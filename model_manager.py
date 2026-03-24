from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger('bentoml.service')

IDLE_CHECK_INTERVAL = 60  # seconds between idle sweeps
# Reserve this fraction of each GPU for CUDA contexts, fragmentation, etc.
GPU_MEMORY_HEADROOM = 0.10


@dataclass
class ModelConfig:
  name: str
  path: str
  tp: int = 1
  max_model_len: int | None = None
  gpu_memory_utilization: float = 0.40
  dtype: str = 'auto'
  max_num_seqs: int | None = None
  max_num_batched_tokens: int | None = None
  extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
  engine: Any  # vllm.AsyncLLMEngine
  tokenizer: Any
  config: ModelConfig
  gpu_ids: list[int] = field(default_factory=list)
  last_access: float = field(default_factory=time.time)


@dataclass
class GPUBudget:
  """Tracks memory utilization budget for a single GPU."""
  gpu_id: int
  used: float = 0.0  # sum of gpu_memory_utilization of loaded models

  @property
  def free(self) -> float:
    return max(0.0, 1.0 - GPU_MEMORY_HEADROOM - self.used)

  def can_fit(self, utilization: float) -> bool:
    return self.free >= utilization


class ModelManager:
  """Manages multiple vLLM engines with per-GPU memory budgeting.

  Multiple models can share the same GPU as long as the sum of their
  ``gpu_memory_utilization`` values (plus headroom) doesn't exceed 1.0.
  Models requiring tensor parallelism (tp > 1) span multiple GPUs and
  reserve ``gpu_memory_utilization`` on each.

  When a requested model cannot fit, the least-recently-used model(s) are
  evicted until enough budget is freed.  An idle timeout monitor runs in the
  background to reclaim memory from models that haven't been accessed.
  """

  def __init__(self, config_path: str = 'models_config.yaml') -> None:
    with open(config_path) as f:
      raw = yaml.safe_load(f)

    self.models_dir: str = raw.get('models_dir', '/data/models')
    self.idle_timeout: int = raw.get('idle_timeout_seconds', 1800)
    self.max_batch_size: int = raw.get('max_batch_size', 128)
    self.model_configs: dict[str, ModelConfig] = {}

    for name, cfg in raw.get('models', {}).items():
      self.model_configs[name] = ModelConfig(
        name=name,
        path=cfg['path'],
        tp=cfg.get('tp', 1),
        max_model_len=cfg.get('max_model_len'),
        gpu_memory_utilization=cfg.get('gpu_memory_utilization', 0.40),
        dtype=cfg.get('dtype', 'auto'),
        max_num_seqs=cfg.get('max_num_seqs'),
        max_num_batched_tokens=cfg.get('max_num_batched_tokens'),
        extra_args=cfg.get('extra_args', {}),
      )

    # Detect available GPUs.
    try:
      import torch
      total_gpus = torch.cuda.device_count()
    except Exception:
      total_gpus = int(os.environ.get('NUM_GPUS', '1'))

    self.total_gpus = total_gpus
    self.gpu_budgets: list[GPUBudget] = [GPUBudget(gpu_id=i) for i in range(total_gpus)]
    self._loaded: dict[str, LoadedModel] = {}
    self._lock = asyncio.Lock()
    self._idle_task: asyncio.Task[None] | None = None

    logger.info('ModelManager: %d GPU(s) detected, %d model(s) registered', total_gpus, len(self.model_configs))

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  async def start_idle_monitor(self) -> None:
    if self._idle_task is None:
      self._idle_task = asyncio.create_task(self._idle_monitor())

  async def get_engine(self, model_name: str) -> LoadedModel:
    """Return the engine for *model_name*, loading it on demand."""
    async with self._lock:
      if model_name in self._loaded:
        loaded = self._loaded[model_name]
        loaded.last_access = time.time()
        return loaded

      if model_name not in self.model_configs:
        raise ValueError(f'Unknown model: {model_name!r}. Available: {list(self.model_configs)}')

      cfg = self.model_configs[model_name]
      if cfg.tp > self.total_gpus:
        raise ValueError(
          f'Model {model_name!r} requires tp={cfg.tp} GPUs but only {self.total_gpus} total GPUs available'
        )

      # Ensure enough memory budget is available (evict LRU if needed).
      gpu_ids = await self._find_or_free_gpus(cfg)
      return await self._do_load(model_name, gpu_ids)

  async def unload_model(self, model_name: str) -> bool:
    """Force-unload a specific model. Returns True if it was loaded."""
    async with self._lock:
      if model_name not in self._loaded:
        return False
      await self._do_unload(model_name)
      return True

  async def unload_all(self) -> list[str]:
    """Force-unload all models."""
    async with self._lock:
      names = list(self._loaded.keys())
      for name in names:
        await self._do_unload(name)
      return names

  def list_models(self) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for name, cfg in self.model_configs.items():
      loaded = name in self._loaded
      entry: dict[str, Any] = {
        'id': name,
        'object': 'model',
        'owned_by': 'local',
        'loaded': loaded,
        'path': cfg.path,
        'tp': cfg.tp,
        'gpu_memory_utilization': cfg.gpu_memory_utilization,
      }
      if loaded:
        entry['gpu_ids'] = self._loaded[name].gpu_ids
      result.append(entry)
    return result

  def status(self) -> dict[str, Any]:
    return {
      'total_gpus': self.total_gpus,
      'gpus': [
        {'gpu_id': b.gpu_id, 'used': round(b.used, 3), 'free': round(b.free, 3)} for b in self.gpu_budgets
      ],
      'loaded_models': {
        name: {
          'gpu_ids': lm.gpu_ids,
          'gpu_memory_utilization': lm.config.gpu_memory_utilization,
          'last_access': lm.last_access,
        }
        for name, lm in self._loaded.items()
      },
      'max_batch_size': self.max_batch_size,
    }

  # ------------------------------------------------------------------
  # GPU budget allocation
  # ------------------------------------------------------------------

  def _find_fitting_gpus(self, cfg: ModelConfig) -> list[int] | None:
    """Find ``cfg.tp`` GPUs that can each fit ``cfg.gpu_memory_utilization``.

    For tp=1, pick the GPU with the most free budget.
    For tp>1, find a contiguous (or any) group of tp GPUs that all have room.
    Returns None if no feasible placement exists without eviction.
    """
    utilization = cfg.gpu_memory_utilization
    eligible = [b for b in self.gpu_budgets if b.can_fit(utilization)]

    if len(eligible) < cfg.tp:
      return None

    if cfg.tp == 1:
      # Pick GPU with the most free space.
      best = max(eligible, key=lambda b: b.free)
      return [best.gpu_id]

    # tp > 1: try contiguous first, then any combination.
    eligible_ids = {b.gpu_id for b in eligible}
    for start in range(self.total_gpus - cfg.tp + 1):
      group = list(range(start, start + cfg.tp))
      if all(g in eligible_ids for g in group):
        return group

    # Non-contiguous fallback (sorted by most free space).
    eligible.sort(key=lambda b: b.free, reverse=True)
    return [b.gpu_id for b in eligible[: cfg.tp]]

  def _reserve_budget(self, gpu_ids: list[int], utilization: float) -> None:
    for gid in gpu_ids:
      self.gpu_budgets[gid].used += utilization

  def _release_budget(self, gpu_ids: list[int], utilization: float) -> None:
    for gid in gpu_ids:
      self.gpu_budgets[gid].used = max(0.0, self.gpu_budgets[gid].used - utilization)

  async def _find_or_free_gpus(self, cfg: ModelConfig) -> list[int]:
    """Find GPUs for *cfg*, evicting LRU models if necessary."""
    gpu_ids = self._find_fitting_gpus(cfg)
    if gpu_ids is not None:
      return gpu_ids

    # Evict LRU models until placement is possible.
    while True:
      if not self._loaded:
        raise RuntimeError(
          f'Cannot fit model {cfg.name!r} (tp={cfg.tp}, util={cfg.gpu_memory_utilization}) '
          f'even with all GPUs empty. Check gpu_memory_utilization + headroom ({GPU_MEMORY_HEADROOM}).'
        )
      lru_name = min(self._loaded, key=lambda n: self._loaded[n].last_access)
      logger.info(
        'Evicting LRU model %s (last access %.0fs ago) to free memory for %s',
        lru_name,
        time.time() - self._loaded[lru_name].last_access,
        cfg.name,
      )
      await self._do_unload(lru_name)

      gpu_ids = self._find_fitting_gpus(cfg)
      if gpu_ids is not None:
        return gpu_ids

  # ------------------------------------------------------------------
  # Engine lifecycle
  # ------------------------------------------------------------------

  async def _do_load(self, model_name: str, gpu_ids: list[int]) -> LoadedModel:
    cfg = self.model_configs[model_name]
    model_path = cfg.path
    if not os.path.isabs(model_path):
      candidate = os.path.join(self.models_dir, model_path)
      if os.path.isdir(candidate):
        model_path = candidate

    self._reserve_budget(gpu_ids, cfg.gpu_memory_utilization)
    logger.info(
      'Loading model %s from %s on GPU(s) %s (tp=%d, util=%.2f)',
      model_name, model_path, gpu_ids, cfg.tp, cfg.gpu_memory_utilization,
    )

    try:
      from vllm import AsyncEngineArgs, AsyncLLMEngine

      kwargs: dict[str, Any] = {}
      if cfg.max_model_len:
        kwargs['max_model_len'] = cfg.max_model_len
      if cfg.max_num_seqs:
        kwargs['max_num_seqs'] = cfg.max_num_seqs
      if cfg.max_num_batched_tokens:
        kwargs['max_num_batched_tokens'] = cfg.max_num_batched_tokens

      engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=cfg.tp,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype=cfg.dtype,
        enforce_eager=False,
        disable_log_requests=True,
        **kwargs,
        **cfg.extra_args,
      )

      # Pin engine to specific GPUs.
      old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)

      try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
      finally:
        if old_cuda_visible is not None:
          os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
        else:
          os.environ.pop('CUDA_VISIBLE_DEVICES', None)

      tokenizer = await engine.get_tokenizer()
    except Exception:
      self._release_budget(gpu_ids, cfg.gpu_memory_utilization)
      raise

    loaded = LoadedModel(engine=engine, tokenizer=tokenizer, config=cfg, gpu_ids=gpu_ids)
    self._loaded[model_name] = loaded
    logger.info('Model %s loaded on GPU(s) %s', model_name, gpu_ids)
    return loaded

  async def _do_unload(self, model_name: str) -> None:
    loaded = self._loaded.pop(model_name, None)
    if loaded is None:
      return

    logger.info('Shutting down engine for %s (GPU(s) %s)', model_name, loaded.gpu_ids)
    loaded.engine.shutdown()
    await asyncio.sleep(0.5)

    self._release_budget(loaded.gpu_ids, loaded.config.gpu_memory_utilization)

    try:
      import torch
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass

    logger.info('Model %s unloaded, freed GPU(s) %s', model_name, loaded.gpu_ids)

  async def _idle_monitor(self) -> None:
    while True:
      await asyncio.sleep(IDLE_CHECK_INTERVAL)
      async with self._lock:
        now = time.time()
        to_unload = [
          name for name, lm in self._loaded.items() if (now - lm.last_access) >= self.idle_timeout
        ]
        for name in to_unload:
          idle_for = now - self._loaded[name].last_access
          logger.info('Model %s idle for %.0fs (timeout=%ds), unloading', name, idle_for, self.idle_timeout)
          await self._do_unload(name)
