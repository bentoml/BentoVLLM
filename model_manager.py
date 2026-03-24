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


@dataclass
class ModelConfig:
  name: str
  path: str
  tp: int = 1
  max_model_len: int | None = None
  gpu_memory_utilization: float = 0.90
  dtype: str = 'auto'
  extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
  engine: Any  # vllm.AsyncLLMEngine
  tokenizer: Any
  config: ModelConfig
  gpu_ids: list[int] = field(default_factory=list)
  last_access: float = field(default_factory=time.time)


class ModelManager:
  """Manages multiple vLLM engines across one or more GPUs.

  Multiple models can be loaded simultaneously as long as GPUs are available.
  Each model declares a tensor-parallelism degree (tp) which determines how
  many GPUs it needs.  When a requested model cannot fit, the least-recently-
  used loaded model(s) are evicted until enough GPUs are free.

  An idle timeout monitor runs in the background to reclaim GPUs from models
  that haven't received requests for a while.
  """

  def __init__(self, config_path: str = 'models_config.yaml') -> None:
    with open(config_path) as f:
      raw = yaml.safe_load(f)

    self.models_dir: str = raw.get('models_dir', '/data/models')
    self.idle_timeout: int = raw.get('idle_timeout_seconds', 1800)
    self.model_configs: dict[str, ModelConfig] = {}

    for name, cfg in raw.get('models', {}).items():
      self.model_configs[name] = ModelConfig(
        name=name,
        path=cfg['path'],
        tp=cfg.get('tp', 1),
        max_model_len=cfg.get('max_model_len'),
        gpu_memory_utilization=cfg.get('gpu_memory_utilization', 0.90),
        dtype=cfg.get('dtype', 'auto'),
        extra_args=cfg.get('extra_args', {}),
      )

    # Detect available GPUs.
    try:
      import torch
      total_gpus = torch.cuda.device_count()
    except Exception:
      total_gpus = int(os.environ.get('NUM_GPUS', '1'))

    self.total_gpus = total_gpus
    self.available_gpus: set[int] = set(range(total_gpus))
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
      # Already loaded – just touch and return.
      if model_name in self._loaded:
        loaded = self._loaded[model_name]
        loaded.last_access = time.time()
        return loaded

      # Need to load – ensure enough GPUs are free.
      if model_name not in self.model_configs:
        raise ValueError(f'Unknown model: {model_name!r}. Available: {list(self.model_configs)}')

      needed = self.model_configs[model_name].tp
      if needed > self.total_gpus:
        raise ValueError(
          f'Model {model_name!r} requires tp={needed} GPUs but only {self.total_gpus} total GPUs available'
        )

      # Evict LRU models until we have enough free GPUs.
      await self._ensure_free_gpus(needed)

      return await self._do_load(model_name)

  async def unload_model(self, model_name: str) -> bool:
    """Force-unload a specific model. Returns True if it was loaded."""
    async with self._lock:
      if model_name not in self._loaded:
        return False
      await self._do_unload(model_name)
      return True

  async def unload_all(self) -> list[str]:
    """Force-unload all models. Returns names of unloaded models."""
    async with self._lock:
      names = list(self._loaded.keys())
      for name in names:
        await self._do_unload(name)
      return names

  def list_models(self) -> list[dict[str, Any]]:
    """Return info about every registered model and its loaded status."""
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
      }
      if loaded:
        entry['gpu_ids'] = self._loaded[name].gpu_ids
      result.append(entry)
    return result

  def status(self) -> dict[str, Any]:
    """Return a summary of GPU allocation."""
    return {
      'total_gpus': self.total_gpus,
      'available_gpus': sorted(self.available_gpus),
      'loaded_models': {
        name: {'gpu_ids': lm.gpu_ids, 'last_access': lm.last_access} for name, lm in self._loaded.items()
      },
    }

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  def _allocate_gpus(self, n: int) -> list[int]:
    """Allocate *n* GPUs from the free pool. Raises if not enough."""
    if len(self.available_gpus) < n:
      raise RuntimeError(
        f'Need {n} free GPU(s) but only {len(self.available_gpus)} available '
        f'(used by: {", ".join(self._loaded.keys()) or "none"})'
      )
    # Take the lowest-numbered available GPUs for determinism.
    allocated = sorted(self.available_gpus)[:n]
    self.available_gpus -= set(allocated)
    return allocated

  def _release_gpus(self, gpu_ids: list[int]) -> None:
    self.available_gpus |= set(gpu_ids)

  async def _ensure_free_gpus(self, needed: int) -> None:
    """Evict least-recently-used models until *needed* GPUs are free."""
    while len(self.available_gpus) < needed:
      if not self._loaded:
        raise RuntimeError(f'Need {needed} GPU(s) but no models to evict and only {len(self.available_gpus)} free')
      # Find LRU model.
      lru_name = min(self._loaded, key=lambda n: self._loaded[n].last_access)
      logger.info(
        'Evicting LRU model %s (last access %.0fs ago) to free %d GPU(s)',
        lru_name,
        time.time() - self._loaded[lru_name].last_access,
        self._loaded[lru_name].config.tp,
      )
      await self._do_unload(lru_name)

  async def _do_load(self, model_name: str) -> LoadedModel:
    cfg = self.model_configs[model_name]
    model_path = cfg.path
    if not os.path.isabs(model_path):
      candidate = os.path.join(self.models_dir, model_path)
      if os.path.isdir(candidate):
        model_path = candidate
      # else keep the original value so vLLM can treat it as a HF repo ID.

    gpu_ids = self._allocate_gpus(cfg.tp)
    logger.info('Loading model %s from %s on GPU(s) %s (tp=%d)', model_name, model_path, gpu_ids, cfg.tp)

    try:
      from vllm import AsyncEngineArgs, AsyncLLMEngine

      engine_args = AsyncEngineArgs(
        model=model_path,
        tensor_parallel_size=cfg.tp,
        gpu_memory_utilization=cfg.gpu_memory_utilization,
        dtype=cfg.dtype,
        enforce_eager=False,
        disable_log_requests=True,
        **(({'max_model_len': cfg.max_model_len}) if cfg.max_model_len else {}),
        **cfg.extra_args,
      )

      # Pin to specific GPUs via environment variable.
      old_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES')
      os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)

      try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
      finally:
        # Restore original CUDA_VISIBLE_DEVICES.
        if old_cuda_visible is not None:
          os.environ['CUDA_VISIBLE_DEVICES'] = old_cuda_visible
        else:
          os.environ.pop('CUDA_VISIBLE_DEVICES', None)

      tokenizer = await engine.get_tokenizer()
    except Exception:
      # Loading failed – return GPUs to pool.
      self._release_gpus(gpu_ids)
      raise

    loaded = LoadedModel(engine=engine, tokenizer=tokenizer, config=cfg, gpu_ids=gpu_ids)
    self._loaded[model_name] = loaded
    logger.info('Model %s loaded successfully on GPU(s) %s', model_name, gpu_ids)
    return loaded

  async def _do_unload(self, model_name: str) -> None:
    loaded = self._loaded.pop(model_name, None)
    if loaded is None:
      return

    logger.info('Shutting down engine for %s (GPU(s) %s)', model_name, loaded.gpu_ids)
    loaded.engine.shutdown()

    # Give the process a moment to release GPU memory.
    await asyncio.sleep(0.5)

    # Return GPUs to the pool.
    self._release_gpus(loaded.gpu_ids)

    # Clear CUDA caches.
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
          logger.info(
            'Model %s idle for %.0fs (timeout=%ds), unloading',
            name,
            idle_for,
            self.idle_timeout,
          )
          await self._do_unload(name)
