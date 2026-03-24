from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger('bentoml.service')

IDLE_CHECK_INTERVAL = 60
GPU_MEMORY_HEADROOM = 0.10
BASE_PORT = 9100  # vllm serve instances get ports starting here


@dataclass
class ModelConfig:
  name: str
  path: str
  tp: int = 1
  max_model_len: int | None = None
  gpu_memory_utilization: float = 0.40
  dtype: str = 'auto'

  # Batch inference tuning
  max_num_seqs: int | None = None
  max_num_batched_tokens: int | None = None

  # Tool calling
  tool_parser: str | None = None  # hermes, llama3_json, pythonic, mistral, jamba, deepseek_v3, ...

  # Reasoning / thinking
  reasoning_parser: str | None = None  # deepseek_r1, qwen3, ...

  # Quantization
  quantization: str | None = None  # awq, gptq, fp8, gguf, bitsandbytes, experts_int8, ...

  # KV cache
  kv_cache_dtype: str | None = None  # auto, fp8, fp8_e4m3, fp8_e5m2

  # Vision / multimodal
  limit_mm_per_prompt: dict[str, int] | None = None  # e.g. {"image": 3}

  # Attention backend
  attn_backend: str = 'FLASH_ATTN'  # FLASH_ATTN, FLASHMLA

  # Misc
  trust_remote_code: bool = False
  chat_template: str | None = None  # path to jinja template
  extra_args: list[str] = field(default_factory=list)  # additional CLI args


@dataclass
class LoadedModel:
  config: ModelConfig
  port: int
  process: asyncio.subprocess.Process
  gpu_ids: list[int] = field(default_factory=list)
  last_access: float = field(default_factory=time.time)

  @property
  def base_url(self) -> str:
    return f'http://127.0.0.1:{self.port}'


@dataclass
class GPUBudget:
  gpu_id: int
  used: float = 0.0

  @property
  def free(self) -> float:
    return max(0.0, 1.0 - GPU_MEMORY_HEADROOM - self.used)

  def can_fit(self, utilization: float) -> bool:
    return self.free >= utilization


class ModelManager:
  """Manages multiple vLLM serve subprocesses with per-GPU memory budgeting.

  Each model runs as a ``vllm serve`` subprocess on its own port, giving full
  OpenAI API compatibility including tool calling, reasoning, vision, and
  quantized model support.  Multiple models can share GPUs as long as the
  sum of their ``gpu_memory_utilization`` values (plus headroom) ≤ 1.0.
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
        tool_parser=cfg.get('tool_parser'),
        reasoning_parser=cfg.get('reasoning_parser'),
        quantization=cfg.get('quantization'),
        kv_cache_dtype=cfg.get('kv_cache_dtype'),
        limit_mm_per_prompt=cfg.get('limit_mm_per_prompt'),
        attn_backend=cfg.get('attn_backend', 'FLASH_ATTN'),
        trust_remote_code=cfg.get('trust_remote_code', False),
        chat_template=cfg.get('chat_template'),
        extra_args=cfg.get('extra_args', []),
      )

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
    self._next_port = BASE_PORT

    logger.info('ModelManager: %d GPU(s), %d model(s) registered', total_gpus, len(self.model_configs))

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  async def start_idle_monitor(self) -> None:
    if self._idle_task is None:
      self._idle_task = asyncio.create_task(self._idle_monitor())

  async def get_model(self, model_name: str) -> LoadedModel:
    """Return the running model for *model_name*, starting it on demand."""
    async with self._lock:
      if model_name in self._loaded:
        loaded = self._loaded[model_name]
        loaded.last_access = time.time()
        return loaded

      if model_name not in self.model_configs:
        raise ValueError(f'Unknown model: {model_name!r}. Available: {list(self.model_configs)}')

      cfg = self.model_configs[model_name]
      if cfg.tp > self.total_gpus:
        raise ValueError(f'Model {model_name!r} requires tp={cfg.tp} but only {self.total_gpus} GPU(s) available')

      gpu_ids = await self._find_or_free_gpus(cfg)
      return await self._do_load(model_name, gpu_ids)

  async def unload_model(self, model_name: str) -> bool:
    async with self._lock:
      if model_name not in self._loaded:
        return False
      await self._do_unload(model_name)
      return True

  async def unload_all(self) -> list[str]:
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
        'features': [],
      }
      if cfg.tool_parser:
        entry['features'].append('tool_calling')
      if cfg.reasoning_parser:
        entry['features'].append('reasoning')
      if cfg.quantization:
        entry['features'].append(f'quantized:{cfg.quantization}')
      if cfg.limit_mm_per_prompt:
        entry['features'].append('vision')
      if loaded:
        lm = self._loaded[name]
        entry['gpu_ids'] = lm.gpu_ids
        entry['port'] = lm.port
      result.append(entry)
    return result

  def status(self) -> dict[str, Any]:
    return {
      'total_gpus': self.total_gpus,
      'gpus': [{'gpu_id': b.gpu_id, 'used': round(b.used, 3), 'free': round(b.free, 3)} for b in self.gpu_budgets],
      'loaded_models': {
        name: {
          'gpu_ids': lm.gpu_ids,
          'port': lm.port,
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
    utilization = cfg.gpu_memory_utilization
    eligible = [b for b in self.gpu_budgets if b.can_fit(utilization)]
    if len(eligible) < cfg.tp:
      return None
    if cfg.tp == 1:
      best = max(eligible, key=lambda b: b.free)
      return [best.gpu_id]
    # tp > 1: try contiguous first.
    eligible_ids = {b.gpu_id for b in eligible}
    for start in range(self.total_gpus - cfg.tp + 1):
      group = list(range(start, start + cfg.tp))
      if all(g in eligible_ids for g in group):
        return group
    eligible.sort(key=lambda b: b.free, reverse=True)
    return [b.gpu_id for b in eligible[: cfg.tp]]

  def _reserve_budget(self, gpu_ids: list[int], utilization: float) -> None:
    for gid in gpu_ids:
      self.gpu_budgets[gid].used += utilization

  def _release_budget(self, gpu_ids: list[int], utilization: float) -> None:
    for gid in gpu_ids:
      self.gpu_budgets[gid].used = max(0.0, self.gpu_budgets[gid].used - utilization)

  async def _find_or_free_gpus(self, cfg: ModelConfig) -> list[int]:
    gpu_ids = self._find_fitting_gpus(cfg)
    if gpu_ids is not None:
      return gpu_ids
    while True:
      if not self._loaded:
        raise RuntimeError(
          f'Cannot fit model {cfg.name!r} (tp={cfg.tp}, util={cfg.gpu_memory_utilization}) '
          f'even with all GPUs empty. Check gpu_memory_utilization + headroom ({GPU_MEMORY_HEADROOM}).'
        )
      lru_name = min(self._loaded, key=lambda n: self._loaded[n].last_access)
      logger.info('Evicting LRU model %s to free memory for %s', lru_name, cfg.name)
      await self._do_unload(lru_name)
      gpu_ids = self._find_fitting_gpus(cfg)
      if gpu_ids is not None:
        return gpu_ids

  # ------------------------------------------------------------------
  # vllm serve subprocess lifecycle
  # ------------------------------------------------------------------

  def _build_vllm_cmd(self, cfg: ModelConfig, port: int, gpu_ids: list[int]) -> list[str]:
    """Build the ``vllm serve`` command line."""
    model_path = cfg.path
    if not os.path.isabs(model_path):
      candidate = os.path.join(self.models_dir, model_path)
      if os.path.isdir(candidate):
        model_path = candidate

    cmd = [
      'vllm', 'serve', model_path,
      '--port', str(port),
      '-tp', str(cfg.tp),
      '--gpu-memory-utilization', str(cfg.gpu_memory_utilization),
      '--dtype', cfg.dtype,
      '--served-model-name', cfg.name,
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      '--no-use-tqdm-on-load',
    ]

    if cfg.max_model_len:
      cmd.extend(['--max-model-len', str(cfg.max_model_len)])
    if cfg.max_num_seqs:
      cmd.extend(['--max-num-seqs', str(cfg.max_num_seqs)])
    if cfg.max_num_batched_tokens:
      cmd.extend(['--max-num-batched-tokens', str(cfg.max_num_batched_tokens)])

    # Tool calling
    if cfg.tool_parser:
      cmd.extend(['--enable-auto-tool-choice', '--tool-call-parser', cfg.tool_parser])

    # Reasoning / thinking
    if cfg.reasoning_parser:
      cmd.extend(['--reasoning-parser', cfg.reasoning_parser])

    # Quantization
    if cfg.quantization:
      cmd.extend(['--quantization', cfg.quantization])

    # KV cache dtype
    if cfg.kv_cache_dtype:
      cmd.extend(['--kv-cache-dtype', cfg.kv_cache_dtype])

    # Vision / multimodal
    if cfg.limit_mm_per_prompt:
      cmd.extend(['--limit-mm-per-prompt', json.dumps(cfg.limit_mm_per_prompt)])

    # Misc
    if cfg.trust_remote_code:
      cmd.append('--trust-remote-code')
    if cfg.chat_template:
      cmd.extend(['--chat-template', cfg.chat_template])

    cmd.extend(cfg.extra_args)
    return cmd

  def _build_env(self, cfg: ModelConfig, gpu_ids: list[int]) -> dict[str, str]:
    """Build environment variables for the vllm serve subprocess."""
    env = {**os.environ}
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(g) for g in gpu_ids)
    env['VLLM_ATTENTION_BACKEND'] = cfg.attn_backend
    return env

  async def _wait_for_health(self, port: int, timeout: float = 300) -> bool:
    """Poll the vllm serve health endpoint until ready or timeout."""
    import httpx

    deadline = time.time() + timeout
    url = f'http://127.0.0.1:{port}/health'
    async with httpx.AsyncClient() as client:
      while time.time() < deadline:
        try:
          resp = await client.get(url, timeout=5)
          if resp.status_code == 200:
            return True
        except (httpx.ConnectError, httpx.RequestError):
          pass
        await asyncio.sleep(2)
    return False

  def _allocate_port(self) -> int:
    port = self._next_port
    self._next_port += 1
    return port

  async def _do_load(self, model_name: str, gpu_ids: list[int]) -> LoadedModel:
    cfg = self.model_configs[model_name]
    port = self._allocate_port()

    self._reserve_budget(gpu_ids, cfg.gpu_memory_utilization)

    cmd = self._build_vllm_cmd(cfg, port, gpu_ids)
    env = self._build_env(cfg, gpu_ids)

    logger.info('Starting vllm serve for %s on port %d, GPU(s) %s', model_name, port, gpu_ids)
    logger.info('Command: %s', ' '.join(cmd))

    try:
      process = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
      )

      # Wait for vLLM to be healthy.
      healthy = await self._wait_for_health(port)
      if not healthy:
        process.terminate()
        await process.wait()
        raise RuntimeError(f'vllm serve for {model_name} failed to start on port {port}')

    except Exception:
      self._release_budget(gpu_ids, cfg.gpu_memory_utilization)
      raise

    loaded = LoadedModel(config=cfg, port=port, process=process, gpu_ids=gpu_ids)
    self._loaded[model_name] = loaded
    logger.info('Model %s ready on port %d, GPU(s) %s', model_name, port, gpu_ids)
    return loaded

  async def _do_unload(self, model_name: str) -> None:
    loaded = self._loaded.pop(model_name, None)
    if loaded is None:
      return

    logger.info('Stopping vllm serve for %s (port %d, GPU(s) %s)', model_name, loaded.port, loaded.gpu_ids)

    # Graceful shutdown: SIGTERM, then SIGKILL after timeout.
    try:
      loaded.process.send_signal(signal.SIGTERM)
      try:
        await asyncio.wait_for(loaded.process.wait(), timeout=15)
      except asyncio.TimeoutError:
        loaded.process.kill()
        await loaded.process.wait()
    except ProcessLookupError:
      pass

    self._release_budget(loaded.gpu_ids, loaded.config.gpu_memory_utilization)

    try:
      import torch
      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass

    logger.info('Model %s stopped, freed GPU(s) %s', model_name, loaded.gpu_ids)

  async def _idle_monitor(self) -> None:
    while True:
      await asyncio.sleep(IDLE_CHECK_INTERVAL)
      async with self._lock:
        now = time.time()
        to_unload = [name for name, lm in self._loaded.items() if (now - lm.last_access) >= self.idle_timeout]
        for name in to_unload:
          idle_for = now - self._loaded[name].last_access
          logger.info('Model %s idle for %.0fs (timeout=%ds), unloading', name, idle_for, self.idle_timeout)
          await self._do_unload(name)
