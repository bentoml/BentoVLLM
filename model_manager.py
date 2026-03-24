from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger('bentoml.service')

IDLE_CHECK_INTERVAL = 60  # seconds between idle sweeps


@dataclass
class ModelConfig:
  name: str
  path: str
  max_model_len: int | None = None
  gpu_memory_utilization: float = 0.90
  dtype: str = 'auto'
  extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadedModel:
  engine: Any  # vllm.AsyncLLMEngine
  tokenizer: Any
  config: ModelConfig
  last_access: float = field(default_factory=time.time)


class ModelManager:
  """Manages a single-GPU vLLM engine with dynamic model swapping.

  Only one model is loaded at a time. When a request arrives for a different
  model the current model is unloaded first, then the requested one is loaded.
  An idle timeout monitor runs in the background to free GPU memory when no
  requests have been served for a while.
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
        max_model_len=cfg.get('max_model_len'),
        gpu_memory_utilization=cfg.get('gpu_memory_utilization', 0.90),
        dtype=cfg.get('dtype', 'auto'),
        extra_args=cfg.get('extra_args', {}),
      )

    self._loaded: LoadedModel | None = None
    self._lock = asyncio.Lock()
    self._idle_task: asyncio.Task[None] | None = None

  # ------------------------------------------------------------------
  # Public API
  # ------------------------------------------------------------------

  async def start_idle_monitor(self) -> None:
    if self._idle_task is None:
      self._idle_task = asyncio.create_task(self._idle_monitor())

  async def get_engine(self, model_name: str) -> LoadedModel:
    """Return the engine for *model_name*, loading it on demand."""
    async with self._lock:
      if self._loaded is not None and self._loaded.config.name == model_name:
        self._loaded.last_access = time.time()
        return self._loaded

      # Need to swap – unload current if any, then load requested.
      if self._loaded is not None:
        logger.info('Unloading model %s to make room for %s', self._loaded.config.name, model_name)
        await self._do_unload()

      return await self._do_load(model_name)

  async def unload_current(self) -> str | None:
    """Force-unload the currently loaded model. Returns its name or None."""
    async with self._lock:
      if self._loaded is None:
        return None
      name = self._loaded.config.name
      await self._do_unload()
      return name

  def list_models(self) -> list[dict[str, Any]]:
    """Return info about every registered model and its loaded status."""
    result: list[dict[str, Any]] = []
    for name, cfg in self.model_configs.items():
      loaded = self._loaded is not None and self._loaded.config.name == name
      result.append({'id': name, 'object': 'model', 'owned_by': 'local', 'loaded': loaded, 'path': cfg.path})
    return result

  # ------------------------------------------------------------------
  # Internal helpers
  # ------------------------------------------------------------------

  async def _do_load(self, model_name: str) -> LoadedModel:
    if model_name not in self.model_configs:
      raise ValueError(f'Unknown model: {model_name!r}. Available: {list(self.model_configs)}')

    cfg = self.model_configs[model_name]
    model_path = cfg.path
    # If the path is not absolute and doesn't look like a HF repo ID with a slash,
    # resolve it relative to models_dir.
    if not os.path.isabs(model_path):
      candidate = os.path.join(self.models_dir, model_path)
      if os.path.isdir(candidate):
        model_path = candidate
      # else keep the original value so vLLM can treat it as a HF repo ID.

    logger.info('Loading model %s from %s', model_name, model_path)

    from vllm import AsyncEngineArgs, AsyncLLMEngine

    engine_args = AsyncEngineArgs(
      model=model_path,
      tensor_parallel_size=1,
      gpu_memory_utilization=cfg.gpu_memory_utilization,
      dtype=cfg.dtype,
      enforce_eager=False,
      disable_log_requests=True,
      **(({'max_model_len': cfg.max_model_len}) if cfg.max_model_len else {}),
      **cfg.extra_args,
    )

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = await engine.get_tokenizer()

    loaded = LoadedModel(engine=engine, tokenizer=tokenizer, config=cfg)
    self._loaded = loaded
    logger.info('Model %s loaded successfully', model_name)
    return loaded

  async def _do_unload(self) -> None:
    if self._loaded is None:
      return
    name = self._loaded.config.name
    engine = self._loaded.engine
    self._loaded = None

    logger.info('Shutting down engine for %s', name)
    engine.shutdown()

    # Give the process a moment to release GPU memory.
    await asyncio.sleep(0.5)

    # Explicitly clear CUDA caches to free GPU memory.
    try:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except Exception:
      pass

    logger.info('Model %s unloaded', name)

  async def _idle_monitor(self) -> None:
    while True:
      await asyncio.sleep(IDLE_CHECK_INTERVAL)
      async with self._lock:
        if self._loaded is None:
          continue
        idle_for = time.time() - self._loaded.last_access
        if idle_for >= self.idle_timeout:
          logger.info(
            'Model %s idle for %.0fs (timeout=%ds), unloading',
            self._loaded.config.name,
            idle_for,
            self.idle_timeout,
          )
          await self._do_unload()
