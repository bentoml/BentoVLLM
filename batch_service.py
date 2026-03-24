"""BentoML batch service with dynamic multi-model management via vllm serve.

Each model runs as a ``vllm serve`` subprocess with full OpenAI API
compatibility — tool calling, reasoning/thinking, vision, quantization,
and KV cache dtype are all supported natively.

Run with:
    bentoml serve batch_service:VLLMBatchService --port 3000

Endpoints (proxied to per-model vllm serve instances):
    POST /v1/chat/completions       – OpenAI chat (tools, reasoning, vision, streaming)
    POST /v1/completions            – OpenAI completions
    POST /v1/batch                  – Batch: array of chat completion requests
    GET  /v1/models                 – List registered models + loaded status + features
    POST /v1/models/{name}/unload   – Force-unload a specific model
    GET  /v1/status                 – GPU allocation summary
    GET  /health                    – Health check
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import bentoml
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_manager import ModelManager

logger = logging.getLogger('bentoml.service')

app = FastAPI()
manager: ModelManager | None = None


# ---------------------------------------------------------------------------
# Proxy helpers
# ---------------------------------------------------------------------------

async def _proxy_request(base_url: str, path: str, body: bytes, headers: dict[str, str], stream: bool):
  """Forward a request to a vllm serve instance and return the response."""
  url = f'{base_url}{path}'
  async with httpx.AsyncClient(timeout=None) as client:
    if stream:
      req = client.build_request('POST', url, content=body, headers=headers)
      resp = await client.send(req, stream=True)
      return StreamingResponse(
        resp.aiter_bytes(),
        status_code=resp.status_code,
        media_type=resp.headers.get('content-type', 'text/event-stream'),
      )
    else:
      resp = await client.post(url, content=body, headers=headers)
      return JSONResponse(resp.json(), status_code=resp.status_code)


async def _get_loaded_model(model_name: str | None):
  """Resolve model name and return the LoadedModel."""
  assert manager is not None
  if not model_name:
    raise HTTPException(400, detail='Missing "model" field in request body')
  try:
    return await manager.get_model(model_name)
  except ValueError as exc:
    raise HTTPException(404, detail=str(exc))
  except RuntimeError as exc:
    raise HTTPException(503, detail=str(exc))


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoints (proxied to vllm serve)
# ---------------------------------------------------------------------------

@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
  """Proxied to vllm serve — supports tool calling, reasoning, vision, streaming."""
  body_bytes = await request.body()
  body_json = await request.json()
  model_name = body_json.get('model')
  loaded = await _get_loaded_model(model_name)

  is_stream = body_json.get('stream', False)
  headers = {'content-type': 'application/json'}

  return await _proxy_request(loaded.base_url, '/v1/chat/completions', body_bytes, headers, stream=is_stream)


@app.post('/v1/completions')
async def completions(request: Request):
  body_bytes = await request.body()
  body_json = await request.json()
  model_name = body_json.get('model')
  loaded = await _get_loaded_model(model_name)

  is_stream = body_json.get('stream', False)
  headers = {'content-type': 'application/json'}

  return await _proxy_request(loaded.base_url, '/v1/completions', body_bytes, headers, stream=is_stream)


# ---------------------------------------------------------------------------
# Batch endpoint
# ---------------------------------------------------------------------------

@app.post('/v1/batch')
async def batch(request: Request):
  """Submit multiple chat completion requests at once.

  All requests go to the same model and are processed concurrently via
  vLLM's continuous batching.  Tool calling, reasoning, and vision are
  supported in each individual request.

  Request body::

      {
        "model": "llama3.1-8b-instruct",
        "requests": [
          {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100},
          {"messages": [...], "tools": [...], "max_tokens": 200}
        ]
      }

  Response::

      { "results": [ <chat.completion>, ... ] }
  """
  assert manager is not None
  body_json = await request.json()
  model_name = body_json.get('model')
  requests = body_json.get('requests', [])

  if not model_name:
    raise HTTPException(400, detail='Missing "model" field')
  if not requests:
    raise HTTPException(400, detail='Missing or empty "requests" array')

  max_batch = manager.max_batch_size
  if len(requests) > max_batch:
    raise HTTPException(
      400,
      detail=f'Batch size {len(requests)} exceeds max_batch_size={max_batch}. Split into smaller batches.',
    )

  loaded = await _get_loaded_model(model_name)

  async def _run_one(req: dict[str, Any]) -> dict[str, Any]:
    # Ensure model name and non-streaming.
    req['model'] = model_name
    req['stream'] = False
    import json as _json

    body = _json.dumps(req).encode()
    headers = {'content-type': 'application/json'}
    async with httpx.AsyncClient(timeout=None) as client:
      resp = await client.post(f'{loaded.base_url}/v1/chat/completions', content=body, headers=headers)
      return resp.json()

  results = await asyncio.gather(*[_run_one(r) for r in requests])
  return JSONResponse({'results': list(results)})


# ---------------------------------------------------------------------------
# Management endpoints
# ---------------------------------------------------------------------------

@app.get('/v1/models')
async def list_models():
  assert manager is not None
  return JSONResponse({'object': 'list', 'data': manager.list_models()})


@app.post('/v1/models/{model_name}/unload')
async def unload_model(model_name: str):
  assert manager is not None
  was_loaded = await manager.unload_model(model_name)
  if not was_loaded:
    raise HTTPException(404, detail=f'Model {model_name!r} is not currently loaded')
  return JSONResponse({'status': 'unloaded', 'model': model_name})


@app.get('/v1/status')
async def gpu_status():
  assert manager is not None
  return JSONResponse(manager.status())


@app.get('/health')
async def health():
  return JSONResponse({'status': 'ok'})


# ---------------------------------------------------------------------------
# BentoML service
# ---------------------------------------------------------------------------

@bentoml.service(
  name='vllm-batch-service',
  traffic={'timeout': 600},
  resources={'gpu': int(os.environ.get('NUM_GPUS', '1'))},
)
@bentoml.mount_asgi_app(app)
class VLLMBatchService:
  def __init__(self) -> None:
    global manager
    config_path = os.environ.get('MODELS_CONFIG', 'models_config.yaml')
    manager = ModelManager(config_path)
    asyncio.get_event_loop().create_task(manager.start_idle_monitor())
    logger.info('VLLMBatchService started, config=%s, gpus=%d', config_path, manager.total_gpus)
