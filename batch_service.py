"""BentoML batch service with dynamic multi-model loading across 1-8 GPUs.

Multiple models can be loaded simultaneously as long as GPUs are available.
When a new model doesn't fit, the least-recently-used model is evicted.

Run with:
    bentoml serve batch_service:VLLMBatchService --port 3000

Endpoints:
    POST /v1/chat/completions       – OpenAI-compatible chat (streams or returns JSON)
    POST /v1/completions            – OpenAI-compatible completions
    POST /v1/batch                  – Batch: array of chat completion requests
    GET  /v1/models                 – List registered models + loaded status
    POST /v1/models/{name}/unload   – Force-unload a specific model
    GET  /v1/status                 – GPU allocation summary
    GET  /health                    – Health check
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator

import bentoml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from model_manager import ModelManager

logger = logging.getLogger('bentoml.service')

app = FastAPI()
manager: ModelManager | None = None  # initialised by BentoML lifecycle


def _uuid() -> str:
  return uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _generate_chat(loaded, request_data: dict[str, Any]) -> dict[str, Any] | AsyncIterator[str]:
  """Run a single chat-completion request through the vLLM engine."""
  from vllm import SamplingParams

  messages = request_data.get('messages', [])
  model_name = loaded.config.name

  prompt = loaded.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

  sampling = SamplingParams(
    max_tokens=request_data.get('max_tokens', 512),
    temperature=request_data.get('temperature', 0.7),
    top_p=request_data.get('top_p', 1.0),
    stop=request_data.get('stop'),
  )

  request_id = f'chatcmpl-{_uuid()}'
  stream = request_data.get('stream', False)

  if stream:
    return _stream_chat(loaded.engine, prompt, sampling, request_id, model_name)

  # Non-streaming: collect the full output.
  final_output = None
  async for output in loaded.engine.generate(prompt, sampling, request_id):
    final_output = output

  text = final_output.outputs[0].text if final_output else ''
  finish_reason = final_output.outputs[0].finish_reason if final_output else 'stop'
  usage = {
    'prompt_tokens': len(final_output.prompt_token_ids) if final_output else 0,
    'completion_tokens': len(final_output.outputs[0].token_ids) if final_output else 0,
    'total_tokens': (len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids)) if final_output else 0,
  }

  return {
    'id': request_id,
    'object': 'chat.completion',
    'created': int(time.time()),
    'model': model_name,
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': text}, 'finish_reason': finish_reason}],
    'usage': usage,
  }


async def _stream_chat(engine, prompt: str, sampling, request_id: str, model_name: str) -> AsyncIterator[str]:
  previous_len = 0
  async for output in engine.generate(prompt, sampling, request_id):
    text = output.outputs[0].text
    delta = text[previous_len:]
    previous_len = len(text)
    if delta:
      chunk = {
        'id': request_id,
        'object': 'chat.completion.chunk',
        'created': int(time.time()),
        'model': model_name,
        'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': None}],
      }
      yield f'data: {json.dumps(chunk)}\n\n'

  # Final chunk with finish_reason.
  final_chunk = {
    'id': request_id,
    'object': 'chat.completion.chunk',
    'created': int(time.time()),
    'model': model_name,
    'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
  }
  yield f'data: {json.dumps(final_chunk)}\n\n'
  yield 'data: [DONE]\n\n'


async def _generate_completion(loaded, request_data: dict[str, Any]) -> dict[str, Any]:
  """Run a single /v1/completions request."""
  from vllm import SamplingParams

  prompt = request_data.get('prompt', '')
  model_name = loaded.config.name

  sampling = SamplingParams(
    max_tokens=request_data.get('max_tokens', 256),
    temperature=request_data.get('temperature', 0.7),
    top_p=request_data.get('top_p', 1.0),
    stop=request_data.get('stop'),
  )

  request_id = f'cmpl-{_uuid()}'
  final_output = None
  async for output in loaded.engine.generate(prompt, sampling, request_id):
    final_output = output

  text = final_output.outputs[0].text if final_output else ''
  finish_reason = final_output.outputs[0].finish_reason if final_output else 'stop'

  return {
    'id': request_id,
    'object': 'text_completion',
    'created': int(time.time()),
    'model': model_name,
    'choices': [{'index': 0, 'text': text, 'finish_reason': finish_reason}],
    'usage': {
      'prompt_tokens': len(final_output.prompt_token_ids) if final_output else 0,
      'completion_tokens': len(final_output.outputs[0].token_ids) if final_output else 0,
      'total_tokens': (len(final_output.prompt_token_ids) + len(final_output.outputs[0].token_ids)) if final_output else 0,
    },
  }


# ---------------------------------------------------------------------------
# FastAPI routes (mounted by BentoML)
# ---------------------------------------------------------------------------

@app.post('/v1/chat/completions')
async def chat_completions(request: Request):
  assert manager is not None
  body = await request.json()
  model_name = body.get('model')
  if not model_name:
    raise HTTPException(400, detail='Missing "model" field')

  try:
    loaded = await manager.get_engine(model_name)
  except ValueError as exc:
    raise HTTPException(404, detail=str(exc))

  result = await _generate_chat(loaded, body)

  if isinstance(result, dict):
    return JSONResponse(result)
  # Streaming
  return StreamingResponse(result, media_type='text/event-stream')


@app.post('/v1/completions')
async def completions(request: Request):
  assert manager is not None
  body = await request.json()
  model_name = body.get('model')
  if not model_name:
    raise HTTPException(400, detail='Missing "model" field')

  try:
    loaded = await manager.get_engine(model_name)
  except ValueError as exc:
    raise HTTPException(404, detail=str(exc))

  result = await _generate_completion(loaded, body)
  return JSONResponse(result)


@app.post('/v1/batch')
async def batch(request: Request):
  """Batch endpoint: submit multiple chat completion requests at once.

  Request body:
      {
        "model": "llama3.1-8b-instruct",
        "requests": [
          {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 100},
          {"messages": [{"role": "user", "content": "World"}], "max_tokens": 100}
        ]
      }

  Response:
      {
        "results": [ <chat.completion>, <chat.completion>, ... ]
      }
  """
  assert manager is not None
  body = await request.json()
  model_name = body.get('model')
  requests = body.get('requests', [])

  if not model_name:
    raise HTTPException(400, detail='Missing "model" field')
  if not requests:
    raise HTTPException(400, detail='Missing or empty "requests" array')

  max_batch = manager.max_batch_size
  if len(requests) > max_batch:
    raise HTTPException(
      400,
      detail=f'Batch size {len(requests)} exceeds max_batch_size={max_batch}. '
      f'Split your requests into smaller batches.',
    )

  try:
    loaded = await manager.get_engine(model_name)
  except ValueError as exc:
    raise HTTPException(404, detail=str(exc))

  # Fan-out all requests concurrently – vLLM continuous batching handles them.
  async def _run(req: dict[str, Any]) -> dict[str, Any]:
    req['model'] = model_name
    req['stream'] = False
    result = await _generate_chat(loaded, req)
    assert isinstance(result, dict)
    return result

  results = await asyncio.gather(*[_run(r) for r in requests])
  return JSONResponse({'results': list(results)})


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
    logger.info(
      'VLLMBatchService started, config=%s, total_gpus=%d',
      config_path,
      manager.total_gpus,
    )
