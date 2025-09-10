# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import asyncio, logging, os, functools, time, uuid, typing as t
import httpx

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from .config import IS_BENTOCLOUD, NIXL_PEER_ALLOC_PORT, NIXL_PEER_INIT_PORT

T = t.TypeVar('T')
DEFAULT_PING_SECONDS = 5
logger = logging.getLogger('bentoml.service')


@functools.lru_cache(maxsize=128)
def _fix_host(netloc: str) -> tuple[str, int]:
  from vllm.utils import get_ip

  host, _, port = netloc.partition(':')
  if host == '127.0.0.1':
    host = get_ip()
  return host, int(port)


app = FastAPI()


def random_uuid() -> str:
  return str(uuid.uuid4().hex)


async def forward_request(url: str, json_data: dict[str, t.Any], request_id: str):
  async with httpx.AsyncClient(timeout=None) as client:
    headers = {'Authorization': f'Bearer {os.environ.get("OPENAI_API_KEY")}', 'X-Request-Id': request_id}
    async with client.stream('POST', url, json=json_data, headers=headers) as resp:
      resp.raise_for_status()
      async for chunk in resp.aiter_bytes():
        yield chunk


@dataclass(frozen=True)
class ClientInfo:
  http_address: str
  # Per-instance NiXL peer ports (single port per paired instance)
  init_port: int | None = None
  alloc_port: int | None = None


@dataclass(frozen=True)
class InstancePair:
  """Represents a paired prefill-decode instance for LMCache + NIXL"""
  prefill: ClientInfo
  decode: ClientInfo
  pair_id: int  # Worker index for this pair


@dataclass
class ServiceDiscovery:
  instance_pairs: list[InstancePair] = field(default_factory=list)
  next_check: float = 0
  count: int = 0
  _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

  if IS_BENTOCLOUD:

    def enumerator(self, iterable: Iterable[T]) -> Iterator[tuple[int, T]]:
      # Do not need port offset for bencloud.
      for x in iterable:
        yield 0, x

  else:

    def enumerator(self, iterable: Iterable[T]) -> Iterator[tuple[int, T]]:
      return enumerate(iterable)

  async def _update_service_info(self) -> None:
    from .decode import Decoder
    from .prefill import Prefiller

    if self.next_check > (current_time := time.time()) and self.instance_pairs:
      return

    async with self._lock:
      # Check again after acquiring the lock
      if self.next_check > current_time and self.instance_pairs:
        return

      # Get raw host lists
      prefill_hosts = await Prefiller.get_hosts()
      decode_hosts = await Decoder.get_hosts()

      # Create paired instances - CRITICAL: maintain 1:1 mapping for LMCache + NIXL
      prefill_clients = [
        ClientInfo(f'{host}:{port + i}')
        for i, (host, port) in self.enumerator([_fix_host(host) for host in prefill_hosts])
      ]
      decode_clients = [
        ClientInfo(
          f'{host}:{port + i}',
          init_port=NIXL_PEER_INIT_PORT + i,
          alloc_port=NIXL_PEER_ALLOC_PORT + i,
        )
        for i, (host, port) in self.enumerator([_fix_host(host) for host in decode_hosts])
      ]

      # Ensure equal number of instances for proper pairing
      min_instances = min(len(prefill_clients), len(decode_clients))
      if len(prefill_clients) != len(decode_clients):
        logger.warning(
          f'Mismatched instance counts: {len(prefill_clients)} prefill, {len(decode_clients)} decode. '
          f'Using first {min_instances} of each for proper pairing.'
        )

      # Create 1:1 paired instances for LMCache + NIXL communication
      self.instance_pairs = [
        InstancePair(
          prefill=prefill_clients[i],
          decode=decode_clients[i],
          pair_id=i + 1  # 1-based for matching worker configs
        )
        for i in range(min_instances)
      ]

      logger.info(f'Updated {len(self.instance_pairs)} paired instances for LMCache + NIXL')
      self.next_check = current_time + DEFAULT_PING_SECONDS

  async def select_pair(self) -> InstancePair:
    """Select a paired prefill-decode instance for LMCache + NIXL coordination"""
    await self._update_service_info()

    if not self.instance_pairs:
      raise RuntimeError('No paired instances available')

    # Round-robin through paired instances (not independent selection)
    selected_pair = self.instance_pairs[self.count % len(self.instance_pairs)]
    self.count += 1

    logger.info(
      f'Selected pair {selected_pair.pair_id}: '
      f'prefill={selected_pair.prefill.http_address} → decode={selected_pair.decode.http_address}'
    )

    return selected_pair


sd = ServiceDiscovery()


@app.post('/v1/completions')
@app.post('/v1/chat/completions')
async def handle_request(request: Request):
  original_request_data = await request.json()

  prefill_request = original_request_data.copy()
  max_tokens = original_request_data.get('max_tokens', 128)
  # change max_tokens = 1 to let it only do prefill
  prefill_request['max_tokens'] = 1
  if 'max_completion_tokens' in prefill_request:
    prefill_request['max_completion_tokens'] = 1
  prefill_request['stream'] = False
  prefill_request.pop('stream_options', None)

  selected_pair = await sd.select_pair()

  # Generate a consistent request_id for KV cache correlation
  request_id = random_uuid()
  disagg_spec = {
    'req_id': request_id,
    'receiver_host': selected_pair.decode.http_address.split(':')[0],
    'receiver_init_port': selected_pair.decode.init_port,
    'receiver_alloc_port': selected_pair.decode.alloc_port,
  }
  prefill_request['kv_transfer_params'] = {'return_first_tok': True, 'disagg_spec': disagg_spec}

  logger.info(
    f'Processing request {request_id} with pair {selected_pair.pair_id}: prefill={selected_pair.prefill.http_address} → decode={selected_pair.decode.http_address}'
  )

  # Step 1: Send prefill request to paired prefiller
  async for i in forward_request(
    f'http://{selected_pair.prefill.http_address}{request.url.path}',
    prefill_request,
    request_id
  ):
    print(i)
    # exhaust the response to completion (no-op per chunk)
    pass

  # Step 2: Send decode request to paired decoder (KV cache transferred via LMCache + NIXL)
  # Include kv_transfer_params with matching req_id so decoder can consume transferred KV
  decode_request = original_request_data.copy()
  decode_request.setdefault('kv_transfer_params', {})
  decode_request['max_tokens'] = max_tokens - 1
  decode_request['kv_transfer_params'].setdefault('disagg_spec', {})
  decode_request['kv_transfer_params']['disagg_spec']['req_id'] = request_id

  generator = forward_request(
    f'http://{selected_pair.decode.http_address}{request.url.path}',
    decode_request,
    request_id
  )

  media_type = 'text/event-stream' if original_request_data.get('stream', False) else 'application/json'
  return StreamingResponse(generator, media_type=media_type)


@app.get('/v1/models')
async def list_models():
  selected_pair = await sd.select_pair()
  async with httpx.AsyncClient() as client:
    response = await client.get(f'http://{selected_pair.prefill.http_address}/v1/models')
    return response.json()


if __name__ == '__main__':
  import uvicorn

  logger.setLevel(logging.INFO)
  logger.addHandler(logging.StreamHandler())

  uvicorn.run(app, host='0.0.0.0', port=10001)
