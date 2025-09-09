# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from __future__ import annotations

import asyncio, logging, os, functools, time, uuid, typing as t
import httpx

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from .config import IS_BENTOCLOUD

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


@dataclass
class ServiceDiscovery:
  prefill_instances: list[ClientInfo] = field(default_factory=list)
  decode_instances: list[ClientInfo] = field(default_factory=list)
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

    if self.next_check > (current_time := time.time()) and self.prefill_instances and not self.decode_instances:
      return
    async with self._lock:
      # Check again after acquiring the lock
      if self.next_check > current_time and self.prefill_instances and not self.decode_instances:
        return

      self.prefill_instances = [
        ClientInfo(f'{host}:{port + i}')
        for i, (host, port) in self.enumerator([_fix_host(host) for host in await Prefiller.get_hosts()])
      ]
      self.decode_instances = [
        ClientInfo(f'{host}:{port + i}')
        for i, (host, port) in self.enumerator([_fix_host(host) for host in await Decoder.get_hosts()])
      ]
      self.next_check = current_time + DEFAULT_PING_SECONDS

  async def select_pair(self) -> tuple[ClientInfo, ClientInfo]:
    await self._update_service_info()
    selected = (
      self.prefill_instances[self.count % len(self.prefill_instances)],
      self.decode_instances[self.count % len(self.decode_instances)],
    )
    self.count += 1
    return selected


sd = ServiceDiscovery()


@app.post('/v1/completions')
@app.post('/v1/chat/completions')
async def handle_request(request: Request):
  original_request_data = await request.json()

  prefill_request = original_request_data.copy()
  # change max_tokens = 1 to let it only do prefill
  prefill_request['max_tokens'] = 1
  if 'max_completion_tokens' in prefill_request:
    prefill_request['max_completion_tokens'] = 1
  prefill_request['stream'] = False
  prefill_request.pop('stream_options', None)

  prefill_client, decode_client = await sd.select_pair()
  logger.info(
    f'handle_request count: {sd.count}, [HTTP:{prefill_client.http_address} 👉 [HTTP:{decode_client.http_address}'
  )

  request_id = random_uuid()
  # disagg_spec = {
  #   "req_id": request_id,
  #   "receiver_host": decode_client.host,
  #   "receiver_init_port": decode_client.init_port,
  #   "receiver_alloc_port": decode_client.alloc_port,
  # }

  # finish prefill
  async for _ in forward_request(
    f'http://{prefill_client.http_address}{request.url.path}', prefill_request, request_id
  ):
    pass
  # return decode
  generator = forward_request(
    f'http://{decode_client.http_address}{request.url.path}', original_request_data, request_id
  )

  media_type = 'text/event-stream' if original_request_data.get('stream', False) else 'application/json'

  return StreamingResponse(generator, media_type=media_type)


@app.get('/v1/models')
async def list_models():
  prefill_client, _ = await sd.select_pair()
  async with httpx.AsyncClient() as client:
    response = await client.get(f'http://{prefill_client.http_address}/v1/models')
    return response.json()


if __name__ == '__main__':
  import uvicorn

  logger.setLevel(logging.INFO)
  logger.addHandler(logging.StreamHandler())

  uvicorn.run(app, host='0.0.0.0', port=10001)
