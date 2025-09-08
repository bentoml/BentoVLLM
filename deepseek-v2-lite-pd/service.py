from __future__ import annotations

import json, asyncio, logging, os, time, uuid, functools, enum, typing as t
import bentoml, pydantic, httpx

from collections.abc import Iterable
from dataclasses import dataclass, field
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

IS_BENTOCLOUD = 'BENTOCLOUD_DEPLOYMENT_URL' in os.environ
PROXY_PORT, PREFILL_KV_PORT, DECODE_KV_PORT = 30001, 20001, 21001
PREFILL_PORT, DECODE_PORT = int(os.getenv('PORT', 3000)), int(os.getenv('PORT', 3000))

T = t.TypeVar('T')
DEFAULT_PING_SECONDS = 5
logger = logging.getLogger('bentoml.service')

class KVConnectorMapping(enum.StrEnum):
  lmcache: str = "LMCacheConnectorV1"
  p2p: str = "P2pNcclConnector"


class BentoArgs(pydantic.BaseModel):
  num_prefill: int = 1
  num_decode: int = 1
  model_id: str = 'deepseek-ai/DeepSeek-V2-Lite-Chat'
  kv_connector: KVConnectorMapping = pydantic.Field(default=KVConnectorMapping.lmcache)


bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
    envs=[{'name': 'HF_TOKEN'}, {"name": "PYTHONHASHSEED", "value": "0"}, {"name": "UCX_TLS", "value": "cuda_ipc,cuda_copy,tcp"}],
  endpoints={'livez': '/health', 'readyz': '/health'},
  resources={'gpu': 1, 'gpu_type': 'nvidia-h100-80gb'},
  extra_ports=[DECODE_KV_PORT],
  workers=bento_args.num_decode,
)
class Decoder:
  model = bentoml.models.HuggingFaceModel(bento_args.model_id, exclude=['*.pth', '*.pt', 'original/**/*'])

  def __command__(self) -> list[str]:
    worker_index = bentoml.server_context.worker_index or 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_index - 1)
    http_port = DECODE_PORT + worker_index - 1
    kv_port = DECODE_KV_PORT + worker_index - 1
    transfer_config = {
      'kv_connector': bento_args.kv_connector,
      'kv_role': 'kv_consumer',
      'kv_buffer_size': '8e9',
      'kv_port': str(kv_port),
      'kv_connector_extra_config': {'http_port': str(http_port), 'send_type': 'PUT_ASYNC', 'nccl_num_channels': '16'},
    }
    return [
      'vllm',
      'serve',
      self.model,
      '--no-use-tqdm-on-load',
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      '--served-model-name',
      bento_args.model_id,
      '--port',
      str(http_port),
      '--max-num-batched-tokens',
      '16384',
      '--kv-transfer-config',
      json.dumps(transfer_config),
      '--enable-auto-tool-choice',
      '--tool-call-parser',
      'hermes',
    ]


@bentoml.service(
  envs=[{'name': 'HF_TOKEN'}],
  endpoints={'livez': '/health', 'readyz': '/health'},
  resources={'gpu': 1, 'gpu_type': 'nvidia-h100-80gb'},
  extra_ports=[PREFILL_KV_PORT],
  workers=bento_args.num_prefill,
)
class Prefiller:
  model = bentoml.models.HuggingFaceModel(bento_args.model_id, exclude=['*.pth', '*.pt', 'original/**/*'])

  def __command__(self) -> list[str]:
    worker_index = bentoml.server_context.worker_index or 1
    gpu_offset = 0 if IS_BENTOCLOUD else 4
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_offset + worker_index - 1)
    http_port = PREFILL_PORT + worker_index - 1
    kv_port = PREFILL_KV_PORT + worker_index - 1
    transfer_config = {
      'kv_connector': 'P2pNcclConnector',
      'kv_role': 'kv_producer',
      'kv_buffer_size': '1e1',
      'kv_port': str(kv_port),
      'kv_connector_extra_config': {'http_port': str(http_port), 'send_type': 'PUT_ASYNC', 'nccl_num_channels': '16'},
    }
    return [
      'vllm',
      'serve',
      self.model,
      '--no-use-tqdm-on-load',
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      '--served-model-name',
      bento_args.model_id,
      '--port',
      str(http_port),
      '--max-num-batched-tokens',
      '16384',
      '--kv-transfer-config',
      json.dumps(transfer_config),
      '--enable-auto-tool-choice',
      '--tool-call-parser',
      'hermes',
    ]


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
  zmq_address: str


@dataclass
class ServiceDiscovery:
  prefill_instances: list[ClientInfo] = field(default_factory=list)
  decode_instances: list[ClientInfo] = field(default_factory=list)
  next_check: float = 0
  count: int = 0
  _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

  if IS_BENTOCLOUD:

    def enumerator(self, iterable: Iterable[T]) -> t.Iterator[tuple[int, T]]:
      # Do not need port offset for bencloud.
      for x in iterable:
        yield 0, x

  else:

    def enumerator(self, iterable: Iterable[T]) -> t.Iterator[tuple[int, T]]:
      return enumerate(iterable)

  async def _update_service_info(self) -> None:
    if self.next_check > (current_time := time.time()) and self.prefill_instances and not self.decode_instances:
      return
    async with self._lock:
      # Check again after acquiring the lock
      if self.next_check > current_time and self.prefill_instances and not self.decode_instances:
        return

      self.prefill_instances = [
        ClientInfo(f'{host}:{port + i}', f'{host}:{PREFILL_KV_PORT + i}')
        for i, (host, port) in self.enumerator([_fix_host(host) for host in await Prefiller.get_hosts()])
      ]
      self.decode_instances = [
        ClientInfo(f'{host}:{port + i}', f'{host}:{DECODE_KV_PORT + i}')
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
    f'handle_request count: {sd.count}, [HTTP:{prefill_client.http_address}, '
    f'ZMQ:{prefill_client.zmq_address}] 👉 [HTTP:{decode_client.http_address}, '
    f'ZMQ:{decode_client.zmq_address}]'
  )

  request_id = (
    f'___prefill_addr_{prefill_client.zmq_address}___decode_addr_{decode_client.zmq_address}_{random_uuid()}'
  )

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


@bentoml.service(
  labels={
    'openai_endpoint': '/v1',
    'reasoning': '0',
    'tool': 'hermes',
    'hf_generation_config': json.dumps({'temperature': 0.6, 'top_p': 0.95}),
  },
  image=bentoml.images.Image().requirements_file('requirements.txt'),
)
@bentoml.asgi_app(app)
class Router:
  decoder = bentoml.depends(Decoder)
  prefiller = bentoml.depends(Prefiller)
