from __future__ import annotations

import contextlib
import json
import logging
import os
import typing

import bentoml
import fastapi
import pydantic

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
  Jsonable = list[str] | list[dict[str, str]] | None
else:
  Jsonable = typing.Any


class BentoArgs(pydantic.BaseModel):
  tp: int = 1
  v1: bool = True
  attn_backend: str = 'FLASHINFER'
  skip_flashinfer: bool = False
  nightly: bool = False
  piecewise_cudagraph: bool = True
  reasoning_parser: str | None = None
  tool_parser: str | None = None
  max_model_len: int | None = None
  autotune: list[int] | None = None
  hf_system_prompt: str | None = None
  include_system_prompt: bool = True

  sharded: bool = False
  name: str = 'llama3.1-8b-instruct'
  gpu_type: str = 'nvidia-h100-80gb'
  model_id: str = 'meta-llama/Meta-Llama-3.1-8B-Instruct'

  kv_transfer_config: dict[str, typing.Any] = pydantic.Field(default_factory=dict)
  post: list[str] = pydantic.Field(default_factory=list)
  cli_args: list[str] = pydantic.Field(default_factory=list)
  envs: list[dict[str, str]] = pydantic.Field(default_factory=list)
  exclude: list[str] = pydantic.Field(default_factory=lambda: ['*.pth', '*.pt', 'original/**/*'])
  hf_generation_config: dict[str, float | int] = pydantic.Field(
    default_factory=lambda: {'repetition_penalty': 1.0, 'temperature': 0.6, 'top_p': 0.9}
  )
  metadata: dict[str, typing.Any] = pydantic.Field(
    default_factory=lambda: {
      'description': 'Llama 3.1 8B Instruct',
      'provider': 'Meta',
      'gpu_recommendation': 'an Nvidia GPU with at least 80GB VRAM (e.g about 1 H100 GPU).',
    }
  )

  @pydantic.field_validator('exclude', 'cli_args', 'post', 'envs', 'hf_generation_config', 'metadata', mode='before')
  @classmethod
  def _coerce_json_or_csv(cls, v: typing.Any) -> Jsonable:
    if v is None or isinstance(v, (list, dict)):
      return typing.cast(Jsonable, v)
    if isinstance(v, str):
      try:
        return typing.cast(Jsonable, json.loads(v))
      except json.JSONDecodeError:
        return [item.strip() for item in v.split(',') if item.strip()]
    return typing.cast(Jsonable, v)

  @property
  def additional_cli_args(self) -> list[str]:
    import torch

    default = ['-tp', f'{torch.cuda.device_count()}', *self.cli_args]
    if self.kv_transfer_config:
      default.extend(['--kv-transfer-config', json.dumps(self.kv_transfer_config)])
    if self.tool_parser:
      default.extend(['--enable-auto-tool-choice', '--tool-call-parser', self.tool_parser])
    if self.reasoning_parser:
      default.extend(['--reasoning-parser', self.reasoning_parser])
    if self.max_model_len:
      default.extend(['--max-model-len', str(self.max_model_len)])
    if self.v1:
      default.extend([
        '--compilation-config',
        json.dumps({
          'level': 3,
          'cudagraph_capture_sizes': [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1],
          'max_capture_size': 128,
          'cudagraph_num_of_warmups': 1,
          'full_cuda_graph': not self.piecewise_cudagraph,
          'compile_sizes': [],  # [1,2,4,6,8] self.autotune if self.autotune else [] , # TODO: enable autotune once we have cache hit
        }),
      ])
    return default

  @property
  def additional_labels(self) -> dict[str, str]:
    default = {
      'hf_generation_config': json.dumps(self.hf_generation_config),
      'reasoning': '1' if self.reasoning_parser else '0',
      'tool': self.tool_parser or '',
      'sharded': self.sharded,
    }
    if self.hf_system_prompt and self.include_system_prompt:
      default['hf_system_prompt'] = json.dumps(self.hf_system_prompt)
    return default

  @property
  def runtime_envs(self) -> list[dict[str, str]]:
    envs = [*self.envs]
    envs.extend([
      {'name': 'VLLM_SKIP_P2P_CHECK', 'value': '1'},
      {'name': 'VLLM_USE_V1', 'value': '1' if self.v1 else '0'},
    ])
    if not self.gpu_type.startswith('amd'):
      envs.append({'name': 'VLLM_ATTENTION_BACKEND', 'value': self.attn_backend})
    if os.getenv('YATAI_T_VERSION'):
      envs.extend([
        {'name': 'HF_HUB_CACHE', 'value': '/home/bentoml/bento/hf-models'},
        {'name': 'VLLM_CACHE_ROOT', 'value': '/home/bentoml/bento/vllm-models'},
      ])
    return envs

  @property
  def runtime_model_id(self) -> str:
    if not self.sharded:
      return self.model_id.lower()
    repo_slug = self.model_id.lower().split('/')[-1]
    return f'aarnphm/{repo_slug}-sharded-tp{self.tp}'


bento_args = bentoml.use_arguments(BentoArgs)

image = (
  bentoml.images.Image(python_version='3.12').system_packages('curl', 'git').requirements_file('requirements.txt')
)
if POST := bento_args.post:
  for cmd in POST:
    image = image.run(cmd)
if not bento_args.skip_flashinfer and bento_args.gpu_type.startswith('nvidia'):
  image = image.run(
    'uv pip install --no-progress https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl'
  )
if bento_args.gpu_type.startswith('amd'):
  image.base_image = 'rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909'
  # Disable locking of Python packages for AMD GPUs to exclude nvidia-* dependencies
  image.lock_python_packages = False
  # The GPU device is accessible by group 992
  image.run('groupadd -g 992 -o rocm && usermod -aG rocm bentoml')
  # Remove the vllm and torch deps to reuse the pre-installed ones in the base image
  image.run('uv pip uninstall vllm torch torchvision torchaudio triton')
if bento_args.nightly:
  image.run('uv pip uninstall vllm')
  image.run('uv pip install -U vllm --torch-backend=auto --extra-index-url https://wheels.vllm.ai/nightly')
hf = bentoml.models.HuggingFaceModel(bento_args.runtime_model_id, exclude=bento_args.exclude)
openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app)
@bentoml.service(
  name=bento_args.name,
  envs=[
    {'name': 'UV_NO_PROGRESS', 'value': '1'},
    {'name': 'UV_TORCH_BACKEND', 'value': 'auto'},
    *bento_args.runtime_envs,
  ],
  image=image,
  labels={
    'owner': 'bentoml-team',
    'type': 'prebuilt',
    'project': 'bentovllm',
    'openai_endpoint': '/v1',
    **bento_args.additional_labels,
  },
  traffic={'timeout': 300},
  endpoints={'livez': '/health', 'readyz': '/ping'},
  resources={'gpu': bento_args.tp, 'gpu_type': bento_args.gpu_type},
)
class LLM:
  hf = hf

  def __init__(self):
    self.stack = contextlib.AsyncExitStack()

  @bentoml.on_startup
  async def init_engine(self):
    import vllm.entrypoints.openai.api_server as vllm_api_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.utils import FlexibleArgumentParser

    args = make_arg_parser(FlexibleArgumentParser()).parse_args([
      '--no-use-tqdm-on-load',
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      *bento_args.additional_cli_args,
      '--served-model-name',
      bento_args.model_id,
    ])
    args.model = self.hf

    router = fastapi.APIRouter(lifespan=vllm_api_server.lifespan)

    OPENAI_ENDPOINTS = [
      ['/v1/chat/completions', vllm_api_server.create_chat_completion, ['POST']],
      ['/v1/completions', vllm_api_server.create_completion, ['POST']],
      ['/v1/responses', vllm_api_server.create_responses, ['POST']],
      ['/v1/models', vllm_api_server.show_available_models, ['GET']],
      ['/health', vllm_api_server.health, ['GET']],
      ['/ping', vllm_api_server.ping, ['GET']],
    ]

    for route, endpoint, methods in OPENAI_ENDPOINTS:
      router.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)
    openai_api_app.include_router(router)

    self.engine = await self.stack.enter_async_context(vllm_api_server.build_async_engine_client(args))
    self.tokenizer = await self.engine.get_tokenizer()
    self.vllm_config = await self.engine.get_vllm_config()
    await vllm_api_server.init_app_state(self.engine, self.vllm_config, openai_api_app.state, args)

  @bentoml.on_shutdown
  async def teardown_engine(self):
    await self.stack.aclose()
