from __future__ import annotations

import json
import logging
import os
import typing

import bentoml
import httpx
import pydantic

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
  Jsonable = list[str] | list[dict[str, str]] | None
else:
  Jsonable = typing.Any


class BentoArgs(pydantic.BaseModel):
  tp: int = 1
  dp: int | None = None
  port: int = 8000
  attn_backend: str = 'FLASH_ATTN'
  nightly: bool = False
  reasoning_parser: str | None = None
  tool_parser: str | None = None
  kv_cache_dtype: str | None = None
  max_model_len: int | None = None
  hf_system_prompt: str | None = None

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
  use_sglang_router: bool = False

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

    auto_tp_device = str(os.environ.get('TP_AUTO_ALLOCATE', True)).lower() in {'1', 'true', 'yes', 'y'}
    if auto_tp_device:
      tp_rank = torch.cuda.device_count()
    else:
      tp_rank = self.tp

    default = ['-tp', f'{tp_rank}', *self.cli_args]
    if self.dp:
      default.extend(['-dp', self.dp])
    if self.kv_cache_dtype:
      default.extend(['--kv-cache-dtype', str(self.kv_cache_dtype)])
    if self.kv_transfer_config:
      default.extend(['--kv-transfer-config', json.dumps(self.kv_transfer_config)])
    if self.tool_parser:
      default.extend(['--enable-auto-tool-choice', '--tool-call-parser', self.tool_parser])
    if self.reasoning_parser:
      default.extend(['--reasoning-parser', self.reasoning_parser])
    if self.max_model_len:
      default.extend(['--max-model-len', str(self.max_model_len)])
    default.extend(['--compilation-config', json.dumps({'level': 3})])
    return default

  @property
  def additional_labels(self) -> dict[str, str]:
    default = {
      'hf_generation_config': json.dumps(self.hf_generation_config),
      'reasoning': '1' if self.reasoning_parser else '0',
      'tool': self.tool_parser or '',
    }
    return default

  @property
  def runtime_envs(self) -> list[dict[str, str]]:
    envs = [*self.envs]
    envs.extend([
      {'name': 'VLLM_SKIP_P2P_CHECK', 'value': '1'},
      {'name': 'UV_NO_PROGRESS', 'value': '1'},
      {'name': 'UV_TORCH_BACKEND', 'value': 'cu128'},
    ])
    if not self.gpu_type.startswith('amd'):
      envs.extend([
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': self.attn_backend},
        {'name': 'TORCH_CUDA_ARCH_LIST', 'value': '7.5 8.0 8.9 9.0a 10.0a 12.0'},
      ])
    if os.getenv('YATAI_T_VERSION'):
      envs.extend([
        {'name': 'HF_HUB_CACHE', 'value': '/home/bentoml/bento/hf-models'},
        {'name': 'VLLM_CACHE_ROOT', 'value': '/home/bentoml/bento/vllm-models'},
      ])
    return envs

  @property
  def image(self) -> bentoml.images.Image:
    image = (
      bentoml.images.Image(python_version='3.12').system_packages('curl', 'git').requirements_file('requirements.txt')
    )
    if self.post:
      for cmd in self.post:
        image = image.run(cmd)

    if False:  # self.gpu_type.startswith('nvidia'):
      image = image.run('uv pip install flashinfer-python flashinfer-cubin --torch-backend=cu128')
      image = image.run('uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128')

    if self.gpu_type.startswith('amd'):
      image.base_image = 'rocm/vllm:rocm6.4.1_vllm_0.10.1_20250909'
      # Disable locking of Python packages for AMD GPUs to exclude nvidia-* dependencies
      image.lock_python_packages = False
      # The GPU device is accessible by group 992
      image.run('groupadd -g 992 -o rocm && usermod -aG rocm bentoml && usermod -aG render bentoml')
      # Remove the vllm and torch deps to reuse the pre-installed ones in the base image
      image.run('uv pip uninstall vllm torch torchvision torchaudio triton')

    if self.nightly:
      image.run('uv pip uninstall vllm')
      image.run('uv pip install -U vllm --torch-backend=cu129 --extra-index-url https://wheels.vllm.ai/nightly')

    return image


bento_args = bentoml.use_arguments(BentoArgs)

if bento_args.use_sglang_router:
  from bento_sgl_router import service
else:
  service = bentoml.service


@service(
  name=bento_args.name,
  envs=bento_args.runtime_envs,
  image=bento_args.image,
  labels={
    'owner': 'bentoml-team',
    'type': 'prebuilt',
    'project': 'bentovllm',
    'openai_endpoint': '/v1',
    **bento_args.additional_labels,
  },
  traffic={'timeout': 300},
  endpoints={'readyz': '/health'},
  resources={'gpu': bento_args.tp, 'gpu_type': bento_args.gpu_type},
)
class LLM:
  hf = bentoml.models.HuggingFaceModel(bento_args.model_id.lower(), exclude=bento_args.exclude)

  def __command__(self) -> list[str]:
    return [
      'vllm',
      'serve',
      self.hf,
      '--port',
      str(bento_args.port),
      '--no-use-tqdm-on-load',
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      *bento_args.additional_cli_args,
      '--served-model-name',
      bento_args.model_id,
    ]

  async def __metrics__(self, content: str) -> str:
    client = typing.cast(httpx.AsyncClient, LLM.context.state['client'])
    try:
      response = await client.get(f'http://localhost:{bento_args.port}/metrics', timeout=5.0)
      response.raise_for_status()
    except (httpx.ConnectError, httpx.RequestError) as e:
      logger.error('Failed to get metrics: %s', e)
      return content
    else:
      return content + '\n' + response.text
