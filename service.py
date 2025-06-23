from __future__ import annotations

import logging, json, os, typing, collections.abc
import pydantic, bentoml
from starlette.responses import RedirectResponse

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
  from starlette.requests import Request
  from starlette.responses import Response

  Jsonable = list[str] | list[dict[str, str]] | None
else:
  Jsonable = typing.Any

async def probes(request: Request, call_next: typing.Callable[[Request], collections.abc.Coroutine[typing.Any, typing.Any, Response]]):
  path = request.url.path
  if path == '/livez':
    return RedirectResponse(url='/health', status_code=301)
  if path == '/readyz':
    return RedirectResponse(url='/ping', status_code=301)
  return await call_next(request)


class BentoArgs(pydantic.BaseModel):
  tp: int = 1
  v1: bool = True
  task: str = 'generate'
  attn_backend: str = 'FLASHINFER'
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
    default = ['-tp', f'{self.tp}', *self.cli_args]
    if os.environ.get('VLLM_LOGGING_LEVEL') is None:
      default.append('--disable-log-stats')
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
          'full_cuda_graph': not bento_args.piecewise_cudagraph,
          'compile_sizes': [], # self.autotune if self.autotune else [] , # TODO: enable autotune once we have cache hit
        }),
      ])
    return default

  @property
  def additional_labels(self) -> dict[str, str]:
    default = {
      'hf_generation_config': json.dumps(self.hf_generation_config),
      'reasoning': '1' if self.reasoning_parser else '0',
      'tool': self.tool_parser or '',
    }
    if self.hf_system_prompt and self.include_system_prompt:
      default['hf_system_prompt'] = json.dumps(self.hf_system_prompt)
    return default

  @property
  def runtime_envs(self) -> list[dict[str, str]]:
    envs = [*self.envs]
    if os.getenv('YATAI_T_VERSION'):
      envs.extend([
        {'name': 'HF_HUB_CACHE', 'value': '/home/bentoml/bento/hf-models'},
        {'name': 'VLLM_CACHE_ROOT', 'value': '/home/bentoml/bento/vllm-models'},
      ])
    return envs

  @property
  def runtime_model_id(self) -> str:
    if not self.sharded: return self.model_id.lower()
    repo_slug = self.model_id.lower().split("/")[-1]
    return f'aarnphm/{repo_slug}-sharded-tp{self.tp}'



bento_args = bentoml.use_arguments(BentoArgs)

image = (
  bentoml.images.Image(python_version='3.11').system_packages('curl', 'git').requirements_file('requirements.txt')
)
if POST := bento_args.post:
  for cmd in POST:
    image = image.run(cmd)
image = image.run(
  'uv pip install --compile-bytecode --no-progress https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl'
)
hf = bentoml.models.HuggingFaceModel(bento_args.runtime_model_id, exclude=bento_args.exclude)

os.environ['PYTHONPATH'] = os.pathsep.join([os.environ.get('PYTHONPATH', ''), os.path.abspath(os.path.dirname(__file__))])

LLM = bentoml.Service(
  name=bento_args.name,
  models=[hf],
  envs=[
    {'name': 'UV_NO_PROGRESS', 'value': '1'},
    {'name': 'VLLM_SKIP_P2P_CHECK', 'value': '1'},
    {'name': 'VLLM_USE_V1', 'value': '1' if bento_args.v1 else '0'},
    {'name': 'VLLM_ATTENTION_BACKEND', 'value': bento_args.attn_backend},
    *bento_args.runtime_envs,
  ],
  image=image,
  cmd=[
    'vllm',
    'serve',
    hf.model_id,
    '--served-model-name',
    bento_args.model_id,
    '--port',
    '$PORT',
    '--task',
    bento_args.task,
    '--no-use-tqdm-on-load',
    '--disable-uvicorn-access-log',
    '--disable-log-requests',
    '--disable-fastapi-docs',
    '--max-log-len',
    '1000',
    '--middleware',
    'service.probes',
    *bento_args.additional_cli_args,
  ],
  labels={
    'owner': 'bentoml-team',
    'type': 'prebuilt',
    'project': 'bentovllm',
    'openai_endpoint': '/v1',
    **bento_args.additional_labels,
  },
  config={
    'traffic': {'timeout': 300},
    'endpoints': {'livez': '/health', 'readyz': '/ping'},
    'resources': {'gpu': bento_args.tp, 'gpu_type': bento_args.gpu_type},
  },
)
