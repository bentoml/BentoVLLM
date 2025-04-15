from __future__ import annotations

import logging, contextlib, typing
import bentoml, fastapi, pydantic

logger = logging.getLogger(__name__)

class BentoArgs(pydantic.BaseModel):
  bentovllm_name: str = "bentovllm-qwq-32b-service"
  bentovllm_model_id: str = 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B'
  bentovllm_ignore_patterns: list[str] = pydantic.Field(default_factory=lambda: ['*.pth', '*.pt', 'original/**/*', 'consolidated*'])
  bentovllm_resources: dict[str, typing.Any] = {'gpu': 1, 'gpu_type': 'nvidia-a100-80gb'}
  bentovllm_system_packages: list[str] | None = None
  bentovllm_post_run: list[str] | None = None
  bentovllm_envs: list[dict[str, str]] = pydantic.Field(default_factory=list)

  disable_log_requests: bool = True
  max_log_len: int = 1000
  request_logger: typing.Any = None
  disable_log_stats: bool = True
  use_tqdm_on_load: bool = False

  max_model_len: int = 4096
  enable_reasoning: bool = True
  reasoning_parser: str = 'deepseek_r1'
  enable_auto_tool_choice: bool = True
  tool_call_parser: str = 'llama3_json'
  max_num_seqs: int = 256

  @pydantic.model_serializer
  def serialize_model(self) -> dict[str, typing.Any]:
    return {k: getattr(self, k) for k in self.__class__.model_fields if not k.startswith("bentovllm_")}

bento_args = bentoml.use_arguments(BentoArgs)
openai_api_app = fastapi.FastAPI()

IMAGE = bentoml.images.Image(python_version='3.11', lock_python_packages=False)
if (system_packages:=bento_args.bentovllm_system_packages) is not None:
  for pkg in system_packages: IMAGE = IMAGE.system_packages(pkg)
IMAGE = IMAGE.pyproject_toml('pyproject.toml')
if (post_run:=bento_args.bentovllm_post_run) is not None:
  for cmd in post_run: IMAGE = IMAGE.run(cmd)
IMAGE = IMAGE.run('uv pip install --compile-bytecode flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.6')

@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
  name=bento_args.bentovllm_name,
  traffic={'timeout': 300},
  resources=bento_args.bentovllm_resources,
  labels={'owner': 'bentoml-team', 'type': 'prebuilt'},
  image=IMAGE,
  envs=[
    {'name': 'HF_TOKEN'},
    {'name': 'UV_NO_BUILD_ISOLATION', 'value': 1},
    {'name': 'UV_NO_PROGRESS', 'value': '1'},
    {'name': 'HF_HUB_DISABLE_PROGRESS_BARS', 'value': '1'},
    {'name': 'VLLM_USE_V1', 'value': '1'},
    *bento_args.bentovllm_envs
  ],
)
class VLLM:
  model = bentoml.models.HuggingFaceModel(bento_args.bentovllm_model_id, exclude=bento_args.bentovllm_ignore_patterns)

  def __init__(self):
    self.exit_stack = contextlib.AsyncExitStack()

  @bentoml.on_startup
  async def init_engine(self) -> None:
    import vllm.entrypoints.openai.api_server as vllm_api_server

    from vllm.utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser

    args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
    args.model = self.model
    args.served_model_name = [bento_args.bentovllm_model_id]
    for key, value in bento_args.model_dump().items(): setattr(args, key, value)

    router = fastapi.APIRouter(lifespan=vllm_api_server.lifespan)
    OPENAI_ENDPOINTS = [
      ["/chat/completions", vllm_api_server.create_chat_completion, ["POST"]],
      ["/completions", vllm_api_server.create_completion, ["POST"]],
      ["/models", vllm_api_server.show_available_models, ["GET"]],
    ]

    for route, endpoint, methods in OPENAI_ENDPOINTS: router.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)
    openai_api_app.include_router(router)

    self.engine = await self.exit_stack.enter_async_context(vllm_api_server.build_async_engine_client(args))
    self.model_config = await self.engine.get_model_config()
    self.tokenizer = await self.engine.get_tokenizer()
    await vllm_api_server.init_app_state(self.engine, self.model_config, openai_api_app.state, args)

  @bentoml.on_shutdown
  async def teardown_engine(self): await self.exit_stack.aclose()
