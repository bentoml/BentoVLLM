from __future__ import annotations

import json, logging, contextlib, typing, uuid
import bentoml, pydantic, fastapi, typing_extensions, annotated_types

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
  from vllm.engine.arg_utils import EngineArgs
  from vllm.config import TaskOption

  class Args(EngineArgs, pydantic.BaseModel):
    pass

else:
  TaskOption = str
  Args = pydantic.BaseModel


class BentoArgs(Args):
  bentovllm_model_id: str = 'mistralai/Ministral-8B-Instruct-2410'
  bentovllm_max_tokens: int = 2048
  bentovllm_full_cudagraph: bool = False
  bentovllm_use_cudagraph: bool = True

  disable_log_requests: bool = True
  max_log_len: int = 1000
  request_logger: typing.Any = None
  disable_log_stats: bool = True
  use_tqdm_on_load: bool = False
  task: TaskOption = 'generate'
  tokenizer_mode: str = 'mistral'
  config_format: str = 'mistral'
  load_format: str = 'mistral'
  max_model_len: int = 4096
  enable_prefix_caching: bool = False
  tool_call_parser: str = 'mistral'
  enable_auto_tool_choice: bool = True
  max_num_seqs: int = 256
  tensor_parallel_size: int = 1

  @pydantic.model_serializer
  def serialize_model(self) -> dict[str, typing.Any]:
    return {k: getattr(self, k) for k in self.__class__.model_fields if not k.startswith('bentovllm_')}


bento_args = bentoml.use_arguments(BentoArgs)
openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
  name='ministral-8b-instruct-2410',
  traffic={'timeout': 300},
  resources={'gpu': bento_args.tensor_parallel_size, 'gpu_type': 'nvidia-l4'},
  envs=[
    {'name': 'HF_TOKEN'},
    {'name': 'UV_NO_PROGRESS', 'value': '1'},
    {'name': 'VLLM_SKIP_P2P_CHECK', 'value': '1'},
    {'name': 'VLLM_USE_V1', 'value': '1'},
  ],
  labels={
    'owner': 'bentoml-team',
    'type': 'prebuilt',
    'project': 'bentovllm',
    'openai_endpoint': '/v1',
    'hf_generation_config': '{"temperature": 0.7, "top_p": 0.95, "max_tokens": 2048}',
    'reasoning': '0',
    'tool': 'mistral',
  },
  image=bentoml.images.Image(python_version='3.11', lock_python_packages=True)
  .system_packages('curl')
  .system_packages('git')
  .requirements_file('requirements.txt')
  .run(
    'uv pip install --compile-bytecode --no-progress https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl'
  ),
)
class VLLM:
  model = bentoml.models.HuggingFaceModel(
    bento_args.bentovllm_model_id, exclude=['model*', '*.pth', '*.pt', 'original/**/*']
  )

  def __init__(self):
    self.exit_stack = contextlib.AsyncExitStack()

  @bentoml.on_startup
  async def init_engine(self) -> None:
    import vllm.entrypoints.openai.api_server as vllm_api_server

    from vllm.utils import FlexibleArgumentParser
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    from vllm.entrypoints.openai.api_server import mount_metrics

    args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
    args.model = self.model
    args.served_model_name = [bento_args.bentovllm_model_id]
    for key, value in bento_args.model_dump().items():
      setattr(args, key, value)

    router = fastapi.APIRouter(lifespan=vllm_api_server.lifespan)
    OPENAI_ENDPOINTS = [
      ['/chat/completions', vllm_api_server.create_chat_completion, ['POST']],
      ['/models', vllm_api_server.show_available_models, ['GET']],
    ]

    for route, endpoint, methods in OPENAI_ENDPOINTS:
      router.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)
    openai_api_app.include_router(router)
    mount_metrics(openai_api_app)

    self.engine = await self.exit_stack.enter_async_context(vllm_api_server.build_async_engine_client(args))
    self.tokenizer = await self.engine.get_tokenizer()
    self.model_config = await self.engine.get_model_config()
    self.vllm_config = await self.engine.get_vllm_config()
    await vllm_api_server.init_app_state(self.engine, self.vllm_config, openai_api_app.state, args)

  @bentoml.on_shutdown
  async def teardown_engine(self):
    await self.exit_stack.aclose()

  @bentoml.api
  async def generate(
    self,
    prompt: str = 'Who are you? Please respond in pirate speak!',
    max_tokens: typing_extensions.Annotated[
      int, annotated_types.Ge(128), annotated_types.Le(bento_args.bentovllm_max_tokens)
    ] = bento_args.bentovllm_max_tokens,
  ) -> typing.AsyncGenerator[str, None]:
    from vllm import SamplingParams, TokensPrompt
    from vllm.entrypoints.chat_utils import apply_mistral_chat_template

    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]

    params = SamplingParams(max_tokens=max_tokens)
    prompt = TokensPrompt(
      prompt_token_ids=apply_mistral_chat_template(self.tokenizer, messages=messages, chat_template=None, tools=None)
    )

    stream = self.engine.generate(request_id=uuid.uuid4().hex, prompt=prompt, sampling_params=params)

    cursor = 0
    async for request_output in stream:
      text = request_output.outputs[0].text
      yield text[cursor:]
      cursor = len(text)
