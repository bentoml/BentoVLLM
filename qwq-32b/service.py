from __future__ import annotations

import logging, contextlib, typing
import bentoml, pydantic, fastapi, typing_extensions, annotated_types

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs

    class Args(EngineArgs, pydantic.BaseModel):
        pass

else:
    Args = pydantic.BaseModel


class BentoArgs(Args):
    bentovllm_model_id: str = 'Qwen/QwQ-32B'
    bentovllm_max_tokens: int = 2048

    disable_log_requests: bool = True
    max_log_len: int = 1000
    request_logger: typing.Any = None
    disable_log_stats: bool = True
    use_tqdm_on_load: bool = False
    task: str = 'generate'
    max_model_len: int = 4096
    enable_reasoning: bool = True
    reasoning_parser: str = 'deepseek_r1'
    max_num_seqs: int = 512
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = 'hermes'
    tensor_parallel_size: int = 2

    @pydantic.model_serializer
    def serialize_model(self) -> dict[str, typing.Any]:
        return {k: getattr(self, k) for k in self.__class__.model_fields if not k.startswith('bentovllm_')}


bento_args = bentoml.use_arguments(BentoArgs)
openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='qwq-32b',
    traffic={'timeout': 300},
    resources={'gpu': bento_args.tensor_parallel_size, 'gpu_type': 'nvidia-tesla-h100'},
    envs=[
        {'name': 'HF_TOKEN'},
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': 'FLASH_ATTN'},
        {'name': 'VLLM_USE_V1', 'value': '1'},
    ],
    labels={'owner': 'bentoml-team', 'type': 'prebuilt', 'project': 'bentovllm'},
    image=bentoml.images.Image(python_version='3.11')
    .requirements_file('requirements.txt')
    .run('uv pip install --compile-bytecode flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.6'),
)
class VLLM:
    model = bentoml.models.HuggingFaceModel(bento_args.bentovllm_model_id, exclude=['*.pth', '*.pt', 'original/**/*'])

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

        engine = await self.exit_stack.enter_async_context(vllm_api_server.build_async_engine_client(args))
        vllm_config = await engine.get_vllm_config()
        await vllm_api_server.init_app_state(engine, vllm_config, openai_api_app.state, args)

    @bentoml.on_shutdown
    async def teardown_engine(self):
        await self.exit_stack.aclose()
