from __future__ import annotations

import logging, os, contextlib, typing
import bentoml, fastapi

logger = logging.getLogger(__name__)

MAX_TOKENS = 1024
ENGINE_CONFIG = {'max_model_len': 2048, 'tensor_parallel_size': 2, 'max_num_seqs': 1024, 'enable_prefix_caching': True}

openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='bentovllm-qwen2.5-72b-instruct-service',
    traffic={'timeout': 300},
    resources={'gpu': 2, 'gpu_type': 'nvidia-a100-80gb'},
    envs=[
        {'name': 'UV_NO_PROGRESS', 'value': '1'},
        {'name': 'HF_HUB_DISABLE_PROGRESS_BARS', 'value': '1'},
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': 'FLASH_ATTN'},
        {'name': 'VLLM_USE_V1', 'value': '1'},
    ],
    labels={'owner': 'bentoml-team', 'type': 'prebuilt'},
    image=bentoml.images.PythonImage(python_version='3.11', lock_python_packages=False)
    .requirements_file('requirements.txt')
    .run('uv pip install --compile-bytecode flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.6'),
)
class VLLM:
    model_id = 'Qwen/Qwen2.5-72B-Instruct'
    model = bentoml.models.HuggingFaceModel(model_id, exclude=['*.pth', '*.pt', 'original/**/*'])

    def __init__(self):
        self.exit_stack = contextlib.AsyncExitStack()

    @bentoml.on_startup
    async def init_engine(self) -> None:
        import vllm.entrypoints.openai.api_server as vllm_api_server

        from vllm.utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser

        args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
        args.model = self.model
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.served_model_name = [self.model_id]
        args.request_logger = None
        args.disable_log_stats = True
        args.use_tqdm_on_load = False
        for key, value in ENGINE_CONFIG.items():
            setattr(args, key, value)
        args.enable_auto_tool_choice = True
        args.tool_call_parser = 'llama3_json'

        router = fastapi.APIRouter(lifespan=vllm_api_server.lifespan)
        OPENAI_ENDPOINTS = [
            ['/chat/completions', vllm_api_server.create_chat_completion, ['POST']],
            ['/models', vllm_api_server.show_available_models, ['GET']],
        ]

        for route, endpoint, methods in OPENAI_ENDPOINTS:
            router.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)
        openai_api_app.include_router(router)

        self.engine = await self.exit_stack.enter_async_context(vllm_api_server.build_async_engine_client(args))
        self.model_config = await self.engine.get_model_config()
        self.tokenizer = await self.engine.get_tokenizer()
        await vllm_api_server.init_app_state(self.engine, self.model_config, openai_api_app.state, args)

    @bentoml.on_shutdown
    async def teardown_engine(self):
        await self.exit_stack.aclose()
