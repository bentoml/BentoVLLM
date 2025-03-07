from __future__ import annotations

import logging, typing, uuid
import bentoml, fastapi, typing_extensions, annotated_types

logger = logging.getLogger(__name__)

ENGINE_CONFIG = {
    'model': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'max_model_len': 4096,
    'enable_prefix_caching': True,
}
MAX_TOKENS = 2048

openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='bentovllm-llama3.1-8b-instruct-service',
    traffic={'timeout': 300},
    resources={'gpu': 1, 'gpu_type': 'nvidia-l4'},
    envs=[
        {'name': 'HF_TOKEN'},
        {'name': 'UV_NO_PROGRESS', 'value': 1},
        {'name': 'HF_HUB_DISABLE_PROGRESS_BARS', 'value': 1},
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': 'FLASH_ATTN'},
        {'name': 'VLLM_USE_V1', 'value': 1},
    ],
    labels={'owner': 'bentoml-team', 'type': 'prebuilt'},
    image=bentoml.images.PythonImage(python_version='3.11', lock_python_packages=False)
    .requirements_file('requirements.txt')
    .run('uv pip install flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.5'),
)
class VLLM:
    model_id = ENGINE_CONFIG['model']
    model = bentoml.models.HuggingFaceModel(model_id, exclude=['*.pth', '*.pt'])

    def __init__(self):
        from openai import AsyncOpenAI

        self.openai = AsyncOpenAI(base_url='http://127.0.0.1:3000/v1', api_key='dummy')

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
        for key, value in ENGINE_CONFIG.items():
            setattr(args, key, value)

        router = fastapi.APIRouter(lifespan=vllm_api_server.lifespan)
        OPENAI_ENDPOINTS = [
            ['/chat/completions', vllm_api_server.create_chat_completion, ['POST']],
            ['/models', vllm_api_server.show_available_models, ['GET']],
        ]

        for route, endpoint, methods in OPENAI_ENDPOINTS:
            router.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)
        openai_api_app.include_router(router)

        self.engine_context = vllm_api_server.build_async_engine_client(args)
        self.engine = await self.engine_context.__aenter__()
        self.model_config = await self.engine.get_model_config()
        self.tokenizer = await self.engine.get_tokenizer()
        args.enable_auto_tool_choice = True
        args.tool_call_parser = 'llama3_json'

        await vllm_api_server.init_app_state(self.engine, self.model_config, openai_api_app.state, args)

    @bentoml.on_shutdown
    async def teardown_engine(self):
        await self.engine_context.__aexit__(GeneratorExit, None, None)

    @bentoml.api
    async def generate(
        self,
        prompt: str = 'Who are you? Please respond in pirate speak!',
        max_tokens: typing_extensions.Annotated[
            int, annotated_types.Ge(128), annotated_types.Le(MAX_TOKENS)
        ] = MAX_TOKENS,
    ) -> typing.AsyncGenerator[str, None]:
        from vllm import SamplingParams, TokensPrompt
        from vllm.entrypoints.chat_utils import parse_chat_messages, apply_hf_chat_template

        params = SamplingParams(max_tokens=max_tokens)
        messages = [dict(role='user', content=[dict(type='text', text=prompt)])]
        conversation, _ = parse_chat_messages(messages, self.model_config, self.tokenizer, content_format='string')
        prompt = TokensPrompt(
            prompt_token_ids=apply_hf_chat_template(
                self.tokenizer,
                conversation=conversation,
                add_generation_prompt=True,
                continue_final_message=False,
                chat_template=None,
                tokenize=True,
            )
        )

        stream = self.engine.generate(request_id=uuid.uuid4().hex, prompt=prompt, sampling_params=params)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
