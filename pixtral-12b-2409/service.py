from __future__ import annotations

import base64, io, logging, os, contextlib, traceback, typing, uuid
import bentoml, fastapi, PIL.Image, typing_extensions, annotated_types

logger = logging.getLogger(__name__)

MAX_TOKENS = 4096
ENGINE_CONFIG = {
    'tokenizer_mode': 'mistral',
    'config_format': 'mistral',
    'load_format': 'mistral',
    'limit_mm_per_prompt': {'image': 5},
    'max_model_len': 32768,
    'enable_prefix_caching': False,
}

openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='bentovllm-pixtral-12b-2409-service',
    traffic={'timeout': 300},
    resources={'gpu': 1, 'gpu_type': 'nvidia-a100-80gb'},
    envs=[
        {'name': 'HF_TOKEN'},
        {'name': 'UV_NO_PROGRESS', 'value': '1'},
        {'name': 'HF_HUB_DISABLE_PROGRESS_BARS', 'value': '1'},
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': 'FLASH_ATTN'},
        {'name': 'VLLM_USE_V1', 'value': '0'},
        {'name': 'VLLM_LOGGING_CONFIG_PATH', 'value': os.path.join(os.path.dirname(__file__), 'logging-config.json')},
    ],
    labels={'owner': 'bentoml-team', 'type': 'prebuilt'},
    image=bentoml.images.PythonImage(python_version='3.11', lock_python_packages=False)
    .requirements_file('requirements.txt')
    .run('uv pip install --compile-bytecode flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.6'),
)
class VLLM:
    model_id = 'mistralai/Pixtral-12B-2409'
    model = bentoml.models.HuggingFaceModel(model_id, exclude=['model*', '*.pth', '*.pt', 'original/**/*'])

    def __init__(self):
        from openai import AsyncOpenAI

        self.openai = AsyncOpenAI(base_url='http://127.0.0.1:3000/v1', api_key='dummy')
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
        args.tool_call_parser = 'mistral'
        args.enable_auto_tool_choice = True

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

    @bentoml.api
    async def generate(
        self,
        prompt: str = 'Who are you? Please respond in pirate speak!',
        max_tokens: typing_extensions.Annotated[
            int, annotated_types.Ge(128), annotated_types.Le(MAX_TOKENS)
        ] = MAX_TOKENS,
    ) -> typing.AsyncGenerator[str, None]:
        from vllm import SamplingParams, TokensPrompt
        from vllm.entrypoints.chat_utils import apply_mistral_chat_template

        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]

        params = SamplingParams(max_tokens=max_tokens)
        prompt = TokensPrompt(prompt_token_ids=apply_mistral_chat_template(self.tokenizer, messages=messages))

        stream = self.engine.generate(request_id=uuid.uuid4().hex, prompt=prompt, sampling_params=params)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)

    @bentoml.api
    async def sights(
        self,
        prompt: str = 'Describe the content of the picture',
        image: typing.Optional['PIL.Image.Image'] = None,
        max_tokens: typing_extensions.Annotated[
            int, annotated_types.Ge(128), annotated_types.Le(MAX_TOKENS)
        ] = MAX_TOKENS,
    ) -> typing.AsyncGenerator[str, None]:
        if image:
            buffered = io.BytesIO()
            image.save(buffered, format='PNG')
            img_str = base64.b64encode(buffered.getvalue()).decode()
            buffered.close()
            image_url = f'data:image/png;base64,{img_str}'
            content = [dict(type='image_url', image_url=dict(url=image_url)), dict(type='text', text=prompt)]
        else:
            content = [dict(type='text', text=prompt)]
        messages = [{'role': 'user', 'content': content}]

        try:
            completion = await self.openai.chat.completions.create(
                model=self.model_id, messages=messages, stream=True, max_tokens=max_tokens
            )
            async for chunk in completion:
                yield chunk.choices[0].delta.content or ''
        except Exception:
            logger.error(traceback.format_exc())
            yield 'Internal error found. Check server logs for more information'
            return
