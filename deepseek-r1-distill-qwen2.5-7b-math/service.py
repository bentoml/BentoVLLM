from __future__ import annotations

import logging, traceback, typing
import bentoml, fastapi, typing_extensions, annotated_types

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENGINE_CONFIG = {
    'model': 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    'max_model_len': 8192,
    'enable_prefix_caching': True,
}
MAX_TOKENS = 4096

openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='bentovllm-r1-qwen2.5-7b-math-service',
    traffic={'timeout': 300},
    resources={'gpu': 1, 'gpu_type': 'nvidia-l4'},
    envs=[{'name': 'HF_TOKEN'}],
    labels={'owner': 'bentoml-team', 'type': 'prebuilt'},
    image=bentoml.images.PythonImage(python_version='3.11').requirements_file('requirements.txt'),
)
class VLLM:
    model_id = ENGINE_CONFIG['model']
    model = bentoml.models.HuggingFaceModel(model_id, exclude=['*.pth', '*.pt'])

    @bentoml.on_startup
    async def init_openai_app(self) -> None:
        import vllm.entrypoints.openai.api_server as vllm_api_server

        from vllm import AsyncEngineArgs, AsyncLLMEngine
        from vllm.utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser

        OPENAI_ENDPOINTS = [
            ['/chat/completions', vllm_api_server.create_chat_completion, ['POST']],
            ['/models', vllm_api_server.show_available_models, ['GET']],
        ]
        for route, endpoint, methods in OPENAI_ENDPOINTS:
            openai_api_app.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)

        ENGINE_ARGS = AsyncEngineArgs(**dict(ENGINE_CONFIG, model=self.model))
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.model_config = self.engine.engine.get_model_config()
        self.tokenizer = await self.engine.engine.get_tokenizer_async()

        args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
        args.model = self.model
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.served_model_name = [self.model_id]
        args.request_logger = None
        args.disable_log_stats = True
        args.enable_reasoning = True
        args.enable_auto_tool_choice = True
        args.reasoning_parser = 'deepseek_r1'
        args.tool_call_parser = 'hermes'

        await vllm_api_server.init_app_state(self.engine, self.model_config, openai_api_app.state, args)

    @bentoml.api
    async def generate(
        self,
        prompt: str = 'Who are you? Please respond in pirate speak!',
        max_tokens: typing_extensions.Annotated[
            int, annotated_types.Ge(128), annotated_types.Le(MAX_TOKENS)
        ] = MAX_TOKENS,
    ) -> typing.AsyncGenerator[str, None]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url='http://127.0.0.1:3000/v1', api_key='dummy')
        try:
            completion = await client.chat.completions.create(
                model=self.model_id,
                messages=[dict(role='user', content=[dict(type='text', text=prompt)])],
                stream=True,
                max_tokens=max_tokens,
            )
            async for chunk in completion:
                yield chunk.choices[0].delta.content or ''
        except Exception:
            logger.error(traceback.format_exc())
            yield 'Internal error found. Check server logs for more information'
            return
