from __future__ import annotations

import logging, typing, uuid
import bentoml, fastapi, typing_extensions, annotated_types

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENGINE_CONFIG = {
    'model': 'ai21labs/AI21-Jamba-1.5-Mini',
    'max_model_len': 204800,
    'tensor_parallel_size': 2,
    'enable_prefix_caching': False,
}
MAX_TOKENS = 4096

openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='bentovllm-jamba1.5-mini-service',
    traffic={'timeout': 300},
    resources={'gpu': 2, 'gpu_type': 'nvidia-a100-80gb'},
    envs=[{'name': 'HF_TOKEN'}, {'name': 'UV_NO_BUILD_ISOLATION', 'value': 1}],
    labels={'owner': 'bentoml-team', 'type': 'prebuilt'},
    image=bentoml.images.PythonImage(python_version='3.11')
    .system_packages('curl')
    .system_packages('git')
    .requirements_file('requirements.txt')
    .run('uv pip install --compile-bytecode torch')
    .run(
        'curl -L -o ./causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
    )
    .run(
        'uv pip install --compile-bytecode ./causal_conv1d-1.5.0.post8+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
    )
    .run(
        'curl -L -o ./mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'
    )
    .run('uv pip install --compile-bytecode ./mamba_ssm-2.2.4+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl'),
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
        args.enable_auto_tool_choice = True
        args.tool_call_parser = 'jamba'

        await vllm_api_server.init_app_state(self.engine, self.model_config, openai_api_app.state, args)

    @bentoml.api
    async def generate(
        self,
        prompt: str = 'Who are you? Please respond in pirate speak!',
        max_tokens: typing_extensions.Annotated[
            int, annotated_types.Ge(128), annotated_types.Le(MAX_TOKENS)
        ] = MAX_TOKENS,
    ) -> typing.AsyncGenerator[str, None]:
        from vllm import SamplingParams
        from vllm.entrypoints.chat_utils import parse_chat_messages, apply_hf_chat_template

        params = SamplingParams(max_tokens=max_tokens)
        messages = [dict(role='user', content=[dict(type='text', text=prompt)])]
        prompt = apply_hf_chat_template(
            self.tokenizer,
            conversation=parse_chat_messages(messages, self.model_config, self.tokenizer, content_format='string')[0],
            add_generation_prompt=True,
            continue_final_message=False,
            chat_template=None,
        )

        stream = await self.engine.add_request(uuid.uuid4().hex, prompt=prompt, params=params)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
