from __future__ import annotations

import logging, contextlib, typing, uuid
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
    bentovllm_model_id: str = 'deepseek-ai/DeepSeek-V3-0324'
    bentovllm_max_tokens: int = 2048

    disable_log_requests: bool = True
    max_log_len: int = 1000
    request_logger: typing.Any = None
    disable_log_stats: bool = True
    use_tqdm_on_load: bool = False
    task: TaskOption = 'generate'
    max_model_len: int = 4096
    max_num_seqs: int = 128
    tensor_parallel_size: int = 8

    @pydantic.model_serializer
    def serialize_model(self) -> dict[str, typing.Any]:
        return {k: getattr(self, k) for k in self.__class__.model_fields if not k.startswith('bentovllm_')}


bento_args = bentoml.use_arguments(BentoArgs)
openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='deepseek-v3-671b',
    traffic={'timeout': 300},
    resources={'gpu': bento_args.tensor_parallel_size, 'gpu_type': 'nvidia-h200-141gb'},
    envs=[
        {'name': 'HF_TOKEN'},
        {'name': 'UV_NO_PROGRESS', 'value': '1'},
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': 'FLASHMLA'},
    ],
    labels={
        'owner': 'bentoml-team',
        'type': 'prebuilt',
        'project': 'bentovllm',
        'openai_endpoint': '/v1',
        'hf_generation_config': '{"temperature": 0.6, "top_p": 0.9}',
    },
    image=bentoml.images.Image(python_version='3.11', lock_python_packages=True)
    .requirements_file('requirements.txt')
    .run(
        'uv pip install --compile-bytecode --no-progress https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl'
    ),
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

        self.engine = await self.exit_stack.enter_async_context(vllm_api_server.build_async_engine_client(args))
        self.tokenizer = await self.engine.get_tokenizer()
        vllm_config = await self.engine.get_vllm_config()
        self.model_config = await self.engine.get_model_config()
        await vllm_api_server.init_app_state(self.engine, vllm_config, openai_api_app.state, args)

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
        from vllm.entrypoints.chat_utils import parse_chat_messages, apply_hf_chat_template

        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]

        params = SamplingParams(max_tokens=max_tokens)
        conversation, _ = parse_chat_messages(messages, self.model_config, self.tokenizer, content_format='string')
        prompt = TokensPrompt(
            prompt_token_ids=apply_hf_chat_template(
                self.tokenizer,
                conversation=conversation,
                tools=None,
                add_generation_prompt=True,
                continue_final_message=False,
                chat_template=None,
                tokenize=True,
                model_config=self.model_config,
            )
        )

        stream = self.engine.generate(request_id=uuid.uuid4().hex, prompt=prompt, sampling_params=params)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
