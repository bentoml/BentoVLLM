from __future__ import annotations

import base64, io, json, logging, contextlib, traceback, typing, uuid
import bentoml, pydantic, fastapi, PIL.Image, typing_extensions, annotated_types

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
    bentovllm_model_id: str = 'meta-llama/Llama-3.2-11B-Vision-Instruct'
    bentovllm_max_tokens: int = 8192
    bentovllm_full_cudagraph: bool = False
    bentovllm_use_cudagraph: bool = True

    disable_log_requests: bool = True
    max_log_len: int = 1000
    request_logger: typing.Any = None
    disable_log_stats: bool = True
    use_tqdm_on_load: bool = False
    task: TaskOption = 'generate'
    enforce_eager: bool = True
    limit_mm_per_prompt: dict[str, typing.Any] = {'image': 1}
    max_model_len: int = 16384
    max_num_seqs: int = 16
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = 'pythonic'
    tensor_parallel_size: int = 1

    @pydantic.model_serializer
    def serialize_model(self) -> dict[str, typing.Any]:
        return {k: getattr(self, k) for k in self.__class__.model_fields if not k.startswith('bentovllm_')}


bento_args = bentoml.use_arguments(BentoArgs)
openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='llama3.2-11b-vision-instruct',
    traffic={'timeout': 300},
    resources={'gpu': bento_args.tensor_parallel_size, 'gpu_type': 'nvidia-h100-80gb'},
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
        'hf_generation_config': '{"temperature": 0.6, "top_p": 0.9}',
        'reasoning': '0',
        'tool': 'pythonic',
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
        bento_args.bentovllm_model_id, exclude=['original', '*.pth', '*.pt', 'original/**/*']
    )

    def __init__(self):
        from openai import AsyncOpenAI

        self.openai = AsyncOpenAI(base_url='http://127.0.0.1:3000/v1', api_key='dummy')
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
        args.compilation_config = {
            'level': 3,
            'use_cudagraph': bento_args.bentovllm_use_cudagraph,
            'cudagraph_capture_sizes': [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32, 24, 16, 8, 4, 2, 1],
            'cudagraph_num_of_warmups': 1,
            'full_cuda_graph': bento_args.bentovllm_full_cudagraph,
        }
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

    @bentoml.api
    async def sights(
        self,
        prompt: str = 'Describe the content of the picture',
        image: typing.Optional['PIL.Image.Image'] = None,
        max_tokens: typing_extensions.Annotated[
            int, annotated_types.Ge(128), annotated_types.Le(bento_args.bentovllm_max_tokens)
        ] = bento_args.bentovllm_max_tokens,
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
                model=bento_args.bentovllm_model_id, messages=messages, stream=True, max_tokens=max_tokens
            )
            async for chunk in completion:
                yield chunk.choices[0].delta.content or ''
        except Exception:
            logger.error(traceback.format_exc())
            yield 'Internal error found. Check server logs for more information'
            return
