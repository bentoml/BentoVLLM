from __future__ import annotations

import base64, io, logging, contextlib, traceback, typing, uuid
import bentoml, pydantic, fastapi, PIL.Image, typing_extensions, annotated_types

logger = logging.getLogger(__name__)
SYSTEM_PROMPT = """You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
You power an AI assistant called Le Chat.
Your knowledge base was last updated on 2023-10-01.
The current date is {today}.

When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. "What are some good restaurants around me?" => "Where are you?" or "When is the next flight to Tokyo" => "Where do you travel from?").
You are always very attentive to dates, in particular you try to resolve dates (e.g. "yesterday" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.
You follow these instructions in all languages, and always respond to the user in the language they use or request.
Next sections describe the capabilities that you have.

# WEB BROWSING INSTRUCTIONS

You cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.

# MULTI-MODAL INSTRUCTIONS

You have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.
You cannot read nor transcribe audio files or videos.
"""

if typing.TYPE_CHECKING:
    from vllm.engine.arg_utils import EngineArgs
    from vllm.config import TaskOption

    class Args(EngineArgs, pydantic.BaseModel):
        pass

else:
    TaskOption = str
    Args = pydantic.BaseModel


class BentoArgs(Args):
    bentovllm_model_id: str = 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'
    bentovllm_max_tokens: int = 4096

    disable_log_requests: bool = True
    max_log_len: int = 1000
    request_logger: typing.Any = None
    disable_log_stats: bool = True
    use_tqdm_on_load: bool = False
    task: TaskOption = 'generate'
    tokenizer_mode: str = 'mistral'
    config_format: str = 'mistral'
    load_format: str = 'mistral'
    max_model_len: int = 8192
    max_num_seqs: int = 256
    limit_mm_per_prompt: dict[str, typing.Any] = {'image': 10}
    enable_prefix_caching: bool = False
    enable_auto_tool_choice: bool = True
    tool_call_parser: str = 'mistral'
    tensor_parallel_size: int = 2

    @pydantic.model_serializer
    def serialize_model(self) -> dict[str, typing.Any]:
        return {k: getattr(self, k) for k in self.__class__.model_fields if not k.startswith('bentovllm_')}


bento_args = bentoml.use_arguments(BentoArgs)
openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path='/v1')
@bentoml.service(
    name='mistral-small-3.1-24b-instruct-2503',
    traffic={'timeout': 300},
    resources={'gpu': bento_args.tensor_parallel_size, 'gpu_type': 'nvidia-a100-80gb'},
    envs=[{'name': 'HF_TOKEN'}, {'name': 'UV_NO_PROGRESS', 'value': '1'}],
    labels={
        'owner': 'bentoml-team',
        'type': 'prebuilt',
        'project': 'bentovllm',
        'openai_endpoint': '/v1',
        'hf_generation_config': '{"temperature": 0.7, "top_p": 0.95}',
        'hf_system_prompt': '"You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\\nYou power an AI assistant called Le Chat.\\nYour knowledge base was last updated on 2023-10-01.\\nThe current date is {today}.\\n\\nWhen you\'re not sure about some information, you say that you don\'t have the information and don\'t make up anything.\\nIf the user\'s question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \\"What are some good restaurants around me?\\" => \\"Where are you?\\" or \\"When is the next flight to Tokyo\\" => \\"Where do you travel from?\\").\\nYou are always very attentive to dates, in particular you try to resolve dates (e.g. \\"yesterday\\" is {yesterday}) and when asked about information at specific dates, you discard information that is at another date.\\nYou follow these instructions in all languages, and always respond to the user in the language they use or request.\\nNext sections describe the capabilities that you have.\\n\\n# WEB BROWSING INSTRUCTIONS\\n\\nYou cannot perform any web search or access internet to open URLs, links etc. If it seems like the user is expecting you to do so, you clarify the situation and ask the user to copy paste the text directly in the chat.\\n\\n# MULTI-MODAL INSTRUCTIONS\\n\\nYou have the ability to read images, but you cannot generate images. You also cannot transcribe audio files or videos.\\nYou cannot read nor transcribe audio files or videos.\\n"',
    },
    image=bentoml.images.Image(python_version='3.11', lock_python_packages=True)
    .requirements_file('requirements.txt')
    .run(
        'uv pip install --compile-bytecode -U vllm --extra-index-url https://wheels.vllm.ai/0.9.1rc1 --torch-backend=cu128'
    )
    .run(
        'uv pip install --compile-bytecode --no-progress https://download.pytorch.org/whl/cu128/flashinfer/flashinfer_python-0.2.6.post1%2Bcu128torch2.7-cp39-abi3-linux_x86_64.whl'
    ),
)
class VLLM:
    model = bentoml.models.HuggingFaceModel(
        bento_args.bentovllm_model_id, exclude=['model*', '*.pth', '*.pt', 'original/**/*']
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
        system_prompt: typing.Optional[str] = SYSTEM_PROMPT,
        max_tokens: typing_extensions.Annotated[
            int, annotated_types.Ge(128), annotated_types.Le(bento_args.bentovllm_max_tokens)
        ] = bento_args.bentovllm_max_tokens,
    ) -> typing.AsyncGenerator[str, None]:
        from vllm import SamplingParams, TokensPrompt
        from vllm.entrypoints.chat_utils import apply_mistral_chat_template

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': [{'type': 'text', 'text': prompt}]},
        ]

        params = SamplingParams(max_tokens=max_tokens)
        prompt = TokensPrompt(
            prompt_token_ids=apply_mistral_chat_template(
                self.tokenizer, messages=messages, chat_template=None, tools=None
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
        system_prompt: typing.Optional[str] = SYSTEM_PROMPT,
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
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': content}]

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
