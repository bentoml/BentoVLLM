from __future__ import annotations

import logging, contextlib, typing
import bentoml, pydantic, fastapi, typing_extensions, annotated_types

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

    class Args(EngineArgs, pydantic.BaseModel):
        pass

else:
    Args = pydantic.BaseModel


class BentoArgs(Args):
    bentovllm_model_id: str = 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'
    bentovllm_max_tokens: int = 4096

    disable_log_requests: bool = True
    max_log_len: int = 1000
    request_logger: typing.Any = None
    disable_log_stats: bool = True
    use_tqdm_on_load: bool = False
    task: str = 'generate'
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
    envs=[
        {'name': 'HF_TOKEN'},
        {'name': 'VLLM_ATTENTION_BACKEND', 'value': 'FLASH_ATTN'},
        {'name': 'VLLM_USE_V1', 'value': '1'},
    ],
    labels={'owner': 'bentoml-team', 'type': 'prebuilt', 'project': 'bentovllm'},
    image=bentoml.images.Image(python_version='3.11')
    .requirements_file('requirements.txt')
    .run('uv pip install --compile-bytecode vllm --pre --extra-index-url https://wheels.vllm.ai/nightly')
    .run('uv pip install --compile-bytecode flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.6'),
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
