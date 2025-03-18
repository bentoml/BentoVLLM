import os
import uuid
from argparse import Namespace
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

import fastapi
openai_api_app = fastapi.FastAPI()

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 8 * 1024))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 4 * 1024))

SYSTEM_PROMPT = """You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.
Your knowledge base was last updated on 2023-10-01. 
When you're not sure about some information, you say that you don't have the information and don't make up anything.
If the user's question is not clear, ambiguous, or does not provide enough context for you to accurately answer the question, you do not try to answer it right away and you rather ask the user to clarify their request (e.g. \"What are some good restaurants around me?\" => \"Where are you?\" or \"When is the next flight to Tokyo\" => \"Where do you travel from?\")"""

MODEL_ID = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"


@bentoml.asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="bentovllm-mistral-small-3.1-service",
    image=bentoml.images.PythonImage(python_version='3.11', lock_python_packages=False)\
    .requirements_file('requirements.txt')\
    .run('uv pip install flashinfer-python --find-links https://flashinfer.ai/whl/cu124/torch2.5'),
    traffic={
        "timeout": 60,
        "concurrency": 20,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class VLLM:
    model = bentoml.models.HuggingFaceModel(MODEL_ID)

    @bentoml.on_startup
    async def init_engine(self) -> None:
        import vllm.entrypoints.openai.api_server as vllm_api_server

        from vllm.utils import FlexibleArgumentParser
        from vllm.entrypoints.openai.cli_args import make_arg_parser

        args = make_arg_parser(FlexibleArgumentParser()).parse_args([])
        args.model = self.model
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.served_model_name = [MODEL_ID]
        args.request_logger = None
        args.disable_log_stats = True
        args.enable_prefix_caching = True
        args.tokenizer_mode = "mistral"
        args.config_format = "mistral"
        args.load_format = "mistral"
        args.max_model_len = MAX_MODEL_LEN
        args.enable_tool_call_parser = True
        args.enable_auto_tool_choice = True
        args.tool_call_parser = "mistral"

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

        await vllm_api_server.init_app_state(self.engine, self.model_config, openai_api_app.state, args)


    @bentoml.on_shutdown
    async def teardown_engine(self):
        await self.engine_context.__aexit__(GeneratorExit, None, None)


    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams, TokensPrompt
        from vllm.entrypoints.chat_utils import apply_mistral_chat_template

        SAMPLING_PARAMS = SamplingParams(max_tokens=max_tokens)

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        prompt = TokensPrompt(prompt_token_ids=apply_mistral_chat_template(self.tokenizer, messages=messages))
        stream = self.engine.generate(prompt=prompt, sampling_params=SAMPLING_PARAMS, request_id=uuid.uuid4().hex)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
