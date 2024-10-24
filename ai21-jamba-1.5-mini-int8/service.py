import os
import uuid
from argparse import Namespace
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

import fastapi
openai_api_app = fastapi.FastAPI()

MAX_MODEL_LEN = 100*1024
MAX_TOKENS = 4096

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

MODEL_ID = "ai21labs/AI21-Jamba-1.5-Mini"


@bentoml.mount_asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    name="bentovllm-llama3.1-70b-instruct-awq-service",
    traffic={
        "timeout": 1200,
        "concurrency": 256,  # Matches the default max_num_seqs in the VLLM engine
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class VLLM:

    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        import vllm.entrypoints.openai.api_server as vllm_api_server
        from vllm.entrypoints.openai.api_server import init_app_state

        os.environ['VLLM_FUSED_MOE_CHUNK_SIZE']='32768'

        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_MODEL_LEN,
            quantization="experts_int8",
            enable_prefix_caching=False,  # prefix caching on jamba is not supported yet
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

        OPENAI_ENDPOINTS = [
            ["/chat/completions", vllm_api_server.create_chat_completion, ["POST"]],
            ["/completions", vllm_api_server.create_completion, ["POST"]],
            ["/models", vllm_api_server.show_available_models, ["GET"]],
        ]

        for route, endpoint, methods in OPENAI_ENDPOINTS:
            openai_api_app.add_api_route(
                path=route,
                endpoint=endpoint,
                methods=methods,
            )

        model_config = self.engine.engine.get_model_config()
        args = Namespace()
        args.model = MODEL_ID
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.response_role = "assistant"
        args.served_model_name = None
        args.chat_template = None
        args.lora_modules = None
        args.prompt_adapters = None
        args.request_logger = None
        args.disable_log_stats = True
        args.return_tokens_as_token_ids = False
        args.enable_tool_call_parser = False
        args.enable_auto_tool_choice = False
        args.tool_call_parser = "jamba"

        vllm_api_server.init_app_state(
            self.engine, model_config, openai_api_app.state, args
        )


    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors in plain English",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
        )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
