from __future__ import annotations

import base64, io, logging, traceback, typing, argparse, asyncio
import bentoml, fastapi, PIL.Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENGINE_CONFIG = {"model": "google/gemma-2-27b-it", "max_model_len": 2048, "dtype": "half"}
SERVICE_CONFIG = {
    "name": "gemma2",
    "traffic": {"timeout": 300},
    "resources": {"gpu": 1, "gpu_type": "nvidia-a100-80gb"},
    "envs": [{"name": "VLLM_ATTENTION_BACKEND", "value": "FLASHINFER"}, {"name": "HF_TOKEN"}],
}
SERVER_CONFIG = {}
REQUIREMENTS_TXT = []


openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    **SERVICE_CONFIG,
    image=bentoml.images.PythonImage(python_version="3.11")
    .python_packages("vllm==0.7.1\n")
    .python_packages("pyyaml\n")
    .python_packages("Pillow\n")
    .python_packages("openai\n")
    .python_packages("bentoml>=1.3.20\n")
    .python_packages(*REQUIREMENTS_TXT),
)
class VLLM:
    model_id = ENGINE_CONFIG["model"]
    model = bentoml.models.HuggingFaceModel(model_id)

    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        import vllm.entrypoints.openai.api_server as vllm_api_server

        OPENAI_ENDPOINTS = [
            ["/chat/completions", vllm_api_server.create_chat_completion, ["POST"]],
            ["/completions", vllm_api_server.create_completion, ["POST"]],
            ["/embeddings", vllm_api_server.create_embedding, ["POST"]],
            ["/models", vllm_api_server.show_available_models, ["GET"]],
        ]
        for route, endpoint, methods in OPENAI_ENDPOINTS:
            openai_api_app.add_api_route(
                path=route,
                endpoint=endpoint,
                methods=methods,
                include_in_schema=True,
            )

        ENGINE_ARGS = AsyncEngineArgs(**dict(ENGINE_CONFIG, model=self.model))
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

        model_config = self.engine.engine.get_model_config()
        args = argparse.Namespace()
        args.model = self.model
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.response_role = "assistant"
        args.served_model_name = [self.model_id]
        args.chat_template = "{% if messages[0]['role'] == 'system' %}\n    {% set offset = 1 %}\n{% else %}\n    {% set offset = 0 %}\n{% endif %}\n\n{% for message in messages %}\n    {% if (message['role'] == 'user') != (loop.index0 % 2 == offset) %}\n        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}\n    {% endif %}\n\n    {{ '<|' + message['role'] + '|>\\n' + message['content'].strip() + '<|end|>' + '\\n' }}\n\n    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}\n        {{ '<|assistant|>\\n' }}\n    {% endif %}\n{% endfor %}\n"
        args.chat_template_content_format = "auto"
        args.lora_modules = None
        args.prompt_adapters = None
        args.request_logger = None
        args.disable_log_stats = True
        args.return_tokens_as_token_ids = False
        args.enable_tool_call_parser = False
        args.enable_auto_tool_choice = False
        args.tool_call_parser = None
        args.enable_prompt_tokens_details = False
        args.enable_reasoning = False
        args.reasoning_parser = None

        for key, value in SERVER_CONFIG.items():
            setattr(args, key, value)

        asyncio.create_task(vllm_api_server.init_app_state(self.engine, model_config, openai_api_app.state, args))

    @bentoml.api
    async def generate(self, prompt: str = "what is this?") -> typing.AsyncGenerator[str, None]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(base_url="http://127.0.0.1:3000/v1", api_key="dummy")

        try:
            completion = await client.chat.completions.create(
                model=self.model_id,
                messages=[dict(role="user", content=[dict(type="text", text=prompt)])],
                stream=True,
            )
            async for chunk in completion:
                yield chunk.choices[0].delta.content or ""
        except Exception:
            yield traceback.format_exc()
