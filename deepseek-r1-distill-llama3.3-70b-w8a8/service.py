from __future__ import annotations

import base64, io, logging, traceback, typing, argparse, asyncio, os
import bentoml, fastapi, PIL.Image

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ENGINE_CONFIG = {"model": "neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8", "max_model_len": 4096}
SERVICE_CONFIG = {
    "name": "bentovllm-r1-llama3.3-70b-w8a8-service",
    "traffic": {"timeout": 300},
    "resources": {"gpu": 1, "gpu_type": "nvidia-a100-80gb"},
    "envs": [{"name": "UV_COMPILE_BYTECODE", "value": 1}],
}
SERVER_CONFIG = {"reasoning_parser": "deepseek_r1"}

openai_api_app = fastapi.FastAPI()


@bentoml.asgi_app(openai_api_app, path="/v1")
@bentoml.service(
    **SERVICE_CONFIG,
    labels={"owner": "bentoml-team", "type": "prebuilt"},
    image=bentoml.images.PythonImage(python_version="3.11").requirements_file("requirements.txt"),
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
            openai_api_app.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)

        # max_model_len
        if (max_model_len := os.getenv("MAX_MODEL_LEN")) is not None:
            try:
                ENGINE_CONFIG["max_model_len"] = int(max_model_len)
                logger.info(
                    f"Updated `max_model_len` to {max_model_len} from environment variable. Make sure that you have enough VRAM to use this given context windows."
                )
            except ValueError:
                logger.warning(f"Invalid MAX_MODEL_LEN value: {max_model_len}. Must be an integer.")

        # reasoning
        if (reasoning := os.getenv("REASONING")) is not None:
            SERVER_CONFIG["enable_reasoning"] = reasoning.lower() in ("1", "true", "y", "yes")
            logger.info("Enable reasoning. This might not work with structured decoding.")

        ENGINE_ARGS = AsyncEngineArgs(**dict(ENGINE_CONFIG, model=self.model, enable_prefix_caching=True))
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

        model_config = self.engine.engine.get_model_config()
        args = argparse.Namespace()
        args.model = self.model
        args.disable_log_requests = True
        args.max_log_len = 1000
        args.response_role = "assistant"
        args.served_model_name = [self.model_id]
        args.chat_template = None
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
    async def generate(
        self, prompt: str = "Who are you? Please respond in pirate speak!"
    ) -> typing.AsyncGenerator[str, None]:
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
