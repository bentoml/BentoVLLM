import bentoml

from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
from typing import Optional, AsyncGenerator, List

MAX_TOKENS = 1024
SAMPLING_PARAM = SamplingParams(max_tokens=MAX_TOKENS)
ENGINE_ARGS = AsyncEngineArgs(
    model='meta-llama/Llama-2-7b-chat-hf',
    max_model_len=MAX_TOKENS
)

@bentoml.service(
    workers=1,
    tiemout=300,
    resources={
        "gpu": 1,
        "memory": "16Gi",
    },
)
class VLLMService:
    def __init__(self) -> None:
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.request_id = 0

    @bentoml.api
    async def generate(self, prompt: str = "Explain superconductors like I'm five years old", tokens: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
        stream = await self.engine.add_request(self.request_id, prompt, SAMPLING_PARAM, prompt_token_ids=tokens)
        self.request_id += 1
        async for request_output in stream:
            yield request_output.outputs[0].text
