import uuid
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

from bentovllm_openai.utils import openai_endpoints


MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<start_of_turn>user
{system_prompt}
{user_prompt}<end_of_turn>
<start_of_turn>model
"""

MODEL_ID = "google/gemma-7b-it"

@openai_endpoints(model_id=MODEL_ID)
@bentoml.service(
    name="gemma-7b-service",
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-l4",
    },
)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_TOKENS,
            gpu_memory_utilization=0.95,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
            system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
