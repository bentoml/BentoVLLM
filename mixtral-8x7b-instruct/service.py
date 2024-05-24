import uuid
from typing import AsyncGenerator

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

from bentovllm_openai.utils import openai_endpoints


MAX_TOKENS = 8192
PROMPT_TEMPLATE = """<s>[INST]
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.

{user_prompt} [/INST] """

MODEL_ID = "TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ"

@openai_endpoints(served_model=MODEL_ID)
@bentoml.service(
    name="mixtral-8x7b-instruct-gptq-service",
    traffic={
        "timeout": 300,
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_TOKENS,
            quantization="gptq",
            dtype="half",
            enable_prefix_caching=True
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate(
        self,
        prompt: str = "Explain superconductors like I'm five years old",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
