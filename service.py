import bentoml

import uuid
from typing import Optional, AsyncGenerator, List

MAX_TOKENS = 1024
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>

{user_prompt} [/INST] """

@bentoml.service(
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
            model='meta-llama/Llama-2-7b-chat-hf',
            max_model_len=MAX_TOKENS
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate(self, prompt: str = "Explain superconductors like I'm five years old", tokens: Optional[List[int]] = None) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(max_tokens=MAX_TOKENS)
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, prompt, SAMPLING_PARAM, prompt_token_ids=tokens)
        async for request_output in stream:
            yield request_output.outputs[0].text
