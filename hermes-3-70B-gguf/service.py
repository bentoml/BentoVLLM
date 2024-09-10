import uuid
from typing import AsyncGenerator, Optional

import bentoml
from annotated_types import Ge, Le
from typing_extensions import Annotated

from bentovllm_openai.utils import openai_endpoints


MAX_TOKENS = 1024
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

MODEL_ID = "NousResearch/Hermes-3-Llama-3.1-70B-GGUF"
MODEL_FILENAME = "Hermes-3-Llama-3.1-70B.Q4_K_M.gguf"
TOKENIZER_ID = "NousResearch/Hermes-3-Llama-3.1-70B"

@openai_endpoints(model_id=MODEL_ID)
@bentoml.service(
    name="bentovllm-hermes-3-70b-gguf",
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
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
        from vllm import AsyncEngineArgs, AsyncLLMEngine

        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)

        model_path = hf_hub_download(MODEL_ID, filename=MODEL_FILENAME)


        ENGINE_ARGS = AsyncEngineArgs(
            model=model_path,
            max_model_len=MAX_TOKENS,
            enable_prefix_caching=True,
            tokenizer=TOKENIZER_ID,
            gpu_memory_utilization=0.85,
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)


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
