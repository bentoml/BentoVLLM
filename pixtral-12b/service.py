import uuid
from typing import AsyncGenerator, Optional

import bentoml
import PIL.Image
from annotated_types import Ge, Le
from typing_extensions import Annotated

from bentovllm_openai.utils import openai_endpoints


MAX_TOKENS = 1024

SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

PROMPT_TEMPLATE = """<s>[INST]{system_prompt}
{user_prompt} [/INST] """

MODEL_ID = "mistral-community/pixtral-12b-240910"

# Hardcoded image_token_id and patch_size for now
IMAGE_TOKEN_ID = 10
PATCH_SIZE = 16
MAX_IMAGE_SIZE = 640


def resize(image: PIL.Image.Image, max_size: int = MAX_IMAGE_SIZE):
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size/image.width, max_size/image.height)
        width = int(image.width * ratio)
        height = int(image.height * ratio)
        image = image.resize((width, height))

    return image

@openai_endpoints(model_id=MODEL_ID)
@bentoml.service(
    name="pixtral-12b-service",
    traffic={
        "timeout": 300,
        "concurrency": 256, # Matches the default max_num_seqs in the VLLM engine
    },
    resources={
        "gpu": 1,
        "gpu_type": "nvidia-a100-80gb",
    },
)
class VLLM:
    def __init__(self) -> None:
        from vllm import AsyncEngineArgs, AsyncLLMEngine, LLM
        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            tokenizer_mode="mistral",
            enable_prefix_caching=False,
            limit_mm_per_prompt=dict(image=4),
            max_num_batched_tokens=16384,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)
        self.tokenizer = None

    @bentoml.api
    async def generate(
        self,
        image: PIL.Image.Image,
        prompt: str = "Describe this image",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
        system_prompt: Optional[str] = SYSTEM_PROMPT,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams, TokensPrompt
        from vllm.multimodal import MultiModalDataBuiltins

        tokenizer = await self.engine.get_tokenizer()

        SAMPLING_PARAM = SamplingParams(max_tokens=max_tokens)

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        prompt = PROMPT_TEMPLATE.format(user_prompt=prompt, system_prompt=system_prompt)
        token_ids = tokenizer(prompt).input_ids

        engine_inputs = TokensPrompt(prompt_token_ids=token_ids)

        image = resize(image)
        mm_data = MultiModalDataBuiltins(image=[image])
        engine_inputs["multi_modal_data"] = mm_data
        image_size = image.width * image.height
        image_token_num = image_size // (PATCH_SIZE ** 2)
        image_token_ids = [IMAGE_TOKEN_ID] * image_token_num
        engine_inputs["prompt_token_ids"] = image_token_ids + engine_inputs["prompt_token_ids"]

        stream = await self.engine.add_request(uuid.uuid4().hex, engine_inputs, SAMPLING_PARAM)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)
