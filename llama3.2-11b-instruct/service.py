import uuid
from typing import AsyncGenerator, Optional

import bentoml
import PIL.Image
from annotated_types import Ge, Le
from typing_extensions import Annotated

from bentovllm_openai.utils import openai_endpoints


MAX_TOKENS = 1024
MAX_IMAGE_SIZE = 640
SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_prompt}<|image|><|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"


def resize(image: PIL.Image.Image, max_size: int = MAX_IMAGE_SIZE):
    if image.width > max_size or image.height > max_size:
        ratio = min(max_size / image.width, max_size / image.height)
        width = int(image.width * ratio)
        height = int(image.height * ratio)
        image = image.resize((width, height))

    return image


@openai_endpoints(
    model_id=MODEL_ID,
    default_chat_completion_parameters=dict(stop=["<|eot_id|>"]),
)
@bentoml.service(
    name="bentovllm-llama32-11b-vision-instruct-service",
    traffic={
        "timeout": 10000,
        "concurrency": 16,  # Matches the default max_num_seqs in the VLLM engine
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

        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            max_model_len=MAX_TOKENS,
            enable_prefix_caching=True,
            enforce_eager=True,
            max_num_seqs=16,
            limit_mm_per_prompt=dict(image=1),
        )

        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.stop_token_ids = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

    @bentoml.api
    async def generate(
        self,
        image: PIL.Image.Image,
        prompt: str = "Describe this image",
        system_prompt: Optional[str] = SYSTEM_PROMPT,
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        SAMPLING_PARAM = SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=self.stop_token_ids,
        )

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT
        engine_inputs = await self.create_image_inputs(
            dict(prompt=prompt, system_prompt=system_prompt, image=resize(image))
        )
        stream = await self.engine.add_request(
            uuid.uuid4().hex, engine_inputs, SAMPLING_PARAM
        )

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)

    async def create_image_inputs(self, inputs):
        from vllm import TextPrompt
        from vllm.multimodal import MultiModalDataBuiltins

        return TextPrompt(
            prompt=PROMPT_TEMPLATE.format(
                user_prompt=inputs["prompt"], system_prompt=inputs["system_prompt"]
            ),
            multi_modal_data=MultiModalDataBuiltins(image=inputs["image"]),
        )
