import uuid
from typing import AsyncGenerator, Optional, List

import bentoml
import PIL.Image
from annotated_types import Ge, Le
from typing_extensions import Annotated

from bentovllm_openai.utils import openai_endpoints

MAX_TOKENS = 1024
MAX_IMAGE_SIZE = 640

SYSTEM_PROMPT = """
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
"""

PROMPT_TEMPLATE = """<s>[INST]{system_prompt}
{user_prompt} [/INST] """

MODEL_ID = "mistral-community/pixtral-12b-240910"


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
        from vllm import AsyncEngineArgs, AsyncLLMEngine
        ENGINE_ARGS = AsyncEngineArgs(
            model=MODEL_ID,
            tokenizer_mode="mistral",
            enable_prefix_caching=True,
            enable_chunked_prefill=False,
            limit_mm_per_prompt=dict(image=1),
            max_model_len=16384,
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(ENGINE_ARGS)

    @bentoml.api
    async def generate(
        self,
        image: PIL.Image.Image,
        prompt: str = "Describe this image",
        max_tokens: Annotated[int, Ge(128), Le(MAX_TOKENS)] = MAX_TOKENS,
        system_prompt: Optional[str] = SYSTEM_PROMPT,
    ) -> AsyncGenerator[str, None]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(max_tokens=max_tokens)

        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT

        engine_inputs = await self.create_image_input([image], prompt, system_prompt)
        stream = await self.engine.add_request(uuid.uuid4().hex, engine_inputs, sampling_params)

        cursor = 0
        async for request_output in stream:
            text = request_output.outputs[0].text
            yield text[cursor:]
            cursor = len(text)


    async def create_image_input(
            self, images: List[PIL.Image.Image], prompt: str, system_prompt: str
    ):
        from vllm import SamplingParams, TokensPrompt
        from vllm.multimodal import MultiModalDataBuiltins
        from mistral_common.protocol.instruct.messages import (
            SystemMessage,
            UserMessage,
            TextChunk,
            ImageChunk,
        )
        from mistral_common.protocol.instruct.request import ChatCompletionRequest

        tokenizer = await self.engine.get_tokenizer()
        tokenizer = tokenizer.mistral

        # tokenize images and text
        messages = [
            UserMessage(
                content=[
                    TextChunk(text=prompt),
                ] + [ImageChunk(image=img) for img in images]
            )
        ]

        if system_prompt:
            system_message = SystemMessage(content=[TextChunk(text=system_prompt)])
            messages = [system_message] + messages

        tokenized = tokenizer.encode_chat_completion(
            ChatCompletionRequest(
                messages=messages,
                model="pixtral",
            )
        )

        engine_inputs = TokensPrompt(prompt_token_ids=tokenized.tokens)

        mm_data = MultiModalDataBuiltins(image=images)
        engine_inputs["multi_modal_data"] = mm_data

        return engine_inputs
