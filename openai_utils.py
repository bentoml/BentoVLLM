from __future__ import annotations

import asyncio
import typing as t

import bentoml

from vllm import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, ChatCompletionResponse,
    ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice,
    ChatCompletionStreamResponse, ChatMessage, DeltaMessage, ErrorResponse,
    UsageInfo
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat

# https://github.com/vllm-project/vllm/issues/2683
class PatchedOpenAIServingChat(OpenAIServingChat):
    def __init__(
        self,
        engine: AsyncLLMEngine,
        served_model: str,
        response_role: str,
        chat_template=None,
    ):
        super(OpenAIServingChat, self).__init__(engine=engine, served_model=served_model)
        self.response_role = response_role
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        if event_loop is not None and event_loop.is_running():
            event_loop.create_task(self._load_chat_template(chat_template))
        else:
            asyncio.run(self._load_chat_template(chat_template))

    async def _load_chat_template(self, chat_template):
        # Simply making this function async is usually already enough to give the parent
        # class time to load the tokenizer (so usually no sleeping happens here)
        # However, it feels safer to be explicit about this since asyncio does not
        # guarantee the order in which scheduled tasks are run
        while self.tokenizer is None:
            await asyncio.sleep(0.1)
        return super()._load_chat_template(chat_template)


def demo_deco(cls):

    async def create_chat_completion(
            self: cls,
            messages: t.Union[str, t.List[t.Dict[str, str]]],
    ) -> dict:


        request = ChatCompletionRequest(messages=messages, model="test_model")

        openai_serving_chat = getattr(self, "_openai_serving_chat", None)
        if openai_serving_chat is None:
            self._openai_serving_chat = PatchedOpenAIServingChat(
                self.engine, "test_model",
                "assistant",
            )
            openai_serving_chat = self._openai_serving_chat

        models = await openai_serving_chat.show_available_models()

        # tmp hack
        class DummyRequest:
            async def is_disconnected(self) -> bool:
                return False
        raw_request = DummyRequest()

        generator = await openai_serving_chat.create_chat_completion(
            request, raw_request)

        return generator.model_dump()

    wrapper = bentoml.api(route="/v1/chat/completions")
    cls.create_chat_completion = wrapper(create_chat_completion)
    return cls

