from __future__ import annotations

import asyncio
import typing as t

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest, CompletionRequest, ErrorResponse
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

from  _bentoml_sdk.service.factory import Service

T = t.TypeVar("T", bound=object)


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


def openai_deco(
        served_model: str,
        response_role: str ="assistant",
        chat_template: t.Optional[str] = None,
):

    def openai_wrapper(svc: Service[T]):

        cls = svc.inner
        app = FastAPI()

        class new_cls(cls):

            def __init__(self):

                from fastapi import Depends, FastAPI, Request
                from fastapi.responses import JSONResponse, StreamingResponse

                super().__init__()

                self.openai_serving_completion = OpenAIServingCompletion(
                    engine=self.engine, served_model=served_model,
                )
                self.openai_serving_chat = PatchedOpenAIServingChat(
                    engine=self.engine,
                    served_model=served_model,
                    response_role=response_role,
                    chat_template=chat_template,
                )

                @app.get("/v1/models")
                async def show_available_models():
                    models = await self.openai_serving_chat.show_available_models()
                    return JSONResponse(content=models.model_dump())

                @app.post("/v1/chat/completions")
                async def create_chat_completion(
                        request: ChatCompletionRequest,
                        raw_request: Request
                ):
                    generator = await self.openai_serving_chat.create_chat_completion(
                        request, raw_request)
                    if isinstance(generator, ErrorResponse):
                        return JSONResponse(content=generator.model_dump(),
                                            status_code=generator.code)
                    if request.stream:
                        return StreamingResponse(content=generator,
                                                 media_type="text/event-stream")
                    else:
                        return JSONResponse(content=generator.model_dump())

                @app.post("/v1/completions")
                async def create_completion(request: CompletionRequest, raw_request: Request):
                    generator = await self.openai_serving_completion.create_completion(
                        request, raw_request)
                    if isinstance(generator, ErrorResponse):
                        return JSONResponse(content=generator.model_dump(),
                                            status_code=generator.code)
                    if request.stream:
                        return StreamingResponse(content=generator,
                                                 media_type="text/event-stream")
                    else:
                        return JSONResponse(content=generator.model_dump())

        svc.inner = new_cls
        svc.mount_asgi_app(app)
        return svc

    return openai_wrapper
