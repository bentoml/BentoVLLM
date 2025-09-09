from __future__ import annotations

import asyncio, json
import bentoml, httpx
from components import Decoder, Prefiller, proxy


@bentoml.service(
  labels={
    'openai_endpoint': '/v1',
    'reasoning': '0',
    'tool': 'hermes',
    'hf_generation_config': json.dumps({'temperature': 0.6, 'top_p': 0.95}),
  },
  image=bentoml.images.Image().requirements_file('requirements.txt'),
)
@bentoml.asgi_app(proxy)
class Router:
  decoder = bentoml.depends(Decoder)
  prefiller = bentoml.depends(Prefiller)

  async def __is_ready__(self) -> bool:
    # Keep P & D running when the router is live.
    async with httpx.AsyncClient() as client:
      responses = await asyncio.gather(
        # Carry the headers so that runner-lb can route to the right service.
        client.get(f'{Prefiller.url}/health', headers={'Runner-Name': 'Prefiller'}),
        client.get(f'{Decoder.url}/health', headers={'Runner-Name': 'Decoder'}),
        return_exceptions=True,
      )
      return all(not isinstance(r, BaseException) and r.is_success for r in responses)
