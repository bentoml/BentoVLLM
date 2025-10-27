from __future__ import annotations

import asyncio, json, logging
import bentoml, httpx

from components import Decoder, Prefiller, proxy
from components.config import PREFILL_GPU_TYPE

logger = logging.getLogger(__name__)

image = (
  bentoml.images.Image(python_version='3.10', lock_python_packages=False)
  .system_packages('curl', 'git', 'ninja-build')
  .requirements_file('requirements.txt')
  .python_packages('./wheels/sglang_router-0.1.9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl')
)
if 'b200' in PREFILL_GPU_TYPE:
  image = image.python_packages('./wheels/deep_gemm-2.1.1+c9f8b34-cp310-cp310-linux_x86_64.whl')


@bentoml.service(
  name='router',
  timeout=6000,
  envs=[
    {'name': 'VLLM_SKIP_P2P_CHECK', 'value': '1'},
    {'name': 'UV_NO_PROGRESS', 'value': '1'},
    {'name': 'UV_TORCH_BACKEND', 'value': 'cu128'},
  ],
  labels={
    'openai_endpoint': '/v1',
    'reasoning': '0',
    'tool': 'hermes',
    'hf_generation_config': json.dumps({'temperature': 0.6, 'top_p': 0.95}),
  },
  image=image,
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
