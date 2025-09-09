from __future__ import annotations

import bentoml, json
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
