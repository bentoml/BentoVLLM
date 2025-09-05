import bentoml
from components import Decoder, Prefiller, proxy

image = bentoml.images.Image().requirements_file('requirements.txt')


@bentoml.service(labels={'openai_endpoint': '/v1'}, image=image)
@bentoml.asgi_app(proxy)
class Router:
  decoder = bentoml.depends(Decoder)
  prefiller = bentoml.depends(Prefiller)
