import os

import pydantic

MODEL_ID = 'meta-llama/Llama-3.1-8B-Instruct'

IS_BENTOCLOUD = 'BENTOCLOUD_DEPLOYMENT_URL' in os.environ

PROXY_PORT = 30001
PREFILL_KV_PORT = 20001
DECODE_KV_PORT = 21001
PREFILL_PORT = int(os.getenv('PORT', 3000))
DECODE_PORT = int(os.getenv('PORT', 3000))


class BentoArgs(pydantic.BaseModel):
  num_prefill: int = 1
  num_decode: int = 1
