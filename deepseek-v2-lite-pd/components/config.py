from __future__ import annotations

import os, pydantic

MODEL_ID = 'deepseek-ai/DeepSeek-V2-Lite-Chat'
IS_BENTOCLOUD = 'BENTOCLOUD_DEPLOYMENT_URL' in os.environ
PROXY_PORT, PREFILL_KV_PORT, DECODE_KV_PORT = 30001, 20001, 21001
NIXL_PEER_INIT_PORT, NIXL_PEER_ALLOC_PORT, NIXL_PROXY_PORT = 22001, 23001, 24001
PREFILL_PORT, DECODE_PORT = int(os.getenv('PORT', 3000)), int(os.getenv('PORT', 3000))


class BentoArgs(pydantic.BaseModel):
  num_prefill: int = 1
  num_decode: int = 1
