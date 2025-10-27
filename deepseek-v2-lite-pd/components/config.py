from __future__ import annotations

import os, pydantic

MODEL_ID = 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8'
IS_BENTOCLOUD = 'BENTOCLOUD_DEPLOYMENT_URL' in os.environ
PROXY_PORT, PREFILL_KV_PORT, DECODE_KV_PORT = 30001, 20001, 21001
NIXL_PEER_INIT_PORT, NIXL_PEER_ALLOC_PORT, NIXL_PROXY_PORT = 22001, 23001, 24001
PREFILL_PORT, DECODE_PORT = 5000, 6000


class BentoArgs(pydantic.BaseModel):
  num_prefill: int = 1
  num_decode: int = 1
