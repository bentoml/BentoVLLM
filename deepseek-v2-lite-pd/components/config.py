from __future__ import annotations

import os, enum, pathlib
import pydantic

MODEL_ID = 'deepseek-ai/DeepSeek-V2-Lite-Chat'
IS_BENTOCLOUD = 'BENTOCLOUD_DEPLOYMENT_URL' in os.environ
PROXY_PORT, PREFILL_KV_PORT, DECODE_KV_PORT = 30001, 20001, 21001
NIXL_PEER_INIT_PORT, NIXL_PEER_ALLOC_PORT, NIXL_PROXY_PORT = 22001, 23001, 24001
PREFILL_PORT, DECODE_PORT = int(os.getenv('PORT', 3000)), int(os.getenv('PORT', 3000))

WORKING_ROOT = pathlib.Path(__file__).parent.parent
ENVS = [
  {'name': 'HF_TOKEN'},
  {'name': 'PYTHONHASHSEED', 'value': '0'},
  {'name': 'UCX_TLS', 'value': 'cuda_ipc,cuda_copy,tcp'},
]


class KVConnectorMapping(enum.StrEnum):
  lmcache: str = 'LMCacheConnectorV1'
  p2p: str = 'P2pNcclConnector'


class BentoArgs(pydantic.BaseModel):
  num_prefill: int = 1
  num_decode: int = 1
  model_id: str = MODEL_ID
  kv_connector: KVConnectorMapping = pydantic.Field(default=KVConnectorMapping.lmcache)
