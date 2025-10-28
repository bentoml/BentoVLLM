from __future__ import annotations

import os, pydantic

MODEL_ID = 'Qwen/Qwen3-30B-A3B-Instruct-2507-FP8'
IS_BENTOCLOUD = 'BENTOCLOUD_DEPLOYMENT_URL' in os.environ
PREFILL_PORT = 5000 if not IS_BENTOCLOUD else 8000
DECODE_PORT = 6000 if not IS_BENTOCLOUD else 8000
PROXY_PORT = 8000
PREFILL_DISAGGREGATION_PORT = 8998 if not IS_BENTOCLOUD else 8998
DECODE_DISAGGREGATION_PORT = 9999 if not IS_BENTOCLOUD else 8998
PREFILL_GPU_TYPE = DECODE_GPU_TYPE = 'nvidia-h100-80gb'
PREFILL_GPU_NUM, DECODE_GPU_NUM = 4, 4
PREFILL_GPU_OFFSET, DECODE_GPU_OFFSET = 0, 4

PREFILL_VISIBLE_DEVICES = ','.join(str(i) for i in range(PREFILL_GPU_OFFSET, PREFILL_GPU_OFFSET + PREFILL_GPU_NUM))

DECODE_VISIBLE_DEVICES = ','.join(str(i) for i in range(DECODE_GPU_OFFSET, DECODE_GPU_OFFSET + DECODE_GPU_NUM))


class BentoArgs(pydantic.BaseModel):
  num_prefill: int = 4
  num_decode: int = 4

  @property
  def envs(self) -> list[dict[str, str]]:
    return [{'name': 'UCX_NET_DEVICES', 'value': 'all'}, {'name': 'UCX_TLS', 'value': 'all'}]
