from __future__ import annotations

import os, enum, pathlib, typing as t
import pydantic

if t.TYPE_CHECKING:
  from vllm.config import VllmConfig, ModelConfig

MODEL_ID = 'deepseek-ai/DeepSeek-V2-Lite-Chat'
IS_BENTOCLOUD = 'BENTOCLOUD_DEPLOYMENT_URL' in os.environ
PROXY_PORT, PREFILL_KV_PORT, DECODE_KV_PORT = 30001, 20001, 21001
NIXL_PEER_INIT_PORT, NIXL_PEER_ALLOC_PORT, NIXL_PROXY_PORT = 22001, 23001, 24001
PREFILL_PORT, DECODE_PORT = int(os.getenv('PORT', 3000)), int(os.getenv('PORT', 3000))

# lmcache value
CHUNK_SIZE = 256

WORKING_ROOT = pathlib.Path(__file__).parent.parent
ENVS = [
  {'name': 'HF_TOKEN'},
  {'name': 'PYTHONHASHSEED', 'value': '0'},
  {'name': 'UCX_TLS', 'value': 'cuda_ipc,cuda_copy,tcp'},
]


def calculate_mtp_layers(vllm_config: VllmConfig, model_config: ModelConfig):
  num_mtp_layers = 0
  if vllm_config is not None and vllm_config.speculative_config is not None:
    # TODO(baoloongmao): Support other MTP methods
    if vllm_config.speculative_config.method == 'deepseek_mtp':
      num_mtp_layers = getattr(model_config.hf_config, 'num_nextn_predict_layers', 0)
  return num_mtp_layers


# return number of aligned bytes for kv shape
def get_aligned_bytes(bento_args, model) -> int:
  import torch
  from vllm.entrypoints.openai.cli_args import make_arg_parser
  from vllm.engine.arg_utils import AsyncEngineArgs
  from vllm.utils import FlexibleArgumentParser, get_kv_cache_torch_dtype

  args = make_arg_parser(FlexibleArgumentParser()).parse_args([
    '--no-use-tqdm-on-load',
    '--disable-uvicorn-access-log',
    '--disable-fastapi-docs',
    '--served-model-name',
    bento_args.model_id,
    '--tensor-parallel-size',
    '1',
    '--no-enable-prefix-caching',
    '--max-num-batched-tokens',
    '10000',
    '--max-model-len',
    '10000',
    '--max-num-seqs',
    '256',
    '--gpu-memory-utilization',
    '0.7',
  ])
  args.model = model
  engine_args = AsyncEngineArgs.from_cli_args(args)

  # we need to calculate the number of kv bytes, for PagedTensorMemoryAllocator
  vllm_config = engine_args.create_engine_config()
  model_config = vllm_config.model_config
  parallel_config = vllm_config.parallel_config
  cache_config = vllm_config.cache_config

  use_mla = hasattr(model_config, 'use_mla') and isinstance(model_config.use_mla, bool) and model_config.use_mla
  kv_dtype = get_kv_cache_torch_dtype(cache_config.cache_dtype, model_config.dtype)

  num_layer = model_config.get_num_layers(parallel_config)
  num_mtp_layers = calculate_mtp_layers(vllm_config, model_config)
  num_layer += num_mtp_layers
  num_kv_head = model_config.get_num_kv_heads(parallel_config)
  head_size = model_config.get_head_size()

  # torch.Size(kv_shape).numel() -> num_elements
  num_elements = torch.Size((num_layer, 1 if use_mla else 2, CHUNK_SIZE, num_kv_head, head_size)).numel()
  bytes_per_element = torch.tensor([], dtype=kv_dtype).element_size()
  return num_elements * bytes_per_element


class KVConnectorMapping(enum.StrEnum):
  lmcache: str = 'LMCacheConnectorV1'
  p2p: str = 'P2pNcclConnector'


class BentoArgs(pydantic.BaseModel):
  num_prefill: int = 1
  num_decode: int = 1
  model_id: str = MODEL_ID
  kv_connector: KVConnectorMapping = pydantic.Field(default=KVConnectorMapping.lmcache)
