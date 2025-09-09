from __future__ import annotations

import json, os

import bentoml, yaml

from .config import CHUNK_SIZE, PREFILL_PORT, ENVS, WORKING_ROOT, NIXL_PROXY_PORT, BentoArgs, get_aligned_bytes

bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
  envs=ENVS,
  endpoints={'livez': '/health', 'readyz': '/health'},
  resources={'gpu': 1, 'gpu_type': 'nvidia-h100-80gb'},
  extra_ports=[],
  workers=bento_args.num_decode,
)
class Prefiller:
  model = bentoml.models.HuggingFaceModel(bento_args.model_id, exclude=['*.pth', '*.pt', 'original/**/*'])

  @bentoml.on_startup
  def setup_config(self):
    worker_index, nixl_proxy_port = self.get_port_config()
    with (WORKING_ROOT / f'prefiller-{worker_index}.yaml').open('w') as f:
      yaml.dump(
        {
          'chunk_size': CHUNK_SIZE,
          'local_cpu': False,
          'max_local_cpu_size': 0,
          'max_local_disk_size': 0,
          'enable_nixl': True,
          'enable_xpyd': True,
          'nixl_role': 'sender',
          'nixl_proxy_host': 'localhost',
          'nixl_proxy_port': nixl_proxy_port,
          'nixl_buffer_size': get_aligned_bytes(bento_args, self.model) * 128,  # 1GB
          'nixl_buffer_device': 'cuda',
          'nixl_backends': ['UCX'],
        },
        stream=f,
      )

  def __command__(self) -> list[str]:
    worker_index, *_ = self.get_port_config()
    http_port = PREFILL_PORT + worker_index - 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_index - 1)
    os.environ['LMCACHE_CONFIG_FILE'] = (WORKING_ROOT / f'prefiller-{worker_index}.yaml').__fspath__()
    transfer_config = {
      'kv_connector': bento_args.kv_connector,
      'kv_role': 'kv_producer',
      'kv_connector_extra_config': {'discard_partial_chunks': False, 'lmcache_rpc_port': f'producer{worker_index}'},
    }

    return [
      'vllm',
      'serve',
      self.model,
      '--no-use-tqdm-on-load',
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      '--served-model-name',
      bento_args.model_id,
      '--port',
      str(http_port),
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
      '--kv-transfer-config',
      json.dumps(transfer_config),
      '--enable-auto-tool-choice',
      '--tool-call-parser',
      'hermes',
    ]

  @staticmethod
  def get_port_config() -> tuple[int, int]:
    worker_index = bentoml.server_context.worker_index or 1
    nixl_proxy_port = NIXL_PROXY_PORT + worker_index - 1
    return worker_index, nixl_proxy_port
