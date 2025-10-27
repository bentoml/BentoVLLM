import json
import os

import bentoml

from .config import IS_BENTOCLOUD, MODEL_ID, PREFILL_KV_PORT, PREFILL_PORT, BentoArgs

bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
  envs=[{'name': 'HF_TOKEN'}],
  endpoints={'livez': '/health', 'readyz': '/health'},
  resources={'gpu': 1, 'gpu_type': 'nvidia-h100-80gb'},
  extra_ports=[PREFILL_KV_PORT],
  workers=bento_args.num_prefill,
)
class Prefiller:
  model = bentoml.models.HuggingFaceModel(MODEL_ID, exclude=['*.pth', '*.pt', 'original/**/*'])

  def __command__(self) -> list[str]:
    worker_index = bentoml.server_context.worker_index or 1
    gpu_offset = 0 if IS_BENTOCLOUD else 4
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_offset + worker_index - 1)
    http_port = PREFILL_PORT + worker_index - 1
    kv_port = PREFILL_KV_PORT + worker_index - 1
    transfer_config = {
      'kv_connector': 'P2pNcclConnector',
      'kv_role': 'kv_producer',
      'kv_buffer_size': '1e1',
      'kv_port': str(kv_port),
      'kv_connector_extra_config': {'http_port': str(http_port), 'send_type': 'PUT_ASYNC', 'nccl_num_channels': '16'},
    }
    return [
      'vllm',
      'serve',
      self.model,
      '--no-use-tqdm-on-load',
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      '--served-model-name',
      MODEL_ID,
      '--no-enable-prefix-caching',
      '--max-num-batched-tokens',
      '10000',
      '--max-model-len',
      '10000',
      '--tensor-parallel-size',
      '1',
      '--port',
      str(http_port),
      '--max-num-batched-tokens',
      '16384',
      '--enable-auto-tool-choice',
      '--tool-call-parser',
      'hermes',
      '--kv-transfer-config',
      json.dumps(transfer_config),
      '--enforce-eager',
    ]
