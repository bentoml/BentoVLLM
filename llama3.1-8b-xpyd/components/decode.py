import json
import os

import bentoml

from .config import DECODE_KV_PORT, DECODE_PORT, MODEL_ID, BentoArgs

bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
  envs=[{'name': 'HF_TOKEN'}],
  endpoints={'livez': '/health', 'readyz': '/health'},
  resources={'gpu': 1, 'gpu_type': 'nvidia-h100-80gb'},
  extra_ports=[DECODE_KV_PORT],
  workers=bento_args.num_decode,
)
class Decoder:
  model = bentoml.models.HuggingFaceModel(MODEL_ID, exclude=['*.pth', '*.pt', 'original/**/*'])

  def __command__(self) -> list[str]:
    worker_index = bentoml.server_context.worker_index or 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(worker_index - 1)
    http_port = DECODE_PORT + worker_index - 1
    kv_port = DECODE_KV_PORT + worker_index - 1
    transfer_config = {
      'kv_connector': 'P2pNcclConnector',
      'kv_role': 'kv_consumer',
      'kv_buffer_size': '8e9',
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
      '--port',
      str(http_port),
      '--max-num-batched-tokens',
      '16384',
      '--kv-transfer-config',
      json.dumps(transfer_config),
    ]
