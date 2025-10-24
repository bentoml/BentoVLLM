import os, json, typing
import bentoml, httpx

from .config import (
  MODEL_ID,
  BentoArgs,
  PREFILL_GPU_TYPE,
  PREFILL_GPU_NUM,
  PREFILL_VISIBLE_DEVICES,
  PREFILL_PORT,
  PREFILL_DISAGGREGATION_PORT,
  IS_BENTOCLOUD,
)

bento_args = bentoml.use_arguments(BentoArgs)


@bentoml.service(
  endpoints={'readyz': '/health'},
  timeout=6000,
  envs=bento_args.envs,
  resources={'gpu': PREFILL_GPU_NUM, 'gpu_type': PREFILL_GPU_TYPE},
  extra_ports=list(range(PREFILL_DISAGGREGATION_PORT, PREFILL_DISAGGREGATION_PORT + PREFILL_GPU_NUM)),
  workers=bento_args.num_prefill,
)
class Prefiller:
  model = bentoml.models.HuggingFaceModel(MODEL_ID.lower(), exclude=['*.pth', '*.pt', 'original/**/*'])

  def __command__(self) -> list[str]:
    if not IS_BENTOCLOUD:
      os.environ['CUDA_VISIBLE_DEVICES'] = PREFILL_VISIBLE_DEVICES
    os.environ['VLLM_NIXL_SIDE_CHANNEL_PORT'] = PREFILL_DISAGGREGATION_PORT
    extra_args = os.environ.get('PREFILL_EXTRA_ARGS')
    kv_transfer_config = {'kv_connector': 'NixlConnector', 'kv_role': 'kv_both'}

    cmd_list = [
      'vllm',
      'serve',
      self.model,
      '--no-use-tqdm-on-load',
      '--disable-uvicorn-access-log',
      '--disable-fastapi-docs',
      '--served-model-name',
      MODEL_ID,
      '--host',
      '0.0.0.0',
      '--port',
      str(PREFILL_PORT),
      '-dp',
      str(PREFILL_GPU_NUM),
      '-tp',
      '1',
      '--enable-auto-tool-choice',
      '--tool-call-parser',
      'hermes',
      '--kv-transfer-config',
      json.dumps(kv_transfer_config),
    ]

    if extra_args:
      extra_list = extra_args.split(';')
      cmd_list = cmd_list + extra_list

    return cmd_list

  async def __is_ready__(self) -> bool:
    client = cast(httpx.AsyncClient, Prefiller.context.state['client'])
    try:
      response = await client.get(f'http://localhost:{PREFILL_PORT}/health', timeout=5.0)
    except (httpx.ConnectError, httpx.RequestError):
      return False
    return response.is_success

  async def __metrics__(self, content: str) -> str:
    client = typing.cast(httpx.AsyncClient, Prefiller.context.state['client'])
    try:
      response = await client.get(f'http://localhost:{PREFILL_PORT}/metrics', timeout=5.0)
      response.raise_for_status()
    except (httpx.ConnectError, httpx.RequestError) as e:
      logger.error('Failed to get metrics: %s', e)
      return content
    else:
      return content + '\n' + response.text
