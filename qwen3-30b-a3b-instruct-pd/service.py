from __future__ import annotations

import asyncio, json, logging, sys
from typing import cast
import bentoml, httpx
from components import Decoder, Prefiller, config

logger = logging.getLogger(__name__)


@bentoml.service(
  name='router',
  timeout=6000,
  envs=[{'name': 'HF_TOKEN'}],
  labels={
    'openai_endpoint': '/v1',
    'reasoning': '0',
    'tool': 'hermes',
    'hf_generation_config': json.dumps({'temperature': 0.6, 'top_p': 0.95}),
  },
  image=bentoml.images.Image(python_version='3.10', lock_python_packages=False)
  .system_packages('curl', 'git', 'ninja-build')
  .requirements_file('requirements.txt')
  .python_packages('./wheels/sglang_router-0.1.9-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'),
)
class Router:
  decoder = bentoml.depends(Decoder)
  prefiller = bentoml.depends(Prefiller)

  def __command__(self) -> list[str]:
    from components.config import PROXY_PORT

    cmd = [
      sys.executable,
      '-m',
      'sglang_router.launch_router',
      '--policy',
      'round_robin',
      '--host',
      '0.0.0.0',
      '--port',
      str(PROXY_PORT),
      '--pd-disaggregation',
    ]

    return cmd

  async def __is_ready__(self) -> bool:
    client = cast(httpx.AsyncClient, Router.context.state['client'])

    # Step 1: Get current hosts from service discovery
    try:
      prefill_hosts = await Prefiller.get_hosts()
      decode_hosts = await Decoder.get_hosts()
    except Exception as e:
      logger.error('Failed to get hosts from service discovery: %s', e)
      return False

    # Step 2: List existing workers in the router
    try:
      response = await client.get('/workers', timeout=5.0)
      response.raise_for_status()
      workers_data = response.json()
      existing_workers = {w['url']: w for w in workers_data.get('workers', [])}
    except Exception as e:
      logger.warning('Failed to list existing workers, will proceed with adding all: %s', e)
      existing_workers = {}

    # Step 3: Build expected worker sets
    expected_prefill_urls = {f'http://{host}' for host in prefill_hosts}
    expected_decode_urls = {f'http://{host}' for host in decode_hosts}
    expected_all_urls = expected_prefill_urls | expected_decode_urls

    # Step 4: Scale UP - Add new workers that don't exist
    workers_to_add = []

    for host in prefill_hosts:
      url = f'http://{host}'
      if url not in existing_workers or existing_workers[url].get('worker_type') != 'prefill':
        workers_to_add.append(
          client.post(
            '/workers',
            json={'url': url, 'worker_type': 'prefill', 'bootstrap_port': config.PREFILL_DISAGGREGATION_PORT},
            timeout=10,
          )
        )

    for host in decode_hosts:
      url = f'http://{host}'
      if url not in existing_workers or existing_workers[url].get('worker_type') != 'decode':
        workers_to_add.append(client.post('/workers', json={'url': url, 'worker_type': 'decode'}, timeout=10))

    if workers_to_add:
      logger.info('Adding %d new workers to the router', len(workers_to_add))
      results = await asyncio.gather(*workers_to_add, return_exceptions=True)
      for i, result in enumerate(results):
        if isinstance(result, Exception):
          logger.warning('Failed to add worker: %s', result)

    # Step 5: Scale DOWN - Remove stale workers that no longer exist
    workers_to_remove = []

    for worker_url, worker_info in existing_workers.items():
      if worker_url not in expected_all_urls:
        logger.info('Removing stale worker: %s', worker_url)
        workers_to_remove.append(client.delete(f'/workers/{worker_url}', timeout=10))

    if workers_to_remove:
      logger.info('Removing %d stale workers from the router', len(workers_to_remove))
      results = await asyncio.gather(*workers_to_remove, return_exceptions=True)
      for i, result in enumerate(results):
        if isinstance(result, Exception):
          logger.warning('Failed to remove worker: %s', result)

    # Step 6: Check router readiness
    try:
      response = await client.get('/readiness', timeout=5.0)
      if not response.is_success:
        logger.error('Router readiness check failed: %s', response.status_code)
        return False
    except (httpx.ConnectError, httpx.RequestError) as e:
      logger.error('Router readiness check failed: %s', e)
      return False

    # Step 7: Verify worker health via runner-lb
    # Create a separate client for health checks to avoid connection pool issues
    async with httpx.AsyncClient() as health_client:
      responses = await asyncio.gather(
        # Carry the headers so that runner-lb can route to the right service
        health_client.get(f'{Prefiller.url}/health', headers={'Runner-Name': 'Prefiller'}, timeout=5.0),
        health_client.get(f'{Decoder.url}/health', headers={'Runner-Name': 'Decoder'}, timeout=5.0),
        return_exceptions=True,
      )

      for i, r in enumerate(responses):
        service_name = 'Prefiller' if i == 0 else 'Decoder'
        if isinstance(r, BaseException):
          logger.error('%s health check failed: %s', service_name, r)
          return False
        if not r.is_success:
          logger.error('%s health check returned %s', service_name, r.status_code)
          return False

      return True
