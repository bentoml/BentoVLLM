This project demonstrates how to build an LLM application using BentoML and [vLLM](https://vllm.ai).

## Prerequisites

- You have installed Python 3.8+ and `pip`. See the [Python downloads page](https://www.python.org/downloads/) to learn more.
- You have a basic understanding of key concepts in BentoML, such as Services. We recommend you read [Quickstart](https://docs.bentoml.com/en/1.2/get-started/quickstart.html) first.
- This example Service uses the model ``meta-llama/Llama-2-7b-chat-hf``. You can choose any other model supported by vLLM based on your needs. If you are using the same model in the project, you need to obtain access to it on the [Meta website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and [Hugging Face](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
- If you want to test the Service locally, you need a Nvidia GPU with at least 16G VRAM.
- (Optional) We recommend you create a virtual environment for dependency isolation for this project. See the [Conda documentation](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) or the [Python documentation](https://docs.python.org/3/library/venv.html) for details.

## Install dependencies

```bash
pip install -r requirements.txt && pip install -f -U "pydantic>=2.0"
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```python
$ bentoml serve .

2024-01-18T07:51:30+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:VLLMService" listening on http://localhost:3000 (Press CTRL+C to quit)
INFO 01-18 07:51:40 model_runner.py:501] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 01-18 07:51:40 model_runner.py:505] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode.
INFO 01-18 07:51:46 model_runner.py:547] Graph capturing finished in 6 secs.
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

CURL

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Explain superconductors like I'\''m five years old",
  "tokens": null
}'
```

BentoML client

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Explain superconductors like I'm five years old",
        tokens=None
    )
    for response in response_generator:
        print(response)
```

## Deploy to production

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. A YAML configuration file (`bentofile.yaml`) is used to define the build options and package your application into a Bento. See [Bento build options](https://docs.bentoml.com/en/latest/concepts/bento.html#bento-build-options) to learn more.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/1.2/bentocloud/how-tos/manage-access-token.html), set your Hugging Face access token in ``bentofile.yaml``, then run the following command in your project directory to deploy the application to BentoCloud.

```bash
bentoml deploy .
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: Alternatively, you can use BentoML to generate a [Docker image](https://docs.bentoml.com/en/1.2/guides/containerization.html) for a custom deployment.