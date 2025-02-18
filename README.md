<div align="center">
    <h1 align="center">Self-host LLMs with vLLM and BentoML</h1>
</div>

This repository contains a group of BentoML example projects, showing you how to serve and deploy open-source Large Language Models using [vLLM](https://vllm.ai), a high-throughput and memory-efficient inference engine. Every model directory contains the code to add OpenAI compatible endpoints to the BentoML Service.

💡 You can use these examples as bases for advanced code customization, such as custom model, inference logic or vLLM options. For simple LLM hosting with OpenAI compatible endpoints without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

The following is an example of serving one of the LLMs in this repository: Mistral 7B Instruct.

## Prerequisites

- If you want to test the Service locally, we recommend you use an Nvidia GPU with at least 16G VRAM.
- Gain access to the model in [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct).

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoVLLM.git
cd BentoVLLM/llama3.2-1b-instruct

# Recommend UV and Python 3.11
uv venv && pip install .

export HF_TOKEN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. Run `bentoml serve` in your project directory to start the Service.

```bash
$ bentoml serve .

2024-01-18T07:51:30+0800 [INFO] [cli] Starting production HTTP BentoServer from "service:VLLM" listening on http://localhost:3000 (Press CTRL+C to quit)
INFO 01-18 07:51:40 model_runner.py:501] Capturing the model for CUDA graphs. This may lead to unexpected consequences if the model is not static. To run the model in eager mode, set 'enforce_eager=True' or use '--enforce-eager' in the CLI.
INFO 01-18 07:51:40 model_runner.py:505] CUDA graphs can take additional 1~3 GiB memory per GPU. If you are running out of memory, consider decreasing `gpu_memory_utilization` or enforcing eager mode.
INFO 01-18 07:51:46 model_runner.py:547] Graph capturing finished in 6 secs.
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details>

<summary>CURL</summary>

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

</details>

<details>

<summary>Python client</summary>

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

</details>

<details>

<summary>OpenAI-compatible endpoints</summary>

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
client.models.list()

chat_completion = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {
            "role": "user",
            "content": "Explain superconductors like I'm five years old"
        }
    ],
    stream=True,
)
for chunk in chat_completion:
    # Extract and print the content of the model's reply
    print(chunk.choices[0].delta.content or "", end="")
```

**Note**: If your Service is deployed with [protected endpoints on BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html#access-protected-deployments), you need to set the environment variable `OPENAI_API_KEY` to your BentoCloud API key first.

```bash
export OPENAI_API_KEY={YOUR_BENTOCLOUD_API_TOKEN}
```

You can then use the following line to replace the client in the above code snippet. Refer to [Obtain the endpoint URL](https://docs.bentoml.com/en/latest/bentocloud/how-tos/call-deployment-endpoints.html#obtain-the-endpoint-url) to retrieve the endpoint URL.

```python
client = OpenAI(base_url='your_bentocloud_deployment_endpoint_url/v1')
```

</details>

For detailed explanations of the Service code, see [vLLM inference](https://docs.bentoml.org/en/latest/use-cases/large-language-models/vllm.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html).

```bash
bentoml cloud login
```

Create a BentoCloud secret to store the required environment variable and reference it for deployment.

```bash
bentoml secret create huggingface HF_TOKEN=$HF_TOKEN

bentoml deploy . --secret huggingface
```

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).

## Featured models

In addition to Llama 3.1 8B Instruct, we also have examples for other models in the subdirectories of this repository:

| Model | Links |
|-------|-------|
| deepseek-v3-671b | [GitHub](deepseek-v3-671b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) |
| deepseek-r1-671b | [GitHub](deepseek-r1-671b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1) |
| deepseek-r1-distill-llama3.3-70b | [GitHub](deepseek-r1-distill-llama3.3-70b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| deepseek-r1-distill-llama3.3-70b-w8a8 | [GitHub](deepseek-r1-distill-llama3.3-70b-w8a8/) • [Hugging Face](https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w8a8) |
| deepseek-r1-distill-llama3.3-70b-w4a16 | [GitHub](deepseek-r1-distill-llama3.3-70b-w4a16/) • [Hugging Face](https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Llama-70B-quantized.w4a16) |
| deepseek-r1-distill-qwen2.5-32b | [GitHub](deepseek-r1-distill-qwen2.5-32b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) |
| deepseek-r1-distill-qwen2.5-32b-w8a8 | [GitHub](deepseek-r1-distill-qwen2.5-32b-w8a8/) • [Hugging Face](https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w8a8) |
| deepseek-r1-distill-qwen2.5-32b-w4a16 | [GitHub](deepseek-r1-distill-qwen2.5-32b-w4a16/) • [Hugging Face](https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Qwen-32B-quantized.w4a16) |
| deepseek-r1-distill-qwen2.5-14b | [GitHub](deepseek-r1-distill-qwen2.5-14b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B) |
| deepseek-r1-distill-qwen2.5-14b-w8a8 | [GitHub](deepseek-r1-distill-qwen2.5-14b-w8a8/) • [Hugging Face](https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w8a8) |
| deepseek-r1-distill-qwen2.5-14b-w4a16 | [GitHub](deepseek-r1-distill-qwen2.5-14b-w4a16/) • [Hugging Face](https://huggingface.co/neuralmagic/DeepSeek-R1-Distill-Qwen-14B-quantized.w4a16) |
| deepseek-r1-distill-qwen2.5-7b-math | [GitHub](deepseek-r1-distill-qwen2.5-7b-math/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) |
| deepseek-r1-distill-llama3.1-8b | [GitHub](deepseek-r1-distill-llama3.1-8b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| deepseek-r1-distill-llama3.1-8b-tool-calling | [GitHub](deepseek-r1-distill-llama3.1-8b-tool-calling/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| gemma2-2b-instruct | [GitHub](gemma2-2b-instruct/) • [Hugging Face](https://huggingface.co/google/gemma-2-2b-it) |
| gemma2-9b-instruct | [GitHub](gemma2-9b-instruct/) • [Hugging Face](https://huggingface.co/google/gemma-2-9b-it) |
| gemma2-27b-instruct | [GitHub](gemma2-27b-instruct/) • [Hugging Face](https://huggingface.co/google/gemma-2-27b-it) |
| jamba1.5-mini | [GitHub](jamba1.5-mini/) • [Hugging Face](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini) |
| jamba1.5-large | [GitHub](jamba1.5-large/) • [Hugging Face](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large) |
| llama3.1-8b-instruct | [GitHub](llama3.1-8b-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| llama3.1-tulu-3.1-8b | [GitHub](llama3.1-tulu-3.1-8b/) • [Hugging Face](https://huggingface.co/allenai/Llama-3.1-Tulu-3.1-8B) |
| llama3.2-1b-instruct | [GitHub](llama3.2-1b-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| llama3.2-3b-instruct | [GitHub](llama3.2-3b-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| llama3.2-11b-vision-instruct | [GitHub](llama3.2-11b-vision-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct) |
| llama3.2-90b-vision-instruct | [GitHub](llama3.2-90b-vision-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct) |
| llama3.3-70b-instruct | [GitHub](llama3.3-70b-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| hermes-3-llama3.1-405b | [GitHub](hermes-3-llama3.1-405b/) • [Hugging Face](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B-FP8) |
| deephermes-3-llama3-8b | [GitHub](deephermes-3-llama3-8b/) • [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) |
| pixtral-12b-2409 | [GitHub](pixtral-12b-2409/) • [Hugging Face](https://huggingface.co/mistralai/Pixtral-12B-2409) |
| pixtral-large-2411 | [GitHub](pixtral-large-2411/) • [Hugging Face](https://huggingface.co/mistralai/Pixtral-Large-Instruct-2411) |
| ministral-8b-instruct-2410 | [GitHub](ministral-8b-instruct-2410/) • [Hugging Face](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) |
| mistral-small-24b-instruct-2501 | [GitHub](mistral-small-24b-instruct-2501/) • [Hugging Face](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) |
| mistral-large-123b-instruct-2411 | [GitHub](mistral-large-123b-instruct-2411/) • [Hugging Face](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411) |
| phi4-14b | [GitHub](phi4-14b/) • [Hugging Face](https://huggingface.co/microsoft/phi-4) |
| qwen2.5-7b-instruct | [GitHub](qwen2.5-7b-instruct/) • [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| qwen2.5-14b-instruct | [GitHub](qwen2.5-14b-instruct/) • [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct) |
| qwen2.5-32b-instruct | [GitHub](qwen2.5-32b-instruct/) • [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) |
| qwen2.5-72b-instruct | [GitHub](qwen2.5-72b-instruct/) • [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) |
| qwen2.5-coder-7b-instruct | [GitHub](qwen2.5-coder-7b-instruct/) • [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| qwen2.5-coder-32b-instruct | [GitHub](qwen2.5-coder-32b-instruct/) • [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) |