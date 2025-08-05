<div align="center">
    <h1 align="center">Self-host LLMs with vLLM and BentoML</h1>
</div>

This repository contains a group of BentoML example projects, showing you how to serve and deploy open-source Large Language Models using [vLLM](https://vllm.ai), a high-throughput and memory-efficient inference engine. Every model directory contains the code to add OpenAI compatible endpoints to the BentoML Service.

💡 You can use these examples as bases for advanced code customization, such as custom model, inference logic or vLLM options. For simple LLM hosting with OpenAI compatible endpoints without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

The following is an example of serving one of the LLMs in this repository: Llama 3.1 8B Instruct.

## Prerequisites

- If you want to test the Service locally, we recommend you use an Nvidia GPU with at least 16G VRAM.
- Gain access to the model in [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoVLLM.git
cd BentoVLLM/llama3.1-8b-instruct

# Recommend UV and Python 3.11
uv venv && uv pip install -r requirements.txt

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

## OpenAI-compatible endpoints

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')


completion = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": "Who are you? Please respond in pirate speak!"
        }
    ],
    stream=True,
)
for chunk in completion:
    # Extract and print the content of the model's reply
    print(chunk.choices[0].delta.content or "", end="")
```

These OpenAI-compatible endpoints also support [vLLM extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters). For example, you can force the chat completion output a JSON object by using the `guided_json` parameters:

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

json_schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    }
}

completion = client.chat.completions.create(
    model=client.models.list().data[0].id,
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    extra_body=dict(guided_json=json_schema),
)
print(completion.choices[0].message.content)  # will return something like: {"city": "Paris"}
```

All supported extra parameters are listed in [vLLM documentation](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters).

**Note**: If your Service is deployed with [protected endpoints on BentoCloud](https://docs.bentoml.com/en/latest/bentocloud/how-tos/manage-access-token.html#access-protected-deployments), you need to set the environment variable `OPENAI_API_KEY` to your BentoCloud API key first.

```bash
export OPENAI_API_KEY={YOUR_BENTOCLOUD_API_TOKEN}
```

You can then use the following line to replace the client in the above code snippet. Refer to [Obtain the endpoint URL](https://docs.bentoml.com/en/latest/bentocloud/how-tos/call-deployment-endpoints.html#obtain-the-endpoint-url) to retrieve the endpoint URL.

```python
client = OpenAI(base_url='your_bentocloud_deployment_endpoint_url/v1')
```

For detailed explanations of the Service code, see [vLLM inference](https://docs.bentoml.org/en/latest/examples/vllm.html).

## Deploy to BentoCloud

After the Service is ready, you can deploy the application to BentoCloud for better management and scalability. [Sign up](https://www.bentoml.com/) if you haven't got a BentoCloud account.

Make sure you have [logged in to BentoCloud](https://docs.bentoml.com/en/latest/scale-with-bentocloud/manage-api-tokens.html).

```bash
bentoml cloud login
```

Create a BentoCloud secret to store the required environment variable and reference it for deployment.

```bash
bentoml secret create huggingface HF_TOKEN=$HF_TOKEN

bentoml deploy . --secret huggingface
```

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).

## Featured models

In addition to Llama 3.1 8B Instruct, we also have examples for other models in the subdirectories of this repository:

| Model | Links |
|-------|-------|
| deepseek-prover-v2-671b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-prover-v2-671b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| deepseek-r1-671b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-r1-671b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| deepseek-v3-671b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-v3-671b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| devstral-small-2505 | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/devstral-small-2505/) • [Hugging Face](https://huggingface.co/mistralai/Devstral-Small-2505) |
| gemma3-4b-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/gemma3-4b-instruct/) • [Hugging Face](https://huggingface.co/google/gemma-3-4b-it) |
| gpt-oss-120b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/gpt-oss-120b/) • [Hugging Face](https://huggingface.co/openai/gpt-oss-120b) |
| jamba1.6-large | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/jamba1.6-large/) • [Hugging Face](https://huggingface.co/ai21labs/AI21-Jamba-Large-1.6) |
| jamba1.6-mini | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/jamba1.6-mini/) • [Hugging Face](https://huggingface.co/ai21labs/AI21-Jamba-Mini-1.6) |
| llama3.1-8b-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama3.1-8b-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) |
| llama3.2-3b-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama3.2-3b-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| llama3.3-70b-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama3.3-70b-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| llama4-17b-maverick-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama4-17b-maverick-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8) |
| llama4-17b-scout-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama4-17b-scout-instruct/) • [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) |
| magistral-small-2506 | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/magistral-small-2506/) • [Hugging Face](https://huggingface.co/mistralai/Magistral-Small-2506) |
| mistral-small-3.1-24b-instruct-2503 | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/mistral-small-3.1-24b-instruct-2503/) • [Hugging Face](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) |
| phi4-14b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/phi4-14b/) • [Hugging Face](https://huggingface.co/microsoft/phi-4) |
| phi4-14b-reasoning | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/phi4-14b-reasoning/) • [Hugging Face](https://huggingface.co/microsoft/Phi-4-reasoning) |
| phi4-14b-reasoning-plus | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/phi4-14b-reasoning-plus/) • [Hugging Face](https://huggingface.co/microsoft/Phi-4-reasoning-plus) |
| qwen3-235b-a22b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/qwen3-235b-a22b/) • [Hugging Face](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8) |
| qwen3-30b-a3b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/qwen3-30b-a3b/) • [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| qwen3-8b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/qwen3-8b/) • [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) |
| qwen3-coder-480b-a35b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/qwen3-coder-480b-a35b/) • [Hugging Face](https://huggingface.co/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8) |
| r1-0528-qwen3-8b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/r1-0528-qwen3-8b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) |
| r1-distill-llama3.3-70b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/r1-distill-llama3.3-70b/) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |