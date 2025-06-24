<div align="center">
    <h1 align="center">Self-host LLMs with vLLM and BentoML</h1>
</div>

Follow this guide to self-host the any LLMs model with BentoCloud in your own cloud account. If your team doesn’t already have access to BentoCloud, please use the link below to contact us and set it up in your cloud environment.

[![Deploy on BentoCloud](https://img.shields.io/badge/Deploy_on_BentoCloud-d0bfff?style=for-the-badge)](https://cloud.bentoml.com/)
[![Talk to sales](https://img.shields.io/badge/Talk_to_sales-eefbe4?style=for-the-badge)](https://bentoml.com/contact)

💡 You can use these examples as bases for advanced code customization, such as custom model, inference logic or vLLM options. For simple LLM hosting with OpenAI compatible endpoints without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

The following is a default example of serving one of the LLMs in this repository: Llama 3.1 8B Instruct.

## Prerequisites

- Gain access to the model in [Hugging Face](https://huggingface.co) if it needs tokens to download the weights.
- NVIDIA GPUs, especially higher end card such as A100s and H100s.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoVLLM.git
cd BentoVLLM/llama3.1-8b-instruct

# Recommend UV and Python 3.11
uv venv && uv pip install -r requirements.txt

export HF_TOKEN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in [`service.py`](/service.py). Run `bentoml serve` in your project directory to start the Service.

```bash
bentoml serve
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

<details open>

<summary>OpenAI-compatible endpoints</summary>

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
model_id = client.models.list().data[0].id

completions = client.chat.completions.create(
  model=model_id,
  messages=[
    {
      "role": "user",
      "content": "Who are you? Please respond in pirate speak!"
    }
  ],
  stream=True,
)
for chunk in completions:
    # Extract and print the content of the model's reply
    print(chunk.choices[0].delta.content or "", end="")
```

These OpenAI-compatible endpoints also support [vLLM extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters). For example:

```python
json_schema = {
  "type": "object",
  "properties": {
    "city": {"type": "string"}
  }
}

completions = client.chat.completions.create(
    model=model_id,
    messages=[
      {
        "role": "user",
        "content": "What is the capital of France?"
      }
    ],
    response_format={
      "type": "json_schema",
      "json_schema": {
        "name": "city_description",
        "schema": json_schema,
      },
    },
)

print(chat_completion.choices[0].message.content)  # will return something like: {"city": "Paris"}
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

</details>

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

In addition to Llama 3.1 8B Instruct, we also have examples for other models, but not limited to:


| Model | Links |
|-------|-------|
| deepseek-v3-671b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-v3-671b.yaml) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) |
| deepseek-r1-671b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-r1-671b.yaml) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) |
| deepseek-prover-v2-671b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-prover-v2-671b.yaml) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-671B) |
| deepseek-prover-v2-7b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-prover-v2-7b.yaml) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B) |
| deepseek-r1-0528-qwen3-8b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-r1-0528-qwen3-8b.yaml) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) |
| deepseek-r1-distill-llama3.3-70b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-r1-distill-llama3.3-70b.yaml) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B) |
| deepseek-r1-distill-llama3.1-8b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/deepseek-r1-distill-llama3.1-8b.yaml) • [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B) |
| gemma3-4b-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/gemma3-4b-instruct.yaml) • [Hugging Face](https://huggingface.co/google/gemma-3-4b-it) |
| jamba1.5-mini | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/jamba1.5-mini.yaml) • [Hugging Face](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Mini) |
| jamba1.5-large | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/jamba1.5-large.yaml) • [Hugging Face](https://huggingface.co/ai21labs/AI21-Jamba-1.5-Large) |
| llama3.2-3b-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama3.2-3b-instruct.yaml) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| llama4-17b-maverick-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama4-17b-maverick-instruct.yaml) • [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8) |
| llama4-17b-scout-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama4-17b-scout-instruct.yaml) • [Hugging Face](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct) |
| llama3.3-70b-instruct | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/llama3.3-70b-instruct.yaml) • [Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| mistral-small-3.1-24b-instruct-2503 | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/mistral-small-3.1-24b-instruct-2503.yaml) • [Hugging Face](https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503) |
| magistral-small-2506 | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/magistral-small-2506.yaml) • [Hugging Face](https://huggingface.co/mistralai/Magistral-Small-2506) |
| phi4-14b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/phi4-14b.yaml) • [Hugging Face](https://huggingface.co/microsoft/phi-4) |
| phi4-14b-reasoning | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/phi4-14b-reasoning.yaml) • [Hugging Face](https://huggingface.co/microsoft/Phi-4-reasoning) |
| qwq-32b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/qwq-32b.yaml) • [Hugging Face](https://huggingface.co/Qwen/QwQ-32B) |
| qwen3-8b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/qwen3-8b.yaml) • [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) |
| qwen3-30b-a3b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/qwen3-30b-a3b.yaml) • [Hugging Face](https://huggingface.co/Qwen/Qwen3-30B-A3B) |
| qwen3-235b-a22b | [GitHub](https://github.com/bentoml/BentoVLLM/tree/main/qwen3-235b-a22b.yaml) • [Hugging Face](https://huggingface.co/Qwen/Qwen3-235B-A22B-FP8) |
