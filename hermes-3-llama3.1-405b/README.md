<div align="center">
    <h1 align="center">Self-host Hermes 3 Llama 3.1 405B FP8 with vLLM and BentoML</h1>
</div>

This is a BentoML example project, showing you how to serve and deploy Hermes 3 Llama 3.1 405B FP8 using [vLLM](https://vllm.ai), a high-throughput and memory-efficient inference engine.

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

💡 This example is served as a basis for advanced code customization, such as custom model, inference logic or vLLM options. For simple LLM hosting with OpenAI compatible endpoint without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

## Prerequisites
- You have gained access to `NousResearch/Hermes-3-Llama-3.1-405B-FP8` on [Hugging Face](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B-FP8).
- If you want to test the Service locally, we recommend you use an Nvidia GPU with at least 16G VRAM.

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoVLLM.git
cd BentoVLLM/hermes-3-llama3.1-405b

# Recommend Python 3.11

pip install -r requirements.txt

export HF_TOKEN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. To run the service do the following:

```python
$ bentoml serve service.py:VLLM
```

The server is now active at [http://localhost:3000](http://localhost:3000/). You can interact with it using the Swagger UI or in other different ways.

> [!NOTE]
> This ships with a default `max_model_len=2048`. If you wish to change this value, set `MAX_MODEL_LEN=<target_context_len>`. Make sure that you have enough VRAM to use this context length. BentoVLLM will only set a conservative value based on this model configuration.

<details>

<summary>cURL</summary>

```bash
curl -X 'POST' \
  'http://localhost:3000/generate' \
  -H 'accept: text/event-stream' \
  -H 'Content-Type: application/json' \
  -d '{
  "prompt": "Who are you? Please respond in pirate speak!",
}'
```

</details>

<details>

<summary>Python client</summary>

```python
import bentoml

with bentoml.SyncHTTPClient("http://localhost:3000") as client:
    response_generator = client.generate(
        prompt="Who are you? Please respond in pirate speak!",
    )
    for response in response_generator:
        print(response, end='')
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
    model="NousResearch/Hermes-3-Llama-3.1-405B-FP8",
    messages=[
        {
            "role": "user",
            "content": "Who are you? Please respond in pirate speak!"
        }
    ],
    stream=True,
)
for chunk in chat_completion:
    # Extract and print the content of the model's reply
    print(chunk.choices[0].delta.content or "", end="")
```

These OpenAI-compatible endpoints also support [vLLM extra parameters](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#extra-parameters). For example, you can force the chat completion output a JSON object by using the `guided_json` parameters:

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:3000/v1', api_key='na')

# Use the following func to get the available models
client.models.list()

json_schema = {
    "type": "object",
    "properties": {
        "city": {"type": "string"}
    }
}

chat_completion = client.chat.completions.create(
    model="NousResearch/Hermes-3-Llama-3.1-405B-FP8",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
    extra_body=dict(guided_json=json_schema),
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

bentoml deploy service:VLLM --secret huggingface
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/guides/containerization.html).