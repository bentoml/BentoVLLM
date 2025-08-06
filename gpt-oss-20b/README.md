<div align="center">
    <h1 align="center">Self-host GPT OSS MoE 20B with vLLM and BentoML</h1>
</div>

Follow this guide to self-host the GPT OSS MoE 20B model with BentoCloud in your own cloud account. If your team doesn’t already have access to BentoCloud, please use the link below to contact us and set it up in your cloud environment.

[![Deploy on BentoCloud](https://img.shields.io/badge/Deploy_on_BentoCloud-d0bfff?style=for-the-badge)](https://cloud.bentoml.com/)
[![Talk to sales](https://img.shields.io/badge/Talk_to_sales-eefbe4?style=for-the-badge)](https://bentoml.com/contact)

See [here](https://docs.bentoml.com/en/latest/examples/overview.html) for a full list of BentoML example projects.

💡 This example is served as a basis for advanced code customization, such as custom model, inference logic or vLLM options. For simple LLM hosting with OpenAI compatible endpoint without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

## Prerequisites
- You have gained access to `openai/gpt-oss-20b` on [Hugging Face](https://huggingface.co/openai/gpt-oss-20b).
- If you want to test the Service locally, we recommend you use Nvidia GPUs with at least 80GB VRAM (e.g about 1 H100 GPU).

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoVLLM.git
cd BentoVLLM/gpt-oss-20b

# Recommend Python 3.11
pip install -r requirements.txt

# if you are running locally, we recommend install additional flashinfer library for better performance.
pip install flashinfer-python --extra-index-url https://flashinfer.ai/whl/cu124/torch2.6

export HF_TOKEN=<your-api-key>
```

## Run the BentoML Service

We have defined a BentoML Service in `service.py`. To run the service do the following:

```python
$ bentoml serve service.py:LLM
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

bentoml deploy service:LLM --secret huggingface
```

Once the application is up and running on BentoCloud, you can access it via the exposed URL.

**Note**: For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).