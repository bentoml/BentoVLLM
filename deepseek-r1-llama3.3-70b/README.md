<div align="center">
    <h1 align="center">Self-host DeepSeek R1 Distill Llama 3.3 70B with vLLM and BentoML</h1>
</div>

💡 This example is served as a basis for advanced code customization, such as custom model, inference logic or vLLM options. For simple LLM hosting with OpenAI compatible endpoint without writing any code, see [OpenLLM](https://github.com/bentoml/OpenLLM).

## Prerequisites
- You have gained access to `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` on [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B).
- If you want to test the Service locally, we recommend you use Nvidia GPUs with at least 80GBx4 VRAM (e.g about 4 H100 GPUs).

## Install dependencies

```bash
git clone https://github.com/bentoml/BentoVLLM.git
cd BentoVLLM/deepseek-r1-llama3.3-70b

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

For custom deployment in your own infrastructure, use [BentoML to generate an OCI-compliant image](https://docs.bentoml.com/en/latest/get-started/packaging-for-deployment.html).