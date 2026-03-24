# vLLM Batch Service — Installation & Usage Guide

A multi-model vLLM serving layer built on BentoML. Each model runs as a
`vllm serve` subprocess with full OpenAI API compatibility — tool calling,
reasoning/thinking, vision, quantization, and streaming all work out of the box.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
  - [Global Settings](#global-settings)
  - [Per-Model Settings](#per-model-settings)
  - [Feature Reference](#feature-reference)
- [Running the Service](#running-the-service)
- [API Reference](#api-reference)
  - [Chat Completions](#chat-completions)
  - [Batch Endpoint](#batch-endpoint)
  - [Model Management](#model-management)
- [Feature Guides](#feature-guides)
  - [Tool Calling (Function Calling)](#tool-calling-function-calling)
  - [Reasoning / Thinking](#reasoning--thinking)
  - [Vision / Multimodal](#vision--multimodal)
  - [Quantized Models](#quantized-models)
  - [FP8 KV Cache](#fp8-kv-cache)
- [GPU Memory Planning](#gpu-memory-planning)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python** 3.10+
- **NVIDIA GPU(s)** with CUDA 12.1+ and driver 535+
- **HuggingFace account** with access to gated models (Llama, etc.)

```bash
# Verify GPU setup
nvidia-smi
python -c "import torch; print(torch.cuda.device_count(), 'GPU(s)')"
```

## Installation

```bash
# Clone the repo
git clone https://github.com/dongha-bahn/bentovllm.git
cd bentovllm

# Install dependencies
pip install -r requirements.txt

# Additional dependency for the batch service proxy
pip install httpx

# Login to HuggingFace (for gated models like Llama)
huggingface-cli login
```

### Verify installation

```bash
python -c "import bentoml, vllm, yaml, httpx; print('All dependencies OK')"
```

---

## Configuration

Edit `models_config.yaml` to register models. The service reads this file at
startup.

### Global Settings

```yaml
models_dir: "/data/models"          # Base dir for local model files
idle_timeout_seconds: 1800          # Unload idle models after 30 min
max_batch_size: 128                 # Max requests per /v1/batch call
```

### Per-Model Settings

```yaml
models:
  my-model:
    path: "meta-llama/Meta-Llama-3.1-8B-Instruct"  # HF repo or local path
    tp: 1                          # Tensor parallelism (number of GPUs)
    gpu_memory_utilization: 0.40   # Fraction of VRAM to reserve
    max_model_len: 8192            # Context length cap
    max_num_seqs: 512              # Max concurrent sequences
    max_num_batched_tokens: 16384  # Max tokens per scheduler step
    dtype: "auto"                  # auto | float16 | bfloat16
```

### Feature Reference

| Field | Type | Description |
|---|---|---|
| `tool_parser` | string | Enable tool/function calling. Values: `hermes`, `llama3_json`, `pythonic`, `mistral`, `jamba`, `deepseek_v3` |
| `reasoning_parser` | string | Enable thinking/reasoning output. Values: `deepseek_r1`, `qwen3` |
| `quantization` | string | Quantization format. Values: `awq`, `gptq`, `fp8`, `gguf`, `bitsandbytes`, `experts_int8`, `marlin` |
| `kv_cache_dtype` | string | KV cache data type. Values: `auto`, `fp8`, `fp8_e4m3`, `fp8_e5m2` |
| `limit_mm_per_prompt` | object | Vision: max media per prompt, e.g. `{"image": 3}` |
| `attn_backend` | string | Attention implementation. Values: `FLASH_ATTN` (default), `FLASHMLA` |
| `trust_remote_code` | bool | Allow custom model code from HuggingFace |
| `chat_template` | string | Path to a custom Jinja chat template |
| `extra_args` | list | Additional `vllm serve` CLI arguments |

---

## Running the Service

### Start the service

```bash
bentoml serve batch_service:VLLMBatchService --port 3000
```

### Environment variables

```bash
# Override config file path
MODELS_CONFIG=my_config.yaml bentoml serve batch_service:VLLMBatchService

# Override GPU count (auto-detected by default)
NUM_GPUS=4 bentoml serve batch_service:VLLMBatchService

# HuggingFace token for gated models
HF_TOKEN=hf_xxx bentoml serve batch_service:VLLMBatchService
```

### What happens at startup

1. The service reads `models_config.yaml` and registers all models
2. **No models are loaded yet** — they start on first request (lazy loading)
3. When a request arrives for a model, a `vllm serve` subprocess starts on a
   dedicated port, the service waits for it to be healthy, then proxies the
   request
4. Idle models are automatically unloaded after `idle_timeout_seconds`

---

## API Reference

The service exposes an OpenAI-compatible API. Use any OpenAI client library.

### Chat Completions

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

**With streaming:**

```bash
curl http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1-8b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

**With the OpenAI Python SDK:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/v1", api_key="unused")

response = client.chat.completions.create(
    model="llama3.1-8b-instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

### Batch Endpoint

Submit multiple requests at once. All go to the same model and are processed
concurrently via vLLM's continuous batching.

```bash
curl http://localhost:3000/v1/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.1-8b-instruct",
    "requests": [
      {"messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 50},
      {"messages": [{"role": "user", "content": "What is 3+3?"}], "max_tokens": 50},
      {"messages": [{"role": "user", "content": "What is 4+4?"}], "max_tokens": 50}
    ]
  }'
```

Response:

```json
{
  "results": [
    {"id": "chatcmpl-...", "choices": [{"message": {"content": "4"}}], ...},
    {"id": "chatcmpl-...", "choices": [{"message": {"content": "6"}}], ...},
    {"id": "chatcmpl-...", "choices": [{"message": {"content": "8"}}], ...}
  ]
}
```

### Model Management

```bash
# List all models with load status and features
curl http://localhost:3000/v1/models

# Check GPU allocation
curl http://localhost:3000/v1/status

# Force-unload a model
curl -X POST http://localhost:3000/v1/models/llama3.1-8b-instruct/unload

# Health check
curl http://localhost:3000/health
```

---

## Feature Guides

### Tool Calling (Function Calling)

Configure `tool_parser` in `models_config.yaml`:

```yaml
models:
  llama3.1-8b-instruct:
    path: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tool_parser: "llama3_json"   # Llama 3.1+ JSON format
```

**Which parser for which model:**

| Model family | Parser | Notes |
|---|---|---|
| Llama 3.1 | `llama3_json` | JSON with `name` + `parameters` |
| Llama 3.2+, Llama 4 | `pythonic` | Python function call syntax |
| Qwen3 | `hermes` | Hermes tool format |
| Mistral | `mistral` | Mistral native format |
| Jamba (AI21) | `jamba` | Jamba native format |
| DeepSeek V3 | `deepseek_v3` | DeepSeek tool format |
| DeepSeek R1 | `hermes` | Uses Hermes format for tools |

**Usage example (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:3000/v1", api_key="unused")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"},
            },
            "required": ["location"],
        },
    },
}]

response = client.chat.completions.create(
    model="llama3.1-8b-instruct",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto",
)

# The model may return a tool_call in the response
tool_calls = response.choices[0].message.tool_calls
if tool_calls:
    print(f"Function: {tool_calls[0].function.name}")
    print(f"Args: {tool_calls[0].function.arguments}")
```

### Reasoning / Thinking

Configure `reasoning_parser` for models with chain-of-thought capabilities:

```yaml
models:
  qwen3-8b:
    path: "Qwen/Qwen3-8B"
    reasoning_parser: "qwen3"      # Extracts thinking content
    tool_parser: "hermes"          # Can combine with tool calling!

  deepseek-r1-qwen3-8b:
    path: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    reasoning_parser: "deepseek_r1"  # Extracts <think> tags
    tool_parser: "hermes"
```

**Available reasoning parsers:**

| Parser | Models | Format |
|---|---|---|
| `deepseek_r1` | DeepSeek R1, Phi-4 Reasoning | `<think>...</think>` tags |
| `qwen3` | Qwen3 | Qwen3 thinking format |

The reasoning content appears in the API response alongside the final answer.

### Vision / Multimodal

Configure `limit_mm_per_prompt` for vision models:

```yaml
models:
  gemma3-4b:
    path: "google/gemma-3-4b-it"
    gpu_memory_utilization: 0.30
    limit_mm_per_prompt: {"image": 3}   # Up to 3 images per request
```

**Usage (OpenAI SDK):**

```python
response = client.chat.completions.create(
    model="gemma3-4b",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "What's in this image?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},
        ],
    }],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

Images can be provided as:
- **HTTP URLs**: `{"url": "https://..."}`
- **Base64**: `{"url": "data:image/png;base64,..."}`
- **Local files**: Requires `extra_args: ["--allowed-local-media-path", "/path"]` in config

### Quantized Models

Quantization reduces VRAM usage significantly. An 8B model:

| Format | Weight size | VRAM (approx) | Notes |
|---|---|---|---|
| FP16 (default) | ~16 GB | ~18 GB | Full precision |
| AWQ INT4 | ~5 GB | ~7 GB | Best quality/speed balance |
| GPTQ INT4 | ~5 GB | ~7 GB | Similar to AWQ |
| GGUF Q4_K_M | ~5 GB | ~7 GB | Experimental in vLLM, slower |
| FP8 | ~8 GB | ~10 GB | Minimal quality loss |

**AWQ (recommended for INT4):**

```yaml
models:
  llama3.1-8b-awq:
    path: "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4"
    gpu_memory_utilization: 0.25    # Smaller footprint
    quantization: "awq"
    tool_parser: "llama3_json"
```

**GPTQ:**

```yaml
models:
  llama3.1-8b-gptq:
    path: "TheBloke/Meta-Llama-3.1-8B-Instruct-GPTQ"
    gpu_memory_utilization: 0.25
    quantization: "gptq"
```

**GGUF Q4_K_M:**

```yaml
# NOTE: GGUF in vLLM is experimental and significantly slower than AWQ/GPTQ.
# Use the "repo_id:quant_type" path format for HF-hosted GGUF files.
models:
  qwen3-8b-gguf:
    path: "unsloth/Qwen3-8B-GGUF:Q4_K_M"
    gpu_memory_utilization: 0.25
    quantization: "gguf"
```

**MoE models with INT8 experts:**

```yaml
models:
  jamba1.6-mini:
    path: "ai21labs/AI21-Jamba-1.6-Mini"
    quantization: "experts_int8"    # Quantize only MoE expert layers
    tool_parser: "jamba"
```

### FP8 KV Cache

Using FP8 for the KV cache **halves** its memory usage, allowing either:
- **2x longer context** at the same concurrency, or
- **2x more concurrent requests** at the same context length

```yaml
models:
  llama3.1-8b-fp8kv:
    path: "meta-llama/Meta-Llama-3.1-8B-Instruct"
    gpu_memory_utilization: 0.40
    max_model_len: 16384          # Can go higher with FP8 KV
    kv_cache_dtype: "fp8"         # 50% KV cache memory savings
    tool_parser: "llama3_json"
```

**Valid `kv_cache_dtype` values:**

| Value | Description |
|---|---|
| `auto` | Match model dtype (default) |
| `fp8` | Hardware-selected FP8 format |
| `fp8_e4m3` | FP8 E4M3 (higher precision, smaller range) |
| `fp8_e5m2` | FP8 E5M2 (lower precision, larger range) |

---

## GPU Memory Planning

Multiple models can share a single GPU. The key constraint:
**sum of `gpu_memory_utilization` + 10% headroom ≤ 100%**

### Single GPU (80 GB A100) examples

**2 full-precision 8B models:**
```yaml
models:
  model-a:
    gpu_memory_utilization: 0.40    # 32 GB
  model-b:
    gpu_memory_utilization: 0.40    # 32 GB
    # Total: 80% + 10% headroom = 90% ✓
```

**3 quantized 8B models (AWQ):**
```yaml
models:
  model-a:
    gpu_memory_utilization: 0.25    # 20 GB
    quantization: "awq"
  model-b:
    gpu_memory_utilization: 0.25    # 20 GB
    quantization: "awq"
  model-c:
    gpu_memory_utilization: 0.25    # 20 GB
    quantization: "awq"
    # Total: 75% + 10% headroom = 85% ✓
```

**1 large model + 1 small model:**
```yaml
models:
  llama-70b:
    tp: 4                           # Across 4 GPUs
    gpu_memory_utilization: 0.85
  llama-8b:
    tp: 1                           # Single GPU (different one)
    gpu_memory_utilization: 0.40
```

### Idle eviction

When a new model doesn't fit, the service automatically evicts the
least-recently-used model. You can also force eviction:

```bash
curl -X POST http://localhost:3000/v1/models/model-a/unload
```

---

## Troubleshooting

### Model fails to start

```bash
# Check service logs for the vllm serve command and stderr
bentoml serve batch_service:VLLMBatchService --port 3000 2>&1 | tee service.log

# Common issues:
# - OOM: Lower gpu_memory_utilization or max_model_len
# - "model not found": Check path is a valid HF repo or local directory
# - Timeout: Large models take longer; the default health timeout is 300s
```

### CUDA out of memory

Reduce memory usage:
1. Lower `gpu_memory_utilization` (e.g. 0.40 → 0.30)
2. Lower `max_model_len` (e.g. 8192 → 4096)
3. Use quantization (`quantization: "awq"`)
4. Use FP8 KV cache (`kv_cache_dtype: "fp8"`)

### GGUF models are slow

This is expected. vLLM's GGUF support is experimental with ~93 tok/s
throughput. Use AWQ or GPTQ instead for production workloads.

### Tool calls not working

1. Ensure `tool_parser` is set in config and matches your model family
2. Pass `tools` in the request body (OpenAI format)
3. Check the model actually supports tool calling (instruction-tuned models only)

### Vision inputs rejected

1. Ensure `limit_mm_per_prompt` is set (e.g. `{"image": 3}`)
2. Don't exceed the configured limit
3. For local files, add `extra_args: ["--allowed-local-media-path", "/path"]`

### Port conflicts

The service allocates ports starting at 9100 for vllm serve subprocesses.
If these conflict with existing services, set `BASE_PORT` in `model_manager.py`.
