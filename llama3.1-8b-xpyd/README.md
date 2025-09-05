# BentoML Disaggregated Prefill-Decode Service

A high-performance LLM serving architecture that separates prefill and decode phases across different GPU instances for optimized throughput and latency.

## Overview

This service implements a disaggregated architecture for Large Language Model (LLM) inference, splitting the workload between:
- **Prefill nodes**: Handle initial prompt processing (compute-intensive)
- **Decode nodes**: Generate tokens (memory bandwidth-intensive)

The architecture uses vLLM's KV-cache transfer mechanism with P2P NCCL communication for efficient state transfer between nodes.

## Features

- **Disaggregated Architecture**: Separate prefill and decode phases for better resource utilization
- **Dynamic Service Discovery**: Automatic registration and health monitoring of prefill/decode instances
- **Load Balancing**: Round-robin distribution across available instances
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI API endpoints
- **KV-Cache Transfer**: Efficient P2P communication using NCCL for cache transfer
- **Auto-scaling Ready**: Built for cloud deployment with BentoML

## Components

### Router Service
- FastAPI application handling incoming requests
- Service discovery via ZeroMQ
- Request routing and load balancing
- OpenAI API compatibility layer

### Prefiller Service
- Processes initial prompts
- Generates KV-cache
- Transfers cache to decode nodes via P2P NCCL
- GPU-optimized for compute-intensive operations

### Decoder Service
- Receives KV-cache from prefill nodes
- Generates output tokens
- Streams responses back to clients
- GPU-optimized for memory bandwidth operations

## Requirements

- Python 3.11+
- NVIDIA GPU (H100 recommended)
- CUDA and NCCL libraries
- Docker (optional)

## Installation

```bash
cd llama3.1-8b-xpyd

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `components/config.py` to configure:

```python
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"  # Model to use
PROXY_PORT = 30001        # ZeroMQ service discovery port
PREFILL_KV_PORT = 20001   # Prefill KV transfer port
DECODE_KV_PORT = 21001    # Decode KV transfer port
```

## Environment Variables

- `HF_TOKEN`: Hugging Face token for model access

## Usage

### Local Development

```bash
# Start the service
bentoml serve

# The service will be available at:
# - Main API: http://localhost:3000
# - OpenAI-compatible endpoints:
#   - /v1/completions
#   - /v1/chat/completions

# Start XPYD
bentoml serve --arg num_prefill 2 --arg num_decode 3
```

### API Examples

```bash
# Chat completion
curl -X POST http://localhost:3000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'

# Text completion
curl -X POST http://localhost:3000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "prompt": "Once upon a time",
    "max_tokens": 100
  }'
```

### Deployment with BentoML

```bash
# Build the bento
bentoml build

# Deploy to BentoCloud
bentoml deploy
```

## Acknowledgments

- Built with [BentoML](https://github.com/bentoml/BentoML)
- Powered by [vLLM](https://github.com/vllm-project/vllm)
- Disaggregated architecture based on vLLM's distributed serving capabilities
