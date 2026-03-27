# MLX Metal Runbook

MLX inference server uses Apple GPU (Metal) for native hardware acceleration.
Docker is not supported for MLX inference in this repo because Linux VM containers do not expose native Apple MLX/Metal execution.

## Table of Contents
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [OpenAI-Compatible Servers](#openai-compatible-servers)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## Quick Start

### 1. Environment setup
```bash
cd "$(git rev-parse --show-toplevel)"
uv venv
source .venv/bin/activate
uv sync
```

Requirements:
- `mlx-lm >= 0.30.7` (Qwen3.5 architecture support)
- `mlx-optiq` (optional, for TurboQuant KV cache compression)

### 2. Optional: Hugging Face token
Use a token for faster downloads and better rate limits:
```bash
# Get token: https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...
```

### 3. Shared defaults
```bash
export HUGGINGFACE_MODEL=mlx-community/Qwen3.5-9B-OptiQ-4bit
export HOST=0.0.0.0
```

`HUGGINGFACE_MODEL` accepts a configured alias or full model ID from `configs/models.yaml`.
Common aliases: `optiq`, `opus`, `claw`, `coder`.

Model card (default alias `optiq`): [mlx-community/Qwen3.5-9B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-OptiQ-4bit)

## Basic Usage

### Generate text
```bash
python -m mlx_lm generate \
  --model mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --prompt "Write a Kotlin coroutine example." \
  --max-tokens 512 \
  --temp 0.2
```

### Interactive chat
```bash
python -m mlx_lm chat \
  --model mlx-community/Qwen3.5-9B-OptiQ-4bit
```

### Python API
```python
from mlx_lm import generate, load

model, tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")
response = generate(model, tokenizer, "Explain vector search briefly.", max_tokens=300)
print(response)
```

### Monitor downloads
```bash
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-OptiQ-4bit
ls -lh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-OptiQ-4bit/blobs/*.incomplete
```

## OpenAI-Compatible Servers

This repo provides two OpenAI-compatible endpoints:
- `mlx-openai-server` on port `8000` (standard MLX)
- `mlx-openai-optiq-server` on port `8080` (TurboQuant KV cache via `mlx-optiq`)

### 1. Standard server (port 8000)
Foreground:
```bash
uv run mlx-openai-server serve
```

Background:
```bash
uv run mlx-openai-server start
```

Stop / status:
```bash
uv run mlx-openai-server stop
uv run mlx-openai-server status
```

### 2. Optiq server (port 8080)
Foreground:
```bash
uv run mlx-openai-optiq-server serve
```

Background:
```bash
uv run mlx-openai-optiq-server start
```

Stop / status:
```bash
uv run mlx-openai-optiq-server stop
uv run mlx-openai-optiq-server status
```

### Server checks
```bash
# Process checks
ps aux | grep -E "mlx-openai-server|mlx-openai-optiq-server" | grep -v grep

# Health checks
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8080/v1/models

# Logs
tail -f logs/mlx-server.log
tail -f logs/mlx-optiq-server.log
```

### Endpoint test
```bash
# Standard server
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    "messages": [{"role": "user", "content": "Explain async/await in JavaScript"}],
    "max_tokens": 200
  }'

# Optiq server
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    "messages": [{"role": "user", "content": "Explain async/await in JavaScript"}],
    "max_tokens": 200
  }'
```

## Troubleshooting

### Repo/path validation error
Error: `Repo id must use alphanumeric chars...`

Use a Hugging Face repo ID, not a local path:
```bash
# Wrong
--model ./qwen35-27b-mlx

# Correct
--model mlx-community/Qwen3.5-9B-OptiQ-4bit
```

### Slow or interrupted downloads
- Set `HF_TOKEN` for authenticated download speed.
- Downloads resume from `.incomplete` files automatically.
- If stuck, rerun the same command and verify cache activity:
  - `du -sh ~/.cache/huggingface/hub/models--*`
  - `ls -lh ~/.cache/huggingface/hub/models--*/blobs/*.incomplete`

### Server does not start
Error: port already in use.

Check port usage and switch ports if needed:
```bash
lsof -i :8000
lsof -i :8080
```

### `curl` returns nothing
1. Use `/v1/chat/completions` (not `/v1/completions`).
2. Confirm process is running and port is correct (`8000` or `8080`).
3. Check logs in `logs/mlx-server.log` or `logs/mlx-optiq-server.log`.

## Resources

- [mlx-openai-server GitHub](https://github.com/cubist38/mlx-openai-server)
- [mlx-openai-server PyPI](https://pypi.org/project/mlx-openai-server/)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
