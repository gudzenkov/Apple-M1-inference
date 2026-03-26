# MLX Metal

Apple GPU (Metal) is used automatically by MLX for hardware acceleration.

## Table of Contents
- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [OpenAI-Compatible Server](#openai-compatible-server)
- [OpenCode Integration](#opencode-integration)
- [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Setup Environment
```bash
uv venv
source .venv/bin/activate
uv pip install mlx mlx-lm huggingface_hub
```

**Requirements:**
- `mlx-lm >= 0.30.7` (for Qwen3.5 architecture support)
- `mlx-optiq` (optional, for TurboQuant KV cache compression)

See [mlx-community/Qwen3.5-9B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.5-9B-OptiQ-4bit) for model details.

### 2. Configure HuggingFace Token (Recommended)
For faster downloads and higher rate limits:
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_...

# Set default model
export HUGGINGFACE_MODEL=mlx-community/Qwen3.5-9B-OptiQ-4bit
```

## Basic Usage

### Generate Text
```bash
# Models auto-download to ~/.cache/huggingface/hub on first use
# Downloads resume automatically if interrupted
python -m mlx_lm generate \
  --model mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --prompt "Write a Kotlin coroutine example." \
  --max-tokens 512 \
  --temp 0.2
```

### Interactive Chat
```bash
python -m mlx_lm chat \
  --model mlx-community/Qwen3.5-9B-OptiQ-4bit
```

### Python API
```python
from mlx_lm import load, generate

# Uses cached model from ~/.cache/huggingface/hub
model, tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")

response = generate(
    model,
    tokenizer,
    "Explain vector search briefly.",
    max_tokens=300
)
print(response)
```

### Monitor Downloads
```bash
# Check cache size
du -sh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-OptiQ-4bit

# Check incomplete files (still downloading)
ls -lh ~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-OptiQ-4bit/blobs/*.incomplete
```

## OpenAI-Compatible Server

### Install Server Package
```bash
uv pip install mlx-openai-server
```

### Start Server

**Foreground (see logs in terminal):**
```bash
# Activate venv first
cd ~/code/Agents/LocalFirst
source .venv/bin/activate

# Start server with 256k context (Ctrl+C to stop)
mlx-openai-server launch \
  --model-path mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --model-type lm \
  --port 8000 \
  --host 0.0.0.0 \
  --context-length 262144
```

**Background (recommended for development):**
```bash
# Activate venv
cd ~/code/Agents/LocalFirst
source .venv/bin/activate

# Start in background with 256k context
nohup mlx-openai-server launch \
  --model-path mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --model-type lm \
  --port 8000 \
  --host 0.0.0.0 \
  --context-length 262144 \
  > /tmp/mlx-server.log 2>&1 &

# View logs
tail -f /tmp/mlx-server.log
```

**With custom settings:**
```bash
mlx-openai-server launch \
  --model-path mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --model-type lm \
  --port 8000 \
  --host 0.0.0.0 \
  --context-length 262144 \
  --max-tokens 2048 \
  --temperature 0.7
```

### Server Management

**Check if server is running:**
```bash
# Check process
ps aux | grep mlx-openai-server | grep -v grep

# Test endpoint
curl -s http://localhost:8000/v1/models
```

**Stop server:**
```bash
# Find process ID
ps aux | grep mlx-openai-server | grep -v grep

# Stop gracefully
pkill -f mlx-openai-server

# Or force stop (if needed)
pkill -9 -f mlx-openai-server
```

**Restart server:**
```bash
# Stop
pkill -f mlx-openai-server

# Wait a moment
sleep 2

# Start again with 256k context
cd ~/code/Agents/LocalFirst
source .venv/bin/activate
nohup mlx-openai-server launch \
  --model-path mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --model-type lm \
  --port 8000 \
  --host 0.0.0.0 \
  --context-length 262144 \
  > /tmp/mlx-server.log 2>&1 &
```

**Check resource usage:**
```bash
# Memory usage (~20-25GB expected for 27B model)
ps aux | grep mlx-openai-server | grep -v grep | awk '{print $6/1024/1024 " GB"}'

# View server logs
tail -f /tmp/mlx-server.log
```

### Test Endpoints
```bash
# List available models
curl http://localhost:8000/v1/models

# Chat completions
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    "messages": [
      {"role": "user", "content": "Explain async/await in JavaScript"}
    ],
    "max_tokens": 200
  }'
```

## OpenCode Integration

### Configuration
Add to `.config/opencode/opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX Local (Qwen3.5-27B)",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "mlx-community/Qwen3.5-9B-OptiQ-4bit": {
          "name": "Qwen3.5-9B-OptiQ-4bit"
        }
      }
    }
  }
}
```

### Usage
1. Start the MLX server (see [OpenAI-Compatible Server](#openai-compatible-server))
2. Restart OpenCode
3. Select "MLX Local (Qwen3.5-27B)" from the model dropdown
4. Start coding with local AI assistance

### Tips for OpenCode
- Use at least 16K context for agentic workflows
- Qwen3-Coder models work best for code generation
- Set temperature to 0.2-0.7 for balanced creativity/accuracy

## Troubleshooting

### Path Validation Errors
**Error:** `Repo id must use alphanumeric chars...`

**Solution:** Use HuggingFace repo ID directly, not local paths:
```bash
# ✗ Wrong
--model ./qwen35-27b-mlx

# ✓ Correct
--model mlx-community/Qwen3.5-9B-OptiQ-4bit
```

### Slow Downloads
**Solution:** Set HF_TOKEN for authenticated access with higher rate limits (see [Quick Start](#quick-start)).

### Download Not Resuming
Downloads automatically resume from `.incomplete` files. If stuck:
1. Check cache: `du -sh ~/.cache/huggingface/hub/models--*`
2. Verify incomplete files exist
3. Restart command - it will resume automatically
4. Files grow even if progress bar shows rounded values (e.g., 2.0G → 2.4G)

### Server Not Starting
**Error:** Port already in use

**Solution:**
1. Check what's using the port: `lsof -i :8000`
2. Use a different port (avoid 8080 - used by OrbStack)
3. Kill conflicting process if needed

### curl Returns Nothing
**Common Issues:**
1. Wrong endpoint - use `/v1/chat/completions` not `/v1/completions`
2. Server not started - check with `ps aux | grep mlx-openai-server`
3. Wrong port - verify server is on expected port
4. Wrong syntax - use `mlx-openai-server launch` not `mlx-openai-server --model`

### Command Not Found: mlx-openai-server
**Error:** `zsh: command not found: mlx-openai-server`

**Cause:** The venv is not activated in your current shell session.

**Solution:**
```bash
# Option 1: Activate the venv
cd ~/code/Agents/LocalFirst
source .venv/bin/activate
mlx-openai-server launch ...

# Option 2: Use full path (any directory)
~/code/Agents/LocalFirst/.venv/bin/mlx-openai-server launch ...

# Option 3: Check if server is already running
ps aux | grep mlx-openai-server | grep -v grep
curl -s http://localhost:8000/v1/models
```

### OpenCode Can't Connect
1. Ensure server is running: `curl http://localhost:8000/v1/models`
2. Check baseURL in opencode.json matches server port
3. Restart OpenCode after config changes
4. Check server logs: `tail -f logs/app.log`

## Additional Resources

- [mlx-openai-server GitHub](https://github.com/cubist38/mlx-openai-server) - OpenAI-compatible API server for MLX
- [OpenCode Providers](https://opencode.ai/docs/providers/) - OpenCode provider configuration
- [OpenCode Models](https://opencode.ai/docs/models/) - OpenCode model setup
- [MLX Examples](https://github.com/ml-explore/mlx-examples) - Official MLX examples

**Sources:**
- [mlx-openai-server · PyPI](https://pypi.org/project/mlx-openai-server/)
- [GitHub - cubist38/mlx-openai-server](https://github.com/cubist38/mlx-openai-server)
- [Providers | OpenCode](https://opencode.ai/docs/providers/)
- [Local LLM with OpenCode](https://tobrun.github.io/blog/add-openai-compatible-endpoint-to-opencode/)
- [Setting Up OpenCode with Local Models](https://theaiops.substack.com/p/setting-up-opencode-with-local-models)
