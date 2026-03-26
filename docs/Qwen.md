# Qwen

## Models
**Models**:
  - mlx-community/Qwen3.5-9B-OptiQ-4bit
  - mlx-community/Qwen3.5-27B-4bit
**Context:** 262144 tokens (256K)
**Memory:** ~8-10GB for model + ~20-30GB for 256K context

## TurboQuant KV Cache Compression

OptiQ models support **TurboQuant KV cache compression** for better long-context performance via `mlx-optiq`.

### Installation
Requirements: `mlx-lm >= 0.30.7` (for Qwen3.5 architecture support)

```bash
uv pip install "mlx-lm>=0.30.7"
uv pip install mlx-optiq
```

### Benefits
- **4.9x smaller KV cache** with 3.5-bit compression
- Near-zero accuracy loss on long-context benchmarks
- Reduced memory footprint for 256K context workflows

## Qwen Code + MLX + TurboQuant Setup

The correct setup for Qwen Code with MLX and TurboQuant KV cache:

**mlx-community/Qwen3.5-9B-OptiQ-4bit + mlx-optiq + custom OpenAI-compatible FastAPI wrapper**

### Option 1: Docker (Recommended)

**Build and run:**
```bash
cd ~/code/Agents/LocalFirst
docker compose up -d --build
```

**View logs:**
```bash
docker logs -f mlx-optiq-server
```

**Stop server:**
```bash
docker compose down
```

**Environment variables:**
```bash
# Optional: Set in .env or pass directly
export HUGGINGFACE_MODEL=mlx-community/Qwen3.5-9B-OptiQ-4bit
export HF_TOKEN=hf_...  # For faster downloads
docker compose up -d
```

### Option 2: Local Python

```bash
cd ~/code/Agents/LocalFirst
source .venv/bin/activate
python src/mlx-openai-optiq-server/mlx-openai-optiq-server.py
```

See `src/mlx-openai-optiq-server/` for the custom FastAPI wrapper that enables TurboQuantKVCache.

### 2. Configure Qwen Code

Qwen Code supports OpenAI-compatible endpoints via `~/.qwen/settings.json`.

#### Config provision
```bash
cp .qwen/settings.json ~/.qwen/settings.json
```

#### Manual setup
**Settings → AI → Custom Provider:**
- **Name:** `MLX TurboQuant`
- **Base URL:** `http://127.0.0.1:8080/v1`
- **API Key:** `mlx-local` (any value works)
- **Model ID:** `mlx-community/Qwen3.5-9B-OptiQ-4bit`

### 3. Verify Connection

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Alternative: Standard MLX Server (No TurboQuant)

For non-TurboQuant usage with `mlx-openai-server`:

```bash
export HUGGINGFACE_MODEL=mlx-community/Qwen3.5-9B-OptiQ-4bit
export CONTEXT_LENGTH=262144

mlx-openai-server launch \
  --model-path "$HUGGINGFACE_MODEL" \
  --model-type lm \
  --port 8000 \
  --host 0.0.0.0 \
  --context-length "$CONTEXT_LENGTH"
```

**Note:** `mlx-openai-server` does not support TurboQuant KV cache configuration.

## Inference params
- Thinking mode for general tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
- Thinking mode for precise coding tasks (e.g. WebDev): temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
- Instruct (or non-thinking) mode for general tasks: temperature=0.7, top_p=0.8, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
- Instruct (or non-thinking) mode for reasoning tasks: temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0

## Tips
- Keep the server running in background while using Qwen Code
- Use `tail -f /tmp/mlx-server.log` to monitor requests
- For best coding performance, set temperature to 0.6
