# Ollama

Run large language models locally with Ollama.

## Table of Contents
- [Installation](#installation)
- [Model Management](#model-management)
- [Server Management](#server-management)
- [OpenAI-Compatible API](#openai-compatible-api)
- [OpenCode Integration](#opencode-integration)

## Installation

### macOS
```bash
# Install via Homebrew
brew install ollama

# Or download from https://ollama.com
```

### Start Ollama Service
```bash
# Start server (runs in background)
ollama serve

# Or use system service (auto-starts on boot)
brew services start ollama
```

## Model Management

### Pull Models
```bash
# Pull current models (with tool support)
ollama pull qwen3.5:9b
ollama pull SimonPu/Qwen3-Coder:30B-Instruct_Q4_K_XL
ollama pull sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M
ollama pull ukjin/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill
```

### List Models
```bash
ollama list
```

### Remove Models
```bash
ollama rm <model-name>
```

### Model Information
```bash
ollama show <model-name>
```

### Set 256k Context (Recommended)

**Option 1: Create Modelfile with extended context**
```bash
# Create Modelfile for each model
./setup-256k-context.sh
```

**Option 2: Set context at runtime**
```bash
# Run with 256k context (temporary)
OLLAMA_CONTEXT_LENGTH=262144 ollama run qwen3.5:9b

# Or via API
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "test"}],
    "num_ctx": 262144
  }'
```

## Server Management

### Check Status
```bash
# Check if Ollama is running
ps aux | grep ollama | grep -v grep

# Test API
curl http://localhost:11434/api/tags
```

### Start/Stop Service
```bash
# Start
brew services start ollama
# Or manually
ollama serve &

# Stop
brew services stop ollama
# Or manually
pkill ollama
```

### Server Configuration
Default settings:
- Port: `11434`
- Host: `127.0.0.1`
- Models directory: `~/.ollama/models`

Environment variables:
```bash
# Change host/port
export OLLAMA_HOST=0.0.0.0:11434

# Change models directory
export OLLAMA_MODELS=/path/to/models
```

## OpenAI-Compatible API

Ollama provides OpenAI-compatible endpoints at `/v1`.

### Chat Completions
```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [
      {"role": "user", "content": "Explain async/await in JavaScript"}
    ],
    "max_tokens": 200,
    "num_ctx": 262144
  }'
```

### List Models
```bash
curl http://localhost:11434/v1/models
```

### Streaming
```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5:9b",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true,
    "num_ctx": 262144
  }'
```

## OpenCode Integration

### Configuration
Add to `~/.config/opencode/opencode.json`:

**⚠️ Important:** OpenCode requires models with tool/function calling support for agentic features.

**Recommended models with tool support:**
```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "ollama": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Ollama Local",
      "options": {
        "baseURL": "http://localhost:11434/v1"
      },
      "models": {
        "qwen3.5:9b": {
          "name": "Qwen3.5-9B (Fast, Thinking+Tools)",
          "supportsTools": true,
          "toolChoice": "auto"
        },
        "SimonPu/Qwen3-Coder:30B-Instruct_Q4_K_XL": {
          "name": "Qwen3-Coder-30B (Baseline, Tools)",
          "supportsTools": true,
          "toolChoice": "auto"
        },
        "sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M": {
          "name": "Qwen3.5-Opus-27B (Thinking+Tools)",
          "supportsTools": true,
          "toolChoice": "auto"
        },
        "ukjin/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill": {
          "name": "Qwen3-Deepseek-30B (Thinking+Tools)",
          "supportsTools": true,
          "toolChoice": "auto"
        }
      }
    }
  }
}
```


### Usage
1. Ensure Ollama is running: `brew services start ollama`
2. Pull the model (if not already done)
3. Restart OpenCode
4. Select "Ollama Local" from the model dropdown

## Performance Tips

### GPU Acceleration
Ollama automatically uses Metal (Apple Silicon GPU) for acceleration.

### Memory Management
```bash
# Check GPU memory usage
sudo powermetrics --samplers gpu_power -i 1000 -n 1

# Limit concurrent requests
export OLLAMA_NUM_PARALLEL=1
```

### Context Length
```bash
# Set 256k context (recommended for agentic workflows)
OLLAMA_CONTEXT_LENGTH=262144 ollama run <model>

# Or smaller for basic tasks
OLLAMA_CONTEXT_LENGTH=32768 ollama run <model>
```

## Troubleshooting

### OpenCode: "does not support tools"
**Error:** `AI_APICallError: <model> does not support tools`

**Cause:** The model doesn't support function calling/tools required by OpenCode.

**Solution:** Use models with tool support:
```bash
# All current models have tool support
ollama pull qwen3.5:9b
ollama pull SimonPu/Qwen3-Coder:30B-Instruct_Q4_K_XL
ollama pull sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M
ollama pull ukjin/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill

# Update OpenCode config to use these models
```

**Models WITH tool support:**
- qwen3.5:9b ✅
- qwen3.5:* ✅
- SimonPu/Qwen3-Coder:30B-Instruct_Q4_K_XL ✅
- sinhang/qwen3.5-claude-4.6-opus:* ✅
- ukjin/Qwen3-30B-A3B-Thinking-* ✅
- llama3.1:* ✅
- mistral:* ✅

### Server Won't Start
```bash
# Check if port is in use
lsof -i :11434

# Kill existing process
pkill ollama

# Restart
ollama serve
```

### Model Download Failed
```bash
# Clear cache and retry
rm -rf ~/.ollama/models/<model-name>
ollama pull <model-name>
```

### Out of Memory
- Close other applications
- Use smaller quantization (Q4 instead of Q8)
- Reduce context length
- Limit parallel requests

## Additional Resources

- [Ollama Official Site](https://ollama.com)
- [Ollama GitHub](https://github.com/ollama/ollama)
- [Model Library](https://ollama.com/library)
- [API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
