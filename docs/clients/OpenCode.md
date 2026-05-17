# OpenCode Integration

OpenCode expects an OpenAI-compatible provider description under `~/.config/opencode/opencode.json`. The repository already includes `opencode.json` with the recommended local Qwen endpoint and models.

## Installation

```bash
mkdir -p ~/.config/opencode
ln -sf opencode.json ~/.config/opencode/opencode.json
```

Reload or restart OpenCode after changing the config.

## Local Qwen setup

### Configuration
Add this provider block to `~/.config/opencode/opencode.json` if you want an explicit local setup:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "qwen-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Qwen Local",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
        "qwen3.6-35b-a3b": {
          "name": "Qwen3.6-35B-A3B vLLM-Metal"
        },
        "mlx-community/Qwen3.5-9B-OptiQ-4bit": {
          "name": "Qwen3.5-9B-OptiQ-4bit"
        },
        "mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit": {
          "name": "Qwen3.5-27B Opus Distilled"
        },
        "mlx-community/Qwen3.5-35B-A3B-4bit": {
          "name": "Qwen3.5-35B-A3B"
        }
      }
    }
  }
}
```

### Usage
1. Start the vLLM-Metal server (see [vLLM-Metal Qwen3.6 Server](../servers/VLLM-Metal.md)).
2. Restart OpenCode.
3. Select "Qwen Local" from the model dropdown.
4. Start coding with local AI assistance.

## Provider highlights

### Qwen Local
- **Base URL:** `http://localhost:8000/v1` on host, or `http://host.docker.internal:8000/v1` from a devcontainer.
- **Models:** `qwen3.6-35b-a3b` for the recommended vLLM-Metal server; MLX model IDs remain usable when the MLX server is running on the same port.
- **Notes:** The config uses the OpenAI-compatible SDK. Tool calls are emitted by the model; OpenCode/client code still executes the tools.

## Verifying the connection

1. Make sure the relevant server is running (`scripts/start-vllm-metal-qwen36.sh` for the recommended path).
2. Run `curl http://<server>/v1/models` to confirm the endpoint responds.
3. Open OpenCode and verify the provider dropdown now lists “Qwen Local”.

## Tips

- Keep the MLX server running in the background; OpenCode will attempt to connect each time you switch models.
- Use at least 16K context for agentic workflows.
- Set temperature to `0.2-0.7` for balanced creativity/accuracy.
