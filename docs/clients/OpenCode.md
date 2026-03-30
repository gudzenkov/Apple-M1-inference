# OpenCode Integration

OpenCode expects an OpenAI-compatible provider description under `~/.config/opencode/opencode.json`. The repository already includes `opencode.json` which registers both the MLX and Ollama endpoints with the recommended models.

## Installation

```bash
mkdir -p ~/.config/opencode
ln -sf opencode.json ~/.config/opencode/opencode.json
```

Reload or restart OpenCode after changing the config.

## MLX-specific setup

### Configuration
Add this provider block to `~/.config/opencode/opencode.json` if you want an explicit MLX-only setup:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "mlx-local": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX Local",
      "options": {
        "baseURL": "http://localhost:8000/v1"
      },
      "models": {
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
1. Start the MLX server (see [MLX OpenAI-Compatible Servers](../servers/MLX.md#openai-compatible-servers)).
2. Restart OpenCode.
3. Select "MLX Local" from the model dropdown.
4. Start coding with local AI assistance.

## Provider highlights

### MLX Local
- **Base URL:** `http://localhost:8000/v1` (or `http://localhost:8080/v1` when running the TurboQuant Optiq server).
- **Models:** Keep this list aligned with `configs/models.yaml` for `runtime: mlx`.
- **Notes:** The config marks the provider with the OpenAI-compatible SDK so OpenCode can show both models alongside Ollama.

### Ollama Local
- **Base URL:** `http://localhost:11434/v1`.
- **Models:** `qwen3.5:9b`, `RogerBen/qwen3.5-35b-opus-distill`, `sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M`.
- **Tool support:** All configured models have `supportsTools` enabled with `toolChoice: auto`, matching the expectation of agentic workflows.

## Verifying the connection

1. Make sure the relevant server is running (`mlx-openai-server`, `mlx-openai-optiq-server`, or Ollama service).
2. Run `curl http://<server>/v1/models` to confirm the endpoint responds.
3. Open OpenCode and verify the provider dropdown now lists “MLX Local” and “Ollama Local”.

## Tips

- Keep the MLX and Ollama servers running in the background; OpenCode will attempt to connect each time you switch models.
- If you create new Ollama models (e.g., `*-256k`), update `opencode.json` accordingly and restart OpenCode.
- Use at least 16K context for agentic workflows.
- Set temperature to `0.2-0.7` for balanced creativity/accuracy.
