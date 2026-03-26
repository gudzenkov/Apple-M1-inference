# OpenCode Integration

OpenCode expects an OpenAI-compatible provider description under `~/.config/opencode/opencode.json`. The repository already includes `opencode.json` which registers both the MLX and Ollama endpoints with the recommended models.

## Installation

```bash
mkdir -p ~/.config/opencode
ln -sf ~/code/Agents/LocalFirst/opencode.json ~/.config/opencode/opencode.json
```

Reload or restart OpenCode after changing the config.

## Provider highlights

### MLX Local
- **Base URL:** `http://localhost:8000/v1` (or `http://localhost:8080/v1` when running the TurboQuant Optiq server).
- **Models:** `mlx-community/Qwen3.5-9B-OptiQ-4bit` and `mlx-community/Qwen3.5-27B-4bit`.
- **Notes:** The config marks the provider with the OpenAI-compatible SDK so OpenCode can show both models alongside Ollama.

### Ollama Local
- **Base URL:** `http://localhost:11434/v1`.
- **Models:** `qwen3.5:9b`, `SimonPu/Qwen3-Coder:30B-Instruct_Q4_K_XL`, `sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M`, and `ukjin/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill`.
- **Tool support:** All configured models have `supportsTools` enabled with `toolChoice: auto`, matching the expectation of agentic workflows.

## Verifying the connection

1. Make sure the relevant server is running (`mlx-openai-server`, `mlx-openai-optiq-server`, or Ollama service).
2. Run `curl http://<server>/v1/models` to confirm the endpoint responds.
3. Open OpenCode and verify the provider dropdown now lists “MLX Local (Qwen3.5-27B)” and “Ollama Local”.

## Tips

- Keep the MLX and Ollama servers running in the background; OpenCode will attempt to connect each time you switch models.
- If you create new Ollama models (e.g., `*-256k`), update `opencode.json` accordingly and restart OpenCode.
