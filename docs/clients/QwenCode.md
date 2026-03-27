# QwenCode Integration

QwenCode can talk directly to the local MLX endpoints once you point it at the OpenAI-compatible server running on 8000 (standard `mlx-openai-server`) or 8080 (TurboQuant `mlx-openai-optiq-server`).

## Quick mode

Copy the shipped settings into your home directory:
```bash
cp .qwen/settings.json ~/.qwen/settings.json
```
The file is already configured for the Optiq server (`http://127.0.0.1:8080/v1`) with `LOCAL_LLM_API_KEY=local` as the API key and a default generation config tuned for longer contexts.

## Manual setup

1. Open QwenCode settings → **AI** → **Custom Provider**.
2. Fill in:
   - **Name:** `MLX Optiq` or similar.
   - **Base URL:** `http://127.0.0.1:8080/v1` for TurboQuant or `http://127.0.0.1:8000/v1` for the standard MLX service.
   - **API Key:** `local` (any string works as long as it matches your `.qwen` config).
   - **Model ID:** Use the exact ID returned by `curl http://127.0.0.1:<port>/v1/models` (for example `mlx-community/Qwen3.5-9B-OptiQ-4bit`).
3. Restart QwenCode after changing providers to apply the new endpoint.

## Switching servers

- **TurboQuant Optiq (port 8080):** keeps the KV cache compressed while honoring 256K context. Use this when you need the best RAM economy.
- **Standard MLX server (port 8000):** easier to script and good for general workflows; visit `http://localhost:8000/v1/models` to confirm readiness.

Keep both entries if you want to toggle between them without reconfiguring: QwenCode will remember the last used provider.

## Tips

- Tail `logs/mlx-server.log` or `logs/mlx-optiq-server.log` to watch requests from QwenCode.
- Ensure the base URL and API key in `.qwen/settings.json` match the entry you selected in the UI.
- If you change sampling params in the JSON file, restart QwenCode so it reloads the config.
