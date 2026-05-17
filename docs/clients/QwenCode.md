# QwenCode Integration

QwenCode can talk directly to the local OpenAI-compatible endpoint once you point it at the host server. Use vLLM-Metal on 8000 for Qwen3.6-35B-A3B chat/tools/thinking; MLX endpoints remain available for older experiments.

## Quick mode

Copy the shipped settings into your home directory:
```bash
cp .qwen/settings.json ~/.qwen/settings.json
```
If you run QwenCode on the host, use `http://127.0.0.1:8000/v1`. From a devcontainer, use `http://host.docker.internal:8000/v1`.

## Manual setup

1. Open QwenCode settings → **AI** → **Custom Provider**.
2. Fill in:
   - **Name:** `Qwen Local` or similar.
   - **Base URL:** `http://127.0.0.1:8000/v1` on host, or `http://host.docker.internal:8000/v1` from a devcontainer.
   - **API Key:** `local` (any string works as long as it matches your `.qwen` config).
   - **Model ID:** Use the exact ID returned by `curl http://127.0.0.1:<port>/v1/models` (for the script default, `qwen3.6-35b-a3b`).
3. Restart QwenCode after changing providers to apply the new endpoint.

## Switching servers

- **vLLM-Metal (port 8000):** recommended for Qwen3.6-35B-A3B tool calling and thinking output.
- **TurboQuant Optiq (port 8080):** legacy long-context experiment path. Probe the port first; it often conflicts with other local tools.
- **Standard MLX server (port 8000):** usable fallback when you do not need vLLM tool-call parsing.

Keep both entries if you want to toggle between them without reconfiguring: QwenCode will remember the last used provider.

## Tips

- Tail the server terminal for vLLM-Metal, or `logs/mlx-server.log` / `logs/mlx-optiq-server.log` for MLX servers.
- Ensure the base URL and API key in `.qwen/settings.json` match the entry you selected in the UI.
- If you change sampling params in the JSON file, restart QwenCode so it reloads the config.
