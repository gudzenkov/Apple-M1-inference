# MLX Runbook

Apple Silicon users can unlock the highest-quality Qwen architectures via MLX, with Metal acceleration handled automatically.

## Prerequisites

1. Create and activate a virtual environment:
   ```bash
   uv venv
   source .venv/bin/activate
   ```
2. Install the MLX stack:
   ```bash
   uv pip install mlx mlx-lm mlx-openai-server mlx-optiq huggingface_hub
   ```
3. (Optional) Export your HuggingFace token and preferred model:
   ```bash
   export HF_TOKEN=hf_...
   export HUGGINGFACE_MODEL=mlx-community/Qwen3.5-9B-OptiQ-4bit
   ```

The model cache lands in `~/.cache/huggingface/hub`; downloads resume automatically when interrupted.

## Basic MLX usage

Use `mlx_lm` for direct runs:
```bash
python -m mlx_lm generate \
  --model mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --prompt "Explain async/await" \
  --max-tokens 512 --temp 0.2
```
The chat helper is identical:
```bash
python -m mlx_lm chat --model mlx-community/Qwen3.5-9B-OptiQ-4bit
```
Need inference tuning or TurboQuant guidance? See `docs/Qwen.md` for model details and sampling presets.

## OpenAI-compatible server

### Standard MLX server (`mlx-openai-server`)
Launch a server for general OpenAPI workflows on port 8000:
```bash
source .venv/bin/activate
mlx-openai-server launch \
  --model-path mlx-community/Qwen3.5-9B-OptiQ-4bit \
  --model-type lm \
  --port 8000 --host 0.0.0.0 \
  --context-length 262144
```
Check status:
```bash
curl -s http://localhost:8000/v1/models
ps aux | grep mlx-openai-server | grep -v grep
```
Stop and restart with `pkill -f mlx-openai-server` followed by the same launch command.

### TurboQuant Optiq server (`mlx-openai-optiq-server`)
This FastAPI wrapper enables the TurboQuant KV cache for lower memory while keeping 256K context alive. You can run it locally from `src/mlx-openai-optiq-server` or via `docker-compose.yml`:
```bash
# Local python run
python src/mlx-openai-optiq-server/mlx-openai-optiq-server.py

# Or build the Docker service (exposes port 8080)
docker compose up -d --build
```
The service exposes the same `/v1/chat/completions` API plus `/v1/models`. The `docker-compose` definition already wires `HUGGINGFACE_MODEL`, `CONTEXT_LENGTH`, and the shared HuggingFace cache volume.

## Monitoring & maintenance

- Tail the logs or the background log file:
  ```bash
  tail -f /tmp/mlx-server.log
  docker logs -f mlx-optiq-server
  ```
- Monitor memory `ps aux | grep mlx-openai-server` or `pmap` for the Optiq service. 4-bit quantized models still eat ~20–25 GB with 256K context.
- Adjust `MAX_TOKENS`, `TEMPERATURE`, etc., via the `mlx-openai-server launch` flags or by crafting OpenAI-compatible requests.

## Troubleshooting

- **Command not found:** Activate `.venv` or call `./.venv/bin/mlx-openai-server` directly.
- **Port busy:** Use `lsof -i :8000`/`:8080` and free the port or pass `--port`/`PORT` overrides.
- **Slow downloads:** Ensure `HF_TOKEN` is set for authenticated HuggingFace access.
- **Cache issues:** Remove `~/.cache/huggingface/hub/models--mlx-community--Qwen3.5-9B-OptiQ-4bit` and restart if the download is corrupted.
- **OpenCode connection failures:** Match the server port and host inside `docs/IDEs/OpenCode.md` and restart OpenCode after any change.

## Resources

- `docs/Qwen.md` – Qwen model lineup, TurboQuant KV strategy, and inference presets.
- `docs/IDEs/QwenCode.md` / `OpenCode.md` / `CCR.md` – Hook your IDEs to the MLX endpoints.
- `docs/benchmark.md` – Throughput and memory comparisons for validation.
