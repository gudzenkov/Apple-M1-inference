# Qwen Models & Inference

## Model lineup
| Model | Parameters | Quantization | Approx RAM (256K) |
|------------------------------|------------|--------------|--------------------|
| mlx-community/Qwen3.5-9B-OptiQ-4bit | 9B | 4-bit OptiQ | ~8–10 GB + 20–22 GB context |
| mlx-community/Qwen3.5-27B-4bit | 27B | 4-bit OptiQ | ~20–25 GB + 20–25 GB context |

All models support 262144 tokens (256K) when the server (MLX or Optiq) is started with the matching context length. See `docs/MLX.md` for runbook-level server commands and `docs/IDEs/QwenCode.md` for connecting QwenCode once the endpoint is live.

## TurboQuant KV cache compression

OptiQ models can leverage `mlx-optiq` to compress the KV cache to roughly 1/5 of the size while maintaining long-context accuracy. This keeps 256K context affordable on 27B models.

- Install requirements:
  ```bash
  uv pip install "mlx-lm>=0.30.7" mlx-optiq
  ```
- Use the custom FastAPI wrapper in `src/mlx-openai-optiq-server` (or the bundled Docker image) to initialize the cache with `TurboQuantKVCache` per attention layer.
- When you need the TurboQuant benefits, prefer the `/v1/chat/completions` endpoint on port 8080 (the Optiq server) and let the wrapper manage `seed+bits` per layer.

## Qwen+MLX inference flows

### General MLX server (`mlx-openai-server`)
- Default server for most experiments.
- Launch with `--context-length 262144` and `--max-tokens`/`--temperature` overrides as needed.
- Good for workloads that simply need OpenAI-compatible chat completions without TurboQuant caching.

### TurboQuant-aware Optiq server (`mlx-openai-optiq-server`)
- Local Python script or `docker compose up -d` will start an endpoint on port 8080 that keeps 256K context in RAM while applying the TurboQuant cache.
- Every request reuses the same quantized cache object, so warm-up is fast after the first prompt and memory stays closer to the 9B footprint.
- The endpoint mirrors `/v1/chat/completions` and `/v1/models`, so any OpenAI client works.

Switch between the two by changing the base URL in your IDE or tooling. `docs/IDEs/QwenCode.md` explains how to map both servers to QwenCode, while `docs/IDEs/OpenCode.md` covers the OpenCode providers.

## Inference parameters

Pick the right sampling profile for the task:
- **Thinking (general):** `temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `repetition_penalty=1.0`.
- **Precise coding (WebDev, CLI):** `temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=0.0`, `repetition_penalty=1.0`.
- **Instruct/general tasks:** `temperature=0.7`, `top_p=0.8`, `top_k=20`, `presence_penalty=1.5`, `repetition_penalty=1.0`.
- **Reasoning-heavy instruct:** `temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `repetition_penalty=1.0`.

Adjust `max_tokens` per prompt (a default of 1024 works for most completions) and include `stream=true` if your client supports streaming.

## Tips

- Keep the MLX/Optiq server running during QwenCode sessions and tail `/tmp/mlx-server.log` or `docker logs mlx-optiq-server` for request visibility.
- Use `curl http://localhost:8000/v1/models` (MLX server) or `http://localhost:8080/v1/models` (Optiq server) to sanity-check readiness before opening an IDE.
- The TurboQuant server uses a fixed seed per layer to keep compression deterministic; avoid restarting the process too frequently if you need consistent caches.

## Resources

- `docs/MLX.md` – server launch commands, monitoring, troubleshooting.
- `docs/IDEs/QwenCode.md` / `OpenCode.md` / `CCR.md` – connect IDEs to the local endpoints.
- `docs/benchmark.md` – performance metrics for MLX, Optiq, and Ollama runtimes.
