# Qwen Models & Inference

## Model lineup
| Model | Parameters | Quantization | Inference server support | Approx RAM (256K) |
|------------------------------|------------|--------------|------------------------------|--------------------|
| mlx-community/Qwen3.5-9B-OptiQ-4bit | 9B | 4-bit OptiQ | `mlx-openai-server` + `mlx-openai-optiq-server` | ~8–10 GB + 20–22 GB context |
| mlx-community/Qwen3.5-9B-4bit | 9B | 4-bit | `mlx-openai-server` | ~8–10 GB + 20–22 GB context |
| mlx-community/Qwen3.5-27B-4bit | 27B | 4-bit | `mlx-openai-server` | ~20–25 GB + 20–25 GB context |
| mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 27B | 4-bit | `mlx-openai-server` | ~20–25 GB + 20–25 GB context |
| mlx-community/Qwen3.5-35B-A3B-4bit | 35B (A3B) | 4-bit | `mlx-openai-server` | ~22–28 GB + 24–30 GB context |
| flovflo/turboquant-mlx-qwen35-kv | 35B (A3B) | 4-bit + TurboQuant KV cache | `tqkv-openai-server` | ~22–28 GB + ~4–6 GB compressed KV context |
| hertz-hwang/Qwen3.5-27B-OpenClaw-mlx-6.5bit | 27B | 6.5-bit | `mlx-openai-server` | ~30+ GB + 20–25 GB context |

All models support 262144 tokens (256K) when the server (MLX or Optiq) is started with the matching context length.

## TurboQuant KV cache compression

OptiQ models can leverage `mlx-optiq` to compress the KV cache to roughly 1/5 of the size while maintaining long-context accuracy. This keeps 256K context affordable on 27B models.

- Install requirements:
  ```bash
  uv sync
  ```
- Use `uv run mlx-openai-optiq-server` to initialize `TurboQuantKVCache` per attention layer.
- When you need the TurboQuant benefits, prefer the `/v1/chat/completions` endpoint on port 8080 (the Optiq server) and let the server manage `seed+bits` per layer.
- MLX inference here requires native macOS runtime access.

## Qwen+MLX inference flows

### General MLX server (`mlx-openai-server`)
- Default server for most experiments.
- Launch with `--context-length 262144` and `--max-tokens`/`--temperature` overrides as needed.
- Good for workloads that simply need OpenAI-compatible chat completions without TurboQuant caching.

### TurboQuant-aware Optiq server (`mlx-openai-optiq-server`)
- `uv run mlx-openai-optiq-server serve` starts an endpoint on port 8080 that keeps 256K context in RAM while applying the TurboQuant cache.
- Every request reuses the same quantized cache object, so warm-up is fast after the first prompt and memory stays closer to the 9B footprint.
- The endpoint mirrors `/v1/chat/completions` and `/v1/models`, so any OpenAI client works.

Switch between the two by changing the base URL in your OpenAI-compatible client.

## Inference parameters

Pick the right sampling profile for the task:
- **Thinking (general):** `temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `repetition_penalty=1.0`.
- **Precise coding (WebDev, CLI):** `temperature=0.6`, `top_p=0.95`, `top_k=20`, `presence_penalty=0.0`, `repetition_penalty=1.0`.
- **Instruct/general tasks:** `temperature=0.7`, `top_p=0.8`, `top_k=20`, `presence_penalty=1.5`, `repetition_penalty=1.0`.
- **Reasoning-heavy instruct:** `temperature=1.0`, `top_p=0.95`, `top_k=20`, `presence_penalty=1.5`, `repetition_penalty=1.0`.

Adjust `max_tokens` per prompt (a default of 1024 works for most completions) and include `stream=true` if your client supports streaming.

## Tips

- Keep the MLX/Optiq server running and tail `logs/mlx-server.log` or `logs/mlx-optiq-server.log` for request visibility.
- Use `curl http://localhost:8000/v1/models` (MLX server) or `http://localhost:8080/v1/models` (Optiq server) to sanity-check readiness before connecting any client.
- The TurboQuant server uses a fixed seed per layer to keep compression deterministic; avoid restarting the process too frequently if you need consistent caches.

## Resources

- `docs/servers/MLX.md` – server launch commands, monitoring, troubleshooting.
- `docs/benchmarks/performance.md` – benchmark workflow for MLX and MLX-Optiq (optional Ollama mode).
