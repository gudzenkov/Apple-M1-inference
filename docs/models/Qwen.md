# Qwen Models & Inference

## Model lineup
| Model | Parameters | Quantization | Inference server support | Approx RAM (256K) |
|------------------------------|------------|--------------|------------------------------|--------------------|
| mlx-community/Qwen3.5-9B-OptiQ-4bit | 9B | 4-bit OptiQ | `mlx-openai-server` + `mlx-openai-optiq-server` | ~8–10 GB + 20–22 GB context |
| mlx-community/Qwen3.5-9B-4bit | 9B | 4-bit | `mlx-openai-server` | ~8–10 GB + 20–22 GB context |
| mlx-community/Qwen3.5-27B-4bit | 27B | 4-bit | `mlx-openai-server` | ~20–25 GB + 20–25 GB context |
| hertz-hwang/Qwen3.5-27B-OpenClaw-mlx-6.5bit | 27B | 6.5-bit | `mlx-openai-server` | ~30+ GB + 20–25 GB context |
| mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit | 27B | 4-bit | `mlx-openai-server` | ~20–25 GB + 20–25 GB context |
| mlx-community/Qwen3.5-35B-A3B-4bit | 35B total / 3B active | 4-bit | `vllm-metal` preferred, `mlx-openai-server` fallback | ~22–28 GB + context |
| mlx-community/Qwen3.6-35B-A3B-4bit-DWQ | 35B total / 3B active | 4-bit DWQ | `vllm-metal` preferred | ~21 GB model + KV cache |
| Qwen/Qwen3-Embedding-0.6B-GGUF | 0.6B | Q8_0/F16 GGUF | `llama-server --embedding` | ~1–2 GB |
| Qwen/Qwen3-Embedding-4B-GGUF | 4B | Q4_K_M/Q5/Q6/Q8/F16 GGUF | `llama-server --embedding` | ~3–7 GB |
| Qwen/Qwen3-Embedding-8B-GGUF | 8B | Q4_K_M/Q5/Q6/Q8/F16 GGUF | `llama-server --embedding` | ~6–12 GB |
| flovflo/turboquant-mlx-qwen35-kv | 35B (A3B) | 4-bit + TurboQuant KV cache | `tqkv-openai-server` | ~22–28 GB + ~4–6 GB compressed KV context |

Qwen3.5 and Qwen3.6 chat models support long context when the server is started with the matching context length. The embedding server should use an explicit practical context such as 32768 unless a larger retrieval workload is validated.

Hard truth: local Qwen3.6-35B-A3B does not provide hosted Qwen built-in tools. Local servers can expose tool-call JSON; the client still executes tools.

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

### Recommended split endpoint
- `vllm-metal` serves Qwen3.6-35B-A3B 4-bit DWQ chat, thinking, and tool-call parsing on `:8000`.
- `llama.cpp` serves Qwen3 embeddings on `:8001`; default to 0.6B when Qwen3.6-35B-A3B is also resident.
- Devcontainers call these through `http://host.docker.internal:8000/v1` and `http://host.docker.internal:8001/v1`.

This is the preferred setup for agent clients that expect OpenAI-compatible APIs.

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

- Keep the vLLM-Metal server running for agent workflows; use MLX/Optiq only for explicit benchmark experiments.
- For Qwen3.6-35B-A3B agent workflows, prefer `scripts/start-vllm-metal-qwen36.sh` and validate tool calls before long runs.
- For embeddings, prefer `scripts/start-qwen3-embedding-llama.cpp.sh` and probe `/v1/embeddings` directly.
- Use `curl http://localhost:8000/v1/models` for the recommended vLLM-Metal endpoint, or `http://localhost:8080/v1/models` only when running the legacy Optiq server.
- The TurboQuant server uses a fixed seed per layer to keep compression deterministic; avoid restarting the process too frequently if you need consistent caches.

## Resources

- `docs/servers/MLX.md` – server launch commands, monitoring, troubleshooting.
- `docs/servers/VLLM-Metal.md` – recommended Qwen3.6-35B-A3B chat/tools/thinking server.
- `docs/servers/llama.cpp.md` – Qwen3 embedding server.
- `docs/benchmarks/performance.md` – benchmark workflow for MLX and MLX-Optiq.
