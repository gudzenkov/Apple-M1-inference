# OptiQ vs TurboQuant: Implementation Differences and Trade-offs

This note is scoped to the current setup (`mlx-openai-optiq-server`) and the installed `mlx-optiq` package in `.venv`.

## Benefits of OptiQ in this setup

- Prefill-side attention path can improve latency in long-context runs, so you may see better `prompt_tps` and `TTFT` even when decode throughput (`generation_tps`) is lower.
- It uses TurboQuant-style rotated attention, which can preserve long-context quality better than naive affine KV quantization. This is a quality claim, not a guaranteed speed win.
- It gives a second runtime path (`mlx-openai-optiq-server`) for A/B experiments against standard MLX.

## Hard truth: this is not a full paper-faithful compressed KV cache implementation

### Why

- Cache storage in the serving path is centroid-space float16 tensors:
  - `optiq/core/turbo_kv_cache.py:61`
  - `optiq/core/turbo_kv_cache.py:62`
- "Quantized storage" fields exist but are marked for future use and are not used by the current serving path:
  - `optiq/core/turbo_kv_cache.py:64`
  - `optiq/core/turbo_kv_cache.py:65`
  - `optiq/core/turbo_kv_cache.py:66`
  - `optiq/core/turbo_kv_cache.py:67`
  - `optiq/core/turbo_kv_cache.py:68`
- Server-side cache construction is fixed to `bits=4`, and no QJL path is enabled in this repo path:
  - `src/servers/mlx_openai_optiq_server.py:34`
  - `src/servers/mlx_openai_optiq_server.py:36`
  - `src/servers/mlx_openai_optiq_server.py:48`
- The OptiQ cache code explicitly states MSE quantizers are used:
  - `optiq/core/turbo_kv_cache.py:54`

## Practical trade-offs for benchmarking

- Expect possible wins in prefill-heavy metrics (`prompt_tps`, sometimes `TTFT`).
- Do not assume decode throughput (`generation_tps`) will improve.
- Treat this as a pragmatic rotated-attention runtime variant, not a strict reproduction of paper/blog KV compression claims.

## Package provenance

- The installed package metadata lists `Author: Thin Signal`, so this should be treated as third-party software, not an official Google release:
  - `mlx_optiq-0.0.2.dist-info/METADATA:5`

## External references

- [TurboQuant Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant paper (arXiv:2504.19874)](https://arxiv.org/pdf/2504.19874)
