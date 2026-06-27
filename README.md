# Apple M1 LLM inference Setup

Local-first guides for running large language models on Apple Silicon with Metal/MLX,
plus OpenAI-compatible endpoints for devcontainers.

## Engines at a Glance

| Engine | Focus | RAM | Notes |
|--------|-------|-----|-------|
| **vLLM-Metal** | Primary Qwen3.6-35B-A3B chat/tools/thinking server | 48GB+ | Best fit for Qwen3.6 MoE on Mac when you need OpenAI-compatible tool calls and reasoning output |
| **llama.cpp** | Qwen3 embeddings and GGUF fallback | 2GB+ | Mature `/v1/embeddings`; use 0.6B on port 8001 when 35B chat also runs |
| **LM Studio** | Local OpenAI-compatible chat or embeddings server | model-dependent | Easiest manual local server; useful when you already loaded the model in LM Studio |
| **MLX** | Existing MLX/OptiQ experiments | 24GB+ | Useful for custom MLX serving and long-context experiments; weaker native tool-call surface than vLLM |

## Documentation

### Inference Server Guides
- 📕 [`docs/servers/VLLM-Metal.md`](docs/servers/VLLM-Metal.md) – Recommended Qwen3.6-35B-A3B 4-bit DWQ chat/tools/thinking server on Apple Silicon.
- 📗 [`docs/servers/llama.cpp.md`](docs/servers/llama.cpp.md) – Qwen3 embedding server and GGUF fallback commands.
- 📔 [`docs/servers/LM-Studio.md`](docs/servers/LM-Studio.md) – Local LM Studio OpenAI-compatible server option for chat or embeddings.
- 📘 [`docs/servers/MLX.md`](docs/servers/MLX.md) – MLX setup, server workflows, and troubleshooting.
- 📙 [`docs/servers/devcontainer.md`](docs/servers/devcontainer.md) – Run Qwen on macOS host and connect it from Orb/Docker Desktop devcontainers.

### Usage Guides
- 🔧 [`docs/models/Qwen.md`](docs/models/Qwen.md) – Qwen model lineup, TurboQuant cache, and inference presets.
- 🔧 [`docs/clients/QwenCode.md`](docs/clients/QwenCode.md) – Point QwenCode at the local vLLM-Metal endpoint.
- 🔧 [`docs/clients/OpenCode.md`](docs/clients/OpenCode.md) – Configure OpenCode providers for the local runtimes.
- 🔧 [`docs/clients/CCR.md`](docs/clients/CCR.md) – Claude Code Router routing notes for the local endpoints.

### Performance
- 📊 [`docs/benchmarks/performance.md`](docs/benchmarks/performance.md) – Benchmark workflow for MLX vs MLX-Optiq.

## Getting started

1. **For agentic Qwen3.6-35B-A3B work, start vLLM-Metal on macOS host.** It exposes OpenAI-compatible chat on `http://127.0.0.1:8000/v1` with Qwen reasoning and tool-call parsers enabled.
2. **For Qwen3 embeddings, start llama.cpp separately.** The default is `Qwen3-Embedding-0.6B-Q8_0`, exposed as OpenAI-compatible embeddings on `http://127.0.0.1:8001/v1`.
3. **From a devcontainer, use `host.docker.internal`.** The matching URLs are `http://host.docker.internal:8000/v1` and `http://host.docker.internal:8001/v1`.

## Local model pair

Recommended local setup:

- Chat LLM: `qwen3.6-35b-a3b` served by vLLM-Metal on `http://127.0.0.1:8000/v1`
- Embeddings: `qwen3-embedding` served by llama.cpp on `http://127.0.0.1:8001/v1`
- Alternative local server: LM Studio on `http://127.0.0.1:1234/v1` for either chat or embeddings, depending on the loaded model

## Tooling & layout

- `pyproject.toml` defines shared Python dependencies for both MLX servers; use `uv sync` once with a single `.venv`.
- Use `uv run benchmark`, `uv run dataset`, and `uv run mlx-cli` for local tooling commands.
- Use `uv run mlx-openai-server` and `uv run mlx-openai-optiq-server` to run or manage servers.
- Use `scripts/patch-vllm-metal-qwen36-text-backbone.sh`, `scripts/start-vllm-metal-qwen36.sh`, `scripts/start-qwen3-embedding-llama.cpp.sh`, and `scripts/probe-openai-local.sh` for the recommended host endpoints.
- `src/servers/` contains the server entrypoints.
- MLX inference in this repo requires native macOS runtime access.
- `scripts/` contains shell utilities for local setup tasks.
