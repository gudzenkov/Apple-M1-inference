# Apple M1 LLM inference Setup

Local-first guides for running large language models on Apple Silicon, with separate provider runbooks for MLX and Ollama plus model/client references.

## Engines at a Glance

| Engine | Focus | RAM | Notes |
|--------|-------|-----|-------|
| **MLX** | Highest-quality Qwen architectures with TurboQuant | 24GB+ | Best quality and compatibility with `mlx-openai-server` / Optiq stack |
| **Ollama** | Easy installation and tool-enabled models | 8–16GB | Reliable service with built-in Metal support |

## Documentation

### Inference Server Guides
- 📘 [`docs/servers/MLX.md`](docs/servers/MLX.md) – MLX setup, server workflows, and troubleshooting.
- 📗 [`docs/servers/ollama.md`](docs/servers/ollama.md) – Ollama installation, model management, and API notes.
- 📙 [`docs/servers/devcontainer.md`](docs/servers/devcontainer.md) – Run Qwen on macOS host and connect it from Orb/Docker Desktop devcontainers.

### Usage Guides
- 🔧 [`docs/models/Qwen.md`](docs/models/Qwen.md) – Qwen model lineup, TurboQuant cache, and inference presets.
- 🔧 [`docs/clients/QwenCode.md`](docs/clients/QwenCode.md) – Point QwenCode at the Optiq/MLX endpoints.
- 🔧 [`docs/clients/OpenCode.md`](docs/clients/OpenCode.md) – Configure OpenCode providers for the local runtimes.
- 🔧 [`docs/clients/CCR.md`](docs/clients/CCR.md) – Claude Code Router routing notes for the local endpoints.

### Performance
- 📊 [`docs/benchmarks/performance.md`](docs/benchmarks/performance.md) – Benchmark workflow for MLX vs MLX-Optiq (with optional Ollama mode).

## Getting started

1. **Choose a runtime.** Use MLX when you need the latest Qwen models and long-context TurboQuant cache, or Ollama for quick setup and community builds. Both guides start with system requirements (Apple Silicon, macOS 13+, 16+ GB RAM).
2. **Follow the provider runbook.** Each will walk you through installing dependencies, configuring models, and operating the OpenAI-compatible servers (`mlx-openai-server` / `mlx-openai-optiq-server` on 8000/8080, Ollama on 11434).
3. **Connect your client.** Use the client runbooks above to point QwenCode, OpenCode, and Claude Code Router at the local endpoints once the server is running.

## Tooling & layout

- `pyproject.toml` defines shared Python dependencies for both MLX servers; use `uv sync` once with a single `.venv`.
- Use `uv run benchmark`, `uv run dataset`, and `uv run mlx-cli` for local tooling commands.
- Use `uv run mlx-openai-server` and `uv run mlx-openai-optiq-server` to run or manage servers.
- `src/servers/` contains the server entrypoints.
- MLX inference in this repo requires native macOS runtime access.
- `scripts/` contains shell utilities (for example Ollama model build helpers).
