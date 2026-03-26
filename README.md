# LocalFirst LLM Setup

Local-first guides for running large language models on Apple Silicon, with separate provider runbooks for MLX and Ollama plus model/IDE references.

## Engines at a Glance

| Engine | Focus | RAM | Notes |
|--------|-------|-----|-------|
| **MLX** | Highest-quality Qwen architectures with TurboQuant | 24GB+ | Best quality and compatibility with `mlx-openai-server` / Optiq stack |
| **Ollama** | Easy installation and tool-enabled models | 8–16GB | Reliable service with built-in Metal support |

## Documentation

### Provider Guides
- 📘 [`docs/MLX.md`](docs/MLX.md) – MLX setup, server workflows, and troubleshooting.
- 📗 [`docs/ollama.md`](docs/ollama.md) – Ollama installation, model management, and API notes.

### Usage Guides
- 🔧 [`docs/Qwen.md`](docs/Qwen.md) – Qwen model lineup, TurboQuant cache, and inference presets.
- 🔧 [`docs/IDEs/QwenCode.md`](docs/IDEs/QwenCode.md) – Point QwenCode at the Optiq/MLX endpoints.
- 🔧 [`docs/IDEs/OpenCode.md`](docs/IDEs/OpenCode.md) – Configure OpenCode providers for the local runtimes.
- 🔧 [`docs/IDEs/CCR.md`](docs/IDEs/CCR.md) – Claude Code Router routing notes for the local endpoints.

### Performance
- 📊 [`docs/benchmark.md`](docs/benchmark.md) – Benchmark comparisons and tooling for MLX vs Ollama.

## Getting started

1. **Choose a runtime.** Use MLX when you need the latest Qwen models and long-context TurboQuant cache, or Ollama for quick setup and community builds. Both guides start with system requirements (Apple Silicon, macOS 13+, 16+ GB RAM).
2. **Follow the provider runbook.** Each will walk you through installing dependencies, configuring models, and operating the OpenAI-compatible servers (`mlx-openai-server` / `mlx-openai-optiq-server` on 8000/8080, Ollama on 11434).
3. **Connect your IDE.** Use the IDE runbooks above to point QwenCode, OpenCode, and Claude Code Router at the local endpoints once the server is running.

## Tooling & layout

- `docker-compose.yml` builds the Optiq-aware FastAPI wrapper (`src/mlx-openai-optiq-server`) for TurboQuant caching and exposes it on port 8080.
- `scripts/` contains helpers such as `benchmark.py` and `setup-256k-context.sh` for Ollama.
