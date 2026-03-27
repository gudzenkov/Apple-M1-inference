# Ollama Runbook

Ollama bundles an OpenAI-compatible service with automatic Metal acceleration, so it is the quickest path to tool-enabled Qwen models on Apple Silicon.
Docker workflows are not supported in this repo; use native macOS installation (`brew` + local service) for consistent behavior.

## Installation & service

```bash
brew install ollama
brew services start ollama
```
You can also run `ollama serve` in a shell if you prefer manual control. Ollama listens on `localhost:11434` by default; set `OLLAMA_HOST=0.0.0.0:11434` if you need remote access.

## Model management

- Pull the curated models:
  ```bash
  ollama pull qwen3.5:9b
  ollama pull SimonPu/Qwen3-Coder:30B-Instruct_Q4_K_XL
  ollama pull sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M
  ollama pull ukjin/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill
  ```
- List, inspect, or delete with `ollama list`, `ollama show <model>`, `ollama rm <model>`.
- Use `scripts/ollama-build-models.sh` to create 256k-context Modelfiles and `ollama create` wrappers for every configured Ollama model.
- The script reads model definitions from `configs/models.yaml` and selects entries with `runtime: ollama`.

## API & context length

Ollama exposes `/v1/chat/completions` with OpenAI semantics. Control runtime context with `OLLAMA_CONTEXT_LENGTH` or via Modelfiles:

```bash
OLLAMA_CONTEXT_LENGTH=262144 ollama run qwen3.5:9b
```
You can also send `num_ctx` in the JSON payload when calling `/v1/chat/completions` or specify `PARAMETER num_ctx` in a Modelfile.

## Server management

- Check the service:
  ```bash
  brew services list | grep ollama
  ps aux | grep ollama | grep -v grep
  curl http://localhost:11434/api/tags
  ```
- Restart with `brew services restart ollama`.
- If running manually in a shell, stop with `Ctrl+C` and then run `ollama serve` again.
- Control concurrency and directory locations with environment variables:
  ```bash
  export OLLAMA_NUM_PARALLEL=1
  export OLLAMA_MODELS=~/ollama-models
  ```

## Performance tips

- Keep `OLLAMA_NUM_PARALLEL=1` when evaluating latency-sensitive workflows.
- Use `OLLAMA_CONTEXT_LENGTH=262144` for agentic tasks; drop to `32768` for quick experiments.
- Monitor GPU/CPU usage with `sudo powermetrics --samplers gpu_power -i 1000 -n 1` and `ollama ps`.

## Troubleshooting

- **Tools not supported:** Ensure you are using the tool-enabled models pulled above; re-pull them if the checksum changed.
- **Port or host mismatch:** Verify `OLLAMA_HOST` matches the base URL used by your client.
- **Model download corrupted:** Remove the model directory under `~/.ollama/models/<model>` and rerun `ollama pull`.
- **Out of memory:** Reduce quantization (Q4 vs Q8), lower `num_ctx`, or shut down other apps.

## Resources

- `scripts/ollama-build-models.sh` – create 256k Modelfiles for configured Ollama models (`runtime: ollama` in `configs/models.yaml`).
- `docs/benchmarks/performance.md` – Benchmark workflow (MLX vs MLX-Optiq, with optional Ollama mode).
