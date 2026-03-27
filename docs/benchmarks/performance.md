# Performance Benchmark: MLX vs MLX-Optiq

Benchmark workflow for comparing the two local MLX inference servers:
- `mlx-openai-server` (standard, port `8000`)
- `mlx-openai-optiq-server` (TurboQuant, port `8080`)

The benchmark runner manages server lifecycle to avoid OOM from running both MLX servers at once.

## Quick start

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

# Default benchmark run from Justfile:
# - quick: 32k context, 5 fast samples
# - long: 256k context, 5 large-payload samples (TurboQuant paper content)
just bench

# Equivalent direct command
uv run benchmark --dataset all
```

Results are written to `results/benchmark_<dataset>.jsonl` by default.

The long dataset reads from `dataset/turboquant_2504_19874v1.md`.
If missing, build it once:
```bash
uv run dataset fetch-parse
```

## Runtime modes

- `--runtime mlx`: benchmark only `mlx-openai-server` on `:8000`.
- `--runtime mlx-optiq`: benchmark only `mlx-openai-optiq-server` on `:8080`.
- `--runtime both`: benchmark both MLX servers sequentially.
- `--runtime ollama`: benchmark external Ollama server on `:11434` (optional).
- `--runtime all`: benchmark MLX, MLX-Optiq, and Ollama.

For managed MLX runs (`mlx`, `mlx-optiq`, `both`, `all`), the script:
1. Stops existing MLX server processes on ports `8000` and `8080`.
2. Starts the selected server with the selected model.
3. Runs benchmark prompts.
4. Stops the server before moving to the next runtime/model.

## Usage examples

```bash
# MLX only
uv run benchmark --runtime mlx --model mlx-community/Qwen3.5-9B-OptiQ-4bit

# MLX-Optiq only
uv run benchmark --runtime mlx-optiq --model mlx-community/Qwen3.5-9B-OptiQ-4bit

# Compare both servers on all configured prompts
uv run benchmark --runtime both --model mlx-community/Qwen3.5-9B-OptiQ-4bit

# Run quick dataset only
uv run benchmark --dataset quick --runtime both

# Run long-context dataset only
uv run benchmark --dataset long --runtime both

# Run all datasets (quick + long) in one execution
uv run benchmark --dataset all --runtime both

# Custom prompt and output file (stored under results/)
uv run benchmark \
  --runtime both \
  --prompt "Explain Rust ownership in one paragraph." \
  --max-tokens 200 \
  --output benchmark_mlx_vs_optiq_qwen9b.jsonl
```

## Metrics recorded

Each JSONL line includes:
- `total_time`
- `completion_tokens`
- `tokens_per_second`
- `memory_gb`

`memory_gb` is the peak resident memory observed during the request window (prefill + generation), not post-run memory.

## Output schema

```json
{
  "success": true,
  "runtime": "mlx-optiq",
  "model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
  "prompt": "Explain async/await in JavaScript",
  "timestamp": "2026-03-27T10:00:00+00:00",
  "total_time": 8.5,
  "prompt_tokens": 0,
  "completion_tokens": 100,
  "total_tokens": 100,
  "tokens_per_second": 11.76,
  "memory_gb": 9.2,
  "response_preview": "async/await is ..."
}
```

## Quick analysis

```bash
# Inspect all results
jq '.' results/benchmark_quick.jsonl
jq '.' results/benchmark_long.jsonl

# Average throughput by runtime
jq -s 'group_by(.runtime) | map({runtime: .[0].runtime, avg_tps: (map(.tokens_per_second) | add / length)})' \
  results/benchmark_quick.jsonl

# Average peak memory by runtime
jq -s 'group_by(.runtime) | map({runtime: .[0].runtime, avg_mem_gb: (map(.memory_gb // 0) | add / length)})' \
  results/benchmark_long.jsonl
```

## Resources

- [MLX server runbook](../servers/MLX.md)
- [Ollama runbook](../servers/ollama.md)
- [Benchmark CLI](../../src/cli/benchmark.py)
