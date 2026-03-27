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
# - quick: 8k context, 5 fast samples
# - long-context: 256k context, 5 large-payload samples (TurboQuant paper content)
just bench

# Equivalent direct command
uv run benchmark --dataset all
```

Results are written to `results/benchmark_<dataset>.jsonl` by default.
Each run also creates:
- `data/<experiment_name_with_global_params_timestamp>/summary.json`
- `data/<experiment_name_with_global_params_timestamp>/summary.md`
- `data/<experiment_name_with_global_params_timestamp>/<run_param>/{payload,response}.json`

The long-context dataset reads from `dataset/turboquant_2504_19874v1.md`.
If missing, build it once:
```bash
uv run dataset fetch-parse
```

## Context sweep experiment (latest)

Experiment date: `2026-03-27`

Setup:
- Runtime: `both` (sequential: `mlx` then `mlx-optiq`)
- Model: alias `optiq` (`mlx-community/Qwen3.5-9B-OptiQ-4bit`)
- Dataset source: `dataset/turboquant_2504_19874v1.md`
- Prompt payload: abstract-focused text extracted from dataset file
- Contexts: `8k,16k,32k,64k`
- Samples: `1` per context
- Output tokens: `64` per request (dataset context mode)

Command:
```bash
uv run benchmark \
  --runtime both \
  --model optiq \
  --context 8,16,32,64 \
  --samples 1 \
  --output benchmark_context_8_16_32_64_both.jsonl
```

Raw results:
- `results/benchmark_context_8_16_32_64_both.jsonl`

### Measurements

| Runtime | Context | Prompt tokens | Completion tokens | Total time (s) | E2E tok/s | Prompt tps | Gen tps | TTFT (s) | Peak RAM (GB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlx | 8k | 7659 | 64 | 35.05 | 1.83 | 230.90 | 43.18 | 33.53 | 8.30 |
| mlx | 16k | 15275 | 64 | 68.87 | 0.93 | 227.71 | 41.80 | 67.31 | 9.10 |
| mlx | 32k | 30495 | 64 | 139.06 | 0.46 | 222.41 | 38.17 | 137.36 | 10.51 |
| mlx | 64k | 60939 | 64 | 301.76 | 0.21 | 203.53 | 31.92 | 299.73 | 13.55 |
| mlx-optiq | 8k | 7659 | 64 | 27.61 | 2.32 | 300.60 | 37.72 | 25.89 | 9.71 |
| mlx-optiq | 16k | 15275 | 64 | 53.64 | 1.19 | 296.24 | 35.50 | 51.82 | 10.52 |
| mlx-optiq | 32k | 30495 | 64 | 116.29 | 0.55 | 267.50 | 32.58 | 114.30 | 11.93 |
| mlx-optiq | 64k | 60939 | 64 | 276.71 | 0.23 | 222.39 | 28.08 | 274.40 | 14.95 |

Notes:
- End-to-end `tok/s` drops with larger context because `TTFT` dominates total time.
- `Gen tps` (decode throughput) is much more stable than end-to-end throughput.
- RAM values above come from `memory_gb` in result rows.
- Benchmark rows keep only compact metrics; full request/response payloads are saved as artifacts.

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

# Run all datasets (quick + long-context) in one execution
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
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `tokens_per_second`
- `prompt_tps`
- `generation_tps`
- `ttft_sec`
- `server_total_time_sec`
- `memory_gb`
- `payload_path`
- `response_path`

Memory metrics:
- `memory_gb`: canonical peak RAM field used by benchmark output.
  For `mlx` / `mlx-optiq`, it is sourced from server-side peak memory stats.

Artifact layout:
- `data/<experiment_name_with_global_params_timestamp>/<run_param>/payload.json`
- `data/<experiment_name_with_global_params_timestamp>/<run_param>/response.json`

## Output schema

```json
{
  "success": true,
  "runtime": "mlx-optiq",
  "model": "mlx-community/Qwen3.5-9B-OptiQ-4bit",
  "timestamp": "2026-03-27T10:00:00+00:00",
  "total_time": 8.5,
  "prompt_tokens": 3000,
  "completion_tokens": 100,
  "total_tokens": 3100,
  "tokens_per_second": 11.76,
  "prompt_tps": 245.1,
  "generation_tps": 34.8,
  "ttft_sec": 7.9,
  "server_total_time_sec": 8.4,
  "memory_gb": 10.7,
  "payload_path": "data/benchmark_context-8k-16k-32k-64k_both_optiq_s1_mt100_20260327T161403Z/mlx-optiq__mlx-community-Qwen3.5-9B-OptiQ-4bit__context-8k-1/payload.json",
  "response_path": "data/benchmark_context-8k-16k-32k-64k_both_optiq_s1_mt100_20260327T161403Z/mlx-optiq__mlx-community-Qwen3.5-9B-OptiQ-4bit__context-8k-1/response.json"
}
```

## Quick analysis

```bash
# Inspect context sweep results
jq '.' results/benchmark_context_8_16_32_64_both.jsonl

# Average throughput by runtime
jq -s 'group_by(.runtime) | map({runtime: .[0].runtime, avg_tps: (map(.tokens_per_second) | add / length)})' \
  results/benchmark_context_8_16_32_64_both.jsonl

# Average peak RAM by runtime
jq -s 'group_by(.runtime) | map({runtime: .[0].runtime, avg_peak_ram_gb: (map(.memory_gb // 0) | add / length)})' \
  results/benchmark_context_8_16_32_64_both.jsonl
```

## Resources

- [MLX server runbook](../servers/MLX.md)
- [Ollama runbook](../servers/ollama.md)
- [Benchmark CLI](../../src/cli/benchmark.py)
