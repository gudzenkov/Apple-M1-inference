# Performance Benchmark: MLX vs MLX-Optiq

Benchmark workflow for comparing the two local MLX inference servers:
- `mlx-openai-server` (standard, port `8000`)
- `mlx-openai-optiq-server` (TurboQuant, port `8080`)

The benchmark runner manages server lifecycle to avoid OOM from running both MLX servers at once.
It now uses 2 dataset modes:
- `short` (fixed `8k`, needle-in-haystack retrieval)
- `long` (variable context via `--context`, default `64k`, needle-in-haystack retrieval)

## Recall

64k needle-in-haystack run (`samples=20`, `use_prompt_cache=true`, date `2026-03-28`)
64k needle-in-haystack run (`samples=3`, `stream=on`, `cache=auto`, `reasoning=off`, date `2026-04-16`):

| Runtime | Model | Context | Prompt tps | Gen tps | Prefill (s) | TTFT (s) | Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlx | Qwen3.5-9B-4bit | 64k | 78.683 | 25.722 | - | 0.868 | 12.94 | 0.967 | 0.800 | 0.175 |
| mlx-optiq | Qwen3.5-9B-OptiQ-4bit | 64k | 93.727 | 26.167 | - | 0.689 | 15.35 | 1.000 | 1.000 | 0.000 |
| delta (mlx vs optiq) | - | - | +19.12% | +1.73% | - | -20.62% | +18.62% | +3.41% | +25.00% | - |
| ollama (`cache=request`) | qwen3.5:9b | 64k | 177.292 | 8.012 | 362.009 | 17.333 | 16.73 | 1.000 | 1.000 | 0.000 |
| mlx (`cache=prefill`) | Qwen3.5-9B-4bit | 64k | 96.536 | 29.255 | 317.255 | 0.737 | 12.94 | 1.000 | 1.000 | 0.000 |
| delta (mlx vs ollama) | - | - | -45.55% | +265.14% | -12.36% | -95.75% | -22.64% | +0.00% | +0.00% | - |

Source summaries:
- `results/mlx-turboquant-s20-mt100-pc1/20260328T110231Z/mlx-qwen3.5-9b-q4-64k-s20.md`
- `results/mlx-optiq-turboquant-s20-mt100-pc1/20260328T111033Z/mlx-optiq-qwen3.5-9b-optiq-q4-64k-s20.md`
- `results/ollama-turboquant-s3-mt32-cache-request/20260330T173436Z/ollama-qwen3.5-9b-64k-s3.json`
- `results/mlx-turboquant-s3-mt100-cache-prefill/20260416T103835Z/mlx-mlx-community-Qwen3.5-9B-4bit-64k-s3.json`

## Performance

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

## Quick start

```bash
cd "$(git rev-parse --show-toplevel)"
source .venv/bin/activate

# Default benchmark run from Justfile:
# - short: 8k context, retrieval-style samples
# - long: 64k context (or --context override), retrieval-style samples
just bench

# Equivalent direct command
uv run benchmark --dataset all
```

Results are written under:
- `results/<experiment-group>/<utc-timestamp>/benchmark_<model>_<context>_s<samples>.jsonl`

Each run also creates:
- `data/benchmark/<experiment-group>/<utc-timestamp>/<run_param>/{payload,response}.json`
- `results/<experiment-group>/<utc-timestamp>/<run_param>.json`
- `results/<experiment-group>/<utc-timestamp>/<run_param>.md`

The long-context dataset reads from `dataset/turboquant_2504_19874v1.md`.
If missing, build it once:
```bash
uv run dataset fetch-parse
```

## Context sweep experiment (latest)

Experiment date: `2026-03-27`

Setup:
- Hardware: M1 Mac (`64GB` LPDDR5 unified memory, `24` GPU cores)
- Runtime: `mlx` and `mlx-optiq` (run as separate commands)
- Model: alias `mlx-optiq-9b` (`mlx-community/Qwen3.5-9B-OptiQ-4bit`)
- Dataset source: `dataset/turboquant_2504_19874v1.md`
- Prompt payload: needle-in-haystack retrieval prompt
- Contexts: `8k,16k,32k,64k`
- Samples: `3` per context (minimum supported)
- Output tokens: `64` per request (long mode with context override)

Command:
```bash
uv run benchmark \
  --runtime mlx \
  --model mlx-optiq-9b \
  --dataset long \
  --context 8,16,32,64 \
  --samples 3 \
  --output benchmark_context_8_16_32_64_mlx.jsonl
```

Raw results:
- `results/benchmark_context_8_16_32_64_mlx.jsonl`

## Runtime modes

- `--runtime auto` (default): resolve runtime from `--model` using `configs/models.yaml`.
- `--runtime mlx`: benchmark only `mlx-openai-server` on `:8000`.
- `--runtime mlx-optiq`: benchmark only `mlx-openai-optiq-server` on `:8080`.
- `--runtime ollama`: benchmark external Ollama server on `:11434` (optional).

For managed MLX runs (`mlx`, `mlx-optiq`), the script:
1. Stops existing MLX server processes on ports `8000` and `8080`.
2. Starts the selected server with the selected model.
3. Runs benchmark prompts.
4. Stops the server before moving to the next runtime/model.

## Usage examples

```bash
# Default runtime from model config (recommended)
uv run benchmark --model mlx-qwen-9b
uv run benchmark --model qwen3.5:9b

# MLX only
uv run benchmark --runtime mlx --model mlx-qwen-9b

# MLX-Optiq only
uv run benchmark --runtime mlx-optiq --model mlx-optiq-9b

# Compare both servers with explicit separate runs
uv run benchmark --runtime mlx --model mlx-qwen-9b
uv run benchmark --runtime mlx-optiq --model mlx-optiq-9b

# Run short dataset only (8k)
uv run benchmark --dataset short --runtime mlx --samples 3

# Run long dataset only (default 64k)
uv run benchmark --dataset long --runtime mlx --samples 3

# Run long dataset context sweep
uv run benchmark --dataset long --context 8,64 --runtime mlx --samples 3

# Run long 64k with 20 needles/samples and prompt caching
uv run benchmark \
  --dataset long \
  --context 64 \
  --runtime mlx \
  --samples 20 \
  --use-prompt-cache \
  --output benchmark_long_64k_s20_cache.jsonl

# Run all datasets (short + long)
uv run benchmark --dataset all --runtime mlx --samples 3

# Custom prompt and output file (stored under results/)
uv run benchmark \
  --runtime mlx \
  --prompt "Explain Rust ownership in one paragraph." \
  --max-tokens 200 \
  --output benchmark_mlx_vs_optiq_qwen9b.jsonl
```

## Metrics recorded

Each JSONL line includes:
- `phase` (`cache-prime` for run0, `benchmark` for run1..N)
- `total_time`
- `prompt_tokens`
- `completion_tokens`
- `total_tokens`
- `tokens_per_second`
- `prompt_tps`
- `generation_tps`
- `prefill_sec`
- `ttft_sec`
- `server_total_time_sec`
- `memory_gb`
- `retrieval_expected`
- `retrieval_predicted`
- `retrieval_score_float`
- `retrieval_exact`
- `used_prompt_cache`
- `payload_path`
- `response_path`

Summary reports also include:
- runtime-level and runtime+context averages for retrieval score and exact rate
- `avg_prefill_sec` (from run0 cache-prime when available)
- `retrieval_exact_ci95_half_width` (95% CI half-width for exact-match rate)

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
  "retrieval_expected": "NIAH-LONG-64K-S01-001234",
  "retrieval_predicted": "NIAH-LONG-64K-S01-001234",
  "retrieval_score_float": 1.0,
  "retrieval_exact": true,
  "used_prompt_cache": true,
  "payload_path": "data/benchmark/mlx-paper-s3-mt100/20260327T161403Z/mlx-optiq-qwen3.5-9b-optiq-q4-64k-s1/payload.json",
  "response_path": "data/benchmark/mlx-paper-s3-mt100/20260327T161403Z/mlx-optiq-qwen3.5-9b-optiq-q4-64k-s1/response.json"
}
```

## Quick analysis

```bash
# Inspect context sweep results
jq '.' results/benchmark_context_8_16_32_64_mlx.jsonl

# Average throughput by runtime
jq -s 'group_by(.runtime) | map({runtime: .[0].runtime, avg_tps: (map(.tokens_per_second) | add / length)})' \
  results/benchmark_context_8_16_32_64_mlx.jsonl

# Average peak RAM by runtime
jq -s 'group_by(.runtime) | map({runtime: .[0].runtime, avg_peak_ram_gb: (map(.memory_gb // 0) | add / length)})' \
  results/benchmark_context_8_16_32_64_mlx.jsonl
```

## Resources

- [MLX server runbook](../servers/MLX.md)
- [Ollama runbook](../servers/ollama.md)
- [Benchmark CLI](../../src/cli/benchmark.py)
