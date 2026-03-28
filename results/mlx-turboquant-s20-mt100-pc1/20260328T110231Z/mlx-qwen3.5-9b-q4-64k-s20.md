# Benchmark Summary

Generated: `2026-03-28T11:10:26.391129+00:00`

## Params

- `dataset`: `long`
- `contexts_k`: `[64]`
- `runtime`: `mlx`
- `runtimes_resolved`: `['mlx']`
- `model`: `qwen3.5-9b-q4`
- `all_models`: `False`
- `samples`: `20`
- `max_tokens`: `100`
- `request_timeout_sec`: `2000`
- `server_start_timeout_sec`: `1200`
- `dataset_file`: `dataset/turboquant_2504_19874v1.md`
- `prompt_mode`: `False`
- `skip_warmup`: `False`
- `use_prompt_cache`: `True`

## Files

- `results_jsonl`: `results/mlx-turboquant-s20-mt100-pc1/20260328T110231Z/benchmark_qwen3.5-9b_64k_s20.jsonl`
- `artifacts_dir`: `data/benchmark/mlx-turboquant-s20-mt100-pc1/20260328T110231Z`
- `summary_json`: `results/mlx-turboquant-s20-mt100-pc1/20260328T110231Z/mlx-qwen3.5-9b-q4-64k-s20.json`

## Counts

- `total`: `20`
- `success`: `20`
- `failure`: `0`

## Runtime Summary

| Runtime | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlx | 20 | 5.911 | 21.811 | 78.683 | 25.722 | 0.868 | 12.940 | 0.967 | 0.800 | 0.175 |

## Runtime + Context Summary

| Runtime | Context | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlx | 64k | 20 | 5.911 | 21.811 | 78.683 | 25.722 | 0.868 | 12.940 | 0.967 | 0.800 | 0.175 |

## Run Results

| Runtime | Model | Case | Context | Time (s) | Tok/s | Prompt tps | Gen tps | TTFT (s) | Peak RAM (GB) | Retrieval score | Retrieval exact | Prompt cache | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-1 | 64k | 4.78 | 26.78 | 98.31 | 31.24 | 0.66 | 12.94 | 0.792 | False | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-10 | 64k | 6.61 | 19.37 | 72.07 | 23.06 | 1.02 | 12.94 | 0.833 | False | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-11 | 64k | 6.35 | 20.16 | 72.97 | 23.68 | 0.91 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-12 | 64k | 5.48 | 23.34 | 78.84 | 27.43 | 0.78 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-13 | 64k | 6.52 | 19.64 | 73.10 | 23.75 | 1.09 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-14 | 64k | 6.26 | 20.45 | 90.17 | 23.67 | 0.82 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-15 | 64k | 6.41 | 19.98 | 73.95 | 23.41 | 0.92 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-16 | 64k | 5.43 | 23.56 | 77.69 | 27.88 | 0.81 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-17 | 64k | 6.26 | 20.46 | 71.99 | 25.04 | 1.09 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-18 | 64k | 5.99 | 21.36 | 78.08 | 25.38 | 0.92 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-19 | 64k | 6.27 | 20.43 | 74.70 | 23.99 | 0.90 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-2 | 64k | 5.87 | 21.79 | 91.20 | 24.76 | 0.68 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-20 | 64k | 5.83 | 21.96 | 77.92 | 26.03 | 0.88 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-3 | 64k | 5.14 | 24.92 | 78.69 | 29.43 | 0.76 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-4 | 64k | 5.69 | 22.49 | 80.47 | 26.15 | 0.77 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-5 | 64k | 5.54 | 23.10 | 80.12 | 26.93 | 0.76 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-6 | 64k | 5.83 | 21.94 | 84.93 | 25.51 | 0.78 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-7 | 64k | 6.32 | 20.25 | 75.87 | 23.60 | 0.87 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-8 | 64k | 5.39 | 23.74 | 70.48 | 28.35 | 0.84 | 12.94 | 0.833 | False | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | long-64k-9 | 64k | 6.24 | 20.51 | 72.10 | 25.15 | 1.11 | 12.94 | 0.875 | False | True | ok |
