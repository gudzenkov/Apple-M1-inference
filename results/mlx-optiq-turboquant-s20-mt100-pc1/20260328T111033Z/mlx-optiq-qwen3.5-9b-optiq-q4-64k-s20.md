# Benchmark Summary

Generated: `2026-03-28T11:17:48.316463+00:00`

## Params

- `dataset`: `long`
- `contexts_k`: `[64]`
- `runtime`: `mlx-optiq`
- `runtimes_resolved`: `['mlx-optiq']`
- `model`: `qwen3.5-9b-optiq-q4`
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

- `results_jsonl`: `results/mlx-optiq-turboquant-s20-mt100-pc1/20260328T111033Z/benchmark_qwen3.5-9b-optiq_64k_s20.jsonl`
- `artifacts_dir`: `data/benchmark/mlx-optiq-turboquant-s20-mt100-pc1/20260328T111033Z`
- `summary_json`: `results/mlx-optiq-turboquant-s20-mt100-pc1/20260328T111033Z/mlx-optiq-qwen3.5-9b-optiq-q4-64k-s20.json`

## Counts

- `total`: `20`
- `success`: `20`
- `failure`: `0`

## Runtime Summary

| Runtime | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlx-optiq | 20 | 5.607 | 22.848 | 93.727 | 26.167 | 0.689 | 15.350 | 1.000 | 1.000 | 0.000 |

## Runtime + Context Summary

| Runtime | Context | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlx-optiq | 64k | 20 | 5.607 | 22.848 | 93.727 | 26.167 | 0.689 | 15.350 | 1.000 | 1.000 | 0.000 |

## Run Results

| Runtime | Model | Case | Context | Time (s) | Tok/s | Prompt tps | Gen tps | TTFT (s) | Peak RAM (GB) | Retrieval score | Retrieval exact | Prompt cache | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-1 | 64k | 5.30 | 24.14 | 102.10 | 27.58 | 0.64 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-10 | 64k | 5.70 | 22.46 | 94.78 | 25.58 | 0.67 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-11 | 64k | 5.66 | 22.63 | 93.40 | 25.85 | 0.68 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-12 | 64k | 5.54 | 23.12 | 90.55 | 26.63 | 0.71 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-13 | 64k | 5.61 | 22.83 | 97.46 | 26.03 | 0.67 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-14 | 64k | 5.59 | 22.88 | 89.93 | 26.32 | 0.71 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-15 | 64k | 5.78 | 22.14 | 95.85 | 25.20 | 0.68 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-16 | 64k | 5.44 | 23.54 | 91.32 | 27.11 | 0.69 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-17 | 64k | 5.74 | 22.29 | 97.14 | 25.37 | 0.67 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-18 | 64k | 5.84 | 21.92 | 87.80 | 25.14 | 0.73 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-19 | 64k | 5.43 | 23.57 | 98.13 | 26.95 | 0.66 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-2 | 64k | 5.55 | 23.07 | 101.12 | 26.31 | 0.66 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-20 | 64k | 5.61 | 22.83 | 95.01 | 26.16 | 0.69 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-3 | 64k | 5.91 | 21.67 | 81.20 | 25.29 | 0.82 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-4 | 64k | 5.44 | 23.54 | 95.48 | 26.95 | 0.67 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-5 | 64k | 5.61 | 22.82 | 96.90 | 26.01 | 0.67 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-6 | 64k | 5.43 | 23.56 | 94.13 | 26.97 | 0.67 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-7 | 64k | 5.58 | 22.92 | 93.64 | 26.17 | 0.67 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-8 | 64k | 5.84 | 21.91 | 90.52 | 25.03 | 0.71 | 15.35 | 1.000 | True | True | ok |
| mlx-optiq | mlx-community/Qwen3.5-9B-OptiQ-4bit | long-64k-9 | 64k | 5.53 | 23.13 | 88.07 | 26.68 | 0.72 | 15.35 | 1.000 | True | True | ok |
