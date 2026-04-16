# Benchmark Summary

Generated: `2026-04-16T10:44:23.692552+00:00`

## Params

- `dataset`: `long`
- `contexts_k`: `[64]`
- `runtime`: `mlx`
- `runtimes_resolved`: `['mlx']`
- `model`: `mlx-community/Qwen3.5-9B-4bit`
- `all_models`: `False`
- `samples`: `3`
- `max_tokens`: `100`
- `request_timeout_sec`: `None`
- `server_start_timeout_sec`: `None`
- `dataset_file`: `dataset/turboquant_2504_19874v1.md`
- `prompt_mode`: `False`
- `skip_warmup`: `False`
- `reasoning_mode`: `off`
- `cache_mode`: `prefill`
- `stream`: `on`
- `transport`: `auto`

## Files

- `results_jsonl`: `/tmp/bench-rerun-mlx-64k-prefill-fixed.jsonl`
- `artifacts_dir`: `data/benchmark/mlx-turboquant-s3-mt100-cache-prefill/20260416T103835Z`
- `summary_json`: `results/mlx-turboquant-s3-mt100-cache-prefill/20260416T103835Z/mlx-mlx-community-Qwen3.5-9B-4bit-64k-s3.json`

## Counts

- `total`: `3`
- `success`: `3`
- `failure`: `0`
- `cache_prime_total`: `1`
- `cache_prime_success`: `1`
- `cache_prime_failure`: `0`

## Setup Metrics

| Runtime | Model | Case build (s) | Download/check (s) | Server start (s) | Warmup (s) | Warmup OK | Setup error |
|---|---|---:|---:|---:|---:|---:|---|
| mlx | mlx-community/Qwen3.5-9B-4bit | 4.073 | 0.000 | 3.036 | 0.635 | True |  |

## Runtime Summary

| Runtime | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg Prefill (s) | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlx | 3 | 5.183 | 24.740 | 96.536 | 29.255 | 317.255 | 0.737 | 12.943 | 1.000 | 1.000 | 0.000 |

## Runtime + Context Summary

| Runtime | Context | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg Prefill (s) | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mlx | 64k | 3 | 5.183 | 24.740 | 96.536 | 29.255 | 317.255 | 0.737 | 12.943 | 1.000 | 1.000 | 0.000 |

## Run Results

| Runtime | Model | Phase | Case | Context | Time (s) | Tok/s | Prompt tps | Gen tps | Prefill (s) | TTFT (s) | Peak RAM (GB) | Retrieval score | Retrieval exact | Cache used | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mlx | mlx-community/Qwen3.5-9B-4bit | cache-prime | long-64k-0 | 64k | 4.73 | 27.09 | 96.54 | 32.04 | 317.25 | 0.66 | 12.94 | 0.000 | False | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | benchmark | long-64k-1 | 64k | 5.46 | 23.42 | 81.30 | 27.52 | 0.00 | 0.75 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | benchmark | long-64k-2 | 64k | 4.92 | 26.00 | 90.16 | 30.70 | 0.00 | 0.69 | 12.94 | 1.000 | True | True | ok |
| mlx | mlx-community/Qwen3.5-9B-4bit | benchmark | long-64k-3 | 64k | 5.16 | 24.80 | 77.70 | 29.55 | 0.00 | 0.77 | 12.94 | 1.000 | True | True | ok |
