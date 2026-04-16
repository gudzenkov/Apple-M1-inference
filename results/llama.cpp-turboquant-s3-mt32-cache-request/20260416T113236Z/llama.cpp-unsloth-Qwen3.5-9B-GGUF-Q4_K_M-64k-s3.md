# Benchmark Summary

Generated: `2026-04-16T11:37:06.051596+00:00`

## Params

- `dataset`: `long`
- `contexts_k`: `[64]`
- `runtime`: `llama.cpp`
- `runtimes_resolved`: `['llama.cpp']`
- `model`: `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`
- `all_models`: `False`
- `samples`: `3`
- `max_tokens`: `32`
- `request_timeout_sec`: `None`
- `server_start_timeout_sec`: `None`
- `dataset_file`: `dataset/turboquant_2504_19874v1.md`
- `prompt_mode`: `False`
- `skip_warmup`: `False`
- `reasoning_mode`: `off`
- `cache_mode`: `request`
- `stream`: `on`
- `transport`: `auto`

## Files

- `results_jsonl`: `/tmp/llama-cpp-64k-s3.jsonl`
- `artifacts_dir`: `data/benchmark/llama.cpp-turboquant-s3-mt32-cache-request/20260416T113236Z`
- `summary_json`: `results/llama.cpp-turboquant-s3-mt32-cache-request/20260416T113236Z/llama.cpp-unsloth-Qwen3.5-9B-GGUF-Q4_K_M-64k-s3.json`

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
| llama.cpp | unsloth/Qwen3.5-9B-GGUF:Q4_K_M | 6.549 | 0.000 | 3.030 | 0.346 | True |  |

## Runtime Summary

| Runtime | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg Prefill (s) | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama.cpp | 3 | 3.970 | 5.710 | 261.343 | 23.806 | 245.299 | 2.802 | 10.310 | 1.000 | 1.000 | 0.000 |

## Runtime + Context Summary

| Runtime | Context | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg Prefill (s) | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama.cpp | 64k | 3 | 3.970 | 5.710 | 261.343 | 23.806 | 245.299 | 2.802 | 10.310 | 1.000 | 1.000 | 0.000 |

## Run Results

| Runtime | Model | Phase | Case | Context | Time (s) | Tok/s | Prompt tps | Gen tps | Prefill (s) | TTFT (s) | Peak RAM (GB) | Retrieval score | Retrieval exact | Cache used | Status |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| llama.cpp | unsloth/Qwen3.5-9B-GGUF:Q4_K_M | cache-prime | long-64k-0 | 64k | 245.30 | 0.03 | 261.34 | 26.55 | 245.30 | 244.83 | 10.31 | 0.000 | False | False | ok |
| llama.cpp | unsloth/Qwen3.5-9B-GGUF:Q4_K_M | benchmark | long-64k-1 | 64k | 3.97 | 5.54 | 192.68 | 23.63 | 0.00 | 2.81 | 10.31 | 1.000 | True | False | ok |
| llama.cpp | unsloth/Qwen3.5-9B-GGUF:Q4_K_M | benchmark | long-64k-2 | 64k | 3.98 | 5.77 | 192.87 | 23.98 | 0.00 | 2.81 | 10.31 | 1.000 | True | False | ok |
| llama.cpp | unsloth/Qwen3.5-9B-GGUF:Q4_K_M | benchmark | long-64k-3 | 64k | 3.95 | 5.82 | 194.82 | 23.81 | 0.00 | 2.78 | 10.31 | 1.000 | True | False | ok |
