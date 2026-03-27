# Benchmark Toolset Research

Last updated: 2026-03-27

## Scope

Target capability set:
- Local OSS tooling
- OpenAI-compatible API target
- Raw serving performance (latency, TTFT, throughput)
- Retrieval quality on Needle-in-a-Haystack style tasks (recall@1 / recall@k)

Hard truth: there is no established single tool that cleanly gives both production-grade load/perf benchmarking and paper-style NIAH recall@k out of the box.

## Decision Matrix

| Tool | OpenAI-compatible local target | Perf benchmarking | Needle / recall benchmarking | Fit |
|---|---:|---:|---:|---|
| GuideLLM | Yes | Yes | No (not built-in) | Best perf replacement for custom perf runner |
| Inference Perf | Yes (`/completions`, `/chat/completions`) | Yes | No (not built-in) | Strong perf/load patterns, no native NIAH suite |
| AIPerf | Yes | Yes | No (not built-in) | Good perf features, no native NIAH recall suite |
| OpenCompass + NeedleBench | Yes (API models via OpenAI/OpenAISDK) | Not load/perf-focused | Yes (single-needle + multi-needle retrieval/reasoning variants) | Best established NIAH evaluator path |
| LLMTest_NeedleInAHaystack | Yes (OpenAI provider) | Limited | Yes (classic NIAH pressure test) | Lightweight NIAH baseline, less standardized than OpenCompass |

## Recommended Stack

Use two tools:

1. GuideLLM for serving performance.
2. OpenCompass NeedleBench for retrieval recall behavior.

Why:
- GuideLLM is now explicitly recommended by vLLM docs for production vLLM server benchmarking.
- OpenCompass provides maintained NeedleBench dataset/task configs (single, multi-retrieval, multi-reasoning, long-context variants) and supports API-based model evaluation via OpenAI-compatible endpoints.

## Recall Metric Mapping

If you need explicit recall-style KPIs from NeedleBench outputs:

- `Recall@1` (single needle): map from single-needle correctness (`score` 0/100) to 0/1.
- `Recall@k` (multi needle): map from matched-needle fraction in multi retrieval outputs (e.g., matched keywords over total keywords).

Note: OpenCompass reports NeedleBench-oriented scores; you may still want a small post-processing script to emit strict `recall@1` and `recall@k` columns in your internal schema.

## Sources

- vLLM benchmarking docs (recommends GuideLLM):  
  https://raw.githubusercontent.com/vllm-project/vllm/main/docs/benchmarking/cli.md
- GuideLLM repo and backend docs:  
  https://github.com/vllm-project/guidellm  
  https://raw.githubusercontent.com/vllm-project/guidellm/main/docs/guides/backends.md
- Inference Perf README (OpenAI completion/chat support):  
  https://raw.githubusercontent.com/kubernetes-sigs/inference-perf/main/README.md
- AIPerf README + OpenAI text endpoint tutorial:  
  https://raw.githubusercontent.com/ai-dynamo/aiperf/main/README.md  
  https://raw.githubusercontent.com/ai-dynamo/aiperf/main/docs/tutorials/openai-text-endpoints.md
- OpenCompass NeedleBench task docs:  
  https://github.com/open-compass/opencompass/blob/main/opencompass/configs/datasets/needlebench/readme.md
- OpenCompass API model usage (OpenAI/OpenAISDK, `openai_api_base`):  
  https://github.com/open-compass/opencompass/blob/main/opencompass/docs/en/user_guides/models.md
- OpenCompass NeedleBench evaluator implementations:  
  https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/needlebench/origin.py  
  https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/needlebench/parallel.py  
  https://github.com/open-compass/opencompass/blob/main/opencompass/datasets/needlebench/multi.py
- Classic Needle-in-a-Haystack harness:  
  https://github.com/gkamradt/LLMTest_NeedleInAHaystack
- TurboQuant blog (benchmarks referenced):  
  https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
