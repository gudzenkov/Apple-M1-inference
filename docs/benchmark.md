# Performance Benchmark: MLX vs Ollama

Comparison of MLX and Ollama for running large language models on Apple Silicon.

## Quick Benchmark

Use the included benchmark script:
```bash
cd ~/code/Agents/LocalFirst

# Benchmark default model (qwen3.5:9b on Ollama)
./benchmark.py

# Benchmark all Ollama models
./benchmark.py --all-models

# Benchmark both runtimes
./benchmark.py --runtime both --all-models

# Custom prompt
./benchmark.py --prompt "Explain Rust ownership" --max-tokens 200

# Results saved to benchmark_results.jsonl
```

## Test Environment

- **Hardware:** Apple Silicon Mac (M-series)
- **OS:** macOS
- **Context:** 256k tokens configured
- **Date:** March 2026

## Current Model Lineup

### MLX
- mlx-community/Qwen3.5-27B-4bit (~15GB, 4-bit quantized)

### Ollama
- qwen3.5:9b (~6.6GB, Q4_K_M) - Fast, thinking+tools
- SimonPu/Qwen3-Coder:30B (~17GB, Q4_K_XL) - Baseline, tools
- sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M (~16GB) - Opus distill, thinking+tools
- ukjin/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill (~18GB) - Deepseek distill, thinking+tools

## Performance Metrics

### Speed Comparison

| Model | Runtime | Size | Throughput | Use Case |
|-------|---------|------|------------|----------|
| qwen3.5:9b | Ollama | 6.6 GB | ~15-20 tok/s | Fast, general |
| Qwen3.5-27B-4bit | MLX | 15 GB | ~6-8 tok/s | Quality, thinking |
| Qwen3-Coder:30B | Ollama | 17 GB | ~5-7 tok/s | Code, baseline |
| qwen3.5-opus:27b | Ollama | 16 GB | ~5-7 tok/s | Reasoning, Opus |
| Qwen3-Deepseek:30B | Ollama | 18 GB | ~5-7 tok/s | Reasoning, Deepseek |

**Note:** Actual speeds vary based on prompt complexity, context length, and hardware.

### Memory Usage

| Model | RAM Usage | Notes |
|-------|-----------|-------|
| qwen3.5:9b | ~8-10 GB | Lightest, fastest |
| Qwen3.5-27B (MLX) | ~20-25 GB | Largest footprint |
| 30B models (Ollama) | ~18-22 GB | Heavy, quality |
| 27B models (Ollama) | ~16-20 GB | Mid-range |

### Model Capabilities

| Model | Tools | Thinking | Speed | Quality | Context |
|-------|-------|----------|-------|---------|---------|
| qwen3.5:9b | ✅ | ✅ | ⚡⚡⚡ | ⭐⭐ | 256k |
| Qwen3-Coder:30B | ✅ | ❌ | ⚡⚡ | ⭐⭐⭐ | 256k |
| qwen3.5-opus:27b | ✅ | ✅ | ⚡⚡ | ⭐⭐⭐⭐ | 256k |
| Qwen3-Deepseek:30B | ✅ | ✅ | ⚡⚡ | ⭐⭐⭐⭐ | 256k |
| Qwen3.5-27B (MLX) | ✅ | ✅ | ⚡⚡ | ⭐⭐⭐⭐ | 256k |

## When to Use Each

### Use qwen3.5:9b (Ollama)
✅ Fast responses needed
✅ Limited RAM (8-16 GB)
✅ Simple queries, code completion
✅ High throughput workloads

### Use Qwen3-Coder:30B (Ollama)
✅ Code generation baseline
✅ Tool use without thinking overhead
✅ Production code tasks

### Use Opus/Deepseek Distills (Ollama)
✅ Complex reasoning tasks
✅ Thinking process required
✅ Comparison between Opus vs Deepseek distillation
✅ Agentic workflows

### Use MLX (Qwen3.5-27B)
✅ Best quality needed
✅ Latest model architecture
✅ Direct HuggingFace model support
✅ Research and experimentation
✅ 24GB+ RAM available

## Benchmark Script

### Usage Examples

**Default benchmark (qwen3.5:9b):**
```bash
./benchmark.py
```

**All Ollama models:**
```bash
./benchmark.py --all-models
```

**Both MLX and Ollama:**
```bash
./benchmark.py --runtime both --all-models
```

**Specific model:**
```bash
./benchmark.py --model "sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M"
```

**Custom test:**
```bash
./benchmark.py \
  --runtime ollama \
  --model "qwen3.5:9b" \
  --prompt "Write a Rust function for binary search" \
  --max-tokens 500 \
  --output my_results.jsonl
```

### Output Format (JSONL)

Each line contains a benchmark result:
```json
{
  "success": true,
  "runtime": "ollama",
  "model": "qwen3.5:9b",
  "prompt": "Explain async/await in JavaScript",
  "timestamp": "2026-03-25T12:00:00.000000",
  "total_time": 8.5,
  "prompt_tokens": 12,
  "completion_tokens": 100,
  "total_tokens": 112,
  "tokens_per_second": 11.76,
  "memory_gb": 8.5,
  "response_preview": "async/await is a modern JavaScript feature..."
}
```

### Analyzing Results

**View results:**
```bash
cat benchmark_results.jsonl | jq '.'
```

**Get average speed per model:**
```bash
cat benchmark_results.jsonl | jq -s 'group_by(.model) | map({model: .[0].model, avg_speed: (map(.tokens_per_second) | add / length)})'
```

**Compare runtimes:**
```bash
cat benchmark_results.jsonl | jq -s 'group_by(.runtime) | map({runtime: .[0].runtime, avg_speed: (map(.tokens_per_second) | add / length)})'
```

## Hardware Requirements

### Minimum

**For 9B models:**
- Mac M1 or better
- 16GB unified memory
- macOS 13.0+
- 15GB free disk space

**For 27-30B models:**
- Mac M2 Pro or better
- 24GB unified memory
- macOS 13.0+
- 25GB free disk space

### Recommended

**For best performance:**
- Mac M3 Pro/Max or M4
- 32GB+ unified memory
- 50GB+ free disk space
- SSD for model storage

## Optimization Tips

### Ollama
- Use 256k context models for agentic workflows
- Set `OLLAMA_NUM_PARALLEL=1` for consistency
- Use Q4_K_M quantization for balance
- Monitor with `ollama ps`

### MLX
- Configure `--context-length 262144`
- Use 4-bit quantized models
- Monitor GPU: `sudo powermetrics --samplers gpu_power`
- Check memory: `ps aux | grep mlx`

## Reproducibility

All benchmarks run with:
- Same prompts across models
- 100 token generation limit
- No streaming
- 256k context configured
- Cold start (first request after server start)

Run your own benchmarks with the provided script for accurate results on your hardware.

## Resources

- [MLX Setup](MLX.md)
- [Ollama Setup](ollama.md)
- [Benchmark Script](../benchmark.py)
- [MLX GitHub](https://github.com/ml-explore/mlx)
- [Ollama GitHub](https://github.com/ollama/ollama)
