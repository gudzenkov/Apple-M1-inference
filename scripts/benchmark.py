#!/usr/bin/env python3
"""
LLM Benchmark Script for MLX and Ollama

Generates JSONL output with performance metrics:
- TTFT (Time To First Token)
- Throughput (tokens/second)
- Total time
- Memory usage
- Token counts

Ensures models are downloaded and warmed up before benchmarking.
"""

import json
import time
import requests
import argparse
import subprocess
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Default configurations
CONFIGS = {
    "mlx": {
        "base_url": "http://localhost:8000/v1/chat/completions",
        "default_model": "mlx-community/Qwen3.5-27B-4bit",
        "models": [
            "mlx-community/Qwen3.5-9B-OptiQ-4bit",
            "mlx-community/Qwen3.5-27B-4bit",
        ]
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1/chat/completions",
        "default_model": "qwen3.5:9b",
        "models": [
            "qwen3.5:9b",
            "SimonPu/Qwen3-Coder:30B-Instruct_Q4_K_XL",
            "sinhang/qwen3.5-claude-4.6-opus:27b-q4_K_M",
            "ukjin/Qwen3-30B-A3B-Thinking-2507-Deepseek-v3.1-Distill",
        ]
    }
}

DEFAULT_PROMPTS = [
    "Explain async/await in JavaScript",
    "Write a Python function to calculate fibonacci numbers",
    "What are the benefits of using Rust for systems programming?",
]


def ensure_model_downloaded(model: str, runtime: str) -> bool:
    """Ensure model is downloaded for the runtime"""
    if runtime == "ollama":
        print(f"Checking if {model} is available...", file=sys.stderr)
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        if model not in result.stdout:
            print(f"Pulling {model}...", file=sys.stderr)
            try:
                subprocess.run(
                    ["ollama", "pull", model],
                    check=True,
                    capture_output=False
                )
                print(f"✓ Model {model} downloaded", file=sys.stderr)
                return True
            except subprocess.CalledProcessError as e:
                print(f"✗ Failed to pull {model}: {e}", file=sys.stderr)
                return False
        else:
            print(f"✓ Model {model} already available", file=sys.stderr)
            return True
    # MLX models are downloaded on first use by the server
    return True


def warmup_model(base_url: str, model: str, runtime: str):
    """Run a warmup request to load the model into memory"""
    print(f"Warming up {model}...", file=sys.stderr)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "stream": False
    }

    try:
        response = requests.post(base_url, json=payload, timeout=120)
        if response.status_code == 200:
            print(f"✓ Model warmed up", file=sys.stderr)
        else:
            print(f"⚠ Warmup returned HTTP {response.status_code}", file=sys.stderr)
    except Exception as e:
        print(f"⚠ Warmup failed: {e}", file=sys.stderr)


def get_memory_usage(process_name: str) -> Optional[float]:
    """Get memory usage in GB for a process"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if process_name in line and 'grep' not in line:
                parts = line.split()
                # Memory in KB, convert to GB
                mem_kb = float(parts[5])
                return round(mem_kb / 1024 / 1024, 2)
    except Exception as e:
        print(f"Warning: Could not get memory usage: {e}", file=sys.stderr)
    return None


def benchmark_model(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int = 100,
    runtime: str = "ollama"
) -> Dict[str, Any]:
    """Benchmark a single model with a prompt"""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }

    start_time = time.time()

    try:
        response = requests.post(base_url, json=payload, timeout=120)
        end_time = time.time()

        if response.status_code != 200:
            return {
                "error": f"HTTP {response.status_code}: {response.text[:200]}",
                "runtime": runtime,
                "model": model,
                "prompt": prompt[:50] + "...",
            }

        data = response.json()
        total_time = end_time - start_time

        # Extract metrics
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        # Calculate throughput
        tokens_per_second = completion_tokens / total_time if total_time > 0 else 0

        # Get memory usage
        process_name = "mlx-openai-server" if runtime == "mlx" else "ollama"
        memory_gb = get_memory_usage(process_name)

        # Extract response
        response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

        return {
            "success": True,
            "runtime": runtime,
            "model": model,
            "prompt": prompt,
            "timestamp": datetime.utcnow().isoformat(),
            "total_time": round(total_time, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_second": round(tokens_per_second, 2),
            "memory_gb": memory_gb,
            "response_preview": response_text[:100] + "..." if len(response_text) > 100 else response_text,
        }

    except requests.exceptions.Timeout:
        return {
            "error": "Request timeout (120s)",
            "runtime": runtime,
            "model": model,
            "prompt": prompt[:50] + "...",
        }
    except Exception as e:
        return {
            "error": str(e),
            "runtime": runtime,
            "model": model,
            "prompt": prompt[:50] + "...",
        }


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM performance")
    parser.add_argument(
        "--runtime",
        choices=["mlx", "ollama", "both"],
        default="ollama",
        help="Runtime to benchmark (default: ollama)"
    )
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-9B-OptiQ-4bit",
        help="Specific model to benchmark (overrides default)"
    )
    parser.add_argument(
        "--prompt",
        help="Custom prompt to use"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens to generate (default: 100)"
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.jsonl",
        help="Output JSONL file (default: benchmark_results.jsonl)"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark all configured models for the runtime"
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup request (not recommended)"
    )

    args = parser.parse_args()

    # Determine runtimes to test
    runtimes = ["mlx", "ollama"] if args.runtime == "both" else [args.runtime]

    # Determine prompts
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    results = []
    warmed_up_models = set()  # Track which models have been warmed up

    for runtime in runtimes:
        config = CONFIGS[runtime]

        # Determine models to test
        if args.model:
            models = [args.model]
        elif args.all_models:
            models = config["models"]
        else:
            models = [config["default_model"]]

        for model in models:
            # Ensure model is downloaded (Ollama only)
            if not ensure_model_downloaded(model, runtime):
                print(f"Skipping {model} - download failed", file=sys.stderr)
                continue

            # Warmup model (once per model)
            if not args.skip_warmup and (runtime, model) not in warmed_up_models:
                warmup_model(
                    base_url=config["base_url"],
                    model=model,
                    runtime=runtime
                )
                warmed_up_models.add((runtime, model))
                time.sleep(1)  # Brief pause after warmup

            # Run benchmarks
            for prompt in prompts:
                print(f"Benchmarking {runtime}/{model}...", file=sys.stderr)
                result = benchmark_model(
                    base_url=config["base_url"],
                    model=model,
                    prompt=prompt,
                    max_tokens=args.max_tokens,
                    runtime=runtime
                )
                results.append(result)

                # Print result
                if result.get("success"):
                    print(f"  ✓ {result['tokens_per_second']} tok/s in {result['total_time']}s", file=sys.stderr)
                else:
                    print(f"  ✗ {result.get('error', 'Unknown error')}", file=sys.stderr)

    # Write results to JSONL
    with open(args.output, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')

    print(f"\n✅ Results written to {args.output}", file=sys.stderr)

    # Print summary
    successful = [r for r in results if r.get("success")]
    if successful:
        print("\n📊 Summary:", file=sys.stderr)
        for runtime in runtimes:
            runtime_results = [r for r in successful if r["runtime"] == runtime]
            if runtime_results:
                avg_speed = sum(r["tokens_per_second"] for r in runtime_results) / len(runtime_results)
                print(f"  {runtime}: {avg_speed:.2f} tok/s average", file=sys.stderr)


if __name__ == "__main__":
    main()
