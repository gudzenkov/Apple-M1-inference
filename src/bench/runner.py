from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bench.config import CONFIGS, resolve_runtimes, select_models
from src.bench.datasets import build_cases
from src.bench.metrics import benchmark_model
from src.bench.process import (
    ensure_model_downloaded,
    start_managed_server,
    stop_managed_process,
    stop_mlx_servers,
    warmup_model,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"


def run_benchmark(args: Any) -> int:
    runtimes = resolve_runtimes(args.runtime)
    cases = build_cases(
        dataset=args.dataset,
        samples=args.samples,
        dataset_file=Path(args.dataset_file),
        prompt=args.prompt,
        prompt_max_tokens=args.max_tokens,
    )

    results: List[Dict[str, Any]] = []

    print("Managed mode: MLX servers run sequentially (never together) to avoid OOM.", file=sys.stderr)

    for runtime in runtimes:
        config = CONFIGS[runtime]
        models = select_models(config, args.model, args.all_models)

        for model in models:
            managed_proc: Optional[Any] = None
            memory_pid: Optional[int] = None

            try:
                if config["managed_server"]:
                    stop_mlx_servers(verbose=True)
                    managed_proc = start_managed_server(
                        runtime=runtime,
                        model=model,
                        timeout_sec=args.server_start_timeout,
                        root_dir=ROOT_DIR,
                        config=config,
                        log_dir=LOG_DIR,
                    )
                    memory_pid = managed_proc.pid
                    if not args.skip_warmup:
                        warmup_model(config["chat_url"], model)
                        time.sleep(1)
                else:
                    if not ensure_model_downloaded(model, runtime):
                        print(f"Skipping {runtime}/{model} (download failed)", file=sys.stderr)
                        continue
                    if not args.skip_warmup:
                        warmup_model(config["chat_url"], model)
                        time.sleep(1)

                for case in cases:
                    print(f"Benchmarking {runtime}/{model} [{case['case_name']}]...", file=sys.stderr)
                    result = benchmark_model(
                        chat_url=config["chat_url"],
                        model=model,
                        prompt=case["prompt"],
                        max_tokens=case["max_tokens"],
                        runtime=runtime,
                        memory_pid=memory_pid,
                        memory_pattern=config.get("process_hint"),
                        request_timeout_sec=args.request_timeout,
                    )
                    result.update(
                        {
                            "dataset": case.get("dataset"),
                            "case_name": case.get("case_name"),
                            "context_tokens_target": case.get("context_tokens_target"),
                            "payload_words": case.get("payload_words"),
                            "payload_source": case.get("payload_source"),
                        }
                    )
                    results.append(result)

                    if result.get("success"):
                        print(
                            f"  ✓ {result['tokens_per_second']} tok/s in {result['total_time']}s",
                            file=sys.stderr,
                        )
                    else:
                        print(f"  ✗ {result.get('error', 'Unknown error')}", file=sys.stderr)
            finally:
                if managed_proc is not None:
                    stop_managed_process(managed_proc)
                if config["managed_server"]:
                    stop_mlx_servers(verbose=False)

    output_name = args.output or f"results/benchmark_{args.dataset}.jsonl"
    output_arg = Path(output_name)
    if not output_arg.is_absolute() and output_arg.parent == Path("."):
        output_arg = Path("results") / output_arg

    output_path = output_arg if output_arg.is_absolute() else (ROOT_DIR / output_arg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

    print(f"\n✅ Results written to {output_path}", file=sys.stderr)

    successful = [r for r in results if r.get("success")]
    if successful:
        print("\n📊 Summary:", file=sys.stderr)
        for runtime in runtimes:
            runtime_results = [r for r in successful if r["runtime"] == runtime]
            if not runtime_results:
                continue
            avg_speed = sum(r["tokens_per_second"] for r in runtime_results) / len(runtime_results)
            avg_time = sum(r["total_time"] for r in runtime_results) / len(runtime_results)
            print(f"  {runtime}: avg {avg_speed:.2f} tok/s, avg {avg_time:.2f}s", file=sys.stderr)

    return 0
