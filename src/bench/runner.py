from __future__ import annotations

import json
import re
import sys
import time
from datetime import datetime, timezone
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
from src.shared.models import get_model_key

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"


def _slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    out = re.sub(r"-{2,}", "-", out).strip("-._")
    return out or "x"


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _dataset_name_from_file(dataset_file: str) -> str:
    stem = Path(dataset_file).stem.lower()
    if not stem:
        return "dataset"
    parts = re.split(r"[_\-]+", stem)
    if parts and parts[0]:
        return _slug(parts[0])
    return _slug(stem)


def _experiment_group(args: Any) -> str:
    runtime_part = _slug(str(args.runtime))
    if args.prompt:
        dataset_part = "prompt"
    elif getattr(args, "contexts_k", None):
        dataset_part = _dataset_name_from_file(str(args.dataset_file))
    elif str(args.dataset) in {"long", "all"}:
        dataset_part = _dataset_name_from_file(str(args.dataset_file))
    else:
        dataset_part = "quick"
    return _slug(f"{runtime_part}-{dataset_part}-s{args.samples}-mt{args.max_tokens}")


def _sample_id(case_name: str) -> str:
    match = re.search(r"-(\d+)$", case_name)
    if match:
        return match.group(1)
    return "1"


def _default_context_k_for_runtime(runtime: str) -> int:
    max_context_tokens = int(CONFIGS.get(runtime, {}).get("max_context_tokens", 256000))
    return max(1, int(max_context_tokens / 1000))


def _context_label(case: Dict[str, Any], default_context_k: int) -> str:
    tokens = case.get("context_tokens_target")
    if isinstance(tokens, int) and tokens > 0:
        return f"{int(tokens / 1000)}k"
    return f"{default_context_k}k"


def _model_label(model_id: str, runtime: str) -> str:
    try:
        return _slug(get_model_key(model_id, runtime=runtime))
    except Exception:  # noqa: BLE001
        return _slug(model_id)


def _run_param(runtime: str, model: str, case: Dict[str, Any], default_context_k: int) -> str:
    model_part = _model_label(model, runtime)
    context_part = _context_label(case, default_context_k=default_context_k)
    sample_part = _sample_id(str(case.get("case_name", "s1")))
    return _slug(f"{runtime}-{model_part}-{context_part}-s{sample_part}")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except Exception:  # noqa: BLE001
        return str(path)


def _avg(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _runtime_summary_rows(results: List[Dict[str, Any]], runtimes: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    successful = [r for r in results if r.get("success")]
    for runtime in runtimes:
        runtime_results = [r for r in successful if r.get("runtime") == runtime]
        if not runtime_results:
            rows.append(
                {
                    "runtime": runtime,
                    "count": 0,
                    "avg_total_time": 0.0,
                    "avg_tokens_per_second": 0.0,
                    "avg_prompt_tps": 0.0,
                    "avg_generation_tps": 0.0,
                    "avg_ttft_sec": 0.0,
                    "avg_memory_gb": 0.0,
                }
            )
            continue
        rows.append(
            {
                "runtime": runtime,
                "count": len(runtime_results),
                "avg_total_time": round(_avg([float(r.get("total_time", 0.0) or 0.0) for r in runtime_results]), 3),
                "avg_tokens_per_second": round(
                    _avg([float(r.get("tokens_per_second", 0.0) or 0.0) for r in runtime_results]), 3
                ),
                "avg_prompt_tps": round(_avg([float(r.get("prompt_tps", 0.0) or 0.0) for r in runtime_results]), 3),
                "avg_generation_tps": round(
                    _avg([float(r.get("generation_tps", 0.0) or 0.0) for r in runtime_results]), 3
                ),
                "avg_ttft_sec": round(_avg([float(r.get("ttft_sec", 0.0) or 0.0) for r in runtime_results]), 3),
                "avg_memory_gb": round(_avg([float(r.get("memory_gb", 0.0) or 0.0) for r in runtime_results]), 3),
            }
        )
    return rows


def _write_summary_reports(
    args: Any,
    results: List[Dict[str, Any]],
    runtimes: List[str],
    output_path: Path,
    experiment_dir: Path,
) -> tuple[Path, Path]:
    generated_at = datetime.now(timezone.utc).isoformat()
    runtime_rows = _runtime_summary_rows(results, runtimes)
    success_count = sum(1 for r in results if r.get("success"))
    failure_count = len(results) - success_count
    contexts_k_raw = getattr(args, "contexts_k", None)
    if contexts_k_raw:
        contexts_k = contexts_k_raw
    else:
        case_contexts = sorted(
            {
                int(r["context_tokens_target"] / 1000)
                for r in results
                if isinstance(r.get("context_tokens_target"), int)
            }
        )
        if case_contexts:
            contexts_k = case_contexts
        else:
            contexts_k = sorted({_default_context_k_for_runtime(runtime) for runtime in runtimes})

    params = {
        "dataset": args.dataset,
        "contexts_k": contexts_k,
        "runtime": args.runtime,
        "runtimes_resolved": runtimes,
        "model": args.model,
        "all_models": bool(args.all_models),
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "request_timeout_sec": args.request_timeout,
        "server_start_timeout_sec": args.server_start_timeout,
        "dataset_file": str(args.dataset_file),
        "prompt_mode": bool(args.prompt),
        "skip_warmup": bool(args.skip_warmup),
    }

    summary_json = {
        "generated_at": generated_at,
        "params": params,
        "files": {
            "results_jsonl": _display_path(output_path),
            "artifacts_dir": _display_path(experiment_dir),
        },
        "counts": {
            "total": len(results),
            "success": success_count,
            "failure": failure_count,
        },
        "runtime_summary": runtime_rows,
        "results": results,
    }

    summary_json_path = experiment_dir / "summary.json"
    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    results_sorted = sorted(
        results,
        key=lambda r: (
            str(r.get("runtime", "")),
            str(r.get("model", "")),
            str(r.get("case_name", "")),
        ),
    )

    lines: List[str] = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"Generated: `{generated_at}`")
    lines.append("")
    lines.append("## Params")
    lines.append("")
    for key in (
        "dataset",
        "contexts_k",
        "runtime",
        "runtimes_resolved",
        "model",
        "all_models",
        "samples",
        "max_tokens",
        "request_timeout_sec",
        "server_start_timeout_sec",
        "dataset_file",
        "prompt_mode",
        "skip_warmup",
    ):
        lines.append(f"- `{key}`: `{params[key]}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- `results_jsonl`: `{_display_path(output_path)}`")
    lines.append(f"- `artifacts_dir`: `{_display_path(experiment_dir)}`")
    lines.append(f"- `summary_json`: `{_display_path(summary_json_path)}`")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- `total`: `{len(results)}`")
    lines.append(f"- `success`: `{success_count}`")
    lines.append(f"- `failure`: `{failure_count}`")
    lines.append("")
    lines.append("## Runtime Summary")
    lines.append("")
    lines.append(
        "| Runtime | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg TTFT (s) | Avg Peak RAM (GB) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in runtime_rows:
        lines.append(
            f"| {row['runtime']} | {row['count']} | {row['avg_total_time']:.3f} | "
            f"{row['avg_tokens_per_second']:.3f} | {row['avg_prompt_tps']:.3f} | "
            f"{row['avg_generation_tps']:.3f} | {row['avg_ttft_sec']:.3f} | "
            f"{row['avg_memory_gb']:.3f} |"
        )
    lines.append("")
    lines.append("## Run Results")
    lines.append("")
    lines.append(
        "| Runtime | Model | Case | Context | Time (s) | Tok/s | Prompt tps | Gen tps | TTFT (s) | Peak RAM (GB) | Status |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in results_sorted:
        context_tokens = row.get("context_tokens_target")
        default_context_k = _default_context_k_for_runtime(str(row.get("runtime", "")))
        context_label = (
            f"{int(context_tokens / 1000)}k"
            if isinstance(context_tokens, int)
            else f"{default_context_k}k"
        )
        status = "ok" if row.get("success") else f"error: {str(row.get('error', 'unknown'))[:80]}"
        lines.append(
            f"| {row.get('runtime', '-')} | {row.get('model', '-')} | {row.get('case_name', '-')} | "
            f"{context_label} | {float(row.get('total_time', 0.0) or 0.0):.2f} | "
            f"{float(row.get('tokens_per_second', 0.0) or 0.0):.2f} | "
            f"{float(row.get('prompt_tps', 0.0) or 0.0):.2f} | "
            f"{float(row.get('generation_tps', 0.0) or 0.0):.2f} | "
            f"{float(row.get('ttft_sec', 0.0) or 0.0):.2f} | "
            f"{float(row.get('memory_gb', 0.0) or 0.0):.2f} | {status} |"
        )

    summary_md_path = experiment_dir / "summary.md"
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json_path, summary_md_path


def run_benchmark(args: Any) -> int:
    runtimes = resolve_runtimes(args.runtime)
    cases = build_cases(
        dataset=args.dataset,
        samples=args.samples,
        dataset_file=Path(args.dataset_file),
        prompt=args.prompt,
        prompt_max_tokens=args.max_tokens,
        contexts_k=getattr(args, "contexts_k", None),
    )

    results: List[Dict[str, Any]] = []
    experiment_dir = ROOT_DIR / "data" / "benchmark" / _experiment_group(args) / _timestamp_slug()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Managed mode: MLX servers run sequentially (never together) to avoid OOM.", file=sys.stderr)
    print(f"Artifacts directory: {experiment_dir}", file=sys.stderr)

    for runtime in runtimes:
        config = CONFIGS[runtime]
        models = select_models(config, args.model, args.all_models)

        for model in models:
            managed_proc: Optional[Any] = None
            memory_pid: Optional[int] = None
            default_context_k = _default_context_k_for_runtime(runtime)

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
                    run_dir = experiment_dir / _run_param(
                        runtime,
                        model,
                        case,
                        default_context_k=default_context_k,
                    )
                    result = benchmark_model(
                        chat_url=config["chat_url"],
                        model=model,
                        prompt=case["prompt"],
                        max_tokens=case["max_tokens"],
                        runtime=runtime,
                        memory_pid=memory_pid,
                        memory_pattern=config.get("process_hint"),
                        request_timeout_sec=args.request_timeout,
                        artifact_dir=run_dir,
                    )
                    case_meta: Dict[str, Any] = {
                        "dataset": case.get("dataset"),
                        "case_name": case.get("case_name"),
                    }
                    for key in ("context_tokens_target", "payload_source"):
                        value = case.get(key)
                        if value is not None:
                            case_meta[key] = value
                    result.update(case_meta)
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

    if args.output:
        output_name = args.output
    elif getattr(args, "contexts_k", None):
        context_slug = "-".join(f"{c}k" for c in args.contexts_k)
        output_name = f"results/benchmark_context_{context_slug}.jsonl"
    else:
        output_name = f"results/benchmark_{args.dataset}.jsonl"
    output_arg = Path(output_name)
    if not output_arg.is_absolute() and output_arg.parent == Path("."):
        output_arg = Path("results") / output_arg

    output_path = output_arg if output_arg.is_absolute() else (ROOT_DIR / output_arg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

    print(f"\n✅ Results written to {output_path}", file=sys.stderr)
    summary_json_path, summary_md_path = _write_summary_reports(
        args=args,
        results=results,
        runtimes=runtimes,
        output_path=output_path,
        experiment_dir=experiment_dir,
    )
    print(f"✅ Summary JSON: {_display_path(summary_json_path)}", file=sys.stderr)
    print(f"✅ Summary MD: {_display_path(summary_md_path)}", file=sys.stderr)

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
