from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from src.bench.composition import default_context_k_for_runtime
from src.bench.runner.stats import (
    avg,
    ci95_half_width_for_rate,
    row_peak_memory,
    row_retrieval_exact,
    row_retrieval_score,
    row_throughput,
    row_total_time,
    row_ttft,
    to_float,
)


def display_path(path: Path, root_dir: Path) -> str:
    try:
        return str(path.relative_to(root_dir))
    except Exception:  # noqa: BLE001
        return str(path)


def runtime_summary_rows(results: list[Dict[str, Any]], runtimes: list[str]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
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
                    "avg_retrieval_score_float": 0.0,
                    "retrieval_exact_rate": 0.0,
                    "retrieval_exact_ci95_half_width": 0.0,
                }
            )
            continue

        retrieval_rows = [r for r in runtime_results if row_retrieval_score(r) is not None]
        retrieval_scores = [float(row_retrieval_score(r) or 0.0) for r in retrieval_rows]
        retrieval_exact_rate = avg(
            [1.0 if bool(row_retrieval_exact(r)) else 0.0 for r in retrieval_rows]
        )
        retrieval_exact_ci95_half_width = ci95_half_width_for_rate(retrieval_exact_rate, len(retrieval_rows))

        rows.append(
            {
                "runtime": runtime,
                "count": len(runtime_results),
                "avg_total_time": round(avg([row_total_time(r) for r in runtime_results]), 3),
                "avg_tokens_per_second": round(
                    avg([row_throughput(r, "tokens_per_second") for r in runtime_results]),
                    3,
                ),
                "avg_prompt_tps": round(avg([row_throughput(r, "prompt_tps") for r in runtime_results]), 3),
                "avg_generation_tps": round(
                    avg([row_throughput(r, "generation_tps") for r in runtime_results]),
                    3,
                ),
                "avg_ttft_sec": round(avg([row_ttft(r) for r in runtime_results]), 3),
                "avg_memory_gb": round(avg([row_peak_memory(r) for r in runtime_results]), 3),
                "avg_retrieval_score_float": round(avg(retrieval_scores), 3),
                "retrieval_exact_rate": round(retrieval_exact_rate, 3),
                "retrieval_exact_ci95_half_width": round(retrieval_exact_ci95_half_width, 3),
            }
        )
    return rows


def runtime_context_summary_rows(results: list[Dict[str, Any]], runtimes: list[str]) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    successful = [r for r in results if r.get("success")]
    for runtime in runtimes:
        runtime_results = [r for r in successful if r.get("runtime") == runtime]
        context_groups: Dict[str, list[Dict[str, Any]]] = {}
        for row in runtime_results:
            context_tokens = row.get("context_tokens_target")
            default_context_k = default_context_k_for_runtime(runtime)
            label = f"{int(context_tokens / 1000)}k" if isinstance(context_tokens, int) else f"{default_context_k}k"
            context_groups.setdefault(label, []).append(row)

        for context_label, group in sorted(context_groups.items(), key=lambda item: item[0]):
            retrieval_rows = [r for r in group if row_retrieval_score(r) is not None]
            retrieval_scores = [float(row_retrieval_score(r) or 0.0) for r in retrieval_rows]
            retrieval_exact_rate = avg(
                [1.0 if bool(row_retrieval_exact(r)) else 0.0 for r in retrieval_rows]
            )
            retrieval_exact_ci95_half_width = ci95_half_width_for_rate(retrieval_exact_rate, len(retrieval_rows))

            rows.append(
                {
                    "runtime": runtime,
                    "context": context_label,
                    "count": len(group),
                    "avg_total_time": round(avg([row_total_time(r) for r in group]), 3),
                    "avg_tokens_per_second": round(
                        avg([row_throughput(r, "tokens_per_second") for r in group]),
                        3,
                    ),
                    "avg_prompt_tps": round(avg([row_throughput(r, "prompt_tps") for r in group]), 3),
                    "avg_generation_tps": round(
                        avg([row_throughput(r, "generation_tps") for r in group]),
                        3,
                    ),
                    "avg_ttft_sec": round(avg([row_ttft(r) for r in group]), 3),
                    "avg_memory_gb": round(avg([row_peak_memory(r) for r in group]), 3),
                    "avg_retrieval_score_float": round(avg(retrieval_scores), 3),
                    "retrieval_exact_rate": round(retrieval_exact_rate, 3),
                    "retrieval_exact_ci95_half_width": round(retrieval_exact_ci95_half_width, 3),
                }
            )
    return rows


def write_summary_reports(
    *,
    args: Any,
    results: list[Dict[str, Any]],
    runtimes: list[str],
    setup_metrics: list[Dict[str, Any]],
    output_path: Path,
    artifact_dir: Path,
    root_dir: Path,
    summary_dir: Optional[Path] = None,
    summary_stem: str = "summary",
) -> tuple[Path, Path]:
    generated_at = datetime.now(timezone.utc).isoformat()
    runtime_rows = runtime_summary_rows(results, runtimes)
    runtime_context_rows = runtime_context_summary_rows(results, runtimes)
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
            contexts_k = sorted({default_context_k_for_runtime(runtime) for runtime in runtimes})

    resolved_models = sorted(
        {
            str(row.get("model", "")).strip()
            for row in setup_metrics
            if isinstance(row.get("model"), str) and str(row.get("model", "")).strip()
        }
    )

    params = {
        "dataset": args.dataset,
        "contexts_k": contexts_k,
        "runtime": args.runtime,
        "runtimes_resolved": runtimes,
        "model": resolved_models[0] if len(resolved_models) == 1 else resolved_models,
        "all_models": bool(args.all_models),
        "samples": args.samples,
        "max_tokens": args.max_tokens,
        "request_timeout_sec": args.request_timeout,
        "server_start_timeout_sec": args.server_start_timeout,
        "dataset_file": str(args.dataset_file),
        "prompt_mode": bool(args.prompt),
        "skip_warmup": bool(args.skip_warmup),
        "reasoning_mode": str(getattr(args, "reasoning_mode")),
        "cache_mode": str(getattr(args, "cache_mode")),
        "stream": str(getattr(args, "stream")),
        "transport": str(getattr(args, "transport")),
    }

    summary_json = {
        "generated_at": generated_at,
        "params": params,
        "files": {
            "results_jsonl": display_path(output_path, root_dir),
            "artifacts_dir": display_path(artifact_dir, root_dir),
        },
        "counts": {
            "total": len(results),
            "success": success_count,
            "failure": failure_count,
        },
        "setup_metrics": setup_metrics,
        "runtime_summary": runtime_rows,
        "runtime_context_summary": runtime_context_rows,
        "results": results,
    }

    target_dir = summary_dir or artifact_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = target_dir / f"{summary_stem}.json"
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

    lines: list[str] = []
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
        "reasoning_mode",
        "cache_mode",
        "stream",
        "transport",
    ):
        lines.append(f"- `{key}`: `{params[key]}`")

    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- `results_jsonl`: `{display_path(output_path, root_dir)}`")
    lines.append(f"- `artifacts_dir`: `{display_path(artifact_dir, root_dir)}`")
    lines.append(f"- `summary_json`: `{display_path(summary_json_path, root_dir)}`")

    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(f"- `total`: `{len(results)}`")
    lines.append(f"- `success`: `{success_count}`")
    lines.append(f"- `failure`: `{failure_count}`")

    lines.append("")
    lines.append("## Setup Metrics")
    lines.append("")
    if setup_metrics:
        lines.append(
            "| Runtime | Model | Case build (s) | Download/check (s) | Server start (s) | Warmup (s) | Warmup OK | Setup error |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|---|")
        for row in setup_metrics:
            lines.append(
                f"| {row.get('runtime', '-')} | {row.get('model', '-')} | "
                f"{to_float(row.get('case_build_sec', 0.0)):.3f} | "
                f"{to_float(row.get('download_or_check_sec', 0.0)):.3f} | "
                f"{to_float(row.get('server_start_sec', 0.0)):.3f} | "
                f"{to_float(row.get('warmup_sec', 0.0)):.3f} | "
                f"{row.get('warmup_success', 'n/a')} | "
                f"{str(row.get('setup_error', ''))[:80]} |"
            )
    else:
        lines.append("- (none)")

    lines.append("")
    lines.append("## Runtime Summary")
    lines.append("")
    lines.append(
        "| Runtime | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in runtime_rows:
        lines.append(
            f"| {row['runtime']} | {row['count']} | {row['avg_total_time']:.3f} | "
            f"{row['avg_tokens_per_second']:.3f} | {row['avg_prompt_tps']:.3f} | "
            f"{row['avg_generation_tps']:.3f} | {row['avg_ttft_sec']:.3f} | "
            f"{row['avg_memory_gb']:.3f} | {row['avg_retrieval_score_float']:.3f} | "
            f"{row['retrieval_exact_rate']:.3f} | {row['retrieval_exact_ci95_half_width']:.3f} |"
        )

    lines.append("")
    lines.append("## Runtime + Context Summary")
    lines.append("")
    lines.append(
        "| Runtime | Context | Count | Avg time (s) | Avg tok/s | Avg prompt tps | Avg gen tps | Avg TTFT (s) | Avg Peak RAM (GB) | Avg retrieval score | Retrieval exact rate | Exact CI95 +/- |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in runtime_context_rows:
        lines.append(
            f"| {row['runtime']} | {row['context']} | {row['count']} | {row['avg_total_time']:.3f} | "
            f"{row['avg_tokens_per_second']:.3f} | {row['avg_prompt_tps']:.3f} | "
            f"{row['avg_generation_tps']:.3f} | {row['avg_ttft_sec']:.3f} | "
            f"{row['avg_memory_gb']:.3f} | {row['avg_retrieval_score_float']:.3f} | "
            f"{row['retrieval_exact_rate']:.3f} | {row['retrieval_exact_ci95_half_width']:.3f} |"
        )

    lines.append("")
    lines.append("## Run Results")
    lines.append("")
    lines.append(
        "| Runtime | Model | Case | Context | Time (s) | Tok/s | Prompt tps | Gen tps | TTFT (s) | Peak RAM (GB) | Retrieval score | Retrieval exact | Cache used | Status |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for row in results_sorted:
        context_tokens = row.get("context_tokens_target")
        default_context_k = default_context_k_for_runtime(str(row.get("runtime", "")))
        context_label = f"{int(context_tokens / 1000)}k" if isinstance(context_tokens, int) else f"{default_context_k}k"
        status = "ok" if row.get("success") else f"error: {str(row.get('error', 'unknown'))[:80]}"
        cache = row.get("cache") if isinstance(row.get("cache"), dict) else {}
        lines.append(
            f"| {row.get('runtime', '-')} | {row.get('model', '-')} | {row.get('case_name', '-')} | "
            f"{context_label} | {row_total_time(row):.2f} | {row_throughput(row, 'tokens_per_second'):.2f} | "
            f"{row_throughput(row, 'prompt_tps'):.2f} | {row_throughput(row, 'generation_tps'):.2f} | "
            f"{row_ttft(row):.2f} | {row_peak_memory(row):.2f} | "
            f"{to_float(row_retrieval_score(row) or 0.0):.3f} | {bool(row_retrieval_exact(row))} | "
            f"{bool(cache.get('used', False))} | {status} |"
        )

    summary_md_path = target_dir / f"{summary_stem}.md"
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json_path, summary_md_path
