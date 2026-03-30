from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bench.config import CONFIGS, resolve_runtimes, select_models
from src.bench.datasets import build_cases
from src.bench.metrics import (
    benchmark_model,
    clear_prompt_cache,
    prefill_prompt_cache,
)
from src.bench.process import (
    ensure_model_downloaded,
    start_managed_server,
    stop_managed_process,
    stop_mlx_servers,
    warmup_model,
)
from src.bench.utils import default_output_filename, default_summary_stem, resolve_experiment_paths
from src.shared.models import (
    get_default_model_id,
    load_models,
    resolve_runtime_for_model_reference,
)

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
        dataset_part = "short"
    cache_part = "pc1" if bool(getattr(args, "use_prompt_cache", False)) else "pc0"
    return _slug(f"{runtime_part}-{dataset_part}-s{args.samples}-mt{args.max_tokens}-{cache_part}")


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
    _ = runtime
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


def _ci95_half_width_for_rate(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p))
    return 1.96 * math.sqrt((p * (1.0 - p)) / float(n))


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def _extract_retrieval_answer(raw_text: str) -> str:
    match = re.search(r"\b(NIAH-[A-Z]+-\d+K-S\d{2}-\d{6})\b", raw_text)
    if match:
        return match.group(1)
    for line in raw_text.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        if candidate.lower().startswith("answer:"):
            candidate = candidate.split(":", 1)[1].strip()
        candidate = candidate.strip(" \t\r\n\"'`.,;:!?")
        return _normalize_text(candidate)
    return _normalize_text(raw_text.strip(" \t\r\n\"'`.,;:!?"))


def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        curr = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def _retrieval_score(predicted: str, expected: str) -> float:
    max_len = max(len(predicted), len(expected), 1)
    dist = _levenshtein_distance(predicted, expected)
    return max(0.0, 1.0 - (float(dist) / float(max_len)))


def _resolve_runtimes_for_args(args: Any) -> List[str]:
    runtime_arg = str(getattr(args, "runtime", "auto") or "auto").strip().lower()
    if runtime_arg != "auto":
        return resolve_runtimes(runtime_arg)

    model_arg = getattr(args, "model", None)
    if isinstance(model_arg, str) and model_arg.strip():
        return [resolve_runtime_for_model_reference(model_arg)]

    if bool(getattr(args, "all_models", False)):
        runtimes: List[str] = []
        for entry in load_models():
            runtime = str(entry["runtime"]).strip().lower()
            if runtime and runtime not in runtimes:
                runtimes.append(runtime)
        if runtimes:
            return runtimes

    default_model = get_default_model_id()
    return [resolve_runtime_for_model_reference(default_model)]


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
                    "avg_retrieval_score_float": 0.0,
                    "retrieval_exact_rate": 0.0,
                    "retrieval_exact_ci95_half_width": 0.0,
                }
            )
            continue
        retrieval_rows = [r for r in runtime_results if isinstance(r.get("retrieval_score_float"), (float, int))]
        retrieval_scores = [float(r.get("retrieval_score_float", 0.0) or 0.0) for r in retrieval_rows]
        retrieval_exact_rate = _avg(
            [1.0 if bool(r.get("retrieval_exact", False)) else 0.0 for r in retrieval_rows]
        )
        retrieval_exact_ci95_half_width = _ci95_half_width_for_rate(retrieval_exact_rate, len(retrieval_rows))
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
                "avg_retrieval_score_float": round(_avg(retrieval_scores), 3),
                "retrieval_exact_rate": round(retrieval_exact_rate, 3),
                "retrieval_exact_ci95_half_width": round(retrieval_exact_ci95_half_width, 3),
            }
        )
    return rows


def _runtime_context_summary_rows(results: List[Dict[str, Any]], runtimes: List[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    successful = [r for r in results if r.get("success")]
    for runtime in runtimes:
        runtime_results = [r for r in successful if r.get("runtime") == runtime]
        context_groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in runtime_results:
            context_tokens = row.get("context_tokens_target")
            default_context_k = _default_context_k_for_runtime(runtime)
            label = (
                f"{int(context_tokens / 1000)}k"
                if isinstance(context_tokens, int)
                else f"{default_context_k}k"
            )
            context_groups.setdefault(label, []).append(row)
        for context_label, group in sorted(context_groups.items(), key=lambda item: item[0]):
            retrieval_rows = [r for r in group if isinstance(r.get("retrieval_score_float"), (float, int))]
            retrieval_scores = [float(r.get("retrieval_score_float", 0.0) or 0.0) for r in retrieval_rows]
            retrieval_exact_rate = _avg(
                [1.0 if bool(r.get("retrieval_exact", False)) else 0.0 for r in retrieval_rows]
            )
            retrieval_exact_ci95_half_width = _ci95_half_width_for_rate(
                retrieval_exact_rate,
                len(retrieval_rows),
            )
            rows.append(
                {
                    "runtime": runtime,
                    "context": context_label,
                    "count": len(group),
                    "avg_total_time": round(_avg([float(r.get("total_time", 0.0) or 0.0) for r in group]), 3),
                    "avg_tokens_per_second": round(
                        _avg([float(r.get("tokens_per_second", 0.0) or 0.0) for r in group]), 3
                    ),
                    "avg_prompt_tps": round(_avg([float(r.get("prompt_tps", 0.0) or 0.0) for r in group]), 3),
                    "avg_generation_tps": round(
                        _avg([float(r.get("generation_tps", 0.0) or 0.0) for r in group]), 3
                    ),
                    "avg_ttft_sec": round(_avg([float(r.get("ttft_sec", 0.0) or 0.0) for r in group]), 3),
                    "avg_memory_gb": round(_avg([float(r.get("memory_gb", 0.0) or 0.0) for r in group]), 3),
                    "avg_retrieval_score_float": round(_avg(retrieval_scores), 3),
                    "retrieval_exact_rate": round(retrieval_exact_rate, 3),
                    "retrieval_exact_ci95_half_width": round(retrieval_exact_ci95_half_width, 3),
                }
            )
    return rows


def _write_summary_reports(
    args: Any,
    results: List[Dict[str, Any]],
    runtimes: List[str],
    setup_metrics: List[Dict[str, Any]],
    output_path: Path,
    artifact_dir: Path,
    summary_dir: Optional[Path] = None,
    summary_stem: str = "summary",
) -> tuple[Path, Path]:
    generated_at = datetime.now(timezone.utc).isoformat()
    runtime_rows = _runtime_summary_rows(results, runtimes)
    runtime_context_rows = _runtime_context_summary_rows(results, runtimes)
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
        "use_prompt_cache": bool(getattr(args, "use_prompt_cache", False)),
        "reasoning": str(getattr(args, "reasoning", "off")),
    }

    summary_json = {
        "generated_at": generated_at,
        "params": params,
        "files": {
            "results_jsonl": _display_path(output_path),
            "artifacts_dir": _display_path(artifact_dir),
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
        "use_prompt_cache",
        "reasoning",
    ):
        lines.append(f"- `{key}`: `{params[key]}`")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- `results_jsonl`: `{_display_path(output_path)}`")
    lines.append(f"- `artifacts_dir`: `{_display_path(artifact_dir)}`")
    lines.append(f"- `summary_json`: `{_display_path(summary_json_path)}`")
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
            "| Runtime | Model | Case build (s) | Download/check (s) | Server start (s) | Warmup (s) | Warmup OK |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for row in setup_metrics:
            lines.append(
                f"| {row.get('runtime', '-')} | {row.get('model', '-')} | "
                f"{float(row.get('case_build_sec', 0.0) or 0.0):.3f} | "
                f"{float(row.get('download_or_check_sec', 0.0) or 0.0):.3f} | "
                f"{float(row.get('server_start_sec', 0.0) or 0.0):.3f} | "
                f"{float(row.get('warmup_sec', 0.0) or 0.0):.3f} | "
                f"{row.get('warmup_success', 'n/a')} |"
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
        "| Runtime | Model | Case | Context | Time (s) | Tok/s | Prompt tps | Gen tps | TTFT (s) | Peak RAM (GB) | Retrieval score | Retrieval exact | Prompt cache | Status |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
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
            f"{float(row.get('memory_gb', 0.0) or 0.0):.2f} | "
            f"{float(row.get('retrieval_score_float', 0.0) or 0.0):.3f} | "
            f"{bool(row.get('retrieval_exact', False))} | "
            f"{bool(row.get('used_prompt_cache', False))} | {status} |"
        )

    summary_md_path = target_dir / f"{summary_stem}.md"
    summary_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary_json_path, summary_md_path


def run_benchmark(args: Any) -> int:
    runtimes = _resolve_runtimes_for_args(args)

    results: List[Dict[str, Any]] = []
    setup_metrics: List[Dict[str, Any]] = []
    experiment_group = _experiment_group(args)
    experiment_stamp = _timestamp_slug()
    experiment_dir, results_dir = resolve_experiment_paths(
        root_dir=ROOT_DIR,
        experiment_group=experiment_group,
        experiment_stamp=experiment_stamp,
    )
    experiment_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    use_prompt_cache = bool(getattr(args, "use_prompt_cache", False))

    print("Managed mode: MLX servers run sequentially (never together) to avoid OOM.", file=sys.stderr)
    print(f"Artifacts directory: {experiment_dir}", file=sys.stderr)
    print(f"Results directory: {results_dir}", file=sys.stderr)

    for runtime in runtimes:
        config = CONFIGS[runtime]
        models = select_models(config, args.model, args.all_models)

        for model in models:
            tokenizer_model_id = model
            if runtime == "ollama":
                tokenizer_model_id = get_default_model_id(runtime="mlx")

            case_build_started = time.perf_counter()
            cases = build_cases(
                dataset=args.dataset,
                samples=args.samples,
                dataset_file=Path(args.dataset_file),
                prompt=args.prompt,
                prompt_max_tokens=args.max_tokens,
                contexts_k=getattr(args, "contexts_k", None),
                tokenizer_model_id=tokenizer_model_id,
            )
            case_build_sec = time.perf_counter() - case_build_started
            managed_proc: Optional[Any] = None
            memory_pid: Optional[int] = None
            default_context_k = _default_context_k_for_runtime(runtime)
            model_cache_ids: set[str] = set()
            setup_entry: Dict[str, Any] = {
                "runtime": runtime,
                "model": model,
                "case_build_sec": round(case_build_sec, 4),
                "download_or_check_sec": 0.0,
                "server_start_sec": 0.0,
                "warmup_sec": 0.0,
                "warmup_success": None,
                "warmup_status_code": None,
                "reasoning": str(getattr(args, "reasoning", "off")),
                "prompt_cache_enabled": bool(use_prompt_cache),
            }

            try:
                if config["managed_server"]:
                    stop_mlx_servers(verbose=True)
                    server_start_started = time.perf_counter()
                    managed_proc = start_managed_server(
                        runtime=runtime,
                        model=model,
                        timeout_sec=args.server_start_timeout,
                        root_dir=ROOT_DIR,
                        config=config,
                        log_dir=LOG_DIR,
                    )
                    setup_entry["server_start_sec"] = round(time.perf_counter() - server_start_started, 4)
                    memory_pid = managed_proc.pid
                    if not args.skip_warmup:
                        warmup_result = warmup_model(
                            config["chat_url"],
                            model,
                            reasoning_mode=str(getattr(args, "reasoning", "off")),
                        )
                        setup_entry["warmup_sec"] = float(warmup_result.get("warmup_sec", 0.0) or 0.0)
                        setup_entry["warmup_success"] = bool(warmup_result.get("success", False))
                        setup_entry["warmup_status_code"] = warmup_result.get("status_code")
                        time.sleep(1)
                else:
                    download_started = time.perf_counter()
                    if not ensure_model_downloaded(model, runtime):
                        print(f"Skipping {runtime}/{model} (download failed)", file=sys.stderr)
                        setup_entry["download_or_check_sec"] = round(time.perf_counter() - download_started, 4)
                        continue
                    setup_entry["download_or_check_sec"] = round(time.perf_counter() - download_started, 4)
                    if not args.skip_warmup:
                        warmup_result = warmup_model(
                            config["chat_url"],
                            model,
                            reasoning_mode=str(getattr(args, "reasoning", "off")),
                        )
                        setup_entry["warmup_sec"] = float(warmup_result.get("warmup_sec", 0.0) or 0.0)
                        setup_entry["warmup_success"] = bool(warmup_result.get("success", False))
                        setup_entry["warmup_status_code"] = warmup_result.get("status_code")
                        time.sleep(1)

                for case in cases:
                    print(f"Benchmarking {runtime}/{model} [{case['case_name']}]...", file=sys.stderr)
                    run_dir = experiment_dir / _run_param(
                        runtime,
                        model,
                        case,
                        default_context_k=default_context_k,
                    )
                    prompt_text = case["prompt"]
                    extra_payload: Optional[Dict[str, Any]] = None
                    if use_prompt_cache:
                        if runtime in {"mlx", "mlx-optiq"}:
                            prompt_prefix = case.get("prompt_prefix")
                            prompt_suffix = case.get("prompt_suffix")
                            prompt_cache_group = case.get("prompt_cache_group")
                            prefill_url = str(config.get("cache_prefill_url", "") or "")
                            if (
                                isinstance(prompt_prefix, str)
                                and isinstance(prompt_suffix, str)
                                and isinstance(prompt_cache_group, str)
                                and prefill_url
                            ):
                                cache_id = _slug(
                                    f"{runtime}-{_model_label(model, runtime)}-{prompt_cache_group}"
                                )
                                if cache_id not in model_cache_ids:
                                    prefill_result = prefill_prompt_cache(
                                        prefill_url=prefill_url,
                                        cache_id=cache_id,
                                        prompt_prefix=prompt_prefix,
                                        request_timeout_sec=args.request_timeout,
                                    )
                                    if not prefill_result.get("success"):
                                        print(
                                            f"  ! Cache prefill failed for {cache_id}: "
                                            f"{prefill_result.get('error', 'unknown')}. "
                                            "Falling back to full prompt.",
                                            file=sys.stderr,
                                        )
                                    else:
                                        model_cache_ids.add(cache_id)
                                if cache_id in model_cache_ids:
                                    prompt_text = prompt_suffix
                                    extra_payload = {
                                        "raw_prompt": prompt_suffix,
                                        "cache_id": cache_id,
                                    }
                        elif runtime == "ollama":
                            prompt_cache_group = case.get("prompt_cache_group")
                            cache_key: Optional[str] = None
                            if isinstance(prompt_cache_group, str) and prompt_cache_group:
                                cache_key = _slug(
                                    f"{runtime}-{_model_label(model, runtime)}-{prompt_cache_group}"
                                )
                            payload_patch: Dict[str, Any] = dict(extra_payload or {})
                            payload_patch["cache_prompt"] = True
                            options = payload_patch.get("options")
                            if not isinstance(options, dict):
                                options = {}
                            options["cache_prompt"] = True
                            payload_patch["options"] = options
                            if cache_key:
                                payload_patch["prompt_cache_key"] = cache_key
                            extra_payload = payload_patch

                    result = benchmark_model(
                        chat_url=config["chat_url"],
                        model=model,
                        prompt=prompt_text,
                        max_tokens=case["max_tokens"],
                        runtime=runtime,
                        memory_pid=memory_pid,
                        memory_pattern=config.get("process_hint"),
                        request_timeout_sec=args.request_timeout,
                        artifact_dir=run_dir,
                        extra_payload=extra_payload,
                        reasoning_mode=str(getattr(args, "reasoning", "off")),
                        enable_stream_metrics=True,
                    )
                    used_prompt_cache = bool(
                        extra_payload
                        and (
                            extra_payload.get("cache_id")
                            or extra_payload.get("cache_prompt")
                            or extra_payload.get("prompt_cache_key")
                        )
                    )
                    case_meta: Dict[str, Any] = {
                        "dataset": case.get("dataset"),
                        "case_name": case.get("case_name"),
                        "used_prompt_cache": used_prompt_cache,
                    }
                    for key in (
                        "context_tokens_target",
                        "payload_source",
                        "needle_key",
                        "needle_value",
                        "needle_position",
                        "prompt_cache_group",
                    ):
                        value = case.get(key)
                        if value is not None:
                            case_meta[key] = value
                    expected_needle = case_meta.get("needle_value")
                    if isinstance(expected_needle, str) and expected_needle:
                        predicted_raw = str(result.get("response_text", "") or "")
                        predicted_answer = _extract_retrieval_answer(predicted_raw)
                        expected_answer = _normalize_text(expected_needle)
                        retrieval_score = _retrieval_score(predicted_answer, expected_answer)
                        case_meta.update(
                            {
                                "retrieval_expected": expected_answer,
                                "retrieval_predicted": predicted_answer,
                                "retrieval_score_float": round(retrieval_score, 6),
                                "retrieval_exact": predicted_answer == expected_answer,
                            }
                        )
                    result.update(case_meta)
                    result.pop("response_text", None)
                    results.append(result)

                    if result.get("success"):
                        print(
                            f"  ✓ {result['tokens_per_second']} tok/s in {result['total_time']}s",
                            file=sys.stderr,
                        )
                    else:
                        print(f"  ✗ {result.get('error', 'Unknown error')}", file=sys.stderr)
            finally:
                setup_metrics.append(setup_entry)
                if runtime in {"mlx", "mlx-optiq"} and model_cache_ids:
                    clear_url = str(config.get("cache_clear_url", "") or "")
                    if clear_url:
                        for cache_id in sorted(model_cache_ids):
                            clear_result = clear_prompt_cache(
                                clear_url=clear_url,
                                cache_id=cache_id,
                            )
                            if not clear_result.get("success"):
                                print(
                                    f"  ! Cache clear failed for {cache_id}: "
                                    f"{clear_result.get('error', 'unknown')}",
                                    file=sys.stderr,
                                )
                if managed_proc is not None:
                    stop_managed_process(managed_proc)
                if config["managed_server"]:
                    stop_mlx_servers(verbose=False)

    if args.output:
        output_name = args.output
    else:
        output_name = str(results_dir / default_output_filename(args, runtimes, results))
    output_arg = Path(output_name)
    if not output_arg.is_absolute() and output_arg.parent == Path("."):
        output_arg = Path("results") / output_arg

    output_path = output_arg if output_arg.is_absolute() else (ROOT_DIR / output_arg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result) + "\n")

    print(f"\n✅ Results written to {output_path}", file=sys.stderr)
    summary_stem = default_summary_stem(args, runtimes, results)
    summary_json_path, summary_md_path = _write_summary_reports(
        args=args,
        results=results,
        runtimes=runtimes,
        setup_metrics=setup_metrics,
        output_path=output_path,
        artifact_dir=experiment_dir,
        summary_dir=results_dir,
        summary_stem=summary_stem,
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
            retrieval_rows = [r for r in runtime_results if isinstance(r.get("retrieval_score_float"), (float, int))]
            avg_retrieval = _avg([float(r.get("retrieval_score_float", 0.0) or 0.0) for r in retrieval_rows])
            exact_rate = _avg([1.0 if bool(r.get("retrieval_exact", False)) else 0.0 for r in retrieval_rows])
            exact_ci95 = _ci95_half_width_for_rate(exact_rate, len(retrieval_rows))
            print(
                f"  {runtime}: avg {avg_speed:.2f} tok/s, avg {avg_time:.2f}s, "
                f"retrieval {avg_retrieval:.3f}, exact {exact_rate:.3f} +/- {exact_ci95:.3f}",
                file=sys.stderr,
            )

    return 0
