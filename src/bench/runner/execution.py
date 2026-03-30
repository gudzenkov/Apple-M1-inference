from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict

from src.bench.composition import compose_benchmark_spec, default_context_k_for_runtime, select_models
from src.bench.dataset.cases import build_cases
from src.bench.handlers import get_runtime_handler
from src.bench.runner.naming import run_param
from src.bench.runner.retrieval import annotate_retrieval
from src.bench.runner.stats import row_throughput, row_total_time
from src.shared.models import get_default_model_id


def run_runtime_matrix(
    *,
    args: Any,
    runtimes: list[str],
    root_dir: Path,
    log_dir: Path,
    experiment_dir: Path,
) -> tuple[list[Dict[str, Any]], list[Dict[str, Any]]]:
    results: list[Dict[str, Any]] = []
    setup_metrics: list[Dict[str, Any]] = []

    for runtime in runtimes:
        models = select_models(runtime, args.model, args.all_models)
        handler = get_runtime_handler(runtime)

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

            try:
                spec = compose_benchmark_spec(runtime=runtime, model=model, args=args)
            except Exception as exc:  # noqa: BLE001
                error_msg = str(exc)
                print(f"Skipping {runtime}/{model}: {error_msg}", file=sys.stderr)
                setup_metrics.append(
                    {
                        "runtime": runtime,
                        "model": model,
                        "case_build_sec": round(case_build_sec, 4),
                        "download_or_check_sec": 0.0,
                        "server_start_sec": 0.0,
                        "warmup_sec": 0.0,
                        "warmup_success": None,
                        "warmup_status_code": None,
                        "setup_error": error_msg,
                    }
                )
                continue

            default_context_k = default_context_k_for_runtime(runtime)
            setup_entry, state = handler.setup_model(
                spec=spec,
                args=args,
                root_dir=root_dir,
                log_dir=log_dir,
                case_build_sec=case_build_sec,
            )

            try:
                if state.setup_failed:
                    print(
                        f"Skipping {runtime}/{spec.model}: {state.setup_error or 'setup failed'}",
                        file=sys.stderr,
                    )
                    continue

                for case in cases:
                    print(f"Benchmarking {runtime}/{spec.model} [{case['case_name']}]...", file=sys.stderr)
                    run_dir = experiment_dir / run_param(
                        runtime,
                        spec.model,
                        case,
                        default_context_k=default_context_k,
                    )

                    result = handler.run_case(
                        spec=spec,
                        args=args,
                        case=case,
                        run_dir=run_dir,
                        state=state,
                    )

                    case_meta: Dict[str, Any] = {
                        "dataset": case.get("dataset"),
                        "case_name": case.get("case_name"),
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

                    annotate_retrieval(result, case_meta.get("needle_value"))

                    result.update(case_meta)
                    result.pop("response_text", None)
                    results.append(result)

                    if result.get("success"):
                        tps = row_throughput(result, "tokens_per_second")
                        total = row_total_time(result)
                        print(f"  ✓ {tps:.2f} tok/s in {total:.2f}s", file=sys.stderr)
                    else:
                        print(f"  ✗ {result.get('error', 'Unknown error')}", file=sys.stderr)
            finally:
                setup_metrics.append(setup_entry)
                handler.teardown_model(spec=spec, state=state)

    return results, setup_metrics
