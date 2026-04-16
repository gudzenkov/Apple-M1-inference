from __future__ import annotations

import re
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


def _case_meta_from_case(case: Dict[str, Any]) -> Dict[str, Any]:
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
    return case_meta


def _prime_case_name(case_name: str) -> str:
    if not case_name:
        return "cache-prime-0"
    if re.search(r"-\d+$", case_name):
        return re.sub(r"-\d+$", "-0", case_name)
    return f"{case_name}-0"


def _cache_prime_key(case: Dict[str, Any]) -> str:
    prompt_cache_group = case.get("prompt_cache_group")
    if isinstance(prompt_cache_group, str) and prompt_cache_group.strip():
        return prompt_cache_group.strip()
    dataset = str(case.get("dataset") or "dataset")
    context = case.get("context_tokens_target")
    context_part = str(context) if isinstance(context, int) and context > 0 else "default"
    return f"{dataset}:{context_part}"


def _build_prime_case(case: Dict[str, Any], *, cache_mode: str) -> Dict[str, Any]:
    prime_case = dict(case)
    prime_case["case_name"] = _prime_case_name(str(case.get("case_name") or "cache-prime"))
    prime_case["phase"] = "cache-prime"

    # MLX prefill needs the real first suffix to preserve retrieval behavior.
    if cache_mode == "prefill":
        return prime_case

    try:
        prime_case["max_tokens"] = min(int(case.get("max_tokens", 16)), 8)
    except Exception:  # noqa: BLE001
        prime_case["max_tokens"] = 8

    prompt_prefix = case.get("prompt_prefix")
    if isinstance(prompt_prefix, str) and prompt_prefix:
        prime_suffix = (
            "Question: Return exactly CACHE-PRIME-ONLY\n"
            "Answer format: CACHE-PRIME-ONLY"
        )
        prime_case["prompt_suffix"] = prime_suffix
        prime_case["prompt"] = f"{prompt_prefix}{prime_suffix}"
    else:
        base_prompt = str(case.get("prompt") or "")
        prime_case["prompt"] = f"{base_prompt}\n\n[cache-prime-only]"

    prime_case.pop("needle_key", None)
    prime_case.pop("needle_value", None)
    prime_case.pop("needle_position", None)
    return prime_case


def _build_cache_prime_cases(cases: list[Dict[str, Any]], cache_mode: str) -> list[Dict[str, Any]]:
    if cache_mode == "none":
        return []
    prime_cases: list[Dict[str, Any]] = []
    seen_keys: set[str] = set()
    for case in cases:
        key = _cache_prime_key(case)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        prime_case = _build_prime_case(case, cache_mode=cache_mode)
        prime_cases.append(prime_case)
    return prime_cases


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

                prime_cases = _build_cache_prime_cases(cases, cache_mode=spec.cache_mode)
                for prime_case in prime_cases:
                    if state.fatal_error:
                        print(
                            f"Skipping cache priming for {runtime}/{spec.model}: {state.fatal_error}",
                            file=sys.stderr,
                        )
                        break
                    print(f"Priming cache {runtime}/{spec.model} [{prime_case['case_name']}]...", file=sys.stderr)
                    run_dir = experiment_dir / run_param(
                        runtime,
                        spec.model,
                        prime_case,
                        default_context_k=default_context_k,
                    )
                    prime_result = handler.run_case(
                        spec=spec,
                        args=args,
                        case=prime_case,
                        run_dir=run_dir,
                        state=state,
                    )
                    prime_result.update(_case_meta_from_case(prime_case))
                    prime_result["phase"] = "cache-prime"
                    prime_result["benchmark_included"] = False
                    prime_result.pop("response_text", None)
                    results.append(prime_result)
                    if prime_result.get("success"):
                        tps = row_throughput(prime_result, "prompt_tps")
                        total = row_total_time(prime_result)
                        print(f"  ↺ cache prime prompt {tps:.2f} tps in {total:.2f}s", file=sys.stderr)
                    else:
                        print(f"  ✗ cache prime failed: {prime_result.get('error', 'Unknown error')}", file=sys.stderr)

                for case in cases:
                    if state.fatal_error:
                        print(f"Skipping remaining cases for {runtime}/{spec.model}: {state.fatal_error}", file=sys.stderr)
                        break
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

                    case_meta = _case_meta_from_case(case)

                    annotate_retrieval(result, case_meta.get("needle_value"))

                    result.update(case_meta)
                    result["phase"] = "benchmark"
                    result["benchmark_included"] = True
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
