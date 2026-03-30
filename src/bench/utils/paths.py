from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.bench.utils.text import slug


def _model_label(model_id: str, runtime: str) -> str:
    _ = runtime
    return slug(model_id)


def _strip_quant_suffix(model_key: str) -> str:
    return re.sub(r"-q\d+$", "", model_key)


def _primary_runtime_for_naming(runtimes: List[str], results: List[Dict[str, Any]]) -> str:
    for runtime in runtimes:
        if any(r.get("runtime") == runtime for r in results):
            return runtime
    return runtimes[0] if runtimes else "runtime"


def _primary_model_for_runtime(results: List[Dict[str, Any]], runtime: str) -> Optional[str]:
    for row in results:
        if row.get("runtime") == runtime:
            model = row.get("model")
            if isinstance(model, str) and model.strip():
                return model
    return None


def _context_part_for_naming(args: Any, results: List[Dict[str, Any]], runtime: str) -> str:
    contexts_from_args = getattr(args, "contexts_k", None)
    if isinstance(contexts_from_args, list) and contexts_from_args:
        if len(contexts_from_args) == 1:
            return f"{contexts_from_args[0]}k"
        return "-".join(f"{context_k}k" for context_k in contexts_from_args)

    contexts: List[int] = sorted(
        {
            int(row["context_tokens_target"] / 1000)
            for row in results
            if row.get("runtime") == runtime and isinstance(row.get("context_tokens_target"), int)
        }
    )
    if len(contexts) == 1:
        return f"{contexts[0]}k"
    if contexts:
        return "-".join(f"{context_k}k" for context_k in contexts)
    return slug(str(getattr(args, "dataset", "dataset")))


def default_output_filename(args: Any, runtimes: List[str], results: List[Dict[str, Any]]) -> str:
    runtime = _primary_runtime_for_naming(runtimes, results)
    model_id = _primary_model_for_runtime(results, runtime)
    if model_id:
        model_part = _strip_quant_suffix(_model_label(model_id, runtime))
        context_part = _context_part_for_naming(args, results, runtime)
        return slug(f"benchmark_{model_part}_{context_part}_s{args.samples}") + ".jsonl"

    if getattr(args, "contexts_k", None):
        context_slug = "-".join(f"{c}k" for c in args.contexts_k)
        return f"benchmark_context_{context_slug}.jsonl"
    return f"benchmark_{args.dataset}.jsonl"


def default_summary_stem(args: Any, runtimes: List[str], results: List[Dict[str, Any]]) -> str:
    runtime = _primary_runtime_for_naming(runtimes, results)
    model_id = _primary_model_for_runtime(results, runtime)
    context_part = _context_part_for_naming(args, results, runtime)
    if model_id:
        model_part = _model_label(model_id, runtime)
    else:
        model_part = "model"
    return slug(f"{runtime}-{model_part}-{context_part}-s{args.samples}")


def resolve_experiment_paths(root_dir: Path, experiment_group: str, experiment_stamp: str) -> tuple[Path, Path]:
    experiment_rel_path = Path(experiment_group) / experiment_stamp
    data_dir = root_dir / "data" / "benchmark" / experiment_rel_path
    results_dir = root_dir / "results" / experiment_rel_path
    return data_dir, results_dir
