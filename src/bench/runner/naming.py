from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from src.bench.utils.text import slug


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def dataset_name_from_file(dataset_file: str) -> str:
    stem = Path(dataset_file).stem.lower()
    if not stem:
        return "dataset"
    parts = re.split(r"[_\-]+", stem)
    if parts and parts[0]:
        return slug(parts[0])
    return slug(stem)


def experiment_group(args: Any) -> str:
    runtime_part = slug(str(args.runtime))
    if args.prompt:
        dataset_part = "prompt"
    elif getattr(args, "contexts_k", None):
        dataset_part = dataset_name_from_file(str(args.dataset_file))
    elif str(args.dataset) in {"long", "all"}:
        dataset_part = dataset_name_from_file(str(args.dataset_file))
    else:
        dataset_part = "short"
    cache_part = slug(f"cache-{getattr(args, 'cache_mode', 'auto')}")
    return slug(f"{runtime_part}-{dataset_part}-s{args.samples}-mt{args.max_tokens}-{cache_part}")


def sample_id(case_name: str) -> str:
    match = re.search(r"-(\d+)$", case_name)
    if match:
        return match.group(1)
    return "1"


def context_label(case: Dict[str, Any], default_context_k: int) -> str:
    tokens = case.get("context_tokens_target")
    if isinstance(tokens, int) and tokens > 0:
        return f"{int(tokens / 1000)}k"
    return f"{default_context_k}k"


def model_label(model_id: str, runtime: str) -> str:
    _ = runtime
    return slug(model_id)


def run_param(runtime: str, model: str, case: Dict[str, Any], default_context_k: int) -> str:
    model_part = model_label(model, runtime)
    context_part = context_label(case, default_context_k=default_context_k)
    sample_part = sample_id(str(case.get("case_name", "s1")))
    return slug(f"{runtime}-{model_part}-{context_part}-s{sample_part}")
