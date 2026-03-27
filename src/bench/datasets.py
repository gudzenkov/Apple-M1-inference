from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, List

SHORT_CONTEXT_TOKENS = 8000
LONG_DEFAULT_CONTEXT_TOKENS = 256000
SHORT_MAX_TOKENS = 64
LONG_MAX_TOKENS = 128
CONTEXT_FILL_RATIO = 0.7
ALLOWED_CONTEXT_K = (8, 16, 32, 64, 128, 256)
BENCH_PROMPTS_PATH = Path("dataset/bench_prompts.json")
LEGACY_DATASET_ALIASES = {
    "quick": "short",
    "context": "long",
}


def normalize_dataset_mode(dataset: str) -> str:
    return LEGACY_DATASET_ALIASES.get(dataset, dataset)


def _build_payload(source_words: List[str], target_words: int, offset: int) -> List[str]:
    if not source_words:
        raise RuntimeError("Cannot build payload from empty source words")
    if target_words <= 0:
        raise RuntimeError("target_words must be > 0")
    n = len(source_words)
    shift = offset % n
    rotated = source_words[shift:] + source_words[:shift]
    repeated = rotated * (target_words // n + 1)
    return repeated[:target_words]


def _load_words(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()


@lru_cache(maxsize=1)
def _load_bench_prompts(path: Path = BENCH_PROMPTS_PATH) -> Dict[str, str]:
    if not path.exists():
        raise RuntimeError(f"Benchmark prompts file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    required_keys = (
        "short_source",
        "short_intro",
        "short_query",
        "long_intro",
        "long_query",
        "needle_template",
        "answer_format",
    )
    prompts: Dict[str, str] = {}
    for key in required_keys:
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                f"Benchmark prompts file {path} is missing non-empty '{key}'"
            )
        prompts[key] = value.strip()
    return prompts


def parse_context_list(raw_value: str) -> List[int]:
    if not raw_value.strip():
        raise RuntimeError("--context value is empty")

    contexts: List[int] = []
    for part in raw_value.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.endswith("k"):
            token = token[:-1]
        if not token.isdigit():
            raise RuntimeError(
                f"Invalid context value '{part}'. Use comma-separated values from: "
                + ", ".join(f"{v}k" for v in ALLOWED_CONTEXT_K)
            )
        value = int(token)
        if value not in ALLOWED_CONTEXT_K:
            raise RuntimeError(
                f"Unsupported context '{value}k'. Allowed: "
                + ", ".join(f"{v}k" for v in ALLOWED_CONTEXT_K)
            )
        if value not in contexts:
            contexts.append(value)

    if not contexts:
        raise RuntimeError("--context produced no values")

    return contexts


def _needle_fields(mode: str, context_k: int, sample_index: int) -> Dict[str, str]:
    tag = mode.upper()
    checksum = (context_k * 1009) + (sample_index * 917) + (13 if mode == "short" else 29)
    needle_key = f"{mode}-needle-{context_k}k-s{sample_index}"
    needle_value = f"NIAH-{tag}-{context_k}K-S{sample_index:02d}-{(checksum % 1_000_000):06d}"
    return {
        "needle_key": needle_key,
        "needle_value": needle_value,
    }


def _embed_needle(payload_words: List[str], needle_text: str, position: int) -> tuple[List[str], int]:
    needle_words = needle_text.split()
    if not needle_words:
        return list(payload_words), 0

    if not payload_words:
        return needle_words, 0

    max_pos = max(0, len(payload_words) - len(needle_words))
    pos = min(max(position, 0), max_pos)
    mutated = list(payload_words)
    mutated[pos:pos + len(needle_words)] = needle_words
    return mutated, pos


def _render_needle(prompts: Dict[str, str], needle_key: str, needle_value: str) -> str:
    try:
        return prompts["needle_template"].format(
            needle_key=needle_key,
            needle_value=needle_value,
        )
    except KeyError as exc:
        raise RuntimeError(
            f"Invalid needle_template placeholder: {exc}. "
            "Allowed placeholders: {needle_key}, {needle_value}"
        )


def _render_query(template: str, needle_key: str) -> str:
    try:
        return template.format(needle_key=needle_key)
    except KeyError as exc:
        raise RuntimeError(
            f"Invalid query placeholder: {exc}. Allowed placeholders: {needle_key}"
        )


def build_short_cases(samples: int) -> List[Dict[str, Any]]:
    prompts = _load_bench_prompts()
    target_words = max(1, int(SHORT_CONTEXT_TOKENS * CONTEXT_FILL_RATIO))
    source_words = prompts["short_source"].split()

    cases: List[Dict[str, Any]] = []
    context_k = int(SHORT_CONTEXT_TOKENS / 1000)
    for idx in range(samples):
        sample_index = idx + 1
        payload_words = _build_payload(source_words, target_words=target_words, offset=idx * 257)
        needle = _needle_fields(mode="short", context_k=context_k, sample_index=sample_index)
        needle_text = _render_needle(prompts, needle["needle_key"], needle["needle_value"])
        payload_words, needle_position = _embed_needle(
            payload_words=payload_words,
            needle_text=needle_text,
            position=(context_k * 173) + (idx * 331),
        )
        payload = " ".join(payload_words)
        query = _render_query(prompts["short_query"], needle["needle_key"])
        prompt = (
            f"[short-8k sample {sample_index}/{samples}] "
            f"{prompts['short_intro']}\n\n"
            f"{payload}\n\n"
            f"Question: {query}\n"
            f"{prompts['answer_format']}"
        )
        cases.append(
            {
                "dataset": "short",
                "case_name": f"short-8k-{sample_index}",
                "prompt": prompt,
                "max_tokens": SHORT_MAX_TOKENS,
                "context_tokens_target": SHORT_CONTEXT_TOKENS,
                "payload_source": "short-source",
                "needle_key": needle["needle_key"],
                "needle_value": needle["needle_value"],
                "needle_position": needle_position,
            }
        )
    return cases


def build_long_cases(
    samples: int,
    dataset_file: Path,
    contexts_k: List[int] | None = None,
) -> List[Dict[str, Any]]:
    if not dataset_file.exists():
        raise RuntimeError(
            f"Dataset file not found: {dataset_file}. "
            "Run 'uv run dataset fetch-parse' first."
        )

    source_words = _load_words(dataset_file)
    prompts = _load_bench_prompts()
    if len(source_words) < 100:
        raise RuntimeError(f"Dataset markdown is too short: {dataset_file}")

    resolved_contexts = contexts_k or [int(LONG_DEFAULT_CONTEXT_TOKENS / 1000)]
    cases: List[Dict[str, Any]] = []
    for context_k in resolved_contexts:
        context_tokens = context_k * 1000
        target_words = max(1, int(context_tokens * CONTEXT_FILL_RATIO))
        shared_payload_words = _build_payload(
            source_words,
            target_words=target_words,
            offset=(context_k * 37),
        )
        context_needles: List[Dict[str, Any]] = []
        for idx in range(samples):
            sample_index = idx + 1
            needle = _needle_fields(mode="long", context_k=context_k, sample_index=sample_index)
            needle_text = _render_needle(prompts, needle["needle_key"], needle["needle_value"])
            shared_payload_words, needle_position = _embed_needle(
                payload_words=shared_payload_words,
                needle_text=needle_text,
                position=(context_k * 173) + (idx * 331),
            )
            context_needles.append(
                {
                    **needle,
                    "sample_index": sample_index,
                    "needle_position": needle_position,
                }
            )

        shared_payload = " ".join(shared_payload_words)
        prompt_prefix = (
            f"[long-{context_k}k shared-context needles {samples}] "
            f"{prompts['long_intro']}\n\n"
            f"{shared_payload}\n\n"
        )
        prompt_cache_group = f"long-{context_k}k"
        for needle_meta in context_needles:
            sample_index = int(needle_meta["sample_index"])
            query = _render_query(prompts["long_query"], str(needle_meta["needle_key"]))
            prompt_suffix = (
                f"Question: {query}\n"
                f"{prompts['answer_format']}"
            )
            prompt = (
                f"{prompt_prefix}"
                f"{prompt_suffix}"
            )
            cases.append(
                {
                    "dataset": "long",
                    "case_name": f"long-{context_k}k-{sample_index}",
                    "prompt": prompt,
                    "prompt_prefix": prompt_prefix,
                    "prompt_suffix": prompt_suffix,
                    "prompt_cache_group": prompt_cache_group,
                    "max_tokens": LONG_MAX_TOKENS,
                    "context_tokens_target": context_tokens,
                    "payload_source": str(dataset_file),
                    "needle_key": str(needle_meta["needle_key"]),
                    "needle_value": str(needle_meta["needle_value"]),
                    "needle_position": int(needle_meta["needle_position"]),
                }
            )
    return cases


def build_cases(
    dataset: str,
    samples: int,
    dataset_file: Path,
    prompt: str | None,
    prompt_max_tokens: int,
    contexts_k: List[int] | None = None,
) -> List[Dict[str, Any]]:
    if prompt:
        return [
            {
                "dataset": "custom",
                "case_name": "custom-1",
                "prompt": prompt,
                "max_tokens": prompt_max_tokens,
                "context_tokens_target": None,
                "payload_source": None,
            }
        ]

    dataset_mode = normalize_dataset_mode(dataset)

    cases: List[Dict[str, Any]] = []
    if dataset_mode in ("short", "all"):
        cases.extend(build_short_cases(samples=samples))
    if dataset_mode in ("long", "all"):
        cases.extend(build_long_cases(samples=samples, dataset_file=dataset_file, contexts_k=contexts_k))

    if not cases:
        raise RuntimeError(f"Unsupported dataset mode: {dataset_mode}")

    return cases
