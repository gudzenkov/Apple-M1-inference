from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
import re
from typing import Any, Dict, List

QUICK_CONTEXT_TOKENS = 8000
LONG_CONTEXT_TOKENS = 256000
QUICK_MAX_TOKENS = 64
LONG_MAX_TOKENS = 128
CONTEXT_FILL_RATIO = 0.7
ALLOWED_CONTEXT_K = (8, 16, 32, 64, 128, 256)
BENCH_PROMPTS_PATH = Path("dataset/bench_prompts.json")


def _build_payload(source_words: List[str], target_words: int, offset: int) -> str:
    if not source_words:
        raise RuntimeError("Cannot build payload from empty source words")
    if target_words <= 0:
        raise RuntimeError("target_words must be > 0")
    n = len(source_words)
    shift = offset % n
    rotated = source_words[shift:] + source_words[:shift]
    repeated = rotated * (target_words // n + 1)
    return " ".join(repeated[:target_words])


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
        "quick_seed",
        "quick_instruction",
        "quick_question",
        "long_intro",
        "long_question",
        "context_intro",
        "context_question",
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


def _load_abstract_words(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"\s+", " ", text).strip()
    best_abstract = ""
    abstract_iter = re.finditer(r"\bAbstract\b", text, flags=re.IGNORECASE)
    for match in abstract_iter:
        tail = text[match.end():]
        intro = re.search(r"\b1(?:\.0)?\s+Introduction\b", tail, flags=re.IGNORECASE)
        if not intro:
            continue
        candidate = tail[:intro.start()].strip()
        if len(candidate.split()) > len(best_abstract.split()):
            best_abstract = candidate

    if best_abstract:
        return best_abstract.split()
    return text.split()


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


def build_context_cases(samples: int, dataset_file: Path, contexts_k: List[int]) -> List[Dict[str, Any]]:
    if not dataset_file.exists():
        raise RuntimeError(
            f"Dataset file not found: {dataset_file}. "
            "Run 'uv run dataset fetch-parse' first."
        )
    source_words = _load_abstract_words(dataset_file)
    prompts = _load_bench_prompts()
    if len(source_words) < 50:
        raise RuntimeError(
            f"Dataset abstract text is too short in {dataset_file}. "
            "Ensure dataset file contains the paper content."
        )

    cases: List[Dict[str, Any]] = []
    for context_k in contexts_k:
        context_tokens = context_k * 1000
        target_words = max(1, int(context_tokens * CONTEXT_FILL_RATIO))
        for idx in range(samples):
            sample_index = idx + 1
            payload = _build_payload(
                source_words,
                target_words=target_words,
                offset=(context_k * 37) + (idx * 997),
            )
            prompt = (
                f"[context-{context_k}k sample {sample_index}/{samples}] "
                f"{prompts['context_intro']}\n\n"
                f"{payload}\n\n"
                f"Question: {prompts['context_question']}"
            )
            cases.append(
                {
                    "dataset": "context",
                    "case_name": f"context-{context_k}k-{sample_index}",
                    "prompt": prompt,
                    "max_tokens": QUICK_MAX_TOKENS,
                    "context_tokens_target": context_tokens,
                    "payload_source": f"{dataset_file}:abstract",
                }
            )
    return cases


def build_quick_cases(samples: int) -> List[Dict[str, Any]]:
    prompts = _load_bench_prompts()
    target_words = max(1, int(QUICK_CONTEXT_TOKENS * CONTEXT_FILL_RATIO))
    source_words = prompts["quick_seed"].split()

    cases: List[Dict[str, Any]] = []
    for idx in range(samples):
        sample_index = idx + 1
        payload = _build_payload(source_words, target_words=target_words, offset=idx * 257)
        prompt = (
            f"[quick-8k sample {sample_index}/{samples}] "
            f"{prompts['quick_instruction']}\n\n"
            f"{payload}\n\n"
            f"Question: {prompts['quick_question']}"
        )
        cases.append(
            {
                "dataset": "quick",
                "case_name": f"quick-8k-{sample_index}",
                "prompt": prompt,
                "max_tokens": QUICK_MAX_TOKENS,
                "context_tokens_target": QUICK_CONTEXT_TOKENS,
                "payload_source": "quick-seed",
            }
        )
    return cases


def build_long_cases(samples: int, dataset_file: Path) -> List[Dict[str, Any]]:
    if not dataset_file.exists():
        raise RuntimeError(
            f"Dataset file not found: {dataset_file}. "
            "Run 'uv run dataset fetch-parse' first."
        )

    source_words = _load_words(dataset_file)
    prompts = _load_bench_prompts()
    if len(source_words) < 100:
        raise RuntimeError(f"Dataset markdown is too short: {dataset_file}")

    target_words = max(1, int(LONG_CONTEXT_TOKENS * CONTEXT_FILL_RATIO))

    cases: List[Dict[str, Any]] = []
    for idx in range(samples):
        sample_index = idx + 1
        payload = _build_payload(source_words, target_words=target_words, offset=idx * 997)
        prompt = (
            f"[long-256k sample {sample_index}/{samples}] "
            f"{prompts['long_intro']}\n\n"
            f"{payload}\n\n"
            f"Question: {prompts['long_question']}"
        )
        cases.append(
            {
                "dataset": "long",
                "case_name": f"long-256k-{sample_index}",
                "prompt": prompt,
                "max_tokens": LONG_MAX_TOKENS,
                "context_tokens_target": LONG_CONTEXT_TOKENS,
                "payload_source": str(dataset_file),
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

    if contexts_k:
        return build_context_cases(
            samples=samples,
            dataset_file=dataset_file,
            contexts_k=contexts_k,
        )

    cases: List[Dict[str, Any]] = []
    if dataset in ("quick", "all"):
        cases.extend(build_quick_cases(samples=samples))
    if dataset in ("long", "all"):
        cases.extend(build_long_cases(samples=samples, dataset_file=dataset_file))

    if not cases:
        raise RuntimeError(f"Unsupported dataset mode: {dataset}")

    return cases
