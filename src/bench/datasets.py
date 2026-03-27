from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List

QUICK_CONTEXT_TOKENS = 32000
LONG_CONTEXT_TOKENS = 256000
QUICK_MAX_TOKENS = 64
LONG_MAX_TOKENS = 128
CONTEXT_FILL_RATIO = 0.7
QUICK_SEED_TEXT = (
    "Local inference benchmarking focuses on latency, throughput, memory, "
    "context handling, and response stability across repeated samples. "
    "Use deterministic prompts and consistent runtime configuration for fair comparisons."
)


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


def build_quick_cases(samples: int) -> List[Dict[str, Any]]:
    target_words = max(1, int(QUICK_CONTEXT_TOKENS * CONTEXT_FILL_RATIO))
    source_words = QUICK_SEED_TEXT.split()

    cases: List[Dict[str, Any]] = []
    for idx in range(samples):
        sample_index = idx + 1
        payload = _build_payload(source_words, target_words=target_words, offset=idx * 257)
        prompt = (
            f"[quick-32k sample {sample_index}/{samples}] "
            "Read the context and answer briefly.\n\n"
            f"{payload}\n\n"
            "Question: Summarize the benchmark intent in <= 20 words."
        )
        cases.append(
            {
                "dataset": "quick",
                "case_name": f"quick-32k-{sample_index}",
                "prompt": prompt,
                "max_tokens": QUICK_MAX_TOKENS,
                "context_tokens_target": QUICK_CONTEXT_TOKENS,
                "payload_words": target_words,
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
    if len(source_words) < 100:
        raise RuntimeError(f"Dataset markdown is too short: {dataset_file}")

    target_words = max(1, int(LONG_CONTEXT_TOKENS * CONTEXT_FILL_RATIO))

    cases: List[Dict[str, Any]] = []
    for idx in range(samples):
        sample_index = idx + 1
        payload = _build_payload(source_words, target_words=target_words, offset=idx * 997)
        prompt = (
            f"[long-256k sample {sample_index}/{samples}] "
            "The context below is extracted from the TurboQuant paper.\n\n"
            f"{payload}\n\n"
            "Question: List 3 key findings from this paper context."
        )
        cases.append(
            {
                "dataset": "long",
                "case_name": f"long-256k-{sample_index}",
                "prompt": prompt,
                "max_tokens": LONG_MAX_TOKENS,
                "context_tokens_target": LONG_CONTEXT_TOKENS,
                "payload_words": target_words,
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
) -> List[Dict[str, Any]]:
    if prompt:
        return [
            {
                "dataset": "custom",
                "case_name": "custom-1",
                "prompt": prompt,
                "max_tokens": prompt_max_tokens,
                "context_tokens_target": None,
                "payload_words": None,
                "payload_source": None,
            }
        ]

    cases: List[Dict[str, Any]] = []
    if dataset in ("quick", "all"):
        cases.extend(build_quick_cases(samples=samples))
    if dataset in ("long", "all"):
        cases.extend(build_long_cases(samples=samples, dataset_file=dataset_file))

    if not cases:
        raise RuntimeError(f"Unsupported dataset mode: {dataset}")

    return cases
