from __future__ import annotations

import re
from typing import Any, Dict, Optional


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip())


def extract_retrieval_answer(raw_text: str) -> str:
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
        return normalize_text(candidate)
    return normalize_text(raw_text.strip(" \t\r\n\"'`.,;:!?"))


def levenshtein_distance(a: str, b: str) -> int:
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


def retrieval_score(predicted: str, expected: str) -> float:
    max_len = max(len(predicted), len(expected), 1)
    dist = levenshtein_distance(predicted, expected)
    return max(0.0, 1.0 - (float(dist) / float(max_len)))


def annotate_retrieval(result: Dict[str, Any], expected_needle: Optional[str]) -> None:
    if not isinstance(expected_needle, str) or not expected_needle:
        return
    predicted_raw = str(result.get("response_text", "") or "")
    predicted_answer = extract_retrieval_answer(predicted_raw)
    expected_answer = normalize_text(expected_needle)
    score = retrieval_score(predicted_answer, expected_answer)
    result["retrieval"] = {
        "expected": expected_answer,
        "predicted": predicted_answer,
        "score_float": round(score, 6),
        "exact": predicted_answer == expected_answer,
    }
