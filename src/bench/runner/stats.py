from __future__ import annotations

import math
from typing import Any, Dict, Optional


def avg(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def ci95_half_width_for_rate(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p))
    return 1.96 * math.sqrt((p * (1.0 - p)) / float(n))


def to_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:  # noqa: BLE001
        return 0.0


def row_total_time(row: Dict[str, Any]) -> float:
    timing = row.get("timing")
    if isinstance(timing, dict):
        client = timing.get("client")
        if isinstance(client, dict):
            return to_float(client.get("total_time_sec"))
    return 0.0


def row_ttft(row: Dict[str, Any]) -> float:
    timing = row.get("timing")
    if isinstance(timing, dict):
        server = timing.get("server")
        if isinstance(server, dict):
            server_ttft = to_float(server.get("ttft_sec"))
            if server_ttft > 0:
                return server_ttft
        client = timing.get("client")
        if isinstance(client, dict):
            client_ttft = to_float(client.get("ttft_sec"))
            if client_ttft > 0:
                return client_ttft
    return 0.0


def row_server_prompt_eval_sec(row: Dict[str, Any]) -> float:
    timing = row.get("timing")
    if isinstance(timing, dict):
        server = timing.get("server")
        if isinstance(server, dict):
            value = to_float(server.get("prompt_eval_sec"))
            if value > 0:
                return value
    return 0.0


def row_cache_prefill_sec(row: Dict[str, Any]) -> float:
    timing = row.get("timing")
    if isinstance(timing, dict):
        cache = timing.get("cache")
        if isinstance(cache, dict):
            value = to_float(cache.get("prefill_sec"))
            if value > 0:
                return value
    return 0.0


def row_prefill_sec(row: Dict[str, Any]) -> float:
    explicit_prefill_sec = row_cache_prefill_sec(row)
    if explicit_prefill_sec > 0:
        return explicit_prefill_sec
    phase = str(row.get("phase") or "").strip().lower()
    if phase == "cache-prime":
        return row_total_time(row)
    prompt_eval_sec = row_server_prompt_eval_sec(row)
    if prompt_eval_sec > 0:
        return prompt_eval_sec
    return 0.0


def row_prompt_tokens(row: Dict[str, Any]) -> float:
    usage = row.get("usage")
    if isinstance(usage, dict):
        normalized = usage.get("normalized")
        if isinstance(normalized, dict):
            tokens = to_float(normalized.get("prompt_tokens"))
            if tokens > 0:
                return tokens
        server = usage.get("server")
        if isinstance(server, dict):
            tokens = to_float(server.get("prompt_tokens"))
            if tokens > 0:
                return tokens
    return 0.0


def row_peak_memory(row: Dict[str, Any]) -> float:
    memory = row.get("memory")
    if isinstance(memory, dict):
        return to_float(memory.get("peak_gb"))
    return 0.0


def row_throughput(row: Dict[str, Any], key: str) -> float:
    throughput = row.get("throughput")
    if isinstance(throughput, dict):
        return to_float(throughput.get(key))
    return 0.0


def row_retrieval_score(row: Dict[str, Any]) -> Optional[float]:
    retrieval = row.get("retrieval")
    if isinstance(retrieval, dict):
        value = retrieval.get("score_float")
        if isinstance(value, (int, float)):
            return float(value)
    return None


def row_retrieval_exact(row: Dict[str, Any]) -> Optional[bool]:
    retrieval = row.get("retrieval")
    if isinstance(retrieval, dict):
        value = retrieval.get("exact")
        if isinstance(value, bool):
            return value
    return None
