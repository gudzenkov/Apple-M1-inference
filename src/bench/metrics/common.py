from __future__ import annotations

import json
from pathlib import Path
import subprocess
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

import requests

from src.bench.process import pid_alive, pids_for_pattern


def get_rss_kb_by_pid(pid: int) -> Optional[int]:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        value = result.stdout.strip()
        if not value:
            return None
        return int(float(value))
    except Exception:  # noqa: BLE001
        return None


def sample_rss_gb(pid: Optional[int] = None, process_pattern: Optional[str] = None) -> Optional[float]:
    pids: Set[int] = set()
    if pid is not None and pid_alive(pid):
        pids.add(pid)
    if process_pattern:
        pids |= pids_for_pattern(process_pattern)

    total_kb = 0
    for proc_pid in pids:
        rss_kb = get_rss_kb_by_pid(proc_pid)
        if rss_kb is not None:
            total_kb += rss_kb

    if total_kb <= 0:
        return None
    return round(total_kb / 1024 / 1024, 4)


class MemoryMonitor:
    def __init__(
        self,
        pid: Optional[int] = None,
        process_pattern: Optional[str] = None,
        interval_sec: float = 0.1,
    ):
        self.pid = pid
        self.process_pattern = process_pattern
        self.interval_sec = interval_sec
        self._stop = threading.Event()
        self._max_kb = 0
        self._thread: Optional[threading.Thread] = None

    def _collect_pids(self) -> Set[int]:
        pids: Set[int] = set()
        if self.pid is not None and pid_alive(self.pid):
            pids.add(self.pid)
        if self.process_pattern:
            pids |= pids_for_pattern(self.process_pattern)
        return pids

    def _sample_once(self) -> None:
        total_kb = 0
        for current_pid in self._collect_pids():
            rss_kb = get_rss_kb_by_pid(current_pid)
            if rss_kb is not None:
                total_kb += rss_kb
        if total_kb > self._max_kb:
            self._max_kb = total_kb

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval_sec)

    def start(self) -> None:
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Optional[float]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._sample_once()
        if self._max_kb <= 0:
            return None
        return round(self._max_kb / 1024 / 1024, 4)


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def display_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:  # noqa: BLE001
        return str(path)


def safe_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except Exception:  # noqa: BLE001
        return 0


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except Exception:  # noqa: BLE001
        return 0.0


def ns_to_sec(value: Any) -> float:
    return safe_float(value) / 1_000_000_000.0


def extract_text_from_message(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text_part = item.get("text")
                if isinstance(text_part, str) and text_part:
                    chunks.append(text_part)
        if chunks:
            return "".join(chunks)
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        return reasoning
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return reasoning_content
    return ""


def iter_sse_lines(response: requests.Response) -> list[str]:
    lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        if line:
            lines.append(line)
    return lines


def error_result(
    *,
    runtime: str,
    model: str,
    error: str,
    memory_peak_gb: Optional[float] = None,
    payload_path: Optional[Path] = None,
    response_path: Optional[Path] = None,
    transport_mode: str,
    reasoning: Dict[str, Any],
    cache: Dict[str, Any],
    stream_enabled: bool,
) -> Dict[str, Any]:
    return {
        "success": False,
        "runtime": runtime,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "error": error,
        "timing": {
            "client": {
                "total_time_sec": 0.0,
                "ttft_sec": 0.0,
            },
            "server": {},
        },
        "usage": {
            "server": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
            "normalized": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        },
        "throughput": {
            "tokens_per_second": 0.0,
            "prompt_tps": 0.0,
            "generation_tps": 0.0,
        },
        "memory": {
            "peak_gb": memory_peak_gb,
            "delta_gb": None,
        },
        "reasoning": reasoning,
        "cache": cache,
        "transport": {"mode": transport_mode},
        "stream": {
            "enabled": bool(stream_enabled),
            "actual_streaming": False,
        },
        "sources": {},
        "artifacts": {
            "payload_path": display_path(payload_path),
            "response_path": display_path(response_path),
        },
    }


def reasoning_payload_openai(reasoning_effective: str) -> Dict[str, Any]:
    mode = str(reasoning_effective or "off").strip().lower()
    if mode == "off":
        return {"reasoning": {"effort": "none"}}
    if mode == "on":
        return {"reasoning": {"effort": "low"}}
    return {}
