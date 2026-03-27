from __future__ import annotations

import subprocess
import threading
import time
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
        if self.pid is not None and pid_alive(self.pid):
            return {self.pid}
        if self.process_pattern:
            return pids_for_pattern(self.process_pattern)
        return set()

    def _sample_once(self) -> None:
        total_kb = 0
        for pid in self._collect_pids():
            rss_kb = get_rss_kb_by_pid(pid)
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
        return round(self._max_kb / 1024 / 1024, 2)


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))


def benchmark_model(
    chat_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    runtime: str,
    memory_pid: Optional[int] = None,
    memory_pattern: Optional[str] = None,
    request_timeout_sec: int = 600,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }

    memory_monitor = MemoryMonitor(pid=memory_pid, process_pattern=memory_pattern)
    memory_monitor.start()
    start_time = time.time()
    try:
        response = requests.post(chat_url, json=payload, timeout=request_timeout_sec)
        end_time = time.time()
    except requests.exceptions.Timeout:
        max_memory_gb = memory_monitor.stop()
        return {
            "error": f"Request timeout ({request_timeout_sec}s)",
            "runtime": runtime,
            "model": model,
            "prompt": prompt[:50] + "...",
            "memory_gb": max_memory_gb,
        }
    except Exception as exc:  # noqa: BLE001
        max_memory_gb = memory_monitor.stop()
        return {
            "error": str(exc),
            "runtime": runtime,
            "model": model,
            "prompt": prompt[:50] + "...",
            "memory_gb": max_memory_gb,
        }
    max_memory_gb = memory_monitor.stop()

    if response.status_code != 200:
        return {
            "error": f"HTTP {response.status_code}: {response.text[:200]}",
            "runtime": runtime,
            "model": model,
            "prompt": prompt[:50] + "...",
            "memory_gb": max_memory_gb,
        }

    data = response.json()
    total_time = end_time - start_time
    usage = data.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if completion_tokens <= 0 and response_text:
        completion_tokens = estimate_tokens(response_text)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    tokens_per_second = completion_tokens / total_time if total_time > 0 else 0.0

    return {
        "success": True,
        "runtime": runtime,
        "model": model,
        "prompt": prompt,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_time": round(total_time, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "memory_gb": max_memory_gb,
        "response_preview": response_text[:100] + "..." if len(response_text) > 100 else response_text,
    }
