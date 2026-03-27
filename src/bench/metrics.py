from __future__ import annotations

import json
from pathlib import Path
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
        pids: Set[int] = set()
        if self.pid is not None and pid_alive(self.pid):
            pids.add(self.pid)
        if self.process_pattern:
            pids |= pids_for_pattern(self.process_pattern)
        return pids

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


def _dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _display_path(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return str(path.relative_to(Path.cwd()))
    except Exception:  # noqa: BLE001
        return str(path)


def benchmark_model(
    chat_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    runtime: str,
    memory_pid: Optional[int] = None,
    memory_pattern: Optional[str] = None,
    request_timeout_sec: int = 2000,
    artifact_dir: Optional[Path] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
    }
    if extra_payload:
        payload.update(extra_payload)
    payload_path: Optional[Path] = None
    response_path: Optional[Path] = None
    if artifact_dir is not None:
        payload_path = artifact_dir / "payload.json"
        response_path = artifact_dir / "response.json"
        _dump_json(
            payload_path,
            {
                "runtime": runtime,
                "payload": payload,
            },
        )

    memory_monitor = MemoryMonitor(pid=memory_pid, process_pattern=memory_pattern)
    memory_monitor.start()
    start_time = time.time()
    try:
        response = requests.post(chat_url, json=payload, timeout=request_timeout_sec)
        end_time = time.time()
    except requests.exceptions.Timeout:
        max_memory_gb = memory_monitor.stop()
        if response_path is not None:
            _dump_json(
                response_path,
                {
                    "error": f"Request timeout ({request_timeout_sec}s)",
                    "runtime": runtime,
                    "model": model,
                },
            )
        return {
            "error": f"Request timeout ({request_timeout_sec}s)",
            "runtime": runtime,
            "model": model,
            "memory_gb": max_memory_gb,
            "payload_path": _display_path(payload_path),
            "response_path": _display_path(response_path),
        }
    except Exception as exc:  # noqa: BLE001
        max_memory_gb = memory_monitor.stop()
        if response_path is not None:
            _dump_json(
                response_path,
                {
                    "error": str(exc),
                    "runtime": runtime,
                    "model": model,
                },
            )
        return {
            "error": str(exc),
            "runtime": runtime,
            "model": model,
            "memory_gb": max_memory_gb,
            "payload_path": _display_path(payload_path),
            "response_path": _display_path(response_path),
        }
    max_memory_gb = memory_monitor.stop()

    if response.status_code != 200:
        if response_path is not None:
            _dump_json(
                response_path,
                {
                    "status_code": response.status_code,
                    "body": response.text,
                    "runtime": runtime,
                    "model": model,
                },
            )
        return {
            "error": f"HTTP {response.status_code}: {response.text[:200]}",
            "runtime": runtime,
            "model": model,
            "memory_gb": max_memory_gb,
            "payload_path": _display_path(payload_path),
            "response_path": _display_path(response_path),
        }

    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        if response_path is not None:
            _dump_json(
                response_path,
                {
                    "status_code": response.status_code,
                    "body": response.text,
                    "error": f"Invalid JSON response: {exc}",
                    "runtime": runtime,
                    "model": model,
                },
            )
        return {
            "error": f"Invalid JSON response: {exc}",
            "runtime": runtime,
            "model": model,
            "memory_gb": max_memory_gb,
            "payload_path": _display_path(payload_path),
            "response_path": _display_path(response_path),
        }
    total_time = end_time - start_time
    usage = data.get("usage", {})
    perf = data.get("perf", {})
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", 0) or 0)
    response_text = data.get("choices", [{}])[0].get("message", {}).get("content", "")

    if prompt_tokens <= 0 and prompt:
        prompt_tokens = estimate_tokens(prompt)
    if completion_tokens <= 0 and response_text:
        completion_tokens = estimate_tokens(response_text)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    tokens_per_second = completion_tokens / total_time if total_time > 0 else 0.0
    prompt_tps = float(perf.get("prompt_tps", 0.0) or 0.0)
    generation_tps = float(perf.get("generation_tps", 0.0) or 0.0)
    ttft_sec = float(perf.get("ttft_sec", 0.0) or 0.0)
    server_total_time_sec = float(perf.get("total_time_sec", 0.0) or 0.0)
    server_peak_memory_gb = float(perf.get("peak_memory_gb", 0.0) or 0.0)
    reported_memory_gb = max_memory_gb
    if runtime in {"mlx", "mlx-optiq"} and server_peak_memory_gb > 0:
        reported_memory_gb = round(server_peak_memory_gb, 2)
    if response_path is not None:
        _dump_json(
            response_path,
            {
                "status_code": response.status_code,
                "runtime": runtime,
                "model": model,
                "body": data,
            },
        )

    return {
        "success": True,
        "runtime": runtime,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_time": round(total_time, 2),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "tokens_per_second": round(tokens_per_second, 2),
        "prompt_tps": round(prompt_tps, 2),
        "generation_tps": round(generation_tps, 2),
        "ttft_sec": round(ttft_sec, 3),
        "server_total_time_sec": round(server_total_time_sec, 3),
        "memory_gb": reported_memory_gb,
        "response_text": response_text,
        "payload_path": _display_path(payload_path),
        "response_path": _display_path(response_path),
    }


def prefill_prompt_cache(
    prefill_url: str,
    cache_id: str,
    prompt_prefix: str,
    request_timeout_sec: int = 2000,
) -> Dict[str, Any]:
    payload = {
        "cache_id": cache_id,
        "raw_prompt": prompt_prefix,
    }
    try:
        response = requests.post(prefill_url, json=payload, timeout=request_timeout_sec)
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"Prefill timeout ({request_timeout_sec}s)"}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}

    if response.status_code != 200:
        return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}

    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Invalid JSON response: {exc}"}

    return {"success": True, "data": data}


def clear_prompt_cache(
    clear_url: str,
    cache_id: str,
    request_timeout_sec: int = 120,
) -> Dict[str, Any]:
    payload = {
        "cache_id": cache_id,
    }
    try:
        response = requests.post(clear_url, json=payload, timeout=request_timeout_sec)
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"Clear timeout ({request_timeout_sec}s)"}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": str(exc)}

    if response.status_code != 200:
        return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}

    try:
        data = response.json()
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "error": f"Invalid JSON response: {exc}"}

    return {"success": True, "data": data}
