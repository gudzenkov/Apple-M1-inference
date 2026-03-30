from __future__ import annotations

import json
from pathlib import Path
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from src.bench.metrics.common import (
    MemoryMonitor,
    display_path,
    dump_json,
    error_result,
    estimate_tokens,
    ns_to_sec,
    safe_int,
)


def benchmark_ollama_native(
    *,
    chat_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    runtime: str,
    transport_mode: str,
    request_options: Optional[Dict[str, Any]] = None,
    memory_pattern: Optional[str] = None,
    request_timeout_sec: int,
    artifact_dir: Optional[Path] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    stream_enabled: bool = True,
    baseline_rss_gb: Optional[float] = None,
    warmup_rss_gb: Optional[float] = None,
    cache_payload_patch: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    reasoning = dict(reasoning or {})
    cache = dict(cache or {})

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": bool(stream_enabled),
        "options": {
            **(dict(request_options or {})),
            "num_predict": int(max_tokens),
        },
    }

    if str(reasoning.get("format") or "").strip().lower() == "ollama-think":
        payload["think"] = str(reasoning.get("effective", "off")).strip().lower() == "on"

    if cache_payload_patch:
        for key, value in cache_payload_patch.items():
            if key == "options" and isinstance(value, dict):
                merged_options = dict(payload.get("options") or {})
                merged_options.update(value)
                payload["options"] = merged_options
            else:
                payload[key] = value

    payload_path: Optional[Path] = None
    response_path: Optional[Path] = None
    if artifact_dir is not None:
        payload_path = artifact_dir / "payload.json"
        response_path = artifact_dir / "response.json"
        dump_json(
            payload_path,
            {
                "runtime": runtime,
                "transport": transport_mode,
                "payload": payload,
            },
        )

    memory_monitor = MemoryMonitor(process_pattern=memory_pattern)
    memory_monitor.start()

    started_at = time.perf_counter()
    response: Optional[requests.Response] = None

    try:
        response = requests.post(
            chat_url,
            json=payload,
            timeout=request_timeout_sec,
            stream=bool(stream_enabled),
        )
    except requests.exceptions.Timeout:
        peak_memory_gb = memory_monitor.stop()
        return error_result(
            runtime=runtime,
            model=model,
            error=f"Request timeout ({request_timeout_sec}s)",
            memory_peak_gb=peak_memory_gb,
            payload_path=payload_path,
            response_path=response_path,
            transport_mode=transport_mode,
            reasoning=reasoning,
            cache=cache,
            stream_enabled=stream_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        peak_memory_gb = memory_monitor.stop()
        return error_result(
            runtime=runtime,
            model=model,
            error=str(exc),
            memory_peak_gb=peak_memory_gb,
            payload_path=payload_path,
            response_path=response_path,
            transport_mode=transport_mode,
            reasoning=reasoning,
            cache=cache,
            stream_enabled=stream_enabled,
        )

    first_token_at: Optional[float] = None
    response_text = ""
    response_json: Dict[str, Any] = {}
    final_chunk: Dict[str, Any] = {}

    try:
        if response.status_code != 200:
            body_text = response.text
            peak_memory_gb = memory_monitor.stop()
            if response_path is not None:
                dump_json(
                    response_path,
                    {
                        "status_code": response.status_code,
                        "body": body_text,
                        "runtime": runtime,
                        "model": model,
                    },
                )
            return error_result(
                runtime=runtime,
                model=model,
                error=f"HTTP {response.status_code}: {body_text[:200]}",
                memory_peak_gb=peak_memory_gb,
                payload_path=payload_path,
                response_path=response_path,
                transport_mode=transport_mode,
                reasoning=reasoning,
                cache=cache,
                stream_enabled=stream_enabled,
            )

        if bool(stream_enabled):
            text_chunks: list[str] = []
            lines: list[Dict[str, Any]] = []
            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(chunk, dict):
                    continue
                lines.append(chunk)

                message = chunk.get("message")
                if isinstance(message, dict):
                    text_piece = message.get("content")
                    if isinstance(text_piece, str):
                        if text_piece and first_token_at is None:
                            first_token_at = time.perf_counter()
                        text_chunks.append(text_piece)

                if chunk.get("done") is True:
                    final_chunk = chunk
                    break

            response_text = "".join(text_chunks).strip()
            if final_chunk:
                message = final_chunk.get("message")
                if isinstance(message, dict):
                    final_text = message.get("content")
                    if isinstance(final_text, str) and final_text and not response_text:
                        response_text = final_text.strip()
                response_json = final_chunk
            elif lines:
                response_json = lines[-1]
        else:
            try:
                response_json = response.json()
            except Exception as exc:  # noqa: BLE001
                peak_memory_gb = memory_monitor.stop()
                if response_path is not None:
                    dump_json(
                        response_path,
                        {
                            "status_code": response.status_code,
                            "body": response.text,
                            "error": f"Invalid JSON response: {exc}",
                            "runtime": runtime,
                            "model": model,
                        },
                    )
                return error_result(
                    runtime=runtime,
                    model=model,
                    error=f"Invalid JSON response: {exc}",
                    memory_peak_gb=peak_memory_gb,
                    payload_path=payload_path,
                    response_path=response_path,
                    transport_mode=transport_mode,
                    reasoning=reasoning,
                    cache=cache,
                    stream_enabled=stream_enabled,
                )
            message = response_json.get("message") if isinstance(response_json, dict) else {}
            if isinstance(message, dict):
                response_text = str(message.get("content") or "").strip()
    finally:
        peak_memory_gb = memory_monitor.stop()
        if response is not None:
            response.close()

    finished_at = time.perf_counter()
    total_time_sec = finished_at - started_at

    prompt_eval_count = safe_int(response_json.get("prompt_eval_count")) if isinstance(response_json, dict) else 0
    eval_count = safe_int(response_json.get("eval_count")) if isinstance(response_json, dict) else 0
    total_tokens_server = prompt_eval_count + eval_count

    prompt_tokens_normalized = prompt_eval_count if prompt_eval_count > 0 else estimate_tokens(prompt)
    completion_tokens_normalized = eval_count if eval_count > 0 else estimate_tokens(response_text or " ")
    total_tokens_normalized = total_tokens_server if total_tokens_server > 0 else (
        prompt_tokens_normalized + completion_tokens_normalized
    )

    total_duration_sec = ns_to_sec(response_json.get("total_duration")) if isinstance(response_json, dict) else 0.0
    load_duration_sec = ns_to_sec(response_json.get("load_duration")) if isinstance(response_json, dict) else 0.0
    prompt_eval_duration_sec = (
        ns_to_sec(response_json.get("prompt_eval_duration"))
        if isinstance(response_json, dict)
        else 0.0
    )
    eval_duration_sec = ns_to_sec(response_json.get("eval_duration")) if isinstance(response_json, dict) else 0.0

    client_ttft_sec = (first_token_at - started_at) if first_token_at is not None else None

    prompt_tps = (
        float(prompt_eval_count) / prompt_eval_duration_sec
        if prompt_eval_count > 0 and prompt_eval_duration_sec > 0
        else 0.0
    )
    generation_tps = float(eval_count) / eval_duration_sec if eval_count > 0 and eval_duration_sec > 0 else 0.0
    tokens_per_second = float(completion_tokens_normalized) / total_time_sec if total_time_sec > 0 else 0.0

    memory_delta_gb: Optional[float] = None
    if peak_memory_gb is not None and baseline_rss_gb is not None:
        memory_delta_gb = round(max(0.0, peak_memory_gb - baseline_rss_gb), 4)

    sources = {
        "timing.client.total_time_sec": "client_derived",
        "timing.client.ttft_sec": "client_derived" if client_ttft_sec is not None else "unavailable_non_stream",
        "timing.server.total_time_sec": "server" if total_duration_sec > 0 else "client_derived",
        "timing.server.load_time_sec": "server" if load_duration_sec > 0 else "estimated",
        "timing.server.prompt_eval_sec": "server" if prompt_eval_duration_sec > 0 else "estimated",
        "timing.server.eval_sec": "server" if eval_duration_sec > 0 else "estimated",
        "usage.server.prompt_tokens": "server" if prompt_eval_count > 0 else "estimated",
        "usage.server.completion_tokens": "server" if eval_count > 0 else "estimated",
        "usage.server.total_tokens": "server" if total_tokens_server > 0 else "estimated",
        "usage.normalized.prompt_tokens": "server" if prompt_eval_count > 0 else "estimated",
        "usage.normalized.completion_tokens": "server" if eval_count > 0 else "estimated",
        "usage.normalized.total_tokens": "server" if total_tokens_server > 0 else "estimated",
        "throughput.tokens_per_second": "client_derived",
        "throughput.prompt_tps": "server" if prompt_tps > 0 else "estimated",
        "throughput.generation_tps": "server" if generation_tps > 0 else "estimated",
        "memory.peak_gb": "client_derived" if peak_memory_gb is not None else "estimated",
        "memory.delta_gb": "client_derived" if memory_delta_gb is not None else "estimated",
    }

    if response_path is not None and response is not None:
        dump_json(
            response_path,
            {
                "status_code": response.status_code,
                "runtime": runtime,
                "model": model,
                "body": response_json,
            },
        )

    return {
        "success": True,
        "runtime": runtime,
        "model": model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timing": {
            "client": {
                "total_time_sec": round(total_time_sec, 4),
                "ttft_sec": round(client_ttft_sec, 4) if client_ttft_sec is not None else None,
            },
            "server": {
                "total_time_sec": round(total_duration_sec if total_duration_sec > 0 else total_time_sec, 4),
                "ttft_sec": None,
                "load_time_sec": round(load_duration_sec, 4) if load_duration_sec > 0 else None,
                "prompt_eval_sec": round(prompt_eval_duration_sec, 4) if prompt_eval_duration_sec > 0 else None,
                "eval_sec": round(eval_duration_sec, 4) if eval_duration_sec > 0 else None,
            },
        },
        "usage": {
            "server": {
                "prompt_tokens": prompt_eval_count,
                "completion_tokens": eval_count,
                "total_tokens": total_tokens_server,
            },
            "normalized": {
                "prompt_tokens": prompt_tokens_normalized,
                "completion_tokens": completion_tokens_normalized,
                "total_tokens": total_tokens_normalized,
            },
        },
        "throughput": {
            "tokens_per_second": round(tokens_per_second, 4),
            "prompt_tps": round(prompt_tps, 4),
            "generation_tps": round(generation_tps, 4),
        },
        "memory": {
            "baseline_rss_gb": baseline_rss_gb,
            "warmup_rss_gb": warmup_rss_gb,
            "peak_gb": peak_memory_gb,
            "delta_gb": memory_delta_gb,
        },
        "reasoning": reasoning,
        "cache": cache,
        "transport": {"mode": transport_mode},
        "stream": {
            "enabled": bool(stream_enabled),
            "actual_streaming": bool(stream_enabled),
        },
        "sources": sources,
        "artifacts": {
            "payload_path": display_path(payload_path),
            "response_path": display_path(response_path),
        },
        "response_text": response_text,
    }


def warmup_ollama_native(
    *,
    chat_url: str,
    model: str,
    request_timeout_sec: int,
    request_options: Optional[Dict[str, Any]] = None,
    reasoning_effective: str = "off",
    reasoning_format: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": False,
        "options": {
            **(dict(request_options or {})),
            "num_predict": 5,
        },
    }

    if str(reasoning_format or "").strip().lower() == "ollama-think":
        payload["think"] = str(reasoning_effective or "off").strip().lower() == "on"

    started_at = time.perf_counter()
    try:
        response = requests.post(chat_url, json=payload, timeout=request_timeout_sec)
        warmup_sec = time.perf_counter() - started_at
        if response.status_code == 200:
            return {
                "success": True,
                "warmup_sec": round(warmup_sec, 4),
                "status_code": response.status_code,
            }
        return {
            "success": False,
            "warmup_sec": round(warmup_sec, 4),
            "status_code": response.status_code,
            "error": f"HTTP {response.status_code}",
        }
    except Exception as exc:  # noqa: BLE001
        warmup_sec = time.perf_counter() - started_at
        return {
            "success": False,
            "warmup_sec": round(warmup_sec, 4),
            "error": str(exc),
        }
