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
    extract_text_from_message,
    iter_sse_lines,
    reasoning_payload_openai,
    safe_float,
    safe_int,
)


def benchmark_openai_compat(
    *,
    chat_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    runtime: str,
    transport_mode: str,
    memory_pid: Optional[int] = None,
    memory_pattern: Optional[str] = None,
    request_timeout_sec: int = 2000,
    artifact_dir: Optional[Path] = None,
    extra_payload: Optional[Dict[str, Any]] = None,
    reasoning: Optional[Dict[str, Any]] = None,
    cache: Optional[Dict[str, Any]] = None,
    stream_enabled: bool = True,
) -> Dict[str, Any]:
    reasoning = dict(reasoning or {})
    cache = dict(cache or {})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": bool(stream_enabled),
    }
    payload.update(reasoning_payload_openai(str(reasoning.get("effective", "off"))))
    if extra_payload:
        payload.update(extra_payload)

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

    memory_monitor = MemoryMonitor(pid=memory_pid, process_pattern=memory_pattern)
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
    stream_mode = False
    response_json: Dict[str, Any] = {}
    usage: Dict[str, Any] = {}
    perf: Dict[str, Any] = {}
    response_text = ""

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

        content_type = (response.headers.get("content-type") or "").lower()
        if bool(stream_enabled) and "text/event-stream" in content_type:
            stream_mode = True
            sse_lines = iter_sse_lines(response)
            text_chunks: list[str] = []
            reasoning_chunks: list[str] = []
            last_chunk: Dict[str, Any] = {}
            for line in sse_lines:
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if not isinstance(chunk, dict):
                    continue
                last_chunk = chunk

                chunk_usage = chunk.get("usage")
                if isinstance(chunk_usage, dict):
                    usage = dict(chunk_usage)

                chunk_perf = chunk.get("perf")
                if isinstance(chunk_perf, dict):
                    perf.update(chunk_perf)

                choices = chunk.get("choices")
                if isinstance(choices, list) and choices:
                    choice0 = choices[0] if isinstance(choices[0], dict) else {}
                    delta = choice0.get("delta")
                    if isinstance(delta, dict):
                        delta_content = delta.get("content")
                        if isinstance(delta_content, str):
                            if delta_content and first_token_at is None:
                                first_token_at = time.perf_counter()
                            text_chunks.append(delta_content)

                        delta_reasoning = delta.get("reasoning")
                        if isinstance(delta_reasoning, str):
                            if delta_reasoning and first_token_at is None:
                                first_token_at = time.perf_counter()
                            reasoning_chunks.append(delta_reasoning)

                    message = choice0.get("message")
                    message_text = extract_text_from_message(message)
                    if message_text and first_token_at is None:
                        first_token_at = time.perf_counter()
                    if message_text and not text_chunks and not reasoning_chunks:
                        text_chunks.append(message_text)

            response_text = "".join(text_chunks).strip()
            if not response_text:
                response_text = "".join(reasoning_chunks).strip()
            response_json = {
                "object": "chat.completion",
                "choices": [{"message": {"role": "assistant", "content": response_text}}],
                "usage": usage,
                "perf": perf,
                "stream_mode": True,
                "last_chunk": last_chunk,
            }
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

            usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
            perf = response_json.get("perf", {}) if isinstance(response_json, dict) else {}
            choices = response_json.get("choices", []) if isinstance(response_json, dict) else []
            if isinstance(choices, list) and choices:
                choice0 = choices[0] if isinstance(choices[0], dict) else {}
                response_text = extract_text_from_message(choice0.get("message"))
            if response_text:
                first_token_at = time.perf_counter()
    finally:
        peak_memory_gb = memory_monitor.stop()
        if response is not None:
            response.close()

    finished_at = time.perf_counter()
    total_time_sec = finished_at - started_at

    prompt_tokens_server = safe_int(usage.get("prompt_tokens")) if isinstance(usage, dict) else 0
    completion_tokens_server = safe_int(usage.get("completion_tokens")) if isinstance(usage, dict) else 0
    total_tokens_server = safe_int(usage.get("total_tokens")) if isinstance(usage, dict) else 0

    prompt_tokens_normalized = prompt_tokens_server
    completion_tokens_normalized = completion_tokens_server
    total_tokens_normalized = total_tokens_server

    sources: Dict[str, str] = {}

    if prompt_tokens_normalized <= 0 and prompt:
        prompt_tokens_normalized = estimate_tokens(prompt)
        sources["usage.normalized.prompt_tokens"] = "estimated"
    else:
        sources["usage.normalized.prompt_tokens"] = "server"

    if completion_tokens_normalized <= 0 and response_text:
        completion_tokens_normalized = estimate_tokens(response_text)
        sources["usage.normalized.completion_tokens"] = "estimated"
    else:
        sources["usage.normalized.completion_tokens"] = "server"

    if total_tokens_normalized <= 0:
        total_tokens_normalized = prompt_tokens_normalized + completion_tokens_normalized
        sources["usage.normalized.total_tokens"] = "estimated"
    else:
        sources["usage.normalized.total_tokens"] = "server"

    sources["usage.server.prompt_tokens"] = "server" if prompt_tokens_server > 0 else "estimated"
    sources["usage.server.completion_tokens"] = "server" if completion_tokens_server > 0 else "estimated"
    sources["usage.server.total_tokens"] = "server" if total_tokens_server > 0 else "estimated"

    client_ttft_sec = (first_token_at - started_at) if first_token_at is not None else total_time_sec
    sources["timing.client.total_time_sec"] = "client_derived"
    sources["timing.client.ttft_sec"] = "client_derived"

    server_ttft_sec = safe_float(perf.get("ttft_sec")) if isinstance(perf, dict) else 0.0
    server_total_time_sec = safe_float(perf.get("total_time_sec")) if isinstance(perf, dict) else 0.0

    if server_ttft_sec > 0:
        sources["timing.server.ttft_sec"] = "server"
    if server_total_time_sec > 0:
        sources["timing.server.total_time_sec"] = "server"
    else:
        server_total_time_sec = total_time_sec
        sources["timing.server.total_time_sec"] = "client_derived"

    prompt_tps = safe_float(perf.get("prompt_tps")) if isinstance(perf, dict) else 0.0
    generation_tps = safe_float(perf.get("generation_tps")) if isinstance(perf, dict) else 0.0

    if prompt_tps > 0:
        sources["throughput.prompt_tps"] = "server"
    elif client_ttft_sec > 0:
        prompt_tps = float(prompt_tokens_normalized) / client_ttft_sec
        sources["throughput.prompt_tps"] = "estimated"

    decode_time = max(total_time_sec - client_ttft_sec, 1e-9)
    if generation_tps > 0:
        sources["throughput.generation_tps"] = "server"
    elif completion_tokens_normalized > 0 and decode_time > 0:
        generation_tps = float(completion_tokens_normalized) / decode_time
        sources["throughput.generation_tps"] = "estimated"

    tokens_per_second = float(completion_tokens_normalized) / total_time_sec if total_time_sec > 0 else 0.0
    sources["throughput.tokens_per_second"] = "client_derived"

    server_peak_memory_gb = safe_float(perf.get("peak_memory_gb")) if isinstance(perf, dict) else 0.0
    if runtime in {"mlx", "mlx-optiq"} and server_peak_memory_gb > 0:
        peak_gb = round(server_peak_memory_gb, 4)
        sources["memory.peak_gb"] = "server"
    else:
        peak_gb = peak_memory_gb
        sources["memory.peak_gb"] = "client_derived" if peak_gb is not None else "estimated"

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
                "ttft_sec": round(client_ttft_sec, 4),
            },
            "server": {
                "total_time_sec": round(server_total_time_sec, 4),
                "ttft_sec": round(server_ttft_sec, 4) if server_ttft_sec > 0 else None,
            },
        },
        "usage": {
            "server": {
                "prompt_tokens": prompt_tokens_server,
                "completion_tokens": completion_tokens_server,
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
            "peak_gb": peak_gb,
            "delta_gb": None,
        },
        "reasoning": reasoning,
        "cache": cache,
        "transport": {"mode": transport_mode},
        "stream": {
            "enabled": bool(stream_enabled),
            "actual_streaming": bool(stream_mode),
        },
        "sources": sources,
        "artifacts": {
            "payload_path": display_path(payload_path),
            "response_path": display_path(response_path),
        },
        "response_text": response_text,
    }


def warmup_openai_compat(
    *,
    chat_url: str,
    model: str,
    request_timeout_sec: int,
    reasoning_effective: str,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 5,
        "stream": False,
    }
    payload.update(reasoning_payload_openai(reasoning_effective))

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
