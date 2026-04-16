from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
import time
from typing import Any, Dict, Optional

from src.bench.composition import ComposedBenchmarkSpec
from src.bench.metrics.cache import clear_prompt_cache, prefill_prompt_cache
from src.bench.metrics.common import error_result
from src.bench.metrics.openai import benchmark_openai_compat, warmup_openai_compat
from src.bench.process import (
    start_managed_server,
    stop_llama_cpp_servers,
    stop_managed_process,
    stop_mlx_servers,
)
from src.bench.utils.text import slug


@dataclass
class ModelRunState:
    setup_failed: bool = False
    setup_error: Optional[str] = None
    managed_proc: Optional[Any] = None
    memory_pid: Optional[int] = None
    cache_ids: set[str] = field(default_factory=set)
    baseline_rss_gb: Optional[float] = None
    warmup_rss_gb: Optional[float] = None
    fatal_error: Optional[str] = None


class RuntimeHandler:
    def setup_model(
        self,
        *,
        spec: ComposedBenchmarkSpec,
        args: Any,
        root_dir: Path,
        log_dir: Path,
        case_build_sec: float,
    ) -> tuple[Dict[str, Any], ModelRunState]:
        raise NotImplementedError

    def run_case(
        self,
        *,
        spec: ComposedBenchmarkSpec,
        args: Any,
        case: Dict[str, Any],
        run_dir: Path,
        state: ModelRunState,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    def teardown_model(self, *, spec: ComposedBenchmarkSpec, state: ModelRunState) -> None:
        raise NotImplementedError


def _stop_runtime_servers(runtime: str, port: Optional[int], *, verbose: bool) -> None:
    runtime_name = runtime.strip().lower()
    if runtime_name in {"mlx", "mlx-optiq"}:
        stop_mlx_servers(verbose=verbose)
        return
    if runtime_name == "llama.cpp":
        stop_llama_cpp_servers(port=port or 8090, verbose=verbose)
        return
    raise ValueError(f"Unsupported runtime: {runtime}")


class OpenAICompatRuntimeHandler(RuntimeHandler):
    def setup_model(
        self,
        *,
        spec: ComposedBenchmarkSpec,
        args: Any,
        root_dir: Path,
        log_dir: Path,
        case_build_sec: float,
    ) -> tuple[Dict[str, Any], ModelRunState]:
        setup_entry: Dict[str, Any] = {
            "runtime": spec.runtime,
            "model": spec.model,
            "model_key": spec.model_key,
            "case_build_sec": round(case_build_sec, 4),
            "download_or_check_sec": 0.0,
            "server_start_sec": 0.0,
            "warmup_sec": 0.0,
            "warmup_success": None,
            "warmup_status_code": None,
            "reasoning_mode": spec.reasoning_effective,
            "cache_mode": spec.cache_mode,
            "transport": spec.transport_mode,
            "stream": spec.stream_enabled,
            "reasoning_supported": spec.reasoning_supported,
            "reasoning_format": spec.reasoning_format,
        }
        state = ModelRunState()

        try:
            if spec.managed_server:
                _stop_runtime_servers(spec.runtime, spec.port, verbose=True)
                server_start_started = time.perf_counter()
                managed_proc = start_managed_server(
                    runtime=spec.runtime,
                    model=spec.model,
                    timeout_sec=spec.server_start_timeout_sec,
                    root_dir=root_dir,
                    config={
                        "log_file": spec.log_file,
                        "port": spec.port,
                        "start_cmd": spec.start_cmd,
                        "health_url": spec.health_url,
                    },
                    log_dir=log_dir,
                )
                setup_entry["server_start_sec"] = round(time.perf_counter() - server_start_started, 4)
                state.managed_proc = managed_proc
                state.memory_pid = managed_proc.pid

            if not args.skip_warmup:
                warmup = warmup_openai_compat(
                    chat_url=spec.chat_url,
                    model=spec.model,
                    request_timeout_sec=spec.request_timeout_sec,
                    reasoning_effective=spec.reasoning_effective,
                )
                setup_entry["warmup_sec"] = float(warmup.get("warmup_sec", 0.0) or 0.0)
                setup_entry["warmup_success"] = bool(warmup.get("success", False))
                setup_entry["warmup_status_code"] = warmup.get("status_code")
                time.sleep(1)
        except Exception as exc:  # noqa: BLE001
            state.setup_failed = True
            state.setup_error = str(exc)
            setup_entry["setup_error"] = state.setup_error

        return setup_entry, state

    def run_case(
        self,
        *,
        spec: ComposedBenchmarkSpec,
        args: Any,
        case: Dict[str, Any],
        run_dir: Path,
        state: ModelRunState,
    ) -> Dict[str, Any]:
        if state.fatal_error:
            return error_result(
                runtime=spec.runtime,
                model=spec.model,
                error=state.fatal_error,
                transport_mode=spec.transport_mode,
                reasoning={
                    "requested": spec.reasoning_requested,
                    "effective": spec.reasoning_effective,
                    "supported": spec.reasoning_supported,
                    "format": spec.reasoning_format,
                    "source": spec.reasoning_source,
                },
                cache={
                    "mode": spec.cache_mode,
                    "used": False,
                    "cache_id": None,
                    "source": spec.cache_source,
                },
                stream_enabled=spec.stream_enabled,
            )

        prompt_text = str(case.get("prompt") or "")
        extra_payload: Optional[Dict[str, Any]] = None
        used_prompt_cache = False
        cache_id: Optional[str] = None
        prefill_sec: Optional[float] = None

        if spec.cache_mode == "prefill" and spec.reasoning_effective == "off":
            prompt_prefix = case.get("prompt_prefix")
            prompt_suffix = case.get("prompt_suffix")
            prompt_cache_group = case.get("prompt_cache_group")
            if (
                spec.cache_prefill_url
                and isinstance(prompt_prefix, str)
                and isinstance(prompt_suffix, str)
                and isinstance(prompt_cache_group, str)
            ):
                cache_id = slug(f"{spec.runtime}-{spec.model_key}-{prompt_cache_group}")
                if cache_id not in state.cache_ids:
                    prefill = prefill_prompt_cache(
                        prefill_url=spec.cache_prefill_url,
                        cache_id=cache_id,
                        prompt_prefix=prompt_prefix,
                        request_timeout_sec=max(spec.request_timeout_sec, 60),
                    )
                    if prefill.get("success"):
                        state.cache_ids.add(cache_id)
                        prefill_sec = float(prefill.get("prefill_sec", 0.0) or 0.0)
                    else:
                        prefill_error = str(prefill.get("error", "unknown"))
                        state.fatal_error = (
                            f"Cache prefill failed for {cache_id}: {prefill_error}"
                        )
                        return error_result(
                            runtime=spec.runtime,
                            model=spec.model,
                            error=state.fatal_error,
                            transport_mode=spec.transport_mode,
                            reasoning={
                                "requested": spec.reasoning_requested,
                                "effective": spec.reasoning_effective,
                                "supported": spec.reasoning_supported,
                                "format": spec.reasoning_format,
                                "source": spec.reasoning_source,
                            },
                            cache={
                                "mode": spec.cache_mode,
                                "used": False,
                                "cache_id": cache_id,
                                "source": spec.cache_source,
                            },
                            stream_enabled=spec.stream_enabled,
                        )
                prompt_text = prompt_suffix
                extra_payload = {
                    "raw_prompt": prompt_suffix,
                    "cache_id": cache_id,
                }
                used_prompt_cache = True

        reasoning = {
            "requested": spec.reasoning_requested,
            "effective": spec.reasoning_effective,
            "supported": spec.reasoning_supported,
            "format": spec.reasoning_format,
            "source": spec.reasoning_source,
        }
        cache = {
            "mode": spec.cache_mode,
            "used": used_prompt_cache,
            "cache_id": cache_id,
            "source": spec.cache_source,
        }

        result = benchmark_openai_compat(
            chat_url=spec.chat_url,
            model=spec.model,
            prompt=prompt_text,
            max_tokens=int(case["max_tokens"]),
            runtime=spec.runtime,
            transport_mode=spec.transport_mode,
            memory_pid=state.memory_pid,
            memory_pattern=spec.process_hint,
            request_timeout_sec=spec.request_timeout_sec,
            artifact_dir=run_dir,
            extra_payload=extra_payload,
            reasoning=reasoning,
            cache=cache,
            stream_enabled=spec.stream_enabled,
        )
        if prefill_sec and prefill_sec > 0:
            timing = result.setdefault("timing", {})
            if not isinstance(timing, dict):
                timing = {}
                result["timing"] = timing
            cache_timing = timing.setdefault("cache", {})
            if not isinstance(cache_timing, dict):
                cache_timing = {}
                timing["cache"] = cache_timing
            cache_timing["prefill_sec"] = round(prefill_sec, 4)

            sources = result.setdefault("sources", {})
            if not isinstance(sources, dict):
                sources = {}
                result["sources"] = sources
            sources["timing.cache.prefill_sec"] = "client_derived"
        return result

    def teardown_model(self, *, spec: ComposedBenchmarkSpec, state: ModelRunState) -> None:
        try:
            if spec.cache_clear_url and state.cache_ids:
                for cache_id in sorted(state.cache_ids):
                    clear_result = clear_prompt_cache(
                        clear_url=spec.cache_clear_url,
                        cache_id=cache_id,
                        request_timeout_sec=spec.request_timeout_sec,
                    )
                    if not clear_result.get("success"):
                        print(
                            f"  ! Cache clear failed for {cache_id}: {clear_result.get('error', 'unknown')}",
                            file=sys.stderr,
                        )
        finally:
            if state.managed_proc is not None:
                stop_managed_process(state.managed_proc)
            if spec.managed_server:
                _stop_runtime_servers(spec.runtime, spec.port, verbose=False)


def get_runtime_handler(runtime: str) -> RuntimeHandler:
    runtime_name = runtime.strip().lower()
    if runtime_name in {"mlx", "mlx-optiq", "llama.cpp"}:
        return OpenAICompatRuntimeHandler()
    raise ValueError(f"Unsupported runtime: {runtime}")
