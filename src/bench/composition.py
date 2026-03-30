from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from src.shared.models import (
    ModelEntry,
    get_default_model_id,
    get_model_entry,
    load_models,
    load_profiles,
    resolve_model_reference,
    resolve_runtime_for_model_reference,
)

BENCH_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "bench.yaml"

CLI_REASONING_MODES = {"auto", "off", "on"}
CLI_CACHE_MODES = {"auto", "prefill", "request", "none"}
CLI_TRANSPORT_MODES = {"auto", "openai-compat", "ollama-native"}
CLI_STREAM_MODES = {"auto", "on", "off"}

CONCRETE_REASONING_MODES = {"off", "on"}
CONCRETE_CACHE_MODES = {"prefill", "request", "none"}
CONCRETE_TRANSPORT_MODES = {"openai-compat", "ollama-native"}
CONCRETE_STREAM_MODES = {"on", "off"}


@dataclass(frozen=True)
class ComposedBenchmarkSpec:
    runtime: str
    model: str
    model_key: str
    family: str
    profile: str
    capabilities: Dict[str, Any]
    managed_server: bool
    chat_url: str
    health_url: str
    cache_prefill_url: Optional[str]
    cache_clear_url: Optional[str]
    port: Optional[int]
    start_cmd: Optional[List[str]]
    log_file: Optional[str]
    process_hint: Optional[str]
    max_context_tokens: int
    request_timeout_sec: int
    server_start_timeout_sec: int
    request_options: Dict[str, Any]
    reasoning_requested: str
    reasoning_effective: str
    reasoning_supported: bool
    reasoning_format: Optional[str]
    reasoning_source: str
    cache_mode: str
    cache_source: str
    transport_mode: str
    transport_source: str
    stream_enabled: bool
    stream_source: str


def _expect_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return value


def _as_non_empty_str(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _as_optional_non_empty_str(value: Any, *, field: str, context: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context}.{field} must be a non-empty string")
    return value.strip()


def _as_int(value: Any, *, field: str, context: str) -> int:
    if isinstance(value, bool):
        raise RuntimeError(f"{context}.{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise RuntimeError(f"{context}.{field} must be an integer")


def _as_optional_int(value: Any, *, field: str, context: str) -> Optional[int]:
    if value is None:
        return None
    return _as_int(value, field=field, context=context)


def _as_bool(value: Any, *, field: str, context: str) -> bool:
    if isinstance(value, bool):
        return value
    raise RuntimeError(f"{context}.{field} must be a boolean")


def _normalize_mode(value: Any, valid: set[str], *, field: str, context: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{context}.{field} must be one of {sorted(valid)}")
    mode = value.strip().lower()
    if mode not in valid:
        raise RuntimeError(f"{context}.{field} must be one of {sorted(valid)}")
    return mode


def _as_mode_from_bool_or_string(value: Any, *, field: str, context: str, valid: set[str]) -> str:
    if isinstance(value, bool):
        mode = "on" if value else "off"
        if mode in valid:
            return mode
    if isinstance(value, str):
        mode = value.strip().lower()
        if mode in valid:
            return mode
    raise RuntimeError(f"{context}.{field} must be one of {sorted(valid)}")


def _as_str_list(value: Any, *, field: str, context: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context}.{field} must be a non-empty list")
    items: List[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"{context}.{field}[{i}] must be a non-empty string")
        items.append(item.strip().lower())
    return list(dict.fromkeys(items))


def _as_command_list(value: Any, *, field: str, context: str) -> List[str]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context}.{field} must be a non-empty list")
    cmd: List[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"{context}.{field}[{i}] must be a non-empty string")
        cmd.append(item.strip())
    return cmd


def _parse_runtime_infra(runtime: str, raw: Mapping[str, Any], context: str) -> Dict[str, Any]:
    managed_server = _as_bool(raw.get("managed_server"), field="managed_server", context=context)
    model_runtime = _as_non_empty_str(raw.get("model_runtime"), field="model_runtime", context=context).lower()
    if model_runtime != runtime:
        raise RuntimeError(f"{context}.model_runtime must match runtime key '{runtime}'")

    health_url = _as_non_empty_str(raw.get("health_url"), field="health_url", context=context)
    process_hint = _as_optional_non_empty_str(raw.get("process_hint"), field="process_hint", context=context)
    max_context_tokens = _as_int(raw.get("max_context_tokens"), field="max_context_tokens", context=context)

    supported_transports = set(_as_str_list(raw.get("supported_transports"), field="supported_transports", context=context))
    invalid_transports = sorted(supported_transports - CONCRETE_TRANSPORT_MODES)
    if invalid_transports:
        raise RuntimeError(f"{context}.supported_transports has invalid values: {invalid_transports}")

    supported_cache_modes = set(_as_str_list(raw.get("supported_cache_modes"), field="supported_cache_modes", context=context))
    invalid_cache_modes = sorted(supported_cache_modes - CONCRETE_CACHE_MODES)
    if invalid_cache_modes:
        raise RuntimeError(f"{context}.supported_cache_modes has invalid values: {invalid_cache_modes}")

    chat_url_openai = _as_optional_non_empty_str(raw.get("chat_url_openai"), field="chat_url_openai", context=context)
    chat_url_native = _as_optional_non_empty_str(raw.get("chat_url_native"), field="chat_url_native", context=context)
    if "openai-compat" in supported_transports and chat_url_openai is None:
        raise RuntimeError(f"{context}.chat_url_openai is required for openai-compat transport")
    if "ollama-native" in supported_transports and chat_url_native is None:
        raise RuntimeError(f"{context}.chat_url_native is required for ollama-native transport")

    cache_prefill_url = _as_optional_non_empty_str(raw.get("cache_prefill_url"), field="cache_prefill_url", context=context)
    cache_clear_url = _as_optional_non_empty_str(raw.get("cache_clear_url"), field="cache_clear_url", context=context)
    if "prefill" in supported_cache_modes and (cache_prefill_url is None or cache_clear_url is None):
        raise RuntimeError(
            f"{context} must define cache_prefill_url and cache_clear_url when prefill cache mode is supported"
        )

    port = _as_optional_int(raw.get("port"), field="port", context=context)
    start_cmd_raw = raw.get("start_cmd")
    start_cmd = _as_command_list(start_cmd_raw, field="start_cmd", context=context) if start_cmd_raw is not None else None
    log_file = _as_optional_non_empty_str(raw.get("log_file"), field="log_file", context=context)
    if managed_server:
        if port is None:
            raise RuntimeError(f"{context}.port is required for managed_server=true")
        if start_cmd is None:
            raise RuntimeError(f"{context}.start_cmd is required for managed_server=true")
        if log_file is None:
            raise RuntimeError(f"{context}.log_file is required for managed_server=true")

    return {
        "managed_server": managed_server,
        "model_runtime": model_runtime,
        "chat_url_openai": chat_url_openai,
        "chat_url_native": chat_url_native,
        "health_url": health_url,
        "cache_prefill_url": cache_prefill_url,
        "cache_clear_url": cache_clear_url,
        "port": port,
        "start_cmd": start_cmd,
        "log_file": log_file,
        "process_hint": process_hint,
        "max_context_tokens": max_context_tokens,
        "supported_transports": supported_transports,
        "supported_cache_modes": supported_cache_modes,
    }


def _normalize_runtime_policy(raw: Mapping[str, Any], context: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    out["reasoning_mode"] = _normalize_mode(
        raw.get("reasoning_mode"),
        CONCRETE_REASONING_MODES,
        field="reasoning_mode",
        context=context,
    )
    out["cache_mode"] = _normalize_mode(
        raw.get("cache_mode"),
        CONCRETE_CACHE_MODES,
        field="cache_mode",
        context=context,
    )
    out["transport"] = _normalize_mode(
        raw.get("transport"),
        CONCRETE_TRANSPORT_MODES,
        field="transport",
        context=context,
    )
    out["stream"] = _as_mode_from_bool_or_string(
        raw.get("stream"),
        field="stream",
        context=context,
        valid=CONCRETE_STREAM_MODES,
    )
    out["request_timeout_sec"] = _as_int(
        raw.get("request_timeout_sec"),
        field="request_timeout_sec",
        context=context,
    )
    out["server_start_timeout_sec"] = _as_int(
        raw.get("server_start_timeout_sec"),
        field="server_start_timeout_sec",
        context=context,
    )

    if "request_options" in raw:
        req_options = _expect_mapping(raw.get("request_options"), f"{context}.request_options")
        out["request_options"] = dict(req_options)

    return out


def _normalize_model_override(raw: Mapping[str, Any], context: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if "reasoning_mode" in raw:
        out["reasoning_mode"] = _normalize_mode(
            raw.get("reasoning_mode"),
            CONCRETE_REASONING_MODES,
            field="reasoning_mode",
            context=context,
        )
    if "cache_mode" in raw:
        out["cache_mode"] = _normalize_mode(
            raw.get("cache_mode"),
            CONCRETE_CACHE_MODES,
            field="cache_mode",
            context=context,
        )
    if "transport" in raw:
        out["transport"] = _normalize_mode(
            raw.get("transport"),
            CONCRETE_TRANSPORT_MODES,
            field="transport",
            context=context,
        )
    if "stream" in raw:
        out["stream"] = _as_mode_from_bool_or_string(
            raw.get("stream"),
            field="stream",
            context=context,
            valid=CONCRETE_STREAM_MODES,
        )

    request_timeout_sec = _as_optional_int(raw.get("request_timeout_sec"), field="request_timeout_sec", context=context)
    if request_timeout_sec is not None:
        out["request_timeout_sec"] = request_timeout_sec

    server_start_timeout_sec = _as_optional_int(
        raw.get("server_start_timeout_sec"),
        field="server_start_timeout_sec",
        context=context,
    )
    if server_start_timeout_sec is not None:
        out["server_start_timeout_sec"] = server_start_timeout_sec

    if "request_options" in raw:
        req_options = _expect_mapping(raw.get("request_options"), f"{context}.request_options")
        out["request_options"] = dict(req_options)

    return out


@lru_cache(maxsize=1)
def _load_bench_config(
    path: Path = BENCH_CONFIG_PATH,
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = _expect_mapping(raw, f"{path}")

    infra_root = _expect_mapping(root.get("infra") or {}, f"{path}: infra")
    infra_runtimes_raw = _expect_mapping(infra_root.get("runtimes") or {}, f"{path}: infra.runtimes")
    if not infra_runtimes_raw:
        raise RuntimeError(f"{path}: infra.runtimes must be a non-empty mapping")

    runtime_infra: Dict[str, Dict[str, Any]] = {}
    for runtime_raw, infra_raw in infra_runtimes_raw.items():
        runtime = str(runtime_raw).strip().lower()
        if not runtime:
            raise RuntimeError(f"{path}: infra.runtimes has empty runtime key")
        if runtime in runtime_infra:
            raise RuntimeError(f"{path}: duplicate runtime in infra.runtimes: {runtime}")
        runtime_map = _expect_mapping(infra_raw, f"{path}: infra.runtimes.{runtime}")
        runtime_infra[runtime] = _parse_runtime_infra(runtime, runtime_map, f"{path}: infra.runtimes.{runtime}")

    defaults_raw = _expect_mapping(root.get("defaults") or {}, f"{path}: defaults")
    runtimes_raw = _expect_mapping(defaults_raw.get("runtimes") or {}, f"{path}: defaults.runtimes")

    runtime_defaults: Dict[str, Dict[str, Any]] = {}
    for runtime, runtime_raw in runtimes_raw.items():
        runtime_key = str(runtime).strip().lower()
        if runtime_key not in runtime_infra:
            raise RuntimeError(f"{path}: defaults.runtimes.{runtime_key} has no matching infra.runtimes entry")
        runtime_map = _expect_mapping(runtime_raw, f"{path}: defaults.runtimes.{runtime_key}")
        runtime_defaults[runtime_key] = _normalize_runtime_policy(runtime_map, f"{path}: defaults.runtimes.{runtime_key}")

    missing_policy = sorted(set(runtime_infra.keys()) - set(runtime_defaults.keys()))
    if missing_policy:
        raise RuntimeError(f"{path}: missing defaults.runtimes entries for runtimes: {missing_policy}")

    model_overrides_raw = _expect_mapping(root.get("models") or {}, f"{path}: models")
    model_overrides: Dict[str, Dict[str, Any]] = {}
    for model_ref_raw, override_raw in model_overrides_raw.items():
        model_ref = str(model_ref_raw).strip().lower()
        if not model_ref:
            raise RuntimeError(f"{path}: models contains empty key")
        override_map = _expect_mapping(override_raw, f"{path}: models.{model_ref}")
        model_overrides[model_ref] = _normalize_model_override(
            override_map,
            f"{path}: models.{model_ref}",
        )

    return runtime_infra, runtime_defaults, model_overrides


def _resolve_setting(
    *,
    cli_value: Any,
    auto_sentinel: Any,
    model_override: Mapping[str, Any],
    runtime_defaults: Mapping[str, Any],
    profile_defaults: Mapping[str, Any],
    key: str,
) -> tuple[Any, str]:
    if cli_value is not None and cli_value != auto_sentinel:
        return cli_value, "cli"
    if key in model_override:
        return model_override[key], "bench_model_override"
    if key in runtime_defaults:
        return runtime_defaults[key], "bench_runtime_default"
    if key in profile_defaults:
        return profile_defaults[key], "profile_default"
    raise ValueError(f"Missing required composed setting '{key}'")


def _stream_mode_to_bool(mode: str) -> bool:
    return mode == "on"


def _lookup_model_override(model_entry: ModelEntry, overrides: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    refs = [
        model_entry["key"].lower(),
        model_entry["model"].lower(),
        *[alias.lower() for alias in model_entry["aliases"]],
    ]
    for ref in refs:
        override = overrides.get(ref)
        if override is not None:
            return dict(override)
    return {}


def list_available_runtimes() -> List[str]:
    runtime_infra, _, _ = _load_bench_config()
    return sorted(runtime_infra.keys())


def resolve_runtimes(runtime_arg: str, model_arg: Optional[str], all_models: bool) -> List[str]:
    runtime_infra, _, _ = _load_bench_config()
    configured_runtimes = set(runtime_infra.keys())

    runtime = str(runtime_arg or "auto").strip().lower()
    if runtime != "auto":
        if runtime not in configured_runtimes:
            raise ValueError(f"Unknown runtime '{runtime}'")
        return [runtime]

    if isinstance(model_arg, str) and model_arg.strip():
        resolved_runtime = resolve_runtime_for_model_reference(model_arg)
        if resolved_runtime not in configured_runtimes:
            raise ValueError(f"Runtime '{resolved_runtime}' for model '{model_arg}' is not configured in bench infra")
        return [resolved_runtime]

    if all_models:
        runtimes: List[str] = []
        for entry in load_models():
            runtime_name = str(entry["runtime"]).strip().lower()
            if runtime_name not in configured_runtimes:
                raise ValueError(f"Model runtime '{runtime_name}' is not configured in bench infra")
            if runtime_name and runtime_name not in runtimes:
                runtimes.append(runtime_name)
        if runtimes:
            return runtimes

    default_model = get_default_model_id()
    resolved_runtime = resolve_runtime_for_model_reference(default_model)
    if resolved_runtime not in configured_runtimes:
        raise ValueError(f"Runtime '{resolved_runtime}' for default model is not configured in bench infra")
    return [resolved_runtime]


def select_models(runtime: str, specific_model: Optional[str], all_models: bool) -> List[str]:
    runtime_name = runtime.strip().lower()
    runtime_infra, _, _ = _load_bench_config()
    if runtime_name not in runtime_infra:
        raise ValueError(f"Unknown runtime '{runtime_name}'")

    if specific_model:
        return [resolve_model_reference(specific_model, runtime=runtime_name)]

    models = [entry["model"] for entry in load_models() if entry["runtime"] == runtime_name]
    models = list(dict.fromkeys(models))
    if not models:
        raise ValueError(f"No configured models for runtime '{runtime_name}'")

    if all_models:
        return models

    return [models[0]]


def compose_benchmark_spec(runtime: str, model: str, args: Any) -> ComposedBenchmarkSpec:
    runtime_name = runtime.strip().lower()
    runtime_infra, runtime_defaults, model_overrides = _load_bench_config()
    if runtime_name not in runtime_infra:
        raise ValueError(f"Unknown runtime '{runtime_name}'")

    runtime_policy = runtime_defaults[runtime_name]
    infra = runtime_infra[runtime_name]

    model_entry = get_model_entry(model, runtime=runtime_name)
    profile_key = model_entry["profile"]
    profiles = load_profiles()
    profile = profiles.get(profile_key)
    profile_defaults: Dict[str, Any] = dict(profile.get("defaults", {}) if profile else {})
    profile_capabilities: Dict[str, Any] = dict(profile.get("capabilities", {}) if profile else {})
    profile_reasoning: Dict[str, Any] = dict(profile.get("reasoning", {}) if profile else {})

    override = _lookup_model_override(model_entry, model_overrides)

    reasoning_mode, reasoning_source = _resolve_setting(
        cli_value=getattr(args, "reasoning_mode", None),
        auto_sentinel="auto",
        model_override=override,
        runtime_defaults=runtime_policy,
        profile_defaults=profile_defaults,
        key="reasoning_mode",
    )
    reasoning_mode = str(reasoning_mode).strip().lower()
    if reasoning_mode not in CONCRETE_REASONING_MODES:
        raise ValueError(f"Invalid reasoning mode '{reasoning_mode}' for {runtime_name}/{model_entry['key']}")

    cache_mode, cache_source = _resolve_setting(
        cli_value=getattr(args, "cache_mode", None),
        auto_sentinel="auto",
        model_override=override,
        runtime_defaults=runtime_policy,
        profile_defaults=profile_defaults,
        key="cache_mode",
    )
    cache_mode = str(cache_mode).strip().lower()
    if cache_mode not in CONCRETE_CACHE_MODES:
        raise ValueError(f"Invalid cache mode '{cache_mode}'")

    transport_mode, transport_source = _resolve_setting(
        cli_value=getattr(args, "transport", None),
        auto_sentinel="auto",
        model_override=override,
        runtime_defaults=runtime_policy,
        profile_defaults=profile_defaults,
        key="transport",
    )
    transport_mode = str(transport_mode).strip().lower()
    if transport_mode not in CONCRETE_TRANSPORT_MODES:
        raise ValueError(f"Invalid transport mode '{transport_mode}'")

    stream_mode, stream_source = _resolve_setting(
        cli_value=getattr(args, "stream", None),
        auto_sentinel="auto",
        model_override=override,
        runtime_defaults=runtime_policy,
        profile_defaults=profile_defaults,
        key="stream",
    )
    stream_mode = str(stream_mode).strip().lower()
    if stream_mode not in CONCRETE_STREAM_MODES:
        raise ValueError(f"Invalid stream mode '{stream_mode}'")
    stream_enabled = _stream_mode_to_bool(stream_mode)

    request_timeout_sec, _ = _resolve_setting(
        cli_value=getattr(args, "request_timeout", None),
        auto_sentinel=None,
        model_override=override,
        runtime_defaults=runtime_policy,
        profile_defaults=profile_defaults,
        key="request_timeout_sec",
    )
    server_start_timeout_sec, _ = _resolve_setting(
        cli_value=getattr(args, "server_start_timeout", None),
        auto_sentinel=None,
        model_override=override,
        runtime_defaults=runtime_policy,
        profile_defaults=profile_defaults,
        key="server_start_timeout_sec",
    )

    profile_request_options = profile_defaults.get("request_options")
    merged_request_options: Dict[str, Any] = {}
    if isinstance(profile_request_options, Mapping):
        merged_request_options.update(dict(profile_request_options))
    runtime_request_options = runtime_policy.get("request_options")
    if isinstance(runtime_request_options, Mapping):
        merged_request_options.update(dict(runtime_request_options))
    override_request_options = override.get("request_options")
    if isinstance(override_request_options, Mapping):
        merged_request_options.update(dict(override_request_options))

    cli_request_options = getattr(args, "request_options", None)
    if isinstance(cli_request_options, Mapping):
        merged_request_options.update(dict(cli_request_options))

    capabilities = dict(profile_capabilities)
    capabilities.update(dict(model_entry.get("capabilities", {})))

    reasoning_supported = bool(capabilities.get("reasoning_supported", False))
    reasoning_format = capabilities.get("reasoning_format")
    if isinstance(reasoning_format, str):
        reasoning_format = reasoning_format.strip().lower() or None
    else:
        reasoning_format = None
    if reasoning_format is None:
        profile_reasoning_format = profile_reasoning.get("format")
        if isinstance(profile_reasoning_format, str) and profile_reasoning_format.strip():
            reasoning_format = profile_reasoning_format.strip().lower()

    reasoning_effective = reasoning_mode
    if reasoning_effective == "on" and not reasoning_supported:
        raise ValueError(
            f"Reasoning is not supported for {runtime_name}/{model_entry['key']} (profile {profile_key})"
        )

    if transport_mode not in set(infra.get("supported_transports", set())):
        raise ValueError(
            f"Transport '{transport_mode}' is not supported for runtime '{runtime_name}'"
        )
    if runtime_name == "ollama" and transport_mode != "ollama-native":
        raise ValueError("Ollama benchmarking is restricted to transport 'ollama-native'")

    if cache_mode not in set(infra.get("supported_cache_modes", set())):
        raise ValueError(
            f"Cache mode '{cache_mode}' is not supported for runtime '{runtime_name}'"
        )

    chat_url_key = "chat_url_native" if transport_mode == "ollama-native" else "chat_url_openai"
    chat_url = str(infra.get(chat_url_key) or "").strip()
    if not chat_url:
        raise ValueError(f"Runtime '{runtime_name}' missing endpoint for transport '{transport_mode}'")

    start_cmd_raw = infra.get("start_cmd")
    start_cmd = list(start_cmd_raw) if isinstance(start_cmd_raw, list) else None

    return ComposedBenchmarkSpec(
        runtime=runtime_name,
        model=model_entry["model"],
        model_key=model_entry["key"],
        family=model_entry["family"],
        profile=profile_key,
        capabilities=capabilities,
        managed_server=bool(infra.get("managed_server", False)),
        chat_url=chat_url,
        health_url=str(infra.get("health_url") or "").strip(),
        cache_prefill_url=str(infra.get("cache_prefill_url") or "").strip() or None,
        cache_clear_url=str(infra.get("cache_clear_url") or "").strip() or None,
        port=int(infra["port"]) if infra.get("port") is not None else None,
        start_cmd=start_cmd,
        log_file=str(infra.get("log_file") or "").strip() or None,
        process_hint=str(infra.get("process_hint") or "").strip() or None,
        max_context_tokens=int(infra.get("max_context_tokens", 256000)),
        request_timeout_sec=int(request_timeout_sec),
        server_start_timeout_sec=int(server_start_timeout_sec),
        request_options=merged_request_options,
        reasoning_requested=reasoning_mode,
        reasoning_effective=reasoning_effective,
        reasoning_supported=reasoning_supported,
        reasoning_format=reasoning_format,
        reasoning_source=reasoning_source,
        cache_mode=cache_mode,
        cache_source=cache_source,
        transport_mode=transport_mode,
        transport_source=transport_source,
        stream_enabled=stream_enabled,
        stream_source=stream_source,
    )


def default_context_k_for_runtime(runtime: str) -> int:
    runtime_name = runtime.strip().lower()
    runtime_infra, _, _ = _load_bench_config()
    max_context_tokens = int(runtime_infra.get(runtime_name, {}).get("max_context_tokens", 256000))
    return max(1, int(max_context_tokens / 1000))
