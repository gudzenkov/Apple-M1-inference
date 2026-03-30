from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, TypedDict

import yaml

MODEL_ALIASES_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


class ModelEntry(TypedDict):
    key: str
    model: str
    runtime: str
    family: str
    profile: str
    size: int | None
    quant: int | None
    aliases: List[str]
    capabilities: Dict[str, Any]


class ProfileEntry(TypedDict):
    key: str
    runtime: str
    family: str
    defaults: Dict[str, Any]
    capabilities: Dict[str, Any]
    reasoning: Dict[str, Any]


def _expect_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return value


def _normalize_alias(alias: Any, *, context: str) -> str:
    if not isinstance(alias, str):
        raise RuntimeError(f"{context} alias must be a string")
    normalized = alias.strip().lower()
    if not normalized:
        raise RuntimeError(f"{context} alias must be non-empty")
    return normalized


def _as_int_or_none(value: Any, context: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be an integer or null")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise RuntimeError(f"{context} must be an integer or null")


def _runtime_candidates(runtime: str) -> set[str]:
    return {runtime.strip().lower()}


def _runtime_matches(entry_runtime: str, runtime: str | None) -> bool:
    if runtime is None:
        return True
    return entry_runtime in _runtime_candidates(runtime)


@lru_cache(maxsize=4)
def _load_models_bundle(path: Path = MODEL_ALIASES_PATH) -> tuple[List[ModelEntry], Dict[str, ProfileEntry]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    root = _expect_mapping(raw, f"{path}")

    defaults = root.get("defaults") or {}
    defaults_map = _expect_mapping(defaults, f"{path}: defaults")
    families_raw = defaults_map.get("families") or {}
    families_map = _expect_mapping(families_raw, f"{path}: defaults.families")

    family_defaults_caps: Dict[str, Dict[str, Any]] = {}
    family_defaults_runtime: Dict[str, Dict[str, Any]] = {}
    for family_key, family_value in families_map.items():
        family = str(family_key).strip().lower()
        if not family:
            raise RuntimeError(f"{path}: family key must be non-empty")
        family_conf = _expect_mapping(family_value, f"{path}: defaults.families.{family}")

        capabilities = family_conf.get("capabilities") or {}
        family_defaults_caps[family] = dict(_expect_mapping(
            capabilities,
            f"{path}: defaults.families.{family}.capabilities",
        ))

        runtime_defaults = {
            key: value
            for key, value in family_conf.items()
            if key != "capabilities"
        }
        family_defaults_runtime[family] = dict(runtime_defaults)

    profiles_raw = root.get("profiles") or {}
    profiles_map = _expect_mapping(profiles_raw, f"{path}: profiles")

    profiles: Dict[str, ProfileEntry] = {}
    for profile_key_raw, profile_value in profiles_map.items():
        profile_key = str(profile_key_raw).strip().lower()
        if not profile_key:
            raise RuntimeError(f"{path}: profile key must be non-empty")

        profile_conf = _expect_mapping(profile_value, f"{path}: profiles.{profile_key}")
        runtime = str(profile_conf.get("runtime") or "").strip().lower()
        family = str(profile_conf.get("family") or "").strip().lower()

        if not runtime or not family:
            if "/" in profile_key:
                key_runtime, key_family = profile_key.split("/", 1)
                runtime = runtime or key_runtime.strip().lower()
                family = family or key_family.strip().lower()
        if not runtime or not family:
            raise RuntimeError(
                f"{path}: profile '{profile_key}' must define runtime/family "
                "or use '<runtime>/<family>' profile key"
            )

        defaults_conf = profile_conf.get("defaults") or {}
        capabilities_conf = profile_conf.get("capabilities") or {}
        reasoning_conf = profile_conf.get("reasoning") or {}

        defaults_dict = dict(_expect_mapping(defaults_conf, f"{path}: profiles.{profile_key}.defaults"))
        capabilities_dict = dict(_expect_mapping(
            capabilities_conf,
            f"{path}: profiles.{profile_key}.capabilities",
        ))
        reasoning_dict = dict(_expect_mapping(reasoning_conf, f"{path}: profiles.{profile_key}.reasoning"))

        merged_caps = dict(family_defaults_caps.get(family, {}))
        merged_caps.update(capabilities_dict)
        if "supported" in reasoning_dict and "reasoning_supported" not in merged_caps:
            merged_caps["reasoning_supported"] = bool(reasoning_dict.get("supported"))
        if "format" in reasoning_dict and "reasoning_format" not in merged_caps:
            merged_caps["reasoning_format"] = reasoning_dict.get("format")

        merged_defaults = dict(family_defaults_runtime.get(family, {}))
        merged_defaults.update(defaults_dict)

        profiles[profile_key] = {
            "key": profile_key,
            "runtime": runtime,
            "family": family,
            "defaults": merged_defaults,
            "capabilities": merged_caps,
            "reasoning": reasoning_dict,
        }

    models_raw = root.get("models")
    if not isinstance(models_raw, list) or not models_raw:
        raise RuntimeError(f"{path}: models must be a non-empty list")

    entries: List[ModelEntry] = []
    seen_keys: set[str] = set()
    for i, raw_item in enumerate(models_raw):
        item_context = f"{path}: models[{i}]"
        item = _expect_mapping(raw_item, item_context)
        if len(item) != 1:
            raise RuntimeError(f"{item_context} must have exactly one key")

        key_raw, conf_raw = next(iter(item.items()))
        key = str(key_raw).strip().lower()
        if not key:
            raise RuntimeError(f"{item_context} model key must be non-empty")
        if key in seen_keys:
            raise RuntimeError(f"{path}: duplicate model key '{key}'")
        seen_keys.add(key)

        conf = _expect_mapping(conf_raw, f"{item_context}.{key}")

        model_conf = conf.get("model")
        if isinstance(model_conf, str):
            model_name = model_conf.strip()
            size = None
            quant = None
        else:
            model_map = _expect_mapping(model_conf, f"{item_context}.{key}.model")
            model_name = str(model_map.get("name") or "").strip()
            size = _as_int_or_none(model_map.get("size"), f"{item_context}.{key}.model.size")
            quant = _as_int_or_none(model_map.get("quant"), f"{item_context}.{key}.model.quant")
        if not model_name:
            raise RuntimeError(f"{item_context}.{key}.model.name must be non-empty")

        runtime_conf = conf.get("runtime")
        runtime: str
        if isinstance(runtime_conf, str):
            runtime = runtime_conf.strip().lower()
        else:
            runtime_map = _expect_mapping(runtime_conf, f"{item_context}.{key}.runtime")
            runtime = str(runtime_map.get("server") or "").strip().lower()
        if not runtime:
            raise RuntimeError(f"{item_context}.{key}.runtime.server must be non-empty")

        family = str(conf.get("family") or "").strip().lower()
        if not family:
            raise RuntimeError(f"{item_context}.{key}.family must be non-empty")

        profile = str(conf.get("profile") or f"{runtime}/{family}").strip().lower()
        if profile:
            profile_entry = profiles.get(profile)
            if profile_entry is not None:
                if profile_entry["runtime"] != runtime or profile_entry["family"] != family:
                    raise RuntimeError(
                        f"{item_context}.{key}.profile '{profile}' runtime/family mismatch"
                    )

        aliases_raw = conf.get("aliases") or []
        if not isinstance(aliases_raw, list):
            raise RuntimeError(f"{item_context}.{key}.aliases must be a list")

        aliases = [key]
        aliases.extend(
            _normalize_alias(alias, context=f"{item_context}.{key}")
            for alias in aliases_raw
        )
        aliases = list(dict.fromkeys(aliases))

        caps_conf = conf.get("capabilities") or {}
        caps_map = dict(_expect_mapping(caps_conf, f"{item_context}.{key}.capabilities"))

        merged_caps = dict(family_defaults_caps.get(family, {}))
        profile_caps = profiles.get(profile, {}).get("capabilities", {})  # type: ignore[union-attr]
        if isinstance(profile_caps, Mapping):
            merged_caps.update(dict(profile_caps))
        merged_caps.update(caps_map)

        reasoning_from_profile = profiles.get(profile, {}).get("reasoning", {})  # type: ignore[union-attr]
        if isinstance(reasoning_from_profile, Mapping):
            if "supported" in reasoning_from_profile and "reasoning_supported" not in merged_caps:
                merged_caps["reasoning_supported"] = bool(reasoning_from_profile.get("supported"))
            if "format" in reasoning_from_profile and "reasoning_format" not in merged_caps:
                merged_caps["reasoning_format"] = reasoning_from_profile.get("format")

        entries.append(
            {
                "key": key,
                "model": model_name,
                "runtime": runtime,
                "family": family,
                "profile": profile,
                "size": size,
                "quant": quant,
                "aliases": aliases,
                "capabilities": merged_caps,
            }
        )

    if not entries:
        raise RuntimeError(f"No model entries loaded from {path}")

    alias_seen: Dict[str, str] = {}
    for entry in entries:
        for alias in entry["aliases"]:
            existing_model = alias_seen.get(alias)
            if existing_model is not None and existing_model != entry["model"]:
                raise RuntimeError(
                    f"Duplicate alias '{alias}' maps to multiple models in {path}"
                )
            alias_seen[alias] = entry["model"]

    return entries, profiles


@lru_cache(maxsize=1)
def load_models(path: Path = MODEL_ALIASES_PATH) -> List[ModelEntry]:
    entries, _ = _load_models_bundle(path)
    return list(entries)


@lru_cache(maxsize=1)
def load_profiles(path: Path = MODEL_ALIASES_PATH) -> Dict[str, ProfileEntry]:
    _, profiles = _load_models_bundle(path)
    return dict(profiles)


@lru_cache(maxsize=16)
def load_model_aliases(path: Path = MODEL_ALIASES_PATH, runtime: str | None = None) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for entry in load_models(path):
        if not _runtime_matches(entry["runtime"], runtime):
            continue
        for alias in entry["aliases"]:
            if alias in aliases and aliases[alias] != entry["model"]:
                raise RuntimeError(
                    f"Duplicate alias '{alias}' maps to multiple models in {path}"
                )
            aliases[alias] = entry["model"]

    if runtime is not None and not aliases:
        raise RuntimeError(f"No model aliases loaded for runtime '{runtime}' from {path}")

    return aliases


def get_models_for_runtime(runtime: str) -> List[str]:
    selected: List[str] = []
    seen: set[str] = set()
    for entry in load_models():
        if not _runtime_matches(entry["runtime"], runtime):
            continue
        model_id = entry["model"]
        if model_id not in seen:
            selected.append(model_id)
            seen.add(model_id)
    return selected


def get_model_key(model_id: str, runtime: str | None = None) -> str:
    target = model_id.strip()
    if not target:
        raise ValueError("Model id is empty")
    for entry in load_models():
        if entry["model"] != target:
            continue
        if _runtime_matches(entry["runtime"], runtime):
            return entry["key"]
    for entry in load_models():
        if entry["model"] == target:
            return entry["key"]
    raise ValueError(f"Model id '{model_id}' is not configured in {MODEL_ALIASES_PATH}")


def get_default_model_id(runtime: str | None = None) -> str:
    if runtime is None:
        return load_models()[0]["model"]
    runtime_models = get_models_for_runtime(runtime)
    if not runtime_models:
        raise RuntimeError(f"No configured models for runtime '{runtime}'")
    return runtime_models[0]


def get_model_entry(model: str, runtime: str | None = None) -> ModelEntry:
    resolved_model = resolve_model_reference(model, runtime=runtime)
    candidates = [entry for entry in load_models() if entry["model"] == resolved_model]
    if runtime is not None:
        runtime_candidates = [entry for entry in candidates if _runtime_matches(entry["runtime"], runtime)]
        if runtime_candidates:
            return runtime_candidates[0]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            f"Model '{resolved_model}' is configured for multiple runtimes: "
            f"{sorted({entry['runtime'] for entry in candidates})}"
        )
    raise ValueError(f"Model '{model}' is not configured in {MODEL_ALIASES_PATH}")


def get_profile_for_model(model: str, runtime: str | None = None) -> ProfileEntry | None:
    entry = get_model_entry(model, runtime=runtime)
    return load_profiles().get(entry["profile"])


def get_capabilities_for_model(model: str, runtime: str | None = None) -> Dict[str, Any]:
    entry = get_model_entry(model, runtime=runtime)
    return dict(entry.get("capabilities", {}))


def resolve_model_reference(model: str, runtime: str | None = None) -> str:
    model_ref = model.strip()
    if not model_ref:
        raise ValueError("Model reference is empty")

    runtime_models = set(get_models_for_runtime(runtime)) if runtime is not None else None
    all_models = {entry["model"] for entry in load_models()}

    if runtime_models is not None and not runtime_models:
        raise ValueError(f"No configured models for runtime '{runtime}'")

    if runtime_models is not None:
        if model_ref in runtime_models:
            return model_ref
    elif model_ref in all_models:
        return model_ref

    aliases = load_model_aliases(runtime=runtime)
    resolved = aliases.get(model_ref.lower())
    if resolved:
        return resolved

    if runtime is not None and model_ref in all_models:
        raise ValueError(
            f"Model '{model_ref}' is configured but not for runtime '{runtime}'."
        )

    raise ValueError(
        f"Unknown model '{model_ref}'. Use configured full model id or alias from {MODEL_ALIASES_PATH}."
    )


def resolve_runtime_for_model_reference(model: str) -> str:
    model_ref = model.strip()
    if not model_ref:
        raise ValueError("Model reference is empty")

    entries = load_models()

    exact_matches = [entry for entry in entries if entry["model"] == model_ref]
    if exact_matches:
        runtimes = {entry["runtime"] for entry in exact_matches}
        if len(runtimes) > 1:
            raise ValueError(
                f"Model '{model_ref}' is configured for multiple runtimes: {sorted(runtimes)}"
            )
        return exact_matches[0]["runtime"]

    ref_lower = model_ref.lower()
    alias_matches = [entry for entry in entries if ref_lower in entry["aliases"]]
    if alias_matches:
        runtimes = {entry["runtime"] for entry in alias_matches}
        if len(runtimes) > 1:
            raise ValueError(
                f"Alias '{model_ref}' is configured for multiple runtimes: {sorted(runtimes)}"
            )
        return alias_matches[0]["runtime"]

    raise ValueError(
        f"Unknown model '{model_ref}'. Use configured full model id or alias from {MODEL_ALIASES_PATH}."
    )


def resolve_model_alias(model: str) -> str:
    return resolve_model_reference(model=model)
