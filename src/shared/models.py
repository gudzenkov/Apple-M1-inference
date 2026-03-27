from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Dict, List, TypedDict

MODEL_ALIASES_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


class ModelEntry(TypedDict):
    key: str
    model: str
    runtime: str
    size: int | None
    quant: int | None
    aliases: List[str]


def _strip_inline_comment(line: str) -> str:
    if "#" not in line:
        return line
    return line.split("#", 1)[0]


def _unquote(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _parse_optional_int(value: str) -> int | None:
    value = _unquote(value).strip()
    if not value:
        return None
    return int(value)


def _runtime_candidates(runtime: str) -> set[str]:
    normalized = runtime.strip().lower()
    if normalized == "mlx-optiq":
        return {"mlx-optiq", "mlx"}
    return {normalized}


def _runtime_matches(entry_runtime: str, runtime: str | None) -> bool:
    if runtime is None:
        return True
    return entry_runtime in _runtime_candidates(runtime)


@lru_cache(maxsize=1)
def load_models(path: Path = MODEL_ALIASES_PATH) -> List[ModelEntry]:
    text = path.read_text(encoding="utf-8")
    entries: List[ModelEntry] = []
    in_models = False

    current_key: str | None = None
    current_model: str | None = None
    current_runtime: str | None = None
    current_size: int | None = None
    current_quant: int | None = None
    current_aliases: List[str] = []
    aliases_indent: int | None = None

    def flush_current() -> None:
        nonlocal current_key, current_model, current_runtime, current_size, current_quant
        nonlocal current_aliases, aliases_indent
        if current_key is None:
            return
        if not current_model:
            raise RuntimeError(
                f"Model entry '{current_key}' is missing required 'model' field in {path}"
            )
        if not current_runtime:
            raise RuntimeError(
                f"Model entry '{current_key}' is missing required 'runtime' field in {path}"
            )
        aliases = list(dict.fromkeys([current_key, *current_aliases]))
        entries.append(
            {
                "key": current_key,
                "model": current_model,
                "runtime": current_runtime,
                "size": current_size,
                "quant": current_quant,
                "aliases": aliases,
            }
        )
        current_key = None
        current_model = None
        current_runtime = None
        current_size = None
        current_quant = None
        current_aliases = []
        aliases_indent = None

    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue

        stripped = line.strip()

        if not in_models:
            if stripped == "models:":
                in_models = True
            continue

        model_entry_match = re.match(r"^\s*-\s*([A-Za-z0-9_.-]+)\s*:\s*$", line)
        if model_entry_match:
            flush_current()
            current_key = model_entry_match.group(1).strip().lower()
            current_model = None
            current_runtime = None
            current_size = None
            current_quant = None
            current_aliases = []
            aliases_indent = None
            continue

        if current_key is None:
            continue

        property_match = re.match(r"^(\s+)([A-Za-z0-9_.-]+)\s*:\s*(.*?)\s*$", line)
        if not property_match:
            if aliases_indent is not None:
                alias_match = re.match(r"^(\s*)-\s*(.+?)\s*$", line)
                if alias_match and len(alias_match.group(1)) > aliases_indent:
                    alias = _unquote(alias_match.group(2)).strip().lower()
                    if alias:
                        current_aliases.append(alias)
                    continue
            aliases_indent = None
            continue

        field_indent = len(property_match.group(1))
        field = property_match.group(2).strip().lower()
        value = property_match.group(3).strip()

        if field == "model":
            current_model = _unquote(value)
            continue

        if field == "runtime":
            current_runtime = _unquote(value).strip().lower()
            continue

        if field == "size":
            current_size = _parse_optional_int(value)
            continue

        if field == "quant":
            current_quant = _parse_optional_int(value)
            continue

        if field == "aliases":
            aliases_indent = field_indent
            continue
        aliases_indent = None

    flush_current()

    if not entries:
        raise RuntimeError(f"No model entries loaded from {path}")

    return entries


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


def resolve_model_alias(model: str) -> str:
    return resolve_model_reference(model=model)
