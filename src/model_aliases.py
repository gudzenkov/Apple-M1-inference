from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import re
from typing import Dict

MODEL_ALIASES_PATH = Path(__file__).resolve().parents[1] / "configs" / "model_aliases.yaml"


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


@lru_cache(maxsize=1)
def load_model_aliases(path: Path = MODEL_ALIASES_PATH) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8")
    aliases: Dict[str, str] = {}
    in_aliases = False

    for raw_line in text.splitlines():
        line = _strip_inline_comment(raw_line).rstrip()
        if not line.strip():
            continue

        if line.strip() == "aliases:":
            in_aliases = True
            continue

        if not in_aliases:
            continue

        match = re.match(r"^\s{2}([A-Za-z0-9_.-]+)\s*:\s*(.+?)\s*$", line)
        if not match:
            continue

        alias = match.group(1).strip().lower()
        model_id = _unquote(match.group(2))
        aliases[alias] = model_id

    if not aliases:
        raise RuntimeError(f"No aliases loaded from {path}")

    return aliases


def get_default_model_id() -> str:
    aliases = load_model_aliases()
    return next(iter(aliases.values()))


def resolve_model_alias(model: str) -> str:
    aliases = load_model_aliases()
    return aliases.get(model.strip().lower(), model)
