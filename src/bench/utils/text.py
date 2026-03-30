from __future__ import annotations

import re


def slug(value: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip())
    out = re.sub(r"-{2,}", "-", out).strip("-._")
    return out or "x"
