from __future__ import annotations

from typing import Any, Dict

import requests


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
