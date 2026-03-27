from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import requests

from src.shared.models import get_default_model_id, resolve_model_reference


def _default_base_url(server: str) -> str:
    host = os.getenv("HOST", "127.0.0.1")
    if server == "mlx":
        port = os.getenv("MLX_PORT", "8000")
    else:
        port = os.getenv("OPTIQ_PORT", "8080")
    return f"http://{host}:{port}"


def _runtime_for_server(server: str) -> str:
    if server == "optiq":
        return "mlx-optiq"
    return "mlx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-cli",
        description="CLI client for local MLX OpenAI-compatible servers",
    )
    parser.add_argument(
        "--server",
        choices=["mlx", "optiq"],
        default="optiq",
        help="Server target (default: optiq)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override server base URL (default from HOST/MLX_PORT/OPTIQ_PORT env)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models from /v1/models",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=None,
        help="Model alias/key or configured full ID",
    )
    parser.add_argument(
        "-p",
        "--prompt",
        default=None,
        help="Prompt text to send to /v1/chat/completions",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max completion tokens (default: 256)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="HTTP timeout in seconds (default: 180)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw JSON response",
    )
    return parser


def _print_json(data: Dict[str, Any]) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _resolve_model(args: argparse.Namespace) -> str:
    runtime = _runtime_for_server(args.server)
    model_ref = args.model or get_default_model_id(runtime=runtime)
    return resolve_model_reference(model_ref, runtime=runtime)


def _request_models(base_url: str, timeout_sec: int) -> Dict[str, Any]:
    resp = requests.get(f"{base_url}/v1/models", timeout=timeout_sec)
    resp.raise_for_status()
    return resp.json()


def _request_chat(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    resp = requests.post(
        f"{base_url}/v1/chat/completions",
        json=payload,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.max_tokens <= 0:
        raise SystemExit("--max-tokens must be > 0")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be > 0")
    if not args.list_models and not args.prompt:
        raise SystemExit("Specify --list-models or provide -p/--prompt")

    base_url = (args.base_url or _default_base_url(args.server)).rstrip("/")

    try:
        if args.list_models:
            models_data = _request_models(base_url=base_url, timeout_sec=args.timeout)
            if args.json:
                _print_json(models_data)
            else:
                for item in models_data.get("data", []):
                    model_id = item.get("id")
                    if model_id:
                        print(model_id)
            if not args.prompt:
                return 0

        model_id = _resolve_model(args)
        response = _request_chat(
            base_url=base_url,
            model=model_id,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout_sec=args.timeout,
        )
        if args.json:
            _print_json(response)
            return 0

        choices = response.get("choices") or []
        if not choices:
            raise RuntimeError("No choices in response")
        message = choices[0].get("message", {})
        content = message.get("content", "")
        print(content)
        return 0
    except requests.HTTPError as exc:
        body = exc.response.text if exc.response is not None else ""
        print(f"HTTP error: {exc}", file=sys.stderr)
        if body:
            print(body, file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
