from __future__ import annotations

import argparse
import sys

from src.bench.dataset import DEFAULT_DATASET_MD, normalize_dataset_mode, parse_context_list
from src.bench.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark", description="Benchmark local LLM performance")
    parser.add_argument(
        "--dataset",
        choices=["short", "long", "all", "quick", "context"],
        default="all",
        help=(
            "Dataset profile: short (8k), long (variable context via --context, default 64k), or all. "
            "Legacy aliases: quick->short, context->long"
        ),
    )
    parser.add_argument(
        "--context",
        default=None,
        help=(
            "Comma-separated context sizes in k-tokens (overrides --dataset). "
            "Allowed: 8,16,32,64,128,256 (with optional 'k' suffix)"
        ),
    )
    parser.add_argument(
        "--runtime",
        choices=["auto", "mlx", "mlx-optiq", "llama.cpp"],
        default="auto",
        help=(
            "Runtime/server selection. Default: auto (resolve from --model via configs/models.yaml; "
            "falls back to first configured model runtime)."
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model alias/key from configs/models.yaml or full model ID."
        ),
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help=(
            "Samples per selected dataset profile (default: 5, minimum: 3). "
            "For long mode this is also the number of retrieval needles/queries per context."
        ),
    )
    parser.add_argument(
        "--dataset-file",
        default=str(DEFAULT_DATASET_MD),
        help="Markdown dataset file used by long-context/all dataset modes",
    )
    parser.add_argument(
        "--prompt",
        help="Adhoc custom prompt (bypasses dataset case generation)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Max tokens for custom --prompt mode (default: 100)",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=None,
        help="Override per-request timeout in seconds (default comes from configs/bench.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output JSONL file. Default is auto-generated under "
            "results/<experiment-group>/<utc-timestamp>/benchmark_<model>_<context>_s<samples>.jsonl"
        ),
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark all configured models for selected runtimes",
    )
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip warmup request",
    )
    parser.add_argument(
        "--reasoning-mode",
        choices=["auto", "off", "on"],
        default="auto",
        help="Reasoning mode (default: auto, resolved from composed runtime/profile policy).",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["auto", "prefill", "request", "none"],
        default="auto",
        help="Cache mode (default: auto, resolved from composed runtime policy).",
    )
    parser.add_argument(
        "--stream",
        choices=["auto", "on", "off"],
        default="auto",
        help="Streaming mode (default: auto, resolved from composed runtime policy).",
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "openai-compat"],
        default="auto",
        help="Transport mode (default: auto, resolved from composed runtime policy).",
    )
    parser.add_argument(
        "--server-start-timeout",
        type=int,
        default=None,
        help="Override seconds to wait for managed benchmark server readiness",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.samples < 3:
        raise SystemExit("--samples must be >= 3")
    if args.request_timeout is not None and args.request_timeout <= 0:
        raise SystemExit("--request-timeout must be > 0")
    if args.server_start_timeout is not None and args.server_start_timeout <= 0:
        raise SystemExit("--server-start-timeout must be > 0")

    raw_dataset = str(args.dataset)
    args.dataset = normalize_dataset_mode(raw_dataset)
    if raw_dataset != args.dataset:
        print(
            f"Warning: --dataset {raw_dataset!r} is deprecated; using {args.dataset!r}",
            file=sys.stderr,
        )

    if args.context:
        try:
            args.contexts_k = parse_context_list(args.context)
        except RuntimeError as exc:
            raise SystemExit(str(exc))
    else:
        args.contexts_k = None

    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
