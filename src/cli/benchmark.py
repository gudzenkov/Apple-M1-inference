from __future__ import annotations

import argparse
import sys

from src.bench.dataset_tools import DEFAULT_DATASET_MD
from src.bench.datasets import normalize_dataset_mode, parse_context_list
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
        choices=["mlx", "mlx-optiq", "both", "ollama", "all"],
        default="both",
        help="Runtime/server selection (default: both = mlx + mlx-optiq)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model alias/key or configured full ID (overrides runtime default)",
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
        default=2000,
        help="Per-request timeout in seconds (default: 2000)",
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
        "--use-prompt-cache",
        action="store_true",
        help=(
            "Use server-side prompt cache for long-mode cases by pre-filling shared context "
            "once and querying suffix prompts against that cached prefix."
        ),
    )
    parser.add_argument(
        "--server-start-timeout",
        type=int,
        default=300,
        help="Seconds to wait for managed MLX server readiness (default: 300)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.samples < 3:
        raise SystemExit("--samples must be >= 3")

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
