from __future__ import annotations

import argparse

from src.bench.dataset_tools import DEFAULT_DATASET_MD
from src.bench.runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="benchmark", description="Benchmark local LLM performance")
    parser.add_argument(
        "--dataset",
        choices=["quick", "long", "all"],
        default="all",
        help="Dataset profile: quick (32k), long (256k), or all",
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
        help="Samples per selected dataset profile (default: 5)",
    )
    parser.add_argument(
        "--dataset-file",
        default=str(DEFAULT_DATASET_MD),
        help="Markdown dataset file used by long/all dataset modes",
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
        default=600,
        help="Per-request timeout in seconds (default: 600)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL file (default: results/benchmark_<dataset>.jsonl)",
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
        "--server-start-timeout",
        type=int,
        default=300,
        help="Seconds to wait for managed MLX server readiness (default: 300)",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.samples <= 0:
        raise SystemExit("--samples must be > 0")

    return run_benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
