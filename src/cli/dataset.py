from __future__ import annotations

import argparse
from pathlib import Path

from src.bench.dataset_tools import (
    DEFAULT_DATASET_HTML,
    DEFAULT_DATASET_MD,
    TURBOQUANT_PAPER_URL,
    fetch_and_parse,
    fetch_html,
    parse_html_to_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset", description="Dataset fetch/parse utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch_parser = subparsers.add_parser("fetch", help="Fetch dataset HTML")
    fetch_parser.add_argument("--url", default=TURBOQUANT_PAPER_URL, help="Source URL")
    fetch_parser.add_argument("--output-html", default=str(DEFAULT_DATASET_HTML), help="Output HTML path")
    fetch_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")

    parse_parser = subparsers.add_parser("parse", help="Parse HTML into markdown dataset")
    parse_parser.add_argument("--input-html", default=str(DEFAULT_DATASET_HTML), help="Input HTML path")
    parse_parser.add_argument("--output-md", default=str(DEFAULT_DATASET_MD), help="Output markdown path")
    parse_parser.add_argument("--source-url", default=TURBOQUANT_PAPER_URL, help="Source URL metadata")

    fetch_parse_parser = subparsers.add_parser("fetch-parse", help="Fetch URL and parse to markdown")
    fetch_parse_parser.add_argument("--url", default=TURBOQUANT_PAPER_URL, help="Source URL")
    fetch_parse_parser.add_argument("--output-md", default=str(DEFAULT_DATASET_MD), help="Output markdown path")
    fetch_parse_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "fetch":
        out = fetch_html(args.url, Path(args.output_html), timeout_sec=args.timeout)
        print(out)
        return 0

    if args.command == "parse":
        out = parse_html_to_markdown(
            input_html=Path(args.input_html),
            output_md=Path(args.output_md),
            source_url=args.source_url,
        )
        print(out)
        return 0

    if args.command == "fetch-parse":
        out = fetch_and_parse(
            url=args.url,
            output_md=Path(args.output_md),
            timeout_sec=args.timeout,
        )
        print(out)
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
