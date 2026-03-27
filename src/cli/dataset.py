from __future__ import annotations

import argparse
from pathlib import Path

from src.bench.dataset_tools import (
    fetch_and_parse,
    fetch_html,
    get_dataset_source,
    load_dataset_sources,
    parse_html_to_markdown,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset", description="Dataset fetch/parse utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    source_names = [source.name for source in load_dataset_sources()]

    fetch_parser = subparsers.add_parser("fetch", help="Fetch dataset HTML")
    fetch_parser.add_argument("--source", choices=source_names, default=None, help="Source key from dataset config")
    fetch_parser.add_argument("--url", default=None, help="Source URL (overrides source config)")
    fetch_parser.add_argument("--output-html", default=None, help="Output HTML path (overrides source config)")
    fetch_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")

    parse_parser = subparsers.add_parser("parse", help="Parse HTML into markdown dataset")
    parse_parser.add_argument("--source", choices=source_names, default=None, help="Source key from dataset config")
    parse_parser.add_argument("--input-html", default=None, help="Input HTML path (overrides source config)")
    parse_parser.add_argument("--output-md", default=None, help="Output markdown path (overrides source config)")
    parse_parser.add_argument("--source-url", default=None, help="Source URL metadata (overrides source config)")

    fetch_parse_parser = subparsers.add_parser("fetch-parse", help="Fetch URL and parse to markdown")
    fetch_parse_parser.add_argument(
        "--source", choices=source_names, default=None, help="Source key from dataset config"
    )
    fetch_parse_parser.add_argument("--url", default=None, help="Source URL (overrides source config)")
    fetch_parse_parser.add_argument("--output-html", default=None, help="Output HTML path (overrides source config)")
    fetch_parse_parser.add_argument("--output-md", default=None, help="Output markdown path (overrides source config)")
    fetch_parse_parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout seconds")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    source = get_dataset_source(args.source)

    if args.command == "fetch":
        url = args.url or source.url
        output_html = Path(args.output_html) if args.output_html else source.html_path
        out = fetch_html(url, output_html, timeout_sec=args.timeout)
        print(out)
        return 0

    if args.command == "parse":
        input_html = Path(args.input_html) if args.input_html else source.html_path
        output_md = Path(args.output_md) if args.output_md else source.md_path
        source_url = args.source_url or source.url
        out = parse_html_to_markdown(
            input_html=input_html,
            output_md=output_md,
            source_url=source_url,
        )
        print(out)
        return 0

    if args.command == "fetch-parse":
        url = args.url or source.url
        output_md = Path(args.output_md) if args.output_md else source.md_path
        output_html = Path(args.output_html) if args.output_html else source.html_path
        out = fetch_and_parse(
            url=url,
            output_md=output_md,
            output_html=output_html,
            timeout_sec=args.timeout,
        )
        print(out)
        return 0

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
