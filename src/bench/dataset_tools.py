from __future__ import annotations

from pathlib import Path
import re
from html import unescape

import requests

TURBOQUANT_PAPER_URL = "https://arxiv.org/html/2504.19874v1"
DEFAULT_DATASET_HTML = Path("dataset/turboquant_2504_19874v1.html")
DEFAULT_DATASET_MD = Path("dataset/turboquant_2504_19874v1.md")


def _html_to_text(html_text: str) -> str:
    html_text = re.sub(r"<script.*?>.*?</script>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
    html_text = re.sub(r"<style.*?>.*?</style>", " ", html_text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", html_text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise RuntimeError("Parsed text is empty")
    return text


def fetch_html(url: str, output_html: Path, timeout_sec: int = 60) -> Path:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=timeout_sec)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch dataset URL {url}: {exc}") from exc

    output_html.write_text(response.text, encoding="utf-8")
    return output_html


def parse_html_to_markdown(input_html: Path, output_md: Path, source_url: str | None = None) -> Path:
    if not input_html.exists():
        raise RuntimeError(f"Input HTML not found: {input_html}")

    html_text = input_html.read_text(encoding="utf-8")
    parsed_text = _html_to_text(html_text)

    source = source_url or "unknown"
    markdown = (
        "# TurboQuant Paper Dataset Cache\n\n"
        f"Source: {source}\n\n"
        "## Extracted Text\n\n"
        f"{parsed_text}\n"
    )

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")
    return output_md


def fetch_and_parse(url: str, output_md: Path, timeout_sec: int = 60) -> Path:
    html_path = DEFAULT_DATASET_HTML
    fetch_html(url=url, output_html=html_path, timeout_sec=timeout_sec)
    return parse_html_to_markdown(input_html=html_path, output_md=output_md, source_url=url)
