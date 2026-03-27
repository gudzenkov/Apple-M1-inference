from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
from html_to_markdown import ConversionOptions, convert
import yaml

DATASET_CONFIG_PATH = Path("dataset/dataset.yaml")
DEFAULT_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass(frozen=True)
class DatasetSource:
    name: str
    url: str
    html_path: Path
    md_path: Path
    default: bool = False


def load_dataset_sources(config_path: Path = DATASET_CONFIG_PATH) -> list[DatasetSource]:
    if not config_path.exists():
        raise RuntimeError(f"Dataset config not found: {config_path}")

    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    raw_sources = raw_config.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise RuntimeError(f"Dataset config has no sources: {config_path}")

    sources: list[DatasetSource] = []
    for entry in raw_sources:
        if not isinstance(entry, dict):
            raise RuntimeError(f"Invalid source entry in {config_path}: {entry!r}")
        name = str(entry.get("name", "")).strip()
        url = str(entry.get("url", "")).strip()
        html_path = str(entry.get("html_path", "")).strip()
        md_path = str(entry.get("md_path", "")).strip()
        if not name or not url or not html_path or not md_path:
            raise RuntimeError(
                "Each source in dataset config must define name, url, html_path, md_path"
            )
        sources.append(
            DatasetSource(
                name=name,
                url=url,
                html_path=Path(html_path),
                md_path=Path(md_path),
                default=bool(entry.get("default", False)),
            )
        )
    return sources


def get_dataset_source(name: str | None = None, config_path: Path = DATASET_CONFIG_PATH) -> DatasetSource:
    sources = load_dataset_sources(config_path=config_path)
    if name:
        for source in sources:
            if source.name == name:
                return source
        available = ", ".join(source.name for source in sources)
        raise RuntimeError(f"Unknown dataset source '{name}'. Available: {available}")

    default_sources = [source for source in sources if source.default]
    if len(default_sources) > 1:
        raise RuntimeError(f"Multiple default dataset sources in {config_path}")
    if len(default_sources) == 1:
        return default_sources[0]
    return sources[0]


def default_dataset_markdown_path(config_path: Path = DATASET_CONFIG_PATH) -> Path:
    return get_dataset_source(config_path=config_path).md_path


_DEFAULT_SOURCE = get_dataset_source()
DEFAULT_DATASET_HTML = _DEFAULT_SOURCE.html_path
DEFAULT_DATASET_MD = _DEFAULT_SOURCE.md_path


def _html_to_markdown(html_text: str) -> str:
    options = ConversionOptions(
        extract_metadata=False,
    )
    markdown = convert(html_text, options=options).strip()
    if not markdown:
        raise RuntimeError("Parsed text is empty")
    return markdown


def fetch_html(url: str, output_html: Path, timeout_sec: int = 60) -> Path:
    output_html.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, timeout=timeout_sec, headers=DEFAULT_FETCH_HEADERS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch dataset URL {url}: {exc}") from exc

    if response.encoding is None:
        response.encoding = response.apparent_encoding
    output_html.write_text(response.text, encoding="utf-8")
    return output_html


def parse_html_to_markdown(input_html: Path, output_md: Path, source_url: str | None = None) -> Path:
    if not input_html.exists():
        raise RuntimeError(f"Input HTML not found: {input_html}")

    html_text = input_html.read_text(encoding="utf-8")
    parsed_text = _html_to_markdown(html_text)

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


def fetch_and_parse(
    url: str,
    output_md: Path,
    output_html: Path | None = None,
    timeout_sec: int = 60,
) -> Path:
    html_path = output_html or output_md.with_suffix(".html")
    fetch_html(url=url, output_html=html_path, timeout_sec=timeout_sec)
    return parse_html_to_markdown(input_html=html_path, output_md=output_md, source_url=url)
