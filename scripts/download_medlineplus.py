#!/usr/bin/env python3
"""Download raw MedlinePlus source data into data/raw/medlineplus."""

import io
from pathlib import Path
import re
from zipfile import ZipFile

import requests

INDEX_URL = "https://medlineplus.gov/xml.html"
FALLBACK_CANDIDATES = [
    "https://medlineplus.gov/xml/mplus_topics_latest.zip",
    "https://medlineplus.gov/xml/mplus_topics_2025-02-26.zip",
]
OUT_DIR = Path("data/raw/medlineplus")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_XML = OUT_DIR / "medlineplus.xml"
OUT_ZIP = OUT_DIR / "medlineplus_topics.zip"

def _download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    return r.content

def _extract_xml_from_zip(zip_bytes: bytes) -> bytes:
    with ZipFile(io.BytesIO(zip_bytes)) as zf:
        xml_names = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_names:
            raise RuntimeError("Zip archive does not contain an XML file.")
        with zf.open(xml_names[0]) as f:
            return f.read()

def _discover_links_from_index() -> list[str]:
    html = _download_bytes(INDEX_URL).decode("utf-8", errors="ignore")
    links = set()
    for match in re.findall(r'href="([^"]+)"', html):
        if "mplus_topics" in match and (match.endswith(".zip") or match.endswith(".xml")):
            if match.startswith("http"):
                links.add(match)
            else:
                links.add(f"https://medlineplus.gov{match}")
    return list(links)

def main() -> None:
    candidates = _discover_links_from_index() + FALLBACK_CANDIDATES
    last_error: Exception | None = None

    for url in candidates:
        try:
            data = _download_bytes(url)
            if url.lower().endswith(".zip"):
                OUT_ZIP.write_bytes(data)
                OUT_XML.write_bytes(_extract_xml_from_zip(data))
                print(f"Saved {OUT_ZIP}")
                print(f"Extracted {OUT_XML}")
            else:
                OUT_XML.write_bytes(data)
                print(f"Saved {OUT_XML}")
            return
        except Exception as exc:  # pragma: no cover - network variability
            last_error = exc
            continue

    raise RuntimeError(f"Failed to download MedlinePlus data. Last error: {last_error}")

if __name__ == "__main__":
    main()