#!/usr/bin/env python3
"""Download raw DailyMed source data into data/raw/dailymed."""


from pathlib import Path
import re
import requests

INDEX_URL = "https://dailymed-data.nlm.nih.gov/public-release-files/"
FALLBACK_URLS = [
    "https://dailymed-data.nlm.nih.gov/public-release-files/dm_spl_release_human_rx_part1.zip",
]

OUT_DIR = Path("data/raw/dailymed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def download(url: str, out_path: Path) -> None:
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

def discover_zip_urls() -> list[str]:
    html = requests.get(INDEX_URL, timeout=120).text
    links = []
    for href in re.findall(r'href="([^"]+)"', html):
        if "dm_spl_release_human_rx_part" in href and href.endswith(".zip"):
            if href.startswith("http"):
                links.append(href)
            else:
                links.append(f"{INDEX_URL}{href.lstrip('/')}")
    return links

def main():
    urls = discover_zip_urls() + FALLBACK_URLS
    out_path = OUT_DIR / "dailymed_part1.zip"
    last_error: Exception | None = None

    for url in urls:
        try:
            download(url, out_path)
            print(f"Saved {out_path} from {url}")
            return
        except Exception as exc:  # pragma: no cover - network variability
            last_error = exc
            continue

    raise RuntimeError(f"Failed to download DailyMed zip. Last error: {last_error}")

if __name__ == "__main__":
    main()