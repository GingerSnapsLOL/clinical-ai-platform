#!/usr/bin/env python3
"""Parse MedlinePlus raw files into data/interim/medlineplus.jsonl."""

from pathlib import Path
from lxml import etree
import orjson

from _text_utils import clean_text, chunk_text

IN_PATH = Path("data/raw/medlineplus/medlineplus.xml")
OUT_PATH = Path("data/interim/medlineplus.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Missing input file: {IN_PATH}. Run scripts/download_medlineplus.py first."
        )

    tree = etree.parse(str(IN_PATH))
    root = tree.getroot()

    count = 0
    chunks_written = 0
    with OUT_PATH.open("wb") as out:
        # Be resilient to tag quirks by selecting by local-name.
        topics = root.findall("health-topic")
        if not topics:
            topics = root.xpath("//*[local-name()='health-topic']")

        for topic in topics:
            title = clean_text(topic.get("title", "") or topic.xpath("string(./title)"))
            summary = clean_text(topic.xpath("string(./full-summary)"))
            if not summary:
                summary = clean_text(topic.get("meta-desc", ""))

            url = topic.get("url")
            if not url:
                url = clean_text(topic.xpath("string(./url)"))

            if not title or not summary:
                continue

            text = f"{title}\n\n{summary}"
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                rec = {
                    "id": f"medlineplus_{count}_{i}",
                    "title": title,
                    "text": chunk,
                    "source": "MedlinePlus",
                    "organization": "NLM",
                    "url": url,
                    "topic": title.lower(),
                    "doc_type": "patient_education",
                    "section": "summary",
                    "lang": "en",
                    "updated_at": None,
                    "license_note": "Official public medical content",
                    "meta": {
                        "raw_source": "medlineplus_xml",
                        "chunk_index": i,
                    },
                }
                out.write(orjson.dumps(rec) + b"\n")
                chunks_written += 1

            count += 1

    print(f"Saved {count} MedlinePlus topics / {chunks_written} chunks to {OUT_PATH}")

if __name__ == "__main__":
    main()