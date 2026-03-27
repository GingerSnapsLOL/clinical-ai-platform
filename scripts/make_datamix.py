#!/usr/bin/env python3
"""Build data/processed/datamix.jsonl from interim datasets."""


from pathlib import Path
import orjson

INPUTS = [
    Path("data/interim/medlineplus.jsonl"),
    Path("data/interim/dailymed.jsonl"),
    Path("data/interim/synthetic.jsonl"),
]

OUT_PATH = Path("data/processed/datamix.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def read_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)


def normalize_record(path: Path, rec: dict, row_index: int) -> dict:
    # Normalize synthetic examples to the same schema used by MedlinePlus/DailyMed.
    if path.name == "synthetic.jsonl":
        metadata = rec.get("metadata") or {}
        title = rec.get("title") or metadata.get("title") or rec.get("doc_id") or "Synthetic Doc"
        source = rec.get("source") or metadata.get("source") or "Synthetic"
        topic = rec.get("topic") or metadata.get("topic") or "synthetic"
        doc_type = rec.get("doc_type") or metadata.get("type") or "synthetic"
        text = rec.get("text") or ""
        return {
            "id": rec.get("id") or rec.get("doc_id") or f"synthetic_{row_index}",
            "title": title,
            "text": text,
            "source": source,
            "organization": rec.get("organization"),
            "url": rec.get("url"),
            "topic": topic,
            "doc_type": doc_type,
            "section": rec.get("section") or "full_text",
            "lang": rec.get("lang") or "en",
            "updated_at": rec.get("updated_at"),
            "license_note": rec.get("license_note") or "Synthetic clinical examples",
            "meta": {
                "raw_source": "synthetic_examples",
                "metadata": metadata,
            },
        }
    return rec

def main():
    seen = set()
    saved = 0

    with OUT_PATH.open("wb") as out:
        for path in INPUTS:
            if not path.exists():
                continue

            for row_index, rec in enumerate(read_jsonl(path)):
                rec = normalize_record(path, rec, row_index)
                key = (
                    rec.get("source"),
                    rec.get("title", "")[:150],
                    rec.get("text", "")[:600],
                )
                if key in seen:
                    continue
                seen.add(key)
                out.write(orjson.dumps(rec) + b"\n")
                saved += 1

    print(f"Saved {saved} records to {OUT_PATH}")

if __name__ == "__main__":
    main()