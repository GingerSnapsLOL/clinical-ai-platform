#!/usr/bin/env python3
"""Parse DailyMed raw files into data/interim/dailymed.jsonl."""


from pathlib import Path
from zipfile import ZipFile
import io
import gzip
from lxml import etree
import orjson

from _text_utils import clean_text, chunk_text

IN_DIR = Path("data/raw/dailymed")
OUT_PATH = Path("data/interim/dailymed.jsonl")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def extract_title(root) -> str:
    texts = [t.strip() for t in root.xpath("//text()") if t and t.strip()]
    if not texts:
        return "Untitled drug label"
    return clean_text(texts[0])[:180]

def main():
    parser = etree.XMLParser(huge_tree=True, recover=True)
    files = sorted(IN_DIR.glob("*.zip"))
    total_docs = 0
    total_chunks = 0
    scanned_xml_files = 0

    with OUT_PATH.open("wb") as out:
        for zip_path in files:
            with ZipFile(zip_path, "r") as zf:
                for name in zf.namelist():
                    lower = name.lower()
                    try:
                        # DailyMed part zips often contain nested zip archives.
                        if lower.endswith(".zip"):
                            with zf.open(name) as nested_file:
                                nested_bytes = nested_file.read()
                            with ZipFile(io.BytesIO(nested_bytes), "r") as nested_zip:
                                for nested_name in nested_zip.namelist():
                                    nested_lower = nested_name.lower()
                                    if not (
                                        nested_lower.endswith(".xml")
                                        or nested_lower.endswith(".xml.gz")
                                    ):
                                        continue
                                    scanned_xml_files += 1
                                    with nested_zip.open(nested_name) as f:
                                        if nested_lower.endswith(".xml.gz"):
                                            content = gzip.decompress(f.read())
                                            root = etree.fromstring(content, parser=parser)
                                        else:
                                            root = etree.parse(f, parser).getroot()

                                    full_text = clean_text(" ".join(root.xpath("//text()")))
                                    if len(full_text) < 1000:
                                        continue

                                    title = extract_title(root)
                                    chunks = chunk_text(
                                        full_text, chunk_size=2600, overlap=300
                                    )

                                    for i, chunk in enumerate(chunks):
                                        rec = {
                                            "id": f"dailymed_{total_docs}_{i}",
                                            "title": title,
                                            "text": chunk,
                                            "source": "DailyMed",
                                            "organization": "NLM/FDA",
                                            "url": None,
                                            "topic": "drug_label",
                                            "doc_type": "drug_label",
                                            "section": "full_label",
                                            "lang": "en",
                                            "updated_at": None,
                                            "license_note": "Official public drug label content",
                                            "meta": {
                                                "raw_source": "dailymed_spl_xml",
                                                "raw_file": f"{name}::{nested_name}",
                                                "chunk_index": i,
                                            },
                                        }
                                        out.write(orjson.dumps(rec) + b"\n")
                                        total_chunks += 1

                                    total_docs += 1
                            continue

                        if not (lower.endswith(".xml") or lower.endswith(".xml.gz")):
                            continue

                        scanned_xml_files += 1
                        with zf.open(name) as f:
                            if lower.endswith(".xml.gz"):
                                content = gzip.decompress(f.read())
                                root = etree.fromstring(content, parser=parser)
                            else:
                                root = etree.parse(f, parser).getroot()

                        full_text = clean_text(" ".join(root.xpath("//text()")))
                        if len(full_text) < 1000:
                            continue

                        title = extract_title(root)
                        chunks = chunk_text(full_text, chunk_size=2600, overlap=300)

                        for i, chunk in enumerate(chunks):
                            rec = {
                                "id": f"dailymed_{total_docs}_{i}",
                                "title": title,
                                "text": chunk,
                                "source": "DailyMed",
                                "organization": "NLM/FDA",
                                "url": None,
                                "topic": "drug_label",
                                "doc_type": "drug_label",
                                "section": "full_label",
                                "lang": "en",
                                "updated_at": None,
                                "license_note": "Official public drug label content",
                                "meta": {
                                    "raw_source": "dailymed_spl_xml",
                                    "raw_file": name,
                                    "chunk_index": i,
                                },
                            }
                            out.write(orjson.dumps(rec) + b"\n")
                            total_chunks += 1

                        total_docs += 1
                    except Exception as e:
                        print(f"Skip {name}: {e}")

    print(
        f"Scanned {scanned_xml_files} XML files. "
        f"Saved {total_docs} DailyMed docs / {total_chunks} chunks to {OUT_PATH}"
    )

if __name__ == "__main__":
    main()