#!/usr/bin/env python3
"""Ingest demo documents into retrieval-service."""
import json
import os
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DOCS_PATH = ROOT_DIR / "examples" / "clinical_docs_extended.json"
RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://localhost:8040")


def main() -> None:
    data = json.loads(DOCS_PATH.read_text(encoding="utf-8"))
    with httpx.Client(timeout=60.0) as client:
        r = client.post(f"{RETRIEVAL_URL.rstrip('/')}/v1/ingest", json=data)
        r.raise_for_status()
    resp = r.json()
    chunks = resp.get("chunks_inserted", 0)
    print(f"Inserted {chunks} chunks.")


if __name__ == "__main__":
    main()
