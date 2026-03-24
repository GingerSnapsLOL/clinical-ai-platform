#!/usr/bin/env python3
"""Call retrieval-service /v1/retrieve and print sources."""
import json
import os
import sys
from pathlib import Path

import httpx

RETRIEVAL_URL = os.getenv("RETRIEVAL_URL", "http://localhost:8040")
DEFAULT_QUERY = "What are the first-line treatments for hypertension?"


def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUERY
    url = f"{RETRIEVAL_URL.rstrip('/')}/v1/retrieve"

    payload = {
        "trace_id": "retrieve-demo-001",
        "query": query,
        "top_k": 5,
    }

    print(f"POST {url}")
    print(f"Query: {query}\n")

    try:
        r = httpx.post(url, json=payload, timeout=90.0)
        r.raise_for_status()
        data = r.json()
        passages = data.get("passages", [])

        print(f"Found {len(passages)} passages:\n")
        for i, p in enumerate(passages, 1):
            print(f"--- Source {i} (score: {p.get('score', 0):.3f}, id: {p.get('source_id', '')}) ---")
            print(p.get("text", ""))
            if p.get("metadata"):
                print(f"Metadata: {p['metadata']}")
            print()
        return 0
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}")
        return 1
    except httpx.RequestError as e:
        print(f"Request failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
