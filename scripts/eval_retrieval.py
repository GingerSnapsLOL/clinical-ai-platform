#!/usr/bin/env python3
"""
Lightweight retrieval evaluation script.

Sends a query to retrieval-service /v1/retrieve and prints the top passages
with scores and basic metadata (doc_id, title, source).
"""
import argparse
import json
import os
import sys
import uuid
from typing import Any, Dict

import httpx


def _post_retrieve(
    base_url: str,
    query: str,
    top_k: int = 10,
    top_n: int = 3,
    rerank: bool = True,
) -> Dict[str, Any]:
    trace_id = str(uuid.uuid4())
    payload = {
        "trace_id": trace_id,
        "query": query,
        "top_k": top_k,
        "top_n": top_n,
        "rerank": rerank,
    }
    url = f"{base_url.rstrip('/')}/v1/retrieve"
    with httpx.Client(timeout=30.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
    return resp.json()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate retrieval-service for a single query.")
    parser.add_argument(
        "--url",
        default=os.getenv("RETRIEVAL_URL", "http://localhost:8040"),
        help="Base URL for retrieval-service (default: %(default)s)",
    )
    parser.add_argument(
        "--query",
        required=False,
        help="Free-text query. If omitted, read from stdin.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of candidates to fetch from Qdrant (default: %(default)s)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of passages to keep after reranking (default: %(default)s)",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable cross-encoder reranking (use raw vector similarity).",
    )

    args = parser.parse_args(argv)

    query = args.query
    if not query:
        query = sys.stdin.read().strip()
        if not query:
            print("No query provided (via --query or stdin).", file=sys.stderr)
            return 1

    try:
        data = _post_retrieve(
            base_url=args.url,
            query=query,
            top_k=args.top_k,
            top_n=args.top_n,
            rerank=not args.no_rerank,
        )
    except Exception as exc:  # pragma: no cover - simple CLI
        print(f"Request to retrieval-service failed: {exc}", file=sys.stderr)
        return 1

    print(f"trace_id: {data.get('trace_id')}")
    passages = data.get("passages", [])
    print(f"passages returned: {len(passages)}\n")

    for idx, p in enumerate(passages, start=1):
        meta = p.get("metadata") or {}
        doc_id = meta.get("doc_id", p.get("source_id"))
        title = meta.get("title", "<no title>")
        source = meta.get("source", "<no source>")
        score = p.get("score", 0.0)
        text = (p.get("text") or "").strip()
        snippet = text[:300] + ("..." if len(text) > 300 else "")

        print(f"#{idx} score={score:.4f}")
        print(f"  doc_id : {doc_id}")
        print(f"  title  : {title}")
        print(f"  source : {source}")
        print(f"  text   : {snippet}\n")

    # Also print raw JSON if needed for debugging
    # print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

