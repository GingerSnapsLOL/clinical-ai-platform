#!/usr/bin/env python3
"""
Retrieval quality evaluation against a golden dataset.

For each query, calls retrieval-service POST /v1/retrieve, takes the first ``k``
passages (document id per passage from ``metadata.doc_id`` or ``source_id``),
and computes:

- **Hit@k**: fraction of queries where at least one gold ``doc_id`` appears in
  those top-``k`` passages.
- **Recall@k**: mean over queries of |gold ∩ retrieved_doc_ids@k| / |gold|.

Golden file formats (text field for retrieval can be ``query`` or ``question``):

- **Canonical**: ``eval/golden_set.json`` — ``[{"question": "...", "expected_doc_ids": ["doc1"]}, ...]``
- **JSONL**: ``{"query": "...", "relevant_doc_ids": ["a"]}`` per line
- **JSON** array: objects with any of the id keys below
- **JSON** object: ``{"queries": [ ... ]}``

Expected document id keys (first match wins): ``expected_doc_ids``, ``relevant_doc_ids``,
``gold_doc_ids``, ``doc_ids``.

By default ``rerank`` is off so ``top_n=k`` returns ``k`` passages. With
``--rerank``, the service returns at most **3** passages; use ``--k`` ≤ 3.

Example::

    uv run python scripts/eval_retrieval.py \\
        --dataset eval/golden_set.json \\
        --url http://localhost:8040 \\
        --k 10 \\
        --log-failures retrieval_eval_failures.jsonl \\
        --log-misses retrieval_eval_misses.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import httpx


def load_golden_dataset(path: Path) -> list[dict[str, Any]]:
    """Load queries with ``relevant_doc_ids`` (non-empty list of strings)."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return []

    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            rows.append(obj)
        return rows

    data = json.loads(raw)
    if isinstance(data, list):
        return list(data)
    if isinstance(data, dict) and "queries" in data:
        q = data["queries"]
        if not isinstance(q, list):
            raise ValueError("JSON object with 'queries' must be a list")
        return list(q)
    raise ValueError("JSON file must be a list or {\"queries\": [...]}")


def _normalize_gold_ids(item: dict[str, Any]) -> list[str]:
    keys = ("expected_doc_ids", "relevant_doc_ids", "gold_doc_ids", "doc_ids")
    for key in keys:
        if key in item:
            ids = item[key]
            break
    else:
        raise KeyError(
            f"golden row missing doc-id list (one of {keys}); keys={list(item.keys())!r}",
        )
    if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
        raise TypeError("expected_doc_ids / relevant_doc_ids must be a list of strings")
    out = [x.strip() for x in ids if x.strip()]
    if not out:
        raise ValueError("expected_doc_ids list must be non-empty after trimming")
    return out


def _post_retrieve(
    client: httpx.Client,
    base_url: str,
    query: str,
    *,
    top_k: int,
    top_n: int,
    rerank: bool,
) -> dict[str, Any]:
    trace_id = str(uuid.uuid4())
    payload = {
        "trace_id": trace_id,
        "query": query,
        "top_k": top_k,
        "top_n": top_n,
        "rerank": rerank,
    }
    url = f"{base_url.rstrip('/')}/v1/retrieve"
    resp = client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


def doc_ids_from_top_passages(passages: Sequence[dict[str, Any]], k: int) -> list[str]:
    """First ``k`` passages → stable doc id string per passage (chunk-level top-k)."""
    out: list[str] = []
    for p in passages[:k]:
        meta = p.get("metadata") or {}
        doc = meta.get("doc_id") or meta.get("id") or p.get("source_id") or ""
        out.append(str(doc))
    return out


def evaluate_row(
    passages: list[dict[str, Any]],
    gold_ids: Sequence[str],
    k: int,
) -> tuple[int, float, list[str]]:
    """Returns (hit 0/1, recall in [0,1], doc_id list for first k passages)."""
    gold = frozenset(gold_ids)
    retrieved = doc_ids_from_top_passages(passages, k)
    rset = set(retrieved)
    hit = 1 if (gold & rset) else 0
    recall = len(gold & rset) / len(gold) if gold else 0.0
    return hit, recall, retrieved


def print_summary_table(
    *,
    k: int,
    n_total: int,
    n_queries: int,
    mean_hit: float,
    mean_recall: float,
    n_failed: int,
) -> None:
    w = 22
    line = "+" + "-" * (w + 2) + "+" + "-" * 12 + "+"
    print(line)
    print(f"| {'metric':<{w}} | {'value':>10} |")
    print(line)
    print(f"| {'queries in dataset':<{w}} | {n_total:>10} |")
    print(f"| {'queries scored':<{w}} | {n_queries:>10} |")
    print(f"| {'failed (HTTP/parse)':<{w}} | {n_failed:>10} |")
    print(f"| {f'mean hit@{k}':<{w}} | {mean_hit:>10.4f} |")
    print(f"| {f'mean recall@{k}':<{w}} | {mean_recall:>10.4f} |")
    print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval-service with a golden JSON/JSONL dataset.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to golden JSONL or JSON (see module docstring).",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:8040",
        help="retrieval-service base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-k passages for metrics (set top_n to this when rerank is off).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Qdrant candidate limit (default: max(50, k*5)).",
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="Use service reranking (truncates to top 3 passages; k should be <= 3).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="HTTP timeout seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--log-failures",
        type=Path,
        default=None,
        help="Append HTTP/parse failures (JSONL: query, gold, error).",
    )
    parser.add_argument(
        "--log-misses",
        type=Path,
        default=None,
        help="Append queries with hit@k==0 (JSONL: query, gold, retrieved_doc_ids).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query misses (hit=0) to stderr.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate golden file only; do not call the service.",
    )
    args = parser.parse_args(argv)

    k = max(1, args.k)
    top_k = args.top_k if args.top_k is not None else max(50, k * 5)
    # Without rerank, top_n caps passages returned. With rerank, service truncates to 3 regardless.
    top_n = k if not args.rerank else 8
    if args.rerank and k > 3:
        print(
            "Warning: rerank=True returns at most 3 passages; use --k <= 3 or --no-rerank.",
            file=sys.stderr,
        )

    rows_in = load_golden_dataset(args.dataset)
    prepared: list[tuple[str, list[str]]] = []
    for i, row in enumerate(rows_in):
        if "query" not in row and "question" not in row:
            print(f"Row {i}: missing 'query' or 'question'", file=sys.stderr)
            return 1
        q = str(row.get("query") or row.get("question") or "").strip()
        if not q:
            print(f"Row {i}: empty query", file=sys.stderr)
            return 1
        try:
            gold = _normalize_gold_ids(row)
        except (KeyError, TypeError, ValueError) as e:
            print(f"Row {i}: {e}", file=sys.stderr)
            return 1
        prepared.append((q, gold))

    if args.dry_run:
        print(f"OK: {len(prepared)} queries loaded from {args.dataset}")
        return 0

    hits: list[int] = []
    recalls: list[float] = []
    n_failed = 0
    failure_log: list[dict[str, Any]] = []
    miss_log: list[dict[str, Any]] = []

    with httpx.Client(timeout=args.timeout) as client:
        for query, gold in prepared:
            try:
                data = _post_retrieve(
                    client,
                    args.url,
                    query,
                    top_k=top_k,
                    top_n=top_n,
                    rerank=args.rerank,
                )
            except Exception as exc:
                n_failed += 1
                rec = {
                    "query": query,
                    "gold": list(gold),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failure_log.append(rec)
                print(f"[FAIL] {query[:80]!r} -> {rec['error']}", file=sys.stderr)
                continue

            passages = data.get("passages") or []
            if not isinstance(passages, list):
                n_failed += 1
                failure_log.append(
                    {
                        "query": query,
                        "gold": list(gold),
                        "error": "invalid response: passages is not a list",
                    },
                )
                continue

            hit, recall, retrieved = evaluate_row(passages, gold, k)
            hits.append(hit)
            recalls.append(recall)
            if hit == 0:
                miss_log.append(
                    {
                        "query": query,
                        "gold": list(gold),
                        "retrieved_doc_ids": retrieved[:k],
                    },
                )
                if args.verbose:
                    print(
                        f"[MISS] recall={recall:.3f} q={query[:100]!r} "
                        f"gold={sorted(gold)[:5]} retrieved={retrieved[:k]}",
                        file=sys.stderr,
                    )

    if args.log_failures and failure_log:
        with args.log_failures.open("a", encoding="utf-8") as fh:
            for rec in failure_log:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.log_misses and miss_log:
        with args.log_misses.open("a", encoding="utf-8") as fh:
            for rec in miss_log:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_ok = len(hits)
    mean_hit = sum(hits) / n_ok if n_ok else 0.0
    mean_recall = sum(recalls) / n_ok if n_ok else 0.0

    print_summary_table(
        k=k,
        n_total=len(prepared),
        n_queries=n_ok,
        mean_hit=mean_hit,
        mean_recall=mean_recall,
        n_failed=n_failed,
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
