#!/usr/bin/env python3
"""Merge chunked JSONL rows (e.g. medlineplus_8_0, medlineplus_8_1) into full documents.

Groups rows by:
  - shared id base: strip a trailing ``_<digits>`` suffix (prefix before last underscore + index), or
  - same non-empty ``url`` (unions groups that share a URL).

Chunks are ordered by numeric index (from id suffix, else ``meta.chunk_index``, else stable fallbacks).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace", newline="") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"skip line {line_no}: {e}", file=sys.stderr)
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                print(f"skip line {line_no}: not a JSON object", file=sys.stderr)
    return rows


def _row_id(rec: dict[str, Any]) -> str:
    rid = rec.get("id")
    if rid is not None and str(rid).strip():
        return str(rid).strip()
    did = rec.get("doc_id")
    if did is not None and str(did).strip():
        return str(did).strip()
    return ""


def _parse_id_chunk(rid: str) -> tuple[str | None, int | None]:
    """If ``rid`` ends with ``_<non-negative int>``, return (base, index); else (None, None)."""
    if not rid:
        return None, None
    base, sep, tail = rid.rpartition("_")
    if not sep or not tail.isdigit():
        return None, None
    return base, int(tail)


def _sort_key(rec: dict[str, Any], list_index: int) -> tuple[int, int, int, str]:
    rid = _row_id(rec)
    _, idx = _parse_id_chunk(rid)
    if idx is not None:
        return (0, idx, 0, rid)
    meta = rec.get("meta")
    if isinstance(meta, dict) and meta.get("chunk_index") is not None:
        try:
            ci = int(meta["chunk_index"])
            return (1, ci, 0, rid)
        except (TypeError, ValueError):
            pass
    ci2 = rec.get("chunk_index")
    if ci2 is not None:
        try:
            return (2, int(ci2), 0, rid)
        except (TypeError, ValueError):
            pass
    return (3, list_index, 0, rid)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._p = list(range(n))

    def find(self, i: int) -> int:
        p = self._p
        while p[i] != i:
            p[i] = p[p[i]]
            i = p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self._p[rj] = ri


def _first_nonempty_str(recs: list[dict[str, Any]], key: str) -> str:
    for r in recs:
        v = r.get(key)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _merge_group(indices: list[int], records: list[dict[str, Any]]) -> dict[str, Any]:
    members = [records[i] for i in indices]
    # Stable: sort by chunk index, then by original list index
    decorated = [(records[i], i) for i in indices]
    decorated.sort(key=lambda t: (_sort_key(t[0], t[1]), t[1]))
    ordered = [t[0] for t in decorated]

    ids_in_order = [_row_id(r) for r in ordered]
    bases = []
    for rid in ids_in_order:
        b, _ = _parse_id_chunk(rid)
        if b is not None:
            bases.append(b)

    if bases and len(set(bases)) == 1:
        doc_id = bases[0]
    else:
        doc_id = ids_in_order[0] if ids_in_order else "unknown"

    texts = [(r.get("text") or "").strip() for r in ordered]
    text = "\n".join(t for t in texts if t)

    title = _first_nonempty_str(ordered, "title")
    url = _first_nonempty_str(ordered, "url")
    topic = _first_nonempty_str(ordered, "topic")
    source = _first_nonempty_str(ordered, "source")

    return {
        "doc_id": doc_id,
        "title": title,
        "text": text,
        "chunk_count": len(ordered),
        "url": url,
        "topic": topic,
        "source": source,
        "original_ids": ids_in_order,
    }


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dedupe_merged(merged: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep first document in stable order for each non-empty URL or identical text hash."""
    merged_sorted = sorted(merged, key=lambda m: m["doc_id"])
    seen_url: set[str] = set()
    seen_hash: set[str] = set()
    out: list[dict[str, Any]] = []
    for m in merged_sorted:
        u = (m.get("url") or "").strip()
        h = _text_hash(m["text"])
        if u and u in seen_url:
            continue
        if h in seen_hash:
            continue
        if u:
            seen_url.add(u)
        seen_hash.add(h)
        out.append(m)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/processed/clean_docs.jsonl"),
        help="Input JSONL (default: data/processed/clean_docs.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/merged_docs.jsonl"),
        help="Output JSONL (default: data/processed/merged_docs.jsonl)",
    )
    args = parser.parse_args()
    in_path: Path = args.input
    out_path: Path = args.output

    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    records = _read_jsonl(in_path)
    if not records:
        raise SystemExit(f"No records read from {in_path}")

    n = len(records)
    uf = _UnionFind(n)

    by_base: dict[str, list[int]] = defaultdict(list)
    by_url: dict[str, list[int]] = defaultdict(list)

    for i, rec in enumerate(records):
        rid = _row_id(rec)
        base, _ = _parse_id_chunk(rid)
        if base is not None:
            by_base[base].append(i)
        url = (rec.get("url") or "").strip()
        if url:
            by_url[url].append(i)

    def _union_bucket(idxs: list[int]) -> None:
        if len(idxs) < 2:
            return
        a = idxs[0]
        for b in idxs[1:]:
            uf.union(a, b)

    for idxs in by_base.values():
        _union_bucket(idxs)
    for idxs in by_url.values():
        _union_bucket(idxs)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)

    merged_raw = [_merge_group(idxs, records) for idxs in groups.values()]
    merged = _dedupe_merged(merged_raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for m in sorted(merged, key=lambda x: x["doc_id"]):
            row = {
                "doc_id": m["doc_id"],
                "title": m["title"],
                "text": m["text"],
                "chunk_count": m["chunk_count"],
                "url": m["url"],
                "topic": m["topic"],
                "source": m["source"],
                "original_ids": m["original_ids"],
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(merged)} merged docs to {out_path} (from {n} rows, {len(groups)} groups pre-dedupe)")


if __name__ == "__main__":
    main()
