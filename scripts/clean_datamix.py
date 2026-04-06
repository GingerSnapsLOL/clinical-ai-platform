#!/usr/bin/env python3
"""Clean raw datamix.json / .jsonl into a normalized English corpus for downstream ML.

Reads a JSON array file, a single JSON object, or JSONL (one JSON object per line).
Writes one normalized object per line: id, title, text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

ALLOWED_DOC_TYPES = frozenset(
    {"patient_education", "reference", "disease_overview"}
)
MIN_TEXT_LEN = 200


def _iter_raw_objects(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    """Yield (source_line_hint, obj) for each dict in the file. line hint is 1-based for JSONL rows."""
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.lstrip("\ufeff \t\n\r")
    if not stripped:
        return

    if stripped[0] == "[":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise SystemExit(f"Invalid JSON array in {path}: {e}") from e
        if not isinstance(data, list):
            raise SystemExit(f"Expected JSON array at root of {path}, got {type(data).__name__}")
        for i, item in enumerate(data):
            if isinstance(item, dict):
                yield (i + 1, item)
            elif item is not None:
                print(f"skip: array element {i} is not an object", file=sys.stderr)
        return

    if stripped[0] == "{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            pass
        else:
            if isinstance(data, dict):
                yield (1, data)
                return
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict):
                        yield (i + 1, item)
                return

    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"skip: line {line_no}: invalid JSON ({e})", file=sys.stderr)
            continue
        if isinstance(obj, dict):
            yield (line_no, obj)
        else:
            print(f"skip: line {line_no}: not a JSON object", file=sys.stderr)


def _stable_id(rec: dict[str, Any], fallback_key: str) -> str:
    for key in ("id", "doc_id"):
        raw = rec.get(key)
        if raw is None:
            continue
        s = str(raw).strip()
        if s:
            return s
    title = (rec.get("title") or "").strip()
    body = (rec.get("text") or "").strip()
    h = hashlib.sha256(f"{title}\0{body[:4000]}".encode("utf-8")).hexdigest()[:16]
    return f"gen_{fallback_key}_{h}"


def _normalize_out(rec: dict[str, Any], fallback_key: str) -> dict[str, str]:
    return {
        "id": _stable_id(rec, fallback_key),
        "title": (rec.get("title") or "").strip(),
        "text": (rec.get("text") or "").strip(),
    }


def _passes_filters(rec: dict[str, Any]) -> tuple[bool, str]:
    lang = rec.get("lang")
    if lang is None:
        return False, "no_lang"
    if str(lang).strip().lower() != "en":
        return False, "not_en"

    if "doc_type" in rec and rec["doc_type"] is not None:
        dt = str(rec["doc_type"]).strip()
        if dt and dt not in ALLOWED_DOC_TYPES:
            return False, "doc_type"

    text = (rec.get("text") or "").strip()
    if not text:
        return False, "empty_text"
    if len(text) < MIN_TEXT_LEN:
        return False, "short_text"

    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/processed/datamix.jsonl"),
        help="Raw datamix.json or .jsonl (default: data/processed/datamix.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/clean_docs.jsonl"),
        help="Output JSONL path (default: data/processed/clean_docs.jsonl)",
    )
    args = parser.parse_args()
    in_path: Path = args.input
    out_path: Path = args.output

    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    stats: Counter[str] = Counter()
    written = 0

    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for hint, rec in _iter_raw_objects(in_path):
            stats["seen"] += 1
            ok, reason = _passes_filters(rec)
            if not ok:
                stats[f"drop_{reason}"] += 1
                continue
            norm = _normalize_out(rec, str(hint))
            out.write(json.dumps(norm, ensure_ascii=False) + "\n")
            written += 1

    print(f"Wrote {written} records to {out_path}")
    drops = {k: v for k, v in stats.items() if k.startswith("drop_")}
    print(f"Counts: seen={stats.get('seen', 0)}, kept={written}, drops={drops}")


if __name__ == "__main__":
    main()
