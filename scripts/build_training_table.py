#!/usr/bin/env python3
"""Build a tabular triage dataset from ``labeled_cases.jsonl``.

Writes ``data/processed/triage_train.parquet`` and ``data/processed/feature_spec.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

NOTE_RE = re.compile(
    r"^(\d+)-year-old\s+(.+?)\s+with\s+(.+?)(?:,\s*history of\s+(.+?))?\.\s*$",
    re.IGNORECASE | re.DOTALL,
)

LABEL_TO_INT = {"low": 0, "medium": 1, "high": 2}

# Canonical sex strings from generate_cases → integer codes (stable for training)
SEX_TO_CODE = {
    "male": 0,
    "female": 1,
    "non-binary patient": 2,
}

CHEST_PAIN_PAT = re.compile(
    r"\bchest\s+pain\b|\bangina\b",
    re.IGNORECASE,
)
DYSPNEA_PAT = re.compile(
    r"\b(shortness\s+of\s+breath|difficulty\s+breathing|dyspnea|sob)\b",
    re.IGNORECASE,
)
NEURO_PAT = re.compile(
    r"\b(numbs?ness|tingling|confusion|seizures?|weakness|stroke|"
    r"tia\b|transient\s+ischemic|altered\s+mental|slurred\s+speech|"
    r"facial\s+droop|hemiparesis|vision\s+loss)\b",
    re.IGNORECASE,
)
SMOKING_PAT = re.compile(
    r"\b(smoking|smokers?|cigarettes?|tobacco|second-?hand\s+smoke|vaping|e-?cigarette)\b",
    re.IGNORECASE,
)
HTN_PAT = re.compile(
    r"\b(hypertension|high\s+blood\s+pressure|elevated\s+blood\s+pressure)\b",
    re.IGNORECASE,
)
DIABETES_PAT = re.compile(
    r"\b(diabetes|diabetic|prediabetes|pre-?diabetes|insulin\s+resistance)\b",
    re.IGNORECASE,
)


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


def _split_natural_list(phrase: str) -> list[str]:
    s = phrase.strip().rstrip(".")
    if not s:
        return []
    if ", and " in s:
        left, right = s.rsplit(", and ", 1)
        head = [x.strip() for x in left.split(",")]
        return [x for x in head + [right.strip()] if x]
    if " and " in s:
        parts = [p.strip() for p in s.split(" and ", 1)]
        return [p for p in parts if p]
    return [s]


def _parse_note(note: str) -> tuple[list[str], list[str]]:
    m = NOTE_RE.match((note or "").strip())
    if not m:
        return [], []
    sym_phrase = m.group(3).strip()
    risk_phrase = (m.group(4) or "").strip()
    return _split_natural_list(sym_phrase), _split_natural_list(risk_phrase)


def _combined_text(case: dict[str, Any]) -> str:
    parts: list[str] = []
    nt = case.get("note_text")
    if isinstance(nt, str) and nt.strip():
        parts.append(nt)
    ent = case.get("entities")
    if isinstance(ent, list):
        parts.extend(str(x) for x in ent if x is not None)
    return " \n ".join(parts).lower()


def _encode_sex(raw: Any) -> tuple[int, str]:
    if raw is None:
        return -1, "unknown"
    s = str(raw).strip().lower()
    if s in SEX_TO_CODE:
        return SEX_TO_CODE[s], s
    return -1, s or "unknown"


def _row_from_case(case: dict[str, Any], *, include_extras: bool) -> dict[str, Any] | None:
    label = case.get("label")
    if label is None:
        return None
    lk = str(label).strip().lower()
    if lk not in LABEL_TO_INT:
        return None

    sf = case.get("structured_features")
    if not isinstance(sf, dict):
        sf = {}

    age_raw = sf.get("age")
    try:
        age = int(age_raw) if age_raw is not None else -1
    except (TypeError, ValueError):
        age = -1

    sex_enc = _encode_sex(sf.get("sex"))[0]

    note = case.get("note_text") if isinstance(case.get("note_text"), str) else ""
    syms, risks = _parse_note(note)
    num_symptoms = len(syms)
    num_risk_factors = len(risks)
    blob = _combined_text(case)

    row: dict[str, Any] = {
        "age": age,
        "sex_enc": sex_enc,
        "num_symptoms": num_symptoms,
        "num_risk_factors": num_risk_factors,
        "has_chest_pain": int(bool(CHEST_PAIN_PAT.search(blob))),
        "has_dyspnea": int(bool(DYSPNEA_PAT.search(blob))),
        "has_neuro_deficit": int(bool(NEURO_PAT.search(blob))),
        "smoking": int(bool(SMOKING_PAT.search(blob))),
        "hypertension": int(bool(HTN_PAT.search(blob))),
        "diabetes": int(bool(DIABETES_PAT.search(blob))),
        "label_int": LABEL_TO_INT[lk],
    }
    if include_extras:
        row["text_length"] = len(note) if note else 0
        ent = case.get("entities")
        row["entity_count"] = len(ent) if isinstance(ent, list) else 0
    return row


def _build_feature_spec(*, include_extras: bool) -> dict[str, Any]:
    feature_cols = [
        "age",
        "sex_enc",
        "num_symptoms",
        "num_risk_factors",
        "has_chest_pain",
        "has_dyspnea",
        "has_neuro_deficit",
        "smoking",
        "hypertension",
        "diabetes",
    ]
    if include_extras:
        feature_cols.extend(["text_length", "entity_count"])

    all_cols = feature_cols + ["label_int"]
    return {
        "version": 1,
        "target_column": "label_int",
        "target_class_map": dict(LABEL_TO_INT),
        "inverse_label_map": {str(v): k for k, v in LABEL_TO_INT.items()},
        "feature_columns": feature_cols,
        "parquet_columns": all_cols,
        "sex_encoding": dict(SEX_TO_CODE),
        "sex_unknown_code": -1,
        "age_missing_code": -1,
        "binary_flag_columns": [
            "has_chest_pain",
            "has_dyspnea",
            "has_neuro_deficit",
            "smoking",
            "hypertension",
            "diabetes",
        ],
        "optional_columns_included": bool(include_extras),
        "notes": (
            "num_symptoms / num_risk_factors are parsed from synthetic note_text "
            "('... with X, history of Y.') when possible; flags use regex over "
            "note_text plus entities."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/processed/labeled_cases.jsonl"),
        help="Input JSONL (default: data/processed/labeled_cases.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/triage_train.parquet"),
        help="Output Parquet (default: data/processed/triage_train.parquet)",
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("data/processed/feature_spec.json"),
        help="Feature spec JSON (default: data/processed/feature_spec.json)",
    )
    parser.add_argument(
        "--no-extras",
        action="store_true",
        help="Omit text_length and entity_count columns",
    )
    args = parser.parse_args()
    in_path: Path = args.input
    out_path: Path = args.output
    spec_path: Path = args.spec
    include_extras = not args.no_extras

    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    cases = _read_jsonl(in_path)
    rows: list[dict[str, Any]] = []
    skipped = 0
    for c in cases:
        r = _row_from_case(c, include_extras=include_extras)
        if r is None:
            skipped += 1
            continue
        rows.append(r)

    if not rows:
        raise SystemExit("No valid labeled rows (need label in low|medium|high).")

    df = pd.DataFrame(rows)
    dtypes = {
        "age": "int32",
        "sex_enc": "int8",
        "num_symptoms": "int8",
        "num_risk_factors": "int8",
        "has_chest_pain": "int8",
        "has_dyspnea": "int8",
        "has_neuro_deficit": "int8",
        "smoking": "int8",
        "hypertension": "int8",
        "diabetes": "int8",
        "label_int": "int8",
    }
    if include_extras:
        dtypes["text_length"] = "int32"
        dtypes["entity_count"] = "int16"
    for col, dt in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    spec = _build_feature_spec(include_extras=include_extras)
    spec["row_count"] = len(df)
    spec["skipped_input_rows"] = skipped
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    print(
        f"Wrote {len(df)} rows to {out_path} and spec to {spec_path} "
        f"(skipped {skipped} rows)",
        flush=True,
    )


if __name__ == "__main__":
    main()
