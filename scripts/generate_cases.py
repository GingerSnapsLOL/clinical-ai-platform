#!/usr/bin/env python3
"""Generate synthetic patient-like cases from ``signals_docs.jsonl`` using templates.

Reads ``data/processed/signals_docs.jsonl`` and writes ``data/processed/synthetic_cases.jsonl``.
Each source document yields 1–5 cases with mild / medium / severe variety.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable

SEVERITIES: tuple[str, ...] = ("mild", "medium", "severe")

HYPERTENSION_MARKERS: frozenset[str] = frozenset(
    {
        "hypertension",
        "high blood pressure",
        "elevated blood pressure",
    }
)

FALLBACK_SYMPTOMS: tuple[str, ...] = (
    "nonspecific fatigue",
    "general malaise",
    "routine health concerns",
)


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
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
                yield obj
            else:
                print(f"skip line {line_no}: not a JSON object", file=sys.stderr)


def _signals(rec: dict[str, Any]) -> dict[str, list[str]]:
    raw = rec.get("signals")
    if not isinstance(raw, dict):
        return {
            "symptoms": [],
            "diseases": [],
            "risk_factors": [],
            "emergency_flags": [],
        }
    out: dict[str, list[str]] = {
        "symptoms": [],
        "diseases": [],
        "risk_factors": [],
        "emergency_flags": [],
    }
    for k in out:
        v = raw.get(k)
        if isinstance(v, list):
            out[k] = [str(x) for x in v if x is not None and str(x).strip()]
    return out


def _symptom_pool(sig: dict[str, list[str]], rng: random.Random) -> list[str]:
    s = list(sig["symptoms"])
    if s:
        return s
    diseases = sig["diseases"]
    if diseases:
        return [f"concerns related to {d}" for d in diseases[:12]]
    return list(FALLBACK_SYMPTOMS)


def _risk_pool(sig: dict[str, list[str]]) -> list[str]:
    return list(sig["risk_factors"])


def _prettify_risk(r: str) -> str:
    if r == "age_mention":
        return "advancing age"
    return r.replace("_", " ")


def _natural_join(items: list[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _severity_counts(severity: str, rng: random.Random) -> tuple[int, int]:
    """Return (n_symptoms, n_risks) before clamping to pool sizes."""
    if severity == "mild":
        return 1, rng.randint(0, 1)
    if severity == "medium":
        return rng.randint(1, 2), rng.randint(0, 2)
    return rng.randint(2, 3), rng.randint(1, 3)


def _sample_age(severity: str, rng: random.Random) -> int:
    if severity == "mild":
        return rng.randint(20, 58)
    if severity == "medium":
        return rng.randint(32, 82)
    return rng.randint(48, 90)


def _sample_sex(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.02:
        return "non-binary patient"
    if r < 0.51:
        return "male"
    return "female"


def _apply_severity_to_symptoms(
    symptoms: list[str], severity: str, rng: random.Random
) -> list[str]:
    if not symptoms:
        return symptoms
    out = list(symptoms)
    if severity == "mild" and rng.random() < 0.55:
        out[0] = f"mild {out[0]}"
    elif severity == "severe" and rng.random() < 0.65:
        out[0] = f"severe {out[0]}"
    elif severity == "medium" and rng.random() < 0.35:
        out[0] = f"progressive {out[0]}"
    return out


def _maybe_blood_pressure(
    severity: str,
    selected_risks: list[str],
    rng: random.Random,
) -> tuple[int | None, int | None]:
    has_htn = any(
        any(m in r.lower() for m in HYPERTENSION_MARKERS) for r in selected_risks
    )
    if severity == "mild":
        if has_htn and rng.random() < 0.35:
            return rng.randint(118, 132), rng.randint(74, 84)
        return None, None
    if severity == "medium":
        if has_htn or rng.random() < 0.4:
            return rng.randint(128, 145), rng.randint(82, 92)
        return None, None
    if has_htn or rng.random() < 0.7:
        return rng.randint(142, 178), rng.randint(88, 108)
    return rng.randint(118, 135), rng.randint(76, 88)


def _build_note(
    age: int,
    sex: str,
    symptoms: list[str],
    risks_display: list[str],
) -> str:
    sym_phrase = _natural_join(symptoms)
    if not risks_display:
        return f"{age}-year-old {sex} with {sym_phrase}."
    risk_phrase = _natural_join(risks_display)
    return f"{age}-year-old {sex} with {sym_phrase}, history of {risk_phrase}."


def _entities(
    symptoms: list[str],
    risks_raw: list[str],
    diseases: list[str],
    rng: random.Random,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for x in symptoms + risks_raw:
        k = x.strip()
        if k and k not in seen:
            seen.add(k)
            ordered.append(k)
    extra_n = rng.randint(0, min(2, len(diseases)))
    if extra_n and diseases:
        for d in rng.sample(diseases, k=extra_n):
            if d not in seen:
                seen.add(d)
                ordered.append(d)
    return ordered


def generate_case_for_doc(
    source_doc_id: str,
    sig: dict[str, list[str]],
    rng: random.Random,
    *,
    severity: str,
) -> dict[str, Any]:
    want_sym, want_risk = _severity_counts(severity, rng)

    sym_pool = _symptom_pool(sig, rng)
    risk_pool = _risk_pool(sig)

    n_sym = min(want_sym, len(sym_pool))
    if n_sym < 1:
        n_sym = 1
    chosen_sym = rng.sample(sym_pool, k=n_sym)
    chosen_sym = _apply_severity_to_symptoms(chosen_sym, severity, rng)

    n_risk = min(want_risk, len(risk_pool))
    chosen_risk = rng.sample(risk_pool, k=n_risk) if n_risk else []

    age = _sample_age(severity, rng)
    sex = _sample_sex(rng)
    sys_bp, dia_bp = _maybe_blood_pressure(severity, chosen_risk, rng)

    risks_display = [_prettify_risk(r) for r in chosen_risk]
    note = _build_note(age, sex, chosen_sym, risks_display)

    feats: dict[str, Any] = {"age": age, "sex": sex}
    if sys_bp is not None and dia_bp is not None:
        feats["systolic_bp"] = sys_bp
        feats["diastolic_bp"] = dia_bp

    return {
        "case_id": str(uuid.uuid4()),
        "note_text": note,
        "structured_features": feats,
        "entities": _entities(chosen_sym, chosen_risk, sig["diseases"], rng),
        "source_doc_id": source_doc_id,
        "generation_type": "template",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/processed/signals_docs.jsonl"),
        help="Input JSONL (default: data/processed/signals_docs.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/synthetic_cases.jsonl"),
        help="Output JSONL (default: data/processed/synthetic_cases.jsonl)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()
    in_path: Path = args.input
    out_path: Path = args.output

    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    rng = random.Random(args.seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_docs = 0
    n_cases = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for rec in _read_jsonl(in_path):
            doc_id = str(rec.get("doc_id") or rec.get("id") or "").strip()
            if not doc_id:
                doc_id = f"unknown_doc_{n_docs}"
            sig = _signals(rec)
            k = rng.randint(1, 5)
            phase = rng.randint(0, len(SEVERITIES) - 1)
            for i in range(k):
                severity = SEVERITIES[(phase + i) % len(SEVERITIES)]
                case = generate_case_for_doc(doc_id, sig, rng, severity=severity)
                row = {
                    "case_id": case["case_id"],
                    "note_text": case["note_text"],
                    "structured_features": case["structured_features"],
                    "entities": case["entities"],
                    "source_doc_id": case["source_doc_id"],
                    "generation_type": case["generation_type"],
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_cases += 1
            n_docs += 1

    print(f"Wrote {n_cases} cases from {n_docs} documents to {out_path}")


if __name__ == "__main__":
    main()
