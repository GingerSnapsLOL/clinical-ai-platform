#!/usr/bin/env python3
"""Extract lightweight clinical signal tags from merged documents (keywords + regex, no ML).

Reads ``data/processed/merged_docs.jsonl`` and writes ``data/processed/signals_docs.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

# --- Symptom phrases (multi-word first so longer matches can be checked before singles if needed) ---
SYMPTOM_TERMS: tuple[str, ...] = (
    "shortness of breath",
    "difficulty breathing",
    "chest pain",
    "abdominal pain",
    "stomach pain",
    "sore throat",
    "muscle pain",
    "joint pain",
    "back pain",
    "neck pain",
    "blurred vision",
    "loss of appetite",
    "weight loss",
    "weight gain",
    "night sweats",
    "swollen lymph",
    "rapid heartbeat",
    "irregular heartbeat",
    "blood in urine",
    "blood in stool",
    "loss of consciousness",
    "fever",
    "chills",
    "cough",
    "headache",
    "migraine",
    "nausea",
    "vomiting",
    "diarrhea",
    "constipation",
    "fatigue",
    "dizziness",
    "weakness",
    "rash",
    "hives",
    "itching",
    "swelling",
    "numbness",
    "tingling",
    "confusion",
    "seizure",
    "seizures",
    "tremor",
    "wheezing",
    "hoarseness",
    "heartburn",
    "indigestion",
    "dehydration",
    "bleeding",
    "bruising",
    "jaundice",
    "palpitations",
    "syncope",
    "fainting",
    "insomnia",
    "anxiety attack",
    "panic attack",
)

# --- Disease / condition names ---
DISEASE_TERMS: tuple[str, ...] = (
    "coronary artery disease",
    "heart failure",
    "heart attack",
    "myocardial infarction",
    "atrial fibrillation",
    "high blood pressure",
    "type 1 diabetes",
    "type 2 diabetes",
    "gestational diabetes",
    "chronic kidney disease",
    "kidney failure",
    "liver disease",
    "cirrhosis",
    "hepatitis b",
    "hepatitis c",
    "copd",
    "emphysema",
    "chronic bronchitis",
    "pneumonia",
    "bronchitis",
    "tuberculosis",
    "influenza",
    "covid-19",
    "covid 19",
    "coronavirus",
    "asthma",
    "hay fever",
    "allergic rhinitis",
    "eczema",
    "psoriasis",
    "lupus",
    "rheumatoid arthritis",
    "osteoarthritis",
    "osteoporosis",
    "fibromyalgia",
    "multiple sclerosis",
    "parkinson",
    "alzheimer",
    "dementia",
    "epilepsy",
    "depression",
    "bipolar",
    "schizophrenia",
    "stroke",
    "transient ischemic attack",
    "tia",
    "deep vein thrombosis",
    "pulmonary embolism",
    "anemia",
    "sickle cell",
    "hypothyroidism",
    "hyperthyroidism",
    "celiac disease",
    "crohn",
    "ulcerative colitis",
    "ibs",
    "irritable bowel",
    "gout",
    "migraine disorder",
    "glaucoma",
    "cataract",
    "macular degeneration",
    "cancer",
    "tumor",
    "tumour",
    "malignancy",
    "melanoma",
    "lymphoma",
    "leukemia",
    "diabetes",
    "hypertension",
    "obesity",
    "hiv",
    "aids",
    "hepatitis",
    "pancreatitis",
    "appendicitis",
    "meningitis",
    "sepsis",
    "endocarditis",
)

# --- Emergency / urgent-care language (patient-education style) ---
EMERGENCY_PHRASES: tuple[str, ...] = (
    "call 911",
    "call 9-1-1",
    "call nine one one",
    "call emergency services",
    "call an ambulance",
    "go to the emergency room",
    "go to the emergency department",
    "go to emergency",
    "go to the er",
    "go to er",
    "emergency room",
    "emergency department",
    "immediate medical attention",
    "seek immediate",
    "get help immediately",
    "right away",
    "without delay",
    "life-threatening",
    "life threatening",
    "medical emergency",
    "urgent care",
    "urgently",
    "severe emergency",
)

EMERGENCY_REGEXES: tuple[tuple[str, str], ...] = (
    ("911_mention", r"\b911\b"),
    ("emergency_number_eu", r"\b(?:112|999)\b"),
    ("er_visit", r"\b(?:er|ed)\s+(?:visit|right\s+away|immediately)\b"),
)

# --- Risk factors ---
RISK_TERMS: tuple[str, ...] = (
    "smoking",
    "smoker",
    "smokers",
    "cigarette",
    "cigarettes",
    "tobacco",
    "secondhand smoke",
    "second-hand smoke",
    "vaping",
    "e-cigarette",
    "hypertension",
    "high blood pressure",
    "elevated blood pressure",
    "diabetes",
    "diabetic",
    "prediabetes",
    "pre-diabetes",
    "insulin resistance",
    "obesity",
    "obese",
    "overweight",
    "bmi",
    "high cholesterol",
    "hyperlipidemia",
    "dyslipidemia",
    "family history",
    "sedentary",
    "physical inactivity",
    "alcohol use",
    "heavy drinking",
    "substance abuse",
    "drug use",
)

# Age-like phrases: capture normalized label once per doc
RISK_AGE_REGEX = re.compile(
    r"(?:\b(?:age|aged)\s*[:\-]?\s*\d{1,3}\b"
    r"|\b\d{1,3}\s*(?:years?\s*old|y\.?o\.?)\b"
    r"|\b(?:over|under|above|below)\s+\d{1,3}\s+years?\b)",
    re.IGNORECASE,
)


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Whole phrase, case-insensitive; allow flexible whitespace."""
    parts = phrase.split()
    if not parts:
        return re.compile(r"(?!x)x")
    inner = r"\s+".join(re.escape(p) for p in parts)
    return re.compile(rf"\b{inner}\b", re.IGNORECASE)


def _single_word_pattern(word: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)


def _build_term_patterns(terms: Iterable[str]) -> list[tuple[str, re.Pattern[str]]]:
    multi = [t for t in terms if " " in t]
    single = [t for t in terms if " " not in t]
    out: list[tuple[str, re.Pattern[str]]] = []
    for t in sorted(multi, key=len, reverse=True):
        out.append((t, _phrase_pattern(t)))
    for t in sorted(single, key=len, reverse=True):
        out.append((t, _single_word_pattern(t)))
    return out


_SYMPTOM_PATTERNS = _build_term_patterns(SYMPTOM_TERMS)
_DISEASE_PATTERNS = _build_term_patterns(DISEASE_TERMS)
_EMERGENCY_PATTERNS = [(p, _phrase_pattern(p)) for p in EMERGENCY_PHRASES]
_RISK_PATTERNS = _build_term_patterns(RISK_TERMS)


def _collect_matches(
    text: str,
    labeled_patterns: list[tuple[str, re.Pattern[str]]],
    *,
    normalize_label: bool = True,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for label, pat in labeled_patterns:
        if pat.search(text):
            out_label = label.lower() if normalize_label else label
            if out_label not in seen:
                seen.add(out_label)
                ordered.append(out_label)
    return ordered


def _collect_emergency(text: str) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    has_phrase_911 = False
    for label, pat in _EMERGENCY_PATTERNS:
        if pat.search(text):
            low = label.lower()
            if "911" in label or "nine one" in low:
                has_phrase_911 = True
            if low not in seen:
                seen.add(low)
                ordered.append(low)
    for key, raw in EMERGENCY_REGEXES:
        if key == "911_mention" and has_phrase_911:
            continue
        if re.search(raw, text, re.IGNORECASE):
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


def _collect_risk_factors(text: str) -> list[str]:
    factors = _collect_matches(text, _RISK_PATTERNS, normalize_label=True)
    if RISK_AGE_REGEX.search(text):
        if "age_mention" not in factors:
            factors.append("age_mention")
    return factors


def extract_signals(text: str) -> dict[str, list[str]]:
    """Return four signal lists for a single document body."""
    if not text or not text.strip():
        return {
            "symptoms": [],
            "diseases": [],
            "risk_factors": [],
            "emergency_flags": [],
        }
    return {
        "symptoms": _collect_matches(text, _SYMPTOM_PATTERNS),
        "diseases": _collect_matches(text, _DISEASE_PATTERNS),
        "risk_factors": _collect_risk_factors(text),
        "emergency_flags": _collect_emergency(text),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("data/processed/merged_docs.jsonl"),
        help="Input JSONL (default: data/processed/merged_docs.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/processed/signals_docs.jsonl"),
        help="Output JSONL (default: data/processed/signals_docs.jsonl)",
    )
    args = parser.parse_args()
    in_path: Path = args.input
    out_path: Path = args.output

    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        for rec in _read_jsonl(in_path):
            doc_id = rec.get("doc_id") or rec.get("id") or ""
            text = (rec.get("text") or "").strip()
            signals = extract_signals(text)
            row = {
                "doc_id": str(doc_id),
                "text": text,
                "signals": signals,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} records to {out_path}")


if __name__ == "__main__":
    main()
