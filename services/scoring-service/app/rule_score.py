"""
Deterministic rule-based triage scoring (no ML).

Rules (first match wins after insufficient check):
1. ``chest pain`` / angina-like phrasing → high
2. ``fever`` and ``cough`` both present → medium
3. else → low

Insufficient: no usable text signals (empty corpus slice for scoring).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from services.shared.schemas_v1 import FeatureContribution, ScoreRequest


@dataclass(frozen=True)
class RuleOutcome:
    risk_available: bool
    label: str  # high | medium | low | insufficient_data
    score: float
    confidence: float
    narrative: str
    contributions: list[FeatureContribution]


_CHEST = re.compile(
    r"\b(chest\s+pain|angina|substernal\s+pain|cardiac\s+pain)\b",
    re.IGNORECASE,
)
_FEVER = re.compile(r"\bfever\b", re.IGNORECASE)
_COUGH = re.compile(r"\bcough\b", re.IGNORECASE)


def _blob_from_request(request: ScoreRequest) -> str:
    parts: list[str] = []
    for ent in request.entities:
        parts.append(ent.text.strip())
    sf = request.structured_features or {}
    for _k, v in sf.items():
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            parts.append(str(v))
        elif isinstance(v, dict):
            parts.append(" ".join(str(x) for x in v.values() if x not in (None, "", [], {})))
        elif isinstance(v, list):
            parts.extend(str(x) for x in v if x not in (None, "", [], {}))
    return " ".join(parts).strip()


def evaluate_rules(request: ScoreRequest) -> RuleOutcome:
    blob = _blob_from_request(request)

    if not blob:
        return RuleOutcome(
            risk_available=False,
            label="insufficient_data",
            score=0.0,
            confidence=0.0,
            narrative="No clinical text or structured features were provided for scoring.",
            contributions=[],
        )

    if _CHEST.search(blob):
        conf = 0.88
        narrative = (
            "Rule: acute chest-pain phrasing maps to elevated triage concern "
            "(rule-based, not diagnostic)."
        )
        return RuleOutcome(
            risk_available=True,
            label="high",
            score=0.9,
            confidence=conf,
            narrative=narrative,
            contributions=[
                FeatureContribution(feature="rule:chest_pain_phrase", contribution=conf),
            ],
        )

    if _FEVER.search(blob) and _COUGH.search(blob):
        conf = 0.72
        narrative = "Rule: fever together with cough maps to intermediate triage concern (rule-based)."
        return RuleOutcome(
            risk_available=True,
            label="medium",
            score=0.55,
            confidence=conf,
            narrative=narrative,
            contributions=[
                FeatureContribution(feature="rule:fever_and_cough", contribution=conf),
            ],
        )

    conf = 0.62
    narrative = (
        "Rule: default low triage band when no higher-priority phrases matched (rule-based)."
    )
    return RuleOutcome(
        risk_available=True,
        label="low",
        score=0.22,
        confidence=conf,
        narrative=narrative,
        contributions=[
            FeatureContribution(feature="rule:default_low", contribution=conf),
        ],
    )
