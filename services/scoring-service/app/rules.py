"""
Deterministic rule tables for the scorer.

Entity rules use case-insensitive substring match with optional NER type filtering.
Structured rules use numeric thresholds (>=) or strict boolean ``True`` flags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from services.shared.schemas_v1 import EntityItem


@dataclass(frozen=True)
class EntityKeywordRule:
    """Match when any entity text contains ``keyword`` (case-insensitive substring)."""

    feature: str
    weight: float
    keyword: str
    entity_types: frozenset[str] | None = None

    def matches_any(self, entities: list[EntityItem]) -> bool:
        kw = self.keyword.lower()
        allowed: set[str] | None = None
        if self.entity_types is not None:
            allowed = {t.upper() for t in self.entity_types}
        for e in entities:
            if allowed is not None and e.type.upper() not in allowed:
                continue
            if kw in e.text.strip().lower():
                return True
        return False


@dataclass(frozen=True)
class StructuredNumericRule:
    """Match when ``key`` resolves to a numeric value ``>= min_inclusive``."""

    feature: str
    weight: float
    key: str
    min_inclusive: float

    def matches(self, structured_features: dict[str, Any]) -> bool:
        v = _coerce_number(structured_features.get(self.key))
        if v is None:
            return False
        return v >= self.min_inclusive


@dataclass(frozen=True)
class StructuredFlagRule:
    """Match when ``key`` is present with JSON/boolean ``true``."""

    feature: str
    weight: float
    key: str

    def matches(self, structured_features: dict[str, Any]) -> bool:
        return structured_features.get(self.key) is True


ENTITY_KEYWORD_RULES: tuple[EntityKeywordRule, ...] = (
    EntityKeywordRule(
        "disease_hypertension",
        0.12,
        "hypertension",
        frozenset({"DISEASE"}),
    ),
    EntityKeywordRule(
        "disease_diabetes",
        0.14,
        "diabetes",
        frozenset({"DISEASE"}),
    ),
    EntityKeywordRule("disease_diabetes", 0.14, "diabetic", frozenset({"DISEASE"})),
    EntityKeywordRule(
        "disease_copd",
        0.11,
        "copd",
        frozenset({"DISEASE"}),
    ),
    EntityKeywordRule(
        "disease_heart_failure",
        0.13,
        "heart failure",
        frozenset({"DISEASE"}),
    ),
    EntityKeywordRule(
        "disease_cad",
        0.13,
        "coronary",
        frozenset({"DISEASE"}),
    ),
    EntityKeywordRule(
        "disease_stroke",
        0.12,
        "stroke",
        frozenset({"DISEASE"}),
    ),
    EntityKeywordRule(
        "disease_ckd",
        0.12,
        "chronic kidney",
        frozenset({"DISEASE"}),
    ),
    EntityKeywordRule(
        "bp_keyword_elevated",
        0.06,
        "high blood pressure",
        None,
    ),
    EntityKeywordRule("symptom_chest_pain", 0.22, "chest pain", None),
    EntityKeywordRule("symptom_chest_pain", 0.2, "substernal", None),
    EntityKeywordRule("symptom_dyspnea", 0.2, "shortness of breath", None),
    EntityKeywordRule("symptom_dyspnea", 0.2, "dyspnea", None),
    EntityKeywordRule("neuro_focal_deficit", 0.24, "facial droop", None),
    EntityKeywordRule("neuro_speech", 0.2, "slurred speech", None),
    EntityKeywordRule("symptom_syncope", 0.16, "syncope", None),
    EntityKeywordRule(
        "infection_sepsis",
        0.2,
        "sepsis",
        frozenset({"DISEASE"}),
    ),
    EntityKeywordRule("infection_sepsis", 0.18, "septic", None),
)

STRUCTURED_NUMERIC_RULES: tuple[StructuredNumericRule, ...] = (
    StructuredNumericRule("bp_systolic_elevated", 0.15, "systolic_bp", 140.0),
    StructuredNumericRule("bp_diastolic_elevated", 0.1, "diastolic_bp", 90.0),
    StructuredNumericRule("age_older_adult", 0.1, "age", 65.0),
    StructuredNumericRule("bmi_obesity", 0.08, "bmi", 30.0),
)

STRUCTURED_FLAG_RULES: tuple[StructuredFlagRule, ...] = (
    StructuredFlagRule("smoking_current", 0.12, "smoking_current"),
    StructuredFlagRule("anticoagulant", 0.09, "on_anticoagulant"),
)


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collect_entity_contributions(entities: list[EntityItem]) -> list[tuple[str, float]]:
    fired: list[tuple[str, float]] = []
    for rule in ENTITY_KEYWORD_RULES:
        if rule.matches_any(entities):
            fired.append((rule.feature, rule.weight))
    return fired


def collect_structured_contributions(
    structured_features: dict[str, Any],
) -> list[tuple[str, float]]:
    fired: list[tuple[str, float]] = []
    for rule in STRUCTURED_NUMERIC_RULES:
        if rule.matches(structured_features):
            fired.append((rule.feature, rule.weight))
    for rule in STRUCTURED_FLAG_RULES:
        if rule.matches(structured_features):
            fired.append((rule.feature, rule.weight))
    return fired
