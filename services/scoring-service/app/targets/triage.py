"""
Triage severity: maps shared extractor signals to triage-specific dimensions.

Thresholds and explanation feature ids are owned here; other targets must not
reuse these names. Add / adjust mappings as clinical policy evolves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from app.features import ExtractedFeatures
from app.score_math import assign_label, clamp_score_and_explanation
from app.targets.base import TargetPrediction


@dataclass(frozen=True)
class _TriageSignalMap:
    """Maps one shared extractor output into a triage explanation dimension."""

    source_signal: str
    triage_feature: str
    weight: float


# Shared clinical signal -> triage axis (max-merge per triage_feature, then sum axes).
_TRIAGE_MAP: tuple[_TriageSignalMap, ...] = (
    _TriageSignalMap("symptom_chest_pain", "triage_acute_cardiopulmonary", 1.0),
    _TriageSignalMap("symptom_dyspnea", "triage_acute_cardiopulmonary", 0.95),
    _TriageSignalMap("neuro_focal_deficit", "triage_neurovascular_acuity", 1.0),
    _TriageSignalMap("neuro_speech", "triage_neurovascular_acuity", 0.92),
    _TriageSignalMap("symptom_syncope", "triage_neurovascular_acuity", 0.75),
    _TriageSignalMap("infection_sepsis", "triage_systemic_illness", 1.0),
    _TriageSignalMap("disease_stroke", "triage_neurovascular_acuity", 0.88),
    _TriageSignalMap("disease_heart_failure", "triage_chronic_cardiometabolic", 0.72),
    _TriageSignalMap("disease_cad", "triage_chronic_cardiometabolic", 0.7),
    _TriageSignalMap("disease_hypertension", "triage_chronic_cardiometabolic", 0.58),
    _TriageSignalMap("disease_diabetes", "triage_chronic_cardiometabolic", 0.62),
    _TriageSignalMap("disease_copd", "triage_chronic_cardiometabolic", 0.55),
    _TriageSignalMap("disease_ckd", "triage_chronic_cardiometabolic", 0.58),
    _TriageSignalMap("bp_keyword_elevated", "triage_hemodynamic_stress", 0.55),
    _TriageSignalMap("bp_systolic_elevated", "triage_hemodynamic_stress", 1.0),
    _TriageSignalMap("bp_diastolic_elevated", "triage_hemodynamic_stress", 0.85),
    _TriageSignalMap("age_older_adult", "triage_age_related_vulnerability", 0.65),
    _TriageSignalMap("bmi_obesity", "triage_chronic_cardiometabolic", 0.48),
    _TriageSignalMap("smoking_current", "triage_chronic_cardiometabolic", 0.42),
    _TriageSignalMap("anticoagulant", "triage_bleeding_medication_context", 0.5),
)


def _thresholds() -> tuple[float, float]:
    return (
        float(os.getenv("SCORING_TRIAGE_MEDIUM_THRESHOLD", "0.30")),
        float(os.getenv("SCORING_TRIAGE_HIGH_THRESHOLD", "0.55")),
    )


def _triage_signal_vector(features: ExtractedFeatures) -> dict[str, float]:
    by_dim: dict[str, float] = {}
    for m in _TRIAGE_MAP:
        raw = features.signals.get(m.source_signal)
        if raw is None or raw <= 0.0:
            continue
        contrib = raw * m.weight
        by_dim[m.triage_feature] = max(by_dim.get(m.triage_feature, 0.0), contrib)
    return by_dim


class TriageSeverityTarget:
    target_id = "triage_severity"

    def predict(self, features: ExtractedFeatures) -> TargetPrediction:
        triage_signals = _triage_signal_vector(features)
        score, explanation = clamp_score_and_explanation(triage_signals)
        med, high = _thresholds()
        label = assign_label(score, med, high)
        return TargetPrediction(
            score=score,
            label=label,
            explanation=explanation,
            ready=True,
            detail=None,
        )
