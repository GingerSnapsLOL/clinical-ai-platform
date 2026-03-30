"""Cardiovascular risk: rule-based model over shared extracted signals."""

from __future__ import annotations

import os

from app.features import ExtractedFeatures
from app.score_math import assign_label, clamp_score_and_explanation
from app.targets.base import TargetPrediction


def _thresholds() -> tuple[float, float]:
    return (
        float(os.getenv("SCORING_CARDIOVASCULAR_MEDIUM_THRESHOLD", "0.35")),
        float(os.getenv("SCORING_CARDIOVASCULAR_HIGH_THRESHOLD", "0.65")),
    )


class CardiovascularRiskTarget:
    """Uses the full shared signal vector (entity + structured extraction rules)."""

    target_id = "cardiovascular_risk"

    def predict(self, features: ExtractedFeatures) -> TargetPrediction:
        score, explanation = clamp_score_and_explanation(features.signals)
        med, high = _thresholds()
        label = assign_label(score, med, high)
        return TargetPrediction(
            score=score,
            label=label,
            explanation=explanation,
            ready=True,
            detail=None,
        )
