"""Shared score aggregation: clamp to [0, 1] and build explanation rows."""

from __future__ import annotations

from typing import Literal

from services.shared.schemas_v1 import FeatureContribution


def clamp_score_and_explanation(
    signals: dict[str, float],
) -> tuple[float, list[FeatureContribution]]:
    """
    Raw score is the sum of signal weights, then clamped to [0, 1].

    When raw > 1, the returned score is 1.0 and explanation rows are scaled
    by ``score / raw`` so contributions sum to the reported score.
    """
    raw = sum(signals.values())
    if raw <= 0.0:
        return 0.0, []

    score = min(1.0, max(0.0, raw))
    scale = score / raw

    explanation = [
        FeatureContribution(feature=k, contribution=round(v * scale, 6))
        for k, v in sorted(signals.items(), key=lambda kv: kv[0])
    ]
    return round(score, 6), explanation


def assign_label(
    score: float,
    medium_threshold: float,
    high_threshold: float,
) -> Literal["low", "medium", "high"]:
    if score < medium_threshold:
        return "low"
    if score < high_threshold:
        return "medium"
    return "high"
