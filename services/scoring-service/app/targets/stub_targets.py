"""
Placeholder targets until trained models are wired in.

To add ``stroke_risk`` / ``diabetes_risk``: implement ``ScoringTarget`` in
``stroke.py`` / ``diabetes.py`` (same pattern as ``triage.py`` / ``cardiovascular.py``),
then swap the ``NotTrainedTarget`` entry in ``registry._build_registry``.
"""

from __future__ import annotations

from app.features import ExtractedFeatures
from app.targets.base import TargetPrediction


class NotTrainedTarget:
    """Preserves API shape; ``ready=False`` and empty explanation."""

    def __init__(self, target_id: str):
        self.target_id = target_id

    def predict(self, features: ExtractedFeatures) -> TargetPrediction:
        return TargetPrediction(
            score=0.0,
            label="low",
            explanation=[],
            ready=False,
            detail="Model not trained",
        )
