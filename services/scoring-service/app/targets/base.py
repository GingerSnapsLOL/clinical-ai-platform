"""Target model protocol and prediction value object."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from services.shared.schemas_v1 import FeatureContribution

from app.features import ExtractedFeatures


@dataclass
class TargetPrediction:
    score: float
    label: Literal["low", "medium", "high"]
    explanation: list[FeatureContribution]
    ready: bool = True
    detail: str | None = None


@runtime_checkable
class ScoringTarget(Protocol):
    target_id: str

    def predict(self, features: ExtractedFeatures) -> TargetPrediction:
        ...
