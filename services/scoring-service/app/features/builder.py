"""
Shared feature builder: turn ``ScoreRequest`` into a canonical signal vector.

All targets consume ``ExtractedFeatures``; individual models map or weight these
signals without re-running extraction. Add new rules here when expanding the
shared clinical signal catalog (e.g. for future ``stroke_risk`` / ``diabetes_risk``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.shared.schemas_v1 import ScoreRequest

from app import rules


@dataclass(frozen=True)
class ExtractedFeatures:
    """Neutral signal weights keyed by extractor output id (shared across targets)."""

    trace_id: str
    signals: dict[str, float]
    structured_features: dict[str, Any] = field(default_factory=dict)
    entity_count: int = 0


def _merge_max_per_key(pairs: list[tuple[str, float]]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for feature, weight in pairs:
        merged[feature] = max(merged.get(feature, 0.0), weight)
    return merged


def extract_features(request: ScoreRequest) -> ExtractedFeatures:
    """Run entity + structured extractors once per request."""
    fired = rules.collect_entity_contributions(
        request.entities
    ) + rules.collect_structured_contributions(request.structured_features)
    signals = _merge_max_per_key(fired)
    structured = dict(request.structured_features or {})
    return ExtractedFeatures(
        trace_id=request.trace_id,
        signals=signals,
        structured_features=structured,
        entity_count=len(request.entities or []),
    )
