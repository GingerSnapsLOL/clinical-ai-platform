"""Model registry: register targets here when adding ``stroke_risk`` / ``diabetes_risk``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.targets.cardiovascular import CardiovascularRiskTarget
from app.targets.stub_targets import NotTrainedTarget
from app.targets.triage import TriageSeverityTarget

if TYPE_CHECKING:
    from app.targets.base import ScoringTarget

# Primary when ``targets`` is omitted from ``ScoreRequest``.
DEFAULT_PRIMARY_TARGET = "triage_severity"

_KNOWN: tuple[str, ...] = (
    "triage_severity",
    "stroke_risk",
    "cardiovascular_risk",
    "diabetes_risk",
)


def _build_registry() -> dict[str, ScoringTarget]:
    return {
        "triage_severity": TriageSeverityTarget(),
        "cardiovascular_risk": CardiovascularRiskTarget(),
        "stroke_risk": NotTrainedTarget("stroke_risk"),
        "diabetes_risk": NotTrainedTarget("diabetes_risk"),
    }


TARGET_REGISTRY: dict[str, ScoringTarget] = _build_registry()


def valid_target_ids() -> frozenset[str]:
    return frozenset(_KNOWN)


def get_target(target_id: str) -> ScoringTarget | None:
    return TARGET_REGISTRY.get(target_id)
