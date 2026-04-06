"""
ClarificationAgent: minimal follow-up questions when history is incomplete.

Deterministic, bounded: maps ``missing_inputs`` and optional structured vitals gap to
a fixed ordered question list (critical clinical fields first).
"""

from __future__ import annotations

from typing import Any

from app.agents.base import AgentResult, AgentRole, monotonic_ms

# Medical field priority: ask in this order when multiple gaps exist.
_PRIORITY: tuple[str, ...] = ("age", "duration", "severity", "vitals")

_QUESTIONS: dict[str, str] = {
    "age": "What is the patient's age (or age range)?",
    "duration": "How long have the current symptoms been present?",
    "severity": (
        "How severe are the symptoms (for example mild, moderate, severe, "
        "or a pain scale if applicable)?"
    ),
    "vitals": (
        "Please share relevant vitals if available "
        "(such as blood pressure, heart rate, temperature, and oxygen saturation)."
    ),
}

_SYNONYMS: dict[str, str] = {
    "patient_age": "age",
    "symptom_duration": "duration",
    "temporal": "duration",
    "time_course": "duration",
    "pain_severity": "severity",
    "symptom_severity": "severity",
    "vital_signs": "vitals",
    "blood_pressure": "vitals",
    "bp": "vitals",
}


def _normalize_key(raw: str) -> str | None:
    s = (raw or "").strip().lower()
    if not s:
        return None
    s = s.removeprefix("missing:").strip()
    return _SYNONYMS.get(s, s)


def _canonicalize(missing_inputs: list[str]) -> list[str]:
    buckets: set[str] = set()
    for item in missing_inputs:
        key = _normalize_key(str(item))
        if key and key in _QUESTIONS:
            buckets.add(key)
    return [p for p in _PRIORITY if p in buckets]


def _vitals_effectively_missing(structured_features: dict[str, Any] | None) -> bool:
    if not structured_features:
        return True
    sf = structured_features
    # Any one useful vital reduces need to ask for vitals broadly.
    if sf.get("systolic_bp") is not None and sf.get("diastolic_bp") is not None:
        return False
    if sf.get("heart_rate") is not None:
        return False
    if sf.get("spo2") is not None:
        return False
    if sf.get("temperature_c") is not None:
        return False
    return True


def run_clarification(
    missing_inputs: list[str],
    *,
    structured_features: dict[str, Any] | None = None,
    include_vitals_if_sparse: bool = True,
) -> AgentResult:
    """
    Build ordered ``questions`` from ``missing_inputs``.

    If ``include_vitals_if_sparse`` and structured vitals look empty, append the
    vitals question when there is already at least one other gap (avoids nagging on
    pure admin notes).
    """
    t0 = monotonic_ms()
    warnings: list[str] = []
    keys = _canonicalize(list(missing_inputs))

    if (
        include_vitals_if_sparse
        and _vitals_effectively_missing(structured_features)
        and "vitals" not in keys
        and keys
    ):
        keys = [*keys, "vitals"]

    keys = list(dict.fromkeys(keys))
    questions = [_QUESTIONS[k] for k in keys]

    if not questions and not missing_inputs:
        warnings.append("clarification:no_missing_fields")
    elif not questions and missing_inputs:
        warnings.append("clarification:unmapped_missing_fields")

    confidence = 0.88 if questions else 0.45
    trace = {
        "agent_role": AgentRole.CLARIFICATION,
        "bounded_remote_calls": 0,
        "normalized_gaps": keys,
        "question_count": len(questions),
    }

    return AgentResult(
        agent_id=AgentRole.CLARIFICATION,
        ok=True,
        confidence=confidence,
        warnings=warnings,
        missing_inputs=list(keys),
        payload={"questions": questions},
        error_detail=None,
        duration_ms=monotonic_ms() - t0,
        trace=trace,
    )


class ClarificationAgent:
    """Namespace for clarification; prefer :func:`run_clarification`."""

    run = staticmethod(run_clarification)
