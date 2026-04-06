"""
SafetyAgent: deterministic emergency / warning triage on clinical text.

No LLM, no network. Flags patterns that require escalation and attaches compliance
actions (no definitive diagnosis; cautious language).
"""

from __future__ import annotations

import re
from typing import Any, Literal

from services.shared.schemas_v1 import EntityItem

from app.agents.base import AgentResult, AgentRole, monotonic_ms

SafetyLevel = Literal["normal", "warning", "emergency"]

_CHEST_PAIN = re.compile(
    r"\b(?:chest pain|chest pressure|substernal pain|angina|anginal|precordial pain|pressure.*chest)\b",
    re.I,
)
_DYSPNEA = re.compile(
    r"\b(?:dyspnea|dyspnoea|shortness of breath|difficulty breathing|air hunger)\b|\bsob\b",
    re.I,
)

_STROKE = re.compile(
    r"\b(?:facial droop|face droop|slurred speech|dysarthria|aphasia|"
    r"sudden weakness|hemiparesis|hemiplegia|unilateral weakness|"
    r"acute stroke|\bstroke\b.*symptom|sudden numbness|"
    r"worst headache|thunderclap|vision loss.*sudden)\b",
    re.I,
)

_SEPSIS_STRONG = re.compile(
    r"\b(?:sepsis|septic shock|severe sepsis|bacteremic shock|"
    r"endotoxic shock|vasopressor.*sepsis)\b",
    re.I,
)
_SEPSIS_INDICATORS = re.compile(
    r"\b(?:rigors|rigour|qsofa|sirs criteria|lactate.*(?:elevated|high)|"
    r"febrile.*hypotens|hypotens.*febrile|peritonitis|"
    r"toxic appearance|altered mental status.*infection)\b",
    re.I,
)

_SINGLE_ACUTE = re.compile(
    r"\b(?:chest pain|chest pressure|substernal|shortness of breath|dyspnea|syncope|"
    r"hemoptysis|acute abdomen)\b",
    re.I,
)


def _scan_blob(note_text: str, question: str | None) -> str:
    parts = [(question or "").strip(), (note_text or "").strip()]
    return "\n\n".join(p for p in parts if p)


def _entity_text_join(entities: list[EntityItem] | None) -> str:
    if not entities:
        return ""
    return " ".join((e.text or "").strip() for e in entities)


def _chest_pain_and_dyspnea(blob: str) -> bool:
    return bool(_CHEST_PAIN.search(blob) and _DYSPNEA.search(blob))


def _stroke_pattern(blob: str) -> bool:
    return bool(_STROKE.search(blob))


def _sepsis_pattern(blob: str) -> bool:
    if _SEPSIS_STRONG.search(blob):
        return True
    return bool(_SEPSIS_INDICATORS.search(blob))


def _warning_without_emergency(blob: str, emergency: bool) -> bool:
    if emergency:
        return False
    if _SINGLE_ACUTE.search(blob) and not _chest_pain_and_dyspnea(blob):
        return True
    return False


def run_safety(
    *,
    note_text: str,
    question: str | None = None,
    entities: list[EntityItem] | None = None,
) -> AgentResult:
    """Assess text + optional entities; return safety envelope in ``payload``."""
    t0 = monotonic_ms()
    blob = _scan_blob(note_text, question)
    entity_blob = _entity_text_join(entities)
    combined = f"{blob}\n{entity_blob}".strip()

    hits: dict[str, bool] = {
        "chest_pain_dyspnea": _chest_pain_and_dyspnea(combined),
        "stroke_symptoms": _stroke_pattern(combined),
        "sepsis_indicators": _sepsis_pattern(combined),
    }
    emergency = any(hits.values())
    warning = _warning_without_emergency(combined, emergency)

    if emergency:
        level: SafetyLevel = "emergency"
        actions = [
            "escalate_emergency_care",
            "seek_immediate_in_person_evaluation",
            "avoid_definitive_diagnosis",
            "use_cautious_language",
        ]
        message_prefix = (
            "Potential emergency pattern detected. This tool does not provide a diagnosis. "
            "If symptoms are severe, new, worsening, or you are concerned, seek urgent in-person "
            "or emergency medical care now. "
        )
        confidence = 0.88
    elif warning:
        level = "warning"
        actions = [
            "avoid_definitive_diagnosis",
            "use_cautious_language",
            "consider_urgent_clinical_review_if_worsening",
        ]
        message_prefix = (
            "Some features may warrant prompt clinical attention. "
            "This is informational only and not a diagnosis. "
        )
        confidence = 0.82
    else:
        level = "normal"
        actions = [
            "avoid_definitive_diagnosis",
            "use_cautious_language",
        ]
        message_prefix = None
        confidence = 0.94

    payload: dict[str, Any] = {
        "safety_level": level,
        "actions": actions,
    }
    if message_prefix is not None:
        payload["message_prefix"] = message_prefix

    trace = {
        "agent_role": AgentRole.SAFETY,
        "bounded_remote_calls": 0,
        "pattern_hits": {k: v for k, v in hits.items() if v},
        "safety_level": level,
    }

    warnings: list[str] = []
    if emergency:
        warnings.append("safety:emergency_pattern")
    elif warning:
        warnings.append("safety:warning_pattern")

    return AgentResult(
        agent_id=AgentRole.SAFETY,
        ok=True,
        confidence=confidence,
        warnings=warnings,
        payload=payload,
        error_detail=None,
        duration_ms=monotonic_ms() - t0,
        trace=trace,
    )


class SafetyAgent:
    """Namespace for safety checks; use :func:`run_safety`."""

    run = staticmethod(run_safety)
