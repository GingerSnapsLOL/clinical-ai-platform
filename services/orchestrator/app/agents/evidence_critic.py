"""
EvidenceCriticAgent: conservative validation of retrieval, scoring, and structuring signals.

Deterministic only (no LLM). Bounded: O(n) over entities, sources, and feature dicts.
"""

from __future__ import annotations

import os
from typing import Any

from services.shared.schemas_v1 import EntityItem, SourceItem

from app.agents.base import AgentResult, AgentRole, monotonic_ms
from app.relevance import retrieval_meets_relevance_bar

# Conservative thresholds
_ACUITY_STRONG_MIN = 3
_SCORE_LOW_FOR_STRONG_ACUITY = 0.12
_SCORE_HIGH_WITHOUT_FEATURES = 0.55
_HEDGE_SCORE_CONTRADICTION = 0.55
_HIGH_LABEL_SCORE_MIN = 0.65
_COVERAGE_WEAK = 0.32
_UPSTREAM_CONF_PRODUCT_SUSPICIOUS = 0.62


def _meaningful_structured_count(sf: dict[str, Any]) -> int:
    n = 0
    for _k, v in sf.items():
        if v is None or v is False or v == "":
            continue
        if isinstance(v, (int, float)) and float(v) == 0.0:
            continue
        n += 1
    return n


def _acuity_hints(entities: list[EntityItem], sf: dict[str, Any]) -> int:
    """Crude 0–5 acuity score aligned with triage-relevant cues (not a full re-score)."""
    n = 0
    acute_kw = (
        "chest pain",
        "substernal",
        "dyspnea",
        "shortness of breath",
        "syncope",
        "sepsis",
        "stroke",
        "worst headache",
        "slurred speech",
    )
    for e in entities:
        t = (e.text or "").lower()
        if any(k in t for k in acute_kw):
            n += 2
    try:
        if float(sf.get("systolic_bp") or 0) >= 140:
            n += 1
    except (TypeError, ValueError):
        pass
    try:
        if float(sf.get("age") or 0) >= 65:
            n += 1
    except (TypeError, ValueError):
        pass
    if sf.get("smoking_current") is True:
        n += 1
    if sf.get("on_anticoagulant") is True:
        n += 1
    return min(5, n)


def run_evidence_critic(
    *,
    sources: list[SourceItem],
    entities: list[EntityItem],
    structured_features: dict[str, Any],
    signals: dict[str, Any],
    scoring_payload: dict[str, Any],
    retrieval_payload: dict[str, Any] | None = None,
    missing_inputs: list[str] | None = None,
    retrieval_step_confidence: float | None = None,
    scoring_step_confidence: float | None = None,
) -> AgentResult:
    """
    Return legacy :class:`AgentResult` with payload
    ``valid``, ``approved`` (mirror), ``issues``, ``confidence_adjustment``.
    """
    t0 = monotonic_ms()
    issues: list[str] = []
    rp = dict(retrieval_payload or {})
    sf = dict(structured_features or {})
    sig = dict(signals or {})
    sp = dict(scoring_payload or {})
    miss = list(missing_inputs or [])

    primary_score = float(sp.get("score") or 0.0)
    label = str(sp.get("label") or "low").lower()
    coverage = rp.get("coverage_score")
    try:
        coverage_f = float(coverage) if coverage is not None else None
    except (TypeError, ValueError):
        coverage_f = None

    acuity = _acuity_hints(entities, sf)
    m_sf = _meaningful_structured_count(sf)

    rel_ok, _top_rel, rel_reason = retrieval_meets_relevance_bar(sources)
    if not sources:
        issues.append("insufficient_data")
    elif not rel_ok:
        issues.append("weak_evidence")

    if coverage_f is not None and coverage_f < _COVERAGE_WEAK and "weak_evidence" not in issues:
        issues.append("weak_evidence")

    if len(miss) >= 2:
        issues.append("insufficient_data")

    if acuity >= _ACUITY_STRONG_MIN and primary_score < _SCORE_LOW_FOR_STRONG_ACUITY:
        issues.append("score_not_supported")

    if (
        acuity == 0
        and not entities
        and m_sf == 0
        and primary_score > _SCORE_HIGH_WITHOUT_FEATURES
    ):
        issues.append("score_not_supported")

    unc = sig.get("uncertainty") if isinstance(sig.get("uncertainty"), dict) else {}
    if unc.get("hedging_language") and primary_score > _HEDGE_SCORE_CONTRADICTION:
        issues.append("contradictory_signals")
    if unc.get("severity_unspecified") and label == "high":
        issues.append("contradictory_signals")
    if unc.get("vague_timing") and label == "high" and acuity >= 2:
        issues.append("contradictory_signals")
    if unc.get("hedging_language") and label == "high" and primary_score > _HIGH_LABEL_SCORE_MIN:
        issues.append("contradictory_signals")

    rc = float(retrieval_step_confidence) if retrieval_step_confidence is not None else None
    sc_ = float(scoring_step_confidence) if scoring_step_confidence is not None else None
    if rc is not None and sc_ is not None:
        prod = rc * sc_
        weak_ctx = "weak_evidence" in issues or (coverage_f is not None and coverage_f < 0.45)
        if weak_ctx and prod > _UPSTREAM_CONF_PRODUCT_SUSPICIOUS:
            issues.append("high_confidence_weak_data")

    issues = list(dict.fromkeys(issues))

    # Strict: any issue => invalid
    valid = len(issues) == 0

    adj = 1.0
    multipliers = {
        "insufficient_data": 0.62,
        "contradictory_signals": 0.72,
        "score_not_supported": 0.78,
        "weak_evidence": 0.84,
        "high_confidence_weak_data": 0.88,
    }
    for issue in issues:
        adj *= multipliers.get(issue, 0.90)
    adj = max(0.15, min(1.0, adj))

    trace = {
        "agent_role": AgentRole.EVIDENCE_CRITIC,
        "bounded_remote_calls": 0,
        "acuity_hint": acuity,
        "relevance_ok": rel_ok if sources else False,
        "relevance_reason": rel_reason if sources else "no_passages",
        "critic_strict": os.getenv("ORCHESTRATOR_EVIDENCE_CRITIC_STRICT", "true").lower()
        not in ("0", "false", "no", "off"),
    }
    duration_ms = monotonic_ms() - t0

    payload: dict[str, Any] = {
        "valid": valid,
        "approved": valid,
        "issues": list(issues),
        "confidence_adjustment": adj,
    }

    return AgentResult(
        agent_id=AgentRole.EVIDENCE_CRITIC,
        ok=valid,
        confidence=adj,
        warnings=list(issues),
        payload=payload,
        error_detail=None if valid else "evidence_critic_rejected",
        duration_ms=duration_ms,
        trace=trace,
    )


class EvidenceCriticAgent:
    """Namespace for the critic; use :func:`run_evidence_critic` for the agent step."""

    run = staticmethod(run_evidence_critic)
