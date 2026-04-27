"""Deterministic rule-based triage scoring (no ML)."""

from __future__ import annotations

from services.shared.logging_util import set_trace_id
from services.shared.schemas_v1 import ScoreRequest, ScoreResponse, TargetScoreResult

from app.rule_score import RuleOutcome, evaluate_rules
from app.targets import DEFAULT_PRIMARY_TARGET


def _dedupe_preserve_order(target_ids: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in target_ids:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _normalize_run_list(request: ScoreRequest) -> list[str]:
    if not request.targets:
        return [DEFAULT_PRIMARY_TARGET]
    return _dedupe_preserve_order(list(request.targets))


def _triage_target_row(triage: RuleOutcome) -> TargetScoreResult:
    if not triage.risk_available:
        return TargetScoreResult(
            target="triage_severity",
            score=0.0,
            label="low",
            ready=False,
            detail="insufficient_data",
            explanation=[],
        )
    return TargetScoreResult(
        target="triage_severity",
        score=triage.score,
        label=triage.label,  # type: ignore[arg-type]
        ready=True,
        detail=None,
        explanation=list(triage.contributions),
    )


def _secondary_stub(tid: str) -> TargetScoreResult:
    return TargetScoreResult(
        target=tid,
        score=0.0,
        label="low",
        ready=False,
        detail="not_computed_rule_based_scorer",
        explanation=[],
    )


def compute_score(request: ScoreRequest) -> ScoreResponse:
    set_trace_id(request.trace_id)
    triage = evaluate_rules(request)
    run_ids = _normalize_run_list(request)

    predictions: dict[str, TargetScoreResult] = {}
    for tid in run_ids:
        if tid == "triage_severity":
            predictions[tid] = _triage_target_row(triage)
        else:
            predictions[tid] = _secondary_stub(tid)

    primary_id = run_ids[0]
    primary_tr = predictions[primary_id]

    if primary_id == "triage_severity":
        top_label = triage.label
        top_score = triage.score
        top_narrative = triage.narrative
        top_contributions = list(triage.contributions)
        top_risk_available = triage.risk_available
        top_conf = triage.confidence if triage.risk_available else None
    else:
        top_label = "low"
        top_score = primary_tr.score
        top_narrative = (
            f"Target {primary_id!r} is not computed by the rule-based scorer; "
            "see triage_severity in target_results when requested."
        )
        top_contributions = []
        top_risk_available = primary_tr.ready
        top_conf = None

    target_results: dict[str, TargetScoreResult] | None = None
    if request.targets:
        target_results = {tid: predictions[tid] for tid in run_ids}

    return ScoreResponse(
        trace_id=request.trace_id,
        score=top_score,
        label=top_label,  # type: ignore[arg-type]
        explanation=top_narrative,
        contributions=top_contributions,
        target_results=target_results,
        risk_available=top_risk_available,
        confidence=top_conf,
    )
