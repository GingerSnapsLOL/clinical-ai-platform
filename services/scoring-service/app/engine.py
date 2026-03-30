"""Multi-target scoring: one extraction pass, then target-specific models."""

from __future__ import annotations

from services.shared.logging_util import set_trace_id
from services.shared.schemas_v1 import ScoreRequest, ScoreResponse, TargetScoreResult

from app.features import extract_features
from app.targets import DEFAULT_PRIMARY_TARGET, TARGET_REGISTRY


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


def compute_score(request: ScoreRequest) -> ScoreResponse:
    set_trace_id(request.trace_id)

    features = extract_features(request)
    run_ids = _normalize_run_list(request)
    predictions = {tid: TARGET_REGISTRY[tid].predict(features) for tid in run_ids}

    primary_id = run_ids[0]
    primary = predictions[primary_id]

    target_results: dict[str, TargetScoreResult] | None = None
    if request.targets:
        target_results = {
            tid: TargetScoreResult(
                target=tid,
                score=pred.score,
                label=pred.label,
                explanation=pred.explanation,
                ready=pred.ready,
                detail=pred.detail,
            )
            for tid, pred in predictions.items()
        }

    return ScoreResponse(
        trace_id=request.trace_id,
        score=primary.score,
        label=primary.label,
        explanation=primary.explanation,
        target_results=target_results,
    )
