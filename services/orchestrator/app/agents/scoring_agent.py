"""
ScoringAgent: selects scoring targets, calls /v1/score, interprets multi-target output.

Always requests triage_severity; optional extra targets from ORCHESTRATOR_SCORING_EXTRA_TARGETS.
"""

from __future__ import annotations

import os
from typing import Any

from services.shared.http_client import post_typed
from services.shared.logging_util import get_logger
from services.shared.schemas_v1 import EntityItem, ScoreRequest, ScoreResponse

from app.agents.base import AgentResult, AgentRole, SupervisorContext, monotonic_ms

logger = get_logger(__name__, "orchestrator")

PRIMARY_TARGET = "triage_severity"
# Must stay aligned with scoring-service registry (optional extras only).
_OPTIONAL_TARGETS = frozenset({"cardiovascular_risk", "stroke_risk", "diabetes_risk"})
_RELEVANT_ENTITY_TYPES = frozenset(
    {
        "SYMPTOM",
        "DISEASE",
        "CONDITION",
        "FINDING",
        "OBSERVATION",
        "PROBLEM",
        "DIAGNOSIS",
    }
)

_MAX_REMOTE_CALLS = 1


class ScoringAgent:
    """Target selection and score response shaping (no HTTP here)."""

    @staticmethod
    def has_relevant_entities(entities: list[EntityItem]) -> bool:
        """Return True when at least one clinically relevant entity is present."""
        for ent in entities:
            et = str(ent.type or "").strip().upper()
            if et in _RELEVANT_ENTITY_TYPES:
                return True
            txt = str(ent.text or "").strip()
            if txt and len(txt) >= 3:
                return True
        return False

    @staticmethod
    def select_targets(signals: dict[str, Any] | None) -> list[str]:
        """triage_severity first; append allowed extras from env (comma-separated)."""
        del signals  # reserved for future routing (e.g. activate cardiovascular when chest pain)
        out: list[str] = [PRIMARY_TARGET]
        raw = os.getenv("ORCHESTRATOR_SCORING_EXTRA_TARGETS", "").strip()
        for part in raw.split(","):
            t = part.strip()
            if t and t in _OPTIONAL_TARGETS and t not in out:
                out.append(t)
        return out

    @staticmethod
    def assess_input_quality(
        structured_features: dict[str, Any] | None,
        signals: dict[str, Any] | None,
        entities: list[EntityItem],
    ) -> tuple[list[str], float]:
        """
        Return supplemental warnings and an input-quality multiplier in (0, 1].

        Lower multiplier reduces agent confidence when features are thin.
        """
        warnings: list[str] = []
        sf = dict(structured_features or {})
        sig = dict(signals or {})
        n_ent = len(entities)
        n_sf = len(sf)
        meaningful_sf = sum(
            1
            for _k, v in sf.items()
            if v is not None and v is not False and v != "" and not (isinstance(v, float) and v == 0.0)
        )

        if n_ent == 0 and n_sf == 0:
            warnings.append("scoring_features_insufficient:no_entities_or_structured")
            return warnings, 0.38

        if n_ent == 0 and meaningful_sf == 0:
            warnings.append("scoring_features_insufficient:sparse_structured")
            return warnings, 0.52

        unc = sig.get("uncertainty") if isinstance(sig.get("uncertainty"), dict) else {}
        if unc.get("hedging_language"):
            warnings.append("scoring_input_hedging_language")

        quality = 0.92
        if n_ent < 1 and meaningful_sf < 2:
            quality = 0.68
            warnings.append("scoring_features_insufficient:thin_features")
        elif n_ent < 2 and meaningful_sf < 2:
            quality = 0.78
            warnings.append("scoring_features_insufficient:thin_features")

        return warnings, quality

    @staticmethod
    def interpret_response(resp: ScoreResponse, requested: list[str]) -> dict[str, Any]:
        """Build scores map, primary block, and overall ready flag."""
        scores: dict[str, Any] = {}
        if resp.target_results:
            for tid, tr in resp.target_results.items():
                scores[tid] = {
                    "score": tr.score,
                    "label": tr.label,
                    "ready": tr.ready,
                    "detail": tr.detail,
                    "explanation": [e.model_dump() for e in tr.explanation],
                }
        else:
            tid = requested[0]
            scores[tid] = {
                "score": resp.score,
                "label": resp.label,
                "ready": resp.risk_available if tid == "triage_severity" else True,
                "detail": None
                if (tid != "triage_severity" or resp.risk_available)
                else "insufficient_data",
                "explanation": [e.model_dump() for e in resp.contributions],
            }

        primary_id = requested[0]
        row = dict(scores.get(primary_id, {}))
        scores[primary_id] = {
            **row,
            "score": resp.score,
            "label": resp.label,
            "explanation": [e.model_dump() for e in resp.contributions],
        }
        if primary_id == "triage_severity":
            scores[primary_id]["ready"] = resp.risk_available
            scores[primary_id]["detail"] = None if resp.risk_available else "insufficient_data"

        primary_row = scores[primary_id]
        primary = {"target": primary_id, **primary_row}

        overall_ready = all(scores.get(t, {}).get("ready", False) for t in requested)
        return {
            "scores": scores,
            "primary": primary,
            "ready": overall_ready,
        }


async def run_scoring_step(
    ctx: SupervisorContext,
    entities: list[EntityItem],
    structured_features: dict[str, Any] | None = None,
    signals: dict[str, Any] | None = None,
) -> AgentResult:
    """POST /v1/score with explicit targets and structured payload for consumers."""
    t0 = monotonic_ms()
    trace: dict[str, Any] = {
        "bounded_remote_calls": _MAX_REMOTE_CALLS,
        "agent_role": AgentRole.SCORING,
    }
    warnings: list[str] = []
    sf = dict(structured_features or {})
    sig = dict(signals or {})

    if not ScoringAgent.has_relevant_entities(entities):
        return AgentResult(
            agent_id=AgentRole.SCORING,
            ok=True,
            confidence=0.0,
            warnings=["scoring_skipped:no_relevant_entities"],
            payload={},
            error_detail=None,
            duration_ms=monotonic_ms() - t0,
            trace=trace,
        )

    targets = ScoringAgent.select_targets(sig)
    trace["scoring_targets"] = list(targets)

    pre_warnings, input_quality = ScoringAgent.assess_input_quality(sf, sig, entities)
    warnings.extend(pre_warnings)
    trace["input_quality_multiplier"] = round(input_quality, 4)

    try:
        score_data, _, _ = await post_typed(
            ctx.client,
            ctx.urls["scoring"],
            ScoreRequest(
                trace_id=ctx.trace_id,
                entities=entities,
                structured_features=sf,
                targets=targets,
            ),
            ScoreResponse,
            timeout=ctx.timeout,
            trace_id=ctx.trace_id,
        )
        duration_ms = monotonic_ms() - t0
        if score_data is None:
            return AgentResult(
                agent_id=AgentRole.SCORING,
                ok=False,
                confidence=0.0,
                warnings=warnings + ["scoring_service_returned_empty"],
                payload={},
                error_detail="no_score",
                duration_ms=duration_ms,
                trace=trace,
            )

        interpreted = ScoringAgent.interpret_response(score_data, targets)
        scores: dict[str, Any] = interpreted["scores"]
        primary_block: dict[str, Any] = interpreted["primary"]
        overall_ready: bool = interpreted["ready"]

        for tid in targets:
            row = scores.get(tid, {})
            if not row.get("ready", True):
                detail = row.get("detail") or "not_ready"
                warnings.append(f"scoring_target_not_ready:{tid}")
                warnings.append(f"scoring_model_detail:{tid}:{detail}")

        if not overall_ready:
            warnings.append("scoring_partial_ready:not_all_targets_ready")

        trace["risk_label"] = primary_block.get("label")
        trace["overall_ready"] = overall_ready

        if score_data.confidence is not None:
            conf = float(score_data.confidence) * input_quality
        else:
            conf = min(1.0, max(0.0, float(primary_block.get("score", 0.0)))) * input_quality
        if not overall_ready:
            conf *= 0.82
        if any("scoring_features_insufficient" in w for w in warnings):
            conf *= 0.88
        conf = min(1.0, max(0.06, conf))

        risk_payload: dict[str, Any] = {
            "scores": interpreted["scores"],
            "primary": interpreted["primary"],
            "ready": overall_ready,
            "score": float(primary_block["score"]),
            "label": primary_block["label"],
            "explanation": primary_block.get("explanation") or [],
            "risk_available": score_data.risk_available,
            "confidence": score_data.confidence,
            "risk_narrative": score_data.explanation,
        }

        return AgentResult(
            agent_id=AgentRole.SCORING,
            ok=True,
            confidence=conf,
            warnings=warnings,
            payload=risk_payload,
            error_detail=None,
            duration_ms=duration_ms,
            trace=trace,
        )
    except Exception as exc:
        duration_ms = monotonic_ms() - t0
        warnings.append(f"scoring_error:{type(exc).__name__}")
        logger.warning(
            "scoring_agent_error",
            extra={"trace_id": ctx.trace_id, "error": str(exc)},
        )
        return AgentResult(
            agent_id=AgentRole.SCORING,
            ok=False,
            confidence=0.0,
            warnings=warnings,
            payload={},
            error_detail=str(exc),
            duration_ms=duration_ms,
            trace=trace,
        )
