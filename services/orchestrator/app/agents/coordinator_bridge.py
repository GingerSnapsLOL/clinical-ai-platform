"""
Bridge :class:`agents.coordinator_agent.CoordinatorAgent` to async clinical service agents.

Provides a ``dispatch`` closure that maps plan steps to structuring, retrieval, scoring,
``run_evidence_critic``, and synthesis (with relevance gate before LLM).
"""

from __future__ import annotations

from typing import Any

from agents.types import AgentResult as FxAgentResult
from agents.types import CoordinatorOutput, CoordinatorRequest, OrchestratorMode

from app.agents.base import AgentResult as LegacyAgentResult
from app.agents.base import SupervisorContext
from app.agents.clinical_structuring import run_clinical_structuring
from app.agents.evidence_critic import run_evidence_critic
from app.agents.scoring_agent import run_scoring_step
from app.agents.retrieval import run_retrieval
from app.agents.safety_agent import run_safety
from app.agents.synthesis_agent import run_synthesis_answer
from app.relevance import retrieval_meets_relevance_bar
from services.shared.schemas_v1 import EntityItem, FeatureContribution, RiskBlock, SourceItem


def _legacy_to_fx(legacy: LegacyAgentResult) -> FxAgentResult:
    trace_lines: list[str] = []
    for k, v in sorted(legacy.trace.items()):
        trace_lines.append(f"{k}={v}")
    trace_lines.extend([f"warn:{w}" for w in legacy.warnings[:12]])
    return FxAgentResult(
        agent_name=str(legacy.agent_id),
        success=legacy.ok,
        confidence=legacy.confidence,
        payload=dict(legacy.payload),
        warnings=list(legacy.warnings),
        missing_inputs=list(legacy.missing_inputs),
        trace=trace_lines[:100],
    )


def create_clinical_coordinator_dispatch(ctx: SupervisorContext):
    """Return ``dispatch(step, state, exec_ctx)`` for :meth:`CoordinatorAgent.arun`."""

    async def dispatch(step: str, state: dict[str, Any], exec_ctx: dict[str, Any]) -> FxAgentResult:
        del exec_ctx  # boundary uses ``ctx`` for I/O; exec_ctx reserved for future flags
        if step == "structuring":
            return _legacy_to_fx(await run_clinical_structuring(ctx))
        if step == "retrieval":
            redacted = state.get("redacted_text") or ctx.note_text
            ent_raw = state.get("entities") or []
            entities = [EntityItem.model_validate(e) for e in ent_raw]
            return _legacy_to_fx(await run_retrieval(ctx, redacted_text=redacted, entities=entities))
        if step == "scoring":
            ent_raw = state.get("entities") or []
            entities = [EntityItem.model_validate(e) for e in ent_raw]
            sf = state.get("structured_features") or {}
            sig = state.get("signals") or {}
            return _legacy_to_fx(
                await run_scoring_step(ctx, entities, structured_features=sf, signals=sig),
            )
        if step == "critic":
            sources = [SourceItem.model_validate(s) for s in state.get("sources") or []]
            ent_raw = state.get("entities") or []
            entities = [EntityItem.model_validate(e) for e in ent_raw]
            sf = state.get("structured_features") or {}
            sig = state.get("signals") or {}
            scoring_payload = {
                k: state[k]
                for k in ("scores", "primary", "ready", "score", "label", "explanation")
                if k in state
            }
            retrieval_payload = {
                k: state[k]
                for k in ("coverage_score", "top_passages", "queries", "evidence_clusters")
                if k in state
            }
            miss = list(state.get("missing_inputs") or [])
            leg = run_evidence_critic(
                sources=sources,
                entities=entities,
                structured_features=sf,
                signals=sig,
                scoring_payload=scoring_payload,
                retrieval_payload=retrieval_payload,
                missing_inputs=miss,
                retrieval_step_confidence=None,
                scoring_step_confidence=None,
            )
            return _legacy_to_fx(leg)
        if step == "synthesis":
            sources_raw = state.get("sources") or []
            sources = [SourceItem.model_validate(s) for s in sources_raw]
            accept, top_rel, gate_reason = retrieval_meets_relevance_bar(sources)
            if not accept:
                return FxAgentResult(
                    agent_name="synthesis",
                    success=False,
                    confidence=0.0,
                    payload={},
                    warnings=[f"relevance_gate:{gate_reason}"],
                    missing_inputs=[],
                    trace=[f"gate_top_score={top_rel}", f"reason={gate_reason}"],
                )
            ent_raw = state.get("entities") or []
            entities = [EntityItem.model_validate(e) for e in ent_raw]
            risk_block: RiskBlock | None = None
            if (
                "score" in state
                and "label" in state
                and state.get("label") != "insufficient_data"
            ):
                risk_block = RiskBlock(
                    score=float(state["score"]),
                    label=str(state["label"]),
                    explanation=[
                        FeatureContribution.model_validate(x)
                        for x in state.get("explanation") or []
                    ],
                )
            redacted = state.get("redacted_text") or ctx.note_text
            safety_res = run_safety(
                note_text=str(redacted),
                question=str(state.get("question") or ctx.question),
                entities=entities,
            )
            scores_payload = {
                k: state[k]
                for k in ("scores", "primary", "ready", "score", "label", "explanation")
                if k in state
            }
            sf = state.get("structured_features") or {}
            leg = await run_synthesis_answer(
                ctx,
                question=str(state.get("question") or ctx.question),
                entities=entities,
                sources=sources,
                structured_features=sf,
                scores=scores_payload or None,
                safety=safety_res.payload,
                risk=risk_block,
            )
            return _legacy_to_fx(leg)
        raise ValueError(f"unknown coordinator step: {step}")

    return dispatch


async def run_clinical_coordinator(
    ctx: SupervisorContext,
    *,
    note_text: str,
    question: str,
    mode: OrchestratorMode = "strict",
) -> CoordinatorOutput:
    """Run the default clinical plan under :class:`~agents.coordinator_agent.CoordinatorAgent`."""
    from agents.coordinator_agent import CoordinatorAgent

    coord = CoordinatorAgent()
    req = CoordinatorRequest(
        trace_id=ctx.trace_id,
        note_text=note_text,
        question=question,
        mode=mode,
    )
    dispatch_fn = create_clinical_coordinator_dispatch(ctx)
    return await coord.arun(req, dispatch=dispatch_fn)
