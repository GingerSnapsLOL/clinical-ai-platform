"""
SupervisorCoordinator: deterministic, traceable workflow over specialized agents.

Workflow (v1, fixed acyclic graph, ≤ :const:`MAX_WORKFLOW_STEPS_V1` agent hand-offs):
  1. ClinicalStructuringAgent → redacted text + entities
  2. RetrievalAgent + inline Scoring step (both bounded; parallelized here with ``asyncio.gather``)
  3. Relevance gate (deterministic; no LLM)
  4. Evidence critic (may block synthesis)
  5. SafetyAgent + :mod:`app.agents.synthesis_agent` (grounded answer if gate and critic allow)

Scoring is handled by :mod:`app.agents.scoring_agent` (``run_scoring_step``).
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field

from services.shared.logging_util import get_logger
from services.shared.schemas_v1 import (
    EntityItem,
    FeatureContribution,
    RiskBlock,
    SourceItem,
)

from app.agents.base import (
    AGENT_VERSION,
    MAX_WORKFLOW_STEPS_V1,
    AgentResult,
    SupervisorContext,
)
from app.agents.clinical_structuring import run_clinical_structuring
from app.agents.retrieval import run_retrieval
from app.agents.evidence_critic import run_evidence_critic
from app.agents.safety_agent import run_safety
from app.agents.scoring_agent import run_scoring_step
from app.agents.synthesis_agent import run_synthesis_answer
from app.relevance import retrieval_meets_relevance_bar

logger = get_logger(__name__, "orchestrator")


class SupervisorRunResult(BaseModel):
    """Traceable outcome of one supervisor run (all intermediate AgentResult steps retained)."""

    model_config = {"extra": "forbid"}

    trace_id: str
    workflow_version: str = AGENT_VERSION
    ok: bool
    steps: list[AgentResult] = Field(default_factory=list)
    gate_accepted: bool = False
    gate_reason: str = ""
    top_relevance_score: float = 0.0
    final_answer: str = ""
    redacted_text: str = ""
    entities: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    risk: dict[str, Any] = Field(default_factory=dict)
    evidence_critic_valid: bool = True
    evidence_critic_issues: list[str] = Field(default_factory=list)
    evidence_critic_adjustment: float = 1.0


class SupervisorCoordinator:
    """
    Supervisor: owns the global step order and parallelization policy (still no loops).
    """

    async def run(self, ctx: SupervisorContext) -> SupervisorRunResult:
        steps: list[AgentResult] = []
        step_budget = MAX_WORKFLOW_STEPS_V1

        struct = await run_clinical_structuring(ctx)
        steps.append(struct)
        step_budget -= 1

        redacted_text = struct.payload.get("redacted_text") or ctx.note_text
        raw_entities = struct.payload.get("entities") or []
        entities = [EntityItem.model_validate(e) for e in raw_entities]

        if not (struct.payload.get("pii_redacted")):
            return SupervisorRunResult(
                trace_id=ctx.trace_id,
                ok=False,
                steps=steps,
                final_answer="Insufficient data",
                redacted_text=redacted_text,
                entities=[e.model_dump() for e in entities],
            )

        if step_budget < 2:
            raise RuntimeError("supervisor step budget exceeded (misconfigured)")

        retrieval_coro = run_retrieval(ctx, redacted_text=redacted_text, entities=entities)
        struct_features = struct.payload.get("structured_features") or {}
        signals = struct.payload.get("signals") or {}
        scoring_coro = run_scoring_step(
            ctx,
            entities,
            structured_features=struct_features,
            signals=signals,
        )
        retrieval_res, scoring_res = await asyncio.gather(retrieval_coro, scoring_coro)
        steps.extend([retrieval_res, scoring_res])
        step_budget -= 2

        sources_payload = retrieval_res.payload.get("sources") or []
        sources = [SourceItem.model_validate(s) for s in sources_payload]

        risk_block: RiskBlock | None = None
        if scoring_res.ok and scoring_res.payload:
            risk_block = RiskBlock(
                score=float(scoring_res.payload["score"]),
                label=scoring_res.payload["label"],
                explanation=[
                    FeatureContribution.model_validate(x)
                    for x in scoring_res.payload.get("explanation", [])
                ],
            )

        accept, top_rel, gate_reason = retrieval_meets_relevance_bar(sources)
        if not accept:
            return SupervisorRunResult(
                trace_id=ctx.trace_id,
                ok=True,
                steps=steps,
                gate_accepted=False,
                gate_reason=gate_reason,
                top_relevance_score=top_rel,
                final_answer="Insufficient data",
                redacted_text=redacted_text,
                entities=[e.model_dump() for e in entities],
                sources=[s.model_dump() for s in sources],
                risk=scoring_res.payload if scoring_res.ok else {},
            )

        scoring_pl: dict[str, Any] = dict(scoring_res.payload) if scoring_res.ok else {}
        retrieval_pl: dict[str, Any] = dict(retrieval_res.payload or {})
        miss_list: list[str] = list(struct.payload.get("missing_inputs") or [])

        critic_res = run_evidence_critic(
            sources=sources,
            entities=entities,
            structured_features=dict(struct_features),
            signals=dict(signals),
            scoring_payload=scoring_pl,
            retrieval_payload=retrieval_pl,
            missing_inputs=miss_list,
            retrieval_step_confidence=retrieval_res.confidence,
            scoring_step_confidence=scoring_res.confidence,
        )
        steps.append(critic_res)
        step_budget -= 1

        critic_blocks = os.getenv(
            "ORCHESTRATOR_EVIDENCE_CRITIC_BLOCKS_SYNTHESIS",
            "true",
        ).lower() not in ("0", "false", "no", "off")
        if critic_blocks and not critic_res.payload.get("valid"):
            issues = list(critic_res.payload.get("issues") or [])
            return SupervisorRunResult(
                trace_id=ctx.trace_id,
                ok=True,
                steps=steps,
                gate_accepted=False,
                gate_reason="evidence_critic:" + ",".join(issues[:6]),
                top_relevance_score=top_rel,
                final_answer="Insufficient data",
                redacted_text=redacted_text,
                entities=[e.model_dump() for e in entities],
                sources=[s.model_dump() for s in sources],
                risk=scoring_res.payload if scoring_res.ok else {},
                evidence_critic_valid=False,
                evidence_critic_issues=issues,
                evidence_critic_adjustment=float(
                    critic_res.payload.get("confidence_adjustment") or 1.0,
                ),
            )

        if step_budget < 2:
            raise RuntimeError("supervisor step budget exceeded before safety and synthesis")

        safety_res = run_safety(
            note_text=redacted_text,
            question=ctx.question,
            entities=entities,
        )
        steps.append(safety_res)
        step_budget -= 1

        if step_budget < 1:
            raise RuntimeError("supervisor step budget exceeded before synthesis")

        syn = await run_synthesis_answer(
            ctx,
            question=ctx.question,
            entities=entities,
            sources=sources,
            structured_features=dict(struct_features),
            scores=scoring_pl,
            safety=safety_res.payload,
            risk=risk_block,
        )
        steps.append(syn)
        answer = syn.payload.get("answer", "") if syn.ok else ""

        return SupervisorRunResult(
            trace_id=ctx.trace_id,
            ok=syn.ok,
            steps=steps,
            gate_accepted=True,
            gate_reason="",
            top_relevance_score=top_rel,
            final_answer=answer or "Insufficient data",
            redacted_text=redacted_text,
            entities=[e.model_dump() for e in entities],
            sources=[s.model_dump() for s in sources],
            risk=scoring_res.payload if scoring_res.ok else {},
            evidence_critic_valid=True,
            evidence_critic_issues=list(critic_res.payload.get("issues") or []),
            evidence_critic_adjustment=float(
                critic_res.payload.get("confidence_adjustment") or 1.0,
            ),
        )


async def run_supervised_pipeline(
    *,
    trace_id: str,
    question: str,
    note_text: str,
    context: SupervisorContext | None = None,
) -> SupervisorRunResult:
    """Convenience entrypoint using a short-lived ``httpx.AsyncClient``."""
    if context is not None:
        return await SupervisorCoordinator().run(context)

    from services.shared.http_client import create_client, get_timeout

    timeout = get_timeout()
    async with create_client(timeout=timeout) as client:
        ctx = SupervisorContext(
            trace_id=trace_id,
            question=question,
            note_text=note_text,
            client=client,
            timeout=timeout,
        )
        return await SupervisorCoordinator().run(ctx)
