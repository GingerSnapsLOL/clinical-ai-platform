"""
Supervised multi-agent pipeline for POST /v1/ask.

Coordinator runs a fixed acyclic sequence with a hard step budget (see MAX_WORKFLOW_STEPS_V1).
Execution is deterministic for a given request and environment: no agent-initiated loops.
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx

from app.agents.base import (
    AGENT_VERSION,
    MAX_WORKFLOW_STEPS_V1,
    AgentRole,
    RetrievalCachePort,
    SupervisorContext,
)
from app.agents.coordinator import SupervisorCoordinator, SupervisorRunResult
from services.shared.logging_util import get_logger
from services.shared.schemas_v1 import (
    AskRequest,
    AskResponse,
    CitationItem,
    EntityItem,
    ErrorInfo,
    FeatureContribution,
    RiskBlock,
    SourceItem,
)

logger = get_logger(__name__, "orchestrator")

_SUPERVISOR_PIPELINE_ENV = "ORCHESTRATOR_SUPERVISOR_PIPELINE"
_AGENT_DEBUG_ENV = "ORCHESTRATOR_AGENT_DEBUG"


def supervisor_pipeline_enabled() -> bool:
    raw = os.getenv(_SUPERVISOR_PIPELINE_ENV, "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


def agent_pipeline_debug(request: AskRequest) -> bool:
    if os.getenv(_AGENT_DEBUG_ENV, "").strip().lower() in ("1", "true", "yes", "on"):
        return True
    uc = request.user_context or {}
    return bool(uc.get("debug") or uc.get("agent_debug"))


def _risk_from_run(run: SupervisorRunResult) -> RiskBlock | None:
    d = run.risk
    if not d or "score" not in d or "label" not in d:
        return None
    expl = d.get("explanation") or []
    return RiskBlock(
        score=float(d["score"]),
        label=d["label"],
        explanation=[FeatureContribution.model_validate(x) for x in expl],
    )


def _citations_from_sources(sources: list[SourceItem]) -> list[CitationItem]:
    citations: list[CitationItem] = []
    seen: set[str] = set()
    for s in sources:
        if s.source_id in seen:
            continue
        seen.add(s.source_id)
        title = s.title
        url = s.url
        if s.metadata:
            title = title or s.metadata.get("title")
            url = url or s.metadata.get("url")
        citations.append(CitationItem(source_id=s.source_id, title=title, url=url))
    return citations


def _pii_redacted_from_steps(run: SupervisorRunResult) -> bool:
    struct = next(
        (s for s in run.steps if s.agent_id == AgentRole.CLINICAL_STRUCTURING.value),
        None,
    )
    if struct is None:
        return False
    return bool(struct.payload.get("pii_redacted"))


def _collect_warnings(run: SupervisorRunResult) -> list[str]:
    out: list[str] = []
    for step in run.steps:
        for w in step.warnings:
            if w not in out:
                out.append(w)
    if run.gate_reason:
        gr = run.gate_reason
        if gr not in out:
            out.append(f"gate:{gr}")
    if run.evidence_critic_issues:
        line = "evidence_critic:" + ",".join(run.evidence_critic_issues[:8])
        if line not in out:
            out.append(line)
    return out


def _log_steps(trace_id: str, run: SupervisorRunResult, *, debug: bool) -> None:
    for i, step in enumerate(run.steps):
        extra: dict[str, Any] = {
            "trace_id": trace_id,
            "step_index": i,
            "agent_id": step.agent_id,
            "ok": step.ok,
            "duration_ms": round(step.duration_ms, 3),
            "confidence": step.confidence,
            "warnings_count": len(step.warnings),
            "workflow_version": AGENT_VERSION,
            "max_workflow_steps": MAX_WORKFLOW_STEPS_V1,
        }
        if debug:
            preview = str(step.trace)[:800] if step.trace else ""
            extra["trace_preview"] = preview
            if step.payload:
                keys = list(step.payload.keys())[:20]
                extra["payload_keys"] = keys
        logger.info("agent_pipeline_step", extra=extra)


def _step_duration(run: SupervisorRunResult, agent_role: AgentRole) -> float | None:
    for s in run.steps:
        if s.agent_id == agent_role.value:
            return float(s.duration_ms)
    return None


async def run_supervised_ask(
    request: AskRequest,
    client: httpx.AsyncClient,
    timeout: float,
    *,
    retrieval_cache: RetrievalCachePort | None = None,
    debug: bool = False,
    t_request_start: float | None = None,
) -> AskResponse:
    trace_id = request.trace_id
    t0 = t_request_start if t_request_start is not None else time.perf_counter()

    logger.info(
        "agent_pipeline_start",
        extra={
            "trace_id": trace_id,
            "workflow_version": AGENT_VERSION,
            "max_workflow_steps": MAX_WORKFLOW_STEPS_V1,
            "debug": debug,
            "request_mode": request.mode,
            "deterministic": True,
        },
    )

    ctx = SupervisorContext(
        trace_id=trace_id,
        question=request.question,
        note_text=request.note_text,
        client=client,
        timeout=timeout,
        retrieval_cache=retrieval_cache,
    )

    try:
        run = await SupervisorCoordinator().run(ctx)
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception(
            "agent_pipeline_error",
            extra={"trace_id": trace_id, "error": str(exc)},
        )
        total_ms = (time.perf_counter() - t0) * 1000.0
        return AskResponse(
            status="error",
            trace_id=trace_id,
            pii_redacted=False,
            answer="Insufficient data",
            entities=[],
            sources=[],
            risk=None,
            warnings=[f"supervised_pipeline:{exc!s}"],
            total_request_time_ms=total_ms,
            error=ErrorInfo(code="supervised_pipeline", message=str(exc)),
            timings={"total_request_time_ms": total_ms},
        )

    _log_steps(trace_id, run, debug=debug)

    entities = [EntityItem.model_validate(e) for e in run.entities]
    sources = [SourceItem.model_validate(s) for s in run.sources]
    risk = _risk_from_run(run)
    warnings = _collect_warnings(run)
    pii_redacted = _pii_redacted_from_steps(run)
    citations = _citations_from_sources(sources)

    retrieval_ms = _step_duration(run, AgentRole.RETRIEVAL)
    llm_ms = _step_duration(run, AgentRole.SYNTHESIS)

    total_ms = (time.perf_counter() - t0) * 1000.0
    timings: dict[str, float] = {
        "total_request_time_ms": total_ms,
        "supervised_pipeline_flag": 1.0,
    }
    if retrieval_ms is not None:
        timings["retrieval_agent_duration_ms"] = retrieval_ms
    if llm_ms is not None:
        timings["synthesis_agent_duration_ms"] = llm_ms
    for i, step in enumerate(run.steps):
        timings[f"agent_step_{i}_{step.agent_id}_ms"] = float(step.duration_ms)

    logger.info(
        "agent_pipeline_summary",
        extra={
            "trace_id": trace_id,
            "total_duration_ms": total_ms,
            "step_count": len(run.steps),
            "gate_accepted": run.gate_accepted,
            "supervisor_ok": run.ok,
            "workflow_version": AGENT_VERSION,
        },
    )

    return AskResponse(
        trace_id=trace_id,
        pii_redacted=pii_redacted,
        answer=run.final_answer or "Insufficient data",
        entities=entities,
        sources=sources,
        risk=risk,
        citations=citations,
        warnings=warnings,
        total_request_time_ms=total_ms,
        retrieval_time_ms=retrieval_ms,
        llm_time_ms=llm_ms,
        timings=timings,
    )
