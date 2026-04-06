"""
SynthesisAgent: single llm-service /v1/generate call (bounded).

Prompt must be supplied by the supervisor (typically from ``app.prompts.llm_ask``).
"""

from __future__ import annotations

import time
from typing import Any

from services.shared.llm_client import LLMClient
from services.shared.logging_util import get_logger

from app.agents.base import AgentResult, AgentRole, SupervisorContext, monotonic_ms

logger = get_logger(__name__, "orchestrator")

_MAX_REMOTE_CALLS = 1


async def run_synthesis(
    ctx: SupervisorContext,
    *,
    prompt: str,
) -> AgentResult:
    t0 = monotonic_ms()
    warnings: list[str] = []
    trace: dict[str, Any] = {
        "bounded_remote_calls": _MAX_REMOTE_CALLS,
        "agent_role": AgentRole.SYNTHESIS,
        "prompt_chars": len(prompt),
    }

    text = ""
    try:
        llm = LLMClient(base_url=ctx.urls["llm_base"], timeout=ctx.timeout)
        try:
            resp = await llm.generate(
                trace_id=ctx.trace_id,
                prompt=prompt,
                max_tokens=ctx.llm_max_tokens,
                temperature=ctx.llm_temperature,
            )
        finally:
            await llm.aclose()
        text = (resp.text or "").strip()
        if not text:
            warnings.append("llm_empty_completion")
    except Exception as exc:
        warnings.append(f"llm_error:{type(exc).__name__}")
        logger.warning(
            "synthesis_agent_error",
            extra={"trace_id": ctx.trace_id, "error": str(exc)},
        )
        return AgentResult(
            agent_id=AgentRole.SYNTHESIS,
            ok=False,
            confidence=0.0,
            warnings=warnings,
            payload={},
            error_detail=str(exc),
            duration_ms=monotonic_ms() - t0,
            trace=trace,
        )

    duration_ms = monotonic_ms() - t0
    ok = bool(text)
    confidence = 0.82 if ok and "insufficient" not in text.lower() else (0.5 if ok else 0.0)

    return AgentResult(
        agent_id=AgentRole.SYNTHESIS,
        ok=ok,
        confidence=confidence,
        warnings=warnings,
        payload={"answer": text},
        error_detail=None if ok else "empty_answer",
        duration_ms=duration_ms,
        trace=trace,
    )
