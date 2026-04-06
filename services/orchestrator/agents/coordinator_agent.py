"""
CoordinatorAgent: supervisor that runs a fixed execution plan with bounded steps.

Decisions after each step: continue, request clarification (stop), or finalize (after synthesis).

The async entrypoint ``arun`` supports I/O-heavy agents via an injected ``dispatch``;
pure sync agents can be registered on ``__init__`` and are run via ``asyncio.to_thread``.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, Literal

from .base import Agent
from .types import AgentResult, CoordinatorOutput, CoordinatorRequest, ExecutionContext

DEFAULT_PLAN: tuple[str, ...] = (
    "structuring",
    "retrieval",
    "scoring",
    "critic",
    "synthesis",
)

StepDecision = Literal["continue", "clarification", "finalize"]

DispatchFn = Callable[[str, dict[str, Any], dict[str, Any]], Awaitable[AgentResult]]

_RESERVED_STATE_KEYS = frozenset({"note_text", "question", "trace_id", "mode"})


class NoOpCriticAgent:
    """Placeholder critic when no evidence review is wired; approves and adds a trace line."""

    @property
    def name(self) -> str:
        return "evidence_critic"

    def run(self, input: dict[str, Any], context: dict[str, Any]) -> AgentResult:
        return AgentResult(
            agent_name=self.name,
            success=True,
            confidence=1.0,
            payload={"approved": True, "notes": []},
            warnings=[],
            missing_inputs=[],
            trace=["noop_critic_approved"],
        )


def _merge_payload_into_state(state: dict[str, Any], payload: dict[str, Any]) -> None:
    for key, value in payload.items():
        if key in _RESERVED_STATE_KEYS:
            continue
        state[key] = value


def _decide_after_step(
    step: str,
    result: AgentResult,
    *,
    mode: str,
) -> StepDecision:
    if result.missing_inputs:
        return "clarification"

    if step == "synthesis":
        return "finalize" if result.success else "clarification"

    if step == "structuring" and not result.success:
        return "clarification"

    if step == "structuring":
        return "continue"

    strict = mode == "strict"

    if step == "retrieval" and not result.success and strict:
        return "clarification"
    if step == "retrieval":
        return "continue"

    if step == "scoring" and not result.success and strict:
        return "clarification"
    if step == "scoring":
        return "continue"

    if step == "critic":
        payload = result.payload
        if payload.get("valid") is False and strict:
            return "clarification"
        if payload.get("approved") is False and strict:
            return "clarification"
        return "continue"

    return "continue"


def _clarification_answer(result: AgentResult, state: dict[str, Any]) -> str:
    parts: list[str] = []
    if result.missing_inputs:
        parts.append("Please provide: " + ", ".join(result.missing_inputs))
    if result.warnings:
        parts.append("Notes: " + "; ".join(result.warnings[:5]))
    if not state.get("pii_redacted", True):
        parts.append("Note text could not be safety-processed; try a shorter note or different wording.")
    if not parts:
        parts.append("More information is needed to answer safely.")
    return " ".join(parts)


def _aggregate_confidence(trace_records: list[dict[str, Any]]) -> float:
    confs = [float(t["confidence"]) for t in trace_records if t.get("success")]
    if not confs:
        return 0.0
    prod = math.prod(confs)
    return float(prod ** (1.0 / len(confs)))


class CoordinatorAgent:
    """
    Supervisor: builds a fixed plan, runs agents in order, propagates state, emits trace.

    Either pass ``agents`` (sync :class:`Agent` per plan step) *or* use ``arun(..., dispatch=…)``
    for async/service-backed steps (clinical stack).
    """

    def __init__(
        self,
        *,
        agents: Mapping[str, Agent] | None = None,
        plan: Sequence[str] | None = None,
    ) -> None:
        self._agents: dict[str, Agent] = dict(agents) if agents else {}
        self._plan = tuple(plan) if plan is not None else DEFAULT_PLAN
        if self._agents:
            missing = [s for s in self._plan if s not in self._agents]
            if missing:
                raise ValueError(
                    f"CoordinatorAgent: plan references steps without agents: {missing}",
                )

    async def _default_dispatch(self, step: str, state: dict[str, Any], ctx: dict[str, Any]) -> AgentResult:
        agent = self._agents[step]
        return await asyncio.to_thread(agent.run, dict(state), dict(ctx))

    async def arun(
        self,
        request: CoordinatorRequest,
        *,
        dispatch: DispatchFn | None = None,
    ) -> CoordinatorOutput:
        invoke: DispatchFn
        if dispatch is not None:
            invoke = dispatch
        elif self._agents:
            invoke = self._default_dispatch
        else:
            raise ValueError("CoordinatorAgent.arun requires `dispatch` or pre-registered `agents`.")

        exec_model = ExecutionContext(
            trace_id=request.trace_id,
            mode=request.mode,
        )
        exec_ctx = exec_model.model_dump()

        state: dict[str, Any] = {
            "note_text": request.note_text,
            "question": request.question,
            "trace_id": request.trace_id,
            "mode": request.mode,
        }

        agent_trace: list[dict[str, Any]] = []
        final_answer = ""
        stopped_by: StepDecision = "continue"

        for step in self._plan:
            result = await invoke(step, state, exec_ctx)
            _merge_payload_into_state(state, result.payload)

            decision = _decide_after_step(step, result, mode=request.mode)
            agent_trace.append(
                {
                    "step": step,
                    "agent_name": result.agent_name,
                    "success": result.success,
                    "confidence": result.confidence,
                    "decision": decision,
                    "trace": list(result.trace),
                    "warnings": list(result.warnings),
                    "missing_inputs": list(result.missing_inputs),
                },
            )
            stopped_by = decision

            if decision == "clarification":
                final_answer = _clarification_answer(result, state)
                break
            if decision == "finalize":
                final_answer = (result.payload.get("answer") or "").strip() or "Insufficient data"
                break
            # continue
        else:
            # plan finished without synthesis finalize (should not happen with default plan)
            final_answer = (state.get("answer") or "").strip() or "Insufficient data"
            stopped_by = "finalize"

        conf = _aggregate_confidence(agent_trace)
        if stopped_by == "clarification":
            conf = min(conf, 0.55) * 0.85

        return CoordinatorOutput(
            final_answer=final_answer,
            agent_trace=agent_trace,
            confidence=min(1.0, max(0.0, conf)),
        )

    def run_sync(self, request: CoordinatorRequest, *, dispatch: DispatchFn | None = None) -> CoordinatorOutput:
        """Run the plan in a fresh event loop (scripts / tests only; not inside running asyncio)."""
        return asyncio.run(self.arun(request, dispatch=dispatch))
