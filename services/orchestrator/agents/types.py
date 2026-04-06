"""
Shared types for the orchestrator agent framework.

Agents are contract-only: implementations must avoid side effects, keep ``run`` pure
for a given (input, context), and must not loop (bounded steps are enforced by callers).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

OrchestratorMode = Literal["strict", "hybrid"]


class AgentResult(BaseModel):
    """Uniform return type for every agent step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    agent_name: str = Field(min_length=1)
    success: bool
    confidence: float = Field(ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)


class CoordinatorRequest(BaseModel):
    """Inbound ask-style payload for the coordinator supervisor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    note_text: str
    question: str
    mode: OrchestratorMode = "strict"
    trace_id: str = Field(min_length=1)


class CoordinatorOutput(BaseModel):
    """Coordinator response: answer text, per-step trace, rolled-up confidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    final_answer: str
    agent_trace: list[dict[str, Any]]
    confidence: float = Field(ge=0.0, le=1.0)


class ExecutionContext(BaseModel):
    """
    Cross-cutting execution metadata passed into agents (typically as ``context`` dict).

    Serialise with ``model_dump()`` when calling ``Agent.run``; agents must treat
    ``context`` as read-only.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    trace_id: str = Field(min_length=1)
    mode: OrchestratorMode = "strict"
    debug: dict[str, bool] = Field(
        default_factory=dict,
        description="Named debug switches (e.g. log_prompts, return_raw_chunks).",
    )
