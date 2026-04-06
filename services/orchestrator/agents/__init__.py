"""Minimal production-grade agent contracts (no LangChain / LangGraph)."""

from .base import Agent
from .coordinator_agent import DEFAULT_PLAN, CoordinatorAgent, NoOpCriticAgent
from .types import (
    AgentResult,
    CoordinatorOutput,
    CoordinatorRequest,
    ExecutionContext,
    OrchestratorMode,
)

__all__ = [
    "Agent",
    "AgentResult",
    "CoordinatorAgent",
    "CoordinatorOutput",
    "CoordinatorRequest",
    "DEFAULT_PLAN",
    "ExecutionContext",
    "NoOpCriticAgent",
    "OrchestratorMode",
]
