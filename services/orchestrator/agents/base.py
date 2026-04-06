"""
Agent contract: named, single-shot ``run`` with structured ``AgentResult``.

Design rules (by convention, documented for implementers):
- No side effects: do not mutate ``input`` or ``context``; do not write globals,
  files, or sockets from agent code (supervisors perform I/O).
- Deterministic: for the same ``input`` and ``context``, return the same result.
- Bounded: no loops in agent logic; supervisors cap how many agents run.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from .types import AgentResult


@runtime_checkable
class Agent(Protocol):
    """
    Structural interface for orchestrator agents.

    ``context`` is usually ``ExecutionContext.model_dump()`` from the caller; values
    must be treated as immutable snapshots.
    """

    @property
    def name(self) -> str:
        """Stable identifier for tracing and metrics."""
        ...

    def run(self, input: dict[str, Any], context: dict[str, Any]) -> AgentResult:
        """Perform one bounded transformation; must not mutate ``input`` or ``context``."""
        ...
