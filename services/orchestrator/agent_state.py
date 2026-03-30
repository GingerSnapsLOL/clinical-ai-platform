"""Typed immutable state for the medical assistant agent flow."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from services.shared.schemas_v1 import EntityItem, RiskBlock, SourceItem


@dataclass(frozen=True, slots=True)
class AgentState:
    """
    Carries ask-flow inputs and artifacts through a node chain.
    Nodes return a new instance via ``evolve`` or ``mark_stop``.
    """

    trace_id: str
    question: str
    entities: list[EntityItem] = field(default_factory=list)
    sources: list[SourceItem] = field(default_factory=list)
    risk: RiskBlock | None = None
    warnings: list[str] = field(default_factory=list)
    selected_sources: list[SourceItem] = field(default_factory=list)
    draft_answer: str | None = None
    draft_used_source_ids: list[str] = field(default_factory=list)
    final_answer: str | None = None
    needs_retry: bool = False
    stop_reason: str | None = None

    def evolve(self, **changes: Any) -> AgentState:
        """Shallow merge: supply only fields that change."""
        return replace(self, **changes)

    def top_k_sources(self, k: int) -> list[SourceItem]:
        """First ``k`` retrieval sources (non-mutating slice)."""
        if k <= 0:
            return []
        return list(self.sources[:k])

    def has_sources(self) -> bool:
        return bool(self.sources)

    def mark_stop(self, reason: str) -> AgentState:
        """End the run before remaining nodes; ``stop_reason`` is non-empty."""
        return replace(self, stop_reason=reason)
