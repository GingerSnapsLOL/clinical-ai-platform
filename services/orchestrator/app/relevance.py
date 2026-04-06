"""Retrieval relevance gate shared by ``/v1/ask`` and the supervisor coordinator."""

from __future__ import annotations

import os

from services.shared.schemas_v1 import SourceItem


def retrieval_relevance_gate_enabled() -> bool:
    raw = os.getenv("ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE", "true").strip().lower()
    return raw not in ("0", "false", "no", "off")


def retrieval_meets_relevance_bar(sources: list[SourceItem]) -> tuple[bool, float, str]:
    if not retrieval_relevance_gate_enabled():
        top = max((float(s.score) for s in sources if s.score is not None), default=0.0)
        return True, top, ""

    if not sources:
        return False, 0.0, "no_passages"

    min_score = float(os.getenv("ORCHESTRATOR_RETRIEVAL_MIN_TOP_SCORE", "1.0"))
    min_snippet = int(os.getenv("ORCHESTRATOR_RETRIEVAL_MIN_TOP_SNIPPET_CHARS", "24"))

    top = max(
        sources,
        key=lambda s: float(s.score) if s.score is not None else float("-inf"),
    )
    top_score_l = float(top.score) if top.score is not None else float("-inf")
    top_score = top_score_l if top_score_l != float("-inf") else 0.0

    if top_score_l < min_score:
        return False, top_score, "below_min_score"

    snippet = (top.snippet or "").strip()
    if len(snippet) < min_snippet:
        return False, top_score, "weak_snippet"

    return True, top_score, ""
