"""
Base contracts for specialized agents: structured results, confidence, warnings, bounded work.

No autonomous loops: each agent performs a fixed maximum of external calls per run.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field


AGENT_VERSION = "2026.03.1"
MAX_WORKFLOW_STEPS_V1 = 6  # hard cap on supervisor-issued agent transitions (no cycles)


class AgentRole(StrEnum):
    """Registered agent identities (supervisor dispatches by role)."""

    COORDINATOR = "coordinator"
    CLINICAL_STRUCTURING = "clinical_structuring"
    RETRIEVAL = "retrieval"
    SCORING = "scoring"
    EVIDENCE_CRITIC = "evidence_critic"
    CLARIFICATION = "clarification"
    SAFETY = "safety"
    SYNTHESIS = "synthesis"


class AgentResult(BaseModel):
    """
    Uniform envelope for every agent step.

    ``payload`` is a JSON-serializable dict (often a nested Pydantic model_dump).
    """

    model_config = {"extra": "forbid"}

    agent_id: str
    ok: bool
    confidence: float = Field(ge=0.0, le=1.0, description="Heuristic self-assessment of output quality.")
    warnings: list[str] = Field(default_factory=list)
    missing_inputs: list[str] = Field(
        default_factory=list,
        description="Critical history elements absent from the note (e.g. age, duration, severity).",
    )
    payload: dict[str, Any] = Field(default_factory=dict)
    error_detail: str | None = None
    duration_ms: float = 0.0
    trace: dict[str, Any] = Field(
        default_factory=dict,
        description="Bounded metadata: service paths, counts, cache flags — no raw PHI.",
    )

    @classmethod
    def failure(
        cls,
        agent_id: str,
        message: str,
        *,
        duration_ms: float = 0.0,
        trace: dict[str, Any] | None = None,
    ) -> AgentResult:
        return cls(
            agent_id=agent_id,
            ok=False,
            confidence=0.0,
            warnings=[],
            missing_inputs=[],
            payload={},
            error_detail=message,
            duration_ms=duration_ms,
            trace=trace or {},
        )


@runtime_checkable
class RetrievalCachePort(Protocol):
    """Optional Redis-backed cache for retrieval responses (same contract as orchestrator cache)."""

    async def get_json(self, trace_id: str, key: str) -> dict[str, Any] | None:
        ...

    async def set_json(self, trace_id: str, key: str, value: dict[str, Any], ttl_sec: int) -> None:
        ...


def default_service_urls() -> dict[str, str]:
    return {
        "pii": os.getenv("PII_SERVICE_URL", "http://pii-service:8020").rstrip("/") + "/v1/redact",
        "ner": os.getenv("NER_SERVICE_URL", "http://ner-service:8030").rstrip("/") + "/v1/extract",
        "retrieval": os.getenv("RETRIEVAL_SERVICE_URL", "http://retrieval-service:8040").rstrip("/")
        + "/v1/retrieve",
        "scoring": os.getenv("SCORING_SERVICE_URL", "http://scoring-service:8050").rstrip("/")
        + "/v1/score",
        "llm_base": os.getenv("LLM_BASE_URL", "http://llm-service:8060").rstrip("/"),
    }


@dataclass
class SupervisorContext:
    """
    Dependencies for a single supervised run (explicit, injectable for tests).

    Bounded: ``client`` is shared; each agent makes at most its documented number of HTTP calls.
    """

    trace_id: str
    question: str
    note_text: str
    client: httpx.AsyncClient
    timeout: float
    urls: dict[str, str] = field(default_factory=default_service_urls)
    retrieval_cache: RetrievalCachePort | None = None
    retrieval_cache_ttl_sec: int = field(
        default_factory=lambda: int(os.getenv("ORCHESTRATOR_RETRIEVAL_CACHE_TTL_SEC", "300"))
    )
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2


def monotonic_ms() -> float:
    return time.perf_counter() * 1000.0
