"""
Shared Pydantic v2 schemas for Clinical AI Platform API contracts.

All request/response models include trace_id for end-to-end traceability.
Services must propagate trace_id from gateway through orchestrator to
pii, ner, retrieval, and scoring services.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl

Mode = Literal["strict", "hybrid"]
Status = Literal["ok", "error"]

# Re-export for typed contracts used by all services
__all__ = [
    "AskDiagnostics",
    "AskRequest",
    "AskResponse",
    "Citation",
    "CitationItem",
    "EntityItem",
    "ErrorInfo",
    "ExtractRequest",
    "ExtractResponse",
    "FeatureContribution",
    "HealthResponse",
    "Mode",
    "PassageItem",
    "PIISpan",
    "RedactRequest",
    "RedactResponse",
    "RetrieveRequest",
    "RetrieveResponse",
    "RiskBlock",
    "ScoreRequest",
    "ScoreResponse",
    "Source",
    "SourceItem",
    "TargetScoreResult",
    "Status",
]


class ErrorInfo(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str
    version: str = "0.1.0"

# -------------------------
# Gateway /v1/ask
# -------------------------
class AskRequest(BaseModel):
    trace_id: str = Field(..., description="UUID for request trace; gateway generates if client omits")
    mode: Mode = "strict"
    note_text: str
    question: str
    user_context: Optional[Dict[str, Any]] = None

class Source(BaseModel):
    """Evidence passage / retrieval hit attached to an ask response."""

    source_id: str
    title: Optional[str] = None
    url: Optional[HttpUrl] = None
    snippet: Optional[str] = None
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class Citation(BaseModel):
    """Compact reference to a source used in the answer."""

    source_id: str
    title: Optional[str] = None
    url: Optional[HttpUrl] = None


# Backward-compatible aliases (orchestrator and tests historically used *Item names).
SourceItem = Source
CitationItem = Citation


class AskDiagnostics(BaseModel):
    """Latency and pipeline telemetry for POST /v1/ask (optional on minimal responses)."""

    model_config = ConfigDict(extra="allow")

    total_request_time_ms: Optional[float] = Field(
        default=None,
        description="End-to-end ask duration including downstream services.",
    )
    retrieval_time_ms: Optional[float] = Field(
        default=None,
        description="Retrieval-service (or retrieval agent) duration when applicable.",
    )
    llm_time_ms: Optional[float] = Field(
        default=None,
        description="LLM or synthesis-agent duration when applicable.",
    )
    timings: Dict[str, float] = Field(
        default_factory=dict,
        description="Additional step timings and numeric flags (ms or unitless scores).",
    )
    retrieval_diagnostics: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Retrieval subsystem diagnostics when the pipeline attaches them.",
    )
    planner_decisions: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Planner / tool-routing decisions when hybrid tooling is enabled.",
    )

class EntityItem(BaseModel):
    type: str
    text: str
    start: int
    end: int
    confidence: Optional[float] = None

class FeatureContribution(BaseModel):
    feature: str
    contribution: float

class RiskBlock(BaseModel):
    score: Optional[float] = None
    label: Optional[Literal["low", "medium", "high"]] = None
    explanation: List[FeatureContribution] = []
    risk_available: bool = True
    confidence: Optional[float] = None
    rationale: Optional[str] = None

class AskResponse(BaseModel):
    """Unified response for POST /v1/ask (gateway and orchestrator)."""

    model_config = ConfigDict(extra="ignore")

    status: Status = "ok"
    trace_id: str
    pii_redacted: bool = True
    answer: str = ""
    sources: List[Source] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)
    entities: Optional[List[EntityItem]] = Field(
        default=None,
        description="NER entities when extraction ran; omitted when null.",
    )
    risk_block: Optional[RiskBlock] = Field(
        default=None,
        validation_alias=AliasChoices("risk_block", "risk"),
        description="Primary risk summary from scoring when available.",
    )
    diagnostics: Optional[AskDiagnostics] = Field(
        default=None,
        description="Latency and pipeline telemetry.",
    )
    warnings: List[str] = Field(default_factory=list)
    error: Optional[ErrorInfo] = None

# -------------------------
# PII service /v1/redact
# -------------------------
class PIISpan(BaseModel):
    type: str
    start: int
    end: int
    replacement: str
    confidence: Optional[float] = None

class RedactRequest(BaseModel):
    trace_id: str
    text: str

class RedactResponse(BaseModel):
    status: Status = "ok"
    trace_id: str
    redacted_text: str
    spans: List[PIISpan] = []
    error: Optional[ErrorInfo] = None

# -------------------------
# NER service /v1/extract
# -------------------------
class ExtractRequest(BaseModel):
    trace_id: str
    text: str

class ExtractResponse(BaseModel):
    status: Status = "ok"
    trace_id: str
    entities: List[EntityItem] = []
    error: Optional[ErrorInfo] = None

# -------------------------
# Retrieval service /v1/retrieve
# -------------------------
class RetrieveRequest(BaseModel):
    trace_id: str
    query: str
    top_k: int = 50
    top_n: int = 8
    rerank: bool = True

class PassageItem(BaseModel):
    source_id: str
    text: str
    score: float
    metadata: Optional[Dict[str, Any]] = None

class RetrieveResponse(BaseModel):
    status: Status = "ok"
    trace_id: str
    passages: List[PassageItem] = []
    error: Optional[ErrorInfo] = None

# -------------------------
# Scoring service /v1/score
# -------------------------
class TargetScoreResult(BaseModel):
    """Per-target output from the multi-target scorer."""

    target: str
    score: float = 0.0
    label: Literal["low", "medium", "high"] = "low"
    explanation: List[FeatureContribution] = []
    ready: bool = True
    detail: Optional[str] = Field(
        default=None,
        description="Unset when ready; otherwise reason (e.g. model not trained).",
    )


class ScoreRequest(BaseModel):
    trace_id: str
    entities: List[EntityItem] = []
    structured_features: Dict[str, Any] = {}
    targets: Optional[List[str]] = Field(
        default=None,
        description=(
            "Risk targets to score. When omitted, only the default primary target "
            "(triage_severity) runs. Unknown ids are rejected by the API."
        ),
    )


ScoreLabel = Literal["low", "medium", "high", "insufficient_data"]


class ScoreResponse(BaseModel):
    status: Status = "ok"
    trace_id: str
    score: float
    label: ScoreLabel
    explanation: str = Field(
        default="",
        description="Human-readable scoring rationale (deterministic rules, not ML).",
    )
    contributions: List[FeatureContribution] = Field(
        default_factory=list,
        description="Optional structured feature rows backing the label.",
    )
    target_results: Optional[Dict[str, TargetScoreResult]] = Field(
        default=None,
        description="Populated when the client explicitly requests `targets`.",
    )
    error: Optional[ErrorInfo] = None
    risk_available: bool = Field(
        default=True,
        description="False when no usable input; label should be insufficient_data.",
    )
    confidence: Optional[float] = Field(
        default=None,
        description="Confidence in [0, 1] when risk_available.",
    )