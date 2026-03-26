"""
Shared Pydantic v2 schemas for Clinical AI Platform API contracts.

All request/response models include trace_id for end-to-end traceability.
Services must propagate trace_id from gateway through orchestrator to
pii, ner, retrieval, and scoring services.
"""
from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, HttpUrl

Mode = Literal["strict", "hybrid"]
Status = Literal["ok", "error"]

# Re-export for typed contracts used by all services
__all__ = [
    "AskRequest",
    "AskResponse",
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
    "SourceItem",
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

class SourceItem(BaseModel):
    source_id: str
    title: Optional[str] = None
    url: Optional[HttpUrl] = None
    snippet: Optional[str] = None
    score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


class CitationItem(BaseModel):
    source_id: str
    title: Optional[str] = None
    url: Optional[HttpUrl] = None

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
    score: float
    label: Literal["low", "medium", "high"]
    explanation: List[FeatureContribution] = []

class AskResponse(BaseModel):
    status: Status = "ok"
    trace_id: str
    pii_redacted: bool = True
    answer: str
    sources: List[SourceItem] = []
    entities: List[EntityItem] = []
    risk: Optional[RiskBlock] = None
    warnings: List[str] = []
    # Latency timings (populated by orchestrator; optional for backward compat)
    total_request_time_ms: Optional[float] = None
    retrieval_time_ms: Optional[float] = None
    llm_time_ms: Optional[float] = None
    timings: Optional[Dict[str, float]] = None
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
class ScoreRequest(BaseModel):
    trace_id: str
    entities: List[EntityItem] = []
    structured_features: Dict[str, Any] = {}

class ScoreResponse(BaseModel):
    status: Status = "ok"
    trace_id: str
    score: float
    label: Literal["low", "medium", "high"]
    explanation: List[FeatureContribution] = []
    error: Optional[ErrorInfo] = None