"""Unit tests for shared schemas."""
import sys
from pathlib import Path

_path = Path(__file__).resolve().parent.parent.parent.parent  # repo root
if str(_path) not in sys.path:
    sys.path.insert(0, str(_path))

import pytest
from pydantic import ValidationError

from services.shared.schemas_v1 import (
    AskRequest,
    AskResponse,
    EntityItem,
    ErrorInfo,
    ExtractRequest,
    ExtractResponse,
    FeatureContribution,
    HealthResponse,
    PassageItem,
    PIISpan,
    RedactRequest,
    RedactResponse,
    RetrieveRequest,
    RetrieveResponse,
    RiskBlock,
    ScoreRequest,
    ScoreResponse,
    SourceItem,
)


class TestHealthResponse:
    def test_default_status_and_version(self):
        r = HealthResponse(service="test")
        assert r.status == "ok"
        assert r.service == "test"
        assert r.version == "0.1.0"


class TestAskRequest:
    def test_requires_trace_id(self):
        with pytest.raises(ValidationError):
            AskRequest(mode="strict", note_text="x", question="y")

    def test_valid_minimal(self):
        r = AskRequest(trace_id="a" * 36, note_text="note", question="q")
        assert r.trace_id == "a" * 36
        assert r.mode == "strict"
        assert r.note_text == "note"
        assert r.question == "q"


class TestAskResponse:
    def test_valid_minimal(self):
        r = AskResponse(trace_id="t1", answer="ans")
        assert r.trace_id == "t1"
        assert r.answer == "ans"
        assert r.status == "ok"
        assert r.pii_redacted is True
        assert r.sources == []
        assert r.entities == []
        assert r.risk is None


class TestRedactRequest:
    def test_requires_trace_id_and_text(self):
        with pytest.raises(ValidationError):
            RedactRequest(text="x")  # missing trace_id

    def test_valid(self):
        r = RedactRequest(trace_id="t1", text="hello")
        assert r.trace_id == "t1"
        assert r.text == "hello"


class TestRedactResponse:
    def test_valid_with_spans(self):
        span = PIISpan(type="NAME", start=0, end=5, replacement="[X]")
        r = RedactResponse(trace_id="t1", redacted_text="[X] world", spans=[span])
        assert r.trace_id == "t1"
        assert r.redacted_text == "[X] world"
        assert len(r.spans) == 1
        assert r.spans[0].type == "NAME"


class TestExtractRequest:
    def test_valid(self):
        r = ExtractRequest(trace_id="t1", text="hypertension")
        assert r.trace_id == "t1"
        assert r.text == "hypertension"


class TestExtractResponse:
    def test_valid_with_entities(self):
        e = EntityItem(type="DISEASE", text="hypertension", start=0, end=11)
        r = ExtractResponse(trace_id="t1", entities=[e])
        assert r.trace_id == "t1"
        assert len(r.entities) == 1
        assert r.entities[0].type == "DISEASE"


class TestRetrieveRequest:
    def test_defaults(self):
        r = RetrieveRequest(trace_id="t1", query="q")
        assert r.trace_id == "t1"
        assert r.query == "q"
        assert r.top_k == 50
        assert r.top_n == 8


class TestRetrieveResponse:
    def test_valid_with_passages(self):
        p = PassageItem(source_id="s1", text="t", score=0.8)
        r = RetrieveResponse(trace_id="t1", passages=[p])
        assert r.trace_id == "t1"
        assert len(r.passages) == 1
        assert r.passages[0].source_id == "s1"


class TestScoreRequest:
    def test_valid(self):
        r = ScoreRequest(trace_id="t1", entities=[], structured_features={"x": 1})
        assert r.trace_id == "t1"
        assert r.structured_features["x"] == 1


class TestScoreResponse:
    def test_valid(self):
        fc = FeatureContribution(feature="f1", contribution=0.1)
        r = ScoreResponse(trace_id="t1", score=0.72, label="high", explanation=[fc])
        assert r.trace_id == "t1"
        assert r.score == 0.72
        assert r.label == "high"
        assert len(r.explanation) == 1


class TestErrorInfo:
    def test_minimal(self):
        e = ErrorInfo(code="X", message="msg")
        assert e.code == "X"
        assert e.message == "msg"
        assert e.details is None
