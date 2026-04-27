"""Endpoint tests for gateway-api: /health, /v1/ask (with mocked orchestrator)."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

_path = Path(__file__).resolve().parent.parent
_root = _path.parent
for p in (str(_root), str(_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from fastapi.testclient import TestClient

from app.main import app
from services.shared.schemas_v1 import AskResponse, AskDiagnostics, FeatureContribution, RiskBlock, SourceItem

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "gateway-api"


def test_ask_validation_empty_note():
    r = client.post("/v1/ask", json={"note_text": "", "question": "q?"})
    assert r.status_code == 400
    assert "detail" in r.json()


def test_ask_validation_empty_question():
    r = client.post("/v1/ask", json={"note_text": "note", "question": ""})
    assert r.status_code == 400


def test_ask_validation_invalid_trace_id():
    r = client.post(
        "/v1/ask",
        json={"trace_id": "not-a-uuid", "note_text": "note", "question": "q?"},
    )
    assert r.status_code == 400


@patch("app.main.post_json", new_callable=AsyncMock)
def test_ask_simple_input_mocked_orchestrator(mock_post_json):
    """
    Minimal integration confidence: /v1/ask with valid body, orchestrator mocked,
    assert core response shape.
    """
    tid = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    mock_response = AskResponse(
        trace_id=tid,
        pii_redacted=True,
        answer="Monitor blood pressure and review medications.",
        sources=[SourceItem(source_id="doc-1", snippet="Evidence text.", score=0.88)],
        citations=[],
        entities=[],
    )
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    mock_post_json.return_value = (mock_response.model_dump(mode="json"), ok_resp, None)

    r = client.post(
        "/v1/ask",
        json={
            "mode": "strict",
            "note_text": "65-year-old with hypertension on lisinopril.",
            "question": "What should we monitor next visit?",
        },
    )

    assert r.status_code == 200
    data = r.json()
    assert isinstance(data.get("answer"), str) and data["answer"].strip() != ""
    assert isinstance(data.get("sources"), list)
    assert data.get("trace_id")
    UUID(data["trace_id"])
    mock_post_json.assert_called_once()


def test_ask_invalid_request_missing_required_field_returns_400():
    """Invalid client request: missing ``question`` → 400 with validation detail."""
    r = client.post(
        "/v1/ask",
        json={"note_text": "Patient note only.", "mode": "strict"},
    )
    assert r.status_code == 400
    body = r.json()
    assert isinstance(body.get("detail"), list)
    assert any(
        isinstance(err, dict) and err.get("loc") and "question" in err["loc"]
        for err in body["detail"]
    )


@patch("app.main.post_json", new_callable=AsyncMock)
def test_ask_integration_mocked_orchestrator(mock_post_json):
    """Integration-style test: gateway calls orchestrator, returns AskResponse. Mock orchestrator response."""
    mock_response = AskResponse(
        trace_id="tid-123",
        pii_redacted=True,
        answer="Stubbed answer",
        sources=[SourceItem(source_id="s1", snippet="snippet", score=0.8)],
        entities=[],
        risk_block=RiskBlock(score=0.72, label="high", explanation=[FeatureContribution(feature="f", contribution=0.1)]),
        diagnostics=AskDiagnostics(total_request_time_ms=5.0, timings={"total_request_time_ms": 5.0}),
    )
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    mock_post_json.return_value = (mock_response.model_dump(mode="json"), ok_resp, None)

    r = client.post(
        "/v1/ask",
        json={"note_text": "55yo with hypertension", "question": "Risk?"},
    )

    assert r.status_code == 200
    data = r.json()
    assert data["trace_id"] == "tid-123"
    assert data["answer"] == "Stubbed answer"
    assert data["pii_redacted"] is True
    assert len(data["sources"]) == 1
    assert data["risk_block"]["score"] == 0.72
    assert data["risk_block"]["label"] == "high"
    mock_post_json.assert_called_once()


@patch("app.main.post_json", new_callable=AsyncMock)
def test_ask_returns_500_when_orchestrator_body_invalid_for_schema(mock_post_json, monkeypatch):
    monkeypatch.delenv("GATEWAY_EXPOSE_ASK_VALIDATION_DETAILS", raising=False)
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    mock_post_json.return_value = ({"answer": "orphan field only"}, ok_resp, None)

    r = client.post(
        "/v1/ask",
        json={"note_text": "55yo with hypertension", "question": "Risk?"},
    )
    assert r.status_code == 500
    data = r.json()
    assert data["status"] == "error"
    assert data["error"]["code"] == "GATEWAY_RESPONSE_VALIDATION_ERROR"
    assert data["error"].get("details") is None


@patch("app.main.post_json", new_callable=AsyncMock)
def test_ask_includes_validation_errors_when_expose_flag(mock_post_json, monkeypatch):
    monkeypatch.setenv("GATEWAY_EXPOSE_ASK_VALIDATION_DETAILS", "true")
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    mock_post_json.return_value = ({"answer": "missing trace_id"}, ok_resp, None)

    r = client.post(
        "/v1/ask",
        json={"note_text": "55yo with hypertension", "question": "Risk?"},
    )
    assert r.status_code == 500
    data = r.json()
    assert data["error"]["code"] == "GATEWAY_RESPONSE_VALIDATION_ERROR"
    assert data["error"]["details"] is not None
    assert "validation_errors" in data["error"]["details"]


@patch("app.main.post_json", new_callable=AsyncMock)
def test_ask_returns_500_on_json_decode_failure(mock_post_json, monkeypatch):
    monkeypatch.setenv("GATEWAY_EXPOSE_ASK_VALIDATION_DETAILS", "true")
    ok_resp = MagicMock()
    ok_resp.status_code = 200
    mock_post_json.return_value = (None, ok_resp, ValueError("simulated json failure"))

    r = client.post(
        "/v1/ask",
        json={"note_text": "55yo with hypertension", "question": "Risk?"},
    )
    assert r.status_code == 500
    data = r.json()
    assert data["error"]["code"] == "GATEWAY_ORCHESTRATOR_JSON_INVALID"
    assert data["error"]["details"] is not None
    assert data["error"]["details"].get("reason")
