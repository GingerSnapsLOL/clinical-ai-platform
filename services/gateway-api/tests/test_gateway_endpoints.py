"""Endpoint tests for gateway-api: /health, /v1/ask (with mocked orchestrator)."""
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

_path = Path(__file__).resolve().parent.parent
_root = _path.parent
for p in (str(_root), str(_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from fastapi.testclient import TestClient

from app.main import app
from services.shared.schemas_v1 import AskResponse, FeatureContribution, RiskBlock, SourceItem

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "gateway-api"


def test_ask_validation_empty_note():
    r = client.post("/v1/ask", json={"note_text": "", "question": "q?"})
    assert r.status_code == 422


def test_ask_validation_empty_question():
    r = client.post("/v1/ask", json={"note_text": "note", "question": ""})
    assert r.status_code == 422


def test_ask_validation_invalid_trace_id():
    r = client.post(
        "/v1/ask",
        json={"trace_id": "not-a-uuid", "note_text": "note", "question": "q?"},
    )
    assert r.status_code == 422


@patch("app.main.post_typed", new_callable=AsyncMock)
def test_ask_integration_mocked_orchestrator(mock_post_typed):
    """Integration-style test: gateway calls orchestrator, returns AskResponse. Mock orchestrator response."""
    mock_response = AskResponse(
        trace_id="tid-123",
        pii_redacted=True,
        answer="Stubbed answer",
        sources=[SourceItem(source_id="s1", snippet="snippet", score=0.8)],
        entities=[],
        risk=RiskBlock(score=0.72, label="high", explanation=[FeatureContribution(feature="f", contribution=0.1)]),
    )
    mock_post_typed.return_value = (mock_response, None, None)

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
    assert data["risk"]["score"] == 0.72
    assert data["risk"]["label"] == "high"
    mock_post_typed.assert_called_once()
