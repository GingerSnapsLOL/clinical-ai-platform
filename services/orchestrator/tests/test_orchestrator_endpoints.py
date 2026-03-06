"""Endpoint tests for orchestrator: /health, /v1/ask."""
import sys
from pathlib import Path

_path = Path(__file__).resolve().parent.parent
_root = _path.parent
for p in (str(_root), str(_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "orchestrator"


def test_ask_contract():
    """Contract endpoint: accepts AskRequest, returns AskResponse with required fields."""
    r = client.post(
        "/v1/ask",
        json={
            "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "mode": "strict",
            "note_text": "Patient with hypertension",
            "question": "What is the risk?",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "trace_id" in data
    assert data["trace_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    assert "answer" in data
    assert "entities" in data
    assert "sources" in data
    assert "risk" in data
    assert "pii_redacted" in data
