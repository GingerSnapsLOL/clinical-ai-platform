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


def test_ask_llm_integration_fallback_friendly():
    """
    Integration-style test for /v1/ask with llm-based synthesis.

    The test is tolerant of environments where llm-service is unavailable:
    it only asserts on properties that should hold in both LLM and fallback modes.
    """
    r = client.post(
        "/v1/ask",
        json={
            "trace_id": "11111111-2222-3333-4444-555555555555",
            "mode": "strict",
            "note_text": "65-year-old with hypertension and diabetes, on ACE inhibitor and statin.",
            "question": "What is this patient's cardiovascular risk and how should they be monitored?",
        },
    )
    assert r.status_code == 200
    data = r.json()

    # Answer should be non-empty regardless of whether LLM or template fallback is used.
    answer = data.get("answer", "")
    assert isinstance(answer, str)
    assert answer.strip() != ""

    # Citations should be present as a list.
    citations = data.get("citations", [])
    assert isinstance(citations, list)

    # Sources should be a list with at most 3 items (top passages).
    sources = data.get("sources", [])
    assert isinstance(sources, list)
    assert len(sources) <= 3

    # Entities and risk should be present.
    entities = data.get("entities", [])
    assert isinstance(entities, list)
    assert "risk" in data
