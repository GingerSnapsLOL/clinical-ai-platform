"""Endpoint tests for scoring-service: /health, /v1/score."""
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
    assert data["service"] == "scoring-service"


def test_score_contract():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-1",
            "entities": [{"type": "DISEASE", "text": "hypertension", "start": 0, "end": 11}],
            "structured_features": {},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["trace_id"] == "tid-1"
    assert "score" in data
    assert "label" in data
    assert "explanation" in data
