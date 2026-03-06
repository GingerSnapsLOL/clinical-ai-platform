"""Endpoint tests for pii-service: /health, /v1/redact."""
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
    assert data["service"] == "pii-service"


def test_redact_contract():
    r = client.post(
        "/v1/redact",
        json={"trace_id": "tid-1", "text": "John Doe presented on 2024-01-15"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["trace_id"] == "tid-1"
    assert "redacted_text" in data
    assert "spans" in data
    assert len(data["spans"]) >= 1
