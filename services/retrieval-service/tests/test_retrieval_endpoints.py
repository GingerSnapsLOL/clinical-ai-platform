"""Endpoint tests for retrieval-service: /health, /v1/retrieve."""
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
    assert data["service"] == "retrieval-service"


def test_retrieve_contract():
    r = client.post(
        "/v1/retrieve",
        json={"trace_id": "tid-1", "query": "hypertension treatment"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["trace_id"] == "tid-1"
    assert "passages" in data
    assert len(data["passages"]) >= 1
