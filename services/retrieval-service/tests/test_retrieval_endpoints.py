"""Endpoint tests for retrieval-service: /health, /v1/retrieve, /v1/ingest."""
import json
import sys
from pathlib import Path

_path = Path(__file__).resolve().parent.parent
_root = _path.parent
_repo_root = _root.parent  # repo root (services/retrieval-service -> services -> repo)
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
    assert isinstance(data["passages"], list)


def test_ingest_and_retrieve_unique_passages():
    """Integration: ingest demo docs, retrieve, verify passages are unique by (source_id, text)."""
    docs_path = _repo_root / "examples" / "clinical_docs.json"
    if not docs_path.exists():
        pytest.skip("examples/clinical_docs.json not found")
    payload = json.loads(docs_path.read_text(encoding="utf-8"))

    r_ingest = client.post("/v1/ingest", json=payload)
    assert r_ingest.status_code == 200
    ingest_data = r_ingest.json()
    assert "chunks_inserted" in ingest_data
    assert ingest_data["chunks_inserted"] >= 1

    r_retrieve = client.post(
        "/v1/retrieve",
        json={
            "trace_id": "integ-1",
            "query": "What are the first-line treatments for hypertension?",
            "top_k": 20,
        },
    )
    assert r_retrieve.status_code == 200
    data = r_retrieve.json()
    passages = data.get("passages", [])

    # Every passage must be unique by (source_id, text)
    pairs = [(p["source_id"], p["text"]) for p in passages]
    assert len(pairs) == len(set(pairs)), "retrieve must return deduplicated passages"
