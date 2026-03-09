"""Endpoint tests for pii-service: /health, /v1/redact."""
import sys
from pathlib import Path
import time

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


@pytest.mark.skip(reason="Requires Presidio + spaCy models installed")
def test_redact_presidio_integration():
    text = "Patient John Doe phone 555-123-4567 email john@test.com"
    r = client.post(
        "/v1/redact",
        json={"trace_id": "tid-2", "text": text},
    )
    assert r.status_code == 200
    data = r.json()

    # Verify top-level fields
    assert data["trace_id"] == "tid-2"
    assert "redacted_text" in data
    assert "spans" in data

    redacted_text = data["redacted_text"]
    assert "[PERSON]" in redacted_text
    assert "[PHONE]" in redacted_text
    assert "[EMAIL]" in redacted_text

    # Verify spans contain expected entity types and replacements
    types = {span["type"] for span in data["spans"]}
    replacements = {span["replacement"] for span in data["spans"]}

    assert "PERSON" in types
    assert "PHONE_NUMBER" in types or "[PHONE]" in replacements
    assert "EMAIL_ADDRESS" in types or "[EMAIL]" in replacements

    # At least one span should include a confidence score
    assert any("confidence" in span for span in data["spans"])
