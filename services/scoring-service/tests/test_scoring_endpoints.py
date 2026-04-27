"""Endpoint tests for scoring-service: /health, /v1/score (rule-based triage)."""
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


def test_insufficient_empty_input():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-empty",
            "entities": [],
            "structured_features": {},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["risk_available"] is False
    assert data["label"] == "insufficient_data"
    assert data["confidence"] is None
    assert "No clinical text" in data["explanation"]
    assert data["contributions"] == []


def test_chest_pain_high():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-cp",
            "entities": [
                {"type": "SYMPTOM", "text": "Substernal chest pain", "start": 0, "end": 23}
            ],
            "structured_features": {},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["risk_available"] is True
    assert data["label"] == "high"
    assert data["confidence"] == pytest.approx(0.88)
    assert "chest-pain" in data["explanation"].lower() or "chest pain" in data["explanation"].lower()
    assert any("chest_pain" in x["feature"] for x in data["contributions"])


def test_fever_and_cough_medium():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-fc",
            "entities": [
                {"type": "SYMPTOM", "text": "fever", "start": 0, "end": 5},
                {"type": "SYMPTOM", "text": "cough", "start": 7, "end": 12},
            ],
            "structured_features": {},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["risk_available"] is True
    assert data["label"] == "medium"
    assert data["confidence"] == pytest.approx(0.72)
    expl = data["explanation"].lower()
    assert "fever" in expl and "cough" in expl


def test_default_low_when_no_rule_hit():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-low",
            "entities": [{"type": "DISEASE", "text": "hypertension", "start": 0, "end": 11}],
            "structured_features": {},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["risk_available"] is True
    assert data["label"] == "low"
    assert data["confidence"] == pytest.approx(0.62)
    assert "default low" in data["explanation"].lower()


def test_implicit_request_omits_target_results_block():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-6",
            "entities": [{"type": "SYMPTOM", "text": "chest pain", "start": 0, "end": 10}],
            "structured_features": {},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data.get("target_results") is None


def test_explicit_targets_triage_and_cardiovascular():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-7",
            "entities": [
                {"type": "DISEASE", "text": "hypertension", "start": 0, "end": 11}
            ],
            "structured_features": {},
            "targets": ["triage_severity", "cardiovascular_risk"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    tr = data["target_results"]
    assert tr is not None
    assert tr["triage_severity"]["ready"] is True
    assert tr["triage_severity"]["label"] == "low"
    assert tr["cardiovascular_risk"]["ready"] is False
    assert tr["cardiovascular_risk"]["detail"] == "not_computed_rule_based_scorer"
    assert data["score"] == pytest.approx(tr["triage_severity"]["score"], rel=1e-4)
    assert data["label"] == "low"


def test_primary_follows_first_requested_target():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-8",
            "entities": [
                {"type": "DISEASE", "text": "hypertension", "start": 0, "end": 11}
            ],
            "structured_features": {},
            "targets": ["stroke_risk", "triage_severity"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["score"] == 0.0
    assert data["label"] == "low"
    assert data["target_results"]["stroke_risk"]["ready"] is False
    assert data["target_results"]["triage_severity"]["ready"] is True


def test_unknown_target_422():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-9",
            "entities": [],
            "structured_features": {},
            "targets": ["triage_severity", "not_a_real_target"],
        },
    )
    assert r.status_code == 422
