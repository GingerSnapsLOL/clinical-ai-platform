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


def test_default_triage_uses_model_specific_explanation():
    """Implicit primary is triage_severity; ML score + signal-based explanation rows."""
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
    assert 0.0 <= data["score"] <= 1.0
    assert data["label"] in ("low", "medium", "high")
    assert len(data["explanation"]) >= 1
    assert data["explanation"][0]["feature"].startswith("triage_input:")
    assert sum(x["contribution"] for x in data["explanation"]) == pytest.approx(
        data["score"], rel=1e-5, abs=1e-5
    )


def test_triage_acute_symptom_signal():
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
    feats = {e["feature"] for e in data["explanation"]}
    assert any("symptom_chest_pain" in f for f in feats)
    assert data["score"] > 0.0


def test_score_no_triggers_empty_explanation():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-2",
            "entities": [],
            "structured_features": {},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] == "low"
    assert data["explanation"] == []
    assert data["score"] < 0.35


def test_score_structured_numeric_triage():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-3",
            "entities": [],
            "structured_features": {"systolic_bp": 150},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["score"] > 0.0
    feats = {e["feature"] for e in data["explanation"]}
    assert any("bp_systolic_elevated" in f for f in feats)


def test_triage_multi_signal_medium_band():
    """Several comorbidities produce a richer feature vector for the triage model."""
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-4",
            "entities": [
                {"type": "DISEASE", "text": "hypertension", "start": 0, "end": 12},
                {"type": "DISEASE", "text": "diabetes mellitus", "start": 14, "end": 31},
                {"type": "DISEASE", "text": "COPD", "start": 33, "end": 37},
                {"type": "DISEASE", "text": "heart failure", "start": 39, "end": 52},
                {"type": "DISEASE", "text": "coronary artery disease", "start": 54, "end": 77},
                {"type": "DISEASE", "text": "prior stroke", "start": 79, "end": 91},
                {"type": "DISEASE", "text": "chronic kidney disease", "start": 93, "end": 117},
            ],
            "structured_features": {"systolic_bp": 160},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["label"] in ("medium", "high")
    assert data["score"] >= 0.25
    assert sum(x["contribution"] for x in data["explanation"]) == pytest.approx(
        data["score"], rel=1e-5
    )


def test_deduplicate_same_canonical_feature_max_weight():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-5",
            "entities": [
                {
                    "type": "DISEASE",
                    "text": "Type 2 diabetes mellitus with diabetic neuropathy",
                    "start": 0,
                    "end": 50,
                }
            ],
            "structured_features": {},
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert 0.0 <= data["score"] <= 1.0
    assert len(data["explanation"]) == 1
    assert "disease_diabetes" in data["explanation"][0]["feature"]


def test_implicit_request_omits_target_results_block():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-6",
            "entities": [],
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
    assert 0.0 <= tr["triage_severity"]["score"] <= 1.0
    assert tr["cardiovascular_risk"]["score"] == pytest.approx(0.12)
    assert tr["cardiovascular_risk"]["explanation"][0]["feature"] == "disease_hypertension"
    assert data["score"] == pytest.approx(tr["triage_severity"]["score"], rel=1e-4)
    assert data["explanation"] == tr["triage_severity"]["explanation"]


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
    assert data["explanation"] == []
    assert data["target_results"]["stroke_risk"]["ready"] is False
    assert data["target_results"]["triage_severity"]["ready"] is True


def test_cardiovascular_target_explanation_uses_extractor_ids():
    r = client.post(
        "/v1/score",
        json={
            "trace_id": "tid-cv",
            "entities": [
                {"type": "DISEASE", "text": "hypertension", "start": 0, "end": 11}
            ],
            "structured_features": {},
            "targets": ["cardiovascular_risk"],
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["explanation"][0]["feature"] == "disease_hypertension"
    assert data["score"] == pytest.approx(0.12)


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
