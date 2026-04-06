"""Unit tests for :class:`ClinicalStructuringAgent` (deterministic parsing / enrichment)."""

from __future__ import annotations

from services.shared.schemas_v1 import EntityItem

from app.agents.clinical_structuring_agent import ClinicalStructuringAgent


def _item(etype: str, text: str, start: int = 0, end: int | None = None) -> EntityItem:
    if end is None:
        end = len(text)
    return EntityItem(type=etype, text=text, start=start, end=end)


def test_enrich_extracts_bp_hr_spo2_and_age() -> None:
    text = (
        "68 yo with chest pain. BP 150/92 mmHg. HR 104. SpO2 94% on room air. "
        "Pain started 2 days ago, moderate."
    )
    entities = [_item("SYMPTOM", "chest pain", 10, 20)]
    out = ClinicalStructuringAgent.enrich(text, entities, pii_redacted=True)
    sf = out["structured_features"]
    assert sf.get("systolic_bp") == 150.0
    assert sf.get("diastolic_bp") == 92.0
    assert sf.get("heart_rate") == 104.0
    assert sf.get("spo2") == 94.0
    assert sf.get("age") == 68.0
    assert "age" not in out["missing_inputs"]
    assert "duration" not in out["missing_inputs"]


def test_missing_duration_and_severity_when_symptom_only() -> None:
    text = "Patient reports substernal chest discomfort."
    entities = [_item("SYMPTOM", "substernal chest discomfort")]
    out = ClinicalStructuringAgent.enrich(text, entities, pii_redacted=True)
    assert "duration" in out["missing_inputs"]
    assert "severity" in out["missing_inputs"]
    assert out["signals"]["uncertainty"]["severity_unspecified"] is True


def test_hedging_sets_uncertainty_flag() -> None:
    text = "Possibly has hypertension, unclear."
    entities = [_item("DISEASE", "hypertension")]
    out = ClinicalStructuringAgent.enrich(text, entities, pii_redacted=True)
    assert out["signals"]["uncertainty"]["hedging_language"] is True


def test_smoking_and_anticoagulant_structured_flags() -> None:
    text = "Current smoker on warfarin."
    out = ClinicalStructuringAgent.enrich(text, [], pii_redacted=True)
    assert out["structured_features"].get("smoking_current") is True
    assert out["structured_features"].get("on_anticoagulant") is True
