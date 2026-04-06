"""Tests for :func:`run_safety`."""

from __future__ import annotations

from services.shared.schemas_v1 import EntityItem

from app.agents.safety_agent import run_safety


def test_emergency_chest_pain_and_dyspnea() -> None:
    res = run_safety(
        note_text="Substernal chest pain with shortness of breath for 1 hour.",
        question="Could this be serious?",
    )
    assert res.payload["safety_level"] == "emergency"
    assert "escalate_emergency_care" in res.payload["actions"]
    assert "message_prefix" in res.payload


def test_emergency_stroke_language() -> None:
    res = run_safety(note_text="Sudden facial droop and slurred speech.")
    assert res.payload["safety_level"] == "emergency"


def test_emergency_sepsis_keyword() -> None:
    res = run_safety(note_text="Patient meets sepsis criteria with hypotension.")
    assert res.payload["safety_level"] == "emergency"


def test_warning_chest_pain_only() -> None:
    res = run_safety(note_text="Episodes of chest pressure without dyspnea.")
    assert res.payload["safety_level"] == "warning"
    assert "avoid_definitive_diagnosis" in res.payload["actions"]


def test_normal_benign() -> None:
    res = run_safety(note_text="Routine follow-up for stable hypertension.")
    assert res.payload["safety_level"] == "normal"
    assert "message_prefix" not in res.payload


def test_entity_text_contributes() -> None:
    res = run_safety(
        note_text="Patient reports symptoms.",
        question="TIA?",
        entities=[
            EntityItem(type="SYMPTOM", text="sudden weakness on left side", start=0, end=10),
        ],
    )
    assert res.payload["safety_level"] == "emergency"
