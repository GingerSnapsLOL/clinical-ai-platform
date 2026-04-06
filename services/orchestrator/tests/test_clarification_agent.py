"""Tests for :func:`run_clarification`."""

from __future__ import annotations

from app.agents.clarification_agent import run_clarification


def test_priority_age_duration_severity() -> None:
    res = run_clarification(
        ["severity", "age", "duration"],
        structured_features={"systolic_bp": 120.0, "diastolic_bp": 80.0},
        include_vitals_if_sparse=False,
    )
    qs = res.payload["questions"]
    assert len(qs) == 3
    assert "age" in qs[0].lower()
    assert "how long" in qs[1].lower()
    assert "severe" in qs[2].lower()


def test_appends_vitals_when_sparse_and_other_gaps() -> None:
    res = run_clarification(
        ["age"],
        structured_features={},
        include_vitals_if_sparse=True,
    )
    assert len(res.payload["questions"]) == 2
    assert "vital" in res.payload["questions"][-1].lower()


def test_no_vitals_nag_when_no_other_gaps() -> None:
    res = run_clarification([], structured_features={}, include_vitals_if_sparse=True)
    assert res.payload["questions"] == []


def test_synonym_patient_age() -> None:
    res = run_clarification(["patient_age"], include_vitals_if_sparse=False)
    assert any("age" in q.lower() for q in res.payload["questions"])
