"""Unit tests for :class:`ScoringAgent` (no HTTP)."""

from __future__ import annotations

from services.shared.schemas_v1 import EntityItem, ScoreResponse, TargetScoreResult

from app.agents.scoring_agent import PRIMARY_TARGET, ScoringAgent


def test_select_targets_always_includes_triage() -> None:
    assert ScoringAgent.select_targets({})[0] == PRIMARY_TARGET
    assert PRIMARY_TARGET == "triage_severity"


def test_select_targets_optional_extra_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ORCHESTRATOR_SCORING_EXTRA_TARGETS", " cardiovascular_risk , bogus ")
    targets = ScoringAgent.select_targets(None)
    assert targets == ["triage_severity", "cardiovascular_risk"]


def test_assess_input_quality_sparse() -> None:
    warns, q = ScoringAgent.assess_input_quality({}, {}, [])
    assert any("scoring_features_insufficient" in w for w in warns)
    assert q < 0.5


def test_assess_input_quality_stronger_with_entities() -> None:
    ent = [
        EntityItem(type="SYMPTOM", text="pain", start=0, end=4),
        EntityItem(type="DISEASE", text="htn", start=10, end=13),
    ]
    warns, q = ScoringAgent.assess_input_quality(
        {"systolic_bp": 150.0, "diastolic_bp": 92.0},
        {},
        ent,
    )
    assert q >= 0.85
    assert not any("no_entities_or_structured" in w for w in warns)


def test_has_relevant_entities_true_for_clinical_types() -> None:
    ent = [EntityItem(type="SYMPTOM", text="chest pain", start=0, end=10)]
    assert ScoringAgent.has_relevant_entities(ent) is True


def test_has_relevant_entities_false_for_empty_list() -> None:
    assert ScoringAgent.has_relevant_entities([]) is False


def test_interpret_response_multi_target_ready() -> None:
    resp = ScoreResponse(
        trace_id="t1",
        score=0.4,
        label="medium",
        explanation="Medium triage (stub).",
        contributions=[],
        target_results={
            "triage_severity": TargetScoreResult(
                target="triage_severity",
                score=0.4,
                label="medium",
                explanation=[],
                ready=True,
            ),
        },
        risk_available=True,
        confidence=0.4,
    )
    out = ScoringAgent.interpret_response(resp, ["triage_severity"])
    assert out["ready"] is True
    assert out["primary"]["score"] == 0.4
    assert "triage_severity" in out["scores"]


def test_interpret_not_ready_secondary() -> None:
    resp = ScoreResponse(
        trace_id="t2",
        score=0.2,
        label="low",
        explanation="",
        contributions=[],
        target_results={
            "triage_severity": TargetScoreResult(
                target="triage_severity",
                score=0.2,
                label="low",
                explanation=[],
                ready=True,
            ),
            "stroke_risk": TargetScoreResult(
                target="stroke_risk",
                score=0.0,
                label="low",
                explanation=[],
                ready=False,
                detail="Model not trained",
            ),
        },
    )
    out = ScoringAgent.interpret_response(resp, ["triage_severity", "stroke_risk"])
    assert out["ready"] is False
    assert out["scores"]["stroke_risk"]["ready"] is False
