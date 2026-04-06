"""Tests for :func:`run_evidence_critic` (deterministic, no HTTP)."""

from __future__ import annotations

from services.shared.schemas_v1 import EntityItem, SourceItem

from app.agents.evidence_critic import run_evidence_critic


def _src(sid: str, score: float, snippet: str) -> SourceItem:
    return SourceItem(source_id=sid, snippet=snippet, score=score)


def test_valid_when_clean_passages_and_scores() -> None:
    sources = [_src("a", 2.0, "This passage has enough characters for relevance.")]
    ent = [EntityItem(type="SYMPTOM", text="mild headache", start=0, end=12)]
    res = run_evidence_critic(
        sources=sources,
        entities=ent,
        structured_features={},
        signals={"uncertainty": {"hedging_language": False}},
        scoring_payload={"score": 0.25, "label": "low"},
        retrieval_payload={"coverage_score": 0.75},
        missing_inputs=[],
        retrieval_step_confidence=0.7,
        scoring_step_confidence=0.6,
    )
    assert res.payload["valid"] is True
    assert res.payload["issues"] == []
    assert res.payload["confidence_adjustment"] == 1.0


def test_weak_evidence_low_top_score() -> None:
    sources = [_src("b", 0.2, "short")]
    res = run_evidence_critic(
        sources=sources,
        entities=[],
        structured_features={},
        signals={},
        scoring_payload={"score": 0.1, "label": "low"},
        retrieval_payload={},
        missing_inputs=[],
    )
    assert "weak_evidence" in res.payload["issues"]
    assert res.payload["valid"] is False


def test_contradictory_hedging_and_high_score() -> None:
    sources = [_src("c", 1.5, "Educational content for testing relevance gate pass.")]
    ent = [EntityItem(type="SYMPTOM", text="pain", start=0, end=4)]
    res = run_evidence_critic(
        sources=sources,
        entities=ent,
        structured_features={},
        signals={"uncertainty": {"hedging_language": True}},
        scoring_payload={"score": 0.8, "label": "medium"},
        retrieval_payload={"coverage_score": 0.8},
        missing_inputs=[],
    )
    assert "contradictory_signals" in res.payload["issues"]
    assert res.payload["valid"] is False
    assert res.payload["confidence_adjustment"] < 1.0


def test_acuity_mismatch_flags_score_not_supported() -> None:
    sources = [_src("d", 1.2, "Another sufficiently long passage for gate rules.")]
    ent = [
        EntityItem(type="SYMPTOM", text="substernal chest pain", start=0, end=10),
        EntityItem(type="SYMPTOM", text="diaphoresis", start=0, end=5),
    ]
    res = run_evidence_critic(
        sources=sources,
        entities=ent,
        structured_features={"systolic_bp": 180.0},
        signals={},
        scoring_payload={"score": 0.05, "label": "low"},
        retrieval_payload={},
        missing_inputs=[],
    )
    assert "score_not_supported" in res.payload["issues"]


def test_insufficient_missing_inputs() -> None:
    sources = [_src("e", 1.3, "Passages must be long enough for the relevance heuristic.")]
    res = run_evidence_critic(
        sources=sources,
        entities=[EntityItem(type="SYMPTOM", text="x", start=0, end=1)],
        structured_features={},
        signals={},
        scoring_payload={"score": 0.2, "label": "low"},
        retrieval_payload={},
        missing_inputs=["age", "duration"],
    )
    assert "insufficient_data" in res.payload["issues"]
