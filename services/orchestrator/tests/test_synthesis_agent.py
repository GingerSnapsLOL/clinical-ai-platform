"""Tests for synthesis prompt assembly (no LLM)."""

from __future__ import annotations

from services.shared.schemas_v1 import EntityItem, RiskBlock, SourceItem

from app.agents.synthesis_agent import build_synthesis_prompt


def test_build_synthesis_prompt_includes_grounding_and_safety() -> None:
    src = [
        SourceItem(
            source_id="s1",
            snippet="Educational passage on hypertension follow-up and monitoring.",
            score=1.3,
        )
    ]
    ent = [EntityItem(type="DISEASE", text="hypertension", start=0, end=12)]
    risk = RiskBlock(score=0.3, label="low", explanation=[])
    safety = {"safety_level": "warning", "actions": ["use_cautious_language"]}
    scores = {"primary": {"target": "triage_severity", "score": 0.3, "label": "low"}}
    sf = {"systolic_bp": 148.0}
    p = build_synthesis_prompt(
        question="Plan?",
        entities=ent,
        sources=src,
        structured_features=sf,
        scores=scores,
        safety=safety,
        risk=risk,
        trace_id="t1",
    )
    assert "Top evidence passages" in p
    assert "Structured features" in p or "systolic_bp" in p
    assert "Safety / policy" in p
    assert "uncertainty" in p.lower()


def test_scores_optional() -> None:
    p = build_synthesis_prompt(
        question="Q",
        entities=[],
        sources=[
            SourceItem(source_id="x", snippet="Some context text for the test.", score=0.9),
        ],
        structured_features={},
        scores=None,
        safety=None,
        risk=None,
        trace_id="t2",
    )
    assert "Question:" in p and "Instructions (strict grounding" in p
