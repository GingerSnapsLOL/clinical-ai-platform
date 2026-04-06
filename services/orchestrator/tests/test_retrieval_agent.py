"""Tests for :class:`RetrievalAgent` query building and merge/dedupe."""

from __future__ import annotations

from services.shared.schemas_v1 import EntityItem, PassageItem

from app.agents.retrieval_agent import RetrievalAgent
from app.note_query import normalize_text_key


def _e(etype: str, text: str) -> EntityItem:
    return EntityItem(type=etype, text=text, start=0, end=len(text))


def test_build_query_axes_symptom_and_condition() -> None:
    entities = [_e("SYMPTOM", "chest pain"), _e("DISEASE", "hypertension")]
    axes = RetrievalAgent.build_query_axes(
        "What is the workup?",
        "Patient with substernal chest pain and HTN.",
        entities,
    )
    labels = [a[0] for a in axes]
    assert labels == ["symptom", "condition"]
    assert "chest pain" in axes[0][1]
    assert "hypertension" in axes[1][1] or "condition" in axes[1][1].lower()


def test_build_query_axes_general_fallback() -> None:
    axes = RetrievalAgent.build_query_axes("Question only", "short", [])
    assert len(axes) == 1
    assert axes[0][0] == "general"


def test_merge_dedupes_by_source_and_text_keeps_best_score() -> None:
    p1 = PassageItem(source_id="s1", text="Same text", score=1.0)
    p2 = PassageItem(source_id="s1", text="Same text", score=4.0)
    p3 = PassageItem(source_id="s1", text="other", score=2.0)
    merged = RetrievalAgent.merge_passages(
        [
            ("symptom", [p1, p3]),
            ("condition", [p2]),
        ]
    )
    assert merged.dedupe_dropped >= 1
    assert len(merged.tagged) == 2
    best = next(
        t for t in merged.tagged if normalize_text_key(t.passage.text) == normalize_text_key("Same text")
    )
    assert best.passage.score == 4.0
    assert best.cluster == "symptom_condition"


def test_coverage_and_clusters() -> None:
    tagged_list = RetrievalAgent.merge_passages(
        [
            (
                "symptom",
                [
                    PassageItem(source_id="a", text="t1", score=3.0),
                    PassageItem(source_id="b", text="t2", score=2.0),
                ],
            ),
        ]
    ).tagged
    cov = RetrievalAgent.coverage_score_for(tagged_list)
    assert 0.0 < cov <= 1.0
    clusters = RetrievalAgent.evidence_clusters_from_tagged(tagged_list)
    assert any(c["name"] == "symptom" for c in clusters)
