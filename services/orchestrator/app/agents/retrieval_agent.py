"""
RetrievalAgent: multi-axis queries (symptom / condition), merge, dedupe, and rank passages.

Deterministic, bounded: at most two retrieval-service calls when both axes have entities.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from services.shared.schemas_v1 import EntityItem, PassageItem, SourceItem

from app.agents.clinical_structuring_agent import ClinicalStructuringAgent
from app.note_query import entities_hint, normalize_text_key, summarize_note

# Tunable thresholds (align with orchestrator relevance gate defaults)
_MIN_PASSAGES_FOR_STRONG_EVIDENCE = 2
_COVERAGE_UNIQUE_SOURCES_TARGET = 5.0
_COVERAGE_TOP_SCORE_REF = 8.0
_COVERAGE_PASSAGE_TARGET = 12.0


@dataclass
class TaggedPassage:
    passage: PassageItem
    cluster: str  # "symptom" | "condition" | "general"


@dataclass
class MergedRetrieval:
    tagged: list[TaggedPassage] = field(default_factory=list)
    dedupe_dropped: int = 0


class RetrievalAgent:
    """Builds queries and merges retrieval results for clinical evidence."""

    @staticmethod
    def build_query_axes(
        question: str,
        redacted_text: str,
        entities: list[EntityItem],
    ) -> list[tuple[str, str]]:
        """
        Return ordered list of (axis, query_text).

        Axes: ``symptom``, ``condition``, and optionally ``general`` as a single fallback.
        """
        symptoms, diseases, _risk = ClinicalStructuringAgent._bucket_entities(entities)
        axes: list[tuple[str, str]] = []

        q = (question or "").strip()
        note_summary = summarize_note(redacted_text)

        if symptoms:
            lines = [
                q,
                "Focus: presenting symptoms, complaints, and associated findings.",
            ]
            sh = entities_hint(symptoms, max_entities=12)
            if sh:
                lines.append(f"Symptom entities: {sh}")
            if note_summary:
                lines.append(f"Clinical note excerpt: {note_summary}")
            axes.append(("symptom", "\n\n".join(lines)))

        if diseases:
            lines = [
                q,
                "Focus: diagnoses, chronic conditions, and disease-specific management.",
            ]
            dh = entities_hint(diseases, max_entities=12)
            if dh:
                lines.append(f"Condition entities: {dh}")
            if note_summary:
                lines.append(f"Clinical note excerpt: {note_summary}")
            axes.append(("condition", "\n\n".join(lines)))

        if not axes:
            lines = [q]
            eh = entities_hint(entities, max_entities=12)
            if eh:
                lines.append(f"Clinical entities: {eh}")
            if note_summary:
                lines.append(f"Clinical note excerpt: {note_summary}")
            axes.append(("general", "\n\n".join(lines)))

        return axes

    @staticmethod
    def merge_passages(per_axis_results: list[tuple[str, list[PassageItem]]]) -> MergedRetrieval:
        """
        Dedupe by (source_id, normalized text); keep highest score; track cluster provenance.
        """
        best: dict[tuple[str, str], tuple[float, PassageItem, set[str]]] = {}
        dropped = 0

        for axis, passages in per_axis_results:
            for p in passages:
                key = (p.source_id, normalize_text_key(p.text))
                score = float(p.score) if p.score is not None else 0.0
                if key not in best:
                    best[key] = (score, p, {axis})
                else:
                    prev_score, prev_p, clusters = best[key]
                    clusters = set(clusters) | {axis}
                    if score > prev_score:
                        best[key] = (score, p, clusters)
                        dropped += 1
                    elif score < prev_score:
                        dropped += 1
                    else:
                        best[key] = (prev_score, prev_p, clusters)

        tagged: list[TaggedPassage] = []
        for _k, (_sc, passage, clusters) in best.items():
            if "symptom" in clusters and "condition" in clusters:
                cluster_name = "symptom_condition"
            elif "symptom" in clusters:
                cluster_name = "symptom"
            elif "condition" in clusters:
                cluster_name = "condition"
            else:
                cluster_name = next(iter(clusters), "general")
            tagged.append(TaggedPassage(passage=passage, cluster=cluster_name))

        tagged.sort(key=lambda t: float(t.passage.score) if t.passage.score is not None else 0.0, reverse=True)

        return MergedRetrieval(tagged=tagged, dedupe_dropped=dropped)

    @staticmethod
    def evidence_clusters_from_tagged(tagged: list[TaggedPassage]) -> list[dict[str, Any]]:
        buckets: dict[str, list[TaggedPassage]] = {}
        for t in tagged:
            buckets.setdefault(t.cluster, []).append(t)
        clusters: list[dict[str, Any]] = []
        for name, items in sorted(buckets.items(), key=lambda x: x[0]):
            scores = [float(i.passage.score) for i in items if i.passage.score is not None]
            clusters.append(
                {
                    "name": name,
                    "passage_count": len(items),
                    "unique_sources": len({i.passage.source_id for i in items}),
                    "max_score": max(scores) if scores else 0.0,
                    "mean_score": sum(scores) / len(scores) if scores else 0.0,
                },
            )
        return clusters

    @staticmethod
    def coverage_score_for(tagged: list[TaggedPassage]) -> float:
        if not tagged:
            return 0.0
        sources = {t.passage.source_id for t in tagged}
        scores = [float(t.passage.score) for t in tagged if t.passage.score is not None]
        top = max(scores) if scores else 0.0
        breadth = min(1.0, len(sources) / _COVERAGE_UNIQUE_SOURCES_TARGET)
        strength = min(1.0, top / _COVERAGE_TOP_SCORE_REF)
        depth = min(1.0, len(tagged) / _COVERAGE_PASSAGE_TARGET)
        return float(max(0.0, min(1.0, 0.38 * breadth + 0.42 * strength + 0.20 * depth)))

    @staticmethod
    def weak_retrieval_warnings(tagged: list[TaggedPassage], coverage: float) -> list[str]:
        out: list[str] = []
        if not tagged:
            return out
        scores = [float(t.passage.score) for t in tagged if t.passage.score is not None]
        top = max(scores) if scores else 0.0
        min_top = float(os.getenv("ORCHESTRATOR_RETRIEVAL_MIN_TOP_SCORE", "1.0"))
        if len(tagged) < _MIN_PASSAGES_FOR_STRONG_EVIDENCE:
            out.append("retrieval_evidence_insufficient:few_passages")
        if top < min_top:
            out.append("retrieval_evidence_insufficient:low_top_score")
        if coverage < 0.35:
            out.append("retrieval_evidence_insufficient:low_coverage")
        return out

    @staticmethod
    def passages_to_source_items(tagged: list[TaggedPassage], limit: int) -> list[SourceItem]:
        out: list[SourceItem] = []
        for t in tagged[:limit]:
            p = t.passage
            meta: dict[str, Any] = dict(p.metadata or {})
            meta["retrieval_cluster"] = t.cluster
            out.append(
                SourceItem(
                    source_id=p.source_id,
                    snippet=p.text,
                    score=p.score,
                    metadata=meta,
                )
            )
        return out

    @staticmethod
    def top_passages_payload(tagged: list[TaggedPassage], limit: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for t in tagged[:limit]:
            p = t.passage
            rows.append(
                {
                    "source_id": p.source_id,
                    "text": p.text,
                    "score": float(p.score) if p.score is not None else 0.0,
                    "metadata": p.metadata,
                    "cluster": t.cluster,
                },
            )
        return rows
