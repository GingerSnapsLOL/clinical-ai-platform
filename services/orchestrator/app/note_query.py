"""Shared note summarization, entity hints, and retrieval cache keys for the orchestrator."""

from __future__ import annotations

import hashlib
import re

from services.shared.schemas_v1 import EntityItem


def normalize_text_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def retrieval_cache_key(query: str, top_k: int, top_n: int, rerank: bool) -> str:
    norm = normalize_text_key(query)
    payload = f"v1|q={norm}|top_k={top_k}|top_n={top_n}|rerank={int(rerank)}"
    return f"orchestrator:retrieval:{sha256_hex(payload)}"


def summarize_note(text: str, max_chars: int = 400) -> str:
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    if not sentences:
        return stripped[:max_chars]
    summary = sentences[0]
    if len(summary) < max_chars and len(sentences) > 1:
        summary = f"{summary} {sentences[1]}"
    return summary[:max_chars]


def entities_hint(entities: list[EntityItem], max_entities: int = 8) -> str:
    if not entities:
        return ""
    parts = [f"{e.type}: {e.text}" for e in entities[:max_entities]]
    return "; ".join(parts)


def build_enriched_retrieval_query(
    question: str,
    redacted_text: str,
    entities: list[EntityItem],
) -> str:
    enriched = question
    ent_hint = entities_hint(entities)
    if ent_hint:
        enriched += f"\n\nKey entities: {ent_hint}"
    note_summary = summarize_note(redacted_text)
    if note_summary:
        enriched += f"\n\nNote summary: {note_summary}"
    return enriched
