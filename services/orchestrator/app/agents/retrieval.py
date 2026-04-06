"""
RetrievalAgent: multi-axis retrieval (symptom / condition), merge, dedupe, rerank.

At most two retrieval-service calls when both symptom and condition axes are present;
one call for general fallback. Optional cache per sub-query.
"""

from __future__ import annotations

from typing import Any

from services.shared.http_client import post_typed
from services.shared.logging_util import get_logger
from services.shared.schemas_v1 import EntityItem, PassageItem, RetrieveRequest, RetrieveResponse

from app.agents.base import AgentResult, AgentRole, RetrievalCachePort, SupervisorContext, monotonic_ms
from app.agents.retrieval_agent import MergedRetrieval, RetrievalAgent, TaggedPassage
from app.note_query import retrieval_cache_key

logger = get_logger(__name__, "orchestrator")


async def _cache_get_adapter(
    port: RetrievalCachePort | None, trace_id: str, key: str
) -> dict[str, Any] | None:
    if port is None:
        return None
    return await port.get_json(trace_id, key)


async def _cache_set_adapter(
    port: RetrievalCachePort | None,
    trace_id: str,
    key: str,
    value: dict[str, Any],
    ttl_sec: int,
) -> None:
    if port is None:
        return
    await port.set_json(trace_id, key, value, ttl_sec)


async def _retrieve_one(
    ctx: SupervisorContext,
    query: str,
    *,
    top_k: int,
    top_n: int,
    rerank: bool,
) -> tuple[RetrieveResponse | None, bool, str]:
    cache_key = retrieval_cache_key(query, top_k=top_k, top_n=top_n, rerank=rerank)
    try:
        cached = await _cache_get_adapter(ctx.retrieval_cache, ctx.trace_id, cache_key)
        if cached is not None and cached.get("v") == 1 and "retrieval" in cached:
            data = RetrieveResponse.model_validate(cached["retrieval"])
            return data, True, cache_key
        data, _, _ = await post_typed(
            ctx.client,
            ctx.urls["retrieval"],
            RetrieveRequest(
                trace_id=ctx.trace_id,
                query=query,
                top_k=top_k,
                top_n=top_n,
                rerank=rerank,
            ),
            RetrieveResponse,
            timeout=ctx.timeout,
            trace_id=ctx.trace_id,
        )
        if data is not None:
            await _cache_set_adapter(
                ctx.retrieval_cache,
                ctx.trace_id,
                cache_key,
                {"v": 1, "retrieval": data.model_dump()},
                ctx.retrieval_cache_ttl_sec,
            )
        return data, False, cache_key
    except Exception as exc:
        logger.warning(
            "retrieval_subquery_error",
            extra={"trace_id": ctx.trace_id, "error": str(exc)},
        )
        return None, False, cache_key


async def run_retrieval(
    ctx: SupervisorContext,
    *,
    redacted_text: str,
    entities: list[EntityItem],
    top_k: int = 50,
    top_n: int = 3,
    rerank: bool = True,
) -> AgentResult:
    t0 = monotonic_ms()
    warnings: list[str] = []
    axes = RetrievalAgent.build_query_axes(ctx.question, redacted_text, entities)
    max_calls = len(axes)
    trace: dict[str, Any] = {
        "bounded_remote_calls": max_calls,
        "agent_role": AgentRole.RETRIEVAL,
        "retrieval_axes": [a[0] for a in axes],
    }

    per_axis: list[tuple[str, list[PassageItem]]] = []
    any_cache_hit = False

    for axis, query in axes:
        retrieval_data, cache_hit, _ = await _retrieve_one(
            ctx, query, top_k=top_k, top_n=top_n, rerank=rerank
        )
        any_cache_hit = any_cache_hit or cache_hit
        passages = list(retrieval_data.passages) if retrieval_data is not None else []
        per_axis.append((axis, passages))
        if retrieval_data is None:
            warnings.append(f"retrieval_axis_empty:{axis}")

    merged: MergedRetrieval = RetrievalAgent.merge_passages(per_axis)
    tagged: list[TaggedPassage] = merged.tagged

    coverage = RetrievalAgent.coverage_score_for(tagged)
    weak = RetrievalAgent.weak_retrieval_warnings(tagged, coverage)
    warnings.extend(weak)

    top_passages_limit = max(12, top_n * 4)
    top_passages = RetrievalAgent.top_passages_payload(tagged, top_passages_limit)
    evidence_clusters = RetrievalAgent.evidence_clusters_from_tagged(tagged)
    sources_objs = RetrievalAgent.passages_to_source_items(tagged, top_n)
    sources = list(sources_objs)

    top_score = 0.0
    if tagged:
        scores = [float(t.passage.score) for t in tagged if t.passage.score is not None]
        top_score = max(scores) if scores else 0.0

    trace["passages_after_merge"] = len(tagged)
    trace["dedupe_drop_count"] = merged.dedupe_dropped
    trace["top_source_score"] = top_score
    trace["coverage_score"] = coverage
    trace["retrieval_cache_hit"] = any_cache_hit
    trace["cache_key_suffix"] = retrieval_cache_key(
        "|".join(q for _, q in axes),
        top_k=top_k,
        top_n=top_n,
        rerank=rerank,
    )[-24:]

    duration_ms = monotonic_ms() - t0

    ok = bool(tagged)
    confidence = min(1.0, max(0.0, 0.55 * coverage + 0.45 * min(1.0, top_score / 10.0)))
    if not ok:
        confidence = 0.0
        warnings.append("no_passages")

    queries_summary = " | ".join(f"{ax}::{q[:120]}..." if len(q) > 120 else f"{ax}::{q}" for ax, q in axes)

    payload: dict[str, Any] = {
        "enriched_query": queries_summary,
        "queries": [{"axis": ax, "query": q} for ax, q in axes],
        "sources": [s.model_dump() for s in sources],
        "top_passages": top_passages,
        "evidence_clusters": evidence_clusters,
        "coverage_score": coverage,
        "cache_hit": any_cache_hit,
    }

    return AgentResult(
        agent_id=AgentRole.RETRIEVAL,
        ok=ok,
        confidence=confidence if ok else 0.0,
        warnings=warnings,
        payload=payload,
        error_detail=None if ok else "retrieval_empty",
        duration_ms=duration_ms,
        trace=trace,
    )
