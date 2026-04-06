import asyncio
import hashlib
import json
import os
import time
from typing import Any, List, Optional

import httpx
from fastapi import FastAPI

from services.shared.http_client import create_client, get_timeout, post_typed
from services.shared.logging_util import get_logger, set_trace_id, structured_log_middleware
from services.shared.llm_client import LLMClient

from agent_nodes import (
    answer_verifier_node,
    draft_answer_node,
    evidence_selector_node,
    finalize_answer_node,
)
from agent_runtime import compile_linear_chain
from agent_state import AgentState
from app.note_query import (
    build_enriched_retrieval_query,
    normalize_text_key,
    retrieval_cache_key,
    sha256_hex,
)
from app.agents.base import SupervisorContext
from app.agents.clinical_structuring_agent import ClinicalStructuringAgent
from app.agents.scoring_agent import run_scoring_step
from app.prompts.llm_ask import build_llm_prompt
from app.agent_pipeline import agent_pipeline_debug, run_supervised_ask, supervisor_pipeline_enabled
from app.relevance import retrieval_meets_relevance_bar

# Backward compat for tests importing the relevance helper from this module.
_retrieval_meets_relevance_bar = retrieval_meets_relevance_bar

try:
    from redis.asyncio import Redis  # type: ignore
except Exception:  # pragma: no cover
    Redis = None  # type: ignore[assignment]

logger = get_logger(__name__, "orchestrator")
from services.shared.schemas_v1 import (
    AskRequest,
    AskResponse,
    CitationItem,
    EntityItem,
    ExtractRequest,
    ExtractResponse,
    FeatureContribution,
    HealthResponse,
    RedactRequest,
    RedactResponse,
    RetrieveRequest,
    RetrieveResponse,
    RiskBlock,
    SourceItem,
)

_redis_client: Optional["Redis"] = None


def _cache_enabled() -> bool:
    raw = os.getenv("ORCHESTRATOR_CACHE_ENABLED", "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _redis_url() -> str:
    url = os.getenv("REDIS_URL")
    if url:
        return url
    host = os.getenv("REDIS_HOST", "redis")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    return f"redis://{host}:{port}/{db}"


def _retrieval_cache_ttl_sec() -> int:
    return int(os.getenv("ORCHESTRATOR_RETRIEVAL_CACHE_TTL_SEC", "300"))


def _answer_cache_ttl_sec() -> int:
    return int(os.getenv("ORCHESTRATOR_ANSWER_CACHE_TTL_SEC", "900"))


def _agent_mode_enabled() -> bool:
    """When True, use multi-step agent runtime (selector → draft → verify → finalize) instead of single-shot LLM."""
    raw = os.getenv("ORCHESTRATOR_AGENT_MODE", "false").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _note_text_hash(note_text: str) -> str:
    return hashlib.sha256((note_text or "").encode("utf-8", errors="ignore")).hexdigest()


def _answer_cache_key(
    question: str,
    note_hash: str,
    top_source_ids: list[str],
    model_name: str,
    *,
    use_agent: bool = False,
) -> str:
    qn = normalize_text_key(question)
    ids = ",".join(top_source_ids)
    if use_agent:
        payload = f"v1|q={qn}|note={note_hash}|ids={ids}|model={model_name}|agent=1"
    else:
        payload = f"v1|q={qn}|note={note_hash}|ids={ids}|model={model_name}"
    return f"orchestrator:answer:{sha256_hex(payload)}"


async def _get_redis() -> Optional["Redis"]:
    global _redis_client
    if not _cache_enabled():
        return None
    if Redis is None:
        return None
    if _redis_client is None:
        _redis_client = Redis.from_url(_redis_url(), decode_responses=True)
    return _redis_client


async def _cache_get_json(trace_id: str, key: str) -> Optional[dict[str, Any]]:
    r = await _get_redis()
    if r is None:
        return None
    try:
        raw = await r.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning(
            "cache_get_error",
            extra={"trace_id": trace_id, "key": key, "error": str(exc)},
        )
        return None


async def _cache_set_json(trace_id: str, key: str, value: dict[str, Any], ttl_sec: int) -> None:
    r = await _get_redis()
    if r is None:
        return
    try:
        await r.set(key, json.dumps(value, separators=(",", ":")), ex=ttl_sec)
    except Exception as exc:
        logger.warning(
            "cache_set_error",
            extra={"trace_id": trace_id, "key": key, "error": str(exc)},
        )


class _OrchestratorRetrievalCache:
    """Adapter: supervised retrieval agent uses the same Redis keys as the legacy path."""

    async def get_json(self, trace_id: str, key: str) -> dict[str, Any] | None:
        return await _cache_get_json(trace_id, key)

    async def set_json(self, trace_id: str, key: str, value: dict[str, Any], ttl_sec: int) -> None:
        await _cache_set_json(trace_id, key, value, ttl_sec)


def _pii_url() -> str:
    base = os.getenv("PII_SERVICE_URL", "http://pii-service:8020")
    return f"{base.rstrip('/')}/v1/redact"


def _ner_url() -> str:
    base = os.getenv("NER_SERVICE_URL", "http://ner-service:8030")
    return f"{base.rstrip('/')}/v1/extract"


def _retrieval_url() -> str:
    base = os.getenv("RETRIEVAL_SERVICE_URL", "http://retrieval-service:8040")
    return f"{base.rstrip('/')}/v1/retrieve"


def _scoring_url() -> str:
    base = os.getenv("SCORING_SERVICE_URL", "http://scoring-service:8050")
    return f"{base.rstrip('/')}/v1/score"


def _synthesize_answer(question: str, sources: List[SourceItem], risk: RiskBlock | None) -> str:
    """
    Simple, deterministic answer synthesis from top passages and risk block.
    Produces a structured markdown-style response with:
    - summary
    - key risks
    - recommended monitoring
    """
    # Summary: use snippets from top sources
    summary_lines: List[str] = []
    for s in sources[:3]:
        if s.snippet:
            summary_lines.append(s.snippet.strip())
    summary = " ".join(summary_lines)[:600] if summary_lines else "No guideline passages were retrieved."

    # Key risks: use scoring-service label/score when available
    risk_lines: List[str] = []
    if risk is not None:
        risk_lines.append(f"Overall risk is **{risk.label}** (score {risk.score:.2f}).")
        if risk.explanation:
            top_feats = ", ".join(feat.feature for feat in risk.explanation[:3])
            risk_lines.append(f"Top contributing factors include: {top_feats}.")
    else:
        risk_lines.append("Risk score is not available for this case.")

    # Recommended monitoring: heuristic text based on common cardiometabolic patterns
    monitoring_lines: List[str] = [
        "Monitor blood pressure regularly and titrate therapy to guideline targets.",
        "Check renal function and electrolytes (especially potassium) after starting or changing ACE inhibitor/ARB therapy.",
        "Reassess cardiovascular risk factors (lipids, diabetes control, smoking status) and reinforce lifestyle measures.",
    ]

    answer = (
        f"**Question**: {question}\n\n"
        f"### Summary\n{summary}\n\n"
        f"### Key risks\n" + "\n".join(f"- {line}" for line in risk_lines) + "\n\n"
        f"### Recommended monitoring\n" + "\n".join(f"- {line}" for line in monitoring_lines)
    )
    return answer


def _stub_risk() -> RiskBlock:
    return RiskBlock(
        score=0.72,
        label="high",
        explanation=[
            FeatureContribution(feature="bp_high", contribution=0.18),
            FeatureContribution(feature="age_bucket_60_70", contribution=0.12),
        ],
    )


app = FastAPI(title="Clinical AI Orchestrator", version="0.1.0")
app.add_middleware(structured_log_middleware("orchestrator"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="orchestrator")


@app.post("/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest) -> AskResponse:
    """
    Call services in order, parallelizing independent stages where safe:

    - Sequential: pii-service -> ner-service
      Reason: ner-service runs on redacted text from pii-service.

    - Parallel after NER: retrieval-service + scoring-service
      Reason: retrieval uses (question + entities + note summary) while scoring uses entities only,
      and scoring does not depend on retrieval output.

    Aggregate results into AskResponse. Answer text is stub for now.
    Returns: answer, entities, sources, risk, trace_id (and pii_redacted).
    """
    trace_id = request.trace_id
    set_trace_id(trace_id)
    t_request_start = time.perf_counter()
    timeout = get_timeout()

    if supervisor_pipeline_enabled():
        logger.info(
            "supervised_pipeline_branch",
            extra={"trace_id": trace_id, "debug": agent_pipeline_debug(request)},
        )
        async with create_client(timeout=timeout) as client:
            cache = _OrchestratorRetrievalCache() if _cache_enabled() else None
            return await run_supervised_ask(
                request,
                client,
                timeout,
                retrieval_cache=cache,
                debug=agent_pipeline_debug(request),
                t_request_start=t_request_start,
            )

    agent_mode_flag = _agent_mode_enabled()
    logger.info(
        "agent_mode_enabled",
        extra={"trace_id": trace_id, "enabled": agent_mode_flag},
    )
    warnings: List[str] = []

    pii_duration_ms: float = 0.0
    ner_duration_ms: float = 0.0
    retrieval_duration_ms: float = 0.0
    scoring_duration_ms: float = 0.0
    llm_duration_ms: float = 0.0
    fallback_synthesis_duration_ms: float = 0.0
    fallback_used: bool = False
    agent_node_timings: dict[str, float] = {}
    accept_retrieval: bool = False
    retrieval_cache_hit: bool = False
    answer_cache_hit: bool = False

    # 1) PII redaction
    redacted_text = request.note_text
    pii_redacted = False
    # 2) NER extraction (uses redacted text)
    entities: List[EntityItem] = []
    structured_features: dict[str, Any] = {}
    structuring_signals: dict[str, Any] = {}
    # 3) Retrieval (sources from retrieval-service only)
    sources: List[SourceItem] = []
    # 4) Scoring (uses entities from NER)
    risk = _stub_risk()

    async with create_client(timeout=timeout) as client:
        # 1) pii-service /v1/redact (trace_id in body + header)
        pii_t0 = time.perf_counter()
        data, _, _ = await post_typed(
            client,
            _pii_url(),
            RedactRequest(trace_id=trace_id, text=request.note_text),
            RedactResponse,
            timeout=timeout,
            trace_id=trace_id,
        )
        pii_duration_ms = (time.perf_counter() - pii_t0) * 1000.0
        logger.info(
            "latency_pii_service",
            extra={"trace_id": trace_id, "duration_ms": pii_duration_ms},
        )
        if data is not None:
            redacted_text = data.redacted_text
            pii_redacted = True

        # 2) ner-service /v1/extract (input: redacted text from step 1)
        ner_t0 = time.perf_counter()
        data, _, _ = await post_typed(
            client,
            _ner_url(),
            ExtractRequest(trace_id=trace_id, text=redacted_text),
            ExtractResponse,
            timeout=timeout,
            trace_id=trace_id,
        )
        ner_duration_ms = (time.perf_counter() - ner_t0) * 1000.0
        logger.info(
            "latency_ner_service",
            extra={"trace_id": trace_id, "duration_ms": ner_duration_ms},
        )
        if data is not None:
            entities = data.entities
            _extra = ClinicalStructuringAgent.enrich(
                redacted_text,
                list(entities),
                pii_redacted=pii_redacted,
            )
            structured_features = dict(_extra["structured_features"])
            structuring_signals = dict(_extra.get("signals") or {})
            for m in _extra.get("missing_inputs") or []:
                warnings.append(f"missing:{m}")

        # 3) Build enriched retrieval query from question, entities, and note summary.
        enriched_query = build_enriched_retrieval_query(request.question, redacted_text, entities)

        async def _run_retrieval() -> RetrieveResponse | None:
            nonlocal retrieval_duration_ms
            nonlocal retrieval_cache_hit
            retrieval_t0 = time.perf_counter()
            cache_key = retrieval_cache_key(enriched_query, top_k=50, top_n=3, rerank=True)
            cached = await _cache_get_json(trace_id, cache_key)
            if cached is not None and cached.get("v") == 1 and "retrieval" in cached:
                try:
                    retrieval_cache_hit = True
                    logger.info(
                        "retrieval_cache_hit",
                        extra={"trace_id": trace_id, "key": cache_key},
                    )
                    parsed = RetrieveResponse.model_validate(cached["retrieval"])
                    retrieval_duration_ms = (time.perf_counter() - retrieval_t0) * 1000.0
                    logger.info(
                        "latency_retrieval_service",
                        extra={"trace_id": trace_id, "duration_ms": retrieval_duration_ms},
                    )
                    return parsed
                except Exception:
                    retrieval_cache_hit = False

            logger.info(
                "retrieval_cache_miss",
                extra={"trace_id": trace_id, "key": cache_key},
            )
            retrieval_data, _, _ = await post_typed(
                client,
                _retrieval_url(),
                RetrieveRequest(
                    trace_id=trace_id,
                    query=enriched_query,
                    top_k=50,
                    top_n=3,
                    rerank=True,
                ),
                RetrieveResponse,
                timeout=timeout,
                trace_id=trace_id,
            )
            retrieval_duration_ms = (time.perf_counter() - retrieval_t0) * 1000.0
            logger.info(
                "latency_retrieval_service",
                extra={"trace_id": trace_id, "duration_ms": retrieval_duration_ms},
            )
            if retrieval_data is not None:
                await _cache_set_json(
                    trace_id,
                    cache_key,
                    {"v": 1, "retrieval": retrieval_data.model_dump()},
                    ttl_sec=_retrieval_cache_ttl_sec(),
                )
            return retrieval_data

        async def _run_scoring_agent() -> None:
            nonlocal risk, scoring_duration_ms
            scoring_t0 = time.perf_counter()
            ctx = SupervisorContext(
                trace_id=trace_id,
                question=request.question,
                note_text=request.note_text,
                client=client,
                timeout=timeout,
            )
            res = await run_scoring_step(
                ctx,
                list(entities),
                structured_features=structured_features,
                signals=structuring_signals,
            )
            scoring_duration_ms = res.duration_ms
            logger.info(
                "latency_scoring_service",
                extra={"trace_id": trace_id, "duration_ms": scoring_duration_ms},
            )
            for w in res.warnings:
                if w not in warnings:
                    warnings.append(w)
            if res.ok and res.payload:
                expl = res.payload.get("explanation") or []
                risk = RiskBlock(
                    score=float(res.payload["score"]),
                    label=res.payload["label"],
                    explanation=[FeatureContribution.model_validate(x) for x in expl],
                )

        # Parallelize retrieval + scoring (independent once entities exist).
        retrieval_result, _ = await asyncio.gather(
            _run_retrieval(),
            _run_scoring_agent(),
            return_exceptions=False,
        )

        num_passages = len(retrieval_result.passages) if retrieval_result else 0
        logger.info(
            "retrieval",
            extra={
                "trace_id": trace_id,
                "retrieval_query": enriched_query[:300],
                "passages_returned": num_passages,
            },
        )
        if retrieval_result is not None and retrieval_result.passages:
            sources = [
                SourceItem(
                    source_id=p.source_id,
                    snippet=p.text,
                    score=p.score,
                    metadata=p.metadata,
                )
                for p in retrieval_result.passages
            ]

    # Relevance gate: refuse before LLM when retrieval is empty or scores/snippets are weak.
    accept_retrieval, top_rel_score, relevance_reason = retrieval_meets_relevance_bar(sources)
    if not accept_retrieval:
        warnings.append(f"retrieval_relevance:{relevance_reason}")
        logger.info(
            "retrieval_relevance_reject",
            extra={
                "trace_id": trace_id,
                "top_score": top_rel_score,
                "reason": relevance_reason,
            },
        )
        answer = "Insufficient data"
    else:
        top_source_ids = [s.source_id for s in sources]
        model_name = os.getenv("LLM_MODEL_NAME", "unknown")
        use_agent = agent_mode_flag
        ans_key = _answer_cache_key(
            question=request.question,
            note_hash=_note_text_hash(request.note_text),
            top_source_ids=top_source_ids,
            model_name=model_name,
            use_agent=use_agent,
        )
        cached_ans = await _cache_get_json(trace_id, ans_key)
        if cached_ans is not None and cached_ans.get("v") == 1 and isinstance(cached_ans.get("answer"), str):
            answer_cache_hit = True
            logger.info(
                "answer_cache_hit",
                extra={"trace_id": trace_id, "key": ans_key},
            )
            answer = cached_ans["answer"]
        else:
            logger.info(
                "answer_cache_miss",
                extra={"trace_id": trace_id, "key": ans_key},
            )
        # Preferred path: synthesize answer via llm-service using retrieved context.
        if not answer_cache_hit:
            try:
                if use_agent:
                    logger.info(
                        "agent_runtime_started",
                        extra={"trace_id": trace_id},
                    )
                    initial_state = AgentState(
                        trace_id=trace_id,
                        question=request.question,
                        entities=entities,
                        sources=sources,
                        risk=risk,
                        warnings=list(warnings),
                    )
                    agent_runtime = compile_linear_chain(
                        evidence_selector_node,
                        draft_answer_node,
                        answer_verifier_node,
                        finalize_answer_node,
                    )
                    final_state, agent_node_timings = await agent_runtime.run(initial_state)
                    llm_duration_ms = agent_node_timings.get("agent_total_duration_ms", 0.0)
                    answer = (final_state.final_answer or "Insufficient data").strip()
                    for w in final_state.warnings:
                        if w not in warnings:
                            warnings.append(w)
                    logger.info(
                        "agent_runtime_finished",
                        extra={
                            "trace_id": trace_id,
                            "duration_ms": llm_duration_ms,
                            "answer_len": len(answer),
                            "agent_selector_duration_ms": round(
                                agent_node_timings.get("agent_selector_duration_ms", 0.0), 3
                            ),
                            "agent_draft_duration_ms": round(
                                agent_node_timings.get("agent_draft_duration_ms", 0.0), 3
                            ),
                            "agent_verifier_duration_ms": round(
                                agent_node_timings.get("agent_verifier_duration_ms", 0.0), 3
                            ),
                            "agent_finalize_duration_ms": round(
                                agent_node_timings.get("agent_finalize_duration_ms", 0.0), 3
                            ),
                        },
                    )
                    logger.info(
                        "latency_llm_service",
                        extra={
                            "trace_id": trace_id,
                            "duration_ms": llm_duration_ms,
                            "agent_mode": True,
                        },
                    )
                    logger.info(
                        "llm_answer_success",
                        extra={
                            "trace_id": trace_id,
                            "agent_mode": True,
                        },
                    )
                else:
                    llm_client = LLMClient()
                    prompt = build_llm_prompt(
                        question=request.question,
                        entities=entities,
                        sources=sources,
                        risk=risk,
                        trace_id=trace_id,
                    )
                    llm_t0 = time.perf_counter()
                    llm_resp = await llm_client.generate(
                        trace_id=trace_id,
                        prompt=prompt,
                        max_tokens=512,
                        temperature=0.2,
                    )
                    llm_duration_ms = (time.perf_counter() - llm_t0) * 1000.0
                    await llm_client.aclose()

                    answer = llm_resp.text
                    logger.info(
                        "latency_llm_service",
                        extra={"trace_id": trace_id, "duration_ms": llm_duration_ms},
                    )
                    logger.info(
                        "llm_answer_success",
                        extra={
                            "trace_id": trace_id,
                            "prompt_length": len(prompt),
                        },
                    )

                await _cache_set_json(
                    trace_id,
                    ans_key,
                    {"v": 1, "answer": answer},
                    ttl_sec=_answer_cache_ttl_sec(),
                )
            except (httpx.HTTPError, Exception) as exc:
                # On any LLM failure, fall back to deterministic template-based synthesis.
                if use_agent:
                    logger.warning(
                        "agent_runtime_fallback",
                        extra={"trace_id": trace_id, "error": str(exc)},
                    )
                logger.warning(
                    "llm_answer_fallback",
                    extra={
                        "trace_id": trace_id,
                        "error": str(exc),
                        "agent_mode": use_agent,
                    },
                )
                fallback_used = True
                fb_t0 = time.perf_counter()
                answer = _synthesize_answer(request.question, sources, risk)
                fallback_synthesis_duration_ms = (time.perf_counter() - fb_t0) * 1000.0
                logger.info(
                    "latency_fallback_synthesis",
                    extra={
                        "trace_id": trace_id,
                        "duration_ms": fallback_synthesis_duration_ms,
                    },
                )

    # Citations: unique source_ids with optional titles extracted from metadata or SourceItem
    citations: List[CitationItem] = []
    seen_ids: set[str] = set()
    for s in sources:
        if s.source_id in seen_ids:
            continue
        seen_ids.add(s.source_id)

        title = s.title
        url = s.url
        if s.metadata:
            title = title or s.metadata.get("title")
            url = url or s.metadata.get("url")

        citations.append(
            CitationItem(
                source_id=s.source_id,
                title=title,
                url=url,
            )
        )

    total_duration_ms = (time.perf_counter() - t_request_start) * 1000.0
    summary_extra: dict[str, Any] = {
        "trace_id": trace_id,
        "total_duration_ms": total_duration_ms,
        "pii_service_duration_ms": pii_duration_ms,
        "ner_service_duration_ms": ner_duration_ms,
        "retrieval_service_duration_ms": retrieval_duration_ms,
        "scoring_service_duration_ms": scoring_duration_ms,
        "llm_service_duration_ms": llm_duration_ms,
        "fallback_synthesis_duration_ms": fallback_synthesis_duration_ms,
        "fallback_used": fallback_used,
        "retrieval_accepted": accept_retrieval,
        "retrieval_cache_hit": retrieval_cache_hit,
        "answer_cache_hit": answer_cache_hit,
        "agent_mode": agent_mode_flag,
    }
    if agent_node_timings:
        summary_extra["agent_total_duration_ms"] = round(
            agent_node_timings.get("agent_total_duration_ms", 0.0), 3
        )
        summary_extra["agent_selector_duration_ms"] = round(
            agent_node_timings.get("agent_selector_duration_ms", 0.0), 3
        )
        summary_extra["agent_draft_duration_ms"] = round(
            agent_node_timings.get("agent_draft_duration_ms", 0.0), 3
        )
        summary_extra["agent_verifier_duration_ms"] = round(
            agent_node_timings.get("agent_verifier_duration_ms", 0.0), 3
        )
        summary_extra["agent_finalize_duration_ms"] = round(
            agent_node_timings.get("agent_finalize_duration_ms", 0.0), 3
        )

    logger.info("ask_latency_summary", extra=summary_extra)

    timings_payload: dict[str, float] = {
        "total_request_time_ms": total_duration_ms,
        "pii_service_duration_ms": pii_duration_ms,
        "ner_service_duration_ms": ner_duration_ms,
        "retrieval_service_duration_ms": retrieval_duration_ms,
        "scoring_service_duration_ms": scoring_duration_ms,
        "llm_service_duration_ms": llm_duration_ms,
        "fallback_synthesis_duration_ms": fallback_synthesis_duration_ms,
    }
    if agent_node_timings:
        timings_payload.update(agent_node_timings)

    return AskResponse(
        trace_id=trace_id,
        pii_redacted=pii_redacted,
        answer=answer,
        entities=entities,
        sources=sources,
        risk=risk,
        citations=citations,
        warnings=warnings,
        total_request_time_ms=total_duration_ms,
        retrieval_time_ms=retrieval_duration_ms,
        llm_time_ms=llm_duration_ms,
        timings=timings_payload,
    )


if __name__ == "__main__":
    import uvicorn

    # Default to 8010 to match docker-compose and ORCHESTRATOR_URL
    port = int(os.getenv("ORCHESTRATOR_PORT", "8010"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

