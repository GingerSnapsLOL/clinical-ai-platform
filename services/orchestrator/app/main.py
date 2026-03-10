import os
import re
from typing import List

import httpx
from fastapi import FastAPI

from services.shared.http_client import create_client, get_timeout, post_typed
from services.shared.logging_util import get_logger, set_trace_id, structured_log_middleware
from services.shared.llm_client import LLMClient

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
    ScoreRequest,
    ScoreResponse,
    SourceItem,
)


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


def _summarize_note(text: str, max_chars: int = 400) -> str:
    """
    Lightweight note summary: first 1–2 sentences, truncated to max_chars.
    Works on redacted text; avoids calling an LLM at this stage.
    """
    if not text:
        return ""
    stripped = text.strip()
    if len(stripped) <= max_chars:
        return stripped

    # Split into sentences and keep the first couple
    sentences = re.split(r"(?<=[.!?])\s+", stripped)
    if not sentences:
        return stripped[:max_chars]

    summary = sentences[0]
    if len(summary) < max_chars and len(sentences) > 1:
        summary = f"{summary} {sentences[1]}"
    return summary[:max_chars]


def _entities_hint(entities: List[EntityItem], max_entities: int = 8) -> str:
    """Render entities into a short hint string for retrieval queries."""
    if not entities:
        return ""
    parts = []
    for e in entities[:max_entities]:
        parts.append(f"{e.type}: {e.text}")
    return "; ".join(parts)


def _build_llm_prompt(
    question: str,
    entities: List[EntityItem],
    sources: List[SourceItem],
    risk: RiskBlock | None,
    trace_id: str,
) -> str:
    """
    Construct a single prompt string for llm-service from question, entities, passages, and risk.

    - Uses only the top 3 passages by score.
    - For each passage, includes source_id, title (when available), and text.
    - Truncates the total context to a safe maximum length.
    - Preserves deterministic ordering by descending score (stable for ties).
    - Logs the number of passages used and total context length.
    """
    MAX_CONTEXT_CHARS = 6000

    lines: List[str] = []

    lines.append(
        "You are a clinical decision support assistant helping clinicians reason "
        "about cardiometabolic and cardiovascular risk. Use the provided entities, "
        "evidence passages, and risk assessment to answer the question clearly and safely."
    )
    lines.append("")
    lines.append(f"Question:\n{question}")

    if entities:
        lines.append("")
        lines.append("Extracted entities:")
        for e in entities[:20]:
            lines.append(f"- {e.type}: {e.text} (span {e.start}-{e.end})")

    # Context from retrieval: top 3 passages by score, deterministic order.
    context_lines: List[str] = []
    num_passages_used = 0
    if sources:
        context_lines.append("")
        context_lines.append("Top evidence passages:")

        # Sort by score descending; stable sort preserves original order for ties.
        top_sources = sorted(sources, key=lambda s: s.score if s.score is not None else 0.0, reverse=True)[
            :3
        ]

        for idx, s in enumerate(top_sources, start=1):
            title = s.title or (s.metadata.get("title") if s.metadata else None)
            header = f"Passage {idx} (source_id={s.source_id})"
            if title:
                header += f" - {title}"
            context_lines.append(header + ":")
            context_lines.append(s.snippet or "")
            context_lines.append("")
            num_passages_used += 1

    # Risk block
    if risk is not None:
        context_lines.append("")
        context_lines.append("Risk assessment:")
        context_lines.append(f"- Overall risk label: {risk.label}")
        context_lines.append(f"- Risk score: {risk.score:.2f}")
        if risk.explanation:
            context_lines.append("- Top contributing factors:")
            for feat in risk.explanation[:5]:
                context_lines.append(f"  - {feat.feature}: {feat.contribution:.3f}")

    # Assemble context and enforce a safe maximum length.
    context_text = "\n".join(context_lines)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]

    # Log context stats before sending to llm-service.
    logger.info(
        "llm_context_built",
        extra={
            "trace_id": trace_id,
            "num_passages": num_passages_used,
            "context_length": len(context_text),
        },
    )

    if context_text:
        lines.append("")
        lines.append(context_text)

    lines.append("")
    lines.append(
        "Instructions:\n"
        "- Provide a concise, clinician-facing answer.\n"
        "- Reference the risk level and key factors.\n"
        "- If evidence is weak or missing, state the uncertainty.\n"
        "- Do NOT fabricate guideline names or numeric thresholds."
    )

    return "\n".join(lines)


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
    Call services in order: pii-service -> ner-service -> retrieval-service -> scoring-service.
    Aggregate results into AskResponse. Answer text is stub for now.
    Returns: answer, entities, sources, risk, trace_id (and pii_redacted).
    """
    trace_id = request.trace_id
    set_trace_id(trace_id)
    timeout = get_timeout()

    # 1) PII redaction
    redacted_text = request.note_text
    pii_redacted = False
    # 2) NER extraction (uses redacted text)
    entities: List[EntityItem] = []
    # 3) Retrieval (sources from retrieval-service only)
    sources: List[SourceItem] = []
    # 4) Scoring (uses entities from NER)
    risk = _stub_risk()

    async with create_client(timeout=timeout) as client:
        # 1) pii-service /v1/redact (trace_id in body + header)
        data, _, _ = await post_typed(
            client,
            _pii_url(),
            RedactRequest(trace_id=trace_id, text=request.note_text),
            RedactResponse,
            timeout=timeout,
            trace_id=trace_id,
        )
        if data is not None:
            redacted_text = data.redacted_text
            pii_redacted = True

        # 2) ner-service /v1/extract (input: redacted text from step 1)
        data, _, _ = await post_typed(
            client,
            _ner_url(),
            ExtractRequest(trace_id=trace_id, text=redacted_text),
            ExtractResponse,
            timeout=timeout,
            trace_id=trace_id,
        )
        if data is not None:
            entities = data.entities

        # 3) Build enriched retrieval query from question, entities, and note summary,
        #    then call retrieval-service /v1/retrieve (sources come only from real retrieval output).
        enriched_query = request.question
        ent_hint = _entities_hint(entities)
        if ent_hint:
            enriched_query += f"\n\nKey entities: {ent_hint}"
        note_summary = _summarize_note(redacted_text)
        if note_summary:
            enriched_query += f"\n\nNote summary: {note_summary}"

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
        num_passages = len(retrieval_data.passages) if retrieval_data else 0
        logger.info(
            "retrieval",
            extra={
                "trace_id": trace_id,
                "retrieval_query": enriched_query[:300],
                "passages_returned": num_passages,
            },
        )
        if retrieval_data is not None and retrieval_data.passages:
            sources = [
                SourceItem(
                    source_id=p.source_id,
                    snippet=p.text,
                    score=p.score,
                    metadata=p.metadata,
                )
                for p in retrieval_data.passages
            ]

        # 4) scoring-service /v1/score (input: entities from step 2)
        data, _, _ = await post_typed(
            client,
            _scoring_url(),
            ScoreRequest(
                trace_id=trace_id,
                entities=entities,
                structured_features={},
            ),
            ScoreResponse,
            timeout=timeout,
            trace_id=trace_id,
        )
        if data is not None:
            risk = RiskBlock(
                score=data.score,
                label=data.label,
                explanation=data.explanation,
            )

    # Preferred path: synthesize answer via llm-service using retrieved context.
    answer: str
    try:
        llm_client = LLMClient()
        prompt = _build_llm_prompt(
            question=request.question,
            entities=entities,
            sources=sources,
            risk=risk,
            trace_id=trace_id,
        )
        llm_resp = await llm_client.generate(
            trace_id=trace_id,
            prompt=prompt,
            max_tokens=512,
            temperature=0.2,
        )
        # Close client only if it owns its underlying httpx client.
        await llm_client.aclose()

        answer = llm_resp.text
        logger.info(
            "llm_answer_success",
            extra={
                "trace_id": trace_id,
                "prompt_length": len(prompt),
            },
        )
    except (httpx.HTTPError, Exception) as exc:
        # On any LLM failure, fall back to deterministic template-based synthesis.
        logger.warning(
            "llm_answer_fallback",
            extra={
                "trace_id": trace_id,
                "error": str(exc),
            },
        )
        answer = _synthesize_answer(request.question, sources, risk)

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

    return AskResponse(
        trace_id=trace_id,
        pii_redacted=pii_redacted,
        answer=answer,
        entities=entities,
        sources=sources,
        risk=risk,
        citations=citations,
    )


if __name__ == "__main__":
    import uvicorn

    # Default to 8010 to match docker-compose and ORCHESTRATOR_URL
    port = int(os.getenv("ORCHESTRATOR_PORT", "8010"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

