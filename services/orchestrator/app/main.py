import os
from typing import List

from fastapi import FastAPI

from services.shared.http_client import create_client, get_timeout, post_typed
from services.shared.logging_util import get_logger, set_trace_id, structured_log_middleware

logger = get_logger(__name__, "orchestrator")
from services.shared.schemas_v1 import (
    AskRequest,
    AskResponse,
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

        # 3) retrieval-service /v1/retrieve with user question; sources come only from here (no stubs)
        retrieval_data, _, _ = await post_typed(
            client,
            _retrieval_url(),
            RetrieveRequest(
                trace_id=trace_id,
                query=request.question,
                top_k=50,
                top_n=8,
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
                "retrieval_query": request.question,
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

    # Stub answer generation (LLM synthesis not implemented yet)
    answer = (
        "This is a stubbed clinical answer based on retrieved context. "
        f"Retrieved {len(sources)} source(s) for your question. "
        "Entities and risk were computed from the pipeline; "
        "full answer synthesis will use an LLM in a later milestone."
    )

    return AskResponse(
        trace_id=trace_id,
        pii_redacted=pii_redacted,
        answer=answer,
        entities=entities,
        sources=sources,
        risk=risk,
    )


if __name__ == "__main__":
    import uvicorn

    # Default to 8010 to match docker-compose and ORCHESTRATOR_URL
    port = int(os.getenv("ORCHESTRATOR_PORT", "8010"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

