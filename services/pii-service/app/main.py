import os
from typing import List
import time
from fastapi import FastAPI

from services.shared.logging_util import get_logger, set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    HealthResponse,
    PIISpan,
    RedactRequest,
    RedactResponse,
)
from app.presidio import get_analyzer, get_anonymizer, init_presidio
from presidio_anonymizer.entities import OperatorConfig


logger = get_logger(__name__, "pii-service")


app = FastAPI(title="PII Redaction Service", version="0.1.0")
app.add_middleware(structured_log_middleware("pii-service"))


@app.on_event("startup")
async def startup() -> None:
    # Load spaCy model and Presidio engines once at process startup.
    init_presidio()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="pii-service")


@app.post("/v1/redact", response_model=RedactResponse)
async def redact(request: RedactRequest) -> RedactResponse:
    """
    Detect and redact PII using Presidio.
    Returns redacted text and spans (type, start, end, replacement, confidence).
    """
    set_trace_id(request.trace_id)

    text = request.text or ""
    text_len = len(text)
    start_time = time.perf_counter()
    if not text.strip():
        return RedactResponse(
            trace_id=request.trace_id,
            redacted_text="",
            spans=[],
        )

    analyzer = get_analyzer()
    anonymizer = get_anonymizer()

    # 1) Detect PII entities
    results = analyzer.analyze(text=text, language="en")

    # 2) Build anonymization operators for selected entity types
    operators = {
        "PERSON": OperatorConfig("replace", {"new_value": "[PERSON]"}),
        "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
        "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
    }

    # 3) Apply anonymization
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators=operators,
    )
    redacted_text = anonymized.text

    # 4) Map results to PIISpan with confidence
    spans: List[PIISpan] = []
    for r in results:
        entity_type = r.entity_type
        replacement = text[r.start : r.end]
        if entity_type == "PERSON":
            replacement = "[PERSON]"
        elif entity_type == "PHONE_NUMBER":
            replacement = "[PHONE]"
        elif entity_type == "EMAIL_ADDRESS":
            replacement = "[EMAIL]"

        spans.append(
            PIISpan(
                type=entity_type,
                start=r.start,
                end=r.end,
                replacement=replacement,
                confidence=float(r.score) if r.score is not None else None,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "pii_redact",
        extra={
            "service": "pii-service",
            "trace_id": request.trace_id,
            "num_entities": len(spans),
            "processing_ms": round(elapsed_ms, 2),
            "text_length": text_len,
        },
    )

    return RedactResponse(
        trace_id=request.trace_id,
        redacted_text=redacted_text,
        spans=spans,
    )


if __name__ == "__main__":
    import uvicorn

    # Default to 8020 to match docker-compose and PII_SERVICE_URL
    port = int(os.getenv("PII_SERVICE_PORT", "8020"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

