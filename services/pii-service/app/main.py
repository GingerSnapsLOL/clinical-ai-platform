import os
from typing import List

from fastapi import FastAPI

from services.shared.logging_util import set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    HealthResponse,
    PIISpan,
    RedactRequest,
    RedactResponse,
)


app = FastAPI(title="PII Redaction Service", version="0.1.0")
app.add_middleware(structured_log_middleware("pii-service"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="pii-service")


@app.post("/v1/redact", response_model=RedactResponse)
async def redact(request: RedactRequest) -> RedactResponse:
    """
    Stub PII redaction: returns redacted text with placeholder spans.
    Accepts RedactRequest, returns RedactResponse; preserves trace_id.
    """
    set_trace_id(request.trace_id)

    # Realistic placeholder: redacted sentence + 1–2 spans
    text = request.text or ""
    spans: List[PIISpan] = [
        PIISpan(type="NAME", start=0, end=8, replacement="[PATIENT]"),
        PIISpan(type="DATE", start=22, end=32, replacement="[DATE]"),
    ]
    redacted_text = (
        "[PATIENT] presented on [DATE] with history of hypertension. "
        "Stub redaction applied (original length {} chars)."
    ).format(len(text)) if text else "[REDACTED TEXT]"

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

