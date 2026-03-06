import os

from fastapi import FastAPI

from services.shared.schemas_v1 import (
    HealthResponse,
    PIISpan,
    RedactRequest,
    RedactResponse,
)


app = FastAPI(title="PII Redaction Service", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="pii-service")


@app.post("/v1/redact", response_model=RedactResponse)
async def redact(request: RedactRequest) -> RedactResponse:
    """
    Stubbed PII redaction.

    Simply wraps the full text in a placeholder and returns a single fake span.
    """
    fake_span = PIISpan(
        type="NAME",
        start=0,
        end=min(len(request.text), 10),
        replacement="[NAME]",
    )
    return RedactResponse(
        trace_id=request.trace_id,
        redacted_text="[REDACTED TEXT]",
        spans=[fake_span],
    )


if __name__ == "__main__":
    import uvicorn

    # Default to 8020 to match docker-compose and PII_SERVICE_URL
    port = int(os.getenv("PII_SERVICE_PORT", "8020"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

