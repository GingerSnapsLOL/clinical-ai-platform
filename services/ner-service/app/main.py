import os
from typing import List

from fastapi import FastAPI

from services.shared.logging_util import set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    EntityItem,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
)


app = FastAPI(title="NER Service", version="0.1.0")
app.add_middleware(structured_log_middleware("ner-service"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="ner-service")


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    """
    Stub NER extraction: returns 1–2 fake medical entities.
    Accepts ExtractRequest, returns ExtractResponse; preserves trace_id.
    """
    set_trace_id(request.trace_id)

    entities: List[EntityItem] = [
        EntityItem(
            type="DISEASE",
            text="hypertension",
            start=0,
            end=11,
            confidence=0.92,
        ),
        EntityItem(
            type="CHEMICAL",
            text="lisinopril",
            start=24,
            end=33,
            confidence=0.88,
        ),
    ]

    return ExtractResponse(
        trace_id=request.trace_id,
        entities=entities,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("NER_SERVICE_PORT", "8030"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

