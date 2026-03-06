import os
from typing import List

from fastapi import FastAPI

from services.shared.schemas_v1 import (
    EntityItem,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
)


app = FastAPI(title="NER Service", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="ner-service")


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    """
    Stubbed NER extraction.

    Returns a single fake DISEASE entity to exercise the end-to-end flow.
    """
    entities: List[EntityItem] = [
        EntityItem(
            type="DISEASE",
            text="hypertension",
            start=0,
            end=11,
            confidence=0.9,
        )
    ]

    return ExtractResponse(
        trace_id=request.trace_id,
        entities=entities,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("NER_SERVICE_PORT", "8030"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

