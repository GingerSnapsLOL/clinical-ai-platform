import os
from typing import List

from fastapi import FastAPI

from services.shared.schemas_v1 import (
    HealthResponse,
    PassageItem,
    RetrieveRequest,
    RetrieveResponse,
)


app = FastAPI(title="Retrieval Service", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="retrieval-service")


@app.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """
    Stubbed retrieval.

    Returns a single fake passage to exercise the end-to-end flow.
    """
    passages: List[PassageItem] = [
        PassageItem(
            source_id="pubmed:123",
            text="This is a stubbed passage from the local guideline corpus.",
            score=0.77,
            metadata={"doc": "stubbed-document"},
        )
    ]

    return RetrieveResponse(
        trace_id=request.trace_id,
        passages=passages,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("RETRIEVAL_SERVICE_PORT", "8040"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

