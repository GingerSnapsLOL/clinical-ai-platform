import os
from typing import List

from fastapi import FastAPI

from services.shared.logging_util import set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    HealthResponse,
    PassageItem,
    RetrieveRequest,
    RetrieveResponse,
)


app = FastAPI(title="Retrieval Service", version="0.1.0")
app.add_middleware(structured_log_middleware("retrieval-service"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="retrieval-service")


@app.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """
    Stub retrieval: returns 2 fake passages.
    Accepts RetrieveRequest, returns RetrieveResponse; preserves trace_id.
    """
    set_trace_id(request.trace_id)

    passages: List[PassageItem] = [
        PassageItem(
            source_id="pubmed:123",
            text=(
                "Hypertension is defined as systolic BP ≥140 mmHg or diastolic BP ≥90 mmHg. "
                "First-line therapy includes ACE inhibitors, thiazide diuretics, or calcium channel blockers."
            ),
            score=0.82,
            metadata={"doc_id": "pubmed:123", "title": "JNC Guideline Excerpt"},
        ),
        PassageItem(
            source_id="pubmed:456",
            text=(
                "Lisinopril is an ACE inhibitor used for hypertension and heart failure. "
                "Common side effects include cough and hyperkalemia; monitor renal function."
            ),
            score=0.76,
            metadata={"doc_id": "pubmed:456", "title": "Drug Monograph Excerpt"},
        ),
    ]

    return RetrieveResponse(
        trace_id=request.trace_id,
        passages=passages,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("RETRIEVAL_SERVICE_PORT", "8040"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

