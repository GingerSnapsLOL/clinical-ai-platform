import os

from fastapi import FastAPI

from services.shared.logging_util import set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    FeatureContribution,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)


app = FastAPI(title="Scoring Service", version="0.1.0")
app.add_middleware(structured_log_middleware("scoring-service"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="scoring-service")


@app.post("/v1/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    """
    Stub risk scoring: returns score 0.72 with feature explanation.
    Accepts ScoreRequest, returns ScoreResponse; preserves trace_id.
    """
    set_trace_id(request.trace_id)

    explanation = [
        FeatureContribution(feature="bp_high", contribution=0.18),
        FeatureContribution(feature="age_bucket_60_70", contribution=0.12),
        FeatureContribution(feature="disease_hypertension", contribution=0.08),
    ]

    return ScoreResponse(
        trace_id=request.trace_id,
        score=0.72,
        label="high",
        explanation=explanation,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SCORING_SERVICE_PORT", "8050"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

