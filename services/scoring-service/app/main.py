import os

from fastapi import FastAPI

from services.shared.schemas_v1 import (
    FeatureContribution,
    HealthResponse,
    RiskBlock,
    ScoreRequest,
    ScoreResponse,
)


app = FastAPI(title="Scoring Service", version="0.1.0")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="scoring-service")


@app.post("/v1/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    """
    Stubbed risk scoring.

    Returns a fixed high-risk score with simple feature contributions.
    """
    explanation = [
        FeatureContribution(feature="bp_high", contribution=0.18),
        FeatureContribution(feature="age_bucket_60_70", contribution=0.12),
    ]

    risk_block = RiskBlock(score=0.72, label="high", explanation=explanation)

    # ScoreResponse is flat; mirror RiskBlock fields.
    return ScoreResponse(
        trace_id=request.trace_id,
        score=risk_block.score,
        label=risk_block.label,
        explanation=risk_block.explanation,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("SCORING_SERVICE_PORT", "8050"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

