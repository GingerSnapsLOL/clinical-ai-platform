from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app import engine
from app.config import settings
from app.targets import valid_target_ids
from services.shared.logging_util import structured_log_middleware
from services.shared.schemas_v1 import HealthResponse, ScoreRequest, ScoreResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="Scoring Service", version="0.1.0", lifespan=lifespan)
app.add_middleware(structured_log_middleware("scoring-service"))


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="scoring-service")


@app.post("/v1/score", response_model=ScoreResponse)
async def score(request: ScoreRequest) -> ScoreResponse:
    """Multi-target triage: deterministic rules (no ML weights)."""
    if request.targets:
        unknown = [t for t in request.targets if t not in valid_target_ids()]
        if unknown:
            raise HTTPException(
                status_code=422,
                detail={"error": "unknown_target", "targets": unknown},
            )
    return engine.compute_score(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=settings.scoring_service_port,
        reload=True,
    )
