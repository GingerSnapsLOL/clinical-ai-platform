import os
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from services.shared.http_client import create_client, get_timeout, post_typed
from services.shared.logging_util import set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    AskRequest,
    AskResponse,
    ErrorInfo,
    HealthResponse,
    Mode,
)


def get_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


# ---------------------------------------------------------------------------
# Request: same shape as AskRequest but trace_id optional (gateway generates if missing)
# ---------------------------------------------------------------------------
class AskRequestIn(BaseModel):
    """
    Public input for POST /v1/ask. Same fields as AskRequest; trace_id is optional.
    """

    trace_id: Optional[str] = None
    mode: Mode = "strict"
    note_text: str = Field(..., min_length=1, description="Clinical note text")
    question: str = Field(..., min_length=1, description="User question")
    user_context: Optional[Dict[str, Any]] = None

    @field_validator("trace_id")
    @classmethod
    def trace_id_uuid_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        try:
            UUID(v)
            return v
        except (ValueError, TypeError):
            raise ValueError("trace_id must be a valid UUID")

    @field_validator("note_text", "question", mode="before")
    @classmethod
    def strip_whitespace(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("note_text", "question")
    @classmethod
    def not_empty_after_strip(cls, v: str) -> str:
        if not v:
            raise ValueError("must not be empty")
        return v


# ---------------------------------------------------------------------------
# Error mapping: map failures to (status_code, ErrorInfo)
# ---------------------------------------------------------------------------
def _map_error(exc: Optional[Exception], resp: Optional[httpx.Response]) -> Tuple[int, ErrorInfo]:
    """Map orchestrator/network errors to HTTP status and structured ErrorInfo."""
    if exc is not None:
        return (
            502,
            ErrorInfo(
                code="ORCHESTRATOR_UNREACHABLE",
                message="Failed to reach orchestrator",
                details={"reason": str(exc)},
            ),
        )
    if resp is None:
        return (
            502,
            ErrorInfo(code="ORCHESTRATOR_ERROR", message="No response from orchestrator", details=None),
        )
    if resp.status_code >= 500:
        return (
            502,
            ErrorInfo(
                code="ORCHESTRATOR_5XX",
                message="Orchestrator returned server error",
                details={"status": resp.status_code, "body": resp.text[:500] if resp.text else None},
            ),
        )
    if resp.status_code >= 400:
        return (
            502,
            ErrorInfo(
                code="ORCHESTRATOR_4XX",
                message="Orchestrator rejected request",
                details={"status": resp.status_code, "body": resp.text[:500] if resp.text else None},
            ),
        )
    return (
        502,
        ErrorInfo(code="ORCHESTRATOR_ERROR", message="Unexpected response", details={"status": resp.status_code}),
    )


app = FastAPI(title="Clinical AI Gateway API", version="0.1.0")

app.add_middleware(structured_log_middleware("gateway-api"))
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="gateway-api")


@app.post("/v1/ask", response_model=AskResponse)
async def ask(request: AskRequestIn) -> AskResponse:
    """
    Accept AskRequest-like input (trace_id optional), generate trace_id if missing,
    forward to orchestrator via httpx, return AskResponse.
    """
    trace_id = request.trace_id if request.trace_id else str(uuid4())
    set_trace_id(trace_id)

    internal_request = AskRequest(
        trace_id=trace_id,
        mode=request.mode,
        note_text=request.note_text,
        question=request.question,
        user_context=request.user_context,
    )

    orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8010")
    url = f"{orchestrator_url.rstrip('/')}/v1/ask"
    timeout = get_timeout()

    async with create_client(timeout=timeout) as client:
        result, resp, exc = await post_typed(
            client,
            url,
            internal_request,
            AskResponse,
            timeout=timeout,
            trace_id=trace_id,
        )

    if result is not None:
        return result

    status_code, error_info = _map_error(exc, resp)
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "trace_id": trace_id,
            "error": error_info.model_dump(),
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("GATEWAY_PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

