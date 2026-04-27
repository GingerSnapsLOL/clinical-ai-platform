import os
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.requests import Request
from pydantic import BaseModel, Field, ValidationError, field_validator

from services.shared.http_client import create_client, get_timeout, post_json
from services.shared.logging_util import get_logger, set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    AskRequest,
    AskResponse,
    ErrorInfo,
    HealthResponse,
    Mode,
)


logger = get_logger(__name__, "gateway-api")


def get_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


def _get_orchestrator_url() -> str:
    url = os.getenv("ORCHESTRATOR_URL") or "http://orchestrator:8010"
    return url.strip().rstrip("/")


def _expose_ask_validation_details() -> bool:
    """When true, HTTP 500 validation failures include Pydantic error payloads (dev only)."""
    return os.getenv("GATEWAY_EXPOSE_ASK_VALIDATION_DETAILS", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _ask_validation_error_response(
    trace_id: str,
    *,
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    body = AskResponse(
        status="error",
        trace_id=trace_id,
        pii_redacted=False,
        answer="",
        warnings=[message],
        error=ErrorInfo(code=code, message=message, details=details),
    )
    return JSONResponse(
        status_code=500,
        content=body.model_dump(mode="json", exclude_none=True),
    )


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


@app.exception_handler(RequestValidationError)
async def request_validation_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 400 for malformed /v1/ask bodies (tests and clients expect 400, not 422)."""
    return JSONResponse(status_code=400, content={"detail": exc.errors()})


@app.on_event("startup")
async def startup() -> None:
    orchestrator_url = _get_orchestrator_url()
    logger.info(
        "gateway_startup",
        extra={
            "service": "gateway-api",
            "orchestrator_url": orchestrator_url,
        },
    )

    # Best-effort connectivity check to orchestrator /health (does not block startup).
    timeout = get_timeout()
    try:
        async with create_client(timeout=timeout) as client:
            resp = await client.get(f"{orchestrator_url}/health", timeout=timeout)
        logger.info(
            "gateway_orchestrator_health_check",
            extra={
                "service": "gateway-api",
                "orchestrator_url": orchestrator_url,
                "orchestrator_health_status": resp.status_code,
            },
        )
    except Exception as exc:
        logger.info(
            "gateway_orchestrator_health_check_failed",
            extra={
                "service": "gateway-api",
                "orchestrator_url": orchestrator_url,
                "error": str(exc),
            },
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="gateway-api")


@app.post(
    "/v1/ask",
    response_model=AskResponse,
    response_model_exclude_none=True,
    responses={
        400: {
            "description": "Invalid request body (validation failed)",
        },
        500: {
            "description": "Orchestrator returned 200 but body failed AskResponse validation (schema drift)",
            "model": AskResponse,
        },
        502: {
            "description": "Orchestrator unreachable or returned an error payload",
            "model": AskResponse,
        },
    },
)
async def ask(request: AskRequestIn) -> Union[AskResponse, JSONResponse]:
    """
    Accept AskRequest-like input (trace_id optional), generate trace_id if missing,
    forward to orchestrator via httpx, return AskResponse.

    Orchestrator JSON is always validated with ``AskResponse.model_validate`` so
    contract drift cannot pass through silently.
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

    orchestrator_url = _get_orchestrator_url()
    url = f"{orchestrator_url}/v1/ask"
    timeout = get_timeout()

    async with create_client(timeout=timeout) as client:
        data, resp, exc = await post_json(
            client,
            url,
            internal_request,
            timeout=timeout,
            trace_id=trace_id,
        )

    # Network failure: no HTTP response
    if exc is not None and resp is None:
        status_code, error_info = _map_error(exc, None)
        error_body = AskResponse(
            status="error",
            trace_id=trace_id,
            pii_redacted=False,
            answer="",
            warnings=[error_info.message],
            error=error_info,
        )
        return JSONResponse(
            status_code=status_code,
            content=error_body.model_dump(mode="json", exclude_none=True),
        )

    # Non-200 from orchestrator
    if resp is not None and resp.status_code != 200:
        status_code, error_info = _map_error(None, resp)
        error_body = AskResponse(
            status="error",
            trace_id=trace_id,
            pii_redacted=False,
            answer="",
            warnings=[error_info.message],
            error=error_info,
        )
        return JSONResponse(
            status_code=status_code,
            content=error_body.model_dump(mode="json", exclude_none=True),
        )

    # 200 but body is not valid JSON
    if exc is not None:
        logger.error(
            "gateway_ask_orchestrator_json_decode_failed",
            extra={"trace_id": trace_id, "error": str(exc), "error_type": type(exc).__name__},
        )
        decode_details: Optional[Dict[str, Any]] = None
        if _expose_ask_validation_details():
            decode_details = {"reason": str(exc), "type": type(exc).__name__}
        return _ask_validation_error_response(
            trace_id,
            code="GATEWAY_ORCHESTRATOR_JSON_INVALID",
            message="Orchestrator returned 200 with a non-JSON or invalid JSON body",
            details=decode_details,
        )

    assert data is not None

    try:
        validated = AskResponse.model_validate(data)
    except ValidationError as e:
        logger.error(
            "gateway_ask_response_validation_failed",
            extra={
                "trace_id": trace_id,
                "error": str(e),
                "pydantic_errors": e.errors(),
            },
        )
        details: Optional[Dict[str, Any]] = None
        if _expose_ask_validation_details():
            details = {"validation_errors": e.errors()}
        return _ask_validation_error_response(
            trace_id,
            code="GATEWAY_RESPONSE_VALIDATION_ERROR",
            message="Orchestrator response failed AskResponse validation",
            details=details,
        )

    return validated


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("GATEWAY_PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

