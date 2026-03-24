"""
Structured JSON logging for Clinical AI Platform services.

Every log entry includes: service name, trace_id (when available),
request path, and status. Machine-readable one-JSON-object-per-line.
"""
import json
import logging
import time
from contextvars import ContextVar
from typing import Any, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# Context var set by endpoints so middleware can include trace_id in logs
trace_id_ctx: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)


def set_trace_id(trace_id: Optional[str]) -> None:
    """Set trace_id for the current request (used by endpoints)."""
    trace_id_ctx.set(trace_id)


def get_trace_id() -> Optional[str]:
    """Get trace_id for the current request."""
    return trace_id_ctx.get()


class JsonLogFormatter(logging.Formatter):
    """Format log records as a single-line JSON object."""

    def __init__(self) -> None:
        super().__init__()

    def format(self, record: logging.LogRecord) -> str:
        log_dict: dict[str, Any] = {
            "timestamp": time.time(),
            "level": record.levelname,
            "message": getattr(record, "message", record.msg) if record.msg else "",
        }
        # Merge extra fields (service, path, status, trace_id, etc.)
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "pathname", "process", "processName", "relativeCreated",
                "stack_info", "exc_info", "exc_text", "thread", "threadName",
                "message", "taskName",
            ) and value is not None:
                log_dict[key] = value
        return json.dumps(log_dict, default=str)


def get_logger(name: str, service_name: str) -> logging.Logger:
    """
    Return a logger that emits JSON. Configure once per service.

    Usage:
        logger = get_logger(__name__, "gateway-api")
        logger.info("request", extra={"path": "/v1/ask", "status": 200, "trace_id": "..."})
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def structured_log_middleware(service_name: str):
    """
    FastAPI/Starlette middleware that logs every request as structured JSON.

    Logged fields: service, path, status, trace_id (when set by endpoint).
    Endpoints that have trace_id should call set_trace_id(trace_id) at the
    start of the handler.
    """

    logger = get_logger(f"clinical_ai.{service_name}", service_name)

    class _Middleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next: Any) -> Response:
            # Clear trace_id for this request (endpoint may set it)
            token = trace_id_ctx.set(None)
            try:
                response = await call_next(request)
                trace_id = get_trace_id()
                logger.info(
                    "request",
                    extra={
                        "service": service_name,
                        "path": request.url.path,
                        "status": response.status_code,
                        "trace_id": trace_id,
                    },
                )
                return response
            finally:
                trace_id_ctx.reset(token)

    return _Middleware
