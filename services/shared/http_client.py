"""
Reusable inter-service HTTP client helpers.

- Timeout configuration (env or argument)
- JSON request/response handling
- Error handling (returns result or response/exception for caller to map)
- trace_id propagation via X-Trace-Id header and request body (caller sets body)
- Typed parsing into Pydantic response models
"""
from __future__ import annotations

import os
from typing import Optional, Tuple, Type, TypeVar

import httpx
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Default timeout; override with env INTER_SERVICE_TIMEOUT (seconds)
DEFAULT_TIMEOUT = float(os.getenv("INTER_SERVICE_TIMEOUT", "30.0"))

# Header for trace_id propagation
TRACE_ID_HEADER = "X-Trace-Id"


def get_timeout() -> float:
    """Return configured inter-service request timeout."""
    return DEFAULT_TIMEOUT


def build_headers(trace_id: Optional[str] = None) -> dict[str, str]:
    """Build request headers with optional trace_id propagation."""
    headers = {"Content-Type": "application/json"}
    if trace_id:
        headers[TRACE_ID_HEADER] = trace_id
    return headers


async def post_typed(
    client: httpx.AsyncClient,
    url: str,
    request_body: BaseModel,
    response_model: Type[T],
    timeout: float = DEFAULT_TIMEOUT,
    trace_id: Optional[str] = None,
) -> Tuple[Optional[T], Optional[httpx.Response], Optional[Exception]]:
    """
    POST JSON body to url, parse response into response_model.

    - request_body: Pydantic model (must include trace_id in body if needed; we also set X-Trace-Id).
    - response_model: Pydantic model class for response.model_validate(json).
    - timeout: request timeout in seconds.
    - trace_id: optional; set on X-Trace-Id header for propagation.

    Returns:
        (parsed_response, http_response, exception).
        On success (200): (parsed_response, http_response, None).
        On HTTP error: (None, http_response, None).
        On network error: (None, None, exception).
    """
    exc: Optional[Exception] = None
    resp: Optional[httpx.Response] = None
    try:
        resp = await client.post(
            url,
            json=request_body.model_dump(),
            headers=build_headers(trace_id),
            timeout=timeout,
        )
    except Exception as e:
        exc = e
        return (None, None, exc)

    if resp.status_code != 200:
        return (None, resp, None)

    try:
        data = resp.json()
        parsed = response_model.model_validate(data)
        return (parsed, resp, None)
    except Exception as e:
        return (None, resp, e)


def create_client(timeout: float = DEFAULT_TIMEOUT) -> httpx.AsyncClient:
    """Create an AsyncClient with the given timeout (for reuse across multiple requests)."""
    return httpx.AsyncClient(timeout=timeout)
