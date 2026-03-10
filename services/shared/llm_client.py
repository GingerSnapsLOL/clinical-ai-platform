"""
Reusable HTTP client for talking to llm-service from orchestrator (and others).

Features:
- Async HTTP calls via httpx.AsyncClient
- Reads base URL from LLM_BASE_URL env var (required)
- Typed request/response using Pydantic models
- Timeout handling and error mapping
- Structured logging with trace_id
"""
from __future__ import annotations

import os
from typing import Optional

import httpx
from pydantic import BaseModel, Field

from services.shared.http_client import (
    TRACE_ID_HEADER,
    create_client,
    get_timeout,
)
from services.shared.logging_util import get_logger
from services.shared.schemas_v1 import Status


logger = get_logger(__name__, "orchestrator-llm-client")


class LLMGenerateRequest(BaseModel):
    trace_id: str = Field(..., description="Trace ID propagated from orchestrator")
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.2


class LLMUsage(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class LLMGenerateResponse(BaseModel):
    status: Status
    trace_id: str
    text: str
    usage: LLMUsage | dict = Field(default_factory=dict)


class LLMClient:
    """
    Thin wrapper around httpx.AsyncClient for calling /v1/generate on llm-service.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url or os.getenv("LLM_BASE_URL")
        if not self.base_url:
            raise ValueError("LLM_BASE_URL environment variable must be set or passed explicitly")

        self.timeout = timeout if timeout is not None else get_timeout()
        self._owns_client = client is None
        self.client = client or create_client(timeout=self.timeout)

    async def aclose(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def generate(
        self,
        trace_id: str,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> LLMGenerateResponse:
        """
        Call llm-service /v1/generate and return a typed response.

        Raises httpx.TimeoutException on timeout and httpx.HTTPStatusError
        on non-2xx responses, after logging with trace_id.
        """
        url = f"{self.base_url.rstrip('/')}/v1/generate"

        body = LLMGenerateRequest(
            trace_id=trace_id,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        logger.info(
            "llm_generate_request",
            extra={
                "service": "orchestrator",
                "trace_id": trace_id,
                "url": url,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )

        try:
            resp = await self.client.post(
                url,
                json=body.model_dump(),
                headers={
                    "Content-Type": "application/json",
                    TRACE_ID_HEADER: trace_id,
                },
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            logger.warning(
                "llm_generate_timeout",
                extra={
                    "service": "orchestrator",
                    "trace_id": trace_id,
                    "url": url,
                    "timeout": self.timeout,
                },
            )
            raise exc
        except Exception as exc:
            logger.exception(
                "llm_generate_network_error",
                extra={
                    "service": "orchestrator",
                    "trace_id": trace_id,
                    "url": url,
                },
            )
            raise exc

        if resp.status_code != 200:
            logger.warning(
                "llm_generate_http_error",
                extra={
                    "service": "orchestrator",
                    "trace_id": trace_id,
                    "url": url,
                    "status_code": resp.status_code,
                },
            )
            resp.raise_for_status()

        data = resp.json()
        parsed = LLMGenerateResponse.model_validate(data)

        logger.info(
            "llm_generate_success",
            extra={
                "service": "orchestrator",
                "trace_id": trace_id,
                "url": url,
                "status": parsed.status,
            },
        )

        return parsed

