import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.shared.schemas_v1 import AskRequest, AskResponse, HealthResponse, Mode


def get_cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    return [o.strip() for o in raw.split(",") if o.strip()]


class GatewayAskIn(BaseModel):
    """
    Public request schema for clients.

    Internally, the gateway augments this with a trace_id and forwards it as
    schemas_v1.AskRequest to the orchestrator.
    """

    mode: Mode = "strict"
    note_text: str
    question: str
    user_context: Optional[Dict[str, Any]] = None


app = FastAPI(title="Clinical AI Gateway API", version="0.1.0")

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
async def ask(request: GatewayAskIn) -> Any:
    """
    Public entrypoint for the Clinical AI Platform.

    For Milestone 0 this endpoint proxies to the orchestrator service, which
    returns a stubbed response.
    """
    orchestrator_url = os.getenv("ORCHESTRATOR_URL", "http://orchestrator:8010")
    url = f"{orchestrator_url.rstrip('/')}/v1/ask"

    internal_request = AskRequest(
        trace_id=str(uuid4()),
        mode=request.mode,
        note_text=request.note_text,
        question=request.question,
        user_context=request.user_context,
    )

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, json=internal_request.model_dump())
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to reach orchestrator: {exc}",
        ) from exc

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Orchestrator error: {resp.text}",
        )

    data = resp.json()
    return AskResponse.model_validate(data)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("GATEWAY_PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

