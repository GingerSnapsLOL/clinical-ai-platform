import os
from typing import Dict

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from services.shared.logging_util import get_logger, set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import HealthResponse, Status


logger = get_logger(__name__, "llm-service")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

# Global model/tokenizer, loaded once at startup
tokenizer = None
model = None


class GenerateRequest(BaseModel):
    trace_id: str = Field(..., description="UUID for request trace; propagated from gateway/orchestrator")
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.2


class GenerateResponse(BaseModel):
    status: Status = "ok"
    trace_id: str
    text: str
    usage: Dict[str, int] = {}


app = FastAPI(title="LLM Service", version="0.1.0")
app.add_middleware(structured_log_middleware("llm-service"))


@app.on_event("startup")
async def startup() -> None:
    """
    Load Qwen2.5-7B-Instruct once at process startup.
    Uses float16 when CUDA is available and device_map=\"auto\" for placement.
    """
    global tokenizer, model

    logger.info(
        "llm_startup_begin",
        extra={
            "service": "llm-service",
            "model_name": MODEL_NAME,
        },
    )

    try:
        tokenizer_local = AutoTokenizer.from_pretrained(MODEL_NAME)

        torch_dtype = torch.float16 if torch.cuda.is_available() else None

        model_local = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

        tokenizer = tokenizer_local
        model = model_local

        logger.info(
            "llm_startup_success",
            extra={
                "service": "llm-service",
                "model_name": MODEL_NAME,
                "cuda_available": torch.cuda.is_available(),
                "dtype": str(torch_dtype) if torch_dtype is not None else "default",
            },
        )
    except Exception as exc:
        logger.exception(
            "llm_startup_failure",
            extra={
                "service": "llm-service",
                "model_name": MODEL_NAME,
                "error": str(exc),
            },
        )
        raise


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="llm-service")


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    LLM generation endpoint backed by Qwen2.5-7B-Instruct.

    The model is loaded once at process startup; this handler only performs
    tokenization, generation, and decoding.
    """
    set_trace_id(request.trace_id)

    if tokenizer is None or model is None:
        # Should not happen in normal operation; startup should have failed fast.
        raise RuntimeError("LLM model is not loaded")

    # Build a simple chat-style prompt for Qwen Instruct
    prompt_text = request.prompt

    input_ids = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to(model.device)

    generation_kwargs = {
        "max_new_tokens": request.max_tokens,
        "temperature": float(request.temperature),
        "do_sample": request.temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }

    with torch.no_grad():
        output = model.generate(**input_ids, **generation_kwargs)

    generated_ids = output[0][input_ids["input_ids"].shape[-1] :]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    usage = {
        "prompt_tokens": int(input_ids["input_ids"].numel()),
        "completion_tokens": int(generated_ids.numel()),
        "total_tokens": int(input_ids["input_ids"].numel() + generated_ids.numel()),
    }

    return GenerateResponse(
        trace_id=request.trace_id,
        text=text,
        usage=usage,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("LLM_SERVICE_PORT", "8060"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

