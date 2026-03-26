import os
import re
from typing import Dict, Protocol

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

from services.shared.logging_util import get_logger, set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import HealthResponse, Status


logger = get_logger(__name__, "llm-service")
MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_DEFAULT_MAX_TOKENS", "192"))
MAX_TOKENS_CAP = int(os.getenv("LLM_MAX_TOKENS_CAP", "256"))
MIN_MAX_TOKENS = 32
DEFAULT_TEMPERATURE = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.0"))
DEFAULT_DO_SAMPLE = os.getenv("LLM_DEFAULT_DO_SAMPLE", "false").strip().lower() in ("1", "true", "yes", "on")
DEFAULT_TOP_P = float(os.getenv("LLM_DEFAULT_TOP_P", "1.0"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("LLM_DEFAULT_REPETITION_PENALTY", "1.05"))
PROMPT_MAX_CHARS = int(os.getenv("LLM_PROMPT_MAX_CHARS", "7000"))
LLM_BACKEND = os.getenv("LLM_BACKEND", "transformers").strip().lower()
LLM_BACKEND_VLLM_ENGINE = os.getenv("LLM_BACKEND_VLLM_ENGINE", "")
LLM_BACKEND_VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("LLM_BACKEND_VLLM_TENSOR_PARALLEL_SIZE", "1"))
# Optional vLLM tuning hooks. Empty means "use vLLM defaults".
# - LLM_BACKEND_VLLM_MAX_MODEL_LEN: positive integer context length cap
# - LLM_BACKEND_VLLM_GPU_MEMORY_UTILIZATION: float in (0, 1]
# - LLM_BACKEND_VLLM_TRUST_REMOTE_CODE: bool (default false)
LLM_BACKEND_VLLM_MAX_MODEL_LEN = os.getenv("LLM_BACKEND_VLLM_MAX_MODEL_LEN", "").strip()
LLM_BACKEND_VLLM_GPU_MEMORY_UTILIZATION = os.getenv("LLM_BACKEND_VLLM_GPU_MEMORY_UTILIZATION", "").strip()
LLM_BACKEND_VLLM_TRUST_REMOTE_CODE = os.getenv("LLM_BACKEND_VLLM_TRUST_REMOTE_CODE", "false").strip().lower()


def _parse_bool_env(name: str, raw: str) -> bool:
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be a boolean (true/false), got {raw!r}")


def _parse_optional_positive_int_env(name: str, raw: str) -> int | None:
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _parse_optional_unit_float_env(name: str, raw: str) -> float | None:
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {raw!r}") from exc
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in (0, 1], got {value}")
    return value


class GenerateConfig(BaseModel):
    max_new_tokens: int
    temperature: float
    do_sample: bool
    top_p: float
    repetition_penalty: float


class NormalizedGenerationResult(BaseModel):
    """
    Unified backend output for /v1/generate.

    text: completion text only (no prompt prefix).
    usage: normalized token counters exposed by API contract.
    usage_exact_*: internal metadata for observability; not returned to clients.
    """
    text: str
    usage: Dict[str, int]
    usage_exact_prompt: bool = False
    usage_exact_completion: bool = False


class BackendGenerationError(RuntimeError):
    """Raised by backends so handler can map to one error shape."""


class LLMBackend(Protocol):
    backend_name: str

    def load_backend(self) -> None:
        ...

    def generate_text(self, prompt: str, cfg: GenerateConfig) -> NormalizedGenerationResult:
        ...


def _estimate_token_count(text: str) -> int:
    # Lightweight fallback when backend does not expose token ids.
    return max(1, len((text or "").split())) if (text or "").strip() else 0


def _normalize_usage(prompt_tokens: int, completion_tokens: int) -> Dict[str, int]:
    p = max(0, int(prompt_tokens))
    c = max(0, int(completion_tokens))
    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": p + c,
    }


class TransformersBackend:
    backend_name = "transformers"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def load_backend(self) -> None:
        tokenizer_local = AutoTokenizer.from_pretrained(self.model_name)
        torch_dtype = torch.float16 if torch.cuda.is_available() else None
        model_local = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.tokenizer = tokenizer_local
        self.model = model_local

    def generate_text(self, prompt: str, cfg: GenerateConfig) -> NormalizedGenerationResult:
        if self.tokenizer is None or self.model is None:
            raise BackendGenerationError("Transformers backend is not loaded")

        chat_tmpl = getattr(self.tokenizer, "chat_template", None)
        apply_tmpl = getattr(self.tokenizer, "apply_chat_template", None)
        if isinstance(chat_tmpl, str) and chat_tmpl.strip() and callable(apply_tmpl):
            messages = [{"role": "user", "content": prompt}]
            prompt_text = apply_tmpl(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = self.tokenizer(
                prompt_text,
                return_tensors="pt",
                add_special_tokens=False,
            ).to(self.model.device)
        else:
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
            ).to(self.model.device)

        generation_kwargs = {
            "max_new_tokens": cfg.max_new_tokens,
            "temperature": cfg.temperature,
            "do_sample": cfg.do_sample,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        try:
            with torch.no_grad():
                output = self.model.generate(**input_ids, **generation_kwargs)
            generated_ids = output[0][input_ids["input_ids"].shape[-1] :]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            usage = _normalize_usage(
                prompt_tokens=int(input_ids["input_ids"].numel()),
                completion_tokens=int(generated_ids.numel()),
            )
            return NormalizedGenerationResult(
                text=text,
                usage=usage,
                usage_exact_prompt=True,
                usage_exact_completion=True,
            )
        except Exception as exc:
            raise BackendGenerationError(f"transformers generate_text failed: {exc}") from exc


class VLLMBackend:
    backend_name = "vllm"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.engine = None
        self.SamplingParams = None

    def load_backend(self) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "vLLM backend selected but vllm is not available. Install vllm and dependencies."
            ) from exc

        if LLM_BACKEND_VLLM_TENSOR_PARALLEL_SIZE <= 0:
            raise ValueError(
                "LLM_BACKEND_VLLM_TENSOR_PARALLEL_SIZE must be > 0, "
                f"got {LLM_BACKEND_VLLM_TENSOR_PARALLEL_SIZE}"
            )

        max_model_len = _parse_optional_positive_int_env(
            "LLM_BACKEND_VLLM_MAX_MODEL_LEN",
            LLM_BACKEND_VLLM_MAX_MODEL_LEN,
        )
        gpu_memory_utilization = _parse_optional_unit_float_env(
            "LLM_BACKEND_VLLM_GPU_MEMORY_UTILIZATION",
            LLM_BACKEND_VLLM_GPU_MEMORY_UTILIZATION,
        )
        trust_remote_code = _parse_bool_env(
            "LLM_BACKEND_VLLM_TRUST_REMOTE_CODE",
            LLM_BACKEND_VLLM_TRUST_REMOTE_CODE,
        )

        # Keep startup loading centralized: construct engine once here.
        # LLM_BACKEND_VLLM_ENGINE can point to a local model path in future deployments.
        model_ref = LLM_BACKEND_VLLM_ENGINE.strip() or self.model_name
        llm_kwargs: Dict[str, object] = {
            "model": model_ref,
            "tensor_parallel_size": LLM_BACKEND_VLLM_TENSOR_PARALLEL_SIZE,
            "trust_remote_code": trust_remote_code,
        }
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len
        if gpu_memory_utilization is not None:
            llm_kwargs["gpu_memory_utilization"] = gpu_memory_utilization

        self.engine = LLM(**llm_kwargs)
        self.SamplingParams = SamplingParams
        logger.info(
            "llm_backend_vllm_loaded",
            extra={
                "service": "llm-service",
                "model_name": self.model_name,
                "vllm_engine": model_ref,
                "tensor_parallel_size": LLM_BACKEND_VLLM_TENSOR_PARALLEL_SIZE,
                "max_model_len": max_model_len,
                "gpu_memory_utilization": gpu_memory_utilization,
                "trust_remote_code": trust_remote_code,
            },
        )

    def generate_text(self, prompt: str, cfg: GenerateConfig) -> NormalizedGenerationResult:
        if self.engine is None or self.SamplingParams is None:
            raise BackendGenerationError("vLLM backend is not loaded")

        sampling_params = self.SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_new_tokens,
            repetition_penalty=cfg.repetition_penalty,
        )
        try:
            outputs = self.engine.generate([prompt], sampling_params)
            first = outputs[0]
            first_completion = first.outputs[0] if first.outputs else None
            text = first_completion.text if first_completion is not None else ""

            prompt_ids = getattr(first, "prompt_token_ids", None)
            completion_ids = getattr(first_completion, "token_ids", None) if first_completion else None

            prompt_exact = bool(prompt_ids is not None)
            completion_exact = bool(completion_ids is not None)
            prompt_tokens = len(prompt_ids or []) if prompt_exact else _estimate_token_count(prompt)
            completion_tokens = len(completion_ids or []) if completion_exact else _estimate_token_count(text)
            usage = _normalize_usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)

            return NormalizedGenerationResult(
                text=text,
                usage=usage,
                usage_exact_prompt=prompt_exact,
                usage_exact_completion=completion_exact,
            )
        except Exception as exc:
            raise BackendGenerationError(f"vllm generate_text failed: {exc}") from exc


# Global backend, loaded once at startup
backend: LLMBackend | None = None


class GenerateRequest(BaseModel):
    trace_id: str = Field(..., description="UUID for request trace; propagated from gateway/orchestrator")
    prompt: str
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE


class GenerateResponse(BaseModel):
    status: Status = "ok"
    trace_id: str
    text: str
    usage: Dict[str, int] = {}


app = FastAPI(title="LLM Service", version="0.1.0")
app.add_middleware(structured_log_middleware("llm-service"))


def _sanitize_prompt(prompt: str) -> str:
    """
    Keep prompt compact to reduce generation latency.
    """
    text = (prompt or "").strip()
    if not text:
        return text

    # Collapse excessive blank lines and trailing spaces.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text[:PROMPT_MAX_CHARS]


def _sanitize_output(text: str, prompt: str) -> str:
    """
    Remove common prompt leakage and unrequested markdown headings.
    """
    out = (text or "").strip()
    if not out:
        return out

    leaked_prefixes = (
        "Question:",
        "Top evidence passages:",
        "Extracted entities:",
        "Risk prediction block",
        "You are a clinical decision support assistant",
    )
    for prefix in leaked_prefixes:
        if out.startswith(prefix):
            idx = out.find("\n\n")
            if idx != -1:
                out = out[idx + 2 :].lstrip()

    prompt_lower = (prompt or "").lower()
    if "heading" not in prompt_lower and "section" not in prompt_lower:
        lines = [ln for ln in out.splitlines() if not re.match(r"^\s{0,3}#{1,6}\s+\S+", ln)]
        out = "\n".join(lines).strip()

    return out


def load_backend(model_name: str, backend_name: str) -> LLMBackend:
    """Centralized backend loader for startup; selected by env."""
    selected = backend_name.strip().lower()
    if selected == "transformers":
        b = TransformersBackend(model_name)
    elif selected == "vllm":
        b = VLLMBackend(model_name)
    else:
        raise ValueError(f"Unsupported LLM_BACKEND={backend_name!r}; expected 'transformers' or 'vllm'")
    b.load_backend()
    return b


@app.on_event("startup")
async def startup() -> None:
    """
    Load Qwen2.5-7B-Instruct once at process startup.
    Uses float16 when CUDA is available and device_map=\"auto\" for placement.
    """
    global backend

    logger.info(
        "llm_startup_begin",
        extra={
            "service": "llm-service",
            "model_name": MODEL_NAME,
            "backend": LLM_BACKEND,
        },
    )

    try:
        backend = load_backend(MODEL_NAME, LLM_BACKEND)

        logger.info(
            "llm_startup_success",
            extra={
                "service": "llm-service",
                "model_name": MODEL_NAME,
                "backend": backend.backend_name,
                "cuda_available": torch.cuda.is_available(),
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
    LLM generation endpoint. Public API stays stable while backend is swappable.
    """
    set_trace_id(request.trace_id)

    if backend is None:
        # Should not happen in normal operation; startup should have failed fast.
        raise RuntimeError("LLM backend is not loaded")

    raw_prompt = _sanitize_prompt(request.prompt)

    max_new_tokens = max(MIN_MAX_TOKENS, min(int(request.max_tokens), MAX_TOKENS_CAP))
    temperature = float(request.temperature)
    # Conservative defaults for low-latency, deterministic clinical answers.
    do_sample = DEFAULT_DO_SAMPLE
    top_p = DEFAULT_TOP_P
    repetition_penalty = DEFAULT_REPETITION_PENALTY
    if do_sample:
        temperature = max(0.0, temperature)
    else:
        temperature = DEFAULT_TEMPERATURE

    cfg = GenerateConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=do_sample,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )
    try:
        result = backend.generate_text(raw_prompt, cfg)
    except BackendGenerationError:
        # Unified service-level error shape irrespective of backend implementation.
        raise RuntimeError("LLM generation failed")

    text = _sanitize_output(result.text, raw_prompt)
    usage = result.usage
    logger.info(
        "llm_usage_normalized",
        extra={
            "trace_id": request.trace_id,
            "backend": backend.backend_name,
            "usage_exact_prompt": result.usage_exact_prompt,
            "usage_exact_completion": result.usage_exact_completion,
        },
    )

    return GenerateResponse(
        trace_id=request.trace_id,
        text=text,
        usage=usage,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("LLM_SERVICE_PORT", "8060"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

