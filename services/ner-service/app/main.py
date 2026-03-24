import os
from typing import List

from fastapi import FastAPI
import time

from services.shared.logging_util import get_logger, set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    EntityItem,
    ExtractRequest,
    ExtractResponse,
    HealthResponse,
)
from app.ner_model import get_nlp, init_scispacy


logger = get_logger(__name__, "ner-service")


app = FastAPI(title="NER Service", version="0.1.0")
app.add_middleware(structured_log_middleware("ner-service"))


_LABEL_MAP = {
    # Direct mappings
    "DISEASE": "DISEASE",
    "CHEMICAL": "CHEMICAL",
    "SYMPTOM": "SYMPTOM",
    # Normalizations / aliases
    "DRUG": "CHEMICAL",
}


@app.on_event("startup")
async def startup() -> None:
    # Load SciSpaCy NER model once at process startup.
    init_scispacy()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="ner-service")


@app.post("/v1/extract", response_model=ExtractResponse)
async def extract(request: ExtractRequest) -> ExtractResponse:
    """
    Run biomedical NER using the SciSpaCy model initialized at startup.
    Accepts ExtractRequest, returns ExtractResponse; preserves trace_id.
    """
    set_trace_id(request.trace_id)

    text = request.text or ""
    text_len = len(text)

    if not text.strip():
        logger.info(
            "ner_extract",
            extra={
                "service": "ner-service",
                "trace_id": request.trace_id,
                "num_entities": 0,
                "processing_ms": 0.0,
                "text_length": text_len,
            },
        )
        return ExtractResponse(trace_id=request.trace_id, entities=[])

    start_time = time.perf_counter()
    nlp = get_nlp()
    doc = nlp(text)

    entities: List[EntityItem] = []
    for ent in doc.ents:
        label = _LABEL_MAP.get(ent.label_, ent.label_)
        entities.append(
            EntityItem(
                type=label,
                text=ent.text,
                start=ent.start_char,
                end=ent.end_char,
                confidence=None,
            )
        )

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    logger.info(
        "ner_extract",
        extra={
            "service": "ner-service",
            "trace_id": request.trace_id,
            "num_entities": len(entities),
            "processing_ms": round(elapsed_ms, 2),
            "text_length": text_len,
        },
    )

    return ExtractResponse(
        trace_id=request.trace_id,
        entities=entities,
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("NER_SERVICE_PORT", "8030"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

