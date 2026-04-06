"""
ClinicalStructuringAgent: PII redaction then NER on redacted text (sequential, two service calls max).
"""

from __future__ import annotations

import time
from typing import Any

from services.shared.http_client import post_typed
from services.shared.logging_util import get_logger
from services.shared.schemas_v1 import EntityItem, ExtractRequest, ExtractResponse, RedactRequest, RedactResponse

from app.agents.base import AgentResult, AgentRole, SupervisorContext, monotonic_ms
from app.agents.clinical_structuring_agent import ClinicalStructuringAgent

logger = get_logger(__name__, "orchestrator")

# Bounded: exactly two upstream HTTP calls when healthy (PII → NER).
_MAX_REMOTE_CALLS = 2


async def run_clinical_structuring(ctx: SupervisorContext) -> AgentResult:
    t0 = monotonic_ms()
    warnings: list[str] = []
    trace: dict[str, Any] = {
        "bounded_remote_calls": _MAX_REMOTE_CALLS,
        "agent_role": AgentRole.CLINICAL_STRUCTURING,
    }

    redacted_text = ctx.note_text
    pii_redacted = False
    entities: list[EntityItem] = []

    try:
        pii_data, _, _ = await post_typed(
            ctx.client,
            ctx.urls["pii"],
            RedactRequest(trace_id=ctx.trace_id, text=ctx.note_text),
            RedactResponse,
            timeout=ctx.timeout,
            trace_id=ctx.trace_id,
        )
        if pii_data is not None:
            redacted_text = pii_data.redacted_text
            pii_redacted = True
        else:
            warnings.append("pii_service_returned_empty")
    except Exception as exc:
        warnings.append(f"pii_service_error:{type(exc).__name__}")
        logger.warning(
            "clinical_structuring_pii_error",
            extra={"trace_id": ctx.trace_id, "error": str(exc)},
        )

    try:
        ner_data, _, _ = await post_typed(
            ctx.client,
            ctx.urls["ner"],
            ExtractRequest(trace_id=ctx.trace_id, text=redacted_text),
            ExtractResponse,
            timeout=ctx.timeout,
            trace_id=ctx.trace_id,
        )
        if ner_data is not None:
            entities = list(ner_data.entities)
        else:
            warnings.append("ner_service_returned_empty")
    except Exception as exc:
        warnings.append(f"ner_service_error:{type(exc).__name__}")
        logger.warning(
            "clinical_structuring_ner_error",
            extra={"trace_id": ctx.trace_id, "error": str(exc)},
        )

    duration_ms = monotonic_ms() - t0
    trace["entity_count"] = len(entities)
    trace["pii_redacted"] = pii_redacted

    enrich = ClinicalStructuringAgent.enrich(
        redacted_text,
        entities,
        pii_redacted=pii_redacted,
    )
    confidence = float(enrich["structuring_confidence_hint"])
    missing_inputs: list[str] = list(enrich["missing_inputs"])
    if not entities:
        warnings.append("no_entities_extracted")
        confidence = min(confidence, 0.52)

    buckets = enrich["signals"]["clinical_buckets"]
    trace["structuring_symptom_count"] = len(buckets["symptoms"])
    trace["structuring_disease_count"] = len(buckets["diseases"])
    trace["structured_feature_keys"] = len(enrich["structured_features"])

    ok = pii_redacted
    if missing_inputs:
        warnings.extend([f"missing:{m}" for m in missing_inputs])

    payload = {
        "redacted_text": redacted_text,
        "pii_redacted": pii_redacted,
        "entities": [e.model_dump() for e in entities],
        "structured_features": enrich["structured_features"],
        "signals": enrich["signals"],
        "missing_inputs": list(missing_inputs),
    }

    return AgentResult(
        agent_id=AgentRole.CLINICAL_STRUCTURING,
        ok=ok,
        confidence=confidence,
        warnings=warnings,
        missing_inputs=missing_inputs,
        payload=payload,
        error_detail=None if pii_redacted else "pii_failed",
        duration_ms=duration_ms,
        trace=trace,
    )
