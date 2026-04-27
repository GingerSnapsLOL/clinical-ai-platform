"""Async agent nodes for evidence-grounded answer synthesis (orchestrator internal)."""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from agent_state import AgentState
from services.shared.llm_client import LLMClient
from services.shared.logging_util import get_logger
from services.shared.schemas_v1 import EntityItem, RiskBlock, SourceItem

logger = get_logger(__name__, "orchestrator")

# Async node: (state) -> updated state
AgentNode = Callable[[AgentState], Awaitable[AgentState]]

_SELECTED_K = 3
_MAX_EVIDENCE_CHARS = 6000
_VERIFICATION_PREFIX = "VERIFICATION_RESULT:"
_VERIFICATION_PASS = f"{_VERIFICATION_PREFIX}pass"
_VERIFICATION_FAIL = f"{_VERIFICATION_PREFIX}fail"

def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s.strip()


def _extract_first_json_object(raw: str) -> str | None:
    """Locate first balanced `{ ... }` substring; returns None if not found."""
    start = raw.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    quote: str | None = None
    for i in range(start, len(raw)):
        c = raw[i]
        if in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == quote:
                in_string = False
                quote = None
            continue
        if c in ("'", '"'):
            in_string = True
            quote = c
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    return None


def parse_llm_json_object(text: str, *, trace_id: str, purpose: str) -> dict[str, Any] | None:
    """
    Parse a single JSON object from model output: full string, fenced, or embedded.
    Returns None on failure after logging.
    """
    raw = text if text else ""
    stripped = _strip_code_fences(raw)
    candidates = [stripped, raw.strip()]
    blob: str | None = None
    for cand in candidates:
        if not cand:
            continue
        try:
            val = json.loads(cand)
            if isinstance(val, dict):
                return val
        except json.JSONDecodeError:
            pass
        extracted = _extract_first_json_object(cand)
        if extracted:
            blob = extracted
            break
    if blob is None and stripped:
        blob = _extract_first_json_object(stripped)
    if blob:
        try:
            val = json.loads(blob)
            if isinstance(val, dict):
                return val
            logger.warning(
                "agent_llm_json_parse_failed",
                extra={
                    "trace_id": trace_id,
                    "purpose": purpose,
                    "error": "not_a_json_object",
                    "snippet": raw[:400],
                },
            )
            return None
        except json.JSONDecodeError as exc:
            logger.warning(
                "agent_llm_json_parse_failed",
                extra={
                    "trace_id": trace_id,
                    "purpose": purpose,
                    "error": str(exc),
                    "snippet": raw[:400],
                },
            )
            return None
    logger.warning(
        "agent_llm_json_parse_failed",
        extra={
            "trace_id": trace_id,
            "purpose": purpose,
            "error": "no_valid_object",
            "snippet": raw[:400],
        },
    )
    return None


def _coerce_bool(val: Any) -> bool | None:
    if val is True or val is False:
        return bool(val)
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    return None


def _coerce_str_list(val: Any) -> list[str]:
    if not isinstance(val, list):
        return []
    out: list[str] = []
    for x in val:
        if x is None:
            continue
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def parse_draft_json_payload(
    text: str,
    *,
    trace_id: str,
    allowed_source_ids: frozenset[str],
) -> tuple[str, list[str]] | None:
    """
    Expect: {"answer": str, "used_source_ids": list[str]}.
    Sanitize IDs to the intersection with ``allowed_source_ids``.
    Returns None if payload is unusable.
    """
    obj = parse_llm_json_object(text, trace_id=trace_id, purpose="draft")
    if obj is None:
        return None
    ans = obj.get("answer")
    if not isinstance(ans, str):
        logger.warning(
            "agent_draft_json_shape",
            extra={"trace_id": trace_id, "field": "answer", "type": type(ans).__name__},
        )
        return None
    answer = ans.strip()
    ids_raw = _coerce_str_list(obj.get("used_source_ids"))
    sanitized = [i for i in ids_raw if i in allowed_source_ids]
    if ids_raw and len(sanitized) < len(ids_raw):
        logger.info(
            "agent_draft_source_ids_filtered",
            extra={
                "trace_id": trace_id,
                "dropped": len(ids_raw) - len(sanitized),
            },
        )
    return answer, sanitized


def parse_verifier_json_payload(
    text: str,
    *,
    trace_id: str,
) -> tuple[bool, bool, list[str]] | None:
    """
    Expect: {"is_grounded": bool, "has_sufficient_evidence": bool, "problems": list[str]}.
    Returns None if required boolean fields are missing or invalid.
    """
    obj = parse_llm_json_object(text, trace_id=trace_id, purpose="verifier")
    if obj is None:
        return None
    g = _coerce_bool(obj.get("is_grounded"))
    s = _coerce_bool(obj.get("has_sufficient_evidence"))
    if g is None or s is None:
        logger.warning(
            "agent_verifier_json_shape",
            extra={
                "trace_id": trace_id,
                "is_grounded": obj.get("is_grounded"),
                "has_sufficient_evidence": obj.get("has_sufficient_evidence"),
            },
        )
        return None
    problems = _coerce_str_list(obj.get("problems"))
    return g, s, problems


def _strip_verification_warnings(warnings: list[str]) -> list[str]:
    return [w for w in warnings if not w.startswith(_VERIFICATION_PREFIX)]


def _last_verification_passed(state: AgentState) -> bool:
    for w in reversed(state.warnings):
        if w.startswith(_VERIFICATION_PREFIX):
            return w == _VERIFICATION_PASS
    return False


def _sort_sources_by_score(sources: list[SourceItem]) -> list[SourceItem]:
    return sorted(
        sources,
        key=lambda s: float(s.score) if s.score is not None else 0.0,
        reverse=True,
    )


def _format_evidence_passages(sources: list[SourceItem]) -> str:
    """Build a single evidence block (top passages already selected upstream)."""
    lines: list[str] = ["Top evidence passages:"]
    for idx, s in enumerate(sources, start=1):
        title = s.title or (s.metadata.get("title") if s.metadata else None)
        header = f"Passage {idx} (source_id={s.source_id})"
        if title:
            header += f" - {title}"
        lines.append(header + ":")
        lines.append(s.snippet or "")
        lines.append("")
    text = "\n".join(lines)
    if len(text) > _MAX_EVIDENCE_CHARS:
        text = text[:_MAX_EVIDENCE_CHARS]
    return text


def _allowed_source_id_lines(selected: list[SourceItem]) -> str:
    return "\n".join(f"- {s.source_id}" for s in selected)


def _entities_block(entities: list[EntityItem]) -> str:
    if not entities:
        return ""
    lines = ["Extracted entities (auxiliary — do not state facts unless also in passages):"]
    for e in entities[:20]:
        lines.append(f"- {e.type}: {e.text}")
    return "\n".join(lines)


def _risk_block(risk: RiskBlock | None) -> str:
    if (
        risk is None
        or not risk.risk_available
        or risk.label is None
        or risk.score is None
    ):
        return ""
    lines = [
        "Risk assessment (auxiliary — not evidence; do not use unless same facts appear in passages):",
        f"- label: {risk.label}, score: {risk.score:.2f}",
    ]
    return "\n".join(lines)


def _build_draft_prompt(state: AgentState) -> str:
    evidence = _format_evidence_passages(state.selected_sources)
    id_list = _allowed_source_id_lines(state.selected_sources)
    parts = [
        "You are a clinical decision support assistant. Answer using ONLY the text under "
        "'Top evidence passages.' Do not use outside knowledge.",
        "",
        f"Question:\n{state.question}",
        "",
        _entities_block(state.entities),
        "",
        _risk_block(state.risk),
        "",
        evidence,
        "",
        "Valid source_id values (each used_source_ids entry must be exactly one of these):",
        id_list,
        "",
        "Output requirements:\n"
        "- Reply with ONE JSON object only. No markdown, no prose before or after.\n"
        "- Keys exactly: \"answer\" (string), \"used_source_ids\" (JSON array of strings).\n"
        "- Every string in used_source_ids must be a source_id from the list above that you "
        "actually relied on for factual content in answer.\n"
        '- If you cannot answer from the passages, set answer to exactly \"Insufficient data\" '
        "and used_source_ids to [].",
    ]
    return "\n".join(parts)


def _build_verifier_prompt(state: AgentState) -> str:
    evidence = _format_evidence_passages(state.selected_sources)
    draft = state.draft_answer or ""
    used_ids = state.draft_used_source_ids
    parts = [
        "You verify grounding of a draft answer against evidence passages only.",
        "Reply with ONE JSON object only. No markdown, no prose before or after.",
        'Keys exactly: \"is_grounded\" (boolean), \"has_sufficient_evidence\" (boolean), '
        '"problems" (JSON array of short strings describing issues, or []).',
        "is_grounded: every factual claim in the draft is directly supported by the passages.",
        "has_sufficient_evidence: passages contain enough material to justify the draft as written.",
        "",
        f"Question:\n{state.question}",
        "",
        f"Draft used_source_ids (for reference): {json.dumps(used_ids)}",
        "",
        "--- EVIDENCE (only source of truth) ---",
        evidence,
        "",
        "--- DRAFT ANSWER ---",
        draft,
    ]
    return "\n".join(parts)


async def evidence_selector_node(state: AgentState) -> AgentState:
    """Keep up to the best ``_SELECTED_K`` sources by retrieval score."""
    if not state.sources:
        logger.info(
            "agent_evidence_selector_empty",
            extra={"trace_id": state.trace_id, "selected": 0},
        )
        return state.evolve(selected_sources=[])

    ranked = _sort_sources_by_score(list(state.sources))
    picked = ranked[:_SELECTED_K]
    logger.info(
        "agent_evidence_selector",
        extra={
            "trace_id": state.trace_id,
            "selected": len(picked),
            "source_ids": [s.source_id for s in picked],
        },
    )
    return state.evolve(selected_sources=picked)


async def draft_answer_node(state: AgentState) -> AgentState:
    """Call LLM to produce a strictly grounded draft from ``selected_sources`` (JSON)."""
    if not state.selected_sources:
        logger.info(
            "agent_draft_skip_no_evidence",
            extra={"trace_id": state.trace_id},
        )
        return state.evolve(draft_answer="Insufficient data", draft_used_source_ids=[])

    allowed = frozenset(s.source_id for s in state.selected_sources)
    llm: LLMClient | None = None
    try:
        llm = LLMClient()
        prompt = _build_draft_prompt(state)
        resp = await llm.generate(
            trace_id=state.trace_id,
            prompt=prompt,
            max_tokens=512,
            temperature=0.2,
        )
        text = resp.text or ""
        parsed = parse_draft_json_payload(
            text,
            trace_id=state.trace_id,
            allowed_source_ids=allowed,
        )
        if parsed is None:
            logger.warning(
                "agent_draft_fallback_plaintext",
                extra={"trace_id": state.trace_id},
            )
            stripped = text.strip()
            if stripped == "Insufficient data" or not stripped:
                return state.evolve(draft_answer="Insufficient data", draft_used_source_ids=[])
            first_line = stripped.splitlines()[0].strip()
            if first_line == "Insufficient data":
                return state.evolve(draft_answer="Insufficient data", draft_used_source_ids=[])
            return state.evolve(draft_answer=stripped[:8000], draft_used_source_ids=[])

        answer, used_ids = parsed
        if not answer:
            answer = "Insufficient data"
        logger.info(
            "agent_draft_ok",
            extra={
                "trace_id": state.trace_id,
                "draft_len": len(answer),
                "used_source_ids_count": len(used_ids),
            },
        )
        return state.evolve(draft_answer=answer, draft_used_source_ids=used_ids)
    except (httpx.HTTPError, httpx.RequestError, ValueError) as exc:
        logger.warning(
            "agent_draft_error",
            extra={"trace_id": state.trace_id, "error": str(exc)},
        )
        return state.evolve(
            draft_answer="Insufficient data",
            draft_used_source_ids=[],
            warnings=[*state.warnings, "agent:draft_llm_error"],
        )
    finally:
        if llm is not None:
            await llm.aclose()


async def answer_verifier_node(state: AgentState) -> AgentState:
    """Ask LLM whether the draft is grounded (structured JSON)."""
    draft = (state.draft_answer or "").strip()
    ws_base = _strip_verification_warnings(list(state.warnings))

    if draft == "Insufficient data" or not draft:
        logger.info(
            "agent_verifier_skip",
            extra={"trace_id": state.trace_id, "reason": "no_substantive_draft"},
        )
        return state.evolve(warnings=[*ws_base, _VERIFICATION_PASS])

    if not state.selected_sources:
        return state.evolve(warnings=[*ws_base, _VERIFICATION_FAIL])

    llm: LLMClient | None = None
    try:
        llm = LLMClient()
        prompt = _build_verifier_prompt(state)
        resp = await llm.generate(
            trace_id=state.trace_id,
            prompt=prompt,
            max_tokens=256,
            temperature=0.0,
        )
        parsed = parse_verifier_json_payload(resp.text or "", trace_id=state.trace_id)
        if parsed is None:
            logger.warning(
                "agent_verifier_parse_fail",
                extra={"trace_id": state.trace_id, "snippet": (resp.text or "")[:200]},
            )
            supported = False
            problems: list[str] = ["parse_failure"]
        else:
            g, s, problems = parsed
            supported = bool(g and s)
            if problems:
                logger.info(
                    "agent_verifier_problems",
                    extra={
                        "trace_id": state.trace_id,
                        "problems": problems[:5],
                        "supported": supported,
                    },
                )

        result = _VERIFICATION_PASS if supported else _VERIFICATION_FAIL
        logger.info(
            "agent_verifier_done",
            extra={"trace_id": state.trace_id, "supported": supported},
        )
        return state.evolve(warnings=[*ws_base, result])
    except (httpx.HTTPError, httpx.RequestError, ValueError) as exc:
        logger.warning(
            "agent_verifier_error",
            extra={"trace_id": state.trace_id, "error": str(exc)},
        )
        return state.evolve(
            warnings=[
                *ws_base,
                _VERIFICATION_FAIL,
                "agent:verifier_llm_error",
            ],
        )
    finally:
        if llm is not None:
            await llm.aclose()


async def finalize_answer_node(state: AgentState) -> AgentState:
    """Emit final answer or Insufficient data if verification failed."""
    draft = (state.draft_answer or "").strip()
    ok = _last_verification_passed(state)

    if ok and draft and draft != "Insufficient data":
        logger.info(
            "agent_finalize_verified",
            extra={
                "trace_id": state.trace_id,
                "final_len": len(draft),
                "cited_sources": len(state.draft_used_source_ids),
            },
        )
        return state.evolve(final_answer=draft, needs_retry=False)

    logger.info(
        "agent_finalize_insufficient",
        extra={"trace_id": state.trace_id, "had_draft": bool(draft)},
    )
    return state.evolve(final_answer="Insufficient data", needs_retry=False)
