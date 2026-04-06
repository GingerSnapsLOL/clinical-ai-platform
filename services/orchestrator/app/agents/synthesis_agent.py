"""
SynthesisAgent: grounded final answers from evidence, structured features, scores, and safety.

One LLM call (via :func:`run_synthesis`). Prompt enforces grounding, uncertainty, and
non-hallucination; output payload includes answer, cited sources snapshot, and confidence.
"""

from __future__ import annotations

import json
from typing import Any

from services.shared.schemas_v1 import EntityItem, RiskBlock, SourceItem

from app.agents.base import AgentResult, AgentRole, SupervisorContext
from app.agents.synthesis import run_synthesis
from app.prompts.llm_ask import build_llm_prompt


_MAX_STRUCT_CHARS = 1400
_MAX_EXTRA_TARGETS_LINES = 12


def _truncate_val(v: Any, max_len: int = 200) -> str:
    if isinstance(v, (dict, list)):
        s = json.dumps(v, default=str)[:max_len]
    else:
        s = str(v)[:max_len]
    return s


def _format_structured_features(sf: dict[str, Any]) -> str:
    if not sf:
        return ""
    lines = [
        "Structured features extracted from the note (auxiliary context only — not evidence text; "
        "do not treat as facts unless the same appears in 'Top evidence passages'):",
    ]
    for i, (k, v) in enumerate(sorted(sf.items())):
        if i >= 48:
            lines.append("- … (truncated)")
            break
        lines.append(f"- {k}: {_truncate_val(v)}")
    text = "\n".join(lines)
    return text[:_MAX_STRUCT_CHARS]


def _format_scores_aux(scores: dict[str, Any] | None) -> str:
    if not scores:
        return ""
    primary = scores.get("primary")
    lines = [
        "Score summary (auxiliary; risk model output — do not state as clinical facts unless "
        "supported verbatim or by clear paraphrase in Top evidence passages):",
    ]
    if isinstance(primary, dict):
        lines.append(
            f"- Primary target ({primary.get('target', 'triage')}): "
            f"label={primary.get('label')}, score={primary.get('score')}"
        )
    all_scores = scores.get("scores")
    if isinstance(all_scores, dict) and len(all_scores) > 1:
        lines.append("- Additional targets:")
        for tid, row in list(all_scores.items())[:_MAX_EXTRA_TARGETS_LINES]:
            if not isinstance(row, dict):
                continue
            lines.append(
                f"  - {tid}: label={row.get('label')} score={row.get('score')} ready={row.get('ready')}"
            )
    return "\n".join(lines)


def _format_safety(safety: dict[str, Any] | None) -> str:
    if not safety:
        return ""
    level = safety.get("safety_level", "normal")
    actions = safety.get("actions") or []
    lines = [
        "Safety / policy context (tone and urgency — not a source of clinical facts):",
        f"- safety_level: {level}",
        "(A compliance prefix may be applied server-side to the final user-visible answer; "
        "do not repeat that boilerplate verbatim in your completion.)",
    ]
    lines.append(f"- required_actions: {', '.join(str(a) for a in actions)}")
    lines.append(
        "- Do not provide a definitive diagnosis. Acknowledge uncertainty when evidence is thin."
    )
    return "\n".join(lines)


def _uncertainty_grounding_addon() -> str:
    return (
        "\n\nAdditional synthesis rules (required):\n"
        "- Be structured and readable: use short sections or bullets when helpful.\n"
        "- Explicitly state uncertainty, limitations, or gaps when passages are sparse, ambiguous, "
        "or only partially address the question.\n"
        "- Where you cite content, tie it to passage/source context mentally; do not fabricate "
        "source_ids or quotes.\n"
        "- Prefer cautious language ('suggests', 'may', 'based only on the passages above').\n"
    )


def build_synthesis_prompt(
    *,
    question: str,
    entities: list[EntityItem],
    sources: list[SourceItem],
    structured_features: dict[str, Any],
    scores: dict[str, Any] | None,
    safety: dict[str, Any] | None,
    risk: RiskBlock | None,
    trace_id: str,
) -> str:
    """Compose full prompt: policy + structured/scores + core grounded prompt + uncertainty rules."""
    header_parts: list[str] = []
    s_block = _format_safety(safety)
    if s_block:
        header_parts.append(s_block)
    st_block = _format_structured_features(structured_features)
    if st_block:
        header_parts.append(st_block)
    sc_block = _format_scores_aux(scores)
    if sc_block:
        header_parts.append(sc_block)

    core = build_llm_prompt(
        question=question,
        entities=entities,
        sources=sources,
        risk=risk,
        trace_id=trace_id,
    )
    assembled = "\n\n".join(header_parts)
    if assembled:
        assembled += "\n\n---\n\n"
    assembled += core + _uncertainty_grounding_addon()
    return assembled


def _evidence_strength(sources: list[SourceItem]) -> float:
    if not sources:
        return 0.0
    scores = [float(s.score) for s in sources if s.score is not None]
    if not scores:
        return 0.38
    return max(0.0, min(1.0, max(scores) / 10.0))


def _combined_answer_confidence(
    *,
    base_confidence: float,
    answer_text: str,
    sources: list[SourceItem],
    safety: dict[str, Any] | None,
) -> float:
    ev = _evidence_strength(sources)
    c = 0.52 * float(base_confidence) + 0.48 * ev
    if safety:
        lvl = str(safety.get("safety_level") or "normal").lower()
        if lvl == "emergency":
            c *= 0.9
        elif lvl == "warning":
            c *= 0.96
    low = answer_text.strip().lower()
    if not answer_text.strip():
        return 0.0
    if "insufficient data" in low:
        c = min(c, 0.42)
    return max(0.0, min(1.0, c))


def _sources_for_payload(sources: list[SourceItem], limit: int = 8) -> list[dict[str, Any]]:
    ranked = sorted(
        sources,
        key=lambda s: float(s.score) if s.score is not None else 0.0,
        reverse=True,
    )[:limit]
    return [s.model_dump() for s in ranked]


async def run_synthesis_answer(
    ctx: SupervisorContext,
    *,
    question: str,
    entities: list[EntityItem],
    sources: list[SourceItem],
    structured_features: dict[str, Any],
    scores: dict[str, Any] | None,
    safety: dict[str, Any] | None,
    risk: RiskBlock | None = None,
    prepend_safety_prefix: bool = True,
) -> AgentResult:
    """
    Build prompt from structured inputs, call LLM once, return answer + sources + confidence.

    When ``prepend_safety_prefix`` and safety provides ``message_prefix``, it is prepended
    to non-placeholder answers.
    """
    prompt = build_synthesis_prompt(
        question=question,
        entities=entities,
        sources=sources,
        structured_features=dict(structured_features or {}),
        scores=scores,
        safety=safety,
        risk=risk,
        trace_id=ctx.trace_id,
    )
    res = await run_synthesis(ctx, prompt=prompt)
    if not res.ok:
        return res

    text = (res.payload.get("answer") or "").strip()
    if (
        prepend_safety_prefix
        and safety
        and safety.get("message_prefix")
        and text
        and text.lower() != "insufficient data"
    ):
        text = f"{safety['message_prefix'].strip()} {text}".strip()

    conf = _combined_answer_confidence(
        base_confidence=res.confidence,
        answer_text=text,
        sources=sources,
        safety=safety,
    )

    payload: dict[str, Any] = {
        "answer": text,
        "sources": _sources_for_payload(sources),
        "confidence": conf,
    }
    trace = dict(res.trace)
    trace["synthesis_agent"] = True
    trace["source_count_payload"] = len(payload["sources"])

    return AgentResult(
        agent_id=AgentRole.SYNTHESIS,
        ok=True,
        confidence=conf,
        warnings=list(res.warnings),
        payload=payload,
        error_detail=None,
        duration_ms=res.duration_ms,
        trace=trace,
    )


class SynthesisAgent:
    """High-level synthesis; prefer :func:`run_synthesis_answer`."""

    build_prompt = staticmethod(build_synthesis_prompt)
    run = staticmethod(run_synthesis_answer)
