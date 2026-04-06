"""Strict grounding prompt for clinical Q&A over retrieved passages."""

from __future__ import annotations

from typing import List

from services.shared.logging_util import get_logger
from services.shared.schemas_v1 import EntityItem, RiskBlock, SourceItem

logger = get_logger(__name__, "orchestrator")


def build_llm_prompt(
    question: str,
    entities: List[EntityItem],
    sources: List[SourceItem],
    risk: RiskBlock | None,
    trace_id: str,
) -> str:
    MAX_CONTEXT_CHARS = 6000
    lines: List[str] = []

    lines.append(
        "You are a clinical decision support assistant. Your answer must follow strict "
        "grounding rules below. All clinical and factual content in your answer must come "
        "only from the section titled 'Top evidence passages.' Do not use general medical "
        "knowledge, guidelines you were trained on, or inference beyond what is explicitly "
        "written there. Extracted entities and risk scores are auxiliary context only: do not "
        "state any fact that is not directly supported by wording in those passages."
    )
    lines.append("")
    lines.append(f"Question:\n{question}")

    if entities:
        lines.append("")
        lines.append("Extracted entities:")
        for e in entities[:20]:
            lines.append(f"- {e.type}: {e.text} (span {e.start}-{e.end})")

    context_lines: List[str] = []
    num_passages_used = 0
    if sources:
        context_lines.append("")
        context_lines.append("Top evidence passages:")
        top_sources = sorted(
            sources,
            key=lambda s: s.score if s.score is not None else 0.0,
            reverse=True,
        )[:3]
        for idx, s in enumerate(top_sources, start=1):
            title = s.title or (s.metadata.get("title") if s.metadata else None)
            header = f"Passage {idx} (source_id={s.source_id})"
            if title:
                header += f" - {title}"
            context_lines.append(header + ":")
            context_lines.append(s.snippet or "")
            context_lines.append("")
            num_passages_used += 1

    if risk is not None:
        context_lines.append("")
        context_lines.append(
            "Risk assessment (not part of evidence passages — do not use in the answer "
            "unless the same facts appear verbatim or by clear paraphrase in Top evidence passages above):"
        )
        context_lines.append(f"- Overall risk label: {risk.label}")
        context_lines.append(f"- Risk score: {risk.score:.2f}")
        if risk.explanation:
            context_lines.append("- Top contributing factors:")
            for feat in risk.explanation[:5]:
                context_lines.append(f"  - {feat.feature}: {feat.contribution:.3f}")

    context_text = "\n".join(context_lines)
    if len(context_text) > MAX_CONTEXT_CHARS:
        context_text = context_text[:MAX_CONTEXT_CHARS]

    logger.info(
        "llm_context_built",
        extra={
            "trace_id": trace_id,
            "num_passages": num_passages_used,
            "context_length": len(context_text),
        },
    )

    if context_text:
        lines.append("")
        lines.append(context_text)

    lines.append("")
    lines.append(
        "Instructions (strict grounding — follow in order):\n"
        "1. Answer the question using ONLY information explicitly stated in "
        "'Top evidence passages' above. Quote or paraphrase only what appears there.\n"
        "2. If there are no evidence passages, or the passages do not contain enough "
        "information to answer the question, respond with exactly this single line and nothing else:\n"
        "Insufficient data\n"
        "3. Do NOT add facts, drug names, doses, guideline names, pathophysiology, or "
        "treatment recommendations unless they appear in those passages.\n"
        "4. Do NOT infer, speculate, or fill gaps with external or prior knowledge. "
        "If the question cannot be answered from the passages alone, output only: Insufficient data\n"
        "5. Keep the answer concise and clinician-facing when you do answer from passages."
    )

    return "\n".join(lines)
