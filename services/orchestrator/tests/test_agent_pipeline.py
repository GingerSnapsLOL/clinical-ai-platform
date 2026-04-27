"""Agent runtime pipeline tests (no network; LLM mocked)."""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

_path = Path(__file__).resolve().parent.parent
_root = _path.parent
_repo_root = _root.parent
for p in (str(_repo_root), str(_root), str(_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent_nodes import (  # noqa: E402
    answer_verifier_node,
    draft_answer_node,
    evidence_selector_node,
    finalize_answer_node,
)
from agent_runtime import compile_linear_chain  # noqa: E402
from agent_state import AgentState  # noqa: E402
from services.shared.llm_client import LLMGenerateResponse  # noqa: E402
from services.shared.schemas_v1 import (  # noqa: E402
    ExtractResponse,
    PassageItem,
    RedactResponse,
    RetrieveResponse,
    ScoreResponse,
    SourceItem,
)

from app import main as main_module  # noqa: E402


def _run_chain(state: AgentState, llm_texts: list[str]) -> AgentState:
    queue = list(llm_texts)

    async def on_generate(trace_id: str, prompt: str = "", **kwargs: object) -> LLMGenerateResponse:
        if not queue:
            raise RuntimeError("unexpected LLM generate call")
        text = queue.pop(0)
        return LLMGenerateResponse(status="ok", trace_id=trace_id, text=text)

    def llm_factory(*_a: object, **_kw: object) -> MagicMock:
        m = MagicMock()
        m.generate = AsyncMock(side_effect=on_generate)
        m.aclose = AsyncMock()
        return m

    runtime = compile_linear_chain(
        evidence_selector_node,
        draft_answer_node,
        answer_verifier_node,
        finalize_answer_node,
    )
    with patch("agent_nodes.LLMClient", side_effect=llm_factory):
        final, _timings = asyncio.run(runtime.run(state))
    assert not queue, f"unused LLM stubs: {queue}"
    return final


def test_no_sources_yields_insufficient_data():
    state = AgentState(trace_id="t1", question="What to do?", sources=[])
    final = _run_chain(state, [])
    assert final.final_answer == "Insufficient data"
    assert final.draft_answer == "Insufficient data"


def test_verifier_rejects_weak_grounding():
    src = [
        SourceItem(
            source_id="src_a",
            snippet="Daily low-dose aspirin may reduce cardiovascular events in some adults.",
            score=1.5,
        )
    ]
    draft_body = json.dumps({"answer": "Take aspirin daily.", "used_source_ids": ["src_a"]})
    verifier_body = json.dumps(
        {
            "is_grounded": False,
            "has_sufficient_evidence": False,
            "problems": ["overstates indication"],
        }
    )
    state = AgentState(trace_id="t2", question="Should I take aspirin?", sources=src)
    final = _run_chain(state, [draft_body, verifier_body])
    assert final.final_answer == "Insufficient data"


def test_verifier_accepts_grounded_answer():
    src = [
        SourceItem(
            source_id="src_b",
            snippet="Daily low-dose aspirin may reduce cardiovascular events in some adults.",
            score=2.0,
        )
    ]
    draft_body = json.dumps(
        {"answer": "Guidelines discuss aspirin for prevention; follow clinician advice.", "used_source_ids": ["src_b"]}
    )
    verifier_body = json.dumps(
        {"is_grounded": True, "has_sufficient_evidence": True, "problems": []}
    )
    state = AgentState(trace_id="t3", question="Aspirin?", sources=src)
    final = _run_chain(state, [draft_body, verifier_body])
    assert "aspirin" in (final.final_answer or "").lower()
    assert final.final_answer != "Insufficient data"


def test_malformed_verifier_json_conservative_insufficient():
    src = [
        SourceItem(
            source_id="src_c",
            snippet="Beta blockers may be used for rate control in atrial fibrillation.",
            score=2.0,
        )
    ]
    draft_body = json.dumps({"answer": "Beta blockers can be used for rate control.", "used_source_ids": ["src_c"]})
    state = AgentState(trace_id="t4", question="Rate control?", sources=src)
    final = _run_chain(state, [draft_body, "NOT JSON {{{"])
    assert final.final_answer == "Insufficient data"


def test_feature_flag_disabled_single_llm_path(monkeypatch):
    """ORCHESTRATOR_AGENT_MODE=false: never runs agent multi-step; one LLM call."""
    monkeypatch.setenv("ORCHESTRATOR_AGENT_MODE", "false")
    monkeypatch.setenv("ORCHESTRATOR_CACHE_ENABLED", "false")
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE", "false")

    tid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    called_models: list[str] = []

    async def fake_post_typed(client, url, request_body, response_model, timeout=None, trace_id=None):
        name = response_model.__name__
        called_models.append(name)
        if name == "RedactResponse":
            return RedactResponse(trace_id=tid, redacted_text=request_body.text), None, None
        if name == "ExtractResponse":
            return ExtractResponse(trace_id=tid, entities=[]), None, None
        if name == "RetrieveResponse":
            return (
                RetrieveResponse(
                    trace_id=tid,
                    passages=[
                        PassageItem(
                            source_id="p1",
                            text="Educational passage about hypertension management and follow-up.",
                            score=1.2,
                        )
                    ],
                ),
                None,
                None,
            )
        if name == "ScoreResponse":
            return (
                ScoreResponse(
                    trace_id=tid,
                    score=0.2,
                    label="low",
                    explanation="",
                    contributions=[],
                ),
                None,
                None,
            )
        raise AssertionError(f"unexpected response_model {name}")

    llm_instance = MagicMock()
    llm_instance.generate = AsyncMock(
        return_value=LLMGenerateResponse(
            status="ok", trace_id=tid, text="Single-call LLM answer body."
        )
    )
    llm_instance.aclose = AsyncMock()

    monkeypatch.setattr(main_module, "post_typed", fake_post_typed)
    trace_store_mock = AsyncMock(return_value=(True, None))
    monkeypatch.setattr(main_module, "save_request_trace", trace_store_mock)
    monkeypatch.setenv("LLM_BASE_URL", "http://mock-llm")

    with patch.object(main_module, "LLMClient", return_value=llm_instance):
        client = TestClient(main_module.app)
        r = client.post(
            "/v1/ask",
            json={
                "trace_id": tid,
                "mode": "strict",
                "note_text": "Patient has hypertension.",
                "question": "What is the plan?",
                "user_context": {"debug": True},
            },
        )

    assert r.status_code == 200
    data = r.json()
    assert data["answer"] == "Single-call LLM answer body."
    trace_store_mock.assert_awaited_once()
    assert ((data.get("diagnostics") or {}).get("trace_storage") or {}).get("saved") is True
    llm_instance.generate.assert_awaited_once()
    assert "ScoreResponse" not in called_models
    scoring_diag = (data.get("diagnostics") or {}).get("scoring_diagnostics") or {}
    assert scoring_diag.get("skipped") is True
    assert scoring_diag.get("skip_reason") == "no_relevant_entities"
    timing_keys = set((data.get("diagnostics") or {}).get("timings", {}).keys())
    assert "agent_total_duration_ms" not in timing_keys
