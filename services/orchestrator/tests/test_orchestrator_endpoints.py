"""Endpoint tests for orchestrator: /health, /v1/ask."""
import sys
from pathlib import Path

_path = Path(__file__).resolve().parent.parent
_root = _path.parent
_repo_root = _root.parent
for p in (str(_repo_root), str(_root), str(_path)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pytest
from fastapi.testclient import TestClient

from services.shared.schemas_v1 import SourceItem

from app.main import app, _retrieval_meets_relevance_bar

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["service"] == "orchestrator"


def test_ask_contract():
    """Contract endpoint: accepts AskRequest, returns AskResponse with required fields."""
    r = client.post(
        "/v1/ask",
        json={
            "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
            "mode": "strict",
            "note_text": "Patient with hypertension",
            "question": "What is the risk?",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "trace_id" in data
    assert data["trace_id"] == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    assert "answer" in data
    assert "entities" in data
    assert "sources" in data
    assert isinstance(data.get("risk_block"), (dict, type(None)))
    assert data.get("diagnostics") is None
    assert "pii_redacted" in data


def test_ask_contract_debug_includes_diagnostics():
    r = client.post(
        "/v1/ask",
        json={
            "trace_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567891",
            "mode": "strict",
            "note_text": "Patient with hypertension",
            "question": "What is the risk?",
            "user_context": {"debug": True},
        },
    )
    assert r.status_code == 200
    data = r.json()
    diagnostics = data.get("diagnostics")
    assert isinstance(diagnostics, dict)
    assert isinstance(diagnostics.get("timings"), dict)
    assert "warnings" in diagnostics
    assert "fallback_used" in diagnostics


def test_ask_llm_integration_fallback_friendly(monkeypatch):
    """
    Integration-style test for /v1/ask with llm-based synthesis.

    The test is tolerant of environments where llm-service is unavailable:
    it only asserts on properties that should hold in both LLM and fallback modes.
    """
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE", "false")

    r = client.post(
        "/v1/ask",
        json={
            "trace_id": "11111111-2222-3333-4444-555555555555",
            "mode": "strict",
            "note_text": "65-year-old with hypertension and diabetes, on ACE inhibitor and statin.",
            "question": "What is this patient's cardiovascular risk and how should they be monitored?",
        },
    )
    assert r.status_code == 200
    data = r.json()

    # Answer should be non-empty regardless of whether LLM or template fallback is used.
    answer = data.get("answer", "")
    assert isinstance(answer, str)
    assert answer.strip() != ""

    # Citations should be present as a list.
    citations = data.get("citations", [])
    assert isinstance(citations, list)

    # Sources should be a list with at most 3 items (top passages).
    sources = data.get("sources", [])
    assert isinstance(sources, list)
    assert len(sources) <= 3

    # Entities and risk should be present.
    entities = data.get("entities", [])
    assert isinstance(entities, list)
    assert isinstance(data.get("risk_block"), (dict, type(None)))


def test_retrieval_meets_relevance_bar_no_passages():
    ok, score, reason = _retrieval_meets_relevance_bar([])
    assert ok is False
    assert score == 0.0
    assert reason == "no_passages"


def test_retrieval_meets_relevance_bar_below_min_score(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE", "true")
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_MIN_TOP_SCORE", "5.0")
    sources = [SourceItem(source_id="a", snippet="x" * 30, score=0.5)]
    ok, _, reason = _retrieval_meets_relevance_bar(sources)
    assert ok is False
    assert reason == "below_min_score"


def test_retrieval_meets_relevance_bar_weak_snippet(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE", "true")
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_MIN_TOP_SCORE", "0.0")
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_MIN_TOP_SNIPPET_CHARS", "100")
    sources = [SourceItem(source_id="a", snippet="short", score=10.0)]
    ok, _, reason = _retrieval_meets_relevance_bar(sources)
    assert ok is False
    assert reason == "weak_snippet"


def test_retrieval_meets_relevance_bar_gate_disabled(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE", "false")
    sources = [SourceItem(source_id="a", snippet="x", score=-99.0)]
    ok, score, reason = _retrieval_meets_relevance_bar(sources)
    assert ok is True
    assert score == -99.0
    assert reason == ""


def test_retrieval_meets_relevance_bar_accepts(monkeypatch):
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_RELEVANCE_GATE", "true")
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_MIN_TOP_SCORE", "0.5")
    monkeypatch.setenv("ORCHESTRATOR_RETRIEVAL_MIN_TOP_SNIPPET_CHARS", "10")
    sources = [SourceItem(source_id="a", snippet="adequate length", score=1.0)]
    ok, score, reason = _retrieval_meets_relevance_bar(sources)
    assert ok is True
    assert score == 1.0
    assert reason == ""


def test_ask_supervised_pipeline_branch_mocked(monkeypatch):
    """Supervised coordinator path returns AskResponse without calling downstream HTTP."""
    from app.agents.base import AgentRole, AgentResult
    from app.agents.coordinator import SupervisorCoordinator, SupervisorRunResult

    async def fake_run(self, ctx):
        struct = AgentResult(
            agent_id=AgentRole.CLINICAL_STRUCTURING.value,
            ok=True,
            confidence=1.0,
            payload={
                "pii_redacted": True,
                "entities": [],
                "redacted_text": ctx.note_text,
            },
        )
        return SupervisorRunResult(
            trace_id=ctx.trace_id,
            ok=True,
            steps=[struct],
            gate_accepted=False,
            gate_reason="mock",
            final_answer="Supervised answer",
            entities=[],
            sources=[],
            risk={"score": 0.5, "label": "medium", "explanation": []},
        )

    monkeypatch.setenv("ORCHESTRATOR_SUPERVISOR_PIPELINE", "true")
    monkeypatch.setattr(SupervisorCoordinator, "run", fake_run)

    r = client.post(
        "/v1/ask",
        json={
            "trace_id": "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            "mode": "strict",
            "note_text": "Patient note for supervised test.",
            "question": "What is the plan?",
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert data["trace_id"] == "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb"
    assert data["answer"] == "Supervised answer"
    assert "gate:mock" in data.get("warnings", [])
    assert data.get("diagnostics", {}).get("timings", {}).get("supervised_pipeline_flag") == 1.0
