"""Unit tests for internal agent runtime (no HTTP)."""
import asyncio
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
_repo = _root.parent.parent
for p in (str(_repo), str(_root)):
    if p not in sys.path:
        sys.path.insert(0, p)

from agent_runtime import AgentRuntime, compile_linear_chain
from agent_state import AgentState
from services.shared.schemas_v1 import SourceItem


def test_runtime_runs_nodes_in_order():
    order: list[str] = []

    async def a(state: AgentState) -> AgentState:
        order.append("a")
        return state.evolve(warnings=[*state.warnings, "a"])

    async def b(state: AgentState) -> AgentState:
        order.append("b")
        return state.evolve(warnings=[*state.warnings, "b"])

    rt = compile_linear_chain(a, b)
    out, timings = asyncio.run(
        rt.run(AgentState(trace_id="t1", sources=[], question="q"))
    )
    assert order == ["a", "b"]
    assert out.trace_id == "t1"
    assert out.question == "q"
    assert out.warnings == ["a", "b"]
    assert "agent_total_duration_ms" in timings


def test_runtime_halts_when_mark_stop():
    async def early_stop(state: AgentState) -> AgentState:
        return state.mark_stop("test_done")

    async def never(state: AgentState) -> AgentState:
        raise AssertionError("should not run")

    rt = AgentRuntime([early_stop, never])
    out, timings = asyncio.run(rt.run(AgentState(trace_id="t2", question="")))
    assert out.stop_reason == "test_done"
    assert timings["agent_total_duration_ms"] >= 0.0


def test_top_k_sources_and_has_sources():
    s = AgentState(
        trace_id="x",
        question="q",
        sources=[
            SourceItem(source_id="1", score=0.9),
            SourceItem(source_id="2", score=0.5),
        ],
    )
    assert s.has_sources()
    assert [x.source_id for x in s.top_k_sources(1)] == ["1"]
    assert len(s.top_k_sources(10)) == 2
    empty = AgentState(trace_id="y", question="q", sources=[])
    assert not empty.has_sources()
    assert empty.top_k_sources(3) == []
