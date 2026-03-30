"""Step-by-step async execution of agent nodes over ``AgentState``."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable, Sequence

from services.shared.logging_util import get_logger

from agent_nodes import AgentNode
from agent_state import AgentState

logger = get_logger(__name__, "orchestrator")

_AGENT_NODE_DURATION_KEYS: dict[str, str] = {
    "evidence_selector_node": "agent_selector_duration_ms",
    "draft_answer_node": "agent_draft_duration_ms",
    "answer_verifier_node": "agent_verifier_duration_ms",
    "finalize_answer_node": "agent_finalize_duration_ms",
}


class AgentRuntime:
    """
    Runs an ordered list of async nodes. Each node receives the latest state and
    must return a new ``AgentState``. Execution stops early if ``state.stop_reason``
    is set (e.g. via ``mark_stop``).

    ``run`` returns per-node wall times (ms) for observability.
    """

    def __init__(self, nodes: Sequence[AgentNode]):
        self._nodes = tuple(nodes)

    @property
    def nodes(self) -> tuple[AgentNode, ...]:
        return self._nodes

    async def run(self, initial: AgentState) -> tuple[AgentState, dict[str, float]]:
        timing_keys = frozenset(_AGENT_NODE_DURATION_KEYS.values())
        timings_ms: dict[str, float] = {k: 0.0 for k in timing_keys}
        timings_ms["agent_total_duration_ms"] = 0.0

        state = initial
        trace_id = state.trace_id
        total_t0 = time.perf_counter()

        for step_i, node in enumerate(self._nodes):
            if state.stop_reason is not None:
                break
            name = getattr(node, "__name__", type(node).__name__)
            timing_field = _AGENT_NODE_DURATION_KEYS.get(name)

            logger.debug(
                "agent_node_start",
                extra={"trace_id": trace_id, "node": name, "step": step_i},
            )

            node_t0 = time.perf_counter()
            state = await node(state)
            node_ms = (time.perf_counter() - node_t0) * 1000.0

            if timing_field is not None:
                timings_ms[timing_field] = node_ms

            logger.info(
                "agent_node_latency",
                extra={
                    "trace_id": trace_id,
                    "node": name,
                    "step": step_i,
                    "duration_ms": round(node_ms, 3),
                    "timing_key": timing_field or name,
                },
            )

            logger.debug(
                "agent_node_done",
                extra={"trace_id": trace_id, "node": name, "step": step_i},
            )

        timings_ms["agent_total_duration_ms"] = (time.perf_counter() - total_t0) * 1000.0

        logger.info(
            "agent_runtime_timing_summary",
            extra={
                "trace_id": trace_id,
                "agent_total_duration_ms": round(timings_ms["agent_total_duration_ms"], 3),
                "agent_selector_duration_ms": round(timings_ms["agent_selector_duration_ms"], 3),
                "agent_draft_duration_ms": round(timings_ms["agent_draft_duration_ms"], 3),
                "agent_verifier_duration_ms": round(timings_ms["agent_verifier_duration_ms"], 3),
                "agent_finalize_duration_ms": round(timings_ms["agent_finalize_duration_ms"], 3),
            },
        )

        return state, timings_ms


def compile_linear_chain(
    *nodes: AgentNode,
) -> AgentRuntime:
    """Convenience: build a runtime from positional node functions."""
    return AgentRuntime(nodes)
