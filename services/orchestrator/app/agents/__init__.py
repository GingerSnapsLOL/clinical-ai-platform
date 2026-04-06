"""
Supervisor-pattern multi-agent orchestration (bounded, deterministic, traceable).

v1 ships: CoordinatorAgent, ClinicalStructuringAgent, RetrievalAgent, SynthesisAgent.
Scoring: :class:`ScoringAgent`. Evidence: :class:`EvidenceCriticAgent`. Safety: :class:`SafetyAgent`.
"""

from app.agents.base import (
    AGENT_VERSION,
    MAX_WORKFLOW_STEPS_V1,
    AgentResult,
    AgentRole,
    RetrievalCachePort,
    SupervisorContext,
)
from app.agents.clinical_structuring_agent import ClinicalStructuringAgent
from app.agents.clarification_agent import ClarificationAgent, run_clarification
from app.agents.evidence_critic import EvidenceCriticAgent, run_evidence_critic
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.coordinator import SupervisorCoordinator
from app.agents.safety_agent import SafetyAgent, run_safety
from app.agents.scoring_agent import ScoringAgent, run_scoring_step
from app.agents.synthesis_agent import SynthesisAgent, run_synthesis_answer
from app.agents.coordinator_bridge import (
    create_clinical_coordinator_dispatch,
    run_clinical_coordinator,
)

__all__ = [
    "ClarificationAgent",
    "ClinicalStructuringAgent",
    "run_clarification",
    "EvidenceCriticAgent",
    "run_evidence_critic",
    "RetrievalAgent",
    "SafetyAgent",
    "run_safety",
    "SynthesisAgent",
    "run_synthesis_answer",
    "ScoringAgent",
    "AGENT_VERSION",
    "MAX_WORKFLOW_STEPS_V1",
    "AgentResult",
    "AgentRole",
    "RetrievalCachePort",
    "SupervisorContext",
    "SupervisorCoordinator",
    "create_clinical_coordinator_dispatch",
    "run_clinical_coordinator",
    "run_scoring_step",
]
