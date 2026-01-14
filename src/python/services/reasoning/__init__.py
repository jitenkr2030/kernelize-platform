"""
Reasoning Services Package
==========================

Reasoning and inference services.

Modules:
- multi_hop_reasoning: Multi-hop question answering
- causal_reasoning: Causal inference and analysis
"""

from .multi_hop_reasoning import (
    ReasoningType,
    EvidenceSource,
    MultiHopReasoningEngine,
    QuestionDecomposer,
    EvidenceCollector,
    ReasoningChainExecutor,
    ConfidenceAggregator,
    ExplanationGenerator,
    ReasoningTrace,
    ReasoningStep,
    Evidence,
    SubQuestion,
    MultiHopResult,
)

from .causal_reasoning import (
    CausalRelationType,
    CausalQueryType,
    CausalReasoningEngine,
    CausalGraphBuilder,
    CausalPatternExtractor,
    CounterfactualEngine,
    CausalExplanation,
    CausalNode,
    CausalEdge,
    CausalPath,
    CounterfactualScenario,
)

__all__ = [
    # Multi-hop reasoning
    "ReasoningType",
    "EvidenceSource",
    "MultiHopReasoningEngine",
    "QuestionDecomposer",
    "EvidenceCollector",
    "ReasoningChainExecutor",
    "ConfidenceAggregator",
    "ExplanationGenerator",
    "ReasoningTrace",
    "ReasoningStep",
    "Evidence",
    "SubQuestion",
    "MultiHopResult",
    # Causal reasoning
    "CausalRelationType",
    "CausalQueryType",
    "CausalReasoningEngine",
    "CausalGraphBuilder",
    "CausalPatternExtractor",
    "CounterfactualEngine",
    "CausalExplanation",
    "CausalNode",
    "CausalEdge",
    "CausalPath",
    "CounterfactualScenario",
]
