"""
KERNELIZE Platform - Reasoning Research Package
================================================

Research modules for advancing reasoning capabilities:
- Neuro-symbolic reasoning combining neural networks with symbolic logic
- Knowledge graph integration for structured knowledge representation

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

from .neuro_symbolic_reasoning import (
    NeuroSymbolicReasoner,
    NeuralModuleNetwork,
    SymbolicKnowledgeBase,
    HybridReasoningEngine,
    MultiStepInferenceEngine,
    ReasoningType,
    InferenceResult
)

from .knowledge_graph import (
    KnowledgeGraphExtractor,
    GraphNeuralNetwork,
    HybridSearchEngine,
    KnowledgeGraphQuerier,
    KernelGraphManager,
    GraphQueryResult
)

__all__ = [
    # Neuro-symbolic reasoning
    'NeuroSymbolicReasoner',
    'NeuralModuleNetwork',
    'SymbolicKnowledgeBase',
    'HybridReasoningEngine',
    'MultiStepInferenceEngine',
    'ReasoningType',
    'InferenceResult',
    
    # Knowledge graph
    'KnowledgeGraphExtractor',
    'GraphNeuralNetwork',
    'HybridSearchEngine',
    'KnowledgeGraphQuerier',
    'KernelGraphManager',
    'GraphQueryResult'
]
