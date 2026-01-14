"""
KERNELIZE Platform - Research Services Package
================================================

Long-term research initiatives for compression breakthroughs
and reasoning capability advancement.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

__version__ = "1.0.0"
__author__ = "KERNELIZE Team"

from .compression.neural_semantic_compression import (
    NeuralSemanticCompressor,
    HierarchicalAttentionCompressor,
    NeuralSymbolicCompressor,
    KnowledgeGraphCompressor,
    DiffusionCompressor,
    CompressionResearchFramework
)

from .compression.domain_models import (
    DomainCompressionModel,
    HealthcareCompressionModel,
    FinanceCompressionModel,
    LegalCompressionModel,
    ScientificCompressionModel,
    GovernmentCompressionModel,
    DomainModelRegistry
)

from .reasoning.neuro_symbolic_reasoning import (
    NeuroSymbolicReasoner,
    NeuralModuleNetwork,
    SymbolicKnowledgeBase,
    HybridReasoningEngine,
    MultiStepInferenceEngine
)

from .reasoning.knowledge_graph import (
    KnowledgeGraphExtractor,
    GraphNeuralNetwork,
    HybridSearchEngine,
    KnowledgeGraphQuerier,
    KernelGraphManager
)

__all__ = [
    # Compression research
    'NeuralSemanticCompressor',
    'HierarchicalAttentionCompressor',
    'NeuralSymbolicCompressor',
    'KnowledgeGraphCompressor',
    'DiffusionCompressor',
    'CompressionResearchFramework',
    
    # Domain models
    'DomainCompressionModel',
    'HealthcareCompressionModel',
    'FinanceCompressionModel',
    'LegalCompressionModel',
    'ScientificCompressionModel',
    'GovernmentCompressionModel',
    'DomainModelRegistry',
    
    # Reasoning research
    'NeuroSymbolicReasoner',
    'NeuralModuleNetwork',
    'SymbolicKnowledgeBase',
    'HybridReasoningEngine',
    'MultiStepInferenceEngine',
    
    # Knowledge graph
    'KnowledgeGraphExtractor',
    'GraphNeuralNetwork',
    'HybridSearchEngine',
    'KnowledgeGraphQuerier',
    'KernelGraphManager'
]
