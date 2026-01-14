"""
KERNELIZE Platform - Compression Research Package
==================================================

Research modules for advanced compression techniques
focusing on neural semantic and domain-specific approaches.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

from .neural_semantic_compression import (
    NeuralSemanticCompressor,
    HierarchicalAttentionCompressor,
    NeuralSymbolicCompressor,
    KnowledgeGraphCompressor,
    DiffusionCompressor,
    CompressionResearchFramework,
    CompressionStrategy,
    CompressionLevel
)

from .domain_models import (
    DomainCompressionModel,
    HealthcareCompressionModel,
    FinanceCompressionModel,
    LegalCompressionModel,
    ScientificCompressionModel,
    GovernmentCompressionModel,
    DomainModelRegistry,
    DomainType
)

__all__ = [
    'NeuralSemanticCompressor',
    'HierarchicalAttentionCompressor',
    'NeuralSymbolicCompressor',
    'KnowledgeGraphCompressor',
    'DiffusionCompressor',
    'CompressionResearchFramework',
    'CompressionStrategy',
    'CompressionLevel',
    'DomainCompressionModel',
    'HealthcareCompressionModel',
    'FinanceCompressionModel',
    'LegalCompressionModel',
    'ScientificCompressionModel',
    'GovernmentCompressionModel',
    'DomainModelRegistry',
    'DomainType'
]
