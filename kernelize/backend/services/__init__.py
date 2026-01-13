"""
KERNELIZE Platform - Services Package
======================================

This package contains all core services for the KERNELIZE Knowledge
Compression Platform including compression, query, monitoring, and
multi-modal processing capabilities.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

from .compression_engine import (
    KernelCompressionEngine,
    CompressionResult,
    Entity,
    Relationship,
    CausalChain,
    CompressionLevel,
    ContentType,
    compression_engine,
    compress_knowledge,
)

from .query_engine import (
    KernelQueryEngine,
    QueryResult,
    QueryMetrics,
    QueryType,
    CacheManager,
    EmbeddingGenerator,
    ExactMatcher,
    HybridSearchEngine,
    query_engine,
    semantic_search,
    hybrid_search,
)

__all__ = [
    # Compression Engine
    "KernelCompressionEngine",
    "CompressionResult",
    "Entity",
    "Relationship",
    "CausalChain",
    "CompressionLevel",
    "ContentType",
    "compression_engine",
    "compress_knowledge",
    
    # Query Engine
    "KernelQueryEngine",
    "QueryResult",
    "QueryMetrics",
    "QueryType",
    "CacheManager",
    "EmbeddingGenerator",
    "ExactMatcher",
    "HybridSearchEngine",
    "query_engine",
    "semantic_search",
    "hybrid_search",
]
