"""
KERNELIZE Platform - Query Engine Tests
=========================================

This module contains unit tests for the semantic query engine.
Tests cover query types, caching, and search functionality.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQueryEngine:
    """Test cases for the query engine"""
    
    def test_semantic_query(self):
        """Test basic semantic query functionality"""
        from services.query_engine import KernelQueryEngine, QueryType
        
        engine = KernelQueryEngine()
        
        # Index some test content
        engine.index_kernel(
            kernel_id="test_1",
            content="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        )
        engine.index_kernel(
            kernel_id="test_2",
            content="Deep learning uses neural networks with multiple layers to learn complex patterns.",
        )
        
        # Query
        results, metrics = engine.query(
            query_text="How do neural networks learn?",
            query_type=QueryType.SEMANTIC,
            top_k=2,
        )
        
        assert metrics is not None
        assert metrics.query_type == "semantic"
        assert metrics.query_time_ms >= 0
    
    def test_exact_query(self):
        """Test exact match query functionality"""
        from services.query_engine import KernelQueryEngine, QueryType
        
        engine = KernelQueryEngine()
        
        # Index content
        engine.index_kernel(
            kernel_id="exact_1",
            content="Machine learning algorithms include decision trees, random forests, and gradient boosting.",
        )
        
        # Query with exact match
        results, metrics = engine.query(
            query_text="decision trees",
            query_type=QueryType.EXACT,
            top_k=1,
        )
        
        assert metrics is not None
        assert metrics.query_type == "exact"
    
    def test_fuzzy_query(self):
        """Test fuzzy matching query"""
        from services.query_engine import KernelQueryEngine, QueryType
        
        engine = KernelQueryEngine()
        
        # Index content
        engine.index_kernel(
            kernel_id="fuzzy_1",
            content="Natural language processing enables computers to understand human language.",
        )
        
        # Query with typos
        results, metrics = engine.query(
            query_text="naturall languag procesing",  # Intentional typos
            query_type=QueryType.FUZZY,
            top_k=1,
        )
        
        assert metrics is not None
        assert metrics.query_type == "fuzzy"
    
    def test_hybrid_query(self):
        """Test hybrid search combining semantic and exact"""
        from services.query_engine import KernelQueryEngine, QueryType
        
        engine = KernelQueryEngine()
        
        # Index content
        engine.index_kernel(
            kernel_id="hybrid_1",
            content="Python is a popular programming language for data science and machine learning.",
        )
        engine.index_kernel(
            kernel_id="hybrid_2",
            content="Java is widely used for enterprise applications and Android development.",
        )
        
        # Hybrid query
        results, metrics = engine.query(
            query_text="Python programming language",
            query_type=QueryType.HYBRID,
            top_k=2,
        )
        
        assert metrics is not None
        assert metrics.query_type == "hybrid"
    
    def test_kernel_indexing(self):
        """Test kernel indexing functionality"""
        from services.query_engine import KernelQueryEngine
        
        engine = KernelQueryEngine()
        
        # Index multiple kernels
        engine.index_kernel("kernel_1", "Content about artificial intelligence.")
        engine.index_kernel("kernel_2", "Content about machine learning.")
        engine.index_kernel("kernel_3", "Content about deep learning.")
        
        stats = engine.get_stats()
        
        assert stats["indexed_kernels"] == 3
        assert "semantic" in stats["search_types"]
        assert "exact" in stats["search_types"]
    
    def test_kernel_deletion(self):
        """Test kernel deletion from index"""
        from services.query_engine import KernelQueryEngine
        
        engine = KernelQueryEngine()
        
        # Index kernel
        engine.index_kernel("delete_test", "Test content to delete.")
        
        # Verify indexed
        assert "delete_test" in engine.knowledge_base
        
        # Delete
        success = engine.delete_kernel("delete_test")
        assert success is True
        
        # Verify deleted
        assert "delete_test" not in engine.knowledge_base
    
    def test_cache_functionality(self):
        """Test query result caching"""
        from services.query_engine import KernelQueryEngine, QueryType
        
        engine = KernelQueryEngine()
        
        # Index content
        engine.index_kernel("cache_1", "Cache test content.")
        
        # First query (cache miss)
        results1, metrics1 = engine.query(
            query_text="Cache test",
            query_type=QueryType.SEMANTIC,
        )
        
        # Second query (cache hit)
        results2, metrics2 = engine.query(
            query_text="Cache test",
            query_type=QueryType.SEMANTIC,
        )
        
        cache_stats = engine.cache.get_stats()
        assert cache_stats["size"] >= 1


class TestEmbeddingGenerator:
    """Test cases for embedding generation"""
    
    def test_embedding_generation(self):
        """Test basic embedding generation"""
        from services.query_engine import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        embedding = generator.generate("Test sentence for embedding")
        
        assert embedding is not None
        assert len(embedding) == generator.dimensions
        assert all(isinstance(x, float) for x in embedding)
    
    def test_batch_embedding_generation(self):
        """Test batch embedding generation"""
        from services.query_engine import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence",
        ]
        
        embeddings = generator.batch_generate(texts)
        
        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == generator.dimensions
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        from services.query_engine import EmbeddingGenerator
        
        generator = EmbeddingGenerator()
        
        # Similar sentences should have higher similarity
        emb1 = generator.generate("The cat sat on the mat")
        emb2 = generator.generate("The cat is sitting on the rug")
        
        similarity = generator.cosine_similarity(emb1, emb2)
        
        assert 0.0 <= similarity <= 1.0
    
    def test_embedding_stats(self):
        """Test embedding generator stats"""
        from services.query_engine import EmbeddingGenerator
        
        generator = EmbeddingGenerator(dimensions=384)
        
        stats = generator.get_stats()
        
        assert stats["model_name"] == "all-MiniLM-L6-v2"
        assert stats["dimensions"] == 384
        assert stats["status"] == "ready"


class TestCacheManager:
    """Test cases for cache management"""
    
    def test_cache_set_get(self):
        """Test basic cache set and get"""
        from services.query_engine import CacheManager
        
        cache = CacheManager(max_size=10)
        
        cache.set("key1", "value1")
        result = cache.get("key1")
        
        assert result == "value1"
    
    def test_cache_ttl(self):
        """Test cache TTL expiration"""
        from services.query_engine import CacheManager
        
        # Create cache with 0 TTL (immediate expiration)
        cache = CacheManager(max_size=10, ttl_hours=0)
        
        cache.set("key1", "value1")
        result = cache.get("key1")
        
        # Should be None due to TTL
        assert result is None
    
    def test_cache_eviction(self):
        """Test cache LRU eviction"""
        from services.query_engine import CacheManager
        
        cache = CacheManager(max_size=3, ttl_hours=24)
        
        # Fill cache
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Add one more (should evict key1)
        cache.set("key4", "value4")
        
        assert cache.get("key1") is None  # Evicted
        assert cache.get("key2") is not None  # Still there
        assert cache.get("key3") is not None  # Still there
        assert cache.get("key4") == "value4"  # New entry
    
    def test_cache_stats(self):
        """Test cache statistics"""
        from services.query_engine import CacheManager
        
        cache = CacheManager(max_size=10)
        
        # Access cache
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestQueryEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_index_query(self):
        """Test querying empty index"""
        from services.query_engine import KernelQueryEngine, QueryType
        
        engine = KernelQueryEngine()
        
        results, metrics = engine.query(
            query_text="Any query",
            query_type=QueryType.SEMANTIC,
        )
        
        assert len(results) == 0
        assert metrics.total_results == 0
    
    def test_filtered_query(self):
        """Test querying specific kernel IDs"""
        from services.query_engine import KernelQueryEngine, QueryType
        
        engine = KernelQueryEngine()
        
        # Index content
        engine.index_kernel("filter_1", "First document")
        engine.index_kernel("filter_2", "Second document")
        engine.index_kernel("filter_3", "Third document")
        
        # Query only specific kernels
        results, metrics = engine.query(
            query_text="document",
            kernel_ids=["filter_1", "filter_2"],
            query_type=QueryType.EXACT,
        )
        
        # Should only find results in filtered kernels
        for result in results:
            assert result.kernel_id in ["filter_1", "filter_2"]
    
    def test_top_k_parameter(self):
        """Test top_k parameter limits results"""
        from services.query_engine import KernelQueryEngine, QueryType
        
        engine = KernelQueryEngine()
        
        # Index many kernels
        for i in range(10):
            engine.index_kernel(f"many_{i}", f"Content about topic {i}")
        
        # Query with top_k=3
        results, metrics = engine.query(
            query_text="Content",
            top_k=3,
            query_type=QueryType.SEMANTIC,
        )
        
        assert len(results) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
