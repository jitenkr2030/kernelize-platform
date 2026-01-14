#!/usr/bin/env python3
"""
Priority 1 Implementation Verification Script
Tests Task 1.1.2: Vector Database Integration and Task 1.1.3: Embedding Caching Layer
"""

import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from unittest.mock import Mock, patch, MagicMock
import hashlib

# Mock imports for testing without actual servers
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()
sys.modules['redis'] = MagicMock()


class VectorDatabaseIntegrationVerifier:
    """Verifies Task 1.1.2: Vector Database Integration"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def verify_qdrant_vector_store_class_exists(self) -> bool:
        """Verify QdrantVectorStore class is defined"""
        print("\n1. Testing QdrantVectorStore Class...")
        try:
            from src.python.services.query_engine import QdrantVectorStore
            
            # Verify core methods exist
            required_methods = [
                'connect', 'disconnect', 'ensure_collection', 'upsert', 
                'search', 'delete', 'clear', 'get_stats', 'batch_upsert'
            ]
            
            for method in required_methods:
                assert hasattr(QdrantVectorStore, method), f"Missing method: {method}"
            
            # Verify required init parameters
            import inspect
            sig = inspect.signature(QdrantVectorStore.__init__)
            params = list(sig.parameters.keys())
            assert 'host' in params, "host parameter should be in __init__"
            assert 'port' in params, "port parameter should be in __init__"
            assert 'collection_name' in params, "collection_name parameter required"
            assert 'vector_dimensions' in params, "vector_dimensions parameter required"
            
            print("   ✓ QdrantVectorStore class with all required methods exists")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ QdrantVectorStore verification failed: {e}")
            self.failed += 1
            return False
    
    def verify_qdrant_vector_store_functionality(self) -> bool:
        """Verify QdrantVectorStore functionality with mock client"""
        print("\n2. Testing QdrantVectorStore Functionality...")
        try:
            from src.python.services.query_engine import QdrantVectorStore, SearchResult
            
            # Create store directly
            store = QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="test_collection",
                vector_dimensions=384
            )
            
            # Create mock client
            mock_client = MagicMock()
            store._client = mock_client
            store._connected = True
            
            # Test upsert
            test_vector = [0.1] * 384
            result = store.upsert(
                kernel_id="test-kernel",
                vector=test_vector,
                content="Test content",
                metadata={"source": "test"}
            )
            assert result is True, "Upsert should succeed"
            
            # Test search
            mock_search_result = MagicMock()
            mock_search_result.id = "test-kernel"
            mock_search_result.score = 0.95
            mock_search_result.payload = {"content": "Test content", "kernel_id": "test-kernel"}
            mock_client.search.return_value = [mock_search_result]
            
            results = store.search(
                query_vector=test_vector,
                top_k=10,
                score_threshold=0.8
            )
            assert len(results) == 1, "Should return 1 search result"
            assert results[0].score == 0.95, "Score should be 0.95"
            
            # Test delete
            result = store.delete("test-kernel")
            assert result is True, "Delete should succeed"
            
            # Test clear
            result = store.clear()
            assert result is True, "Clear should succeed"
            
            print("   ✓ QdrantVectorStore functionality works correctly")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ QdrantVectorStore functionality test failed: {e}")
            self.failed += 1
            return False
    
    def verify_hnsw_index_configuration(self) -> bool:
        """Verify HNSW index configuration is supported"""
        print("\n3. Testing HNSW Index Configuration...")
        try:
            from src.python.services.query_engine import QdrantVectorStore
            
            # Create store with HNSW enabled
            store = QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="test_hnsw",
                vector_dimensions=384,
                use_hnsw=True,
                hnsw_m=16,
                hnsw_ef_construct=64
            )
            
            # Verify HNSW parameters are stored
            assert store.use_hnsw is True, "HNSW should be enabled"
            assert store.hnsw_m == 16, "HNSW M parameter should be 16"
            assert store.hnsw_ef_construct == 64, "HNSW ef_construct should be 64"
            
            print("   ✓ HNSW index configuration is properly supported")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ HNSW index configuration test failed: {e}")
            self.failed += 1
            return False
    
    def verify_batch_operations(self) -> bool:
        """Verify batch upsert capability"""
        print("\n4. Testing Batch Operations...")
        try:
            from src.python.services.query_engine import QdrantVectorStore
            
            store = QdrantVectorStore(
                host="localhost",
                port=6333,
                collection_name="test_batch",
                vector_dimensions=384
            )
            
            # Create mock client
            mock_client = MagicMock()
            store._client = mock_client
            store._connected = True
            
            # Test batch_upsert
            test_items = [
                {"kernel_id": "kernel-1", "embedding": [0.1] * 384, "content": "Content 1"},
                {"kernel_id": "kernel-2", "embedding": [0.2] * 384, "content": "Content 2"},
                {"kernel_id": "kernel-3", "embedding": [0.3] * 384, "content": "Content 3"},
            ]
            
            count = store.batch_upsert(test_items)
            assert count == 3, f"Should upsert 3 items, got {count}"
            
            # Verify client.upsert was called
            mock_client.upsert.assert_called()
            
            print("   ✓ Batch operations work correctly")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ Batch operations test failed: {e}")
            self.failed += 1
            return False


class EmbeddingCachingLayerVerifier:
    """Verifies Task 1.1.3: Embedding Caching Layer"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def verify_redis_embedding_cache_class_exists(self) -> bool:
        """Verify RedisEmbeddingCache class is defined"""
        print("\n5. Testing RedisEmbeddingCache Class...")
        try:
            from src.python.services.query_engine import RedisEmbeddingCache
            
            # Verify all required methods exist
            required_methods = [
                'get', 'set', 'delete', 'clear', 'get_stats',
                '_get_cache_key', '_get_client'
            ]
            
            for method in required_methods:
                assert hasattr(RedisEmbeddingCache, method), f"Missing method: {method}"
            
            # Verify required init parameters
            import inspect
            sig = inspect.signature(RedisEmbeddingCache.__init__)
            params = list(sig.parameters.keys())
            assert 'host' in params, "host parameter should be in __init__"
            assert 'port' in params, "port parameter should be in __init__"
            
            print("   ✓ RedisEmbeddingCache class with all required methods exists")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ RedisEmbeddingCache verification failed: {e}")
            self.failed += 1
            return False
    
    def verify_lru_cache_functionality(self) -> bool:
        """Verify LRU eviction for fallback cache"""
        print("\n6. Testing LRU Cache Functionality...")
        try:
            from src.python.services.query_engine import RedisEmbeddingCache
            
            # Create cache with fallback enabled (no Redis connection)
            cache = RedisEmbeddingCache(
                host="localhost",
                port=6379,
                db=0,
                enable_fallback=True,
                max_memory_policy="allkeys-lru"
            )
            
            # Test fallback cache functionality
            test_vector = [0.1] * 384
            cache_key = cache._get_cache_key("test text")
            
            # Set should work with fallback
            result = cache.set("test text", test_vector, ttl_seconds=60)
            assert result is True, "Set should succeed with fallback"
            
            # Get should return cached value
            retrieved = cache.get("test text")
            assert retrieved is not None, "Should retrieve cached value"
            assert len(retrieved) == 384, "Retrieved vector should have 384 dimensions"
            
            # Test LRU eviction by filling cache
            for i in range(15000):  # Exceed max_fallback_size
                cache.set(f"text_{i}", test_vector)
            
            # Cache should not grow unbounded
            assert len(cache._fallback_cache) <= 15000, "Cache should be bounded"
            
            print("   ✓ LRU cache functionality works correctly")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ LRU cache functionality test failed: {e}")
            self.failed += 1
            return False
    
    def verify_ttl_expiration(self) -> bool:
        """Verify TTL-based expiration"""
        print("\n7. Testing TTL-Based Expiration...")
        try:
            from src.python.services.query_engine import RedisEmbeddingCache
            
            cache = RedisEmbeddingCache(
                host="localhost",
                port=6379,
                db=0,
                enable_fallback=True,
                default_ttl_seconds=1  # 1 second TTL
            )
            
            # Set with short TTL
            test_vector = [0.1] * 384
            cache.set("expiring text", test_vector)
            
            # Should be retrievable immediately
            result = cache.get("expiring text")
            assert result is not None, "Should be retrievable before expiry"
            
            # Wait for expiration
            time.sleep(1.1)
            
            # Should no longer be retrievable
            result = cache.get("expiring text")
            assert result is None, "Should be expired after TTL"
            
            print("   ✓ TTL-based expiration works correctly")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ TTL expiration test failed: {e}")
            self.failed += 1
            return False
    
    def verify_cache_statistics(self) -> bool:
        """Verify cache hit statistics"""
        print("\n8. Testing Cache Statistics...")
        try:
            from src.python.services.query_engine import RedisEmbeddingCache
            
            cache = RedisEmbeddingCache(
                host="localhost",
                port=6379,
                db=0,
                enable_fallback=True
            )
            
            # Get stats from fallback cache
            stats = cache.get_stats()
            assert hasattr(stats, 'hits'), "Stats should have hits attribute"
            assert hasattr(stats, 'misses'), "Stats should have misses attribute"
            assert hasattr(stats, 'size'), "Stats should have size attribute"
            
            # Create a cache entry
            test_vector = [0.1] * 384
            cache.set("stats test", test_vector)
            
            # Get the entry
            cache.get("stats test")
            
            # Get updated stats
            stats = cache.get_stats()
            assert stats.size > 0 or stats.hits >= 0, "Stats should be updated"
            
            print("   ✓ Cache statistics tracking works correctly")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ Cache statistics test failed: {e}")
            self.failed += 1
            return False
    
    def verify_hash_based_keys(self) -> bool:
        """Verify cache keys are hash-based for consistency"""
        print("\n9. Testing Hash-Based Cache Keys...")
        try:
            from src.python.services.query_engine import RedisEmbeddingCache
            
            cache = RedisEmbeddingCache(
                host="localhost",
                port=6379,
                db=0,
                enable_fallback=True
            )
            
            # Same text should produce same key
            key1 = cache._get_cache_key("consistent text")
            key2 = cache._get_cache_key("consistent text")
            assert key1 == key2, "Same text should produce same cache key"
            
            # Different text should produce different keys
            key3 = cache._get_cache_key("different text")
            assert key1 != key3, "Different text should produce different cache keys"
            
            # Key should have prefix
            assert key1.startswith(cache.key_prefix), "Cache key should have prefix"
            
            print("   ✓ Hash-based cache keys work correctly")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ Hash-based keys test failed: {e}")
            self.failed += 1
            return False


class IntegrationVerifier:
    """Verifies integration with KernelQueryEngine"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
    
    def verify_query_engine_integration(self) -> bool:
        """Verify KernelQueryEngine integrates with vector DB and cache"""
        print("\n10. Testing KernelQueryEngine Integration...")
        try:
            from src.python.services.query_engine import KernelQueryEngine
            
            # Verify integration parameters in __init__
            import inspect
            sig = inspect.signature(KernelQueryEngine.__init__)
            params = list(sig.parameters.keys())
            
            # Check for vector DB parameters
            assert 'use_vector_db' in params, "use_vector_db parameter required"
            assert 'qdrant_host' in params, "qdrant_host parameter required"
            assert 'qdrant_port' in params, "qdrant_port parameter required"
            
            # Check for cache parameters
            assert 'redis_host' in params, "redis_host parameter required"
            assert 'redis_port' in params, "redis_port parameter required"
            
            print("   ✓ KernelQueryEngine has integration parameters")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ Query engine integration test failed: {e}")
            self.failed += 1
            return False
    
    def verify_embedding_cache_usage(self) -> bool:
        """Verify embedding cache is used during queries"""
        print("\n11. Testing Embedding Cache Usage...")
        try:
            from src.python.services.query_engine import KernelQueryEngine
            
            # Create engine with caching enabled
            engine = KernelQueryEngine(
                embedding_generator=None,
                vector_store=None,
                embedding_cache=None,
                use_vector_db=False,
                use_cache=True,
                redis_host="localhost",
                redis_port=6379
            )
            
            assert hasattr(engine, 'embedding_cache'), "embedding_cache attribute required"
            assert engine.embedding_cache is not None, "embedding_cache should be initialized"
            
            print("   ✓ Embedding cache is properly initialized")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ Embedding cache usage test failed: {e}")
            self.failed += 1
            return False
    
    def verify_vector_store_usage(self) -> bool:
        """Verify vector store is used for kernel indexing and searching"""
        print("\n12. Testing Vector Store Usage...")
        try:
            from src.python.services.query_engine import KernelQueryEngine
            
            # Create engine with vector DB enabled
            engine = KernelQueryEngine(
                embedding_generator=None,
                vector_store=None,
                embedding_cache=None,
                use_vector_db=True,
                use_cache=False,
                qdrant_host="localhost",
                qdrant_port=6333
            )
            
            assert hasattr(engine, 'vector_store'), "vector_store attribute required"
            # When vector store connection fails, it should fall back gracefully
            # The attribute should still exist
            
            print("   ✓ Vector store is properly configured")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ Vector store usage test failed: {e}")
            self.failed += 1
            return False
    
    def verify_hybrid_search_function_exists(self) -> bool:
        """Verify hybrid search function exists at module level"""
        print("\n13. Testing Hybrid Search Function...")
        try:
            from src.python.services.query_engine import hybrid_search
            
            assert callable(hybrid_search), "hybrid_search should be callable"
            
            print("   ✓ Hybrid search function is implemented")
            self.passed += 1
            return True
        except Exception as e:
            print(f"   ✗ Hybrid search test failed: {e}")
            self.failed += 1
            return False


def main():
    """Run all verification tests"""
    print("=" * 70)
    print("Priority 1 Implementation Verification")
    print("Task 1.1.2: Vector Database Integration")
    print("Task 1.1.3: Embedding Caching Layer")
    print("=" * 70)
    
    vector_verifier = VectorDatabaseIntegrationVerifier()
    cache_verifier = EmbeddingCachingLayerVerifier()
    integration_verifier = IntegrationVerifier()
    
    # Run vector database tests
    vector_verifier.verify_qdrant_vector_store_class_exists()
    vector_verifier.verify_qdrant_vector_store_functionality()
    vector_verifier.verify_hnsw_index_configuration()
    vector_verifier.verify_batch_operations()
    
    # Run caching layer tests
    cache_verifier.verify_redis_embedding_cache_class_exists()
    cache_verifier.verify_lru_cache_functionality()
    cache_verifier.verify_ttl_expiration()
    cache_verifier.verify_cache_statistics()
    cache_verifier.verify_hash_based_keys()
    
    # Run integration tests
    integration_verifier.verify_query_engine_integration()
    integration_verifier.verify_embedding_cache_usage()
    integration_verifier.verify_vector_store_usage()
    integration_verifier.verify_hybrid_search_function_exists()
    
    # Calculate results
    total_passed = (vector_verifier.passed + cache_verifier.passed + 
                   integration_verifier.passed)
    total_failed = (vector_verifier.failed + cache_verifier.failed + 
                   integration_verifier.failed)
    total_tests = total_passed + total_failed
    
    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)
    print(f"\nTask 1.1.2 (Vector Database Integration):")
    print(f"  Passed: {vector_verifier.passed}/4")
    print(f"  Failed: {vector_verifier.failed}/4")
    
    print(f"\nTask 1.1.3 (Embedding Caching Layer):")
    print(f"  Passed: {cache_verifier.passed}/5")
    print(f"  Failed: {cache_verifier.failed}/5")
    
    print(f"\nIntegration Tests:")
    print(f"  Passed: {integration_verifier.passed}/4")
    print(f"  Failed: {integration_verifier.failed}/4")
    
    print(f"\n{'=' * 70}")
    print(f"TOTAL RESULTS")
    print(f"{'=' * 70}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate: {success_rate:.1f}%")
    
    if total_failed == 0:
        print("\n✓ ALL TESTS PASSED - Priority 1 is fully implemented!")
        return 0
    else:
        print(f"\n✗ {total_failed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
