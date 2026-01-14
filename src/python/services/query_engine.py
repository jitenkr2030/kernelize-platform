"""
KERNELIZE Platform - Query Engine
===================================

This module implements the semantic query engine for the KERNELIZE Platform.
It provides fast, accurate semantic search across compressed knowledge kernels
with support for multiple query types and optimization strategies.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Query type enumeration"""
    SEMANTIC = "semantic"     # Semantic search
    EXACT = "exact"           # Exact match
    FUZZY = "fuzzy"           # Fuzzy match
    HYBRID = "hybrid"         # Hybrid search


@dataclass
class QueryResult:
    """Query result"""
    kernel_id: str
    content: str
    similarity_score: float
    rank: int
    highlights: List[str]
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "similarity_score": self.similarity_score,
            "rank": self.rank,
            "highlights": self.highlights,
            "metadata": self.metadata,
        }


@dataclass
class QueryMetrics:
    """Query metrics"""
    query_time_ms: int
    total_results: int
    query_type: str
    cache_hit: bool
    embeddings_generated: bool


class CacheManager:
    """
    Cache Manager
    
    Provides query result caching and embedding vector caching with LRU eviction strategy.
    """
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache = {}
        self.access_order = []
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.hits += 1
                self.access_order.remove(key)
                self.access_order.append(key)
                return value
            else:
                del self.cache[key]
                self.access_order.remove(key)
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Set cached value"""
        if key in self.cache:
            self.access_order.remove(key)
        
        self.cache[key] = (value, time.time())
        self.access_order.append(key)
        
        # LRU eviction
        while len(self.cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
        }
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


class EmbeddingGenerator:
    """
    Embedding Vector Generator
    
    Generates semantic embedding vectors using sentence-transformers for accurate
    semantic similarity. Supports lazy loading, model caching, and fallback to
    hash-based embeddings for resilience.
    """
    
    # Class-level model instance for singleton pattern
    _model_instance = None
    _model_name = "all-MiniLM-L6-v2"
    _dimensions = 384
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimensions: int = 384):
        """
        Initialize the embedding generator.
        
        Args:
            model_name: Name of the sentence-transformers model to use
            dimensions: Expected embedding dimensions (default 384 for all-MiniLM-L6-v2)
        """
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None
        self._use_fallback = False
        self._fallback_calls = 0
        self._model_calls = 0
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
    
    def _load_model(self):
        """
        Load the sentence-transformers model lazily.
        
        Uses singleton pattern to ensure model is loaded only once.
        Falls back to hash-based embeddings if model loading fails.
        """
        if self._model is not None and not self._use_fallback:
            return True
            
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            self._use_fallback = False
            logger.info(f"Successfully loaded model: {self.model_name}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load sentence-transformers model: {e}")
            logger.warning("Falling back to hash-based embeddings")
            self._use_fallback = True
            self._model = None
            return False
    
    def generate(self, text: str) -> List[float]:
        """
        Generate text embedding vector using sentence-transformers.
        
        Args:
            text: Input text to generate embedding for
            
        Returns:
            List of floats representing the embedding vector (384 dimensions)
        """
        # Attempt to load model if not already loaded
        self._load_model()
        
        if not self._use_fallback and self._model is not None:
            try:
                # Use sentence-transformers for real semantic embeddings
                self._model_calls += 1
                embedding = self._model.encode(text, convert_to_numpy=True)
                # Ensure it's a list and has correct dimensions
                if embedding.ndim == 1:
                    embedding = embedding.tolist()
                else:
                    embedding = embedding.flatten().tolist()
                return embedding[:self.dimensions]
            except Exception as e:
                logger.error(f"Inference error with sentence-transformers: {e}")
                logger.warning("Falling back to hash-based embeddings")
                self._use_fallback = True
        
        # Fallback to hash-based embeddings
        return self._fallback_generate(text)
    
    def _fallback_generate(self, text: str) -> List[float]:
        """
        Generate hash-based embedding as fallback.
        
        This preserves backward compatibility when sentence-transformers
        is unavailable or fails.
        """
        self._fallback_calls += 1
        
        # Preprocess text
        text = text.lower().strip()
        words = re.findall(r'\b[a-z]{2,}\b', text)
        
        # Simulate word vectors (legacy method)
        vector = np.zeros(self.dimensions)
        for i, word in enumerate(words[:self.dimensions]):
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vector[i] = (hash_val % 1000) / 1000.0
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def batch_generate(self, texts: List[str]) -> List[List[float]]:
        """
        Batch generate embedding vectors for efficiency.
        
        Uses matrix operations for significant speedup over individual calls.
        
        Args:
            texts: List of input texts to generate embeddings for
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Attempt to load model if not already loaded
        self._load_model()
        
        if not self._use_fallback and self._model is not None:
            try:
                # Use batch encoding for efficiency
                self._model_calls += len(texts)
                embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                
                # Convert to list of lists with correct dimensions
                result = []
                for emb in embeddings:
                    if emb.ndim == 1:
                        result.append(emb[:self.dimensions].tolist())
                    else:
                        result.append(emb.flatten()[:self.dimensions].tolist())
                return result
            except Exception as e:
                logger.error(f"Batch inference error: {e}")
                logger.warning("Falling back to individual hash-based embeddings")
                self._use_fallback = True
        
        # Fallback to individual processing
        return [self._fallback_generate(text) for text in texts]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two embedding vectors.
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score between 0 and 1
        """
        if not vec1 or not vec2:
            return 0.0
        
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def semantic_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts.
        
        Convenience method that generates embeddings and calculates similarity.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Semantic similarity score between 0 and 1
        """
        emb1 = self.generate(text1)
        emb2 = self.generate(text2)
        return self.cosine_similarity(emb1, emb2)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get embedding generator statistics.
        
        Returns:
            Dictionary with model info, dimensions, and usage statistics
        """
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "status": "active" if not self._use_fallback else "fallback",
            "model_calls": self._model_calls,
            "fallback_calls": self._fallback_calls,
            "using_transformer": not self._use_fallback,
        }
    
    def clear_cache(self) -> None:
        """Clear any internal caches (placeholder for future caching implementations)"""
        pass


class ExactMatcher:
    """
    Exact Matcher
    
    Provides keyword-based exact and fuzzy matching functionality.
    """
    
    def __init__(self):
        self.index = {}
        logger.info("ExactMatcher initialized")
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """Build document index"""
        for doc in documents:
            kernel_id = doc.get("kernel_id")
            content = doc.get("compressed_content", "")
            
            # Extract terms
            words = self._tokenize(content)
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                if kernel_id not in self.index[word]:
                    self.index[word].append(kernel_id)
    
    def _tokenize(self, text: str) -> set:
        """Tokenize text"""
        text = text.lower()
        words = re.findall(r'\b[a-z]{2,}\b', text)
        return set(words)
    
    def exact_search(self, query: str) -> List[str]:
        """Exact search"""
        query_words = self._tokenize(query)
        results = {}
        
        for word in query_words:
            if word in self.index:
                for kernel_id in self.index[word]:
                    results[kernel_id] = results.get(kernel_id, 0) + 1
        
        # Sort by number of matching words
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [kernel_id for kernel_id, _ in sorted_results]
    
    def fuzzy_search(self, query: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """Fuzzy search (based on edit distance)"""
        query_words = self._tokenize(query)
        results = {}
        
        for indexed_word, kernel_ids in self.index.items():
            for query_word in query_words:
                distance = self._levenshtein_distance(query_word, indexed_word)
                if distance <= max_distance:
                    for kernel_id in kernel_ids:
                        results[kernel_id] = results.get(kernel_id, 0) + (1 - distance / max_distance)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [(kernel_id, int(score)) for kernel_id, score in sorted_results]
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def clear(self) -> None:
        """Clear index"""
        self.index.clear()


class HybridSearchEngine:
    """
    Hybrid Search Engine
    
    Combines the advantages of semantic search and exact search to provide
    optimal search results. Uses weighted fusion strategy to balance results
    from both search methods.
    """
    
    def __init__(self):
        self.semantic_weight = 0.6
        self.exact_weight = 0.4
        logger.info("HybridSearchEngine initialized")
    
    def search(
        self,
        query: str,
        semantic_results: List[Tuple[str, float]],
        exact_results: List[Tuple[str, int]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Fuse search results
        
        Combines semantic similarity and exact match scores to return final
        ranked results.
        """
        # Normalize scores
        max_semantic = max((s for _, s in semantic_results), default=1)
        max_exact = max((e for _, e in exact_results), default=1)
        
        normalized_semantic = {
            kernel_id: score / max_semantic
            for kernel_id, score in semantic_results
        }
        normalized_exact = {
            kernel_id: score / max_exact
            for kernel_id, score in exact_results
        }
        
        # Get all unique kernel IDs
        all_kernel_ids = set(normalized_semantic.keys()) | set(normalized_exact.keys())
        
        # Calculate fused scores
        fused_scores = {}
        for kernel_id in all_kernel_ids:
            semantic_score = normalized_semantic.get(kernel_id, 0)
            exact_score = normalized_exact.get(kernel_id, 0)
            
            fused_score = (
                self.semantic_weight * semantic_score +
                self.exact_weight * exact_score
            )
            fused_scores[kernel_id] = fused_score
        
        # Sort and return top_k
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def set_weights(self, semantic: float, exact: float) -> None:
        """Set fusion weights"""
        total = semantic + exact
        self.semantic_weight = semantic / total
        self.exact_weight = exact / total


class KernelQueryEngine:
    """
    Knowledge Kernel Query Engine
    
    Top-level query interface that integrates semantic search, exact matching,
    and hybrid search to provide unified query API and performance optimization.
    """
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.exact_matcher = ExactMatcher()
        self.hybrid_engine = HybridSearchEngine()
        self.cache = CacheManager(max_size=10000, ttl_hours=24)
        
        # Simulated knowledge base storage
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        
        logger.info("KernelQueryEngine initialized")
    
    def index_kernel(self, kernel_id: str, content: str, embedding: Optional[List[float]] = None) -> None:
        """
        Index knowledge kernel
        
        Adds knowledge kernel to search index.
        """
        kernel_data = {
            "kernel_id": kernel_id,
            "content": content,
            "embedding": embedding or self.embedding_generator.generate(content),
            "indexed_at": datetime.utcnow().isoformat(),
        }
        
        self.knowledge_base[kernel_id] = kernel_data
        self.exact_matcher.build_index([kernel_data])
        
        logger.info(f"Indexed kernel: {kernel_id}")
    
    def query(
        self,
        query_text: str,
        kernel_ids: Optional[List[str]] = None,
        query_type: QueryType = QueryType.SEMANTIC,
        top_k: int = 10,
        min_similarity: float = 0.0,
    ) -> Tuple[List[QueryResult], QueryMetrics]:
        """
        Execute query
        
        Executes appropriate search strategy based on specified query type.
        
        Args:
            query_text: Query text
            kernel_ids: Optional, list of kernel IDs to query
            query_type: Query type
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
        
        Returns:
            (Query result list, Query metrics)
        """
        start_time = time.time()
        
        # Check cache
        cache_key = f"{query_type.value}:{query_text}:{kernel_ids}:{top_k}"
        cached_result = self.cache.get(cache_key)
        
        if cached_result:
            return cached_result, QueryMetrics(
                query_time_ms=0,
                total_results=0,
                query_type=query_type.value,
                cache_hit=True,
                embeddings_generated=False,
            )
        
        # Filter knowledge base
        if kernel_ids:
            filtered_kernels = {
                k: v for k, v in self.knowledge_base.items()
                if k in kernel_ids
            }
        else:
            filtered_kernels = self.knowledge_base
        
        if not filtered_kernels:
            return [], QueryMetrics(
                query_time_ms=int((time.time() - start_time) * 1000),
                total_results=0,
                query_type=query_type.value,
                cache_hit=False,
                embeddings_generated=False,
            )
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate(query_text)
        
        # Execute search based on query type
        if query_type == QueryType.SEMANTIC:
            results = self._semantic_search(query_text, query_embedding, filtered_kernels, top_k)
        elif query_type == QueryType.EXACT:
            results = self._exact_search(query_text, filtered_kernels, top_k)
        elif query_type == QueryType.FUZZY:
            results = self._fuzzy_search(query_text, filtered_kernels, top_k)
        else:  # HYBRID
            results = self._hybrid_search(query_text, query_embedding, filtered_kernels, top_k)
        
        # Generate highlights
        for result in results:
            result.highlights = self._generate_highlights(query_text, result.content)
        
        # Calculate metrics
        query_time_ms = int((time.time() - start_time) * 1000)
        metrics = QueryMetrics(
            query_time_ms=query_time_ms,
            total_results=len(results),
            query_type=query_type.value,
            cache_hit=False,
            embeddings_generated=True,
        )
        
        # Cache results
        self.cache.set(cache_key, results)
        
        return results, metrics
    
    def _semantic_search(
        self,
        query_text: str,
        query_embedding: List[float],
        kernels: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> List[QueryResult]:
        """Semantic search"""
        similarities = []
        
        for kernel_id, kernel_data in kernels.items():
            embedding = kernel_data.get("embedding", [])
            if embedding:
                similarity = self.embedding_generator.cosine_similarity(query_embedding, embedding)
                similarities.append((kernel_id, similarity))
        
        # Sort
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for rank, (kernel_id, score) in enumerate(similarities[:top_k]):
            kernel_data = kernels[kernel_id]
            results.append(QueryResult(
                kernel_id=kernel_id,
                content=kernel_data["content"],
                similarity_score=score,
                rank=rank + 1,
                highlights=[],
                metadata=kernel_data.get("metadata", {}),
            ))
        
        return results
    
    def _exact_search(
        self,
        query_text: str,
        kernels: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> List[QueryResult]:
        """Exact search"""
        # Update index
        self.exact_matcher.build_index(list(kernels.values()))
        
        # Execute exact search
        matched_ids = self.exact_matcher.exact_search(query_text)
        
        results = []
        for rank, kernel_id in enumerate(matched_ids[:top_k]):
            if kernel_id in kernels:
                kernel_data = kernels[kernel_id]
                results.append(QueryResult(
                    kernel_id=kernel_id,
                    content=kernel_data["content"],
                    similarity_score=1.0,
                    rank=rank + 1,
                    highlights=[],
                    metadata=kernel_data.get("metadata", {}),
                ))
        
        return results
    
    def _fuzzy_search(
        self,
        query_text: str,
        kernels: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> List[QueryResult]:
        """Fuzzy search"""
        # Update index
        self.exact_matcher.build_index(list(kernels.values()))
        
        # Execute fuzzy search
        fuzzy_results = self.exact_matcher.fuzzy_search(query_text)
        
        results = []
        for rank, (kernel_id, score) in enumerate(fuzzy_results[:top_k]):
            if kernel_id in kernels:
                kernel_data = kernels[kernel_id]
                results.append(QueryResult(
                    kernel_id=kernel_id,
                    content=kernel_data["content"],
                    similarity_score=score / 100.0,  # Normalize
                    rank=rank + 1,
                    highlights=[],
                    metadata=kernel_data.get("metadata", {}),
                ))
        
        return results
    
    def _hybrid_search(
        self,
        query_text: str,
        query_embedding: List[float],
        kernels: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> List[QueryResult]:
        """Hybrid search"""
        # Execute both searches
        semantic_results = []
        for kernel_id, kernel_data in kernels.items():
            embedding = kernel_data.get("embedding", [])
            if embedding:
                similarity = self.embedding_generator.cosine_similarity(query_embedding, embedding)
                semantic_results.append((kernel_id, similarity))
        
        # Update index and execute exact search
        self.exact_matcher.build_index(list(kernels.values()))
        exact_results = self.exact_matcher.exact_search(query_text)
        exact_scores = [(kid, 1) for kid in exact_results]
        
        # Fuse results
        fused = self.hybrid_engine.search(
            query_text,
            semantic_results,
            exact_scores,
            top_k,
        )
        
        results = []
        for rank, (kernel_id, score) in enumerate(fused):
            if kernel_id in kernels:
                kernel_data = kernels[kernel_id]
                results.append(QueryResult(
                    kernel_id=kernel_id,
                    content=kernel_data["content"],
                    similarity_score=score,
                    rank=rank + 1,
                    highlights=[],
                    metadata=kernel_data.get("metadata", {}),
                ))
        
        return results
    
    def _generate_highlights(self, query: str, content: str) -> List[str]:
        """Generate query highlights"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        highlights = []
        
        sentences = re.split(r'[.!?]+', content)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            matched_words = query_words & set(re.findall(r'\b\w+\b', sentence_lower))
            if matched_words:
                highlights.append(sentence.strip())
        
        return highlights[:3]
    
    def delete_kernel(self, kernel_id: str) -> bool:
        """Delete knowledge kernel"""
        if kernel_id in self.knowledge_base:
            del self.knowledge_base[kernel_id]
            self.exact_matcher.clear()
            self.exact_matcher.build_index(list(self.knowledge_base.values()))
            return True
        return False
    
    def clear_index(self) -> None:
        """Clear index"""
        self.knowledge_base.clear()
        self.exact_matcher.clear()
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        return {
            "indexed_kernels": len(self.knowledge_base),
            "cache_stats": self.cache.get_stats(),
            "embedding_stats": self.embedding_generator.get_stats(),
            "search_types": [e.value for e in QueryType],
        }


# Create global query engine instance
query_engine = KernelQueryEngine()


# Convenience functions
def semantic_search(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.0,
) -> Tuple[List[QueryResult], QueryMetrics]:
    """Quick semantic search function"""
    return query_engine.query(
        query_text=query,
        query_type=QueryType.SEMANTIC,
        top_k=top_k,
        min_similarity=min_similarity,
    )


def hybrid_search(
    query: str,
    top_k: int = 10,
) -> Tuple[List[QueryResult], QueryMetrics]:
    """Quick hybrid search function"""
    return query_engine.query(
        query_text=query,
        query_type=QueryType.HYBRID,
        top_k=top_k,
    )
