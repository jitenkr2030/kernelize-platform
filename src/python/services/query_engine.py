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
import os
import pickle
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

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
class SearchResult:
    """Search result from vector database"""
    kernel_id: str
    score: float
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CacheStats:
    """Cache statistics"""
    hits: int = 0
    misses: int = 0
    size: int = 0
    hit_rate: float = 0.0


@dataclass
class VectorStoreStats:
    """Vector store statistics"""
    total_vectors: int = 0
    collection_name: str = ""
    status: str = "unknown"
    index_type: str = ""


@dataclass
class QueryMetrics:
    """Query metrics"""
    query_time_ms: int = 0
    total_results: int = 0
    query_type: str = ""
    cache_hit: bool = False
    embeddings_generated: bool = False
    cache_stats: Optional[CacheStats] = None
    vector_stats: Optional[VectorStoreStats] = None


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


# ============================================================================
# Vector Database Integration (Task 1.1.2)
# ============================================================================

class VectorStoreAdapter(ABC):
    """
    Abstract Base Class for Vector Database Adapters
    
    Provides a unified interface for vector database operations, enabling
    support for multiple vector databases (Qdrant, Weaviate, Milvus, etc.)
    through a consistent API.
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the vector database"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection to the vector database"""
        pass
    
    @abstractmethod
    def ensure_collection(self, collection_name: str, vector_dimensions: int) -> bool:
        """
        Ensure collection exists with proper configuration
        
        Args:
            collection_name: Name of the collection
            vector_dimensions: Dimensionality of embedding vectors
            
        Returns:
            True if collection exists or was created successfully
        """
        pass
    
    @abstractmethod
    def upsert(
        self,
        kernel_id: str,
        vector: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Insert or update a kernel embedding in the database
        
        Args:
            kernel_id: Unique identifier for the kernel
            vector: Embedding vector
            content: Original text content
            metadata: Optional additional metadata
            
        Returns:
            True if operation succeeded
        """
        pass
    
    @abstractmethod
    def batch_upsert(
        self,
        items: List[Dict[str, Any]],
        vector_field: str = "embedding",
        id_field: str = "kernel_id"
    ) -> int:
        """
        Batch insert or update multiple kernel embeddings
        
        Args:
            items: List of kernel dictionaries
            vector_field: Field name containing the embedding
            id_field: Field name containing the kernel ID
            
        Returns:
            Number of successfully inserted/updated items
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """
        Perform ANN (Approximate Nearest Neighbor) search
        
        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filter_ids: Optional list of kernel IDs to filter by
            
        Returns:
            List of SearchResult objects ranked by similarity
        """
        pass
    
    @abstractmethod
    def delete(self, kernel_id: str) -> bool:
        """Delete a kernel from the database"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all data from the collection"""
        pass
    
    @abstractmethod
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics"""
        pass


class QdrantVectorStore(VectorStoreAdapter):
    """
    Qdrant Vector Database Adapter
    
    Implements vector storage and ANN search using Qdrant, a high-performance
    vector database written in Rust. Uses HNSW index for efficient similarity search.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        collection_name: str = "kernel_vectors",
        vector_dimensions: int = 384,
        use_hnsw: bool = True,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 128,
        on_disk: bool = False,
    ):
        """
        Initialize Qdrant vector store adapter
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for authentication
            collection_name: Name of the collection to use
            vector_dimensions: Embedding vector dimensions
            use_hnsw: Use HNSW index for faster search
            hnsw_m: HNSW M parameter (higher = better recall, more memory)
            hnsw_ef_construct: HNSW ef_construct parameter
            on_disk: Store vectors on disk instead of memory
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.collection_name = collection_name
        self.vector_dimensions = vector_dimensions
        self.use_hnsw = use_hnsw
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct
        self.on_disk = on_disk
        
        self._client = None
        self._connected = False
        
        logger.info(f"QdrantVectorStore initialized: {host}:{port}/{collection_name}")
    
    def _get_client(self):
        """Get or create Qdrant client"""
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
                from qdrant_client.models import Distance, VectorParams, PointStruct
                
                # Create client with appropriate configuration
                if self.api_key:
                    self._client = QdrantClient(
                        host=self.host,
                        port=self.port,
                        api_key=self.api_key,
                    )
                else:
                    self._client = QdrantClient(
                        host=self.host,
                        port=self.port,
                    )
            except ImportError:
                logger.error("qdrant-client not installed. Install with: pip install qdrant-client")
                raise ImportError("Qdrant client library not installed")
            except Exception as e:
                logger.error(f"Failed to connect to Qdrant: {e}")
                raise
        
        return self._client
    
    def connect(self) -> bool:
        """Establish connection to Qdrant"""
        try:
            client = self._get_client()
            # Test connection by getting collections
            collections = client.get_collections()
            logger.info(f"Connected to Qdrant: {len(collections.collections)} collections")
            self._connected = True
            
            # Ensure collection exists
            self.ensure_collection(self.collection_name, self.vector_dimensions)
            
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Qdrant: {e}")
            self._connected = False
            return False
    
    def disconnect(self) -> None:
        """Close connection to Qdrant"""
        if self._client:
            # Qdrant client doesn't have explicit disconnect, just cleanup
            self._client = None
            self._connected = False
            logger.info("Disconnected from Qdrant")
    
    def ensure_collection(self, collection_name: str, vector_dimensions: int) -> bool:
        """Ensure collection exists with proper configuration"""
        try:
            client = self._get_client()
            from qdrant_client.models import Distance, VectorParams
            
            # Check if collection exists
            collections = client.get_collections()
            collection_names = [c.name for c in collections.collections]
            
            if collection_name not in collection_names:
                # Create collection with HNSW index
                if self.use_hnsw:
                    # HNSW index configuration
                    hnsw_config = {
                        "m": self.hnsw_m,
                        "ef_construct": self.hnsw_ef_construct,
                        "full_scan_threshold": 10000,
                    }
                else:
                    hnsw_config = None
                
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_dimensions,
                        distance=Distance.COSINE,
                        on_disk=self.on_disk,
                        hnsw_config=hnsw_config,
                    ),
                )
                logger.info(f"Created Qdrant collection: {collection_name}")
            else:
                logger.debug(f"Collection {collection_name} already exists")
            
            return True
        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            return False
    
    def upsert(
        self,
        kernel_id: str,
        vector: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Insert or update a kernel embedding"""
        try:
            from qdrant_client.models import PointStruct
            
            client = self._get_client()
            
            point = PointStruct(
                id=kernel_id,
                vector=vector,
                payload={
                    "content": content,
                    "kernel_id": kernel_id,
                    **(metadata or {}),
                }
            )
            
            client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )
            
            logger.debug(f"Upserted kernel: {kernel_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to upsert kernel {kernel_id}: {e}")
            return False
    
    def batch_upsert(
        self,
        items: List[Dict[str, Any]],
        vector_field: str = "embedding",
        id_field: str = "kernel_id"
    ) -> int:
        """Batch insert or update multiple kernel embeddings"""
        try:
            from qdrant_client.models import PointStruct
            
            if not items:
                return 0
            
            client = self._get_client()
            
            points = []
            for item in items:
                kernel_id = item.get(id_field)
                vector = item.get(vector_field, [])
                content = item.get("content", "")
                metadata = {k: v for k, v in item.items() 
                           if k not in [id_field, vector_field, "content"]}
                
                if kernel_id and vector:
                    points.append(PointStruct(
                        id=kernel_id,
                        vector=vector,
                        payload={
                            "content": content,
                            "kernel_id": kernel_id,
                            **metadata,
                        }
                    ))
            
            if points:
                client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
                logger.info(f"Batch upserted {len(points)} kernels")
            
            return len(points)
        except Exception as e:
            logger.error(f"Batch upsert failed: {e}")
            return 0
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Perform ANN search using Qdrant"""
        try:
            client = self._get_client()
            from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
            
            # Build filter if ids provided
            search_filter = None
            if filter_ids:
                # MatchAny expects list of strings, but we need to convert properly
                # Using Filter with conditions
                conditions = []
                for fid in filter_ids:
                    conditions.append(
                        FieldCondition(
                            key="kernel_id",
                            match=MatchValue(value=fid),
                        )
                    )
                # If multiple IDs, use should (OR) logic
                if len(conditions) > 1:
                    search_filter = Filter(should=conditions)
                elif len(conditions) == 1:
                    search_filter = Filter(must=conditions)
            
            # Search with optional filtering - use new API
            search_args = {
                "collection_name": self.collection_name,
                "query": query_vector,  # Use 'query' instead of 'query_vector'
                "limit": top_k,
            }
            
            if search_filter:
                search_args["query_filter"] = search_filter
            
            if score_threshold:
                search_args["score_threshold"] = score_threshold
            
            # Use search method with proper arguments
            results = client.search(**search_args)
            
            search_results = []
            for hit in results:
                payload = hit.payload or {}
                search_results.append(SearchResult(
                    kernel_id=str(hit.id),
                    score=hit.score,
                    content=payload.get("content"),
                    metadata={k: v for k, v in payload.items() 
                             if k not in ["content", "kernel_id"]},
                ))
            
            logger.debug(f"Qdrant search returned {len(search_results)} results")
            return search_results
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def delete(self, kernel_id: str) -> bool:
        """Delete a kernel from the database"""
        try:
            client = self._get_client()
            from qdrant_client.models import PointIdsList
            client.delete(
                collection_name=self.collection_name,
                points=PointIdsList(points=[kernel_id]),
            )
            logger.debug(f"Deleted kernel: {kernel_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete kernel {kernel_id}: {e}")
            return False
    
    def clear(self) -> bool:
        """Clear all data from the collection"""
        try:
            client = self._get_client()
            client.delete_collection(self.collection_name)
            logger.info(f"Cleared collection: {self.collection_name}")
            # Recreate the collection
            self.ensure_collection(self.collection_name, self.vector_dimensions)
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics"""
        try:
            client = self._get_client()
            collection_info = client.get_collection(self.collection_name)
            
            # Get points count
            try:
                points_count = collection_info.points_count
            except AttributeError:
                # Try alternative attribute
                points_count = getattr(collection_info, 'points_count', 0) or 0
            
            return VectorStoreStats(
                total_vectors=points_count,
                collection_name=self.collection_name,
                status="connected" if self._connected else "disconnected",
                index_type="HNSW" if self.use_hnsw else "flat",
            )
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return VectorStoreStats(
                status="error",
                collection_name=self.collection_name,
            )


class InMemoryVectorStore(VectorStoreAdapter):
    """
    In-Memory Vector Store (Fallback)
    
    Simple in-memory implementation for development and testing when
    a full vector database is not available.
    """
    
    def __init__(
        self,
        collection_name: str = "kernel_vectors",
        vector_dimensions: int = 384,
    ):
        """
        Initialize in-memory vector store
        
        Args:
            collection_name: Name of the collection (for compatibility)
            vector_dimensions: Embedding vector dimensions
        """
        self.collection_name = collection_name
        self.vector_dimensions = vector_dimensions
        self._vectors: Dict[str, Dict[str, Any]] = {}
        self._connected = True
        
        logger.info("InMemoryVectorStore initialized (fallback mode)")
    
    def connect(self) -> bool:
        """Always connected in memory"""
        self._connected = True
        logger.info("Connected to in-memory vector store")
        return True
    
    def disconnect(self) -> None:
        """No-op for in-memory store"""
        self._connected = False
        logger.info("Disconnected from in-memory vector store")
    
    def ensure_collection(self, collection_name: str, vector_dimensions: int) -> bool:
        """No-op for in-memory store"""
        return True
    
    def upsert(
        self,
        kernel_id: str,
        vector: List[float],
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Store vector in memory"""
        self._vectors[kernel_id] = {
            "vector": vector,
            "content": content,
            "metadata": metadata or {},
        }
        return True
    
    def batch_upsert(
        self,
        items: List[Dict[str, Any]],
        vector_field: str = "embedding",
        id_field: str = "kernel_id"
    ) -> int:
        """Batch store vectors in memory"""
        count = 0
        for item in items:
            kernel_id = item.get(id_field)
            vector = item.get(vector_field, [])
            content = item.get("content", "")
            metadata = {k: v for k, v in item.items() 
                       if k not in [id_field, vector_field, "content"]}
            
            if kernel_id and vector:
                self._vectors[kernel_id] = {
                    "vector": vector,
                    "content": content,
                    "metadata": metadata,
                }
                count += 1
        
        logger.info(f"Batch upserted {count} vectors to in-memory store")
        return count
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        filter_ids: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Brute-force search (for small datasets)"""
        import numpy as np
        
        query_vec = np.array(query_vector)
        results = []
        
        for kernel_id, data in self._vectors.items():
            # Skip if filtering by IDs
            if filter_ids and kernel_id not in filter_ids:
                continue
            
            vector = np.array(data["vector"])
            
            # Calculate cosine similarity
            norm_query = np.linalg.norm(query_vec)
            norm_vector = np.linalg.norm(vector)
            
            if norm_query > 0 and norm_vector > 0:
                score = float(np.dot(query_vec, vector) / (norm_query * norm_vector))
            else:
                score = 0.0
            
            # Apply threshold
            if score_threshold is None or score >= score_threshold:
                results.append(SearchResult(
                    kernel_id=kernel_id,
                    score=score,
                    content=data.get("content"),
                    metadata=data.get("metadata"),
                ))
        
        # Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:top_k]
    
    def delete(self, kernel_id: str) -> bool:
        """Delete vector from memory"""
        if kernel_id in self._vectors:
            del self._vectors[kernel_id]
            return True
        return False
    
    def clear(self) -> bool:
        """Clear all vectors from memory"""
        self._vectors.clear()
        logger.info("Cleared in-memory vector store")
        return True
    
    def get_stats(self) -> VectorStoreStats:
        """Get vector store statistics"""
        return VectorStoreStats(
            total_vectors=len(self._vectors),
            collection_name=self.collection_name,
            status="connected" if self._connected else "disconnected",
            index_type="brute-force",
        )


# ============================================================================
# Embedding Caching Layer (Task 1.1.3)
# ============================================================================

class EmbeddingCache(ABC):
    """
    Abstract Base Class for Embedding Cache
    
    Provides a unified interface for embedding cache implementations,
    supporting both Redis-based distributed caching and in-memory caching.
    """
    
    @abstractmethod
    def get(self, text: str) -> Optional[List[float]]:
        """
        Retrieve cached embedding for text
        
        Args:
            text: Input text to look up
            
        Returns:
            Cached embedding vector or None if not found
        """
        pass
    
    @abstractmethod
    def set(self, text: str, vector: List[float], ttl_seconds: Optional[int] = None) -> bool:
        """
        Store embedding in cache
        
        Args:
            text: Input text
            embedding: Vector to cache
            ttl_seconds: Optional TTL in seconds
            
        Returns:
            True if operation succeeded
        """
        pass
    
    @abstractmethod
    def delete(self, text: str) -> bool:
        """Delete cached embedding for text"""
        pass
    
    @abstractmethod
    def clear(self) -> bool:
        """Clear all cached embeddings"""
        pass
    
    @abstractmethod
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        pass


class RedisEmbeddingCache(EmbeddingCache):
    """
    Redis-based Embedding Cache
    
    Provides high-performance distributed caching using Redis.
    Supports TTL-based expiration and LRU eviction when memory is full.
    Falls back to in-memory caching if Redis is unavailable.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "kernel_emb:",
        max_memory: str = "1gb",
        max_memory_policy: str = "allkeys-lru",
        default_ttl_seconds: int = 86400,  # 24 hours
        enable_fallback: bool = True,
    ):
        """
        Initialize Redis embedding cache
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Optional Redis password
            key_prefix: Prefix for cache keys
            max_memory: Redis max memory setting
            max_memory_policy: Eviction policy (allkeys-lru, volatile-lru, etc.)
            default_ttl_seconds: Default TTL for cached embeddings
            enable_fallback: Fall back to in-memory if Redis unavailable
        """
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.key_prefix = key_prefix
        self.default_ttl_seconds = default_ttl_seconds
        self.enable_fallback = enable_fallback
        self.max_memory = max_memory
        self.max_memory_policy = max_memory_policy
        
        self._client = None
        self._fallback_cache: Dict[str, Tuple[List[float], float]] = {}
        self._fallback_hits = 0
        self._fallback_misses = 0
        self._using_fallback = False
        self._using_redis = False
        
        logger.info(f"RedisEmbeddingCache initialized: {host}:{port}/{db}")
    
    def _get_client(self):
        """Get or create Redis client"""
        if self._client is None:
            try:
                import redis
                self._client = redis.Redis(
                    host=self.host,
                    port=self.port,
                    db=self.db,
                    password=self.password,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                )
                # Test connection
                self._client.ping()
                logger.info("Connected to Redis")
                self._using_redis = True
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                if self.enable_fallback:
                    logger.warning("Falling back to in-memory cache")
                    self._using_fallback = True
                else:
                    raise
        return self._client
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text using SHA256 hash"""
        hash_value = hashlib.sha256(text.encode('utf-8')).hexdigest()
        return f"{self.key_prefix}{hash_value}"
    
    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding"""
        cache_key = self._get_cache_key(text)
        
        # Try Redis first
        if self._using_redis and self._client:
            try:
                data = self._client.get(cache_key)
                if data:
                    # Deserialize vector
                    vector = pickle.loads(data)
                    return vector
            except Exception as e:
                logger.warning(f"Redis get failed: {e}, using fallback")
                self._using_redis = False
        
        # Use fallback cache
        if self.enable_fallback:
            if cache_key in self._fallback_cache:
                vector, timestamp = self._fallback_cache[cache_key]
                # Check TTL
                if time.time() - timestamp < self.default_ttl_seconds:
                    self._fallback_hits += 1
                    return vector
                else:
                    del self._fallback_cache[cache_key]
            self._fallback_misses += 1
        
        return None
    
    def set(self, text: str, vector: List[float], ttl_seconds: Optional[int] = None) -> bool:
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text)
        ttl = ttl_seconds or self.default_ttl_seconds
        
        # Try Redis first
        if self._using_redis and self._client:
            try:
                # Serialize vector
                data = pickle.dumps(vector)
                self._client.setex(cache_key, ttl, data)
                return True
            except Exception as e:
                logger.warning(f"Redis set failed: {e}, using fallback")
                self._using_redis = False
        
        # Use fallback cache
        if self.enable_fallback:
            self._fallback_cache[cache_key] = (vector, time.time())
            
            # LRU eviction if cache is too large
            max_fallback_size = 10000
            if len(self._fallback_cache) > max_fallback_size:
                # Remove oldest entries
                oldest_keys = sorted(
                    self._fallback_cache.keys(),
                    key=lambda k: self._fallback_cache[k][1]
                )[:max_fallback_size // 10]
                for key in oldest_keys:
                    del self._fallback_cache[key]
            
            return True
        
        return False
    
    def delete(self, text: str) -> bool:
        """Delete cached embedding"""
        cache_key = self._get_cache_key(text)
        
        # Try Redis first
        if self._using_redis and self._client:
            try:
                self._client.delete(cache_key)
                return True
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Use fallback
        if self.enable_fallback:
            if cache_key in self._fallback_cache:
                del self._fallback_cache[cache_key]
                return True
        
        return False
    
    def clear(self) -> bool:
        """Clear all cached embeddings"""
        # Try Redis first
        if self._using_redis and self._client:
            try:
                # Delete only keys with our prefix
                pattern = f"{self.key_prefix}*"
                cursor = 0
                while True:
                    cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                    if keys:
                        self._client.delete(*keys)
                    if cursor == 0:
                        break
                logger.info("Cleared Redis cache")
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        # Clear fallback
        if self.enable_fallback:
            self._fallback_cache.clear()
            self._fallback_hits = 0
            self._fallback_misses = 0
            logger.info("Cleared fallback cache")
        
        return True
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        # Get Redis stats
        redis_hits = 0
        redis_misses = 0
        redis_size = 0
        
        if self._using_redis and self._client:
            try:
                info = self._client.info("stats")
                redis_hits = info.get("keyspace_hits", 0)
                redis_misses = info.get("keyspace_misses", 0)
                
                # Count our keys
                pattern = f"{self.key_prefix}*"
                cursor = 0
                count = 0
                while True:
                    cursor, keys = self._client.scan(cursor, match=pattern, count=100)
                    count += len(keys)
                    if cursor == 0:
                        break
                redis_size = count
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")
        
        # Combine with fallback stats
        total_hits = redis_hits + self._fallback_hits
        total_misses = redis_misses + self._fallback_misses
        total_size = redis_size + len(self._fallback_cache)
        
        total_requests = total_hits + total_misses
        hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return CacheStats(
            hits=total_hits,
            misses=total_misses,
            size=total_size,
            hit_rate=hit_rate,
        )
    
    def is_using_fallback(self) -> bool:
        """Check if cache is using fallback mode"""
        return self._using_fallback


class InMemoryEmbeddingCache(EmbeddingCache):
    """
    In-Memory Embedding Cache (Fallback)
    
    Simple in-memory LRU cache for embedding vectors.
    Used as fallback when Redis is unavailable.
    """
    
    def __init__(
        self,
        max_size: int = 10000,
        default_ttl_seconds: int = 86400,  # 24 hours
    ):
        """
        Initialize in-memory embedding cache
        
        Args:
            max_size: Maximum number of cached embeddings
            default_ttl_seconds: Default TTL for cached embeddings
        """
        self.max_size = max_size
        self.default_ttl_seconds = default_ttl_seconds
        
        self._cache: Dict[str, Tuple[List[float], float]] = {}
        self._access_order: List[str] = []
        self._hits = 0
        self._misses = 0
        
        logger.info(f"InMemoryEmbeddingCache initialized (max_size={max_size})")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text using SHA256 hash"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Retrieve cached embedding"""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._cache:
            vector, timestamp = self._cache[cache_key]
            # Check TTL
            if time.time() - timestamp < self.default_ttl_seconds:
                self._hits += 1
                # Update access order (LRU)
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
                self._access_order.append(cache_key)
                return vector
            else:
                # Expired
                del self._cache[cache_key]
                if cache_key in self._access_order:
                    self._access_order.remove(cache_key)
        
        self._misses += 1
        return None
    
    def set(self, text: str, vector: List[float], ttl_seconds: Optional[int] = None) -> bool:
        """Store embedding in cache"""
        cache_key = self._get_cache_key(text)
        ttl = ttl_seconds or self.default_ttl_seconds
        
        # Remove old entry if exists
        if cache_key in self._cache:
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
        
        # Add new entry
        self._cache[cache_key] = (vector, time.time() + ttl)
        self._access_order.append(cache_key)
        
        # LRU eviction
        while len(self._cache) > self.max_size:
            if self._access_order:
                oldest_key = self._access_order.pop(0)
                if oldest_key in self._cache:
                    del self._cache[oldest_key]
        
        return True
    
    def delete(self, text: str) -> bool:
        """Delete cached embedding"""
        cache_key = self._get_cache_key(text)
        
        if cache_key in self._cache:
            del self._cache[cache_key]
            if cache_key in self._access_order:
                self._access_order.remove(cache_key)
            return True
        
        return False
    
    def clear(self) -> bool:
        """Clear all cached embeddings"""
        self._cache.clear()
        self._access_order.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cleared in-memory cache")
        return True
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics"""
        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests * 100) if total_requests > 0 else 0.0
        
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size=len(self._cache),
            hit_rate=hit_rate,
        )


class EmbeddingCacheManager:
    """
    Embedding Cache Manager
    
    Wrapper that provides unified access to embedding cache with
    automatic fallback between Redis and in-memory implementations.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        key_prefix: str = "kernel_emb:",
        default_ttl_seconds: int = 86400,
        enable_redis: bool = True,
        in_memory_max_size: int = 10000,
    ):
        """
        Initialize cache manager with automatic fallback
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            redis_password: Optional Redis password
            key_prefix: Prefix for cache keys
            default_ttl_seconds: Default TTL for cached embeddings
            enable_redis: Try Redis first, fall back to in-memory
            in_memory_max_size: Fallback cache maximum size
        """
        self._redis_cache = None
        self._in_memory_cache = None
        self._use_redis = False
        self._enable_redis = enable_redis
        self._default_ttl = default_ttl_seconds
        self._key_prefix = key_prefix
        
        # Initialize in-memory cache (always available as fallback)
        self._in_memory_cache = InMemoryEmbeddingCache(
            max_size=in_memory_max_size,
            default_ttl_seconds=default_ttl_seconds,
        )
        
        # Try to initialize Redis cache
        if enable_redis:
            try:
                self._redis_cache = RedisEmbeddingCache(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    key_prefix=key_prefix,
                    default_ttl_seconds=default_ttl_seconds,
                    enable_fallback=False,  # We have our own fallback
                )
                # Test connection
                self._redis_cache.get("test_connection")
                self._use_redis = True
                logger.info("Redis cache manager initialized (Redis active)")
            except Exception as e:
                logger.warning(f"Redis cache initialization failed: {e}")
                logger.warning("Using in-memory cache only")
                self._use_redis = False
        else:
            logger.info("Redis disabled, using in-memory cache only")
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache"""
        # Try Redis first if available
        if self._use_redis and self._redis_cache:
            try:
                vector = self._redis_cache.get(text)
                if vector is not None:
                    return vector
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
                self._use_redis = False
        
        # Fall back to in-memory
        if self._in_memory_cache:
            return self._in_memory_cache.get(text)
        
        return None
    
    def set(self, text: str, vector: List[float], ttl_seconds: Optional[int] = None) -> bool:
        """Store embedding in cache"""
        ttl = ttl_seconds or self._default_ttl
        
        # Try Redis first if available
        if self._use_redis and self._redis_cache:
            try:
                if self._redis_cache.set(text, vector, ttl):
                    return True
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
                self._use_redis = False
        
        # Fall back to in-memory
        if self._in_memory_cache:
            return self._in_memory_cache.set(text, vector, ttl)
        
        return False
    
    def delete(self, text: str) -> bool:
        """Delete embedding from cache"""
        # Try Redis first if available
        if self._use_redis and self._redis_cache:
            try:
                self._redis_cache.delete(text)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Also delete from in-memory
        if self._in_memory_cache:
            self._in_memory_cache.delete(text)
        
        return True
    
    def clear(self) -> bool:
        """Clear all cached embeddings"""
        # Clear Redis
        if self._use_redis and self._redis_cache:
            try:
                self._redis_cache.clear()
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        # Clear in-memory
        if self._in_memory_cache:
            self._in_memory_cache.clear()
        
        return True
    
    def get_stats(self) -> CacheStats:
        """Get combined cache statistics"""
        if self._use_redis and self._redis_cache:
            return self._redis_cache.get_stats()
        elif self._in_memory_cache:
            return self._in_memory_cache.get_stats()
        else:
            return CacheStats()
    
    def is_using_redis(self) -> bool:
        """Check if Redis is currently active"""
        return self._use_redis


# ============================================================================
# Kernel Query Engine (Updated with Vector DB and Cache Support)
# ============================================================================

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
    vector database storage, and embedding caching. Provides unified query API
    with performance optimization through vector databases and caching.
    
    Supports multiple backends:
    - Vector Store: Qdrant (production) or InMemory (development/testing)
    - Cache: Redis (production) or InMemory (development/testing)
    """
    
    def __init__(
        self,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        vector_store: Optional[VectorStoreAdapter] = None,
        embedding_cache: Optional[EmbeddingCacheManager] = None,
        use_vector_db: bool = False,
        use_cache: bool = True,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "kernel_vectors",
        vector_dimensions: int = 384,
    ):
        """
        Initialize the kernel query engine
        
        Args:
            embedding_generator: Custom embedding generator (auto-created if None)
            vector_store: Custom vector store adapter (auto-created if use_vector_db=True)
            embedding_cache: Custom cache manager (auto-created if use_cache=True)
            use_vector_db: Enable Qdrant vector database integration
            use_cache: Enable embedding caching
            redis_host: Redis server host for caching
            redis_port: Redis server port
            qdrant_host: Qdrant server host for vector storage
            qdrant_port: Qdrant server port
            collection_name: Qdrant collection name
            vector_dimensions: Embedding vector dimensions
        """
        # Initialize embedding generator
        self.embedding_generator = embedding_generator or EmbeddingGenerator(
            model_name="all-MiniLM-L6-v2",
            dimensions=vector_dimensions,
        )
        
        # Initialize vector store
        self._use_vector_db = use_vector_db
        if vector_store:
            self.vector_store = vector_store
        elif use_vector_db:
            try:
                self.vector_store = QdrantVectorStore(
                    host=qdrant_host,
                    port=qdrant_port,
                    collection_name=collection_name,
                    vector_dimensions=vector_dimensions,
                )
                self.vector_store.connect()
                logger.info(f"Connected to Qdrant: {qdrant_host}:{qdrant_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Qdrant: {e}")
                logger.warning("Falling back to in-memory vector store")
                self.vector_store = InMemoryVectorStore(
                    collection_name=collection_name,
                    vector_dimensions=vector_dimensions,
                )
                self._use_vector_db = False
        else:
            self.vector_store = InMemoryVectorStore(
                collection_name=collection_name,
                vector_dimensions=vector_dimensions,
            )
        
        # Initialize embedding cache
        self._use_cache = use_cache
        if embedding_cache:
            self.embedding_cache = embedding_cache
        elif use_cache:
            self.embedding_cache = EmbeddingCacheManager(
                redis_host=redis_host,
                redis_port=redis_port,
                key_prefix="kernel_emb:",
                default_ttl_seconds=86400,  # 24 hours
                enable_redis=True,
                in_memory_max_size=10000,
            )
            logger.info(f"Embedding cache initialized (Redis: {self.embedding_cache.is_using_redis()})")
        else:
            self.embedding_cache = None
        
        # Initialize exact matcher and hybrid engine
        self.exact_matcher = ExactMatcher()
        self.hybrid_engine = HybridSearchEngine()
        
        # Legacy in-memory knowledge base (for exact matching and when vector DB disabled)
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        
        logger.info("KernelQueryEngine initialized")
    
    def index_kernel(
        self,
        kernel_id: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        store_in_vector_db: bool = None,
    ) -> bool:
        """
        Index knowledge kernel
        
        Adds knowledge kernel to search index and optionally to vector database.
        
        Args:
            kernel_id: Unique identifier for the kernel
            content: Text content to index
            embedding: Pre-computed embedding (auto-generated if None)
            metadata: Optional metadata to store
            store_in_vector_db: Override for vector database storage
            
        Returns:
            True if indexing succeeded
        """
        use_vector_db = store_in_vector_db if store_in_vector_db is not None else self._use_vector_db
        
        # Generate embedding if not provided
        if embedding is None:
            embedding = self.embedding_generator.generate(content)
        
        # Store kernel data
        kernel_data = {
            "kernel_id": kernel_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "indexed_at": datetime.utcnow().isoformat(),
        }
        
        self.knowledge_base[kernel_id] = kernel_data
        self.exact_matcher.build_index([kernel_data])
        
        # Store in vector database if enabled
        if use_vector_db and self.vector_store:
            try:
                self.vector_store.upsert(
                    kernel_id=kernel_id,
                    vector=embedding,
                    content=content,
                    metadata=metadata,
                )
                logger.debug(f"Indexed kernel in vector DB: {kernel_id}")
            except Exception as e:
                logger.error(f"Failed to index kernel in vector DB: {e}")
                return False
        
        logger.info(f"Indexed kernel: {kernel_id}")
        return True
    
    def batch_index(
        self,
        kernels: List[Dict[str, Any]],
        store_in_vector_db: bool = None,
    ) -> int:
        """
        Batch index multiple knowledge kernels
        
        Args:
            kernels: List of kernel dictionaries with kernel_id, content, and optional embedding
            store_in_vector_db: Override for vector database storage
            
        Returns:
            Number of successfully indexed kernels
        """
        use_vector_db = store_in_vector_db if store_in_vector_db is not None else self._use_vector_db
        
        indexed_count = 0
        embeddings_to_store = []
        
        for kernel in kernels:
            kernel_id = kernel.get("kernel_id")
            content = kernel.get("content", "")
            metadata = {k: v for k, v in kernel.items() 
                       if k not in ["kernel_id", "content", "embedding"]}
            
            if not kernel_id:
                continue
            
            # Generate embedding if not provided
            embedding = kernel.get("embedding")
            if embedding is None:
                embedding = self.embedding_generator.generate(content)
            
            # Store in knowledge base
            kernel_data = {
                "kernel_id": kernel_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata,
                "indexed_at": datetime.utcnow().isoformat(),
            }
            self.knowledge_base[kernel_id] = kernel_data
            
            # Collect for batch vector DB storage
            if use_vector_db:
                embeddings_to_store.append({
                    "kernel_id": kernel_id,
                    "embedding": embedding,
                    "content": content,
                    **metadata,
                })
            
            indexed_count += 1
        
        # Update exact matcher
        self.exact_matcher.build_index(list(self.knowledge_base.values()))
        
        # Batch store in vector database
        if use_vector_db and embeddings_to_store and self.vector_store:
            try:
                batch_count = self.vector_store.batch_upsert(embeddings_to_store)
                logger.info(f"Batch indexed {batch_count} kernels in vector DB")
            except Exception as e:
                logger.error(f"Batch vector DB indexing failed: {e}")
        
        logger.info(f"Batch indexed {indexed_count} kernels")
        return indexed_count
    
    def query(
        self,
        query_text: str,
        kernel_ids: Optional[List[str]] = None,
        query_type: QueryType = QueryType.SEMANTIC,
        top_k: int = 10,
        min_similarity: float = 0.0,
        use_cache: bool = None,
        use_vector_db: bool = None,
    ) -> Tuple[List[QueryResult], QueryMetrics]:
        """
        Execute query
        
        Executes appropriate search strategy based on specified query type.
        Uses vector database for semantic search when available.
        
        Args:
            query_text: Query text
            kernel_ids: Optional, list of kernel IDs to query
            query_type: Query type (SEMANTIC, EXACT, FUZZY, HYBRID)
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            use_cache: Override for cache usage
            use_vector_db: Override for vector database usage
            
        Returns:
            (Query result list, Query metrics)
        """
        start_time = time.time()
        
        # Determine settings
        cache_enabled = use_cache if use_cache is not None else self._use_cache
        vector_db_enabled = use_vector_db if use_vector_db is not None else self._use_vector_db
        
        # Check embedding cache
        embedding_cache_hit = False
        if cache_enabled and self.embedding_cache:
            cached_embedding = self.embedding_cache.get(query_text)
            if cached_embedding is not None:
                query_embedding = cached_embedding
                embedding_cache_hit = True
            else:
                query_embedding = self.embedding_generator.generate(query_text)
                self.embedding_cache.set(query_text, query_embedding)
        else:
            query_embedding = self.embedding_generator.generate(query_text)
        
        # Execute search based on query type
        if query_type == QueryType.SEMANTIC:
            if vector_db_enabled and self.vector_store:
                results = self._vector_semantic_search(
                    query_embedding, kernel_ids, top_k, min_similarity
                )
            else:
                results = self._semantic_search(query_text, query_embedding, top_k)
        elif query_type == QueryType.EXACT:
            results = self._exact_search(query_text, top_k)
        elif query_type == QueryType.FUZZY:
            results = self._fuzzy_search(query_text, top_k)
        else:  # HYBRID
            results = self._hybrid_search(query_text, query_embedding, top_k)
        
        # Generate highlights
        for result in results:
            result.highlights = self._generate_highlights(query_text, result.content)
        
        # Calculate metrics
        query_time_ms = int((time.time() - start_time) * 1000)
        
        # Get cache stats
        cache_stats = None
        if self.embedding_cache:
            cache_stats = self.embedding_cache.get_stats()
        
        # Get vector store stats
        vector_stats = None
        if self.vector_store:
            vector_stats = self.vector_store.get_stats()
        
        metrics = QueryMetrics(
            query_time_ms=query_time_ms,
            total_results=len(results),
            query_type=query_type.value,
            cache_hit=embedding_cache_hit,
            embeddings_generated=not embedding_cache_hit,
            cache_stats=cache_stats,
            vector_stats=vector_stats,
        )
        
        return results, metrics
    
    def _vector_semantic_search(
        self,
        query_embedding: List[float],
        filter_ids: Optional[List[str]],
        top_k: int,
        min_similarity: float,
    ) -> List[QueryResult]:
        """Perform semantic search using vector database"""
        if not self.vector_store:
            return []
        
        try:
            # Search vector database
            search_results = self.vector_store.search(
                query_vector=query_embedding,
                top_k=top_k,
                score_threshold=min_similarity if min_similarity > 0 else None,
                filter_ids=filter_ids,
            )
            
            results = []
            for rank, search_result in enumerate(search_results):
                results.append(QueryResult(
                    kernel_id=search_result.kernel_id,
                    content=search_result.content or "",
                    similarity_score=search_result.score,
                    rank=rank + 1,
                    highlights=[],
                    metadata=search_result.metadata or {},
                ))
            
            return results
        except Exception as e:
            logger.warning(f"Vector DB search failed, falling back to in-memory: {e}")
            # Fall back to in-memory semantic search
            query_text = ""  # We don't have the query text here, but it won't be used
            return self._semantic_search(query_text, query_embedding, top_k)
    
    def _semantic_search(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int,
    ) -> List[QueryResult]:
        """In-memory semantic search"""
        if not self.knowledge_base:
            return []
        
        similarities = []
        
        for kernel_id, kernel_data in self.knowledge_base.items():
            embedding = kernel_data.get("embedding", [])
            if embedding:
                similarity = self.embedding_generator.cosine_similarity(query_embedding, embedding)
                similarities.append((kernel_id, similarity))
        
        # Sort
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Build results
        results = []
        for rank, (kernel_id, score) in enumerate(similarities[:top_k]):
            kernel_data = self.knowledge_base[kernel_id]
            results.append(QueryResult(
                kernel_id=kernel_id,
                content=kernel_data["content"],
                similarity_score=score,
                rank=rank + 1,
                highlights=[],
                metadata=kernel_data.get("metadata", {}),
            ))
        
        return results
    
    def _exact_search(self, query_text: str, top_k: int) -> List[QueryResult]:
        """Exact keyword search"""
        # Update index
        self.exact_matcher.build_index(list(self.knowledge_base.values()))
        
        # Execute exact search
        matched_ids = self.exact_matcher.exact_search(query_text)
        
        results = []
        for rank, kernel_id in enumerate(matched_ids[:top_k]):
            if kernel_id in self.knowledge_base:
                kernel_data = self.knowledge_base[kernel_id]
                results.append(QueryResult(
                    kernel_id=kernel_id,
                    content=kernel_data["content"],
                    similarity_score=1.0,
                    rank=rank + 1,
                    highlights=[],
                    metadata=kernel_data.get("metadata", {}),
                ))
        
        return results
    
    def _fuzzy_search(self, query_text: str, top_k: int) -> List[QueryResult]:
        """Fuzzy keyword search"""
        # Update index
        self.exact_matcher.build_index(list(self.knowledge_base.values()))
        
        # Execute fuzzy search
        fuzzy_results = self.exact_matcher.fuzzy_search(query_text)
        
        results = []
        for rank, (kernel_id, score) in enumerate(fuzzy_results[:top_k]):
            if kernel_id in self.knowledge_base:
                kernel_data = self.knowledge_base[kernel_id]
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
        top_k: int,
    ) -> List[QueryResult]:
        """Hybrid search combining semantic and exact matching"""
        # Execute semantic search
        semantic_results = []
        for kernel_id, kernel_data in self.knowledge_base.items():
            embedding = kernel_data.get("embedding", [])
            if embedding:
                similarity = self.embedding_generator.cosine_similarity(query_embedding, embedding)
                semantic_results.append((kernel_id, similarity))
        
        # Update index and execute exact search
        self.exact_matcher.build_index(list(self.knowledge_base.values()))
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
            if kernel_id in self.knowledge_base:
                kernel_data = self.knowledge_base[kernel_id]
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
            
            # Delete from vector database
            if self.vector_store:
                try:
                    self.vector_store.delete(kernel_id)
                except Exception as e:
                    logger.warning(f"Failed to delete from vector DB: {e}")
            
            self.exact_matcher.clear()
            self.exact_matcher.build_index(list(self.knowledge_base.values()))
            return True
        return False
    
    def clear_index(self) -> None:
        """Clear all indexed kernels"""
        self.knowledge_base.clear()
        self.exact_matcher.clear()
        
        # Clear vector database
        if self.vector_store:
            try:
                self.vector_store.clear()
            except Exception as e:
                logger.warning(f"Failed to clear vector DB: {e}")
        
        # Clear embedding cache
        if self.embedding_cache:
            try:
                self.embedding_cache.clear()
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        stats = {
            "indexed_kernels": len(self.knowledge_base),
            "embedding_stats": self.embedding_generator.get_stats(),
            "search_types": [e.value for e in QueryType],
        }
        
        # Cache stats
        if self.embedding_cache:
            cache_stats = self.embedding_cache.get_stats()
            stats["cache_stats"] = {
                "hits": cache_stats.hits,
                "misses": cache_stats.misses,
                "size": cache_stats.size,
                "hit_rate": f"{cache_stats.hit_rate:.2f}%",
                "using_redis": self.embedding_cache.is_using_redis(),
            }
        
        # Vector store stats
        if self.vector_store:
            vector_stats = self.vector_store.get_stats()
            stats["vector_stats"] = {
                "total_vectors": vector_stats.total_vectors,
                "collection_name": vector_stats.collection_name,
                "status": vector_stats.status,
                "index_type": vector_stats.index_type,
            }
        
        return stats


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
