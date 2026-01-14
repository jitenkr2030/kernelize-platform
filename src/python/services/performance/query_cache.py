"""
KERNELIZE Platform - Query Result Caching System
=================================================

Intelligent caching for frequently requested queries.
Implements query pattern detection, cache invalidation, cache warming,
and comprehensive hit rate monitoring.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Tuple
from threading import RLock
from collections import OrderedDict
import re
import numpy as np

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache strategies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In First Out
    TTL = "ttl"  # Time To Live based


@dataclass
class CachedQuery:
    """Cached query result"""
    query_hash: str = ""
    query_text: str = ""
    query_params: Dict[str, Any] = field(default_factory=dict)
    
    result: Any = None
    result_size_bytes: int = 0
    
    # Cache metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    access_count: int = 0
    
    # TTL
    ttl_seconds: int = 3600
    expires_at: str = field(default_factory=lambda: (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).isoformat())
    
    # Invalidation
    invalidation_version: int = 0
    depends_on_kernels: List[str] = field(default_factory=list)
    
    # Statistics
    generation_time_ms: float = 0.0
    
    def is_expired(self) -> bool:
        """Check if cached result is expired"""
        if datetime.now(timezone.utc) > datetime.fromisoformat(self.expires_at):
            return True
        return False
    
    def touch(self):
        """Update access time and count"""
        self.last_accessed = datetime.now(timezone.utc).isoformat()
        self.access_count += 1


@dataclass
class CacheConfiguration:
    """Cache configuration"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Cache"
    
    # Size limits
    max_entries: int = 10000
    max_memory_mb: int = 1024  # 1GB default
    
    # TTL settings
    default_ttl_seconds: int = 3600
    max_ttl_seconds: int = 86400  # 24 hours
    min_ttl_seconds: int = 60
    
    # Invalidation
    auto_invalidate_on_kernel_update: bool = True
    invalidation_batch_size: int = 100
    
    # Warming
    enable_cache_warming: bool = True
    warming_interval_seconds: int = 300
    popular_query_threshold: int = 10  # Queries accessed N times are "popular"
    
    # Monitoring
    enable_hit_rate_monitoring: bool = True
    monitoring_interval_seconds: int = 60
    
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True


@dataclass
class CacheStats:
    """Cache statistics"""
    # Hit/miss counts
    hits: int = 0
    misses: int = 0
    
    # Entry counts
    current_entries: int = 0
    current_memory_bytes: int = 0
    
    # Performance
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    
    # Operations
    evictions: int = 0
    invalidations: int = 0
    expirations: int = 0
    
    # Query patterns
    unique_queries: int = 0
    popular_queries: List[Dict[str, Any]] = field(default_factory=list)
    
    # Timestamps
    last_access: Optional[str] = None
    last_miss: Optional[str] = None
    last_hit: Optional[str] = None
    
    def hit_rate(self) -> float:
        """Calculate hit rate percentage"""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': f"{self.hit_rate():.2f}%",
            'current_entries': self.current_entries,
            'current_memory_mb': round(self.current_memory_bytes / (1024 * 1024), 2),
            'avg_hit_latency_ms': round(self.avg_hit_latency_ms, 2),
            'avg_miss_latency_ms': round(self.avg_miss_latency_ms, 2),
            'evictions': self.evictions,
            'invalidations': self.invalidations,
            'expirations': self.expirations,
            'unique_queries': self.unique_queries,
            'popular_queries': self.popular_queries
        }


class QueryPatternDetector:
    """Detects query patterns for intelligent caching"""
    
    def __init__(self):
        """Initialize pattern detector"""
        self._query_history: List[Dict[str, Any]] = []
        self._pattern_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()
        
        # Query pattern templates
        self._templates = {
            'keyword_search': r'^(?:\w+\s+){1,5}(?:search|find|query|look)',
            'semantic_query': r'(?:explain|describe|what|how|why)',
            'filter_query': r'(?:filter|where|with|having)',
            'aggregation': r'(?:count|sum|average|total|group)',
        }
    
    def analyze_query(self, query_text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze query to extract patterns
        
        Args:
            query_text: Query text
            params: Query parameters
            
        Returns:
            Analysis results
        """
        analysis = {
            'query_hash': self._hash_query(query_text, params),
            'query_text': query_text,
            'query_type': self._classify_query(query_text),
            'complexity': self._calculate_complexity(query_text),
            'keywords': self._extract_keywords(query_text),
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        with self._lock:
            self._query_history.append(analysis)
            
            # Update pattern cache
            query_type = analysis['query_type']
            if query_type not in self._pattern_cache:
                self._pattern_cache[query_type] = {
                    'count': 0,
                    'total_complexity': 0,
                    'examples': []
                }
            
            self._pattern_cache[query_type]['count'] += 1
            self._pattern_cache[query_type]['total_complexity'] += analysis['complexity']
            
            # Keep examples
            if len(self._pattern_cache[query_type]['examples']) < 5:
                self._pattern_cache[query_type]['examples'].append(query_text)
        
        return analysis
    
    def _hash_query(self, query_text: str, params: Dict[str, Any]) -> str:
        """Generate hash for query"""
        query_data = {
            'text': query_text.lower().strip(),
            'params': {k: v for k, v in params.items() if k not in ['sensitive']}
        }
        query_str = json.dumps(query_data, sort_keys=True)
        return hashlib.sha256(query_str.encode()).hexdigest()[:16]
    
    def _classify_query(self, query_text: str) -> str:
        """Classify query type based on text"""
        query_lower = query_text.lower()
        
        for pattern_type, pattern in self._templates.items():
            if re.search(pattern, query_lower):
                return pattern_type
        
        # Check for other common patterns
        if '?' in query_text:
            return 'question'
        elif len(query_text.split()) <= 3:
            return 'short_query'
        else:
            return 'general'
    
    def _calculate_complexity(self, query_text: str) -> int:
        """Calculate query complexity score"""
        score = 0
        
        # Length-based complexity
        score += len(query_text.split()) * 0.5
        
        # Contains special operators
        if ' AND ' in query_text or ' OR ' in query_text:
            score += 2
        if ' NOT ' in query_text:
            score += 1
        if any(op in query_text for op in ['>', '<', '=', '!=']):
            score += 2
        
        # Contains aggregations
        aggregations = ['count', 'sum', 'avg', 'max', 'min', 'group']
        if any(a in query_text.lower() for a in aggregations):
            score += 3
        
        return min(score, 10)  # Cap at 10
    
    def _extract_keywords(self, query_text: str) -> List[str]:
        """Extract key terms from query"""
        # Simple stopword removal
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'can', 'need', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
                     'by', 'from', 'as', 'into', 'through', 'during', 'before',
                     'after', 'above', 'below', 'between', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when', 'where',
                     'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
                     'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
                     'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                     'because', 'until', 'while', 'about', 'against', 'this', 'that'}
        
        words = re.findall(r'\b\w+\b', query_text.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        return keywords[:10]  # Limit to top 10 keywords
    
    def get_popular_queries(self, threshold: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed queries"""
        query_counts = {}
        
        with self._lock:
            for entry in self._query_history:
                query_hash = entry['query_hash']
                if query_hash not in query_counts:
                    query_counts[query_hash] = {
                        'query_hash': query_hash,
                        'query_text': entry['query_text'],
                        'query_type': entry['query_type'],
                        'count': 0,
                        'last_seen': entry['timestamp']
                    }
                query_counts[query_hash]['count'] += 1
                query_counts[query_hash]['last_seen'] = entry['timestamp']
        
        # Sort by count
        popular = sorted(
            query_counts.values(),
            key=lambda x: x['count'],
            reverse=True
        )
        
        return [q for q in popular if q['count'] >= threshold]
    
    def get_pattern_statistics(self) -> Dict[str, Any]:
        """Get pattern statistics"""
        with self._lock:
            total_queries = len(self._query_history)
            
            pattern_stats = {}
            for pattern, data in self._pattern_cache.items():
                pattern_stats[pattern] = {
                    'count': data['count'],
                    'percentage': (data['count'] / total_queries * 100) if total_queries > 0 else 0,
                    'avg_complexity': data['total_complexity'] / data['count'] if data['count'] > 0 else 0,
                    'examples': data['examples'][:3]
                }
            
            return {
                'total_queries': total_queries,
                'pattern_distribution': pattern_stats
            }


class QueryResultCache:
    """
    Main query result cache
    
    Implements intelligent caching with pattern detection,
    LRU/LFU strategies, and automatic invalidation.
    """
    
    def __init__(self, config: Optional[CacheConfiguration] = None):
        """
        Initialize query result cache
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfiguration()
        
        # Storage
        self._cache: OrderedDict[str, CachedQuery] = OrderedDict()
        self._index_by_kernel: Dict[str, Set[str]] = {}  # kernel_id -> query_hashes
        
        # Pattern detection
        self._pattern_detector = QueryPatternDetector()
        
        # Statistics
        self._stats = CacheStats()
        
        # Lock
        self._lock = RLock()
        
        # Background tasks
        self._running = False
        self._maintenance_thread = None
        
        # Start background maintenance
        self._start_maintenance()
    
    def _start_maintenance(self):
        """Start background maintenance tasks"""
        self._running = True
        self._maintenance_thread = None  # Would start thread in production
    
    def get(
        self,
        query_text: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Optional[Any], bool]:
        """
        Get cached query result
        
        Args:
            query_text: Query text
            params: Query parameters
            
        Returns:
            (result, cache_hit)
        """
        params = params or {}
        query_hash = self._pattern_detector._hash_query(query_text, params)
        
        start_time = time.perf_counter()
        hit = False
        
        with self._lock:
            if query_hash in self._cache:
                cached = self._cache[query_hash]
                
                if not cached.is_expired():
                    result = cached.result
                    cached.touch()
                    
                    # Move to end (most recently used)
                    self._cache.move_to_end(query_hash)
                    
                    # Update stats
                    self._stats.hits += 1
                    self._stats.last_hit = datetime.now(timezone.utc).isoformat()
                    self._stats.last_access = self._stats.last_hit
                    
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    self._update_avg_hit_latency(latency_ms)
                    
                    hit = True
                    return result, True
                else:
                    # Remove expired entry
                    self._remove_entry(query_hash)
                    self._stats.expirations += 1
            
            # Cache miss
            self._stats.misses += 1
            self._stats.last_miss = datetime.now(timezone.utc).isoformat()
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            self._update_avg_miss_latency(latency_ms)
            
            return None, False
    
    def set(
        self,
        query_text: str,
        result: Any,
        params: Optional[Dict[str, Any]] = None,
        ttl_seconds: Optional[int] = None,
        generation_time_ms: float = 0.0,
        depends_on_kernels: Optional[List[str]] = None
    ) -> str:
        """
        Cache query result
        
        Args:
            query_text: Query text
            result: Query result to cache
            params: Query parameters
            ttl_seconds: TTL in seconds
            generation_time_ms: Time to generate result
            depends_on_kernels: List of kernel IDs this depends on
            
        Returns:
            Query hash
        """
        params = params or {}
        ttl = ttl_seconds or self.config.default_ttl_seconds
        ttl = min(ttl, self.config.max_ttl_seconds)
        
        query_hash = self._pattern_detector._hash_query(query_text, params)
        
        # Analyze query
        analysis = self._pattern_detector.analyze_query(query_text, params)
        
        with self._lock:
            # Check if we need to evict
            if len(self._cache) >= self.config.max_entries:
                self._evict_entries(self.config.max_entries // 10)
            
            # Calculate result size
            result_size = self._estimate_size(result)
            
            # Create cached entry
            cached = CachedQuery(
                query_hash=query_hash,
                query_text=query_text,
                query_params=params,
                result=result,
                result_size_bytes=result_size,
                ttl_seconds=ttl,
                expires_at=(datetime.now(timezone.utc) + timedelta(seconds=ttl)).isoformat(),
                generation_time_ms=generation_time_ms,
                depends_on_kernels=depends_on_kernels or []
            )
            
            # Store in cache
            self._cache[query_hash] = cached
            
            # Update kernel index
            for kernel_id in cached.depends_on_kernels:
                if kernel_id not in self._index_by_kernel:
                    self._index_by_kernel[kernel_id] = set()
                self._index_by_kernel[kernel_id].add(query_hash)
            
            # Update stats
            self._stats.current_entries = len(self._cache)
            self._stats.current_memory_bytes += result_size
            self._stats.unique_queries = len(self._cache)
        
        return query_hash
    
    def invalidate_kernel(self, kernel_id: str) -> int:
        """
        Invalidate all cached results depending on a kernel
        
        Args:
            kernel_id: Kernel that was updated
            
        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        
        with self._lock:
            if kernel_id in self._index_by_kernel:
                query_hashes = list(self._index_by_kernel[kernel_id])
                
                for query_hash in query_hashes:
                    if query_hash in self._cache:
                        self._remove_entry(query_hash)
                        invalidated += 1
                
                del self._index_by_kernel[kernel_id]
                
                self._stats.invalidations += invalidated
        
        logger.info(f"Invalidated {invalidated} cache entries for kernel {kernel_id}")
        return invalidated
    
    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate cached results matching a pattern
        
        Args:
            pattern: Regex pattern to match
            
        Returns:
            Number of entries invalidated
        """
        invalidated = 0
        regex = re.compile(pattern)
        
        with self._lock:
            to_remove = [
                query_hash for query_hash, cached in self._cache.items()
                if regex.search(cached.query_text)
            ]
            
            for query_hash in to_remove:
                self._remove_entry(query_hash)
                invalidated += 1
        
        logger.info(f"Invalidated {invalidated} cache entries matching pattern: {pattern}")
        return invalidated
    
    def warm_cache(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Warm cache with popular or important queries
        
        Args:
            queries: List of query configurations with text, params, and generator
            
        Returns:
            Warming results
        """
        results = {
            'total': len(queries),
            'cached': 0,
            'failed': 0,
            'skipped': 0
        }
        
        for query_config in queries:
            try:
                query_text = query_config.get('text')
                params = query_config.get('params', {})
                generator = query_config.get('generator')
                ttl = query_config.get('ttl')
                
                # Check if already cached
                _, hit = self.get(query_text, params)
                
                if hit:
                    results['skipped'] += 1
                    continue
                
                # Generate result
                if generator:
                    start_time = time.perf_counter()
                    result = generator(query_text, params)
                    generation_time_ms = (time.perf_counter() - start_time) * 1000
                    
                    # Cache the result
                    self.set(
                        query_text=query_text,
                        result=result,
                        params=params,
                        ttl_seconds=ttl,
                        generation_time_ms=generation_time_ms
                    )
                    
                    results['cached'] += 1
                else:
                    results['skipped'] += 1
            
            except Exception as e:
                logger.error(f"Failed to warm cache for query: {e}")
                results['failed'] += 1
        
        logger.info(f"Cache warming complete: {results}")
        return results
    
    def _remove_entry(self, query_hash: str):
        """Remove entry from cache"""
        if query_hash in self._cache:
            cached = self._cache[query_hash]
            
            # Update memory stats
            self._stats.current_memory_bytes -= cached.result_size_bytes
            
            # Remove from kernel index
            for kernel_id in cached.depends_on_kernels:
                if kernel_id in self._index_by_kernel:
                    self._index_by_kernel[kernel_id].discard(query_hash)
            
            # Remove from cache
            del self._cache[query_hash]
            
            self._stats.current_entries = len(self._cache)
    
    def _evict_entries(self, count: int):
        """Evict least valuable entries"""
        if not self._cache:
            return
        
        # Sort by various criteria based on strategy
        if self.config.name.lower() == 'lru':
            # Remove oldest accessed
            to_remove = list(self._cache.items())[:count]
        elif self.config.name.lower() == 'lfu':
            # Remove least frequently accessed
            sorted_items = sorted(
                self._cache.items(),
                key=lambda x: x[1].access_count
            )
            to_remove = sorted_items[:count]
        else:  # FIFO or default
            # Remove oldest created
            to_remove = list(self._cache.items())[:count]
        
        for query_hash, _ in to_remove:
            self._remove_entry(query_hash)
            self._stats.evictions += 1
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate size of object in bytes"""
        try:
            import sys
            return len(json.dumps(obj).encode()) if obj else 0
        except:
            return 0
    
    def _update_avg_hit_latency(self, latency_ms: float):
        """Update average hit latency"""
        total = self._stats.hits
        if total == 1:
            self._stats.avg_hit_latency_ms = latency_ms
        else:
            self._stats.avg_hit_latency_ms = (
                (self._stats.avg_hit_latency_ms * (total - 1) + latency_ms) / total
            )
    
    def _update_avg_miss_latency(self, latency_ms: float):
        """Update average miss latency"""
        total = self._stats.misses
        if total == 1:
            self._stats.avg_miss_latency_ms = latency_ms
        else:
            self._stats.avg_miss_latency_ms = (
                (self._stats.avg_miss_latency_ms * (total - 1) + latency_ms) / total
            )
    
    def clear(self):
        """Clear all cached entries"""
        with self._lock:
            self._cache.clear()
            self._index_by_kernel.clear()
            self._stats = CacheStats()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            stats = self._stats.to_dict()
            
            # Add popular queries
            popular = self._pattern_detector.get_popular_queries(
                self.config.popular_query_threshold
            )
            stats['popular_queries'] = popular[:10]
            
            # Add pattern statistics
            stats['pattern_stats'] = self._pattern_detector.get_pattern_statistics()
        
        return stats
    
    def shutdown(self):
        """Shutdown cache and cleanup"""
        self._running = False
        self.clear()


# Singleton instance
_query_cache: Optional[QueryResultCache] = None


def get_query_cache() -> QueryResultCache:
    """Get query cache singleton"""
    global _query_cache
    
    if _query_cache is None:
        _query_cache = QueryResultCache()
    
    return _query_cache


def init_query_cache(
    max_entries: int = 10000,
    max_memory_mb: int = 1024,
    default_ttl_seconds: int = 3600
) -> QueryResultCache:
    """Initialize query cache system"""
    global _query_cache
    
    config = CacheConfiguration(
        max_entries=max_entries,
        max_memory_mb=max_memory_mb,
        default_ttl_seconds=default_ttl_seconds
    )
    
    _query_cache = QueryResultCache(config)
    
    return _query_cache
