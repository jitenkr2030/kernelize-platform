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
    """查询类型枚举"""
    SEMANTIC = "semantic"     # 语义搜索
    EXACT = "exact"           # 精确匹配
    FUZZY = "fuzzy"           # 模糊匹配
    HYBRID = "hybrid"         # 混合搜索


@dataclass
class QueryResult:
    """查询结果"""
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
    """查询指标"""
    query_time_ms: int
    total_results: int
    query_type: str
    cache_hit: bool
    embeddings_generated: bool


class CacheManager:
    """
    缓存管理器
    
    提供查询结果缓存和嵌入向量缓存，支持LRU淘汰策略。
    """
    
    def __init__(self, max_size: int = 10000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_seconds = ttl_hours * 3600
        self.cache = {}
        self.access_order = []
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
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
        """设置缓存值"""
        if key in self.cache:
            self.access_order.remove(key)
        
        self.cache[key] = (value, time.time())
        self.access_order.append(key)
        
        # LRU淘汰
        while len(self.cache) > self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.2f}%",
        }
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()
        self.hits = 0
        self.misses = 0


class EmbeddingGenerator:
    """
    嵌入向量生成器
    
    为文本生成语义嵌入向量，支持多种模型和优化策略。
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", dimensions: int = 384):
        self.model_name = model_name
        self.dimensions = dimensions
        self._model = None
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
    
    def generate(self, text: str) -> List[float]:
        """
        生成文本嵌入向量
        
        使用预训练模型将文本转换为语义向量。
        注意：实际实现中会加载transformers模型。
        """
        # 模拟嵌入生成（实际会使用sentence-transformers）
        # 这里生成一个基于文本内容的确定性向量
        
        # 预处理文本
        text = text.lower().strip()
        words = re.findall(r'\b[a-z]{2,}\b', text)
        
        # 使用词向量模拟
        vector = np.zeros(self.dimensions)
        for i, word in enumerate(words[:self.dimensions]):
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            vector[i] = (hash_val % 1000) / 1000.0
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def batch_generate(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入向量"""
        return [self.generate(text) for text in texts]
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
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
    
    def get_stats(self) -> Dict[str, Any]:
        """获取嵌入生成器统计"""
        return {
            "model_name": self.model_name,
            "dimensions": self.dimensions,
            "status": "ready",
        }


class ExactMatcher:
    """
    精确匹配器
    
    提供基于关键词的精确和模糊匹配功能。
    """
    
    def __init__(self):
        self.index = {}
        logger.info("ExactMatcher initialized")
    
    def build_index(self, documents: List[Dict[str, Any]]) -> None:
        """构建文档索引"""
        for doc in documents:
            kernel_id = doc.get("kernel_id")
            content = doc.get("compressed_content", "")
            
            # 提取词项
            words = self._tokenize(content)
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                if kernel_id not in self.index[word]:
                    self.index[word].append(kernel_id)
    
    def _tokenize(self, text: str) -> set:
        """文本分词"""
        text = text.lower()
        words = re.findall(r'\b[a-z]{2,}\b', text)
        return set(words)
    
    def exact_search(self, query: str) -> List[str]:
        """精确搜索"""
        query_words = self._tokenize(query)
        results = {}
        
        for word in query_words:
            if word in self.index:
                for kernel_id in self.index[word]:
                    results[kernel_id] = results.get(kernel_id, 0) + 1
        
        # 按匹配词数排序
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return [kernel_id for kernel_id, _ in sorted_results]
    
    def fuzzy_search(self, query: str, max_distance: int = 2) -> List[Tuple[str, int]]:
        """模糊搜索（基于编辑距离）"""
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
        """计算编辑距离"""
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
        """清空索引"""
        self.index.clear()


class HybridSearchEngine:
    """
    混合搜索引擎
    
    结合语义搜索和精确搜索的优势，提供最佳的搜索结果。
    使用加权融合策略平衡两种搜索方式的结果。
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
        融合搜索结果
        
        综合语义相似度和精确匹配分数，返回最终排序结果。
        """
        # 归一化分数
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
        
        # 获取所有唯一的kernel_id
        all_kernel_ids = set(normalized_semantic.keys()) | set(normalized_exact.keys())
        
        # 计算融合分数
        fused_scores = {}
        for kernel_id in all_kernel_ids:
            semantic_score = normalized_semantic.get(kernel_id, 0)
            exact_score = normalized_exact.get(kernel_id, 0)
            
            fused_score = (
                self.semantic_weight * semantic_score +
                self.exact_weight * exact_score
            )
            fused_scores[kernel_id] = fused_score
        
        # 排序并返回top_k
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_results[:top_k]
    
    def set_weights(self, semantic: float, exact: float) -> None:
        """设置融合权重"""
        total = semantic + exact
        self.semantic_weight = semantic / total
        self.exact_weight = exact / total


class KernelQueryEngine:
    """
    知识内核查询引擎
    
    顶层查询接口，整合语义搜索、精确匹配和混合搜索，
    提供统一的查询API和性能优化。
    """
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.exact_matcher = ExactMatcher()
        self.hybrid_engine = HybridSearchEngine()
        self.cache = CacheManager(max_size=10000, ttl_hours=24)
        
        # 模拟知识库存储
        self.knowledge_base: Dict[str, Dict[str, Any]] = {}
        
        logger.info("KernelQueryEngine initialized")
    
    def index_kernel(self, kernel_id: str, content: str, embedding: Optional[List[float]] = None) -> None:
        """
        索引知识内核
        
        将知识内核添加到搜索索引中。
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
        执行查询
        
        根据指定的查询类型执行相应的搜索策略。
        
        Args:
            query_text: 查询文本
            kernel_ids: 可选，限制查询的内核ID列表
            query_type: 查询类型
            top_k: 返回结果数量
            min_similarity: 最小相似度阈值
        
        Returns:
            (查询结果列表, 查询指标)
        """
        start_time = time.time()
        
        # 检查缓存
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
        
        # 过滤知识库
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
        
        # 生成查询嵌入
        query_embedding = self.embedding_generator.generate(query_text)
        
        # 根据查询类型执行搜索
        if query_type == QueryType.SEMANTIC:
            results = self._semantic_search(query_text, query_embedding, filtered_kernels, top_k)
        elif query_type == QueryType.EXACT:
            results = self._exact_search(query_text, filtered_kernels, top_k)
        elif query_type == QueryType.FUZZY:
            results = self._fuzzy_search(query_text, filtered_kernels, top_k)
        else:  # HYBRID
            results = self._hybrid_search(query_text, query_embedding, filtered_kernels, top_k)
        
        # 生成高亮
        for result in results:
            result.highlights = self._generate_highlights(query_text, result.content)
        
        # 计算指标
        query_time_ms = int((time.time() - start_time) * 1000)
        metrics = QueryMetrics(
            query_time_ms=query_time_ms,
            total_results=len(results),
            query_type=query_type.value,
            cache_hit=False,
            embeddings_generated=True,
        )
        
        # 缓存结果
        self.cache.set(cache_key, results)
        
        return results, metrics
    
    def _semantic_search(
        self,
        query_text: str,
        query_embedding: List[float],
        kernels: Dict[str, Dict[str, Any]],
        top_k: int,
    ) -> List[QueryResult]:
        """语义搜索"""
        similarities = []
        
        for kernel_id, kernel_data in kernels.items():
            embedding = kernel_data.get("embedding", [])
            if embedding:
                similarity = self.embedding_generator.cosine_similarity(query_embedding, embedding)
                similarities.append((kernel_id, similarity))
        
        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 构建结果
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
        """精确搜索"""
        # 更新索引
        self.exact_matcher.build_index(list(kernels.values()))
        
        # 执行精确搜索
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
        """模糊搜索"""
        # 更新索引
        self.exact_matcher.build_index(list(kernels.values()))
        
        # 执行模糊搜索
        fuzzy_results = self.exact_matcher.fuzzy_search(query_text)
        
        results = []
        for rank, (kernel_id, score) in enumerate(fuzzy_results[:top_k]):
            if kernel_id in kernels:
                kernel_data = kernels[kernel_id]
                results.append(QueryResult(
                    kernel_id=kernel_id,
                    content=kernel_data["content"],
                    similarity_score=score / 100.0,  # 归一化
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
        """混合搜索"""
        # 执行两种搜索
        semantic_results = []
        for kernel_id, kernel_data in kernels.items():
            embedding = kernel_data.get("embedding", [])
            if embedding:
                similarity = self.embedding_generator.cosine_similarity(query_embedding, embedding)
                semantic_results.append((kernel_id, similarity))
        
        # 更新索引并执行精确搜索
        self.exact_matcher.build_index(list(kernels.values()))
        exact_results = self.exact_matcher.exact_search(query_text)
        exact_scores = [(kid, 1) for kid in exact_results]
        
        # 融合结果
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
        """生成查询高亮"""
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
        """删除知识内核"""
        if kernel_id in self.knowledge_base:
            del self.knowledge_base[kernel_id]
            self.exact_matcher.clear()
            self.exact_matcher.build_index(list(self.knowledge_base.values()))
            return True
        return False
    
    def clear_index(self) -> None:
        """清空索引"""
        self.knowledge_base.clear()
        self.exact_matcher.clear()
        self.cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取引擎统计"""
        return {
            "indexed_kernels": len(self.knowledge_base),
            "cache_stats": self.cache.get_stats(),
            "embedding_stats": self.embedding_generator.get_stats(),
            "search_types": [e.value for e in QueryType],
        }


# 创建全局查询引擎实例
query_engine = KernelQueryEngine()


# 便捷函数
def semantic_search(
    query: str,
    top_k: int = 10,
    min_similarity: float = 0.0,
) -> Tuple[List[QueryResult], QueryMetrics]:
    """快捷语义搜索函数"""
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
    """快捷混合搜索函数"""
    return query_engine.query(
        query_text=query,
        query_type=QueryType.HYBRID,
        top_k=top_k,
    )
