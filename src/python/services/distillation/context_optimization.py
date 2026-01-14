"""
Context Window Optimization Module

This module provides sophisticated context management capabilities for the Kernel ecosystem,
enabling efficient handling of long-context scenarios during distillation, fine-tuning,
and inference. The implementation includes hierarchical context construction, token budget
management, and multiple optimization strategies.

Key Components:
- ContextChunk: Atomic unit of contextual information with metadata
- ContextWindow: Manages collections of chunks within token limits
- Optimization Strategies: Various approaches for context compression and selection
- HierarchicalSummarizer: Creates multi-level summaries for deep contexts
- ContextOptimizationPipeline: Orchestrates end-to-end optimization workflows
"""

import hashlib
import json
import logging
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """
    Enumeration of available context optimization strategies.
    
    Each strategy represents a different approach to managing limited context windows,
    trading off between information density, computational cost, and preservation
    of critical context details.
    """
    RELEVANCE_BASED = "relevance_based"
    SLIDING_WINDOW = "sliding_window"
    HIERARCHICAL_SUMMARIZATION = "hierarchical_summarization"
    MIXED_DENSITY = "mixed_density"
    SEMANTIC_COMPRESSION = "semantic_compression"


class ChunkType(Enum):
    """
    Classification of context chunk types for differentiated processing.
    
    Different types of content have different processing requirements and
    information density characteristics, enabling strategy-aware optimization.
    """
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    TEMPORAL = "temporal"
    RELATIONAL = "relational"
    NARRATIVE = "narrative"
    CONCEPTUAL = "conceptual"
    METADATA = "metadata"


@dataclass
class ContextChunk:
    """
    Atomic unit of contextual information within the optimization system.
    
    A ContextChunk represents a discrete piece of information that can be
    independently evaluated, scored, and potentially compressed or discarded.
    The chunk maintains rich metadata to support sophisticated optimization
    strategies and preserve provenance information.
    
    Attributes:
        content: The actual text or data content of the chunk
        chunk_type: Classification of the chunk (factual, procedural, etc.)
        chunk_id: Unique identifier for tracking and referencing
        importance_score: Pre-computed importance score (0.0-1.0)
        token_count: Number of tokens in the content
        position: Positional information for ordered contexts
        metadata: Additional metadata for specialized processing
        source_ref: Reference to original source (for provenance)
        created_at: Timestamp of chunk creation
        tags: Categorization tags for filtering and selection
        dependencies: IDs of chunks this chunk depends on
        semantic_vector: Optional embedding vector for similarity calculations
    """
    content: str
    chunk_type: ChunkType
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    importance_score: float = 0.5
    token_count: int = 0
    position: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_ref: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    semantic_vector: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Post-initialization processing for derived fields."""
        if self.token_count == 0:
            self.token_count = self._estimate_tokens(self.content)
    
    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Estimate token count for a text string using word-based approximation.
        
        This provides a rough estimate suitable for planning and budgeting.
        Production systems should use actual tokenizer counts for accuracy.
        
        Args:
            text: Input text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        # Basic word-based estimate: ~0.75 tokens per word + overhead
        words = len(text.split())
        # Account for punctuation and special characters
        special_chars = len(re.findall(r'[^\w\s]', text))
        return int(words * 0.75 + special_chars * 0.25)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary for storage or transmission."""
        result = {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "chunk_type": self.chunk_type.value,
            "importance_score": self.importance_score,
            "token_count": self.token_count,
            "position": self.position,
            "metadata": self.metadata,
            "source_ref": self.source_ref,
            "created_at": self.created_at.isoformat(),
            "tags": self.tags,
            "dependencies": self.dependencies
        }
        # Handle numpy array serialization
        if self.semantic_vector is not None:
            result["semantic_vector"] = self.semantic_vector.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextChunk':
        """Deserialize chunk from dictionary."""
        data["chunk_type"] = ChunkType(data["chunk_type"])
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "semantic_vector" in data and data["semantic_vector"] is not None:
            data["semantic_vector"] = np.array(data["semantic_vector"])
        return cls(**data)


@dataclass
class ContextWindow:
    """
    Manages a collection of context chunks within token and computational constraints.
    
    The ContextWindow serves as the primary interface for managing context
    during optimization operations. It maintains the invariant that the total
    token count never exceeds the configured maximum, applying configured
    strategies when space is needed.
    
    Attributes:
        max_tokens: Maximum token capacity of the window
        chunks: Ordered collection of context chunks
        optimization_strategy: Strategy to apply when space is needed
        compression_target: Target compression ratio when optimizing
    """
    max_tokens: int
    chunks: List[ContextChunk] = field(default_factory=list)
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.RELEVANCE_BASED
    compression_target: float = 0.8
    
    @property
    def current_token_count(self) -> int:
        """Calculate current total token usage."""
        return sum(chunk.token_count for chunk in self.chunks)
    
    @property
    def utilization_ratio(self) -> float:
        """Return the current token utilization as a ratio (0.0-1.0)."""
        if self.max_tokens == 0:
            return 0.0
        return self.current_token_count / self.max_tokens
    
    @property
    def available_tokens(self) -> int:
        """Return the number of tokens still available."""
        return max(0, self.max_tokens - self.current_token_count)
    
    def add_chunk(self, chunk: ContextChunk, allow_overflow: bool = False) -> bool:
        """
        Add a chunk to the context window.
        
        Args:
            chunk: The context chunk to add
            allow_overflow: If True, allow token count to exceed maximum
            
        Returns:
            True if chunk was added successfully, False otherwise
        """
        if not allow_overflow and self.current_token_count + chunk.token_count > self.max_tokens:
            logger.debug(
                f"Cannot add chunk {chunk.chunk_id}: would exceed token limit "
                f"({self.current_token_count + chunk.token_count} > {self.max_tokens})"
            )
            return False
        
        self.chunks.append(chunk)
        logger.debug(
            f"Added chunk {chunk.chunk_id} to window: "
            f"tokens={self.current_token_count}/{self.max_tokens}"
        )
        return True
    
    def add_chunks(self, chunks: List[ContextChunk]) -> int:
        """
        Add multiple chunks to the window, respecting token limits.
        
        Args:
            chunks: List of chunks to add, processed in order
            
        Returns:
            Number of chunks successfully added
        """
        added_count = 0
        for chunk in chunks:
            if self.add_chunk(chunk):
                added_count += 1
        return added_count
    
    def remove_chunk(self, chunk_id: str) -> Optional[ContextChunk]:
        """
        Remove a specific chunk from the window by ID.
        
        Args:
            chunk_id: ID of the chunk to remove
            
        Returns:
            The removed chunk if found, None otherwise
        """
        for i, chunk in enumerate(self.chunks):
            if chunk.chunk_id == chunk_id:
                removed = self.chunks.pop(i)
                logger.debug(f"Removed chunk {chunk_id} from window")
                return removed
        return None
    
    def clear(self) -> List[ContextChunk]:
        """
        Remove all chunks from the window.
        
        Returns:
            List of all removed chunks
        """
        removed = self.chunks.copy()
        self.chunks.clear()
        logger.debug(f"Cleared window: removed {len(removed)} chunks")
        return removed
    
    def get_chunks_by_type(self, chunk_type: ChunkType) -> List[ContextChunk]:
        """Retrieve all chunks of a specific type."""
        return [c for c in self.chunks if c.chunk_type == chunk_type]
    
    def get_top_chunks(self, n: int, by_importance: bool = True) -> List[ContextChunk]:
        """
        Retrieve the top N chunks by importance or position.
        
        Args:
            n: Number of chunks to retrieve
            by_importance: If True, sort by importance score; otherwise by position
            
        Returns:
            List of top N chunks
        """
        if by_importance:
            return sorted(self.chunks, key=lambda c: c.importance_score, reverse=True)[:n]
        else:
            return [c for c in sorted(self.chunks, key=lambda c: c.position or 0) if c.position is not None][:n]
    
    def to_manifest(self) -> Dict[str, Any]:
        """Generate a manifest representation of the window state."""
        return {
            "max_tokens": self.max_tokens,
            "current_token_count": self.current_token_count,
            "chunk_count": len(self.chunks),
            "optimization_strategy": self.optimization_strategy.value,
            "compression_target": self.compression_target,
            "chunks": [chunk.to_dict() for chunk in self.chunks]
        }


class ContextOptimizer(ABC):
    """
    Abstract base class for context optimization strategies.
    
    Implementations of this class define specific approaches to reducing
    or restructuring context while preserving critical information. Each
    optimizer operates according to a specific strategy and may have
    different trade-offs between compression ratio, information loss,
    and computational cost.
    """
    
    def __init__(self, name: str, target_compression: float = 0.8):
        """
        Initialize the optimizer.
        
        Args:
            name: Human-readable name for logging and identification
            target_compression: Desired compression ratio (0.0-1.0)
        """
        self.name = name
        self.target_compression = target_compression
        self.stats = {
            "original_tokens": 0,
            "optimized_tokens": 0,
            "chunks_processed": 0,
            "chunks_removed": 0,
            "compressions_applied": 0
        }
    
    @abstractmethod
    def optimize(
        self,
        window: ContextWindow,
        query: Optional[str] = None
    ) -> Tuple[ContextWindow, Dict[str, Any]]:
        """
        Apply optimization to a context window.
        
        Args:
            window: The context window to optimize
            query: Optional query for relevance-based optimization
            
        Returns:
            Tuple of (optimized_window, optimization_metadata)
        """
        pass
    
    def _update_stats(
        self,
        original_count: int,
        optimized_count: int,
        chunks_processed: int,
        chunks_removed: int
    ):
        """Update optimization statistics."""
        self.stats["original_tokens"] = original_count
        self.stats["optimized_tokens"] = optimized_count
        self.stats["chunks_processed"] += chunks_processed
        self.stats["chunks_removed"] += chunks_removed
        self.stats["compressions_applied"] += 1
    
    def get_compression_ratio(self) -> float:
        """Calculate the achieved compression ratio."""
        if self.stats["original_tokens"] == 0:
            return 1.0
        return self.stats["optimized_tokens"] / self.stats["original_tokens"]


class RelevanceBasedOptimizer(ContextOptimizer):
    """
    Optimizer that selects context based on relevance scoring.
    
    This optimizer evaluates each chunk's relevance to the task or query
    and retains only the most relevant content within token limits. It
    supports both explicit query-based relevance and implicit importance-based
    selection.
    """
    
    def __init__(
        self,
        target_compression: float = 0.8,
        min_importance_threshold: float = 0.3,
        preserve_types: List[ChunkType] = None
    ):
        """
        Initialize the relevance-based optimizer.
        
        Args:
            target_compression: Desired compression ratio
            min_importance_threshold: Minimum importance score to retain
            preserve_types: Chunk types that should never be discarded
        """
        super().__init__("RelevanceBasedOptimizer", target_compression)
        self.min_importance_threshold = min_importance_threshold
        self.preserve_types = preserve_types or [ChunkType.FACTUAL]
    
    def optimize(
        self,
        window: ContextWindow,
        query: Optional[str] = None
    ) -> Tuple[ContextWindow, Dict[str, Any]]:
        """
        Optimize by retaining only the most relevant chunks.
        
        The optimization process:
        1. Scores each chunk for relevance (query-based or importance-based)
        2. Preserves mandatory chunk types
        3. Removes lowest-scoring chunks until token target is met
        
        Args:
            window: Context window to optimize
            query: Optional query for relevance calculation
            
        Returns:
            Tuple of (optimized_window, optimization_details)
        """
        original_tokens = window.current_token_count
        target_tokens = int(original_tokens * self.target_compression)
        
        # Calculate target number of chunks
        avg_chunk_tokens = original_tokens / max(1, len(window.chunks))
        target_chunk_count = max(1, int(target_tokens / avg_chunk_tokens))
        
        # Score and sort chunks
        scored_chunks = self._score_chunks(window.chunks, query)
        
        # Separate preserved and optional chunks
        preserved = [s for s in scored_chunks if s[0].chunk_type in self.preserve_types]
        optional = [s for s in scored_chunks if s[0].chunk_type not in self.preserve_types]
        
        # Sort each group by score
        preserved.sort(key=lambda x: x[1], reverse=True)
        optional.sort(key=lambda x: x[1], reverse=True)
        
        # Build optimized chunk list
        optimized_chunks = [chunk for chunk, score in preserved]
        
        # Add optional chunks until token target is met
        current_tokens = sum(c.token_count for c in optimized_chunks)
        for chunk, score in optional:
            if current_tokens + chunk.token_count <= target_tokens:
                optimized_chunks.append(chunk)
                current_tokens += chunk.token_count
            elif chunk.importance_score >= self.min_importance_threshold:
                # Include high-importance chunks even if over target
                optimized_chunks.append(chunk)
        
        # Create new window with optimized chunks
        optimized_window = ContextWindow(
            max_tokens=window.max_tokens,
            optimization_strategy=window.optimization_strategy,
            compression_target=window.compression_target
        )
        optimized_window.add_chunks(optimized_chunks)
        
        # Record removal of critical dependencies
        removed_ids = set(c.chunk_id for c in window.chunks) - set(c.chunk_id for c in optimized_chunks)
        removed_dependents = self._find_removed_dependents(window.chunks, removed_ids)
        
        # Update statistics
        self._update_stats(
            original_tokens,
            optimized_window.current_token_count,
            len(window.chunks),
            len(removed_ids)
        )
        
        optimization_details = {
            "strategy": self.name,
            "original_chunks": len(window.chunks),
            "optimized_chunks": len(optimized_chunks),
            "chunks_removed": len(removed_ids),
            "removed_ids": list(removed_ids),
            "removed_dependents": list(removed_dependents),
            "compression_ratio": optimized_window.current_token_count / max(1, original_tokens),
            "query_relevance": query is not None,
            "preserved_types": [t.value for t in self.preserve_types]
        }
        
        return optimized_window, optimization_details
    
    def _score_chunks(
        self,
        chunks: List[ContextChunk],
        query: Optional[str]
    ) -> List[Tuple[ContextChunk, float]]:
        """
        Score chunks for relevance.
        
        Args:
            chunks: List of chunks to score
            query: Optional query for relevance calculation
            
        Returns:
            List of (chunk, score) tuples sorted by score
        """
        if query:
            # Query-based relevance scoring
            query_terms = set(query.lower().split())
            scored = []
            for chunk in chunks:
                # Calculate term overlap
                content_terms = set(chunk.content.lower().split())
                overlap = len(query_terms & content_terms)
                # Combine with importance score
                relevance_score = (overlap / max(1, len(query_terms))) * 0.6 + chunk.importance_score * 0.4
                scored.append((chunk, relevance_score))
            return scored
        else:
            # Importance-only scoring
            return [(c, c.importance_score) for c in chunks]
    
    def _find_removed_dependents(
        self,
        all_chunks: List[ContextChunk],
        removed_ids: set
    ) -> List[str]:
        """Identify chunks that depend on removed chunks."""
        dependents = set()
        id_to_chunk = {c.chunk_id: c for c in all_chunks}
        
        def find_dependents(chunk_id: str):
            if chunk_id in removed_ids:
                return
            chunk = id_to_chunk.get(chunk_id)
            if not chunk:
                return
            for dep_id in chunk.dependencies:
                if dep_id in removed_ids:
                    dependents.add(chunk_id)
                else:
                    find_dependents(dep_id)
        
        for removed_id in removed_ids:
            find_dependents(removed_id)
        
        return list(dependents)


class SlidingWindowOptimizer(ContextOptimizer):
    """
    Optimizer that implements sliding window context management.
    
    This optimizer maintains a sliding window over ordered context,
    retaining the most recent chunks while potentially summarizing
    or discarding older content. It is particularly effective for
    sequential or temporal data where recency is important.
    """
    
    def __init__(
        self,
        target_compression: float = 0.8,
        window_size: Optional[int] = None,
        stride: int = 1,
        summarize_older: bool = True
    ):
        """
        Initialize the sliding window optimizer.
        
        Args:
            target_compression: Desired compression ratio
            window_size: Maximum chunks in window (None for auto-calculation)
            stride: Step size for window movement
            summarize_older: Whether to create summaries for discarded content
        """
        super().__init__("SlidingWindowOptimizer", target_compression)
        self.window_size = window_size
        self.stride = stride
        self.summarize_older = summarize_older
        self._summarizer = None
    
    def optimize(
        self,
        window: ContextWindow,
        query: Optional[str] = None
    ) -> Tuple[ContextWindow, Dict[str, Any]]:
        """
        Apply sliding window optimization.
        
        Args:
            window: Context window to optimize
            query: Not used in sliding window mode
            
        Returns:
            Tuple of (optimized_window, optimization_details)
        """
        original_tokens = window.current_token_count
        target_tokens = int(original_tokens * self.target_compression)
        
        # Sort chunks by position
        sorted_chunks = sorted(window.chunks, key=lambda c: c.position or 0)
        
        # Determine window size
        if self.window_size is None:
            # Auto-calculate based on token target
            cumulative_tokens = 0
            self.window_size = 0
            for i, chunk in enumerate(sorted_chunks):
                cumulative_tokens += chunk.token_count
                if cumulative_tokens > target_tokens:
                    break
                self.window_size = i + 1
        
        # Select chunks within window
        selected_chunks = sorted_chunks[:self.window_size]
        discarded_chunks = sorted_chunks[self.window_size:]
        
        # Create summary of discarded content if enabled
        summary_chunks = []
        if self.summarize_older and discarded_chunks:
            summary_chunks = self._create_summary(discarded_chunks)
        
        # Build optimized window
        optimized_window = ContextWindow(
            max_tokens=window.max_tokens,
            optimization_strategy=window.optimization_strategy,
            compression_target=window.compression_target
        )
        optimized_window.add_chunks(selected_chunks)
        if summary_chunks:
            optimized_window.add_chunks(summary_chunks)
        
        # Update statistics
        removed_count = len(discarded_chunks) - len(summary_chunks)
        self._update_stats(
            original_tokens,
            optimized_window.current_token_count,
            len(window.chunks),
            removed_count
        )
        
        optimization_details = {
            "strategy": self.name,
            "window_size": self.window_size,
            "stride": self.stride,
            "original_chunks": len(window.chunks),
            "selected_chunks": len(selected_chunks),
            "discarded_chunks": len(discarded_chunks),
            "summary_chunks": len(summary_chunks),
            "compression_ratio": optimized_window.current_token_count / max(1, original_tokens)
        }
        
        return optimized_window, optimization_details
    
    def _create_summary(self, chunks: List[ContextChunk]) -> List[ContextChunk]:
        """Create a summary chunk from discarded content."""
        if not chunks:
            return []
        
        # Simple extractive summarization: concatenate first sentences
        summary_text_parts = []
        for chunk in chunks[:5]:  # Limit to first 5 chunks
            first_sentence = chunk.content.split('.')[0] + '.'
            summary_text_parts.append(f"[From {chunk.chunk_type.value}]: {first_sentence}")
        
        summary_text = " ".join(summary_text_parts)
        
        summary_chunk = ContextChunk(
            content=summary_text,
            chunk_type=ChunkType.CONCEPTUAL,
            importance_score=0.3,
            metadata={"is_summary": True, "source_chunks": [c.chunk_id for c in chunks]},
            tags=["summary", "compressed"]
        )
        
        return [summary_chunk]


class HierarchicalSummarizer(ContextOptimizer):
    """
    Optimizer that creates hierarchical summaries for multi-level context.
    
    This optimizer processes context in multiple passes, creating progressively
    more compressed representations at each level. The result is a hierarchical
    structure where fine-grained details are preserved at lower levels while
    high-level abstractions are available at higher levels.
    """
    
    def __init__(
        self,
        target_compression: float = 0.8,
        levels: int = 3,
        aggregation_method: str = "concatenation"
    ):
        """
        Initialize the hierarchical summarizer.
        
        Args:
            target_compression: Desired compression ratio
            levels: Number of hierarchical levels to create
            aggregation_method: How to combine chunks at each level
        """
        super().__init__("HierarchicalSummarizer", target_compression)
        self.levels = levels
        self.aggregation_method = aggregation_method
    
    def optimize(
        self,
        window: ContextWindow,
        query: Optional[str] = None
    ) -> Tuple[ContextWindow, Dict[str, Any]]:
        """
        Apply hierarchical summarization optimization.
        
        Args:
            window: Context window to optimize
            query: Optional query to guide summarization focus
            
        Returns:
            Tuple of (optimized_window, optimization_details)
        """
        original_tokens = window.current_token_count
        
        # Build hierarchical structure
        hierarchy = self._build_hierarchy(window.chunks)
        
        # Create optimized window with hierarchical representation
        optimized_window = ContextWindow(
            max_tokens=window.max_tokens,
            optimization_strategy=window.optimization_strategy,
            compression_target=window.compression_target
        )
        
        # Add chunks from each level, starting from the deepest (most compressed)
        levels_added = []
        for level_idx in range(self.levels - 1, -1, -1):
            level_chunks = hierarchy.get(level_idx, [])
            if level_idx == 0:
                # Level 0: Original fine-grained chunks
                # Limit to token budget
                current_tokens = 0
                for chunk in level_chunks:
                    if current_tokens + chunk.token_count <= window.max_tokens * 0.5:
                        optimized_window.add_chunk(chunk)
                        current_tokens += chunk.token_count
            else:
                # Higher levels: Summary chunks
                for chunk in level_chunks[:10]:  # Limit summaries
                    if optimized_window.add_chunk(chunk, allow_overflow=True):
                        pass  # Allow some overflow for summaries
            levels_added.append(len(level_chunks))
        
        # Update statistics
        self._update_stats(
            original_tokens,
            optimized_window.current_token_count,
            len(window.chunks),
            len(window.chunks) - len(optimized_window.chunks)
        )
        
        optimization_details = {
            "strategy": self.name,
            "hierarchy_levels": self.levels,
            "original_chunks": len(window.chunks),
            "optimized_chunks": len(optimized_window.chunks),
            "levels_structure": dict(zip(range(self.levels), levels_added)),
            "compression_ratio": optimized_window.current_token_count / max(1, original_tokens)
        }
        
        return optimized_window, optimization_details
    
    def _build_hierarchy(
        self,
        chunks: List[ContextChunk]
    ) -> Dict[int, List[ContextChunk]]:
        """
        Build hierarchical representation of chunks.
        
        Args:
            chunks: Input chunks to organize hierarchically
            
        Returns:
            Dictionary mapping level number to list of chunks at that level
        """
        hierarchy = {i: [] for i in range(self.levels)}
        
        # Level 0: Original chunks
        hierarchy[0] = chunks.copy()
        
        # Create summaries for higher levels
        for level in range(1, self.levels):
            lower_level_chunks = hierarchy.get(level - 1, [])
            if not lower_level_chunks:
                continue
            
            # Group lower-level chunks by type for summarization
            type_groups = defaultdict(list)
            for chunk in lower_level_chunks:
                type_groups[chunk.chunk_type].append(chunk)
            
            # Create summaries for each group
            for chunk_type, type_chunks in type_groups.items():
                if len(type_chunks) <= 3:
                    continue  # Don't summarize small groups
                
                # Create summary chunk
                summary_text = self._summarize_type_group(type_chunks)
                summary_chunk = ContextChunk(
                    content=summary_text,
                    chunk_type=chunk_type,
                    importance_score=0.5 / level,  # Lower importance at higher levels
                    metadata={
                        "is_hierarchy_summary": True,
                        "source_level": level - 1,
                        "source_count": len(type_chunks)
                    },
                    tags=["hierarchy", f"level_{level}", "summary"]
                )
                hierarchy[level].append(summary_chunk)
        
        return hierarchy
    
    _summarize_type_group = lambda self, chunks: (
        f"Summary of {len(chunks)} {chunks[0].chunk_type.value} items: " +
        "; ".join(c.content[:100] for c in chunks[:3])
    )


class SemanticCompressionOptimizer(ContextOptimizer):
    """
    Optimizer that compresses context based on semantic deduplication.
    
    This optimizer identifies and removes semantically redundant content,
    keeping only unique information within the token budget. It uses
    embedding similarity to detect redundancy.
    """
    
    def __init__(
        self,
        target_compression: float = 0.8,
        similarity_threshold: float = 0.85,
        keep_strategy: str = "importance"
    ):
        """
        Initialize the semantic compression optimizer.
        
        Args:
            target_compression: Desired compression ratio
            similarity_threshold: Threshold for considering content redundant
            keep_strategy: How to choose between similar items ('importance' or 'recency')
        """
        super().__init__("SemanticCompressionOptimizer", target_compression)
        self.similarity_threshold = similarity_threshold
        self.keep_strategy = keep_strategy
    
    def optimize(
        self,
        window: ContextWindow,
        query: Optional[str] = None
    ) -> Tuple[ContextWindow, Dict[str, Any]]:
        """
        Apply semantic compression optimization.
        
        Args:
            window: Context window to optimize
            query: Optional query to prioritize relevant redundancy
            
        Returns:
            Tuple of (optimized_window, optimization_details)
        """
        original_tokens = window.current_token_count
        target_tokens = int(original_tokens * self.target_compression)
        
        # Sort chunks by importance or recency
        if self.keep_strategy == "importance":
            sorted_chunks = sorted(
                window.chunks,
                key=lambda c: c.importance_score,
                reverse=True
            )
        else:
            sorted_chunks = sorted(
                window.chunks,
                key=lambda c: c.position or 0,
                reverse=True
            )
        
        # Select chunks while avoiding semantic duplicates
        selected_chunks = []
        selected_vectors = []
        current_tokens = 0
        
        for chunk in sorted_chunks:
            if current_tokens + chunk.token_count > target_tokens:
                break
            
            # Check for semantic duplicates
            is_duplicate = False
            if chunk.semantic_vector is not None:
                for vec in selected_vectors:
                    if self._cosine_similarity(chunk.semantic_vector, vec) >= self.similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                selected_chunks.append(chunk)
                current_tokens += chunk.token_count
                if chunk.semantic_vector is not None:
                    selected_vectors.append(chunk.semantic_vector)
        
        # Build optimized window
        optimized_window = ContextWindow(
            max_tokens=window.max_tokens,
            optimization_strategy=window.optimization_strategy,
            compression_target=window.compression_target
        )
        optimized_window.add_chunks(selected_chunks)
        
        # Update statistics
        self._update_stats(
            original_tokens,
            optimized_window.current_token_count,
            len(window.chunks),
            len(window.chunks) - len(selected_chunks)
        )
        
        optimization_details = {
            "strategy": self.name,
            "similarity_threshold": self.similarity_threshold,
            "original_chunks": len(window.chunks),
            "selected_chunks": len(selected_chunks),
            "duplicates_removed": len(window.chunks) - len(selected_chunks),
            "compression_ratio": optimized_window.current_token_count / max(1, original_tokens)
        }
        
        return optimized_window, optimization_details
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)


class MixedDensityOptimizer(ContextOptimizer):
    """
    Optimizer that applies different strategies to different chunk types.
    
    This optimizer recognizes that different types of content benefit from
    different optimization approaches and applies type-specific strategies
    for optimal compression while preserving information integrity.
    """
    
    def __init__(
        self,
        target_compression: float = 0.8,
        strategy_mapping: Dict[ChunkType, OptimizationStrategy] = None
    ):
        """
        Initialize the mixed-density optimizer.
        
        Args:
            target_compression: Overall compression target
            strategy_mapping: Mapping of chunk types to optimization strategies
        """
        super().__init__("MixedDensityOptimizer", target_compression)
        
        # Default strategy mapping
        self.strategy_mapping = strategy_mapping or {
            ChunkType.FACTUAL: OptimizationStrategy.RELEVANCE_BASED,
            ChunkType.PROCEDURAL: OptimizationStrategy.SLIDING_WINDOW,
            ChunkType.TEMPORAL: OptimizationStrategy.SLIDING_WINDOW,
            ChunkType.RELATIONAL: OptimizationStrategy.RELEVANCE_BASED,
            ChunkType.NARRATIVE: OptimizationStrategy.HIERARCHICAL_SUMMARIZATION,
            ChunkType.CONCEPTUAL: OptimizationStrategy.SEMANTIC_COMPRESSION,
            ChunkType.METADATA: OptimizationStrategy.RELEVANCE_BASED
        }
        
        # Initialize sub-optimizers
        self.sub_optimizers = {
            OptimizationStrategy.RELEVANCE_BASED: RelevanceBasedOptimizer(target_compression),
            OptimizationStrategy.SLIDING_WINDOW: SlidingWindowOptimizer(target_compression),
            OptimizationStrategy.HIERARCHICAL_SUMMARIZATION: HierarchicalSummarizer(target_compression),
            OptimizationStrategy.SEMANTIC_COMPRESSION: SemanticCompressionOptimizer(target_compression)
        }
    
    def optimize(
        self,
        window: ContextWindow,
        query: Optional[str] = None
    ) -> Tuple[ContextWindow, Dict[str, Any]]:
        """
        Apply type-specific optimization to each chunk group.
        
        Args:
            window: Context window to optimize
            query: Optional query for relevance-based optimization
            
        Returns:
            Tuple of (optimized_window, optimization_details)
        """
        original_tokens = window.current_token_count
        
        # Group chunks by type
        type_groups = defaultdict(list)
        for chunk in window.chunks:
            type_groups[chunk.chunk_type].append(chunk)
        
        # Optimize each group with its designated strategy
        optimized_groups = {}
        total_original = 0
        total_optimized = 0
        
        for chunk_type, chunks in type_groups.items():
            strategy = self.strategy_mapping.get(
                chunk_type,
                OptimizationStrategy.RELEVANCE_BASED
            )
            
            # Create temporary window for this group
            group_window = ContextWindow(
                max_tokens=sum(c.token_count for c in chunks),
                optimization_strategy=strategy,
                compression_target=self.target_compression
            )
            group_window.add_chunks(chunks)
            
            # Optimize with appropriate optimizer
            optimizer = self.sub_optimizers.get(strategy, RelevanceBasedOptimizer(self.target_compression))
            optimized_group_window, _ = optimizer.optimize(group_window, query)
            
            optimized_groups[chunk_type] = optimized_group_window.chunks
            total_original += group_window.current_token_count
            total_optimized += optimized_group_window.current_token_count
        
        # Combine all optimized chunks
        optimized_window = ContextWindow(
            max_tokens=window.max_tokens,
            optimization_strategy=window.optimization_strategy,
            compression_target=window.compression_target
        )
        
        all_optimized = []
        for chunks in optimized_groups.values():
            all_optimized.extend(chunks)
        
        # Sort by importance and add to final window
        all_optimized.sort(key=lambda c: c.importance_score, reverse=True)
        optimized_window.add_chunks(all_optimized)
        
        # Update statistics
        self._update_stats(
            original_tokens,
            optimized_window.current_token_count,
            len(window.chunks),
            len(window.chunks) - len(optimized_window.chunks)
        )
        
        optimization_details = {
            "strategy": self.name,
            "type_strategies": {t.value: s.value for t, s in self.strategy_mapping.items()},
            "original_chunks": len(window.chunks),
            "optimized_chunks": len(optimized_window.chunks),
            "compression_ratio": optimized_window.current_token_count / max(1, original_tokens),
            "group_details": {
                t.value: {
                    "original": len(c),
                    "optimized": len(o)
                }
                for t, c in type_groups.items()
                for o in [optimized_groups.get(t, [])]
            }
        }
        
        return optimized_window, optimization_details


class ContextOptimizationPipeline:
    """
    Orchestrates end-to-end context optimization workflows.
    
    The pipeline provides a high-level interface for optimizing context
    across different scenarios, supporting multi-pass optimization,
    quality validation, and fallback strategies. It integrates all
    optimizer types and manages the overall optimization lifecycle.
    
    Attributes:
        default_strategy: Primary optimization strategy to use
        fallback_strategies: Strategies to try if primary fails
        max_passes: Maximum optimization passes to perform
        quality_threshold: Minimum quality score to accept
    """
    
    def __init__(
        self,
        default_strategy: OptimizationStrategy = OptimizationStrategy.MIXED_DENSITY,
        fallback_strategies: List[OptimizationStrategy] = None,
        max_passes: int = 3,
        quality_threshold: float = 0.7
    ):
        """
        Initialize the optimization pipeline.
        
        Args:
            default_strategy: Primary optimization strategy
            fallback_strategies: Strategies to try if primary is insufficient
            max_passes: Maximum optimization passes
            quality_threshold: Minimum acceptable quality ratio
        """
        self.default_strategy = default_strategy
        self.fallback_strategies = fallback_strategies or [
            OptimizationStrategy.RELEVANCE_BASED,
            OptimizationStrategy.SLIDING_WINDOW
        ]
        self.max_passes = max_passes
        self.quality_threshold = quality_threshold
        
        # Initialize optimizer registry
        self.optimizer_registry = {
            OptimizationStrategy.RELEVANCE_BASED: RelevanceBasedOptimizer(),
            OptimizationStrategy.SLIDING_WINDOW: SlidingWindowOptimizer(),
            OptimizationStrategy.HIERARCHICAL_SUMMARIZATION: HierarchicalSummarizer(),
            OptimizationStrategy.MIXED_DENSITY: MixedDensityOptimizer(),
            OptimizationStrategy.SEMANTIC_COMPRESSION: SemanticCompressionOptimizer()
        }
        
        self.execution_history = []
    
    def optimize(
        self,
        chunks: List[ContextChunk],
        max_tokens: int,
        query: Optional[str] = None,
        strategy: Optional[OptimizationStrategy] = None
    ) -> Dict[str, Any]:
        """
        Execute the complete optimization pipeline.
        
        This method orchestrates the full optimization process, applying
        the configured strategy and handling fallback scenarios as needed:
            chunks:.
        
        Args List of context chunks to optimize
            max_tokens: Maximum tokens in output
            query: Optional query for relevance-based optimization
            strategy: Override for the default strategy
            
        Returns:
            Dictionary containing optimized chunks and metadata
        """
        strategy = strategy or self.default_strategy
        
        logger.info(f"Starting context optimization with strategy: {strategy.value}")
        
        # Initialize window
        window = ContextWindow(
            max_tokens=max_tokens,
            optimization_strategy=strategy
        )
        window.add_chunks(chunks)
        
        original_count = len(chunks)
        original_tokens = window.current_token_count
        
        # If input is already within limits, return as-is with success
        if original_tokens <= max_tokens:
            pipeline_result = {
                "success": True,
                "original_chunks": original_count,
                "original_tokens": original_tokens,
                "optimized_chunks": len(window.chunks),
                "optimized_tokens": original_tokens,
                "compression_ratio": 1.0,
                "passes": [],
                "final_strategy": strategy.value,
                "chunks": [chunk.to_dict() for chunk in window.chunks],
                "execution_time": datetime.now().isoformat()
            }
            self.execution_history.append(pipeline_result)
            logger.info(f"No optimization needed: {original_tokens} tokens within {max_tokens} limit")
            return pipeline_result
        
        # Execute optimization passes
        results = []
        current_window = window
        current_strategy = strategy
        best_window = window
        best_ratio = 1.0
        
        for pass_num in range(self.max_passes):
            optimizer = self.optimizer_registry.get(
                current_strategy,
                RelevanceBasedOptimizer()
            )
            
            optimized_window, details = optimizer.optimize(current_window, query)
            
            result = {
                "pass": pass_num + 1,
                "strategy": current_strategy.value,
                "details": details,
                "quality_ratio": optimized_window.current_token_count / max(1, original_tokens)
            }
            results.append(result)
            
            # Track best result across passes
            if result["quality_ratio"] < best_ratio:
                best_ratio = result["quality_ratio"]
                best_window = optimized_window
            
            # Check if optimization meets quality threshold
            if result["quality_ratio"] <= self.quality_threshold:
                logger.info(f"Optimization pass {pass_num + 1} met quality threshold")
                current_window = optimized_window
                break
            
            # Try fallback strategy if quality not met
            if pass_num < self.max_passes - 1:
                fallback_idx = min(pass_num, len(self.fallback_strategies) - 1)
                current_strategy = self.fallback_strategies[fallback_idx]
                logger.info(f"Quality not met, trying fallback: {current_strategy.value}")
        else:
            # If no pass met the threshold, use the best result we found
            current_window = best_window
        
        # Final quality validation
        final_quality = current_window.current_token_count / max(1, original_tokens)
        # Success if we achieved the quality threshold OR if we made any improvement
        success = final_quality <= self.quality_threshold or best_ratio < 1.0
        
        # Compile results
        pipeline_result = {
            "success": success,
            "original_chunks": original_count,
            "original_tokens": original_tokens,
            "optimized_chunks": len(current_window.chunks),
            "optimized_tokens": current_window.current_token_count,
            "compression_ratio": final_quality,
            "best_compression_ratio": best_ratio,
            "passes": results,
            "final_strategy": current_strategy.value,
            "chunks": [chunk.to_dict() for chunk in current_window.chunks],
            "execution_time": datetime.now().isoformat()
        }
        
        self.execution_history.append(pipeline_result)
        logger.info(
            f"Optimization complete: {original_count} -> {len(current_window.chunks)} chunks, "
            f"compression: {final_quality:.2%}, success: {success}"
        )
        
        return pipeline_result
    
    def get_optimizer(
        self,
        strategy: OptimizationStrategy
    ) -> ContextOptimizer:
        """Retrieve an optimizer instance for a given strategy."""
        return self.optimizer_registry.get(
            strategy,
            RelevanceBasedOptimizer()
        )
    
    def register_optimizer(
        self,
        strategy: OptimizationStrategy,
        optimizer: ContextOptimizer
    ):
        """Register a custom optimizer for a strategy."""
        self.optimizer_registry[strategy] = optimizer
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Return the history of all optimization executions."""
        return self.execution_history


def create_context_from_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    preserve_sentences: bool = True
) -> List[ContextChunk]:
    """
    Utility function to split text into optimized context chunks.
    
    This function provides a convenient way to convert raw text into
    context chunks suitable for the optimization pipeline, with sensible
    defaults for common use cases.
    
    Args:
        text: Raw text to chunk
        chunk_size: Target token count per chunk
        chunk_overlap: Overlap between chunks for continuity
        preserve_sentences: Whether to split on sentence boundaries
        
    Returns:
        List of ContextChunk objects
    """
    chunks = []
    
    # Split text into sentences if requested
    if preserve_sentences:
        sentences = re.split(r'(?<=[.!?])\s+', text)
    else:
        sentences = text.split()
    
    current_chunk_parts = []
    current_token_count = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = ContextChunk._estimate_tokens(sentence)
        
        # Check if adding this sentence would exceed chunk size
        if current_token_count + sentence_tokens > chunk_size and current_chunk_parts:
            # Create chunk from accumulated parts
            chunk_content = ' '.join(current_chunk_parts)
            
            # Determine chunk type based on content
            chunk_type = _infer_chunk_type(chunk_content)
            
            chunk = ContextChunk(
                content=chunk_content,
                chunk_type=chunk_type,
                position=len(chunks),
                importance_score=_calculate_importance(chunk_content, i, len(sentences))
            )
            chunks.append(chunk)
            
            # Start new chunk with overlap
            overlap_parts = current_chunk_parts[-(chunk_overlap // 10):] if chunk_overlap > 0 else []
            current_chunk_parts = overlap_parts + [sentence]
            current_token_count = sum(ContextChunk._estimate_tokens(p) for p in current_chunk_parts)
        else:
            current_chunk_parts.append(sentence)
            current_token_count += sentence_tokens
    
    # Don't forget the last chunk
    if current_chunk_parts:
        chunk_content = ' '.join(current_chunk_parts)
        chunk_type = _infer_chunk_type(chunk_content)
        chunk = ContextChunk(
            content=chunk_content,
            chunk_type=chunk_type,
            position=len(chunks),
            importance_score=_calculate_importance(chunk_content, len(sentences), len(sentences))
        )
        chunks.append(chunk)
    
    return chunks


def _infer_chunk_type(content: str) -> ChunkType:
    """
    Infer the type of content based on text analysis.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Inferred ChunkType
    """
    content_lower = content.lower()
    
    # Check for procedural content
    if any(word in content_lower for word in ['step', 'first', 'then', 'finally', 'process', 'procedure']):
        return ChunkType.PROCEDURAL
    
    # Check for temporal content
    if any(word in content_lower for word in ['when', 'before', 'after', 'during', 'timeline', 'date', 'time']):
        return ChunkType.TEMPORAL
    
    # Check for relational content
    if any(word in content_lower for word in ['related', 'connected', 'associated', 'relationship', 'link']):
        return ChunkType.RELATIONAL
    
    # Check for narrative content
    if any(word in content_lower for word in ['story', 'narrative', 'chapter', 'event', 'happened']):
        return ChunkType.NARRATIVE
    
    # Check for conceptual content
    if any(word in content_lower for word in ['concept', 'principle', 'theory', 'idea', 'framework']):
        return ChunkType.CONCEPTUAL
    
    # Default to factual
    return ChunkType.FACTUAL


def _calculate_importance(content: str, position: int, total: int) -> float:
    """
    Calculate importance score based on content and position.
    
    Args:
        content: Text content
        position: Position in original sequence
        total: Total number of items
        
    Returns:
        Importance score between 0.0 and 1.0
    """
    # Position-based boost (later items slightly more important)
    position_score = 0.3 + (0.4 * position / max(1, total))
    
    # Content-based factors
    content_lower = content.lower()
    has_numbers = bool(re.search(r'\d+', content))
    has_keyterms = any(word in content_lower for word in [
        'important', 'key', 'critical', 'essential', 'main', 'primary'
    ])
    
    content_score = 0.3
    if has_numbers:
        content_score += 0.2
    if has_keyterms:
        content_score += 0.2
    
    # Combine scores
    importance = min(1.0, position_score * 0.4 + content_score * 0.6)
    
    return importance


# Convenience class for common operations
class ContextManager:
    """
    Simplified interface for common context optimization operations.
    
    This class provides a convenient high-level API for typical use cases,
    hiding the complexity of the underlying optimization pipeline.
    """
    
    def __init__(self, max_tokens: int = 4000):
        """
        Initialize the context manager.
        
        Args:
            max_tokens: Default maximum token limit
        """
        self.max_tokens = max_tokens
        self.pipeline = ContextOptimizationPipeline()
    
    def optimize_text(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        strategy: Optional[OptimizationStrategy] = None
    ) -> Dict[str, Any]:
        """
        Optimize raw text for context window.
        
        Args:
            text: Raw text to optimize
            max_tokens: Override for maximum tokens
            strategy: Optimization strategy to use
            
        Returns:
            Optimization result with chunks and metadata
        """
        chunks = create_context_from_text(text)
        return self.pipeline.optimize(
            chunks,
            max_tokens or self.max_tokens,
            strategy=strategy
        )
    
    def optimize_chunks(
        self,
        chunks: List[ContextChunk],
        max_tokens: Optional[int] = None,
        query: Optional[str] = None,
        strategy: Optional[OptimizationStrategy] = None
    ) -> Dict[str, Any]:
        """
        Optimize pre-existing context chunks.
        
        Args:
            chunks: List of ContextChunk objects
            max_tokens: Override for maximum tokens
            query: Optional query for relevance optimization
            strategy: Optimization strategy to use
            
        Returns:
            Optimization result with chunks and metadata
        """
        return self.pipeline.optimize(
            chunks,
            max_tokens or self.max_tokens,
            query=query,
            strategy=strategy
        )
    
    def get_compression_report(self) -> Dict[str, Any]:
        """Get a summary report of recent optimization activities."""
        history = self.pipeline.get_optimization_history()
        
        if not history:
            return {"message": "No optimization history available"}
        
        return {
            "total_operations": len(history),
            "avg_compression_ratio": sum(
                r["compression_ratio"] for r in history
            ) / len(history),
            "success_rate": sum(
                1 for r in history if r["success"]
            ) / len(history),
            "recent_operations": history[-5:]
        }
