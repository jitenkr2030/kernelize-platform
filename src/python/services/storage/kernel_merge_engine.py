#!/usr/bin/env python3
"""
Kernel Merge Engine
====================

Implements sophisticated kernel merging capabilities with semantic alignment,
conflict resolution, partial merges, and human-in-the-loop workflows.

Features:
- Semantic alignment for entity conflict resolution
- Consistency validation for contradiction detection
- Partial merges combining specific sections from different kernels
- Merge conflict resolution workflows with human-in-the-loop options
- Automated and manual merge strategies

Author: MiniMax Agent
"""

import json
import hashlib
import uuid
import re
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)


class MergeStrategy(Enum):
    """Merge strategies for kernel combination"""
    CONCATENATE = "concatenate"  # Simple concatenation of content
    SEMANTIC = "semantic"  # Semantic alignment and intelligent merging
    UNIFY = "unify"  # Unify entities and resolve conflicts automatically
    HUMAN_REVIEW = "human_review"  # Flag conflicts for human resolution


class ConflictType(Enum):
    """Types of merge conflicts"""
    ENTITY_MISMATCH = "entity_mismatch"  # Same entity, different values
    DUPLICATE_ENTITY = "duplicate_entity"  # Same entity exists in both kernels
    CONTRADICTION = "contradiction"  # Conflicting information detected
    DEPENDENCY_CYCLE = "dependency_cycle"  # Circular dependency detected
    SCHEMA_INCOMPATIBLE = "schema_incompatible"  # Incompatible schema versions
    DUPLICATE_DOCUMENT = "duplicate_document"  # Document with same title exists


class ConflictResolution(Enum):
    """Resolution strategies for conflicts"""
    KEEP_SOURCE = "keep_source"  # Keep source kernel's version
    KEEP_TARGET = "keep_target"  # Keep target kernel's version
    KEEP_BOTH = "keep_both"  # Keep both versions as separate entries
    MERGE_AUTO = "merge_auto"  # Attempt automatic merge
    RESOLVE_MANUAL = "resolve_manual"  # Require manual resolution


@dataclass
class EntityKey:
    """Identifies an entity for merging purposes"""
    entity_type: str
    primary_key: str  # Main identifier (name, title, etc.)
    secondary_keys: List[str] = field(default_factory=list)  # Alternative identifiers
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "primary_key": self.primary_key,
            "secondary_keys": self.secondary_keys,
        }


@dataclass
class MergeConflict:
    """Represents a conflict detected during merge"""
    conflict_id: str
    conflict_type: ConflictType
    source_kernel_id: str
    target_kernel_id: str
    entity_type: str
    entity_id: str
    source_value: Any
    target_value: Any
    conflict_details: Dict[str, Any]
    suggested_resolution: ConflictResolution = ConflictResolution.RESOLVE_MANUAL
    resolution_notes: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution: Optional[ConflictResolution] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "source_kernel_id": self.source_kernel_id,
            "target_kernel_id": self.target_kernel_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "source_value": self._serialize_value(self.source_value),
            "target_value": self._serialize_value(self.target_value),
            "conflict_details": self.conflict_details,
            "suggested_resolution": self.suggested_resolution.value,
            "resolution_notes": self.resolution_notes,
            "created_at": self.created_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolved_by": self.resolved_by,
            "resolution": self.resolution.value if self.resolution else None,
        }
    
    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON storage"""
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, (list, dict)):
            return value
        return str(value)


@dataclass
class MergeSection:
    """Defines a section to merge from source kernel"""
    section_type: str  # "documents", "chunks", "entities", "relationships"
    filter_criteria: Dict[str, Any]  # Criteria to select specific items
    merge_behavior: str = "append"  # "append", "replace", "merge"


@dataclass
class MergeResult:
    """Result of a merge operation"""
    success: bool
    merged_kernel_id: Optional[str]
    source_kernel_id: str
    target_kernel_id: str
    conflicts: List[MergeConflict]
    unresolved_conflicts: List[MergeConflict]
    merged_documents: int
    merged_chunks: int
    merged_entities: int
    merged_relationships: int
    conflicts_resolved_automatically: int
    conflicts_resolved_manually: int
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "merged_kernel_id": self.merged_kernel_id,
            "source_kernel_id": self.source_kernel_id,
            "target_kernel_id": self.target_kernel_id,
            "total_conflicts": len(self.conflicts),
            "unresolved_conflicts": len(self.unresolved_conflicts),
            "conflicts_resolved": (
                self.conflicts_resolved_automatically + 
                self.conflicts_resolved_manually
            ),
            "merged_documents": self.merged_documents,
            "merged_chunks": self.merged_chunks,
            "merged_entities": self.merged_entities,
            "merged_relationships": self.merged_relationships,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class PartialMergeRequest:
    """Request for partial kernel merge"""
    source_kernel_id: str
    target_kernel_id: Optional[str]  # None for new kernel
    sections: List[MergeSection]
    merge_strategy: MergeStrategy = MergeStrategy.SEMANTIC
    conflict_resolution: ConflictResolution = ConflictResolution.RESOLVE_MANUAL
    custom_resolvers: Dict[str, Callable] = field(default_factory=dict)
    preserve_metadata: bool = True


class EntityExtractor:
    """
    Extracts and normalizes entities from kernel content for merging
    """
    
    ENTITY_PATTERNS = {
        "document": r"document[:\s]+([A-Z][a-zA-Z\s]*)",
        "concept": r"concept[:\s]+([A-Z][a-zA-Z\s]*)",
        "term": r"term[:\s]+([A-Z][a-zA-Z\s]*)",
        "definition": r"definition[:\s]+(.+?)(?:\.\s*[A-Z]|\n\n)",
    }
    
    def __init__(self):
        """Initialize entity extractor"""
        self._compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.ENTITY_PATTERNS.items()
        }
    
    def extract_entities(
        self,
        content: str,
        entity_types: Optional[List[str]] = None,
    ) -> List[EntityKey]:
        """
        Extract entities from content
        
        Args:
            content: Text content to analyze
            entity_types: Types of entities to extract (None for all)
            
        Returns:
            List of EntityKey objects
        """
        entities = []
        
        if entity_types is None:
            entity_types = list(self.ENTITY_PATTERNS.keys())
        
        for entity_type in entity_types:
            if entity_type not in self._compiled_patterns:
                continue
            
            pattern = self._compiled_patterns[entity_type]
            for match in pattern.finditer(content):
                primary_key = match.group(1).strip()
                entities.append(EntityKey(
                    entity_type=entity_type,
                    primary_key=primary_key,
                    secondary_keys=self._extract_secondary_keys(content, primary_key),
                ))
        
        return entities
    
    def _extract_secondary_keys(self, content: str, primary_key: str) -> List[str]:
        """Extract alternative identifiers for an entity"""
        secondary = []
        
        # Find variations in the content
        variations = [
            primary_key.lower(),
            primary_key.upper(),
            primary_key.replace(" ", ""),
            primary_key.replace("-", "_"),
        ]
        
        # Add common variations
        for variation in variations:
            if variation != primary_key.lower():
                secondary.append(variation)
        
        return secondary
    
    def normalize_entity_key(self, key: str) -> str:
        """Normalize an entity key for comparison"""
        # Convert to lowercase
        normalized = key.lower().strip()
        
        # Remove special characters
        normalized = re.sub(r'[^\w\s-]', '', normalized)
        
        # Replace multiple spaces/hyphens with single
        normalized = re.sub(r'[\s-]+', ' ', normalized)
        
        return normalized


class SemanticAligner:
    """
    Performs semantic alignment between entities from different kernels
    """
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Initialize semantic aligner
        
        Args:
            similarity_threshold: Minimum similarity for entity matching (0.0-1.0)
        """
        self.similarity_threshold = similarity_threshold
        self.entity_extractor = EntityExtractor()
    
    def align_entities(
        self,
        source_entities: List[Dict[str, Any]],
        target_entities: List[Dict[str, Any]],
        entity_type: str,
    ) -> List[Tuple[Optional[str], Optional[str]]]:
        """
        Align entities between source and target kernels
        
        Returns:
            List of (source_entity_id, target_entity_id) pairs
        """
        alignments = []
        
        # Build entity maps for efficient lookup
        source_map = self._build_entity_map(source_entities, entity_type)
        target_map = self._build_entity_map(target_entities, entity_type)
        
        for source_entity in source_entities:
            source_id = source_entity.get("id", "")
            source_key = self._get_entity_key(source_entity, entity_type)
            normalized_source = self.entity_extractor.normalize_entity_key(source_key)
            
            best_match = None
            best_similarity = 0.0
            
            for target_entity in target_entities:
                target_id = target_entity.get("id", "")
                target_key = self._get_entity_key(target_entity, entity_type)
                normalized_target = self.entity_extractor.normalize_entity_key(target_key)
                
                # Calculate similarity
                similarity = self._calculate_similarity(normalized_source, normalized_target)
                
                if similarity >= self.similarity_threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_match = target_id
            
            if best_match:
                alignments.append((source_id, best_match))
            else:
                alignments.append((source_id, None))
        
        return alignments
    
    def _build_entity_map(
        self,
        entities: List[Dict[str, Any]],
        entity_type: str,
    ) -> Dict[str, Dict[str, Any]]:
        """Build a map of entities by normalized key"""
        entity_map = {}
        
        for entity in entities:
            key = self._get_entity_key(entity, entity_type)
            normalized = self.entity_extractor.normalize_entity_key(key)
            
            if normalized not in entity_map:
                entity_map[normalized] = []
            entity_map[normalized].append(entity)
        
        return entity_map
    
    def _get_entity_key(self, entity: Dict[str, Any], entity_type: str) -> str:
        """Get the primary key for an entity"""
        if entity_type == "document":
            return entity.get("title", entity.get("name", ""))
        elif entity_type == "chunk":
            return entity.get("text_content", "")[:100]
        elif entity_type == "concept":
            return entity.get("name", "")
        else:
            return entity.get("name", entity.get("title", str(entity.get("id", ""))))
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using multiple methods"""
        # Exact match
        if str1 == str2:
            return 1.0
        
        # Substring matching
        if str1 in str2 or str2 in str1:
            return 0.9
        
        # SequenceMatcher for fuzzy matching
        seq_similarity = SequenceMatcher(None, str1, str2).ratio()
        
        # Token-based Jaccard similarity
        tokens1 = set(str1.split())
        tokens2 = set(str2.split())
        
        if tokens1 and tokens2:
            jaccard = len(tokens1 & tokens2) / len(tokens1 | tokens2)
        else:
            jaccard = 0.0
        
        # Combined score (weighted average)
        combined = seq_similarity * 0.6 + jaccard * 0.4
        
        return combined


class ConsistencyValidator:
    """
    Validates consistency between kernels being merged
    """
    
    CONTRADICTION_PATTERNS = [
        # Direct contradictions
        (r"not.*(yes|true|correct)", r"(yes|true|correct)"),
        (r"never.*(always|frequently)", r"(always|frequently)"),
        (r"impossible.*(possible|can)", r"(possible|can)"),
        
        # Numerical contradictions
        (r"greater.*than\s+(\d+)", r"less.*than\s+(\d+)"),
        (r"before\s+(\d{4})", r"after\s+(\d{4})"),
    ]
    
    def __init__(self):
        """Initialize consistency validator"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile contradiction detection patterns"""
        self._patterns = []
        for positive, negative in self.CONTRADICTION_PATTERNS:
            self._patterns.append((
                re.compile(positive, re.IGNORECASE),
                re.compile(negative, re.IGNORECASE),
            ))
    
    def validate_consistency(
        self,
        source_content: str,
        target_content: str,
        entity_type: str,
    ) -> List[Dict[str, Any]]:
        """
        Check for contradictions between source and target content
        
        Returns:
            List of detected contradictions
        """
        contradictions = []
        
        for pos_pattern, neg_pattern in self._patterns:
            # Find positive assertions in source
            for pos_match in pos_pattern.finditer(source_content):
                pos_context = self._extract_context(source_content, pos_match.start(), pos_match.end())
                
                # Check for negative assertions in target
                for neg_match in neg_pattern.finditer(target_content):
                    neg_context = self._extract_context(
                        target_content, neg_match.start(), neg_match.end()
                    )
                    
                    contradictions.append({
                        "type": "contradiction",
                        "source_context": pos_context,
                        "target_context": neg_context,
                        "confidence": 0.8,
                        "pattern_detected": f"{pos_match.group()} vs {neg_match.group()}",
                    })
        
        return contradictions
    
    def _extract_context(self, text: str, start: int, end: int, context_size: int = 100) -> str:
        """Extract surrounding context for a match"""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        return text[context_start:context_end].strip()
    
    def check_schema_compatibility(
        self,
        source_schema: str,
        target_schema: str,
    ) -> Tuple[bool, str]:
        """
        Check if schemas are compatible for merging
        
        Returns:
            (is_compatible, message)
        """
        # Parse schema versions
        source_version = self._parse_version(source_schema)
        target_version = self._parse_version(target_schema)
        
        if source_version is None or target_version is None:
            return True, "Schema version parsing failed, proceeding with caution"
        
        # Major version mismatch is incompatible
        if source_version[0] != target_version[0]:
            return False, f"Major schema version mismatch: {source_schema} vs {target_schema}"
        
        # Minor version difference requires migration
        if source_version[1] != target_version[1]:
            return (
                True,
                f"Minor version difference detected: {source_schema} to {target_schema}. "
                "Fields may need migration."
            )
        
        return True, "Schemas are compatible"
    
    def _parse_version(self, version_str: str) -> Optional[Tuple[int, int, int]]:
        """Parse semantic version string"""
        try:
            parts = version_str.split(".")
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except (ValueError, IndexError):
            return None


class ConflictResolver:
    """
    Resolves conflicts detected during kernel merging
    """
    
    def __init__(self, custom_resolvers: Optional[Dict[str, Callable]] = None):
        """
        Initialize conflict resolver
        
        Args:
            custom_resolvers: Custom resolution functions by conflict type
        """
        self.custom_resolvers = custom_resolvers or {}
        self.semantic_aligner = SemanticAligner()
        self.resolution_history: List[Dict[str, Any]] = []
    
    def resolve_conflict(
        self,
        conflict: MergeConflict,
        strategy: ConflictResolution,
        source_data: Any,
        target_data: Any,
    ) -> Tuple[Any, str]:
        """
        Resolve a merge conflict
        
        Args:
            conflict: The conflict to resolve
            strategy: Resolution strategy to apply
            source_data: Source kernel data
            target_data: Target kernel data
            
        Returns:
            (resolved_data, resolution_description)
        """
        # Check for custom resolver
        conflict_type_key = conflict.conflict_type.value
        if conflict_type_key in self.custom_resolvers:
            return self.custom_resolvers[conflict_type_key](
                conflict, source_data, target_data
            )
        
        # Apply resolution strategy
        if strategy == ConflictResolution.KEEP_SOURCE:
            return source_data, f"Kept source kernel's version of {conflict.entity_type}"
        
        elif strategy == ConflictResolution.KEEP_TARGET:
            return target_data, f"Kept target kernel's version of {conflict.entity_type}"
        
        elif strategy == ConflictResolution.KEEP_BOTH:
            return {
                "source": source_data,
                "target": target_data,
                "note": f"Dual versions kept due to unresolved conflict",
            }, f"Kept both versions of {conflict.entity_type}"
        
        elif strategy == ConflictResolution.MERGE_AUTO:
            return self._auto_merge(conflict, source_data, target_data)
        
        else:
            return target_data, f"Defaulted to target kernel's version"
    
    def _auto_merge(
        self,
        conflict: MergeConflict,
        source_data: Any,
        target_data: Any,
    ) -> Tuple[Any, str]:
        """Attempt automatic merging of conflicting data"""
        if conflict.conflict_type == ConflictType.DUPLICATE_ENTITY:
            # For duplicates, keep the one with more complete information
            source_completeness = self._calculate_completeness(source_data)
            target_completeness = self._calculate_completeness(target_data)
            
            if source_completeness > target_completeness:
                return source_data, "Auto-merged: kept more complete source version"
            else:
                return target_data, "Auto-merged: kept more complete target version"
        
        elif conflict.conflict_type == ConflictType.ENTITY_MISMATCH:
            # For mismatches, try to merge based on field type
            if isinstance(source_data, dict) and isinstance(target_data, dict):
                merged = {**target_data, **source_data}
                return merged, "Auto-merged: combined fields from both versions"
        
        elif conflict.conflict_type == ConflictType.CONTRADICTION:
            # For contradictions, flag for review
            return {
                "source": source_data,
                "target": target_data,
                "flagged": True,
                "flag_reason": "Contradiction detected",
            }, "Auto-merge failed: contradiction requires manual review"
        
        # Default: keep target
        return target_data, "Auto-merge defaulted to target version"
    
    def _calculate_completeness(self, data: Any) -> float:
        """Calculate completeness score for data structure"""
        if not isinstance(data, dict):
            return 0.5
        
        if not data:
            return 0.0
        
        # Count non-empty fields
        non_empty = sum(1 for v in data.values() if v and v != [])
        total = len(data)
        
        return non_empty / total if total > 0 else 0.0
    
    def record_resolution(
        self,
        conflict: MergeConflict,
        resolution: ConflictResolution,
        resolved_by: str = "system",
    ):
        """Record a resolution for audit trail"""
        self.resolution_history.append({
            "conflict_id": conflict.conflict_id,
            "conflict_type": conflict.conflict_type.value,
            "resolution": resolution.value,
            "resolved_by": resolved_by,
            "timestamp": datetime.utcnow().isoformat(),
        })
        
        conflict.resolved_at = datetime.utcnow()
        conflict.resolved_by = resolved_by
        conflict.resolution = resolution


class KernelMergeEngine:
    """
    Main kernel merge engine with comprehensive merge capabilities
    """
    
    def __init__(
        self,
        storage_backend,
        similarity_threshold: float = 0.85,
        auto_merge_threshold: float = 0.9,
    ):
        """
        Initialize merge engine
        
        Args:
            storage_backend: Storage backend for kernel operations
            similarity_threshold: Entity similarity threshold for alignment
            auto_merge_threshold: Confidence threshold for automatic conflict resolution
        """
        self.storage = storage_backend
        self.similarity_threshold = similarity_threshold
        self.auto_merge_threshold = auto_merge_threshold
        
        # Initialize components
        self.semantic_aligner = SemanticAligner(similarity_threshold)
        self.consistency_validator = ConsistencyValidator()
        self.conflict_resolver = ConflictResolver()
        self.entity_extractor = EntityExtractor()
        
        # Merge statistics
        self._stats = {
            "total_merges": 0,
            "successful_merges": 0,
            "failed_merges": 0,
            "conflicts_detected": 0,
            "auto_resolved": 0,
            "manual_resolved": 0,
        }
    
    def merge_kernels(
        self,
        source_kernel_id: str,
        target_kernel_id: str,
        merge_strategy: MergeStrategy = MergeStrategy.SEMANTIC,
        conflict_resolution: ConflictResolution = ConflictResolution.RESOLVE_MANUAL,
        sections: Optional[List[MergeSection]] = None,
    ) -> MergeResult:
        """
        Merge two kernels together
        
        Args:
            source_kernel_id: ID of source kernel
            target_kernel_id: ID of target kernel
            merge_strategy: Strategy for merging
            conflict_resolution: Default conflict resolution approach
            sections: Specific sections to merge (None for all)
            
        Returns:
            MergeResult with merge outcome
        """
        start_time = time.time()
        
        # Initialize result
        result = MergeResult(
            success=False,
            merged_kernel_id=None,
            source_kernel_id=source_kernel_id,
            target_kernel_id=target_kernel_id,
            conflicts=[],
            unresolved_conflicts=[],
            merged_documents=0,
            merged_chunks=0,
            merged_entities=0,
            merged_relationships=0,
            conflicts_resolved_automatically=0,
            conflicts_resolved_manually=0,
            duration_ms=0,
        )
        
        try:
            # Validate kernels exist
            source_kernel = self.storage.get_kernel(source_kernel_id)
            target_kernel = self.storage.get_kernel(target_kernel_id)
            
            if not source_kernel:
                result.metadata["error"] = f"Source kernel not found: {source_kernel_id}"
                return result
            
            if not target_kernel:
                result.metadata["error"] = f"Target kernel not found: {target_kernel_id}"
                return result
            
            # Check schema compatibility
            compatible, message = self.consistency_validator.check_schema_compatibility(
                source_kernel.get("schema_version", "1.0"),
                target_kernel.get("schema_version", "1.0"),
            )
            
            if not compatible:
                result.metadata["error"] = message
                return result
            
            # Detect conflicts
            conflicts = self._detect_conflicts(source_kernel, target_kernel)
            result.conflicts = conflicts
            
            # Resolve conflicts
            auto_resolved = 0
            manual_resolved = 0
            unresolved = []
            
            for conflict in conflicts:
                # Determine resolution strategy
                if conflict.confidence >= self.auto_merge_threshold:
                    # Auto-resolve high confidence conflicts
                    resolved_data, _ = self.conflict_resolver.resolve_conflict(
                        conflict,
                        conflict.suggested_resolution,
                        conflict.source_value,
                        conflict.target_value,
                    )
                    auto_resolved += 1
                else:
                    # Flag for manual review
                    if conflict_resolution == ConflictResolution.RESOLVE_MANUAL:
                        unresolved.append(conflict)
                    else:
                        resolved_data, _ = self.conflict_resolver.resolve_conflict(
                            conflict,
                            conflict_resolution,
                            conflict.source_value,
                            conflict.target_value,
                        )
                        auto_resolved += 1
            
            result.unresolved_conflicts = unresolved
            result.conflicts_resolved_automatically = auto_resolved
            result.conflicts_resolved_manually = manual_resolved
            
            # If strategy is HUMAN_REVIEW and there are unresolved conflicts
            if merge_strategy == MergeStrategy.HUMAN_REVIEW and unresolved:
                result.metadata["status"] = "pending_review"
                result.metadata["pending_conflicts"] = len(unresolved)
                result.duration_ms = (time.time() - start_time) * 1000
                return result
            
            # Perform merge based on strategy
            if merge_strategy == MergeStrategy.CONCATENATE:
                merged_id = self._merge_concatenate(source_kernel, target_kernel, result)
            elif merge_strategy == MergeStrategy.SEMANTIC:
                merged_id = self._merge_semantic(source_kernel, target_kernel, result)
            elif merge_strategy == MergeStrategy.UNIFY:
                merged_id = self._merge_unify(source_kernel, target_kernel, result)
            else:
                merged_id = target_kernel_id  # Default to target
            
            result.merged_kernel_id = merged_id
            result.success = True
            
            # Update statistics
            self._stats["total_merges"] += 1
            self._stats["successful_merges"] += 1
            self._stats["conflicts_detected"] += len(conflicts)
            self._stats["auto_resolved"] += auto_resolved
            self._stats["manual_resolved"] += manual_resolved
            
        except Exception as e:
            logger.error(f"Kernel merge failed: {e}")
            result.metadata["error"] = str(e)
            self._stats["failed_merges"] += 1
        
        result.duration_ms = (time.time() - start_time) * 1000
        return result
    
    def partial_merge(
        self,
        request: PartialMergeRequest,
    ) -> MergeResult:
        """
        Perform partial merge of specific sections
        
        Args:
            request: Partial merge request with sections to merge
            
        Returns:
            MergeResult with merge outcome
        """
        start_time = time.time()
        
        # Create target kernel if not specified
        if request.target_kernel_id is None:
            from src.python.services.storage.postgres_storage import KernelMetadata
            target_id = self.storage.create_kernel(KernelMetadata(
                name=f"Merged Kernel {datetime.utcnow().isoformat()}",
                description="Created via partial merge",
            ))
        else:
            target_id = request.target_kernel_id
        
        result = MergeResult(
            success=False,
            merged_kernel_id=target_id,
            source_kernel_id=request.source_kernel_id,
            target_kernel_id=target_id,
            conflicts=[],
            unresolved_conflicts=[],
            merged_documents=0,
            merged_chunks=0,
            merged_entities=0,
            merged_relationships=0,
            conflicts_resolved_automatically=0,
            conflicts_resolved_manually=0,
            duration_ms=0,
        )
        
        try:
            for section in request.sections:
                section_result = self._merge_section(
                    request.source_kernel_id,
                    target_id,
                    section,
                    request.conflict_resolution,
                )
                
                # Accumulate results
                result.merged_documents += section_result.get("documents", 0)
                result.merged_chunks += section_result.get("chunks", 0)
                result.merged_entities += section_result.get("entities", 0)
                result.merged_relationships += section_result.get("relationships", 0)
                result.conflicts.extend(section_result.get("conflicts", []))
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Partial merge failed: {e}")
            result.metadata["error"] = str(e)
        
        result.duration_ms = (time.time() - start_time) * 1000
        return result
    
    def _detect_conflicts(
        self,
        source_kernel: Dict[str, Any],
        target_kernel: Dict[str, Any],
    ) -> List[MergeConflict]:
        """Detect conflicts between two kernels"""
        conflicts = []
        
        # Check for duplicate documents
        source_docs = self.storage.get_documents_by_kernel(source_kernel["id"])
        target_docs = self.storage.get_documents_by_kernel(target_kernel["id"])
        
        source_titles = {d.get("title", "").lower() for d in source_docs}
        target_titles = {d.get("title", "").lower() for d in target_docs}
        
        duplicates = source_titles & target_titles
        for dup_title in duplicates:
            conflicts.append(MergeConflict(
                conflict_id=str(uuid.uuid4()),
                conflict_type=ConflictType.DUPLICATE_DOCUMENT,
                source_kernel_id=source_kernel["id"],
                target_kernel_id=target_kernel["id"],
                entity_type="document",
                entity_id=dup_title,
                source_value=[d for d in source_docs if d.get("title", "").lower() == dup_title],
                target_value=[d for d in target_docs if d.get("title", "").lower() == dup_title],
                conflict_details={"title": dup_title},
                suggested_resolution=ConflictResolution.KEEP_BOTH,
            ))
        
        # Check for schema conflicts
        if source_kernel.get("schema_version") != target_kernel.get("schema_version"):
            conflicts.append(MergeConflict(
                conflict_id=str(uuid.uuid4()),
                conflict_type=ConflictType.SCHEMA_INCOMPATIBLE,
                source_kernel_id=source_kernel["id"],
                target_kernel_id=target_kernel["id"],
                entity_type="kernel",
                entity_id="schema",
                source_value=source_kernel.get("schema_version"),
                target_value=target_kernel.get("schema_version"),
                conflict_details={
                    "message": "Schema versions differ",
                    "source": source_kernel.get("schema_version"),
                    "target": target_kernel.get("schema_version"),
                },
                suggested_resolution=ConflictResolution.MERGE_AUTO,
            ))
        
        return conflicts
    
    def _merge_concatenate(
        self,
        source_kernel: Dict[str, Any],
        target_kernel: Dict[str, Any],
        result: MergeResult,
    ) -> str:
        """Perform simple concatenation merge"""
        source_docs = self.storage.get_documents_by_kernel(source_kernel["id"])
        
        for doc in source_docs:
            # Skip documents with duplicate titles
            existing_docs = self.storage.get_documents_by_kernel(target_kernel["id"])
            if any(d.get("title") == doc.get("title") for d in existing_docs):
                continue
            
            # Add document to target
            doc_id = self.storage.add_document(
                kernel_id=target_kernel["id"],
                title=doc.get("title", "Untitled"),
                content_hash=doc.get("content_hash", ""),
                blob_path=doc.get("blob_path"),
                metadata=doc.get("metadata", {}),
                source_url=doc.get("source_url"),
                author=doc.get("author"),
                published_date=doc.get("published_date"),
            )
            result.merged_documents += 1
            
            # Add chunks for this document
            source_chunks = self.storage.get_chunks_by_document(doc["id"])
            for chunk in source_chunks:
                self.storage.add_chunk(
                    document_id=doc_id,
                    kernel_id=target_kernel["id"],
                    chunk_index=chunk.get("chunk_index", 0),
                    text_content=chunk.get("text_content", ""),
                    content_hash=chunk.get("content_hash", ""),
                    qdrant_point_id=chunk.get("qdrant_point_id"),
                    metadata=chunk.get("metadata", {}),
                )
                result.merged_chunks += 1
        
        # Create merge version
        self.storage.create_version(
            kernel_id=target_kernel["id"],
            commit_message=f"Merged from kernel {source_kernel['id']}",
            content_hash=hashlib.md5(json.dumps(source_kernel).encode()).hexdigest(),
            change_summary={
                "action": "concatenate_merge",
                "source_kernel": source_kernel["id"],
                "documents_added": result.merged_documents,
            },
            diff_summary=f"Added {result.merged_documents} documents from merge",
            created_by="merge_engine",
        )
        
        return target_kernel["id"]
    
    def _merge_semantic(
        self,
        source_kernel: Dict[str, Any],
        target_kernel: Dict[str, Any],
        result: MergeResult,
    ) -> str:
        """Perform semantic merge with entity alignment"""
        source_docs = self.storage.get_documents_by_kernel(source_kernel["id"])
        target_docs = self.storage.get_documents_by_kernel(target_kernel["id"])
        
        # Align documents
        alignments = self.semantic_aligner.align_entities(
            source_docs, target_docs, "document"
        )
        
        for source_doc in source_docs:
            source_id = source_doc.get("id")
            
            # Find aligned target document
            aligned = [a for a in alignments if a[0] == source_id]
            
            if aligned and aligned[0][1]:
                # Document aligned - merge content
                target_doc_id = aligned[0][1]
                self._merge_document_content(source_doc, target_doc_id, result)
            else:
                # No alignment - add as new document
                doc_id = self.storage.add_document(
                    kernel_id=target_kernel["id"],
                    title=source_doc.get("title", "Untitled"),
                    content_hash=source_doc.get("content_hash", ""),
                    blob_path=source_doc.get("blob_path"),
                    metadata=source_doc.get("metadata", {}),
                    source_url=source_doc.get("source_url"),
                    author=source_doc.get("author"),
                    published_date=source_doc.get("published_date"),
                )
                result.merged_documents += 1
                
                # Add chunks
                source_chunks = self.storage.get_chunks_by_document(source_id)
                for chunk in source_chunks:
                    self.storage.add_chunk(
                        document_id=doc_id,
                        kernel_id=target_kernel["id"],
                        chunk_index=chunk.get("chunk_index", 0),
                        text_content=chunk.get("text_content", ""),
                        content_hash=chunk.get("content_hash", ""),
                        qdrant_point_id=chunk.get("qdrant_point_id"),
                        metadata=chunk.get("metadata", {}),
                    )
                    result.merged_chunks += 1
        
        # Create merge version
        self.storage.create_version(
            kernel_id=target_kernel["id"],
            commit_message=f"Semantic merge from kernel {source_kernel['id']}",
            content_hash=hashlib.md5(json.dumps(source_kernel).encode()).hexdigest(),
            change_summary={
                "action": "semantic_merge",
                "source_kernel": source_kernel["id"],
                "documents_merged": result.merged_documents,
                "conflicts_resolved": result.conflicts_resolved_automatically,
            },
            diff_summary=f"Semantically merged {result.merged_documents} documents",
            created_by="merge_engine",
        )
        
        return target_kernel["id"]
    
    def _merge_unify(
        self,
        source_kernel: Dict[str, Any],
        target_kernel: Dict[str, Any],
        result: MergeResult,
    ) -> str:
        """Perform unified merge with automatic conflict resolution"""
        # Similar to semantic merge but with aggressive auto-resolution
        result.conflict_resolution = ConflictResolution.MERGE_AUTO
        
        return self._merge_semantic(source_kernel, target_kernel, result)
    
    def _merge_document_content(
        self,
        source_doc: Dict[str, Any],
        target_doc_id: str,
        result: MergeResult,
    ):
        """Merge content of aligned documents"""
        source_chunks = self.storage.get_chunks_by_document(source_doc["id"])
        target_chunks = self.storage.get_chunks_by_document(target_doc_id)
        
        # Find unique chunks from source
        target_hashes = {c.get("content_hash") for c in target_chunks}
        
        for chunk in source_chunks:
            if chunk.get("content_hash") not in target_hashes:
                self.storage.add_chunk(
                    document_id=target_doc_id,
                    kernel_id=chunk.get("kernel_id"),
                    chunk_index=len(target_chunks),
                    text_content=chunk.get("text_content", ""),
                    content_hash=chunk.get("content_hash", ""),
                    qdrant_point_id=chunk.get("qdrant_point_id"),
                    metadata={**chunk.get("metadata", {}), "merged_from": source_doc["id"]},
                )
                result.merged_chunks += 1
        
        result.merged_documents += 1
    
    def _merge_section(
        self,
        source_kernel_id: str,
        target_kernel_id: str,
        section: MergeSection,
        default_resolution: ConflictResolution,
    ) -> Dict[str, Any]:
        """Merge a specific section of a kernel"""
        result = {
            "section_type": section.section_type,
            "documents": 0,
            "chunks": 0,
            "entities": 0,
            "relationships": 0,
            "conflicts": [],
        }
        
        if section.section_type == "documents":
            # Get and merge documents
            source_docs = self.storage.get_documents_by_kernel(source_kernel_id)
            
            for doc in source_docs:
                # Apply filter criteria
                if self._matches_filter(doc, section.filter_criteria):
                    doc_id = self.storage.add_document(
                        kernel_id=target_kernel_id,
                        title=doc.get("title", "Untitled"),
                        content_hash=doc.get("content_hash", ""),
                        blob_path=doc.get("blob_path"),
                        metadata=doc.get("metadata", {}),
                        source_url=doc.get("source_url"),
                        author=doc.get("author"),
                        published_date=doc.get("published_date"),
                    )
                    result["documents"] += 1
        
        elif section.section_type == "chunks":
            # Get and merge chunks
            source_chunks = self.storage.get_chunks_by_kernel(source_kernel_id)
            
            for chunk in source_chunks:
                if self._matches_filter(chunk, section.filter_criteria):
                    self.storage.add_chunk(
                        document_id=chunk.get("document_id"),
                        kernel_id=target_kernel_id,
                        chunk_index=chunk.get("chunk_index", 0),
                        text_content=chunk.get("text_content", ""),
                        content_hash=chunk.get("content_hash", ""),
                        qdrant_point_id=chunk.get("qdrant_point_id"),
                        metadata=chunk.get("metadata", {}),
                    )
                    result["chunks"] += 1
        
        return result
    
    def _matches_filter(
        self,
        item: Dict[str, Any],
        filter_criteria: Dict[str, Any],
    ) -> bool:
        """Check if item matches filter criteria"""
        for key, value in filter_criteria.items():
            item_value = item.get(key)
            
            if isinstance(value, list):
                if item_value not in value:
                    return False
            elif item_value != value:
                return False
        
        return True
    
    def get_merge_statistics(self) -> Dict[str, Any]:
        """Get merge engine statistics"""
        return {
            **self._stats,
            "success_rate": (
                self._stats["successful_merges"] / self._stats["total_merges"]
                if self._stats["total_merges"] > 0 else 0.0
            ),
            "auto_resolution_rate": (
                self._stats["auto_resolved"] / self._stats["conflicts_detected"]
                if self._stats["conflicts_detected"] > 0 else 0.0
            ),
        }
    
    def request_human_review(
        self,
        conflicts: List[MergeConflict],
        reviewer_id: str,
    ) -> List[MergeConflict]:
        """
        Request human review for unresolved conflicts
        
        Args:
            conflicts: List of conflicts requiring review
            reviewer_id: ID of assigned reviewer
            
        Returns:
            Updated conflicts with reviewer notes
        """
        for conflict in conflicts:
            conflict.suggested_resolution = ConflictResolution.RESOLVE_MANUAL
            conflict.resolution_notes = f"Pending review by {reviewer_id}"
        
        return conflicts
