"""
KERNELIZE Platform - Base Domain Processor
===========================================

This module defines the abstract base class for all domain-specific
knowledge compression processors. It provides a common interface
and shared functionality for domain-aware semantic compression.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


@dataclass
class DomainContext:
    """
    Domain-specific compression context.
    
    Contains metadata and processing information specific to a
    particular domain during compression operations.
    """
    domain: str
    preserve_patterns: List[str] = field(default_factory=list)
    critical_terms: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    compression_level: int = 5
    
    def add_preserve_pattern(self, pattern: str) -> None:
        """Add a regex pattern to preserve during compression"""
        self.preserve_patterns.append(pattern)
    
    def add_critical_term(self, term: str) -> None:
        """Add a critical term that must be preserved"""
        self.critical_terms.append(term)


class DomainSchema(BaseModel):
    """Base validation schema for domain-specific metadata"""
    domain: str = Field(..., description="Domain identifier")


class BaseDomainProcessor(ABC):
    """
    Abstract base class for domain-specific knowledge compression.
    
    All domain processors inherit from this class and implement
    the required methods for domain-aware semantic compression.
    """
    
    DOMAIN_NAME: str = "base"
    DISPLAY_NAME: str = "Base Domain"
    DESCRIPTION: str = "Base domain processor"
    
    def __init__(self, default_compression_level: int = 5):
        """
        Initialize the domain processor.
        
        Args:
            default_compression_level: Default compression level for this domain
        """
        self.default_compression_level = default_compression_level
        self._setup_domain_patterns()
        
    def _setup_domain_patterns(self) -> None:
        """Setup domain-specific regex patterns for preservation"""
        self.preserve_patterns: List[str] = []
        self.critical_terms: List[str] = []
        self._configure_patterns()
    
    @abstractmethod
    def _configure_patterns(self) -> None:
        """
        Configure domain-specific patterns and terms.
        
        This method should be implemented by each domain processor
        to define its specific preservation requirements.
        """
        pass
    
    def create_context(
        self,
        compression_level: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DomainContext:
        """
        Create a domain context for compression operations.
        
        Args:
            compression_level: Override default compression level
            metadata: Additional domain-specific metadata
            
        Returns:
            DomainContext configured for this domain
        """
        return DomainContext(
            domain=self.DOMAIN_NAME,
            preserve_patterns=self.preserve_patterns.copy(),
            critical_terms=self.critical_terms.copy(),
            metadata=metadata or {},
            compression_level=compression_level or self.default_compression_level,
        )
    
    def preprocess(
        self,
        content: str,
        context: Optional[DomainContext] = None,
    ) -> str:
        """
        Preprocess content before semantic compression.
        
        Args:
            content: Raw input content
            context: Domain context with patterns and terms
            
        Returns:
            Preprocessed content ready for compression
        """
        if context is None:
            context = self.create_context()
        
        # Protect preserved patterns
        protected = self._protect_patterns(content, context)
        
        return protected
    
    def postprocess(
        self,
        compressed: str,
        original: str,
        context: Optional[DomainContext] = None,
    ) -> str:
        """
        Postprocess compressed content after semantic compression.
        
        Args:
            compressed: Compressed content
            original: Original content for reference
            context: Domain context
            
        Returns:
            Postprocessed content
        """
        if context is None:
            context = self.create_context()
        
        # Restore protected patterns
        restored = self._restore_patterns(compressed, context)
        
        return restored
    
    def _protect_patterns(
        self,
        content: str,
        context: DomainContext,
    ) -> str:
        """
        Protect patterns from being modified during compression.
        
        Args:
            content: Input content
            context: Domain context with patterns
            
        Returns:
            Content with patterns replaced by placeholders
        """
        protected = content
        placeholders = []
        
        for i, pattern in enumerate(context.preserve_patterns):
            try:
                matches = list(re.finditer(pattern, protected))
                for match in reversed(matches):
                    placeholder = f"__PRESERVED_{i}_{match.start()}_{match.end()}__"
                    protected = (
                        protected[:match.start()] +
                        placeholder +
                        protected[match.end():]
                    )
                    placeholders.append((placeholder, match.group()))
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern}, error: {e}")
        
        context.metadata["_placeholders"] = placeholders
        return protected
    
    def _restore_patterns(
        self,
        content: str,
        context: DomainContext,
    ) -> str:
        """
        Restore protected patterns after compression.
        
        Args:
            content: Content with placeholders
            context: Domain context with stored patterns
            
        Returns:
            Content with original patterns restored
        """
        restored = content
        placeholders = context.metadata.get("_placeholders", [])
        
        for placeholder, original in placeholders:
            restored = restored.replace(placeholder, original)
        
        return restored
    
    @abstractmethod
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate domain-specific metadata.
        
        Args:
            metadata: Metadata to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def get_domain_schema(self) -> type:
        """
        Get the Pydantic schema for domain validation.
        
        Returns:
            Pydantic model class for validation
        """
        pass
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get processor information and capabilities.
        
        Returns:
            Dictionary with processor details
        """
        return {
            "domain": self.DOMAIN_NAME,
            "display_name": self.DISPLAY_NAME,
            "description": self.DESCRIPTION,
            "compression_levels": list(range(1, 11)),
            "preserve_patterns_count": len(self.preserve_patterns),
            "critical_terms_count": len(self.critical_terms),
        }
    
    def score_sentence_importance(
        self,
        sentence: str,
        entities: List[str],
        position: int,
        total: int,
        context: Optional[DomainContext] = None,
    ) -> float:
        """
        Score sentence importance with domain-specific adjustments.
        
        Args:
            sentence: Sentence text
            entities: Extracted entities
            position: Sentence position in document
            total: Total number of sentences
            
        Returns:
            Importance score (higher = more important)
        """
        if context is None:
            context = self.create_context()
        
        # Base score calculation
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Entity scoring
        for entity in entities:
            if entity.lower() in sentence_lower:
                score += 2.0
        
        # Position scoring
        if position == 0:
            score += 3.0
        elif position == total - 1:
            score += 2.0
        elif position < total * 0.2:
            score += 1.0
        
        # Domain-specific critical terms
        for term in context.critical_terms:
            if term.lower() in sentence_lower:
                score += 3.0
        
        # Length scoring
        word_count = len(sentence.split())
        if 5 <= word_count <= 30:
            score += 1.0
        elif word_count > 50:
            score -= 0.5
        
        # Numerical data scoring (important for many domains)
        if re.search(r'\d+', sentence):
            score += 1.0
        
        return score


import logging

logger = logging.getLogger(__name__)
