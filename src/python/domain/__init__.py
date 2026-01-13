"""
KERNELIZE Platform - Domain Registry
=====================================

This module provides the domain registry and factory for instantiating
domain-specific knowledge compression processors.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

from typing import Any, Dict, Optional, Type, Union

from .base import BaseDomainProcessor, DomainContext, DomainSchema
from .healthcare import HealthcareProcessor
from .finance import FinanceProcessor
from .legal import LegalProcessor
from .technology import TechnologyProcessor
from .education import EducationProcessor


# Domain registry mapping
_DOMAIN_REGISTRY: Dict[str, Type[BaseDomainProcessor]] = {
    "healthcare": HealthcareProcessor,
    "finance": FinanceProcessor,
    "legal": LegalProcessor,
    "technology": TechnologyProcessor,
    "education": EducationProcessor,
}


# Display name mapping
_DOMAIN_DISPLAY_NAMES: Dict[str, str] = {
    "healthcare": "Healthcare",
    "finance": "Finance",
    "legal": "Legal",
    "technology": "Technology",
    "education": "Education",
    "general": "General",
}


class DomainRegistry:
    """
    Domain Registry and Factory
    
    Provides centralized access to domain-specific processors
    with support for registration, lookup, and instantiation.
    """
    
    _processors: Dict[str, BaseDomainProcessor] = {}
    _default_compression_level: int = 5
    
    @classmethod
    def register(
        cls,
        domain_name: str,
        processor_class: Type[BaseDomainProcessor],
    ) -> None:
        """
        Register a new domain processor.
        
        Args:
            domain_name: Unique domain identifier
            processor_class: Processor class to register
        """
        if not issubclass(processor_class, BaseDomainProcessor):
            raise ValueError(
                f"Processor must be a subclass of BaseDomainProcessor, "
                f"got {processor_class}"
            )
        
        _DOMAIN_REGISTRY[domain_name] = processor_class
        
        logger.info(f"Registered domain processor: {domain_name}")
    
    @classmethod
    def get_processor(
        cls,
        domain_name: str,
        compression_level: Optional[int] = None,
    ) -> BaseDomainProcessor:
        """
        Get a domain processor instance.
        
        Args:
            domain_name: Domain identifier
            compression_level: Optional compression level override
            
        Returns:
            Domain processor instance
            
        Raises:
            ValueError: If domain is not registered
        """
        domain_name = domain_name.lower()
        
        # Check cache
        cache_key = f"{domain_name}_{compression_level}"
        if cache_key in cls._processors:
            return cls._processors[cache_key]
        
        # Get processor class
        processor_class = _DOMAIN_REGISTRY.get(domain_name)
        
        if processor_class is None:
            raise ValueError(
                f"Unknown domain: {domain_name}. "
                f"Available domains: {list(_DOMAIN_REGISTRY.keys())}"
            )
        
        # Create instance
        level = compression_level or cls._default_compression_level
        processor = processor_class(default_compression_level=level)
        
        # Cache instance
        cls._processors[cache_key] = processor
        
        return processor
    
    @classmethod
    def get_domain_names(cls) -> list:
        """Get list of registered domain names"""
        return list(_DOMAIN_REGISTRY.keys())
    
    @classmethod
    def get_domain_display_names(cls) -> Dict[str, str]:
        """Get mapping of domain names to display names"""
        return _DOMAIN_DISPLAY_NAMES.copy()
    
    @classmethod
    def validate_domain(cls, domain_name: str) -> bool:
        """Check if a domain is registered"""
        return domain_name.lower() in _DOMAIN_REGISTRY
    
    @classmethod
    def get_schema(cls, domain_name: str) -> Optional[type]:
        """
        Get the validation schema for a domain.
        
        Args:
            domain_name: Domain identifier
            
        Returns:
            Pydantic schema class or None for unknown domain
        """
        try:
            processor = cls.get_processor(domain_name)
            return processor.get_domain_schema()
        except ValueError:
            return None
    
    @classmethod
    def get_processor_info(cls, domain_name: str) -> Optional[Dict[str, Any]]:
        """
        Get processor information for a domain.
        
        Args:
            domain_name: Domain identifier
            
        Returns:
            Processor info dictionary or None
        """
        try:
            processor = cls.get_processor(domain_name)
            return processor.get_processor_info()
        except ValueError:
            return None
    
    @classmethod
    def create_context(
        cls,
        domain_name: str,
        compression_level: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[DomainContext]:
        """
        Create a domain context for compression.
        
        Args:
            domain_name: Domain identifier
            compression_level: Compression level
            metadata: Domain-specific metadata
            
        Returns:
            DomainContext or None for unknown domain
        """
        try:
            processor = cls.get_processor(domain_name, compression_level)
            return processor.create_context(compression_level, metadata)
        except ValueError:
            return None
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached processor instances"""
        cls._processors.clear()
        logger.info("Domain processor cache cleared")
    
    @classmethod
    def get_all_processor_info(cls) -> Dict[str, Dict[str, Any]]:
        """
        Get information for all registered processors.
        
        Returns:
            Dictionary mapping domain names to processor info
        """
        info = {}
        for domain_name in _DOMAIN_REGISTRY.keys():
            proc_info = cls.get_processor_info(domain_name)
            if proc_info:
                info[domain_name] = proc_info
        return info


# Convenience functions
def get_processor(
    domain: str,
    compression_level: Optional[int] = None,
) -> BaseDomainProcessor:
    """Get a domain processor by name"""
    return DomainRegistry.get_processor(domain, compression_level)


def get_domain_names() -> list:
    """Get list of available domain names"""
    return DomainRegistry.get_domain_names()


def validate_domain(domain: str) -> bool:
    """Check if a domain is valid"""
    return DomainRegistry.validate_domain(domain)


# Import logger
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DomainResult:
    """Result dataclass for domain-specific processing"""
    kernel_id: str
    domain: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    compressed_content: str
    domain_entities: List[Dict[str, Any]]
    domain_metadata: Optional[Dict[str, Any]]
    embedding: Optional[List[float]]
    processing_time_ms: int


class DomainProcessor:
    """
    Unified Domain Processor
    
    Provides a simple interface for domain-specific knowledge compression
    across all supported domains.
    """
    
    @staticmethod
    def process(
        content: str,
        domain: str,
        compression_level: int = 5,
        language: str = "en",
        extract_entities: bool = True,
        generate_embedding: bool = True,
        domain_options: Optional[Dict[str, Any]] = None,
    ) -> DomainResult:
        """
        Process content for a specific domain.
        
        Args:
            content: Knowledge content to process
            domain: Target domain (healthcare, finance, legal, technology, education)
            compression_level: Compression level (1-10)
            language: Language code
            extract_entities: Whether to extract domain entities
            generate_embedding: Whether to generate embedding
            domain_options: Domain-specific options
            
        Returns:
            DomainResult with processed content and metadata
        """
        import time
        import hashlib
        
        start_time = time.time()
        
        # Get the appropriate processor
        processor = DomainRegistry.get_processor(domain, compression_level)
        
        # Create context and process
        context = processor.create_context(compression_level, domain_options or {})
        
        # Process the content
        result = processor.process(content, context)
        
        # Generate kernel ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        kernel_id = f"kz_{domain}_{content_hash}_{int(start_time)}"
        
        # Calculate compression ratio
        original_size = len(content.encode())
        compressed_size = len(result.compressed_content.encode())
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        return DomainResult(
            kernel_id=kernel_id,
            domain=domain,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            compressed_content=result.compressed_content,
            domain_entities=result.entities,
            domain_metadata=result.metadata,
            embedding=None,  # Would be generated by embedding service
            processing_time_ms=processing_time_ms,
        )


# Create convenience instance
domain_processor = DomainProcessor()
