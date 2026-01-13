"""
KERNELIZE Platform - Technology Domain Processor
=================================================

This module implements the technology-specific knowledge compression
processor for the KERNELIZE Platform. It preserves code snippets,
API documentation, and technical specifications.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base import BaseDomainProcessor, DomainContext, DomainSchema


# ==================== Technology Schema ====================

class TechnologyMetadata(DomainSchema):
    """Validation schema for technology domain metadata"""
    language: Optional[str] = Field(
        None,
        description="Programming or markup language",
        pattern="^[a-zA-Z][a-zA-Z0-9_-]*$",
    )
    framework: Optional[str] = Field(None, description="Framework or library name")
    version: Optional[str] = Field(None, description="Version string")
    repository_url: Optional[str] = Field(None, description="Code repository URL")
    api_version: Optional[str] = Field(None, description="API version")
    environment: Optional[str] = Field(
        default="production",
        description="Deployment environment",
        pattern="^(development|staging|production)$",
    )


class TechnologyProcessor(BaseDomainProcessor):
    """
    Technology Domain Knowledge Processor
    
    Specialized compression for technical documentation, code
    comments, and API specifications. Key features:
    - Protection of code blocks and inline code
    - Preservation of API endpoints and parameters
    - Documentation comment extraction
    - Version and configuration preservation
    """
    
    DOMAIN_NAME = "technology"
    DISPLAY_NAME = "Technology"
    DESCRIPTION = "Specialized compression for code documentation and technical specifications"
    
    def _configure_patterns(self) -> None:
        """Configure technology-specific preservation patterns"""
        # Code blocks (markdown)
        self.preserve_patterns.extend([
            r'```[a-zA-Z0-9]*\n[\s\S]*?```',  # Multi-line code blocks
            r'`[^`]+`',  # Inline code
            r'^    [^\n]+$',  # Indented code
        ])
        
        # API specifications
        self.preserve_patterns.extend([
            r'\b(?:GET|POST|PUT|DELETE|PATCH|OPTIONS)\s+[/a-zA-Z0-9_-]+\b',  # HTTP methods
            r'\b/api/v\d+/[-a-zA-Z0-9_/{}]+\b',  # API endpoints
            r'\{[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\}',  # JSON objects
            r'\[[a-zA-Z_][a-zA-Z0-9_]*(?:\s*,\s*[a-zA-Z_][a-zA-Z0-9_]*)*\]',  # JSON arrays
            r'\b(?:200|201|400|401|403|404|500)\s*[a-zA-Z\s]*\b',  # Status codes
        ])
        
        # Configuration and imports
        self.preserve_patterns.extend([
            r'\b(?:import|from|require|using|include)\s+[a-zA-Z0-9_.]+\b',  # Imports
            r'\b(?:const|let|var|def|function|class|interface)\s+[a-zA-Z_][a-zA-Z0-9_]*\b',  # Declarations
            r'\b[A-Z_][A-Z0-9_]*\s*=',  # Constants
            r'\b(?:localhost|127\.0\.0\.1|https?://[^\s]+)\b',  # URLs/hosts
        ])
        
        # Critical technical terminology
        self.critical_terms.extend([
            "function", "method", "class", "interface", "module", "package",  # Code structure
            "api", "endpoint", "request", "response", "header", "parameter",  # API terms
            "database", "query", "index", "schema", "table", "column",  # Data terms
            "error", "exception", "debug", "log", "trace", "monitor",  # Operations
            "async", "await", "promise", "callback", "event", "listener",  # Async terms
            "authentication", "authorization", "token", "session", "cookie",  # Security
        ])
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate technology-specific metadata.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            TechnologyMetadata(**metadata)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_domain_schema(self) -> type:
        """Get the Pydantic schema for technology validation"""
        return TechnologyMetadata
    
    def preprocess(
        self,
        content: str,
        context: Optional[DomainContext] = None,
    ) -> str:
        """
        Preprocess technical content with code protection.
        
        Args:
            content: Raw technical documentation
            context: Domain context
            
        Returns:
            Preprocessed content
        """
        if context is None:
            context = self.create_context()
        
        # Standard preprocessing
        protected = super().preprocess(content, context)
        
        # Additional technical preprocessing
        # Normalize whitespace in code blocks
        protected = re.sub(r'```[a-zA-Z0-9]*\n+', '```\n', protected)
        
        return protected
    
    def score_sentence_importance(
        self,
        sentence: str,
        entities: List[str],
        position: int,
        total: int,
        context: Optional[DomainContext] = None,
    ) -> float:
        """
        Score sentence importance with technology-specific adjustments.
        
        Technical docs have specific priorities:
        - API documentation is high priority
        - Code comments within blocks are important
        - Setup/configuration steps are important
        """
        if context is None:
            context = self.create_context()
        
        # Base score
        score = super().score_sentence_importance(
            sentence, entities, position, total, context
        )
        
        sentence_lower = sentence.lower()
        
        # Boost API documentation
        if any(term in sentence_lower for term in [
            "endpoint", "parameter", "request body", "response schema",
            "authentication required", "rate limit", "example"
        ]):
            score += 2.5
        
        # Boost setup and configuration
        if any(term in sentence_lower for term in [
            "install", "configure", "setup", "prerequisite", "dependency",
            "environment variable", "config", "init", "start"
        ]):
            score += 2.0
        
        # Boost error handling documentation
        if any(term in sentence_lower for term in [
            "error", "exception", "throw", "catch", "fallback",
            "handle", "treat", "response code"
        ]):
            score += 1.5
        
        # Boost key architectural concepts
        if any(term in sentence_lower for term in [
            "architecture", "design pattern", "component", "module",
            "service", "microservice", "integration", "interface"
        ]):
            score += 1.5
        
        # Reduce score for changelog-type content (can be summarized)
        if any(term in sentence_lower for term in [
            "changelog", "version history", "release note", "bug fix"
        ]):
            score -= 0.5
        
        return score
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get technology processor information.
        
        Returns:
            Dictionary with processor details
        """
        info = super().get_processor_info()
        
        info.update({
            "supported_languages": [
                "Python",
                "JavaScript",
                "TypeScript",
                "Java",
                "Go",
                "Rust",
                "C/C++",
                "Ruby",
                "PHP",
                "SQL",
            ],
            "document_types": [
                "API Documentation",
                "Code Comments",
                "Technical Manuals",
                "README Files",
                "Architecture Docs",
            ],
            "preserved_elements": [
                "Code Blocks",
                "API Endpoints",
                "Configuration",
                "Import Statements",
            ],
        })
        
        return info
