"""
KERNELIZE Platform - Legal Domain Processor
============================================

This module implements the legal-specific knowledge compression
processor for the KERNELIZE Platform. It preserves legal citations,
contract terminology, and maintains clause integrity.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .base import BaseDomainProcessor, DomainContext, DomainSchema


# ==================== Legal Schema ====================

class LegalMetadata(DomainSchema):
    """Validation schema for legal domain metadata"""
    jurisdiction: Optional[str] = Field(
        None,
        description="Legal jurisdiction (e.g., 'California', 'Delaware', 'Federal')",
    )
    case_number: Optional[str] = Field(None, description="Case or docket number")
    court_name: Optional[str] = Field(None, description="Name of court")
    document_type: Optional[str] = Field(
        None,
        description="Type of legal document",
        pattern="^(brief|motion|opinion|contract|agreement|pleading|memorandum)$",
    )
    filing_date: Optional[datetime] = Field(None, description="Document filing date")
    client_matter: Optional[str] = Field(None, description="Client matter number")
    confidentiality: Optional[str] = Field(
        default="public",
        description="Confidentiality level",
        pattern="^(public|confidential|privileged|sealed)$",
    )


class LegalProcessor(BaseDomainProcessor):
    """
    Legal Domain Knowledge Processor
    
    specialized compression for legal documents, contracts,
    and court filings. Key features:
    - Preservation of legal citations and references
    - Protection of defined terms and definitions
    - Clause-level integrity for contracts
    - Citation format standardization
    """
    
    DOMAIN_NAME = "legal"
    DISPLAY_NAME = "Legal"
    DESCRIPTION = "Specialized compression for legal documents, contracts, and court filings"
    
    def _configure_patterns(self) -> None:
        """Configure legal-specific preservation patterns"""
        # Legal citations
        self.preserve_patterns.extend([
            r'\b\d+\s+U\.S\.\s+\d+\b',  # Supreme Court citations
            r'\b\d+\s+F\.\d+[d]?\s+\d+\b',  # Federal Reporter citations
            r'\b\d+\s+F\.Supp\.\d+[d]?\s+\d+\b',  # Federal Supplement citations
            r'\b[A-Z][a-z]+\s+\d{4}\b',  # Year-only citations (Smith 2024)
            r'\bId\.?(?:\s+at\s+\d+)?\b',  # Id. citations
            r'\b(?:See|Decision|Order|Rule)\s+No\.?\s*\d+[-]?\d*\b',  # Court orders
            r'\b§\s*\d+(?:\.\d+)*\b',  # Section symbols
            r'\b(?:Article|Section|Clause)\s+\d+(?:\.\d+)*\b',  # Article/Section references
        ])
        
        # Contract terminology
        self.preserve_patterns.extend([
            r'\b(?:Party|Party|Agreement|Contract|Term|Effective Date)\b',  # Defined terms
            r'\b(?:hereto|thereof|herein|therein|wherein|whereto)\b',  # Legal connectives
            r'\b(?:indemnify|hold harmless|terminat|warrant|represent|covenant)\b',  # Key obligations
            r'\b(?:confidentiality|non-compete|non-solicitation|intellectual property)\b',  # Key clauses
        ])
        
        # Critical legal terminology
        self.critical_terms.extend([
            "shall", "may", "must", "will", "agrees", "warrants",  # Obligation language
            "herein", "thereof", "hereto", "whereas",  # Reference terms
            "force majeure", "severability", "entire agreement", "governing law",  # Contract boilerplate
            "objection", "motion", "dismiss", "summary judgment",  # Litigation terms
            "attorney-client privilege", "work product", "confidential",  # Privileged terms
        ])
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate legal-specific metadata.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            LegalMetadata(**metadata)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_domain_schema(self) -> type:
        """Get the Pydantic schema for legal validation"""
        return LegalMetadata
    
    def preprocess(
        self,
        content: str,
        context: Optional[DomainContext] = None,
    ) -> str:
        """
        Preprocess legal content with citation protection.
        
        Args:
            content: Raw legal document
            context: Domain context
            
        Returns:
            Preprocessed content
        """
        if context is None:
            context = self.create_context()
        
        # Standard preprocessing
        protected = super().preprocess(content, context)
        
        # Normalize section symbols
        protected = re.sub(r'§+', '§', protected)
        
        # Ensure proper spacing around citations
        protected = re.sub(r'(\d+)\s+U\.S\.', r'\1 U.S.', protected)
        
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
        Score sentence importance with legal-specific adjustments.
        
        Legal documents have specific structure:
        - Definitions sections should be largely preserved
        - Operative clauses are high priority
        - Boilerplate can be compressed more aggressively
        """
        if context is None:
            context = self.create_context()
        
        # Base score
        score = super().score_sentence_importance(
            sentence, entities, position, total, context
        )
        
        sentence_lower = sentence.lower()
        
        # Boost definitions (typically near beginning of contracts)
        if any(term in sentence_lower for term in [
            "means", "defined as", "refers to", "as used herein"
        ]):
            score += 3.0  # High priority for definitions
        
        # Boost operative clauses (what the parties must do)
        if any(term in sentence_lower for term in [
            "shall", "agrees to", "warrants that", "represents and warrants",
            "covenants and agrees", "obligated"
        ]):
            score += 2.5
        
        # Boost obligations and liabilities
        if any(term in sentence_lower for term in [
            "indemnify", "hold harmless", "liability", "damages",
            "breach", "default", "termination", "remedy"
        ]):
            score += 2.0
        
        # Boost governing law and jurisdiction
        if any(term in sentence_lower for term in [
            "governing law", "jurisdiction", "venue", "arbitration",
            "applicable law", "laws of"
        ]):
            score += 1.5
        
        # Reduce score for boilerplate (can be compressed)
        boilerplate_terms = [
            "entire agreement", "severability", "amendment",
            "waiver", "notices", "headings"
        ]
        if any(term in sentence_lower for term in boilerplate_terms):
            if position > total * 0.3:
                score -= 0.5
        
        return score
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get legal processor information.
        
        Returns:
            Dictionary with processor details
        """
        info = super().get_processor_info()
        
        info.update({
            "supported_citations": [
                "U.S. Supreme Court",
                "Federal Reporter",
                "Federal Supplement",
                "State Reports",
            ],
            "document_types": [
                "Contracts",
                "Briefs",
                "Motions",
                "Opinions",
                "Memoranda",
            ],
            "preserved_elements": [
                "Legal Citations",
                "Defined Terms",
                "Obligation Language",
                "Jurisdiction Clauses",
            ],
        })
        
        return info
