"""
KERNELIZE Platform - Healthcare Domain Processor
=================================================

This module implements the healthcare-specific knowledge compression
processor for the KERNELIZE Platform. It preserves medical terminology,
ICD-10 codes, drug names, and ensures HIPAA compliance considerations.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from .base import BaseDomainProcessor, DomainContext, DomainSchema


# ==================== Healthcare Schema ====================

class HealthcareMetadata(DomainSchema):
    """Validation schema for healthcare domain metadata"""
    patient_id_hash: Optional[str] = Field(
        None,
        description="Hashed patient identifier for privacy",
        min_length=64,
        max_length=64,
    )
    encounter_date: Optional[datetime] = Field(
        None,
        description="Date of medical encounter",
    )
    provider_id: Optional[str] = Field(
        None,
        description="Healthcare provider identifier",
    )
    facility_id: Optional[str] = Field(
        None,
        description="Healthcare facility identifier",
    )
    encounter_type: Optional[str] = Field(
        None,
        description="Type of medical encounter",
        pattern="^(outpatient|inpatient|emergency|consultation|telehealth)$",
    )
    hipaa_compliant: bool = Field(
        default=True,
        description="Whether processing follows HIPAA guidelines",
    )


class HealthcareProcessor(BaseDomainProcessor):
    """
    Healthcare Domain Knowledge Processor
    
    Specialized compression for medical records, clinical notes,
    and healthcare documentation. Key features:
    - Preservation of ICD-10, CPT, and SNOMED codes
    - Protection of negation terms (critical for diagnosis)
    - Drug name preservation (RxNorm compatibility)
    - HIPAA compliance metadata handling
    """
    
    DOMAIN_NAME = "healthcare"
    DISPLAY_NAME = "Healthcare"
    DESCRIPTION = "Specialized compression for medical records and clinical documentation"
    
    def _configure_patterns(self) -> None:
        """Configure healthcare-specific preservation patterns"""
        # ICD-10 Diagnosis Codes (e.g., E11.9, J45.909)
        self.preserve_patterns.extend([
            r'\b[A-Z]\d{2}\.?\d?[A-Z0-9]{0,4}\b',  # ICD-10-CM
            r'\b[A-Z]\d{5}\b',  # ICD-10-PCS procedure codes
            r'\bCPT\s*\d{4,5}\b',  # CPT codes
            r'\bSNOMED[CT]?[:\s]*\d+\b',  # SNOMED codes
        ])
        
        # Drug names and prescriptions (RxNorm style)
        self.preserve_patterns.extend([
            r'\b(mg|mL|mcg|kg|lb)\b',  # Dosage units
            r'\b(qd|bid|tid|qid|prn|ac|pc|hs)\b',  # Frequency codes
            r'\bPO|IM|IV|SC|SL\b',  # Route codes
            r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\s*(?:HCl|Sulfate|Tablet|Capsule|Injection)\b',  # Drug forms
        ])
        
        # Critical medical terminology
        self.critical_terms.extend([
            "denies", "no", "not", "negative", "absence",  # Negation terms
            "pain", "acute", "chronic", "severe", "mild",  # Symptom descriptors
            "history of", "family history", "past medical",  # History markers
            "vital signs", "blood pressure", "heart rate", "temperature",  # Vitals
            "diagnosis", "impression", "assessment", "plan",  # Documentation sections
        ])
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate healthcare-specific metadata.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            HealthcareMetadata(**metadata)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_domain_schema(self) -> type:
        """Get the Pydantic schema for healthcare validation"""
        return HealthcareMetadata
    
    def preprocess(
        self,
        content: str,
        context: Optional[DomainContext] = None,
    ) -> str:
        """
        Preprocess healthcare content with HIPAA considerations.
        
        Args:
            content: Raw medical documentation
            context: Domain context
            
        Returns:
            Preprocessed content
        """
        if context is None:
            context = self.create_context()
        
        # Standard preprocessing
        protected = super().preprocess(content, context)
        
        # Additional healthcare-specific preprocessing
        # Normalize whitespace in medical abbreviations
        protected = re.sub(r'\s+', ' ', protected)
        
        # Ensure proper spacing around punctuation
        protected = re.sub(r'\s+([.,;:])', r'\1', protected)
        
        context.metadata["_placeholders"] = context.metadata.get("_placeholders", [])
        
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
        Score sentence importance with healthcare-specific adjustments.
        
        Healthcare notes follow SOAP format: Subjective, Objective, Assessment, Plan.
        Assessment and Plan sections are typically most critical.
        """
        if context is None:
            context = self.create_context()
        
        # Base score
        score = super().score_sentence_importance(
            sentence, entities, position, total, context
        )
        
        sentence_lower = sentence.lower()
        
        # Boost assessment and plan sections
        if any(term in sentence_lower for term in [
            "assessment:", "impression:", "diagnosis:", "plan:",
            "recommendation:", "prescribed", "discharge"
        ]):
            score += 2.0
        
        # Boost vital signs and measurements
        if any(term in sentence_lower for term in [
            "bp:", "blood pressure", "heart rate", "rr:", "o2 sat",
            "temperature:", "weight:", "height:"
        ]):
            score += 1.5
        
        # Boost medication-related content
        if any(term in sentence_lower for term in [
            "medication", "prescribed", "dosage", "mg", "mg/ml",
            "started on", "continued", "discontinued"
        ]):
            score += 1.5
        
        # Protect negation context heavily
        negation_patterns = [
            "denies", "no", "not", "negative", "without", "absence of",
            "denied", "none", "unremarkable"
        ]
        if any(term in sentence_lower for term in negation_patterns):
            score += 2.5  # High priority for negative findings
        
        return score
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get healthcare processor information.
        
        Returns:
            Dictionary with processor details
        """
        info = super().get_processor_info()
        
        info.update({
            "supported_code_systems": [
                "ICD-10-CM",
                "ICD-10-PCS",
                "CPT",
                "SNOMED-CT",
                "RxNorm",
            ],
            "compliance_standards": ["HIPAA"],
            "documentation_formats": ["SOAP", "Progress Notes", "Discharge Summary"],
        })
        
        return info
