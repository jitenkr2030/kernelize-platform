"""
KERNELIZE Platform - Education Domain Processor
================================================

This module implements the education-specific knowledge compression
processor for the KERNELIZE Platform. It preserves learning objectives,
curriculum structures, and assessment content.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator

from .base import BaseDomainProcessor, DomainContext, DomainSchema


# ==================== Education Schema ====================

class EducationMetadata(DomainSchema):
    """Validation schema for education domain metadata"""
    grade_level: Optional[str] = Field(
        None,
        description="Target grade level (e.g., 'K-12', 'Undergraduate', 'Graduate')",
        pattern="^(K-|Grade\s*\d+|Pre-K|Undergraduate|Graduate|Professional)$",
    )
    subject: Optional[str] = Field(
        None,
        description="Academic subject area",
    )
    standard_alignment: Optional[str] = Field(
        None,
        description="Educational standard alignment (e.g., 'Common Core', 'NGSS')",
    )
    Bloom_level: Optional[str] = Field(
        default="apply",
        description="Target Bloom's Taxonomy level",
        pattern="^(remember|understand|apply|analyze|evaluate|create)$",
    )
    course_code: Optional[str] = Field(None, description="Course identifier")
    semester: Optional[str] = Field(
        None,
        description="Academic term",
        pattern="^(Fall|Spring|Summer|Winter)\s*\d{4}$",
    )


class EducationProcessor(BaseDomainProcessor):
    """
    Education Domain Knowledge Processor
    
    Specialized compression for educational content, curriculum,
    and learning materials. Key features:
    - Preservation of learning objectives (Bloom's taxonomy)
    - Hierarchical structure for curriculum
    - Assessment question protection
    - Standard alignment preservation
    """
    
    DOMAIN_NAME = "education"
    DISPLAY_NAME = "Education"
    DESCRIPTION = "Specialized compression for curriculum and learning materials"
    
    def _configure_patterns(self) -> None:
        """Configure education-specific preservation patterns"""
        # Learning objectives
        self.preserve_patterns.extend([
            r'\b(?:students will be able to|swbat|learners will|objective:)\s*.+',  # Objective statements
            r'\b(?:by the end of|upon completion|after this)\s*.+',  # Outcome statements
            r'\b(?:Bloom|Bloom\'s)\s*(?:Taxonomy)?\s*(?:level)?\s*[:\s]*(remember|understand|apply|analyze|evaluate|create)\b',  # Bloom's levels
        ])
        
        # Curriculum structure
        self.preserve_patterns.extend([
            r'\b(?:Module|Unit|Lesson|Chapter|Section|Topic|Subtopic)\s*\d*[:\s]*.+',  # Content organization
            r'\b(?:Learning Objective|Learning Goal|Competency|Skill)\s*[:\s]*.+',  # Objectives
            r'\b(?:Key Concept|Main Idea|Essential Question|Guiding Question)\s*[:\s]*.+',  # Key concepts
        ])
        
        # Assessment content
        self.preserve_patterns.extend([
            r'\b(?:Question|Problem|Exercise|Task|Prompt)\s*\d*[:\s]*.+',  # Assessment items
            r'\b(?:Answer|Key|Solution|Rubric|Criteria)\s*[:\s]*.+',  # Assessment keys
            r'\b(?:True|False|Multiple Choice|Short Answer|Essay)\s*[-:]?\s*.+',  # Question types
        ])
        
        # Critical educational terminology
        self.critical_terms.extend([
            "objective", "goal", "outcome", "competency", "skill",  # Outcome terms
            "understand", "analyze", "evaluate", "create",  # Bloom's verbs
            "curriculum", "syllabus", "lesson plan", " pacing guide",  # Planning terms
            "assessment", "evaluation", "rubric", "criteria",  # Assessment terms
            "standard", "benchmark", "framework", "guideline",  # Standards terms
        ])
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate education-specific metadata.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            EducationMetadata(**metadata)
            return True, None
        except Exception as e:
            return False, str(e)
    
    def get_domain_schema(self) -> type:
        """Get the Pydantic schema for education validation"""
        return EducationMetadata
    
    def score_sentence_importance(
        self,
        sentence: str,
        entities: List[str],
        position: int,
        total: int,
        context: Optional[DomainContext] = None,
    ) -> float:
        """
        Score sentence importance with education-specific adjustments.
        
        Educational content has hierarchical importance:
        - Learning objectives are highest priority
        - Key concepts and essential questions are high priority
        - Detailed examples can be compressed
        - Assessment items should be preserved
        """
        if context is None:
            context = self.create_context()
        
        # Base score
        score = super().score_sentence_importance(
            sentence, entities, position, total, context
        )
        
        sentence_lower = sentence.lower()
        
        # Boost learning objectives (highest priority)
        if any(term in sentence_lower for term in [
            "objective", "goal", "outcome", "competency", "swbat",
            "students will", "learners will", "by the end of"
        ]):
            score += 3.0
        
        # Boost essential questions and key concepts
        if any(term in sentence_lower for term in [
            "essential question", "guiding question", "key concept",
            "main idea", "central question", "big idea"
        ]):
            score += 2.5
        
        # Boost assessment items
        if any(term in sentence_lower for term in [
            "question", "problem", "exercise", "task", "prompt",
            "what is", "how would", "explain why", "analyze"
        ]):
            score += 2.0
        
        # Boost standards alignment
        if any(term in sentence_lower for term in [
            "standard", "benchmark", "aligned to", "meets standard",
            "common core", "ngss", "state standard"
        ]):
            score += 1.5
        
        # Reduce score for extended examples and elaborations
        if position > total * 0.6:
            if any(term in sentence_lower for term in [
                "for example", "for instance", "such as", "including"
            ]):
                score -= 0.5
        
        return score
    
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get education processor information.
        
        Returns:
            Dictionary with processor details
        """
        info = super().get_processor_info()
        
        info.update({
            "supported_levels": [
                "K-12",
                "Undergraduate",
                "Graduate",
                "Professional",
            ],
            "content_types": [
                "Learning Objectives",
                "Curriculum",
                "Lesson Plans",
                "Assessments",
                "Standards Alignment",
            ],
            "bloom_levels": [
                "Remember",
                "Understand",
                "Apply",
                "Analyze",
                "Evaluate",
                "Create",
            ],
        })
        
        return info
