"""
KERNELIZE Platform - Domain-Specific Compression Models
=========================================================

Specialized compression models trained for each target domain:
- Healthcare: Medical records, clinical notes, research papers
- Finance: Financial reports, market data, regulatory filings
- Legal: Contracts, court documents, legislation
- Scientific: Research papers, technical documentation
- Government: Policy documents, public records, regulations

Each model includes domain-specific training data configurations,
fine-tuning objectives, and expert review processes.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import json
import logging
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class DomainType(Enum):
    """Domain types for specialized compression"""
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    LEGAL = "legal"
    SCIENTIFIC = "scientific"
    GOVERNMENT = "government"
    GENERAL = "general"


@dataclass
class DomainConfig:
    """Configuration for domain-specific compression"""
    domain_type: str
    model_name: str
    
    # Compression parameters
    target_compression_ratio: float = 50.0
    quality_preservation_target: float = 0.85
    
    # Domain-specific vocabulary
    vocabulary_files: List[str] = field(default_factory=list)
    medical_term_file: Optional[str] = None
    legal_term_file: Optional[str] = None
    financial_term_file: Optional[str] = None
    scientific_term_file: Optional[str] = None
    
    # Entity preservation priorities
    priority_entities: List[str] = field(default_factory=list)
    
    # Quality thresholds
    min_entity_preservation: float = 0.9
    min_relation_preservation: float = 0.85
    min_fact_preservation: float = 0.8
    
    # Review requirements
    requires_expert_review: bool = False
    review_threshold_quality: float = 0.7


@dataclass
class DomainTrainingData:
    """Training data configuration for domain models"""
    domain_type: str
    source_files: List[str]
    total_documents: int
    average_document_length: int
    
    # Data quality metrics
    annotation_quality: float = 0.9
    expert_review_coverage: float = 0.8
    
    # Preprocessing
    cleaning_rules: List[str] = field(default_factory=list)
    anonymization_required: bool = True
    
    # Augmentation
    augmentation_enabled: bool = True
    augmentation_factors: List[float] = field(default_factory=lambda: [1.0, 1.5, 2.0])


@dataclass
class DomainBenchmarkResult:
    """Benchmark results for domain compression"""
    domain_type: str
    test_dataset_size: int
    average_compression_ratio: float
    quality_score: float
    entity_preservation: float
    relation_preservation: float
    fact_preservation: float
    processing_time_ms: float
    
    # Per-category results
    category_results: Dict[str, Dict[str, float]] = field(default_factory=dict)


@dataclass
class ExpertReview:
    """Expert review record for quality validation"""
    review_id: str
    domain_type: str
    document_id: str
    reviewer_id: str
    
    # Review metrics
    overall_quality: float
    factual_accuracy: float
    completeness: float
    coherence: float
    
    # Issues found
    critical_issues: List[str] = field(default_factory=list)
    minor_issues: List[str] = field(default_factory=list)
    
    # Recommendation
    approved: bool = False
    revision_required: bool = False
    revision_notes: str = ""
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class DomainCompressionModel(ABC):
    """
    Base class for domain-specific compression models.
    
    Provides common functionality for specialized compression
    including vocabulary management, entity recognition, and
    quality validation.
    """
    
    def __init__(self, config: DomainConfig):
        """
        Initialize domain compression model
        
        Args:
            config: Domain configuration
        """
        self.config = config
        self.domain_type = config.domain_type
        
        # Domain vocabulary
        self._domain_vocabulary: Dict[str, Dict[str, Any]] = {}
        self._entity_patterns: Dict[str, List[Tuple[str, str]]] = {}
        
        # Quality tracking
        self._quality_scores: List[float] = []
        self._review_history: List[ExpertReview] = []
        
        # Initialize vocabulary
        self._load_domain_vocabulary()
        self._compile_entity_patterns()
    
    @abstractmethod
    def _load_domain_vocabulary(self):
        """Load domain-specific vocabulary"""
        pass
    
    @abstractmethod
    def _compile_entity_patterns(self):
        """Compile entity recognition patterns"""
        pass
    
    @abstractmethod
    def compress(
        self,
        content: str,
        target_ratio: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compress domain-specific content
        
        Args:
            content: Content to compress
            target_ratio: Optional target compression ratio
            
        Returns:
            Compressed content and metrics
        """
        pass
    
    @abstractmethod
    def validate_quality(
        self,
        original: str,
        compressed: str
    ) -> Dict[str, float]:
        """
        Validate compression quality for domain
        
        Args:
            original: Original content
            compressed: Compressed content
            
        Returns:
            Quality metrics
        """
        pass
    
    def _preserve_entities(
        self,
        content: str,
        compressed: str
    ) -> Tuple[List[str], List[str]]:
        """Identify entities in original and compressed content"""
        original_entities = self._extract_domain_entities(content)
        compressed_entities = self._extract_domain_entities(compressed)
        
        return original_entities, compressed_entities
    
    @abstractmethod
    def _extract_domain_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract domain-specific entities"""
        pass
    
    def _calculate_entity_preservation(
        self,
        original: List[Dict[str, Any]],
        compressed: List[Dict[str, Any]]
    ) -> float:
        """Calculate entity preservation rate"""
        if not original:
            return 1.0
        
        original_texts = set(e['text'] for e in original)
        compressed_texts = set(e['text'] for e in compressed)
        
        preserved = original_texts & compressed_texts
        
        return len(preserved) / max(len(original_texts), 1)
    
    def record_quality_score(self, score: float):
        """Record a quality score for monitoring"""
        self._quality_scores.append(score)
        
        # Keep only last 100 scores
        if len(self._quality_scores) > 100:
            self._quality_scores = self._quality_scores[-100:]
    
    def get_average_quality(self) -> float:
        """Get average quality score"""
        if not self._quality_scores:
            return 0.0
        
        return sum(self._quality_scores) / len(self._quality_scores)
    
    def add_expert_review(self, review: ExpertReview):
        """Add expert review record"""
        self._review_history.append(review)
        
        if review.approved:
            logger.info(f"Expert review approved for {review.document_id}")
        else:
            logger.warning(f"Expert review rejected for {review.document_id}: {review.revision_notes}")
    
    def get_review_statistics(self) -> Dict[str, Any]:
        """Get review statistics"""
        total = len(self._review_history)
        approved = sum(1 for r in self._review_history if r.approved)
        rejected = total - approved
        
        return {
            'total_reviews': total,
            'approved': approved,
            'rejected': rejected,
            'approval_rate': approved / max(total, 1),
            'average_quality': sum(r.overall_quality for r in self._review_history) / max(total, 1)
        }


class HealthcareCompressionModel(DomainCompressionModel):
    """
    Healthcare domain compression model.
    
    Specialized for:
    - Medical records and clinical notes
    - Healthcare research papers
    - Patient information management
    - Clinical trial documentation
    
    Prioritizes preservation of:
    - Medical terminology and codes (ICD-10, CPT)
    - Patient identifiers (anonymized)
    - Clinical measurements and values
    - Diagnosis and treatment information
    """
    
    def __init__(self):
        """Initialize healthcare compression model"""
        config = DomainConfig(
            domain_type=DomainType.HEALTHCARE.value,
            model_name="kernelize-healthcare-v1",
            target_compression_ratio=50.0,
            quality_preservation_target=0.9,
            
            # Medical vocabulary
            medical_term_file="vocabularies/medical_terms.json",
            priority_entities=[
                "PATIENT_ID", "DIAGNOSIS", "MEDICATION", "PROCEDURE",
                "LAB_RESULT", "VITAL_SIGN", "ALLERGY", "IMMUNIZATION"
            ],
            
            # Strict preservation requirements
            min_entity_preservation=0.95,
            min_relation_preservation=0.9,
            min_fact_preservation=0.88,
            
            # Expert review for sensitive content
            requires_expert_review=True,
            review_threshold_quality=0.85
        )
        
        super().__init__(config)
    
    def _load_domain_vocabulary(self):
        """Load medical terminology vocabulary"""
        # Medical vocabulary categories
        self._domain_vocabulary = {
            "icd10_codes": {
                "description": "ICD-10 diagnosis codes",
                "priority": "critical",
                "examples": ["E11.9", "J45.20", "I10", "F32.1"]
            },
            "cpt_codes": {
                "description": "CPT procedure codes",
                "priority": "critical",
                "examples": ["99213", "93000", "27130"]
            },
            "medications": {
                "description": "Common medications",
                "priority": "high",
                "examples": ["metformin", "lisinopril", "atorvastatin"]
            },
            "vital_signs": {
                "description": "Vital sign measurements",
                "priority": "high",
                "examples": ["blood pressure", "heart rate", "temperature"]
            },
            "lab_tests": {
                "description": "Laboratory test names",
                "priority": "medium",
                "examples": ["HbA1c", "creatinine", "cholesterol"]
            }
        }
        
        logger.info("Loaded healthcare domain vocabulary")
    
    def _compile_entity_patterns(self):
        """Compile healthcare entity recognition patterns"""
        self._entity_patterns = {
            "PATIENT_ID": [
                (r'MR[Nn]?\s*#?\s*\d+', "Medical Record Number"),
                (r'Pt\s*ID\s*#?\s*\d+', "Patient ID"),
                (r'DOB\s*:?\s*\d{2}/\d{2}/\d{4}', "Date of Birth"),
            ],
            "DIAGNOSIS": [
                (r'\b[A-Z]\d{2}\.?\d*[a-z]?\b', "ICD-10 Code"),
                (r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\s*(?:syndrome|disease|disorder|condition)\b', "Disease Name"),
            ],
            "MEDICATION": [
                (r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\s*(?:mg|ml|mcg|g|units?)\b', "Dosage"),
                (r'\b(?:mg|ml|mcg|g)\s*\d+(?:\.\d+)?\b', "Numeric Dosage"),
                (r'\b(?:QD|BID|TID|QID|PRN|PO|IV|IM|SQ)\b', "Frequency/Route"),
            ],
            "PROCEDURE": [
                (r'\b\d{5}\b', "CPT Code"),
                (r'\b(?:biopsy|resection|incision|drainage|excision)\b', "Surgical Terms"),
            ],
            "LAB_RESULT": [
                (r'\b\d+(?:\.\d+)?\s*(?:mg/dL|mmol/L|mEq/L|g/dL|%|cells/µL)\b', "Lab Value"),
                (r'\b(?:positive|negative|abnormal|critical|normal)\b', "Lab Status"),
            ],
            "VITAL_SIGN": [
                (r'\d{2,3}/\d{2,3}\s*(?:mmHg)?', "Blood Pressure"),
                (r'\d{2,3}\s*(?:bpm|beats/min)', "Heart Rate"),
                (r'\d{2}\.?\d*\s*°[CF]', "Temperature"),
            ]
        }
    
    def compress(
        self,
        content: str,
        target_ratio: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Compress healthcare document"""
        start_time = time.time()
        original_size = len(content)
        
        target_ratio = target_ratio or self.config.target_compression_ratio
        
        # Extract entities for preservation
        entities = self._extract_domain_entities(content)
        
        # Identify critical sections (preservation priority)
        critical_sections = self._identify_critical_sections(content)
        
        # Compress non-critical sections more aggressively
        compressed_sections = []
        for section_type, section_content in critical_sections.items():
            if section_type in ['MEDICATIONS', 'DIAGNOSIS', 'LABS']:
                # Preserve critical sections
                compressed_sections.append(section_content)
            else:
                # Compress non-critical sections
                compressed = self._compress_section(section_content, target_ratio * 0.7)
                compressed_sections.append(compressed)
        
        compressed = '\n'.join(compressed_sections)
        
        # Calculate metrics
        processing_time = (time.time() - start_time) * 1000
        compressed_size = len(compressed)
        ratio = original_size / max(compressed_size, 1)
        
        # Validate quality
        quality_metrics = self.validate_quality(content, compressed)
        
        return compressed, {
            'compression_ratio': ratio,
            'quality_score': quality_metrics.get('overall', 0.0),
            'entity_preservation': quality_metrics.get('entity_preservation', 0.0),
            'processing_time_ms': processing_time,
            'entities_extracted': len(entities),
            'critical_sections_preserved': len([s for s in critical_sections.keys() if s in ['MEDICATIONS', 'DIAGNOSIS', 'LABS']])
        }
    
    def _identify_critical_sections(self, content: str) -> Dict[str, str]:
        """Identify critical healthcare sections"""
        sections = {}
        
        section_patterns = {
            'CHIEF_COMPLAINT': r'(?i)chief\s*complaint[:\s]*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            'MEDICATIONS': r'(?i)medications?[:\s]*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            'DIAGNOSIS': r'(?i)diagnosis[:\s]*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            'ASSESSMENT': r'(?i)assessment[:\s]*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            'PLAN': r'(?i)plan[:\s]*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            'LABS': r'(?i)laboratory[:\s]*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            'VITALS': r'(?i)vital\s*signs?[:\s]*(.+?)(?=\n\n|\n[A-Z]|\Z)',
            'HISTORY': r'(?i)history\s*of\s*present\s*illness[:\s]*(.+?)(?=\n\n|\n[A-Z]|\Z)',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        # If no sections found, treat entire content
        if not sections:
            sections['FULL_DOCUMENT'] = content
        
        return sections
    
    def _compress_section(
        self,
        section: str,
        target_ratio: float
    ) -> str:
        """Compress a section of the document"""
        if not section:
            return section
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', section)
        
        # Score sentences
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence)
            scored_sentences.append((i, sentence, score))
        
        # Select sentences to keep
        keep_ratio = 1.0 / target_ratio
        num_keep = max(1, int(len(sentences) * keep_ratio))
        
        # Keep highest scoring sentences
        scored_sentences.sort(key=lambda x: x[2], reverse=True)
        selected = scored_sentences[:num_keep]
        
        # Sort by original order
        selected.sort(key=lambda x: x[0])
        
        return ' '.join(s[1] for s in selected)
    
    def _score_sentence(self, sentence: str) -> float:
        """Score sentence importance"""
        score = 0.0
        
        # Check for medical terminology
        medical_terms = 0
        for category, vocab in self._domain_vocabulary.items():
            examples = vocab.get('examples', [])
            for term in examples:
                if term.lower() in sentence.lower():
                    medical_terms += 1
        
        score += medical_terms * 0.3
        
        # Check for entity patterns
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                if re.search(pattern, sentence):
                    score += 0.2
                    break
        
        # Position bonus (first/last sentences often important)
        # This will be adjusted by caller
        
        return score
    
    def _extract_domain_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract healthcare entities"""
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                for match in re.finditer(pattern, content):
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'description': description,
                        'position': match.start()
                    })
        
        return entities
    
    def validate_quality(
        self,
        original: str,
        compressed: str
    ) -> Dict[str, float]:
        """Validate healthcare compression quality"""
        # Extract entities from both
        orig_entities = self._extract_domain_entities(original)
        comp_entities = self._extract_domain_entities(compressed)
        
        entity_preservation = self._calculate_entity_preservation(orig_entities, comp_entities)
        
        # Check for critical entity types
        critical_entities = ['DIAGNOSIS', 'MEDICATION', 'PROCEDURE']
        critical_preservation = 1.0
        
        for entity_type in critical_entities:
            orig_type = [e for e in orig_entities if e['type'] == entity_type]
            comp_type = [e for e in comp_entities if e['type'] == entity_type]
            
            if orig_type:
                preservation = len(set(e['text'] for e in comp_type)) / len(set(e['text'] for e in orig_type))
                critical_preservation = min(critical_preservation, preservation)
        
        # Overall quality
        overall = (
            entity_preservation * 0.4 +
            critical_preservation * 0.4 +
            (1.0 if len(compressed) < len(original) else 0) * 0.2
        )
        
        return {
            'overall': overall,
            'entity_preservation': entity_preservation,
            'critical_preservation': critical_preservation,
            'compression_ratio': len(original) / max(len(compressed), 1)
        }


class FinanceCompressionModel(DomainCompressionModel):
    """
    Finance domain compression model.
    
    Specialized for:
    - Financial reports and statements
    - Market data and analysis
    - Regulatory filings (SEC, etc.)
    - Investment research documents
    
    Prioritizes preservation of:
    - Financial figures and metrics
    - Company identifiers (tickers, CIK)
    - Dates and periods
    - Key performance indicators
    """
    
    def __init__(self):
        """Initialize finance compression model"""
        config = DomainConfig(
            domain_type=DomainType.FINANCE.value,
            model_name="kernelize-finance-v1",
            target_compression_ratio=60.0,
            quality_preservation_target=0.88,
            
            # Financial vocabulary
            financial_term_file="vocabularies/financial_terms.json",
            priority_entities=[
                "TICKER", "CIK", "REVENUE", "PROFIT", "EPS",
                "QUARTER", "YEAR", "FORECAST"
            ],
            
            min_entity_preservation=0.92,
            min_relation_preservation=0.88,
            min_fact_preservation=0.85,
            
            requires_expert_review=False,
            review_threshold_quality=0.75
        )
        
        super().__init__(config)
    
    def _load_domain_vocabulary(self):
        """Load financial terminology vocabulary"""
        self._domain_vocabulary = {
            "financial_metrics": {
                "description": "Key financial metrics",
                "priority": "critical",
                "examples": ["revenue", "net income", "EBITDA", "EPS", "ROE"]
            },
            "market_terms": {
                "description": "Market-related terms",
                "priority": "high",
                "examples": ["market cap", "shares outstanding", "dividend", "yield"]
            },
            "filing_periods": {
                "description": "Reporting periods",
                "priority": "high",
                "examples": ["Q1", "Q2", "Q3", "Q4", "fiscal year", "FY"]
            },
            "regulatory_terms": {
                "description": "Regulatory filing terms",
                "priority": "medium",
                "examples": ["10-K", "10-Q", "8-K", "Form S-1"]
            }
        }
        
        logger.info("Loaded finance domain vocabulary")
    
    def _compile_entity_patterns(self):
        """Compile financial entity recognition patterns"""
        self._entity_patterns = {
            "TICKER": [
                (r'\b[A-Z]{1,5}\b', "Stock Ticker"),
                (r'\([A-Z]{1,5}\)', "Ticker in Parens"),
            ],
            "CIK": [
                (r'CIK\s*[:#]?\s*\d{10}', "CIK Number"),
                (r'\b\d{10}\b', "10-digit CIK"),
            ],
            "REVENUE": [
                (r'\$\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:million|billion|trillion)?\s*(?:in\s+)?revenue', "Revenue"),
                (r'revenue\s+(?:of\s+)?\$\d+(?:,\d{3})*(?:\.\d{2})?', "Revenue Mention"),
            ],
            "PROFIT": [
                (r'\$\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:net\s+)?profit', "Profit"),
                (r'(?:net\s+)?income\s+(?:of\s+)?\$\d+(?:,\d{3})*(?:\.\d{2})?', "Income"),
            ],
            "EPS": [
                (r'\$\d+(?:\.\d{2})?\s*(?:earnings?|EPS)\s*(?:per\s+share)?', "EPS"),
                (r'earnings?\s*per\s*share\s*[:=]\s*\$\d+(?:\.\d{2})?', "EPS Statement"),
            ],
            "QUARTER": [
                (r'Q[1-4]\s*\d{4}', "Quarter Year"),
                (r'(?:first|second|third|fourth)\s*quarter\s*\d{4}', "Quarter Full"),
            ],
            "YEAR": [
                (r'\b(?:19|20)\d{2}\b', "Year"),
                (r'fiscal\s*(?:year)?\s*(?:20)?\d{2}', "Fiscal Year"),
            ],
            "PERCENTAGE": [
                (r'\d+(?:\.\d+)?%', "Percentage"),
                (r'\d+(?:\.\d+)?\s*(?:basis)?\s*points?', "Basis Points"),
            ],
            "MONEY": [
                (r'\$\d+(?:,\d{3})*(?:\.\d{2})?', "Dollar Amount"),
                (r'\d+(?:,\d{3})?\s*(?:million|billion|trillion)\s*(?:USD|\$)', "Scaled Amount"),
            ]
        }
    
    def compress(
        self,
        content: str,
        target_ratio: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Compress financial document"""
        start_time = time.time()
        original_size = len(content)
        
        target_ratio = target_ratio or self.config.target_compression_ratio
        
        # Extract financial entities
        entities = self._extract_domain_entities(content)
        
        # Extract financial tables/summaries (high preservation)
        tables = self._extract_tables(content)
        
        # Compress narrative sections more aggressively
        compressed_parts = []
        
        for table in tables:
            compressed_parts.append(table)  # Preserve tables
        
        # Compress narrative
        narrative = self._extract_narrative(content)
        compressed_narrative = self._compress_text(narrative, target_ratio * 1.2)
        compressed_parts.append(compressed_narrative)
        
        compressed = '\n'.join(compressed_parts)
        
        processing_time = (time.time() - start_time) * 1000
        compressed_size = len(compressed)
        ratio = original_size / max(compressed_size, 1)
        
        quality_metrics = self.validate_quality(content, compressed)
        
        return compressed, {
            'compression_ratio': ratio,
            'quality_score': quality_metrics.get('overall', 0.0),
            'entity_preservation': quality_metrics.get('entity_preservation', 0.0),
            'processing_time_ms': processing_time,
            'entities_extracted': len(entities),
            'tables_preserved': len(tables)
        }
    
    def _extract_tables(self, content: str) -> List[str]:
        """Extract tabular data (high preservation priority)"""
        tables = []
        
        # Look for tabular patterns
        table_pattern = r'(?i)(?:table|figure|exhibit)[:\s]*\d*[\s\S]*?(?=\n\n|\Z)'
        matches = re.findall(table_pattern, content)
        
        # Also extract line-item data
        line_item_pattern = r'^\s*\w+(?:\s+\w+)?\s+\$?\d+(?:,\d{3})*(?:\.\d{2})?\s*\$?\d+(?:,\d{3})*(?:\.\d{2})?\s*$'
        for line in content.split('\n'):
            if re.match(line_item_pattern, line):
                if line not in tables:
                    tables.append(line)
        
        return tables
    
    def _extract_narrative(self, content: str) -> str:
        """Extract narrative text"""
        # Remove tables and figures
        narrative = re.sub(r'(?i)(?:table|figure|exhibit)[:\s]*\d*[\s\S]*?(?=\n\n|\Z)', '', content)
        
        # Remove line items
        lines = [l for l in narrative.split('\n') 
                 if not re.match(r'^\s*\w+(?:\s+\w+)?\s+\$?\d+', l)]
        
        return '\n'.join(lines)
    
    def _compress_text(self, text: str, target_ratio: float) -> str:
        """Compress text section"""
        if not text:
            return text
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence(sentence)
            scored.append((i, sentence, score))
        
        # Select sentences to keep
        keep_ratio = 1.0 / target_ratio
        num_keep = max(1, int(len(sentences) * keep_ratio))
        
        # Keep highest scoring
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = scored[:num_keep]
        
        # Sort by original order
        selected.sort(key=lambda x: x[0])
        
        return ' '.join(s[1] for s in selected)
    
    def _score_sentence(self, sentence: str) -> float:
        """Score sentence importance in financial context"""
        score = 0.0
        
        # Check for financial metrics
        for category, vocab in self._domain_vocabulary.items():
            examples = vocab.get('examples', [])
            for term in examples:
                if term.lower() in sentence.lower():
                    score += 0.25
                    if vocab['priority'] == 'critical':
                        score += 0.15
        
        # Check for entity patterns
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                if re.search(pattern, sentence):
                    if entity_type in ['REVENUE', 'PROFIT', 'EPS', 'TICKER']:
                        score += 0.3
                    else:
                        score += 0.15
                    break
        
        # Numbers indicate important data
        if re.search(r'\$[\d,]+(?:\.\d{2})?|\d+(?:\.\d+)?%', sentence):
            score += 0.2
        
        return score
    
    def _extract_domain_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract financial entities"""
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                for match in re.finditer(pattern, content):
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'description': description,
                        'position': match.start()
                    })
        
        return entities
    
    def validate_quality(
        self,
        original: str,
        compressed: str
    ) -> Dict[str, float]:
        """Validate financial compression quality"""
        orig_entities = self._extract_domain_entities(original)
        comp_entities = self._extract_domain_entities(compressed)
        
        entity_preservation = self._calculate_entity_preservation(orig_entities, comp_entities)
        
        # Critical financial entity preservation
        critical_entities = ['TICKER', 'REVENUE', 'PROFIT', 'EPS']
        critical_preservation = 1.0
        
        for entity_type in critical_entities:
            orig_type = [e for e in orig_entities if e['type'] == entity_type]
            comp_type = [e for e in comp_entities if e['type'] == entity_type]
            
            if orig_type:
                preservation = len(set(e['text'] for e in comp_type)) / len(set(e['text'] for e in orig_type))
                critical_preservation = min(critical_preservation, preservation)
        
        overall = (
            entity_preservation * 0.35 +
            critical_preservation * 0.45 +
            (1.0 if len(compressed) < len(original) else 0) * 0.2
        )
        
        return {
            'overall': overall,
            'entity_preservation': entity_preservation,
            'critical_preservation': critical_preservation,
            'compression_ratio': len(original) / max(len(compressed), 1)
        }


class LegalCompressionModel(DomainCompressionModel):
    """
    Legal domain compression model.
    
    Specialized for:
    - Contracts and agreements
    - Court documents and opinions
    - Legislation and regulations
    - Legal memoranda
    
    Prioritizes preservation of:
    - Defined terms and definitions
    - Obligations and commitments
    - Dates and deadlines
    - Parties and jurisdictions
    """
    
    def __init__(self):
        """Initialize legal compression model"""
        config = DomainConfig(
            domain_type=DomainType.LEGAL.value,
            model_name="kernelize-legal-v1",
            target_compression_ratio=40.0,
            quality_preservation_target=0.92,
            
            legal_term_file="vocabularies/legal_terms.json",
            priority_entities=[
                "PARTY", "DEFINITION", "OBLIGATION", "DATE",
                "JURISDICTION", "SECTION", "CLAUSE"
            ],
            
            min_entity_preservation=0.95,
            min_relation_preservation=0.9,
            min_fact_preservation=0.9,
            
            requires_expert_review=True,
            review_threshold_quality=0.88
        )
        
        super().__init__(config)
    
    def _load_domain_vocabulary(self):
        """Load legal terminology vocabulary"""
        self._domain_vocabulary = {
            "contract_terms": {
                "description": "Contract-related terms",
                "priority": "critical",
                "examples": ["party", "agreement", "term", "condition", "covenant"]
            },
            "legal_actions": {
                "description": "Legal action terms",
                "priority": "high",
                "examples": ["shall", "must", "will", "agrees to", "warrants"]
            },
            "definitions": {
                "description": "Definition indicators",
                "priority": "critical",
                "examples": ["means", "defined as", "refers to"]
            },
            "temporal_terms": {
                "description": "Time-related terms",
                "priority": "high",
                "examples": ["effective date", "termination", "deadline", "within"]
            }
        }
        
        logger.info("Loaded legal domain vocabulary")
    
    def _compile_entity_patterns(self):
        """Compile legal entity recognition patterns"""
        self._entity_patterns = {
            "PARTY": [
                (r'(?i)"[^"]+"\s*(?:hereinafter|referred\s+to\s+as)', "Party Definition"),
                (r'(?i)(?:Party|Signatory|Company|Corporation|LLC|Inc\.)\s*[:#]?\s*[A-Z][A-Za-z\s,\.]+', "Party Reference"),
                (r'(?i)\b[A-Z][A-Za-z\s]+(?:\s+Inc\.|\s+LLC|\s+Corp\.|\s+Ltd\.)\b', "Organization"),
            ],
            "DEFINITION": [
                (r'(?i)"([^"]+)"\s+(?:means|is\s+defined\s+as|refers\s+to)', "Quoted Definition"),
                (r'(?i)Definition[:\s]+"([^"]+)"', "Section Definition"),
            ],
            "OBLIGATION": [
                (r'(?i)\b(?:shall|must|will|agrees?\s+to|is\s+required\s+to)\b[^.!?]+[.!?]', "Obligation Statement"),
                (r'(?i)Party\s+(?:A|B|1|2)\s+(?:shall|must|will)', "Party Obligation"),
            ],
            "DATE": [
                (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', "Full Date"),
                (r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', "Short Date"),
                (r'(?i)\beffective\s+date[:\s]+\S+', "Effective Date"),
            ],
            "JURISDICTION": [
                (r'(?i)\b(?:State of|New York|California|Texas|Delaware)\b', "State"),
                (r'(?i)\b(?:United States|Federal|Law of)\b', "Federal"),
                (r'(?i)\b\d+\s+(?:U\.S\.C\.|C\.F\.R\.)\b', "USC/CFR Reference"),
            ],
            "SECTION": [
                (r'(?i)Section\s+\d+(?:\.\d+)*', "Section Reference"),
                (r'(?i)Article\s+\d+(?:\.\d+)*', "Article Reference"),
                (r'(?i)Exhibit\s+[A-Z]', "Exhibit Reference"),
            ],
            "CLAUSE": [
                (r'(?i)(?:whereas|provided\s+that|in\s+witness\s+whereof)', "Clue Words"),
                (r'(?i)\(?[a-z]\)\s+\w+[^.!?]+[.!?]', "Lettered Clause"),
            ]
        }
    
    def compress(
        self,
        content: str,
        target_ratio: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Compress legal document"""
        start_time = time.time()
        original_size = len(content)
        
        target_ratio = target_ratio or self.config.target_compression_ratio
        
        # Extract legal entities
        entities = self._extract_domain_entities(content)
        
        # Preserve definitions, obligations, key dates
        critical_sections = self._identify_critical_sections(content)
        
        # Compress background/recitals more aggressively
        compressed_parts = []
        
        for section_name, section_content in critical_sections.items():
            if section_name in ['DEFINITIONS', 'OBLIGATIONS', 'DATES']:
                compressed_parts.append(section_content)  # Preserve
            else:
                ratio = target_ratio * 1.5 if section_name in ['BACKGROUND', 'RECITALS'] else target_ratio
                compressed = self._compress_legal_text(section_content, ratio)
                compressed_parts.append(compressed)
        
        compressed = '\n'.join(compressed_parts)
        
        processing_time = (time.time() - start_time) * 1000
        compressed_size = len(compressed)
        ratio = original_size / max(compressed_size, 1)
        
        quality_metrics = self.validate_quality(content, compressed)
        
        return compressed, {
            'compression_ratio': ratio,
            'quality_score': quality_metrics.get('overall', 0.0),
            'entity_preservation': quality_metrics.get('entity_preservation', 0.0),
            'processing_time_ms': processing_time,
            'entities_extracted': len(entities),
            'critical_sections_preserved': len([s for s in critical_sections.keys() if s in ['DEFINITIONS', 'OBLIGATIONS', 'DATES']])
        }
    
    def _identify_critical_sections(self, content: str) -> Dict[str, str]:
        """Identify critical legal sections"""
        sections = {}
        
        section_patterns = {
            'BACKGROUND': r'(?i)background[:\s]*(.+?)(?=\n\n|\n(?:NOW|THEREFORE|WHEREAS)\Z)',
            'RECITALS': r'(?i)recitals?[:\s]*(.+?)(?=\n\n|\n(?:NOW|THEREFORE)\Z)',
            'DEFINITIONS': r'(?i)definitions?[:\s]*(.+?)(?=\n\n|\n(?:OBLIGATIONS|TERMS)\Z)',
            'OBLIGATIONS': r'(?i)obligations?[:\s]*(.+?)(?=\n\n|\n(?:TERMS|CONCLUSION)\Z)',
            'DATES': r'(?i)dates?[:\s]*(.+?)(?=\n\n|\n(?:TERMS)\Z)',
            'TERMS': r'(?i)terms?[:\s]*(.+?)(?=\n\n|\n(?:CONCLUSION|SIGNATURES)\Z)',
            'SIGNATURES': r'(?i)signatures?[:\s]*(.+?)(?=\Z)',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        if not sections:
            sections['FULL_DOCUMENT'] = content
        
        return sections
    
    def _compress_legal_text(self, text: str, target_ratio: float) -> str:
        """Compress legal text while preserving structure"""
        if not text:
            return text
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score sentences
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_legal_sentence(sentence)
            scored.append((i, sentence, score))
        
        # Select sentences to keep
        keep_ratio = 1.0 / target_ratio
        num_keep = max(1, int(len(sentences) * keep_ratio))
        
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = scored[:num_keep]
        
        selected.sort(key=lambda x: x[0])
        
        return ' '.join(s[1] for s in selected)
    
    def _score_legal_sentence(self, sentence: str) -> float:
        """Score legal sentence importance"""
        score = 0.0
        
        # Check for legal vocabulary
        for category, vocab in self._domain_vocabulary.items():
            examples = vocab.get('examples', [])
            for term in examples:
                if term.lower() in sentence.lower():
                    score += 0.25
                    if vocab['priority'] == 'critical':
                        score += 0.2
        
        # Check for entity patterns
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                if re.search(pattern, sentence):
                    if entity_type in ['DEFINITION', 'OBLIGATION', 'PARTY']:
                        score += 0.35
                    else:
                        score += 0.15
                    break
        
        # Legal action words
        if re.search(r'\b(?:shall|must|will|agrees|warrants|covenant)\b', sentence, re.I):
            score += 0.25
        
        # Numbers/dates
        if re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b(?:January|February)\b \d{1,2},? \d{4}', sentence):
            score += 0.15
        
        return score
    
    def _extract_domain_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract legal entities"""
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                for match in re.finditer(pattern, content):
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'description': description,
                        'position': match.start()
                    })
        
        return entities
    
    def validate_quality(
        self,
        original: str,
        compressed: str
    ) -> Dict[str, float]:
        """Validate legal compression quality"""
        orig_entities = self._extract_domain_entities(original)
        comp_entities = self._extract_domain_entities(compressed)
        
        entity_preservation = self._calculate_entity_preservation(orig_entities, comp_entities)
        
        # Critical legal entity preservation
        critical_entities = ['DEFINITION', 'OBLIGATION', 'PARTY', 'DATE']
        critical_preservation = 1.0
        
        for entity_type in critical_entities:
            orig_type = [e for e in orig_entities if e['type'] == entity_type]
            comp_type = [e for e in comp_entities if e['type'] == entity_type]
            
            if orig_type:
                preservation = len(set(e['text'] for e in comp_type)) / len(set(e['text'] for e in orig_type))
                critical_preservation = min(critical_preservation, preservation)
        
        overall = (
            entity_preservation * 0.3 +
            critical_preservation * 0.5 +
            (1.0 if len(compressed) < len(original) else 0) * 0.2
        )
        
        return {
            'overall': overall,
            'entity_preservation': entity_preservation,
            'critical_preservation': critical_preservation,
            'compression_ratio': len(original) / max(len(compressed), 1)
        }


class ScientificCompressionModel(DomainCompressionModel):
    """
    Scientific domain compression model.
    
    Specialized for:
    - Research papers and publications
    - Technical documentation
    - Laboratory reports
    - Grant proposals
    
    Prioritizes preservation of:
    - Methodology and experimental setup
    - Data and results
    - Statistical findings
    - Citations and references
    """
    
    def __init__(self):
        """Initialize scientific compression model"""
        config = DomainConfig(
            domain_type=DomainType.SCIENTIFIC.value,
            model_name="kernelize-scientific-v1",
            target_compression_ratio=55.0,
            quality_preservation_target=0.88,
            
            scientific_term_file="vocabularies/scientific_terms.json",
            priority_entities=[
                "METHOD", "RESULT", "STATISTIC", "CITATION",
                "HYPOTHESIS", "CONCLUSION", "DATA"
            ],
            
            min_entity_preservation=0.9,
            min_relation_preservation=0.85,
            min_fact_preservation=0.82,
            
            requires_expert_review=False,
            review_threshold_quality=0.75
        )
        
        super().__init__(config)
    
    def _load_domain_vocabulary(self):
        """Load scientific terminology vocabulary"""
        self._domain_vocabulary = {
            "methodology": {
                "description": "Methodology terms",
                "priority": "critical",
                "examples": ["method", "experiment", "analysis", "survey", "measurement"]
            },
            "statistics": {
                "description": "Statistical terms",
                "priority": "critical",
                "examples": ["p-value", "significance", "confidence interval", "regression"]
            },
            "results": {
                "description": "Results terminology",
                "priority": "critical",
                "examples": ["result", "finding", "observation", "data", "outcome"]
            },
            "discipline_terms": {
                "description": "Field-specific terms",
                "priority": "high",
                "examples": []
            }
        }
        
        logger.info("Loaded scientific domain vocabulary")
    
    def _compile_entity_patterns(self):
        """Compile scientific entity recognition patterns"""
        self._entity_patterns = {
            "METHOD": [
                (r'(?i)\b(?:method|approach|technique|procedure|protocol)s?[:\s]*.+?(?=\n\n|\Z)', "Method Section"),
                (r'(?i)we\s+(?:used|employed|implemented|performed)\b[^.!?]+[.!?]', "Method Mention"),
            ],
            "RESULT": [
                (r'(?i)\bresults?[:\s]*.+?(?=\n\n|\n(?:DISCUSSION|CONCLUSION)|\Z)', "Results Section"),
                (r'(?i)we\s+(?:found|observed|discovered|measured)\b[^.!?]+[.!?]', "Result Statement"),
            ],
            "STATISTIC": [
                (r'\b0\.\d{2,}\b', "Decimal Probability"),
                (r'\bp\s*[<>=]\s*0?\.\d+', "P-value"),
                (r'\b\d+(?:\.\d+)?\s*(?:%|percent)\b', "Percentage"),
                (r'\b\d+(?:,\d{3})*(?:\.\d+)?\s*(?:±|\+/-)\s*\d+(?:,\d{3})*(?:\.\d+)?', "Mean ± SD"),
                (r'\b(?:R|r)\^?2\s*[=:]\s*0?\.\d+', "R-squared"),
            ],
            "CITATION": [
                (r'\[\d+(?:,\s*\d+)*\]', "Number Citation"),
                (r'(?i)\([A-Z][a-z]+(?:\s+et\s+al\.?)?,?\s+\d{4}\)', "Author-Date Citation"),
                (r'(?i)doi[:\s]*\S+', "DOI Reference"),
            ],
            "HYPOTHESIS": [
                (r'(?i)hypothesis[:\s]*.+?[.!?]', "Hypothesis Statement"),
                (r'(?i)we\s+hypothesized\s+that\b[^.!?]+[.!?]', "Hypothesis Mention"),
            ],
            "CONCLUSION": [
                (r'(?i)\bconclusion[s]?[:\s]*.+?(?=\n\n|\Z)', "Conclusion Section"),
                (r'(?i)in\s+(?:conclusion|summary)\b[^.!?]+[.!?]', "Conclusion Mention"),
            ],
            "DATA": [
                (r'(?i)data\s+(?:were|was)\s+collected\s+from\b[^.!?]+[.!?]', "Data Collection"),
                (r'(?i)\d+(?:,\d{3})*(?:\.\d+)?\s*(?:participants|subjects|samples|observations)\b', "Sample Size"),
            ],
            "FIGURE": [
                (r'(?i)Figure\s+\d+(?:[.:]\s*\d+)?', "Figure Reference"),
                (r'(?i)Table\s+\d+(?:[.:]\s*\d+)?', "Table Reference"),
            ]
        }
    
    def compress(
        self,
        content: str,
        target_ratio: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Compress scientific document"""
        start_time = time.time()
        original_size = len(content)
        
        target_ratio = target_ratio or self.config.target_compression_ratio
        
        # Extract scientific entities
        entities = self._extract_domain_entities(content)
        
        # Preserve methods, results, conclusions
        critical_sections = self._identify_critical_sections(content)
        
        compressed_parts = []
        
        for section_name, section_content in critical_sections.items():
            if section_name in ['METHODS', 'RESULTS', 'CONCLUSIONS']:
                compressed_parts.append(section_content)
            else:
                ratio = target_ratio * 1.3 if section_name in ['BACKGROUND', 'INTRODUCTION'] else target_ratio
                compressed = self._compress_scientific_text(section_content, ratio)
                compressed_parts.append(compressed)
        
        compressed = '\n'.join(compressed_parts)
        
        processing_time = (time.time() - start_time) * 1000
        compressed_size = len(compressed)
        ratio = original_size / max(compressed_size, 1)
        
        quality_metrics = self.validate_quality(content, compressed)
        
        return compressed, {
            'compression_ratio': ratio,
            'quality_score': quality_metrics.get('overall', 0.0),
            'entity_preservation': quality_metrics.get('entity_preservation', 0.0),
            'processing_time_ms': processing_time,
            'entities_extracted': len(entities)
        }
    
    def _identify_critical_sections(self, content: str) -> Dict[str, str]:
        """Identify critical scientific sections"""
        sections = {}
        
        section_patterns = {
            'ABSTRACT': r'(?i)abstract[:\s]*(.+?)(?=\n\n|\n(?:INTRODUCTION|METHODS)\Z)',
            'INTRODUCTION': r'(?i)introduction[:\s]*(.+?)(?=\n\n|\n(?:METHODS|BACKGROUND)\Z)',
            'BACKGROUND': r'(?i)background[:\s]*(.+?)(?=\n\n|\n(?:METHODS)\Z)',
            'METHODS': r'(?i)(?:methods|methodology|materials?\s+and?\s+methods)[:\s]*(.+?)(?=\n\n|\n(?:RESULTS|DISCUSSION)\Z)',
            'RESULTS': r'(?i)results?[:\s]*(.+?)(?=\n\n|\n(?:DISCUSSION|CONCLUSION)\Z)',
            'DISCUSSION': r'(?i)discussion[:\s]*(.+?)(?=\n\n|\n(?:CONCLUSION)\Z)',
            'CONCLUSIONS': r'(?i)conclusion[s]?[:\s]*(.+?)(?=\n\n|\n(?:REFERENCES)\Z)',
            'REFERENCES': r'(?i)references?[:\s]*(.+?)(?=\Z)',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        if not sections:
            sections['FULL_DOCUMENT'] = content
        
        return sections
    
    def _compress_scientific_text(self, text: str, target_ratio: float) -> str:
        """Compress scientific text"""
        if not text:
            return text
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_scientific_sentence(sentence)
            scored.append((i, sentence, score))
        
        keep_ratio = 1.0 / target_ratio
        num_keep = max(1, int(len(sentences) * keep_ratio))
        
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = scored[:num_keep]
        
        selected.sort(key=lambda x: x[0])
        
        return ' '.join(s[1] for s in selected)
    
    def _score_scientific_sentence(self, sentence: str) -> float:
        """Score scientific sentence importance"""
        score = 0.0
        
        # Check vocabulary
        for category, vocab in self._domain_vocabulary.items():
            examples = vocab.get('examples', [])
            for term in examples:
                if term.lower() in sentence.lower():
                    score += 0.25
                    if vocab['priority'] == 'critical':
                        score += 0.15
        
        # Check entity patterns
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                if re.search(pattern, sentence):
                    if entity_type in ['STATISTIC', 'RESULT', 'METHOD', 'CITATION']:
                        score += 0.3
                    else:
                        score += 0.15
                    break
        
        # Statistical significance
        if re.search(r'\bp\s*[<>=]\s*0?\.\d+', sentence):
            score += 0.35
        
        # Numbers/data
        if re.search(r'\d+(?:,\d{3})*(?:\.\d+)?', sentence):
            score += 0.1
        
        return score
    
    def _extract_domain_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract scientific entities"""
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                for match in re.finditer(pattern, content):
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'description': description,
                        'position': match.start()
                    })
        
        return entities
    
    def validate_quality(
        self,
        original: str,
        compressed: str
    ) -> Dict[str, float]:
        """Validate scientific compression quality"""
        orig_entities = self._extract_domain_entities(original)
        comp_entities = self._extract_domain_entities(compressed)
        
        entity_preservation = self._calculate_entity_preservation(orig_entities, comp_entities)
        
        # Critical entity preservation
        critical_entities = ['STATISTIC', 'RESULT', 'METHOD', 'HYPOTHESIS']
        critical_preservation = 1.0
        
        for entity_type in critical_entities:
            orig_type = [e for e in orig_entities if e['type'] == entity_type]
            comp_type = [e for e in comp_entities if e['type'] == entity_type]
            
            if orig_type:
                preservation = len(set(e['text'] for e in comp_type)) / len(set(e['text'] for e in orig_type))
                critical_preservation = min(critical_preservation, preservation)
        
        overall = (
            entity_preservation * 0.35 +
            critical_preservation * 0.45 +
            (1.0 if len(compressed) < len(original) else 0) * 0.2
        )
        
        return {
            'overall': overall,
            'entity_preservation': entity_preservation,
            'critical_preservation': critical_preservation,
            'compression_ratio': len(original) / max(len(compressed), 1)
        }


class GovernmentCompressionModel(DomainCompressionModel):
    """
    Government domain compression model.
    
    Specialized for:
    - Policy documents
    - Public records
    - Government regulations
    - Official correspondence
    
    Prioritizes preservation of:
    - Policy positions and decisions
    - Legal references and citations
    - Procedural requirements
    - Official designations
    """
    
    def __init__(self):
        """Initialize government compression model"""
        config = DomainConfig(
            domain_type=DomainType.GOVERNMENT.value,
            model_name="kernelize-government-v1",
            target_compression_ratio=45.0,
            quality_preservation_target=0.9,
            
            priority_entities=[
                "AGENCY", "REGULATION", "POLICY", "OFFICIAL",
                "EFFECTIVE_DATE", "AUTHORITY"
            ],
            
            min_entity_preservation=0.92,
            min_relation_preservation=0.88,
            min_fact_preservation=0.85,
            
            requires_expert_review=False,
            review_threshold_quality=0.8
        )
        
        super().__init__(config)
    
    def _load_domain_vocabulary(self):
        """Load government terminology vocabulary"""
        self._domain_vocabulary = {
            "government_agencies": {
                "description": "Government agency names",
                "priority": "critical",
                "examples": ["Department of", "Agency", "Bureau", "Office", "Commission"]
            },
            "regulatory_terms": {
                "description": "Regulatory terminology",
                "priority": "critical",
                "examples": ["regulation", "rule", "compliance", "requirement", "mandate"]
            },
            "policy_terms": {
                "description": "Policy-related terms",
                "priority": "high",
                "examples": ["policy", "guideline", "procedure", "directive", "order"]
            },
            "legal_references": {
                "description": "Legal citation terms",
                "priority": "high",
                "examples": ["U.S.C.", "C.F.R.", "Public Law", "Statute"]
            }
        }
        
        logger.info("Loaded government domain vocabulary")
    
    def _compile_entity_patterns(self):
        """Compile government entity recognition patterns"""
        self._entity_patterns = {
            "AGENCY": [
                (r'(?i)(?:Department|Agency|Bureau|Office|Administration|Commission)\s+of\s+[A-Z][A-Za-z\s]+', "Federal Agency"),
                (r'(?i)(?:U\.S\.|United\s+States)\s+[A-Z][A-Za-z\s]+(?:Department|Agency|Bureau)', "US Agency"),
            ],
            "REGULATION": [
                (r'(?i)\d{1,3}\s*C\.F\.R\.\s*\S+', "CFR Citation"),
                (r'(?i)Regulation\s*[:#]?\s*\d+', "Regulation Reference"),
                (r'(?i)\d+\s*U\.S\.C\.\s*§?\s*\d+', "USC Citation"),
            ],
            "POLICY": [
                (r'(?i)Policy\s*[:#]?\s*\w+(?:\s+\w+)*', "Policy Reference"),
                (r'(?i)(?:Executive\s+)?Order\s*[:#]?\s*\d+', "Executive Order"),
            ],
            "OFFICIAL": [
                (r'(?i)(?:Secretary|Director|Administrator|Chair|Commissioner)\s+[A-Z][A-Za-z]+', "Official Title"),
                (r'(?i)Honorable\s+[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?', "Honorable Address"),
            ],
            "EFFECTIVE_DATE": [
                (r'(?i)effective\s+(?:date|from)\s*:?\s*\S+', "Effective Date"),
                (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', "Full Date"),
            ],
            "AUTHORITY": [
                (r'(?i)authorized?\s+by\s+\S+', "Authority Statement"),
                (r'(?i)pursuant\s+to\s+\S+', "Authority Citation"),
                (r'(?i)under\s+(?:the\s+)?(?:authority\s+of\s+)?\S+', "Under Authority"),
            ],
            "PUBLIC_LAW": [
                (r'(?i)Public\s+Law\s*\d+-\d+', "Public Law"),
                (r'(?i)Pub\.?\s*L\.?\s*\d+-\d+', "Pub Law Abbreviated"),
            ]
        }
    
    def compress(
        self,
        content: str,
        target_ratio: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Compress government document"""
        start_time = time.time()
        original_size = len(content)
        
        target_ratio = target_ratio or self.config.target_compression_ratio
        
        entities = self._extract_domain_entities(content)
        critical_sections = self._identify_critical_sections(content)
        
        compressed_parts = []
        
        for section_name, section_content in critical_sections.items():
            if section_name in ['AUTHORITY', 'POLICY', 'REGULATIONS']:
                compressed_parts.append(section_content)
            else:
                ratio = target_ratio * 1.4 if section_name in ['BACKGROUND', 'PURPOSE'] else target_ratio
                compressed = self._compress_government_text(section_content, ratio)
                compressed_parts.append(compressed)
        
        compressed = '\n'.join(compressed_parts)
        
        processing_time = (time.time() - start_time) * 1000
        compressed_size = len(compressed)
        ratio = original_size / max(compressed_size, 1)
        
        quality_metrics = self.validate_quality(content, compressed)
        
        return compressed, {
            'compression_ratio': ratio,
            'quality_score': quality_metrics.get('overall', 0.0),
            'entity_preservation': quality_metrics.get('entity_preservation', 0.0),
            'processing_time_ms': processing_time,
            'entities_extracted': len(entities)
        }
    
    def _identify_critical_sections(self, content: str) -> Dict[str, str]:
        """Identify critical government sections"""
        sections = {}
        
        section_patterns = {
            'PURPOSE': r'(?i)purpose[:\s]*(.+?)(?=\n\n|\n(?:BACKGROUND|AUTHORITY)\Z)',
            'BACKGROUND': r'(?i)background[:\s]*(.+?)(?=\n\n|\n(?:PURPOSE)\Z)',
            'AUTHORITY': r'(?i)authority[:\s]*(.+?)(?=\n\n|\n(?:POLICY|REGULATIONS)\Z)',
            'POLICY': r'(?i)policy[:\s]*(.+?)(?=\n\n|\n(?:PROCEDURES|REGULATIONS)\Z)',
            'PROCEDURES': r'(?i)procedures?[:\s]*(.+?)(?=\n\n|\n(?:REGULATIONS)\Z)',
            'REGULATIONS': r'(?i)regulations?[:\s]*(.+?)(?=\n\n|\n(?:DEFINITIONS)\Z)',
            'DEFINITIONS': r'(?i)definitions?[:\s]*(.+?)(?=\n\n|\n(?:CONTACT)\Z)',
            'CONTACT': r'(?i)contact[:\s]*(.+?)(?=\n\n|\Z)',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, content, re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        if not sections:
            sections['FULL_DOCUMENT'] = content
        
        return sections
    
    def _compress_government_text(self, text: str, target_ratio: float) -> str:
        """Compress government text"""
        if not text:
            return text
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        scored = []
        for i, sentence in enumerate(sentences):
            score = self._score_government_sentence(sentence)
            scored.append((i, sentence, score))
        
        keep_ratio = 1.0 / target_ratio
        num_keep = max(1, int(len(sentences) * keep_ratio))
        
        scored.sort(key=lambda x: x[2], reverse=True)
        selected = scored[:num_keep]
        
        selected.sort(key=lambda x: x[0])
        
        return ' '.join(s[1] for s in selected)
    
    def _score_government_sentence(self, sentence: str) -> float:
        """Score government sentence importance"""
        score = 0.0
        
        for category, vocab in self._domain_vocabulary.items():
            examples = vocab.get('examples', [])
            for term in examples:
                if term.lower() in sentence.lower():
                    score += 0.25
                    if vocab['priority'] == 'critical':
                        score += 0.15
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                if re.search(pattern, sentence):
                    if entity_type in ['REGULATION', 'POLICY', 'AUTHORITY', 'AGENCY']:
                        score += 0.3
                    else:
                        score += 0.15
                    break
        
        return score
    
    def _extract_domain_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract government entities"""
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                for match in re.finditer(pattern, content):
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'description': description,
                        'position': match.start()
                    })
        
        return entities
    
    def validate_quality(
        self,
        original: str,
        compressed: str
    ) -> Dict[str, float]:
        """Validate government compression quality"""
        orig_entities = self._extract_domain_entities(original)
        comp_entities = self._extract_domain_entities(compressed)
        
        entity_preservation = self._calculate_entity_preservation(orig_entities, comp_entities)
        
        critical_entities = ['REGULATION', 'POLICY', 'AUTHORITY', 'AGENCY']
        critical_preservation = 1.0
        
        for entity_type in critical_entities:
            orig_type = [e for e in orig_entities if e['type'] == entity_type]
            comp_type = [e for e in comp_entities if e['type'] == entity_type]
            
            if orig_type:
                preservation = len(set(e['text'] for e in comp_type)) / len(set(e['text'] for e in orig_type))
                critical_preservation = min(critical_preservation, preservation)
        
        overall = (
            entity_preservation * 0.35 +
            critical_preservation * 0.45 +
            (1.0 if len(compressed) < len(original) else 0) * 0.2
        )
        
        return {
            'overall': overall,
            'entity_preservation': entity_preservation,
            'critical_preservation': critical_preservation,
            'compression_ratio': len(original) / max(len(compressed), 1)
        }


class DomainModelRegistry:
    """
    Registry for domain-specific compression models.
    
    Provides model selection, configuration management,
    and benchmark tracking across domains.
    """
    
    def __init__(self):
        """Initialize domain model registry"""
        self._models: Dict[DomainType, DomainCompressionModel] = {}
        self._configs: Dict[DomainType, DomainConfig] = {}
        self._benchmarks: Dict[DomainType, List[DomainBenchmarkResult]] = {}
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default domain models"""
        self.register_model(DomainType.HEALTHCARE, HealthcareCompressionModel())
        self.register_model(DomainType.FINANCE, FinanceCompressionModel())
        self.register_model(DomainType.LEGAL, LegalCompressionModel())
        self.register_model(DomainType.SCIENTIFIC, ScientificCompressionModel())
        self.register_model(DomainType.GOVERNMENT, GovernmentCompressionModel())
    
    def register_model(
        self,
        domain_type: DomainType,
        model: DomainCompressionModel
    ):
        """Register a domain model"""
        self._models[domain_type] = model
        self._configs[domain_type] = model.config
        
        logger.info(f"Registered domain model: {domain_type.value}")
    
    def get_model(self, domain_type: DomainType) -> Optional[DomainCompressionModel]:
        """Get model for domain type"""
        return self._models.get(domain_type)
    
    def get_model_by_name(self, domain_name: str) -> Optional[DomainCompressionModel]:
        """Get model by domain name string"""
        try:
            domain_type = DomainType(domain_name)
            return self.get_model(domain_type)
        except ValueError:
            return None
    
    def auto_detect_domain(self, content: str) -> DomainType:
        """Automatically detect domain of content"""
        scores = {}
        
        for domain_type, model in self._models.items():
            score = self._score_domain(content, model)
            scores[domain_type] = score
        
        # Return highest scoring domain
        best_domain = max(scores, key=scores.get)
        
        return best_domain
    
    def _score_domain(self, content: str, model: DomainCompressionModel) -> float:
        """Score content for domain match"""
        # Extract entities from model
        patterns = model._entity_patterns
        
        score = 0.0
        total_patterns = sum(len(p) for p in patterns.values())
        
        for entity_type, pattern_list in patterns.items():
            for pattern, description in pattern_list:
                matches = len(re.findall(pattern, content))
                score += matches * 0.1
        
        return min(1.0, score / total_patterns)
    
    def compress(
        self,
        content: str,
        domain: Optional[DomainType] = None,
        target_ratio: Optional[float] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Compress content using domain-specific model
        
        Args:
            content: Content to compress
            domain: Optional domain type (auto-detected if not provided)
            target_ratio: Target compression ratio
            
        Returns:
            Compressed content and metrics
        """
        # Auto-detect domain if not provided
        if domain is None:
            domain = self.auto_detect_domain(content)
        
        model = self.get_model(domain)
        
        if model is None:
            raise ValueError(f"No model registered for domain: {domain}")
        
        return model.compress(content, target_ratio)
    
    def add_benchmark(
        self,
        domain_type: DomainType,
        result: DomainBenchmarkResult
    ):
        """Add benchmark result for domain"""
        if domain_type not in self._benchmarks:
            self._benchmarks[domain_type] = []
        
        self._benchmarks[domain_type].append(result)
    
    def get_benchmark_summary(self, domain_type: DomainType) -> Dict[str, Any]:
        """Get benchmark summary for domain"""
        results = self._benchmarks.get(domain_type, [])
        
        if not results:
            return {'count': 0}
        
        return {
            'count': len(results),
            'avg_compression_ratio': sum(r.average_compression_ratio for r in results) / len(results),
            'avg_quality': sum(r.quality_score for r in results) / len(results),
            'avg_entity_preservation': sum(r.entity_preservation for r in results) / len(results)
        }
    
    def list_domains(self) -> List[str]:
        """List available domains"""
        return [dt.value for dt in self._models.keys()]


# Singleton instance
_domain_registry: Optional[DomainModelRegistry] = None


def get_domain_registry() -> DomainModelRegistry:
    """Get domain model registry singleton"""
    global _domain_registry
    
    if _domain_registry is None:
        _domain_registry = DomainModelRegistry()
    
    return _domain_registry


def init_domain_models() -> DomainModelRegistry:
    """Initialize domain models system"""
    global _domain_registry
    
    _domain_registry = DomainModelRegistry()
    
    return _domain_registry
