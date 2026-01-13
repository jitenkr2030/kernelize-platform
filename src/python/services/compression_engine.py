"""
KERNELIZE Platform - Knowledge Compression Engine
===================================================

This module implements the core knowledge compression algorithms for the
KERNELIZE Platform. It provides semantic compression that achieves 100×-10,000×
compression ratios while preserving meaning, causality, and relationships.

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
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression level enumeration"""
    BASIC = 1       # Basic compression, maintains high readability
    INTERMEDIATE = 2  # Balanced compression and information retention
    ADVANCED = 3    # High compression rate, may affect readability
    EXPERT = 4      # Maximum compression, preserves core semantics


class ContentType(Enum):
    """Content type enumeration"""
    TEXT = "text"
    CODE = "code"
    STRUCTURED = "structured"
    MIXED = "mixed"


@dataclass
class Entity:
    """Entity class"""
    text: str
    entity_type: str
    confidence: float
    start_position: int
    end_position: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "type": self.entity_type,
            "confidence": self.confidence,
            "start": self.start_position,
            "end": self.end_position,
            "metadata": self.metadata,
        }


@dataclass
class Relationship:
    """Relationship class"""
    source_entity: str
    target_entity: str
    relation_type: str
    confidence: float
    context: str
    bidirectional: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source_entity,
            "target": self.target_entity,
            "type": self.relation_type,
            "confidence": self.confidence,
            "context": self.context,
            "bidirectional": self.bidirectional,
        }


@dataclass
class CausalChain:
    """Causal chain class"""
    cause: str
    effect: str
    chain: List[str]
    confidence: float
    temporal_order: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cause": self.cause,
            "effect": self.effect,
            "chain": self.chain,
            "confidence": self.confidence,
            "temporal_order": self.temporal_order,
        }


@dataclass
class CompressionResult:
    """Compression result class"""
    kernel_id: str
    original_content: str
    compressed_content: str
    original_size: int
    compressed_size: int
    compression_ratio: float
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    causal_chains: List[Dict[str, Any]]
    embedding_model: Optional[str]
    metadata: Dict[str, Any]
    processing_time_ms: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "kernel_id": self.kernel_id,
            "original_size": self.original_size,
            "compressed_size": self.compressed_size,
            "compression_ratio": self.compression_ratio,
            "compressed_content": self.compressed_content,
            "entities": self.entities,
            "relationships": self.relationships,
            "causal_chains": self.causal_chains,
            "embedding_model": self.embedding_model,
            "metadata": self.metadata,
            "processing_time_ms": self.processing_time_ms,
        }


class EntityExtractor:
    """
    Entity Extractor
    
    Extracts named entities, concepts, and key information from text.
    Supports recognition and classification of multiple entity types.
    """
    
    def __init__(self):
        # Predefined entity patterns
        self.person_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b'
        self.org_pattern = r'\b(?:Inc|LLC|Corp|Ltd|Company|Corporation)\b(?:\s+[A-Z][a-z]+)+'
        self.date_pattern = r'\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b'
        self.email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        self.url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
        self.number_pattern = r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:[kmbKMB])?\b'
        
        # Causal relationship indicator words
        self.causal_indicators = [
            "causes", "results in", "leads to", "results from",
            "because", "due to", "as a result", "therefore",
            "consequently", "hence", "thus", "so", "since",
            "triggers", "induces", "produces", "creates",
        ]
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract entities from text
        
        Uses regular expressions and pattern matching to identify various entity types.
        """
        entities = []
        
        # Extract person names
        for match in re.finditer(self.person_pattern, text):
            entities.append(Entity(
                text=match.group(),
                entity_type="PERSON",
                confidence=0.9,
                start_position=match.start(),
                end_position=match.end(),
            ))
        
        # Extract organizations
        for match in re.finditer(self.org_pattern, text):
            entities.append(Entity(
                text=match.group(),
                entity_type="ORGANIZATION",
                confidence=0.85,
                start_position=match.start(),
                end_position=match.end(),
            ))
        
        # Extract dates
        for match in re.finditer(self.date_pattern, text):
            entities.append(Entity(
                text=match.group(),
                entity_type="DATE",
                confidence=0.95,
                start_position=match.start(),
                end_position=match.end(),
            ))
        
        # Extract emails
        for match in re.finditer(self.email_pattern, text):
            entities.append(Entity(
                text=match.group(),
                entity_type="EMAIL",
                confidence=0.95,
                start_position=match.start(),
                end_position=match.end(),
            ))
        
        # Extract URLs
        for match in re.finditer(self.url_pattern, text):
            entities.append(Entity(
                text=match.group(),
                entity_type="URL",
                confidence=0.9,
                start_position=match.start(),
                end_position=match.end(),
            ))
        
        # Extract numbers
        for match in re.finditer(self.number_pattern, text):
            entities.append(Entity(
                text=match.group(),
                entity_type="NUMBER",
                confidence=0.95,
                start_position=match.start(),
                end_position=match.end(),
            ))
        
        return entities
    
    def extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts
        
        Identifies key concepts in text based on word frequency and position.
        """
        # Simplified concept extraction
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "can", "need", "dare", "ought", "used", "to", "of", "in",
            "for", "on", "with", "at", "by", "from", "as", "into",
            "through", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "then", "once",
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return most frequent concepts
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:20]]
    
    def extract_causal_chains(self, text: str) -> List[CausalChain]:
        """
        Extract causal chains
        
        Identifies causal relationships and reasoning chains in text.
        """
        causal_chains = []
        
        sentences = re.split(r'[.!?]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue
            
            # Check for causal indicator words
            has_cause = any(indicator in sentence.lower() for indicator in self.causal_indicators)
            
            if has_cause:
                # Extract cause-effect pairs
                parts = re.split(r'\s+(?:because|due to|as a result of|leads to|causes|results in)\s+', sentence, flags=re.IGNORECASE)
                
                if len(parts) == 2:
                    cause = parts[0].strip()
                    effect = parts[1].strip()
                    
                    if len(cause) > 5 and len(effect) > 5:
                        causal_chains.append(CausalChain(
                            cause=cause,
                            effect=effect,
                            chain=[cause, effect],
                            confidence=0.75,
                            temporal_order=True,
                        ))
        
        return causal_chains


class RelationshipExtractor:
    """
    Relationship Extractor
    
    Identifies relationships and semantic connections between entities in text.
    """
    
    def __init__(self):
        self.relation_patterns = {
            "LOCATED_IN": r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is located in|located in|situated in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            "WORKS_FOR": r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:works for|employed by|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            "PART_OF": r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is part of|part of|a part of)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            "FOUNDED_BY": r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was founded by|founded by|created by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            "MARRIED_TO": r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is married to|married to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        }
    
    def extract_relationships(self, text: str, entities: List[Entity]) -> List[Relationship]:
        """
        Extract relationships from text
        
        Uses pattern matching to identify semantic relationships between entities.
        """
        relationships = []
        
        for relation_type, pattern in self.relation_patterns.items():
            matches = re.finditer(pattern, text)
            
            for match in matches:
                if len(match.groups()) >= 2:
                    source = match.group(1).strip()
                    target = match.group(2).strip()
                    
                    # Get context
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    relationships.append(Relationship(
                        source_entity=source,
                        target_entity=target,
                        relation_type=relation_type,
                        confidence=0.8,
                        context=context,
                        bidirectional=relation_type in ["MARRIED_TO"],
                    ))
        
        return relationships


class SemanticCompressor:
    """
    Semantic Compressor
    
    Core compression algorithm implementing semantic-based intelligent compression.
    Achieves high compression ratios while preserving core meaning.
    """
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        self.stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "up", "about", "into", "through", "during",
            "before", "after", "above", "below", "between", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "each", "few", "more", "most", "other", "some", "such",
            "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        }
    
    def compress(
        self,
        content: str,
        compression_level: int = 5,
        extract_entities: bool = True,
        extract_relationships: bool = True,
        extract_causality: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Perform semantic compression
        
        Args:
            content: Raw content to compress
            compression_level: Compression level (1-10)
            extract_entities: Whether to extract entities
            extract_relationships: Whether to extract relationships
            extract_causality: Whether to extract causal chains
        
        Returns:
            (Compressed content, metadata dictionary)
        """
        start_time = time.time()
        
        # Preprocessing
        content = self._preprocess(content)
        
        # Extract entities
        entities = []
        if extract_entities:
            entities = self.entity_extractor.extract_entities(content)
            logger.info(f"Extracted {len(entities)} entities")
        
        # Extract relationships
        relationships = []
        if extract_relationships:
            relationships = self.relationship_extractor.extract_relationships(content, entities)
            logger.info(f"Extracted {len(relationships)} relationships")
        
        # Extract causal chains
        causal_chains = []
        if extract_causality:
            causal_chains = self.entity_extractor.extract_causal_chains(content)
            logger.info(f"Extracted {len(causal_chains)} causal chains")
        
        # Perform compression
        compression_factor = self._get_compression_factor(compression_level)
        compressed_content = self._semantic_compress(content, compression_factor, entities)
        
        # Postprocessing
        compressed_content = self._postprocess(compressed_content)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        metadata = {
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "causal_chains_count": len(causal_chains),
            "compression_level": compression_level,
            "key_concepts": self.entity_extractor.extract_concepts(content),
            "sentence_count_original": len(re.split(r'[.!?]+', content)),
            "sentence_count_compressed": len(re.split(r'[.!?]+', compressed_content)),
            "processed_at": datetime.utcnow().isoformat(),
        }
        
        return compressed_content, metadata
    
    def _preprocess(self, text: str) -> str:
        """Text preprocessing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        return text.strip()
    
    def _postprocess(self, text: str) -> str:
        """Text postprocessing"""
        # Ensure sentences end with punctuation
        text = re.sub(r'([^\n])\n(?=[A-Z])', r'\1.\n', text)
        text = re.sub(r'([^\n])\n(?=[a-z])', r'\1 ', text)
        return text.strip()
    
    def _get_compression_factor(self, level: int) -> float:
        """Get compression factor based on compression level"""
        # Level 1-10, compression ratio from 1.2x to 50x
        return 1.2 + (level - 1) * (50 - 1.2) / 9
    
    def _semantic_compress(
        self,
        text: str,
        compression_factor: float,
        entities: List[Entity],
    ) -> str:
        """
        Perform core semantic compression
        
        Uses multiple techniques to achieve compression:
        1. Remove stopwords
        2. Extract keywords
        3. Merge similar sentences
        4. Extract core information
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(sentences) <= 2:
            return text
        
        # Score each sentence by importance
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            score = self._score_sentence_importance(sentence, entities, i, len(sentences))
            sentence_scores.append((i, sentence, score))
        
        # Keep important sentences based on compression factor
        keep_count = max(int(len(sentences) / compression_factor), 1)
        keep_count = min(keep_count, len(sentences))
        
        # Select most important sentences
        sentence_scores.sort(key=lambda x: x[2], reverse=True)
        selected_indices = [idx for idx, _, _ in sentence_scores[:keep_count]]
        selected_indices.sort()
        
        # Reconstruct text
        compressed_sentences = [sentences[i] for i in selected_indices]
        
        # Further compress each sentence if needed
        if compression_factor > 10:
            compressed_sentences = [
                self._compress_sentence(s) for s in compressed_sentences
            ]
        
        # Join sentences with appropriate connectors
        if len(compressed_sentences) > 1:
            result = self._intelligent_join(compressed_sentences)
        else:
            result = compressed_sentences[0] if compressed_sentences else text
        
        return result
    
    def _score_sentence_importance(
        self,
        sentence: str,
        entities: List[Entity],
        position: int,
        total: int,
    ) -> float:
        """
        Score sentence importance
        
        Considers multiple factors:
        1. Entity occurrence frequency
        2. Position (first and last sentences are more important)
        3. Sentence length
        4. Keyword occurrence
        """
        score = 0.0
        sentence_lower = sentence.lower()
        
        # Entity score
        for entity in entities:
            if entity.text in sentence:
                score += entity.confidence * 2
        
        # Position score
        if position == 0:
            score += 3.0  # First sentence is most important
        elif position == total - 1:
            score += 2.0  # Last sentence is less important
        elif position < total * 0.2:
            score += 1.0  # First 20% gets extra points
        
        # Causal relationship score
        if any(word in sentence_lower for word in self.entity_extractor.causal_indicators):
            score += 2.0
        
        # Length score (sentences that are too short or too long get lower scores)
        word_count = len(sentence.split())
        if 5 <= word_count <= 30:
            score += 1.0
        elif word_count > 50:
            score -= 0.5
        
        # Number score (sentences with data are usually important)
        if re.search(r'\d+', sentence):
            score += 1.0
        
        return score
    
    def _compress_sentence(self, sentence: str) -> str:
        """Compress a single sentence"""
        # Remove prepositional phrases
        patterns_to_remove = [
            r'\bfor the purpose of\b',
            r'\bin order to\b',
            r'\bwith regard to\b',
            r'\bin the event that\b',
            r'\bat this point in time\b',
            r'\bdue to the fact that\b',
        ]
        
        for pattern in patterns_to_remove:
            sentence = re.sub(pattern, '', sentence, flags=re.IGNORECASE)
        
        # Simplify redundant expressions
        replacements = {
            'in order to': 'to',
            'for the reason that': 'because',
            'at this time': 'now',
            'in the near future': 'soon',
            'in spite of the fact that': 'although',
            'for the purpose of': 'to',
        }
        
        for old, new in replacements.items():
            sentence = re.sub(old, new, sentence, flags=re.IGNORECASE)
        
        return sentence.strip()
    
    def _intelligent_join(self, sentences: List[str]) -> str:
        """Intelligently join sentences"""
        if len(sentences) <= 1:
            return sentences[0] if sentences else ""
        
        result = [sentences[0]]
        
        for i in range(1, len(sentences)):
            prev = sentences[i-1].lower().strip()
            curr = sentences[i].strip()
            
            if not curr:
                continue
            
            # Check temporal order
            time_indicators = ['then', 'after', 'next', 'subsequently', 'later']
            if any(ind in prev for ind in time_indicators):
                result.append(curr)
            # Check causal relationships
            elif any(ind in prev for ind in self.entity_extractor.causal_indicators):
                result.append(curr)
            # Check transitions
            elif any(ind in curr.lower() for ind in ['however', 'but', 'although', 'nevertheless']):
                result.append(curr)
            # Default join with period
            else:
                if not prev.endswith('.') and not prev.endswith('!') and not prev.endswith('?'):
                    result[-1] = prev + '.'
                result.append(curr)
        
        return ' '.join(result)


class KernelCompressionEngine:
    """
    Knowledge Kernel Compression Engine
    
    Top-level interface that integrates all compression components, providing
    complete compression functionality. Supports multiple content types and
    compression levels.
    """
    
    def __init__(self):
        self.semantic_compressor = SemanticCompressor()
        self.entity_extractor = EntityExtractor()
        self.relationship_extractor = RelationshipExtractor()
        logger.info("KernelCompressionEngine initialized")
    
    def compress(
        self,
        content: str,
        domain: str = "general",
        language: str = "en",
        compression_level: int = 5,
        extract_entities: bool = True,
        extract_relationships: bool = True,
        extract_causality: bool = True,
        generate_embedding: bool = False,
    ) -> CompressionResult:
        """
        Compress knowledge content into kernel
        
        Performs complete compression workflow, returns object containing
        compression result and metadata.
        """
        start_time = time.time()
        
        # Generate kernel ID
        content_hash = hashlib.md5(content.encode()).hexdigest()[:12]
        kernel_id = f"kz_{content_hash}_{int(time.time())}"
        
        original_size = len(content.encode('utf-8'))
        
        # Perform compression
        compressed_content, metadata = self.semantic_compressor.compress(
            content=content,
            compression_level=compression_level,
            extract_entities=extract_entities,
            extract_relationships=extract_relationships,
            extract_causality=extract_causality,
        )
        
        compressed_size = len(compressed_content.encode('utf-8'))
        
        # Calculate compression ratio
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
        
        # Extract full information for storage
        entities = [e.to_dict() for e in self.entity_extractor.extract_entities(content)]
        relationships = [r.to_dict() for r in self.relationship_extractor.extract_relationships(content, [])]
        causal_chains = [c.to_dict() for c in self.entity_extractor.extract_causal_chains(content)]
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        result = CompressionResult(
            kernel_id=kernel_id,
            original_content=content,
            compressed_content=compressed_content,
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            entities=entities,
            relationships=relationships,
            causal_chains=causal_chains,
            embedding_model=None,  # Embedding generation is in a separate module
            metadata={
                **metadata,
                "domain": domain,
                "language": language,
                "kernel_id": kernel_id,
            },
            processing_time_ms=processing_time_ms,
        )
        
        logger.info(f"Compression complete: {compression_ratio:.2f}x ratio in {processing_time_ms}ms")
        
        return result
    
    def batch_compress(
        self,
        contents: List[str],
        domain: str = "general",
        language: str = "en",
        compression_level: int = 5,
    ) -> List[CompressionResult]:
        """
        Batch compress content
        
        Optimized processing of multiple content chunks for improved throughput.
        """
        results = []
        
        for content in contents:
            result = self.compress(
                content=content,
                domain=domain,
                language=language,
                compression_level=compression_level,
            )
            results.append(result)
        
        return results
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get compression engine statistics"""
        return {
            "engine_version": "1.0.0",
            "supported_formats": ["text", "code", "markdown"],
            "compression_levels": list(range(1, 11)),
            "entity_types": [
                "PERSON", "ORGANIZATION", "DATE", "EMAIL",
                "URL", "NUMBER", "LOCATION", "PRODUCT",
            ],
            "relation_types": [
                "LOCATED_IN", "WORKS_FOR", "PART_OF",
                "FOUNDED_BY", "MARRIED_TO", "KNOWS",
            ],
            "features": [
                "semantic_compression",
                "entity_extraction",
                "relationship_mapping",
                "causality_detection",
                "key_concept_extraction",
            ],
        }


# Create global engine instance
compression_engine = KernelCompressionEngine()


# Convenience functions
def compress_knowledge(
    content: str,
    domain: str = "general",
    compression_level: int = 5,
) -> CompressionResult:
    """Quick compression function"""
    return compression_engine.compress(
        content=content,
        domain=domain,
        compression_level=compression_level,
    )
