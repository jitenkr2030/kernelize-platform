#!/usr/bin/env python3
"""
Natural Language Query Understanding
=====================================

This module implements intelligent query processing that interprets user intent,
expands queries semantically, handles complex boolean logic, and supports
temporal constraints.

Components:
- Query Intent Classification
- Entity Extraction and Temporal Parsing
- Query Expansion with Synonyms
- Boolean Query Parsing
- Query Rewrite Pipeline

Author: MiniMax Agent
"""

import re
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """Classification of query intents"""
    FACT_RETRIEVAL = "fact_retrieval"
    EXPLANATION = "explanation"
    COMPARISON = "comparison"
    CAUSAL = "causal"
    MULTI_HOP = "multi_hop"
    SUMMARY = "summary"
    TEMPORAL = "temporal"
    DEFINITION = "definition"
    LIST = "list"
    PROCEDURAL = "procedural"
    UNKNOWN = "unknown"


class TemporalRelation(Enum):
    """Temporal relationships"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    BETWEEN = "between"
    RECENT = "recent"
    ALL_TIME = "all_time"


class QueryOperator(Enum):
    """Boolean operators"""
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    MAYBE = "MAYBE"


@dataclass
class ExtractedEntity:
    """Entity extracted from query"""
    text: str
    entity_type: str
    value: Any
    start_pos: int
    end_pos: int
    confidence: float


@dataclass
class TemporalConstraint:
    """Temporal constraint from query"""
    relation: TemporalRelation
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    original_text: str
    confidence: float


@dataclass
class QueryComponent:
    """Component of a parsed query"""
    text: str
    operator: Optional[QueryOperator]
    is_negated: bool
    weight: float
    entities: List[ExtractedEntity]
    temporal: Optional[TemporalConstraint]


@dataclass
class QueryRewrite:
    """Rewritten query with expanded terms"""
    original_query: str
    rewritten_query: str
    intent: QueryIntent
    components: List[QueryComponent]
    entities: List[ExtractedEntity]
    temporal_constraints: List[TemporalConstraint]
    synonyms_added: List[str]
    confidence: float
    suggested_top_k: int
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryIntentClassifier:
    """
    Classifies query intent using pattern matching and embedding similarity
    """
    
    # Intent patterns (regex-based for speed)
    INTENT_PATTERNS = {
        QueryIntent.CAUSAL: [
            r"(why|cause|reason|result|effect|impact|lead to|due to|because of)",
            r"what if",
            r"how did.*happen",
            r"what made.*possible",
        ],
        QueryIntent.COMPARISON: [
            r"(compare|versus|vs|difference|between.*and|similarly|unlike)",
            r"better.*than|worse.*than",
            r"same as|different from",
        ],
        QueryIntent.EXPLANATION: [
            r"(explain|describe|elaborate|tell me about)",
            r"what is.*mean|meaning of",
            r"how does.*work",
        ],
        QueryIntent.DEFINITION: [
            r"(what is|what are|define|definition of)",
            r"meaning of",
        ],
        QueryIntent.LIST: [
            r"(list|name|give me|provide|what are)",
            r"examples of|instances of",
        ],
        QueryIntent.PROCEDURAL: [
            r"(how to|steps|process|procedure|instructions)",
            r"way to.*do|method for",
        ],
        QueryIntent.TEMPORAL: [
            r"(when|during|period|time|from.*to)",
            r"in \d{4}|year \d{4}",
            r"last \w+|recent",
        ],
        QueryIntent.MULTI_HOP: [
            r"(who.*that|which.*that|.*of the .* who)",
            r"first.*then|before.*after",
            r"connected to|related to.*that",
        ],
    }
    
    def __init__(self, embedding_model=None):
        """
        Initialize classifier
        
        Args:
            embedding_model: Optional sentence-transformers model for semantic classification
        """
        self.embedding_model = embedding_model
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency"""
        self._compiled_patterns = {}
        for intent, patterns in self.INTENT_PATTERNS.items():
            compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
            self._compiled_patterns[intent] = compiled
    
    def classify(self, query: str) -> Tuple[QueryIntent, float]:
        """
        Classify query intent
        
        Args:
            query: User query string
            
        Returns:
            (intent, confidence_score)
        """
        query_lower = query.lower()
        
        # Try pattern matching first
        best_intent = QueryIntent.UNKNOWN
        best_confidence = 0.0
        
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(query_lower)
                if match:
                    # Confidence based on match position and length
                    position_score = 1.0 - (match.start() / len(query))
                    length_score = len(match.group()) / len(query)
                    confidence = (position_score * 0.6 + length_score * 0.4)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_intent = intent
        
        # Use embedding model for refinement if available
        if self.embedding_model and best_intent == QueryIntent.UNKNOWN:
            semantic_intent, semantic_confidence = self._classify_semantic(query)
            if semantic_confidence > best_confidence:
                return semantic_intent, semantic_confidence
        
        # Detect multi-hop patterns
        if best_intent == QueryIntent.UNKNOWN:
            if self._detect_multi_hop(query):
                return QueryIntent.MULTI_HOP, 0.75
        
        return best_intent, best_confidence if best_confidence > 0 else 0.5
    
    def _detect_multi_hop(self, query: str) -> bool:
        """Detect if query likely requires multi-hop reasoning"""
        multi_hop_patterns = [
            r"\b\w+\b.*\b\w+\b.*\b\w+\b",  # Multiple entities
            r"(who|what|which).*(that|who|which).*(is|are|was|were)",  # Relative clause
            r"(company|person|place).*whose",  # Possessive
        ]
        
        for pattern in multi_hop_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def _classify_semantic(self, query: str) -> Tuple[QueryIntent, float]:
        """Use embedding model for semantic classification"""
        # This is a placeholder - would use actual embeddings
        # For now, return default
        return QueryIntent.FACT_RETRIEVAL, 0.6


class EntityExtractor:
    """
    Extracts named entities and temporal expressions from queries
    """
    
    ENTITY_TYPES = [
        "DATE", "TIME", "DURATION", "PERSON", "ORG", "LOCATION",
        "PRODUCT", "EVENT", "CONCEPT", "NUMBER", "PERCENTAGE", "MONEY",
    ]
    
    # Patterns for entity extraction
    DATE_PATTERNS = [
        r"\d{4}-\d{2}-\d{2}",  # ISO format
        r"\d{1,2}/\d{1,2}/\d{2,4}",  # US format
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]* \d{1,2},? \d{4}",  # Month format
        r"the \d{1,2}(st|nd|rd|th)? of (jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*",
        r"\d{4}",  # Year only
    ]
    
    TEMPORAL_EXPRESSION_PATTERNS = {
        "today": lambda: datetime.utcnow(),
        "yesterday": lambda: datetime.utcnow() - timedelta(days=1),
        "tomorrow": lambda: datetime.utcnow() + timedelta(days=1),
        "last week": lambda: datetime.utcnow() - timedelta(weeks=1),
        "last month": lambda: datetime.utcnow() - timedelta(days=30),
        "last year": lambda: datetime.utcnow() - timedelta(days=365),
        "recently": lambda: datetime.utcnow() - timedelta(days=30),
        "this week": lambda: datetime.utcnow() - timedelta(days=7),
        "this month": lambda: datetime.utcnow() - timedelta(days=30),
        "this year": lambda: datetime.utcnow() - timedelta(days=365),
    }
    
    def __init__(self):
        """Initialize entity extractor"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self._date_patterns = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self._temporal_patterns = {
            k: re.compile(v, re.IGNORECASE) 
            for k, v in [
                (k, re.escape(k)) for k in self.TEMPORAL_EXPRESSION_PATTERNS.keys()
            ]
        }
        
        # Entity patterns (simplified NER)
        self._entity_patterns = {
            "PERSON": r"\b[A-Z][a-z]+ [A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b",
            "ORG": r"\b(?:Inc|LLC|Corp|Ltd|Company|Corporation|Association)\b",
            "LOCATION": r"\b(?:New York|London|Paris|Tokyo|San Francisco|Mountain View)\b",
            "NUMBER": r"\b\d+(?:,\d{3})*(?:\.\d+)?\b",
            "PERCENTAGE": r"\d+(?:\.\d+)?%",
            "MONEY": r"\$[\d,]+(?:\.\d{2})?",
        }
    
    def extract_entities(self, query: str) -> List[ExtractedEntity]:
        """
        Extract entities from query
        
        Args:
            query: User query string
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract dates
        entities.extend(self._extract_dates(query))
        
        # Extract temporal expressions
        entities.extend(self._extract_temporal(query))
        
        # Extract named entities
        entities.extend(self._extract_named_entities(query))
        
        return entities
    
    def _extract_dates(self, query: str) -> List[ExtractedEntity]:
        """Extract date entities"""
        entities = []
        
        for pattern in self._date_patterns:
            for match in pattern.finditer(query):
                try:
                    date = self._parse_date(match.group())
                    entities.append(ExtractedEntity(
                        text=match.group(),
                        entity_type="DATE",
                        value=date,
                        start_pos=match.start(),
                        end_pos=match.end(),
                        confidence=0.9,
                    ))
                except:
                    pass
        
        return entities
    
    def _extract_temporal(self, query: str) -> List[ExtractedEntity]:
        """Extract temporal expressions like 'last week'"""
        entities = []
        
        for expr, pattern in self._temporal_patterns.items():
            for match in pattern.finditer(query):
                date = self.TEMPORAL_EXPRESSION_PATTERNS[expr]()
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type="TEMPORAL",
                    value=date,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.8,
                ))
        
        return entities
    
    def _extract_named_entities(self, query: str) -> List[ExtractedEntity]:
        """Extract named entities using patterns"""
        entities = []
        
        for entity_type, pattern in self._entity_patterns.items():
            compiled = re.compile(pattern)
            for match in compiled.finditer(query):
                entities.append(ExtractedEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    value=match.group(),
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.7,
                ))
        
        return entities
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date string to datetime"""
        date_str = date_str.strip()
        
        # Try ISO format
        try:
            return datetime.fromisoformat(date_str.replace("/", "-"))
        except:
            pass
        
        # Try month format
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        
        match = re.match(r"(\w+)\s+(\d{1,2}),?\s+(\d{4})", date_str, re.IGNORECASE)
        if match:
            month = months.get(match.group(1).lower()[:3], 1)
            day = int(match.group(2))
            year = int(match.group(3))
            return datetime(year, month, day)
        
        # Try year only
        match = re.match(r"(\d{4})", date_str)
        if match:
            return datetime(int(match.group(1)), 1, 1)
        
        raise ValueError(f"Cannot parse date: {date_str}")


class TemporalQueryParser:
    """
    Parses temporal constraints from queries
    """
    
    def parse(self, query: str) -> List[TemporalConstraint]:
        """
        Parse temporal constraints from query
        
        Args:
            query: User query string
            
        Returns:
            List of temporal constraints
        """
        constraints = []
        query_lower = query.lower()
        
        # Check for specific date ranges
        between_match = re.search(
            r"(between|from)\s+(.+?)\s+(to|until)\s+(.+)",
            query_lower
        )
        if between_match:
            start = self._parse_temporal_expr(between_match.group(2))
            end = self._parse_temporal_expr(between_match.group(4))
            if start and end:
                constraints.append(TemporalConstraint(
                    relation=TemporalRelation.BETWEEN,
                    start_date=start,
                    end_date=end,
                    original_text=between_match.group(0),
                    confidence=0.9,
                ))
        
        # Check for "before" constraints
        before_match = re.search(
            r"before\s+(.+?)(?:\.|,|$)",
            query_lower
        )
        if before_match:
            date = self._parse_temporal_expr(before_match.group(1))
            if date:
                constraints.append(TemporalConstraint(
                    relation=TemporalRelation.BEFORE,
                    start_date=None,
                    end_date=date,
                    original_text=before_match.group(0),
                    confidence=0.85,
                ))
        
        # Check for "after" constraints
        after_match = re.search(
            r"after\s+(.+?)(?:\.|,|$)",
            query_lower
        )
        if after_match:
            date = self._parse_temporal_expr(after_match.group(1))
            if date:
                constraints.append(TemporalConstraint(
                    relation=TemporalRelation.AFTER,
                    start_date=date,
                    end_date=None,
                    original_text=after_match.group(0),
                    confidence=0.85,
                ))
        
        # Check for "during" constraints
        during_match = re.search(
            r"during\s+(.+?)(?:\.|,|$)",
            query_lower
        )
        if during_match:
            date = self._parse_temporal_expr(during_match.group(1))
            if date:
                constraints.append(TemporalConstraint(
                    relation=TemporalRelation.DURING,
                    start_date=date - timedelta(days=365),
                    end_date=date + timedelta(days=365),
                    original_text=during_match.group(0),
                    confidence=0.8,
                ))
        
        # Check for recent time periods
        if re.search(r"(recent|last \w+|past \w+)", query_lower):
            days_map = {
                "day": 1, "week": 7, "month": 30, "year": 365,
                "week": 7, "month": 30,
            }
            
            match = re.search(r"(last|past) (\w+)", query_lower)
            if match:
                period = match.group(2).rstrip("s")  # Remove plural
                days = days_map.get(period, 30)
                
                constraints.append(TemporalConstraint(
                    relation=TemporalRelation.RECENT,
                    start_date=datetime.utcnow() - timedelta(days=days),
                    end_date=datetime.utcnow(),
                    original_text=match.group(0),
                    confidence=0.85,
                ))
        
        return constraints
    
    def _parse_temporal_expr(self, expr: str) -> Optional[datetime]:
        """Parse a temporal expression"""
        expr = expr.strip()
        
        # Check for date patterns
        date_extractor = EntityExtractor()
        entities = date_extractor.extract_entities(expr)
        
        for entity in entities:
            if entity.entity_type in ("DATE", "TEMPORAL"):
                return entity.value
        
        # Check for relative expressions
        expr_lower = expr.lower()
        now = datetime.utcnow()
        
        if "today" in expr_lower:
            return now
        elif "yesterday" in expr_lower:
            return now - timedelta(days=1)
        elif "last week" in expr_lower:
            return now - timedelta(weeks=1)
        elif "last month" in expr_lower:
            return now - timedelta(days=30)
        elif "last year" in expr_lower:
            return now - timedelta(days=365)
        
        return None


class QueryExpander:
    """
    Expands queries using synonyms and semantic similarity
    """
    
    # Common domain-specific synonyms
    SYNONYMS = {
        "ai": ["artificial intelligence", "machine intelligence", "ML"],
        "machine learning": ["ML", "deep learning", "neural networks"],
        "neural network": ["neural net", "ANN", "deep neural network"],
        "data science": ["data analysis", "analytics", "data mining"],
        "nlp": ["natural language processing", "computational linguistics"],
        "computer vision": ["image recognition", "visual AI"],
        "cloud computing": ["cloud", "cloud services", "AWS", "Azure", "GCP"],
        "api": ["application programming interface", "endpoint"],
        "database": ["DB", "data store", "data repository"],
        "software": ["code", "program", "application", "app"],
        "company": ["business", "corporation", "firm", "enterprise"],
        "document": ["doc", "file", "paper", "report"],
        "study": ["research", "analysis", "investigation"],
        "find": ["discover", "locate", "identify", "retrieve"],
        "use": ["utilize", "employ", "apply", "leverage"],
        "show": ["display", "present", "demonstrate", "illustrate"],
    }
    
    def __init__(self, wordnet_path: Optional[str] = None, embedding_model=None):
        """
        Initialize query expander
        
        Args:
            wordnet_path: Optional path to WordNet database
            embedding_model: Optional sentence-transformers model for semantic expansion
        """
        self.wordnet_path = wordnet_path
        self.embedding_model = embedding_model
        self._load_wordnet()
    
    def _load_wordnet(self):
        """Load WordNet if available"""
        try:
            from nltk.corpus import wordnet as wn
            self._wordnet_available = True
        except ImportError:
            self._wordnet_available = False
            logger.info("NLTK WordNet not available, using built-in synonyms")
    
    def expand(self, query: str, max_expansions: int = 5) -> Tuple[List[str], List[str]]:
        """
        Expand query with synonyms
        
        Args:
            query: Original query
            max_expansions: Maximum number of expansions
            
        Returns:
            (expanded_queries, synonyms_used)
        """
        expansions = []
        synonyms_used = []
        
        query_lower = query.lower()
        words = re.findall(r'\b\w+\b', query_lower)
        
        # Apply built-in synonyms
        for word in words:
            for synonym_set in self.SYNONYMS.values():
                if word in [s.lower() for s in synonym_set]:
                    for syn in synonym_set:
                        if syn.lower() != word and len(synonyms_used) < max_expansions:
                            expanded = query.replace(word, syn, 1)
                            if expanded not in expansions:
                                expansions.append(expanded)
                            synonyms_used.append(f"{word} -> {syn}")
                    break
        
        # Add WordNet synonyms if available
        if self._wordnet_available:
            wordnet_synonyms = self._get_wordnet_synonyms(query_lower)
            for syn in wordnet_synonyms[:max_expansions - len(synonyms_used)]:
                if syn not in expansions:
                    expansions.append(syn)
                    synonyms_used.append(f"wordnet: {syn}")
        
        return expansions, synonyms_used
    
    def _get_wordnet_synonyms(self, query: str) -> List[str]:
        """Get synonyms from WordNet"""
        if not self._wordnet_available:
            return []
        
        try:
            from nltk.corpus import wordnet as wn
            
            synonyms = set()
            words = re.findall(r'\b\w+\b', query)
            
            for word in words:
                for syn in wn.synsets(word):
                    for lemma in syn.lemmas():
                        lemma_name = lemma.name().replace('_', ' ')
                        if lemma_name.lower() != word.lower():
                            synonyms.add(lemma_name)
            
            return list(synonyms)[:10]
            
        except Exception as e:
            logger.warning(f"WordNet error: {e}")
            return []


class BooleanQueryParser:
    """
    Parses boolean query expressions
    """
    
    OPERATOR_WORDS = {
        "AND": ["and", "&", "+", "with"],
        "OR": ["or", "|", "/"],
        "NOT": ["not", "without", "-", "except", "but not"],
    }
    
    def parse(self, query: str) -> List[QueryComponent]:
        """
        Parse query into components with boolean operators
        
        Args:
            query: User query string
            
        Returns:
            List of query components
        """
        components = []
        
        # Normalize query
        normalized = query.strip()
        
        # Split by operators while keeping operator info
        tokens = self._tokenize(normalized)
        
        current_component = QueryComponent(
            text="",
            operator=None,
            is_negated=False,
            weight=1.0,
            entities=[],
            temporal=None,
        )
        
        for token, token_type in tokens:
            if token_type == "OPERATOR":
                # Save current component
                if current_component.text.strip():
                    components.append(current_component)
                
                # Start new component
                op = self._get_operator(token)
                current_component = QueryComponent(
                    text="",
                    operator=op,
                    is_negated=op == QueryOperator.NOT,
                    weight=0.5 if op == QueryOperator.NOT else 1.0,
                    entities=[],
                    temporal=None,
                )
            else:
                current_component.text += token + " "
        
        # Add final component
        if current_component.text.strip():
            components.append(current_component)
        
        # If no operators found, create single component
        if not components:
            components.append(QueryComponent(
                text=normalized,
                operator=None,
                is_negated=False,
                weight=1.0,
                entities=[],
                temporal=None,
            ))
        
        return components
    
    def _tokenize(self, query: str) -> List[Tuple[str, str]]:
        """Tokenize query into words and operators"""
        tokens = []
        current_token = ""
        current_type = "WORD"
        
        i = 0
        while i < len(query):
            char = query[i]
            
            if char.isspace():
                if current_token:
                    tokens.append((current_token, current_type))
                    current_token = ""
                    current_type = "WORD"
                i += 1
                continue
            
            # Check for operators
            if char in "-&+|,":
                if current_token:
                    tokens.append((current_token, current_type))
                    current_token = ""
                
                # Handle multi-char operators
                if query[i:i+3] == "AND" or query[i:i+3] == "NOT":
                    tokens.append((query[i:i+3], "OPERATOR"))
                    i += 3
                    continue
                elif query[i:i+2] == "OR":
                    tokens.append((query[i:i+2], "OPERATOR"))
                    i += 2
                    continue
                
                tokens.append((char, "OPERATOR"))
                i += 1
                continue
            
            current_token += char
            i += 1
        
        if current_token:
            tokens.append((current_token, current_type))
        
        return tokens
    
    def _get_operator(self, token: str) -> QueryOperator:
        """Get operator from token"""
        token_lower = token.lower()
        
        for op, words in self.OPERATOR_WORDS.items():
            if token_lower in words:
                return QueryOperator(op)
        
        return QueryOperator.AND  # Default
    
    def to_filter_expression(
        self,
        components: List[QueryComponent],
    ) -> Dict[str, Any]:
        """
        Convert parsed components to filter expression for vector DB
        
        Args:
            components: Parsed query components
            
        Returns:
            Filter expression dictionary
        """
        must_conditions = []
        should_conditions = []
        must_not_conditions = []
        
        for component in components:
            text_filter = {"text": component.text}
            
            if component.is_negated:
                must_not_conditions.append(text_filter)
            elif component.operator == QueryOperator.OR:
                should_conditions.append(text_filter)
            else:  # AND is default
                must_conditions.append(text_filter)
        
        filter_expr = {}
        
        if must_conditions:
            filter_expr["must"] = must_conditions
        if should_conditions:
            filter_expr["should"] = should_conditions
        if must_not_conditions:
            filter_expr["must_not"] = must_not_conditions
        
        return filter_expr if filter_expr else {"should": [{"text": components[0].text if components else ""}]}


class QueryUnderstandingPipeline:
    """
    Complete query understanding pipeline
    """
    
    def __init__(
        self,
        embedding_model=None,
        use_semantic_classification: bool = False,
    ):
        """
        Initialize pipeline
        
        Args:
            embedding_model: Optional sentence-transformers model
            use_semantic_classification: Whether to use embeddings for classification
        """
        self.classifier = QueryIntentClassifier(embedding_model if use_semantic_classification else None)
        self.entity_extractor = EntityExtractor()
        self.temporal_parser = TemporalQueryParser()
        self.expander = QueryExpander(embedding_model=embedding_model)
        self.boolean_parser = BooleanQueryParser()
    
    def process(self, query: str) -> QueryRewrite:
        """
        Process a query through the full pipeline
        
        Args:
            query: User query string
            
        Returns:
            QueryRewrite with processed query information
        """
        # Step 1: Extract entities
        entities = self.entity_extractor.extract_entities(query)
        
        # Step 2: Parse temporal constraints
        temporal_constraints = self.temporal_parser.parse(query)
        
        # Step 3: Classify intent
        intent, confidence = self.classifier.classify(query)
        
        # Step 4: Parse boolean structure
        components = self.boolean_parser.parse(query)
        
        # Step 5: Expand query
        expansions, synonyms_used = self.expander.expand(query)
        
        # Build rewritten query
        rewritten = self._build_rewritten_query(query, expansions, intent)
        
        # Suggest top_k based on intent
        suggested_top_k = self._suggest_top_k(intent)
        
        return QueryRewrite(
            original_query=query,
            rewritten_query=rewritten,
            intent=intent,
            components=components,
            entities=entities,
            temporal_constraints=temporal_constraints,
            synonyms_added=synonyms_used,
            confidence=confidence,
            suggested_top_k=suggested_top_k,
            metadata={
                "expanded_queries": expansions,
                "requires_reasoning": intent in [QueryIntent.CAUSAL, QueryIntent.MULTI_HOP],
                "requires_temporal": len(temporal_constraints) > 0,
                "requires_comparison": intent == QueryIntent.COMPARISON,
            },
        )
    
    def _build_rewritten_query(
        self,
        original: str,
        expansions: List[str],
        intent: QueryIntent,
    ) -> str:
        """Build rewritten query string"""
        # Start with original
        rewritten = original
        
        # Add expansions if available
        if expansions:
            # Combine with OR for semantic expansion
            expanded = " OR ".join(expansions[:3])
            rewritten = f"({original}) OR ({expanded})"
        
        return rewritten
    
    def _suggest_top_k(self, intent: QueryIntent) -> int:
        """Suggest number of results based on intent"""
        suggestions = {
            QueryIntent.FACT_RETRIEVAL: 5,
            QueryIntent.EXPLANATION: 3,
            QueryIntent.COMPARISON: 5,
            QueryIntent.CAUSAL: 10,
            QueryIntent.MULTI_HOP: 10,
            QueryIntent.SUMMARY: 3,
            QueryIntent.TEMPORAL: 10,
            QueryIntent.DEFINITION: 3,
            QueryIntent.LIST: 10,
            QueryIntent.PROCEDURAL: 5,
            QueryIntent.UNKNOWN: 5,
        }
        return suggestions.get(intent, 5)


class QueryCache:
    """
    Caches query understanding results
    """
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """
        Initialize cache
        
        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Tuple[QueryRewrite, datetime]] = {}
    
    def get(self, query: str) -> Optional[QueryRewrite]:
        """Get cached result for query"""
        key = self._get_key(query)
        
        if key in self._cache:
            rewrite, timestamp = self._cache[key]
            age = (datetime.utcnow() - timestamp).total_seconds()
            
            if age < self.ttl_seconds:
                return rewrite
            else:
                del self._cache[key]
        
        return None
    
    def set(self, query: str, rewrite: QueryRewrite):
        """Cache query result"""
        key = self._get_key(query)
        
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size:
            oldest_key = min(
                self._cache.keys(),
                key=lambda k: self._cache[k][1]
            )
            del self._cache[oldest_key]
        
        self._cache[key] = (rewrite, datetime.utcnow())
    
    def _get_key(self, query: str) -> str:
        """Generate cache key for query"""
        # Normalize query for caching
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def clear(self):
        """Clear cache"""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = len(self._cache)
        if total == 0:
            return {"size": 0, "hit_rate": 0.0, "hits": 0, "misses": 0}
        
        hits = sum(1 for _, timestamp in self._cache.values())
        # Note: misses would need separate tracking in production
        return {
            "size": total,
            "hit_rate": 0.0,  # Would need proper tracking
            "hits": hits,
            "misses": 0,
        }


# Convenience function
def understand_query(
    query: str,
    use_cache: bool = True,
    embedding_model=None,
) -> QueryRewrite:
    """
    Process a query through the understanding pipeline
    
    Args:
        query: User query string
        use_cache: Whether to use query caching
        embedding_model: Optional embedding model
        
    Returns:
        QueryRewrite with processed query information
    """
    # This is a placeholder for easy use
    # In production, you would instantiate the pipeline once
    pipeline = QueryUnderstandingPipeline(embedding_model=embedding_model)
    return pipeline.process(query)
