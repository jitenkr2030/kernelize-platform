#!/usr/bin/env python3
"""
Multi-Hop Reasoning Engine
===========================

Enables answering complex questions that require reasoning across multiple
knowledge kernels. Implements question decomposition, evidence chaining,
confidence aggregation, and explanation generation.

Author: MiniMax Agent
"""

import json
import re
import uuid
import time
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, deque
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    GRAPH_TRAVERSAL = "graph_traversal"
    TEMPORAL_REASONING = "temporal_reasoning"
    CAUSAL_REASONING = "causal_reasoning"


class EvidenceSource(Enum):
    """Sources of evidence"""
    VECTOR_SEARCH = "vector_search"
    EXACT_MATCH = "exact_match"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    TEMPORAL_FILTER = "temporal_filter"
    CROSS_DOCUMENT = "cross_document"


@dataclass
class SubQuestion:
    """A decomposed sub-question"""
    id: str
    question: str
    parent_id: Optional[str]
    position: int
    dependencies: List[str]  # IDs of questions this depends on
    context: str  # Context from previous questions
    expected_answer_type: str
    priority: float


@dataclass
class Evidence:
    """Evidence supporting an answer"""
    id: str
    source: EvidenceSource
    chunk_id: str
    kernel_id: str
    content: str
    relevance_score: float
    confidence_score: float
    supporting_score: float
    metadata: Dict[str, Any]
    retrieved_at: datetime


@dataclass
class ReasoningStep:
    """A step in the reasoning chain"""
    id: str
    step_type: ReasoningType
    question: str
    answer: str
    evidence_used: List[str]
    confidence: float
    reasoning_path: List[str]
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """Complete reasoning trace for explanation"""
    trace_id: str
    original_query: str
    decomposed_questions: List[SubQuestion]
    reasoning_steps: List[ReasoningStep]
    evidence_collected: List[Evidence]
    final_answer: str
    confidence_score: float
    reasoning_type: ReasoningType
    execution_time_ms: float
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiHopResult:
    """Result from multi-hop reasoning"""
    success: bool
    answer: str
    confidence: float
    trace: Optional[ReasoningTrace]
    sub_answers: Dict[str, str]
    errors: List[str]
    execution_time_ms: float


class QuestionDecomposer:
    """
    Decomposes complex questions into answerable sub-questions
    """
    
    DECOMPOSITION_PATTERNS = [
        # Entity-based decomposition
        (
            r"(?:who|what|which) (?:is|are) the (.+?) of (?:the )?(.+)",
            lambda m: [
                f"Who or what is {m.group(2)}?",
                f"What is the {m.group(1)} of {m.group(2)}?",
            ]
        ),
        # Location-based decomposition
        (
            r"(?:who|what) (.+) in (.+)",
            lambda m: [
                f"What or who is {m.group(1)}?",
                f"Information about {m.group(1)} in {m.group(2)}?",
            ]
        ),
        # Temporal decomposition
        (
            r"(?:how|what) (.+) since (.+)",
            lambda m: [
                f"What is {m.group(1)}?",
                f"Changes since {m.group(2)}?",
            ]
        ),
        # Comparative decomposition
        (
            r"(?:what|how) (.+) (?:compare|differs) between (.+) and (.+)",
            lambda m: [
                f"Information about {m.group(2)}?",
                f"Information about {m.group(3)}?",
                f"Comparison between {m.group(2)} and {m.group(3)} regarding {m.group(1)}?",
            ]
        ),
        # Causal decomposition
        (
            r"(?:why|what caused) (.+) to (.+)",
            lambda m: [
                f"What caused {m.group(1)} to {m.group(2)}?",
                f"Factors influencing {m.group(1)} {m.group(2)}?",
            ]
        ),
    ]
    
    # Question templates for common patterns
    TEMPLATES = {
        "definition": [
            "What is {entity}?",
            "Define {entity}",
            "Explain what {entity} means",
        ],
        "relationship": [
            "What is the relationship between {entity1} and {entity2}?",
            "How are {entity1} and {entity2} connected?",
        ],
        "process": [
            "What is the process of {process}?",
            "How does {process} work?",
            "What are the steps in {process}?",
        ],
        "comparison": [
            "What are the differences between {entity1} and {entity2}?",
            "Compare {entity1} with {entity2}",
        ],
        "temporal": [
            "What happened to {entity} during {time_period}?",
            "What changed about {entity} after {event}?",
        ],
    }
    
    def __init__(self, embedding_model=None):
        """
        Initialize decomposer
        
        Args:
            embedding_model: Optional embedding model for semantic similarity
        """
        self.embedding_model = embedding_model
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self._patterns = [
            (re.compile(p, re.IGNORECASE), f) 
            for p, f in self.DECOMPOSITION_PATTERNS
        ]
    
    def decompose(self, query: str) -> List[SubQuestion]:
        """
        Decompose a complex query into sub-questions
        
        Args:
            query: Original complex query
            
        Returns:
            List of sub-questions with dependencies
        """
        sub_questions = []
        
        # Try pattern-based decomposition
        for pattern, extract_func in self._patterns:
            match = pattern.search(query)
            if match:
                sub_qs = extract_func(match)
                for i, sq in enumerate(sub_qs):
                    sub_questions.append(SubQuestion(
                        id=str(uuid.uuid4()),
                        question=sq,
                        parent_id=None,
                        position=i,
                        dependencies=[] if i == 0 else [sub_questions[i-1].id if sub_questions else None],
                        context="",
                        expected_answer_type="text",
                        priority=1.0 - (i * 0.1),
                    ))
                return sub_questions
        
        # Use LLM-based decomposition if available
        if self.embedding_model:
            return self._semantic_decomposition(query)
        
        # Default: return query as single question
        return [SubQuestion(
            id=str(uuid.uuid4()),
            question=query,
            parent_id=None,
            position=0,
            dependencies=[],
            context="",
            expected_answer_type="text",
            priority=1.0,
        )]
    
    def _semantic_decomposition(self, query: str) -> List[SubQuestion]:
        """
        Use embeddings for semantic decomposition
        
        This is a placeholder - actual implementation would use an LLM
        """
        # For now, use simple heuristics
        if " and " in query.lower():
            parts = query.lower().split(" and ")
            return self._decompose_conjunctive(query, parts)
        elif " or " in query.lower():
            parts = query.lower().split(" or ")
            return self._decompose_disjunctive(query, parts)
        
        return [SubQuestion(
            id=str(uuid.uuid4()),
            question=query,
            parent_id=None,
            position=0,
            dependencies=[],
            context="",
            expected_answer_type="text",
            priority=1.0,
        )]
    
    def _decompose_conjunctive(self, query: str, parts: List[str]) -> List[SubQuestion]:
        """Decompose 'and' questions"""
        sub_questions = []
        
        for i, part in enumerate(parts):
            # Find the entity/thing being discussed
            entity_match = re.search(r"(?:the )?(\w+(?:\s+\w+)?)", part.strip())
            entity = entity_match.group(1) if entity_match else "it"
            
            sub_questions.append(SubQuestion(
                id=str(uuid.uuid4()),
                question=f"Information about {entity}",
                parent_id=None,
                position=i,
                dependencies=[] if i == 0 else [sub_questions[i-1]["id"]],
                context="",
                expected_answer_type="text",
                priority=1.0 - (i * 0.1),
            ))
        
        # Add synthesis question
        sub_questions.append(SubQuestion(
            id=str(uuid.uuid4()),
            question=f"Synthesize information about: {query}",
            parent_id=None,
            position=len(sub_questions),
            dependencies=[sq["id"] for sq in sub_questions],
            context="Combined information from previous questions",
            expected_answer_type="synthesis",
            priority=0.5,
        ))
        
        return sub_questions
    
    def _decompose_disjunctive(self, query: str, parts: List[str]) -> List[SubQuestion]:
        """Decompose 'or' questions"""
        sub_questions = []
        
        for i, part in enumerate(parts):
            sub_questions.append(SubQuestion(
                id=str(uuid.uuid4()),
                question=f"Information about: {part.strip()}",
                parent_id=None,
                position=i,
                dependencies=[],
                context="",
                expected_answer_type="text",
                priority=1.0 / (i + 1),
            ))
        
        return sub_questions


class EvidenceCollector:
    """
    Collects evidence from various sources
    """
    
    def __init__(
        self,
        vector_store,
        knowledge_graph,
        exact_matcher,
    ):
        """
        Initialize collector
        
        Args:
            vector_store: Vector database for semantic search
            knowledge_graph: Knowledge graph for relationship queries
            exact_matcher: Keyword matcher for exact search
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.exact_matcher = exact_matcher
    
    def collect_evidence(
        self,
        query: str,
        kernel_ids: Optional[List[str]] = None,
        temporal_filter: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
    ) -> List[Evidence]:
        """
        Collect evidence for a query from multiple sources
        
        Args:
            query: Search query
            kernel_ids: Optional list of kernel IDs to search
            temporal_filter: Optional temporal constraints
            top_k: Number of results per source
            
        Returns:
            List of evidence with scores
        """
        evidence = []
        
        # 1. Vector search
        vector_evidence = self._collect_vector_evidence(query, top_k)
        evidence.extend(vector_evidence)
        
        # 2. Exact match
        exact_evidence = self._collect_exact_evidence(query, top_k)
        evidence.extend(exact_evidence)
        
        # 3. Knowledge graph (if entities found)
        kg_evidence = self._collect_graph_evidence(query, top_k)
        evidence.extend(kg_evidence)
        
        # 4. Temporal filtering
        if temporal_filter:
            evidence = self._apply_temporal_filter(evidence, temporal_filter)
        
        # Deduplicate and rank
        evidence = self._deduplicate_evidence(evidence)
        evidence = self._rank_evidence(evidence)
        
        return evidence[:top_k]
    
    def _collect_vector_evidence(self, query: str, top_k: int) -> List[Evidence]:
        """Collect evidence via vector search"""
        try:
            results = self.vector_store.search(query_vector=[], top_k=top_k)
            
            return [
                Evidence(
                    id=str(uuid.uuid4()),
                    source=EvidenceSource.VECTOR_SEARCH,
                    chunk_id=hit.id if hasattr(hit, 'id') else str(hit),
                    kernel_id="",
                    content=hit.payload.get("content", "") if hasattr(hit, 'payload') else "",
                    relevance_score=hit.score if hasattr(hit, 'score') else 0.5,
                    confidence_score=0.7,
                    supporting_score=0.8,
                    metadata={"type": "vector_search"},
                    retrieved_at=datetime.utcnow(),
                )
                for hit in results
            ]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []
    
    def _collect_exact_evidence(self, query: str, top_k: int) -> List[Evidence]:
        """Collect evidence via exact match"""
        try:
            matched_ids = self.exact_matcher.exact_search(query)
            
            return [
                Evidence(
                    id=str(uuid.uuid4()),
                    source=EvidenceSource.EXACT_MATCH,
                    chunk_id=mid,
                    kernel_id="",
                    content="",
                    relevance_score=0.9,
                    confidence_score=0.9,
                    supporting_score=0.9,
                    metadata={"type": "exact_match", "matched_terms": query},
                    retrieved_at=datetime.utcnow(),
                )
                for mid in matched_ids[:top_k]
            ]
        except Exception as e:
            logger.warning(f"Exact match failed: {e}")
            return []
    
    def _collect_graph_evidence(self, query: str, top_k: int) -> List[Evidence]:
        """Collect evidence via knowledge graph"""
        # Placeholder - would extract entities and query relationships
        return []
    
    def _apply_temporal_filter(
        self,
        evidence: List[Evidence],
        temporal_filter: Dict[str, Any],
    ) -> List[Evidence]:
        """Apply temporal constraints to evidence"""
        if "start_date" not in temporal_filter and "end_date" not in temporal_filter:
            return evidence
        
        start_date = temporal_filter.get("start_date")
        end_date = temporal_filter.get("end_date")
        
        filtered = []
        for ev in evidence:
            ev_date = ev.metadata.get("date")
            if ev_date:
                if start_date and ev_date < start_date:
                    continue
                if end_date and ev_date > end_date:
                    continue
            filtered.append(ev)
        
        return filtered
    
    def _deduplicate_evidence(self, evidence: List[Evidence]) -> List[Evidence]:
        """Remove duplicate evidence"""
        seen = set()
        unique = []
        
        for ev in evidence:
            key = (ev.chunk_id, ev.source)
            if key not in seen:
                seen.add(key)
                unique.append(ev)
        
        return unique
    
    def _rank_evidence(self, evidence: List[Evidence]) -> List[Evidence]:
        """Rank evidence by combined score"""
        for ev in evidence:
            ev.confidence_score = (
                ev.relevance_score * 0.4 +
                ev.confidence_score * 0.3 +
                ev.supporting_score * 0.3
            )
        
        return sorted(evidence, key=lambda e: e.confidence_score, reverse=True)


class ReasoningChainExecutor:
    """
    Executes reasoning chains across sub-questions
    """
    
    def __init__(
        self,
        evidence_collector: EvidenceCollector,
        llm_provider=None,
    ):
        """
        Initialize executor
        
        Args:
            evidence_collector: Evidence collector instance
            llm_provider: Optional LLM for synthesis
        """
        self.evidence_collector = evidence_collector
        self.llm_provider = llm_provider
    
    def execute_chain(
        self,
        sub_questions: List[SubQuestion],
        reasoning_type: ReasoningType = ReasoningType.CHAIN_OF_THOUGHT,
    ) -> Tuple[List[ReasoningStep], Dict[str, str]]:
        """
        Execute reasoning chain for sub-questions
        
        Args:
            sub_questions: List of decomposed questions
            reasoning_type: Type of reasoning to use
            
        Returns:
            (reasoning_steps, sub_answers)
        """
        steps = []
        sub_answers = {}
        
        # Sort by dependencies (topological order)
        sorted_questions = self._topological_sort(sub_questions)
        
        for question in sorted_questions:
            # Collect evidence for this question
            evidence = self.evidence_collector.collect_evidence(
                question.question,
                temporal_filter=None,
                top_k=5,
            )
            
            # Generate answer
            answer = self._generate_answer(question, evidence)
            sub_answers[question.id] = answer
            
            # Record reasoning step
            step = ReasoningStep(
                id=str(uuid.uuid4()),
                step_type=reasoning_type,
                question=question.question,
                answer=answer,
                evidence_used=[e.id for e in evidence],
                confidence=self._calculate_confidence(evidence),
                reasoning_path=[question.id],
                timestamp=datetime.utcnow(),
                metadata={
                    "dependencies": question.dependencies,
                    "context": question.context,
                },
            )
            steps.append(step)
        
        return steps, sub_answers
    
    def _topological_sort(self, questions: List[SubQuestion]) -> List[SubQuestion]:
        """Sort questions by dependency"""
        # Build dependency graph
        in_degree = {q.id: 0 for q in questions}
        graph = {q.id: [] for q in questions}
        
        for q in questions:
            for dep in q.dependencies:
                if dep in graph:
                    graph[dep].append(q.id)
                    in_degree[q.id] = in_degree.get(q.id, 0) + 1
        
        # Kahn's algorithm
        queue = deque([q for q in questions if in_degree[q.id] == 0])
        sorted_qs = []
        
        while queue:
            q = queue.popleft()
            sorted_qs.append(q)
            
            for neighbor in graph[q.id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Add any remaining (shouldn't happen with valid dependencies)
        sorted_qs.extend([q for q in questions if q.id not in in_degree])
        
        return sorted_qs
    
    def _generate_answer(
        self,
        question: SubQuestion,
        evidence: List[Evidence],
    ) -> str:
        """Generate answer from evidence"""
        if not evidence:
            return "No relevant information found."
        
        # Use LLM if available
        if self.llm_provider:
            return self._llm_synthesize(question, evidence)
        
        # Fallback: use top evidence
        best_evidence = evidence[0]
        return best_evidence.content[:500] + "..." if len(best_evidence.content) > 500 else best_evidence.content
    
    def _llm_synthesize(
        self,
        question: SubQuestion,
        evidence: List[Evidence],
    ) -> str:
        """Use LLM to synthesize answer"""
        # Placeholder - would call actual LLM
        context = "\n".join([e.content for e in evidence[:3]])
        return f"Based on the evidence: {context[:200]}..."
    
    def _calculate_confidence(self, evidence: List[Evidence]) -> float:
        """Calculate confidence from evidence"""
        if not evidence:
            return 0.0
        
        avg_confidence = sum(e.confidence_score for e in evidence) / len(evidence)
        
        # Boost confidence if multiple sources agree
        sources = set(e.source for e in evidence)
        source_boost = min(len(sources) * 0.1, 0.3)
        
        return min(avg_confidence + source_boost, 1.0)


class ConfidenceAggregator:
    """
    Aggregates confidence scores from multiple sources
    """
    
    @staticmethod
    def aggregate(
        evidence: List[Evidence],
        sub_answers: Dict[str, str],
        steps: List[ReasoningStep],
    ) -> float:
        """
        Aggregate confidence from evidence and reasoning steps
        
        Args:
            evidence: Collected evidence
            sub_answers: Answers to sub-questions
            steps: Reasoning steps
            
        Returns:
            Aggregated confidence score
        """
        if not evidence and not steps:
            return 0.0
        
        # Weight factors
        evidence_weight = 0.4
        reasoning_weight = 0.4
        coherence_weight = 0.2
        
        # Evidence confidence
        if evidence:
            evidence_confidence = sum(e.confidence_score for e in evidence) / len(evidence)
        else:
            evidence_confidence = 0.0
        
        # Reasoning confidence
        if steps:
            reasoning_confidence = sum(s.confidence for s in steps) / len(steps)
        else:
            reasoning_confidence = 0.0
        
        # Coherence (answer consistency)
        coherence = ConfidenceAggregator._calculate_coherence(sub_answers)
        
        total = (
            evidence_confidence * evidence_weight +
            reasoning_confidence * reasoning_weight +
            coherence * coherence_weight
        )
        
        return min(total, 1.0)
    
    @staticmethod
    def _calculate_coherence(answers: Dict[str, str]) -> float:
        """Calculate coherence of answers"""
        if len(answers) <= 1:
            return 1.0
        
        # Check for contradictions (simplified)
        answer_texts = list(answers.values())
        coherence_scores = []
        
        for i in range(1, len(answer_texts)):
            # Check overlap
            words1 = set(answer_texts[i-1].lower().split())
            words2 = set(answer_texts[i].lower().split())
            overlap = len(words1 & words2) / max(len(words1 | words2), 1)
            coherence_scores.append(overlap)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.5


class ExplanationGenerator:
    """
    Generates human-readable explanations of reasoning
    """
    
    @staticmethod
    def generate_explanation(trace: ReasoningTrace) -> str:
        """
        Generate explanation from reasoning trace
        
        Args:
            trace: Reasoning trace
            
        Returns:
            Human-readable explanation
        """
        if not trace.reasoning_steps:
            return "No reasoning steps recorded."
        
        lines = [
            f"## Reasoning Analysis\n",
            f"**Original Question:** {trace.original_query}\n",
            f"**Reasoning Type:** {trace.reasoning_type.value}\n",
            f"**Confidence:** {trace.confidence_score:.1%}\n",
            f"**Execution Time:** {trace.execution_time_ms:.0f}ms\n\n",
        ]
        
        if trace.decomposed_questions:
            lines.append("### Question Decomposition\n")
            for i, sq in enumerate(trace.decomposed_questions, 1):
                lines.append(f"{i}. {sq.question}\n")
                if sq.dependencies:
                    lines.append(f"   - Depends on: {', '.join(sq.dependencies)}\n")
            lines.append("\n")
        
        lines.append("### Reasoning Steps\n")
        for i, step in enumerate(trace.reasoning_steps, 1):
            lines.append(f"**Step {i}:** {step.step_type.value}\n")
            lines.append(f"- **Question:** {step.question}\n")
            lines.append(f"- **Answer:** {step.answer[:200]}{'...' if len(step.answer) > 200 else ''}\n")
            lines.append(f"- **Confidence:** {step.confidence:.1%}\n")
            lines.append(f"- **Evidence Used:** {len(step.evidence_used)} sources\n")
            lines.append("\n")
        
        lines.append(f"### Final Answer\n")
        lines.append(f"{trace.final_answer}\n")
        
        if trace.evidence_collected:
            lines.append("\n### Evidence Sources\n")
            for i, ev in enumerate(trace.evidence_collected[:5], 1):
                lines.append(f"{i}. [{ev.source.value}] Score: {ev.relevance_score:.2f}\n")
        
        return "".join(lines)
    
    @staticmethod
    def generate_step_explanation(step: ReasoningStep) -> str:
        """Generate explanation for a single step"""
        return f"""
Step: {step.step_type.value}
Question: {step.question}
Answer: {step.answer}
Confidence: {step.confidence:.1%}
Evidence: {len(step.evidence_used)} sources used
""".strip()


class MultiHopReasoningEngine:
    """
    Main multi-hop reasoning engine
    """
    
    def __init__(
        self,
        vector_store,
        knowledge_graph=None,
        embedding_model=None,
        llm_provider=None,
        max_hops: int = 5,
        timeout_seconds: int = 30,
    ):
        """
        Initialize engine
        
        Args:
            vector_store: Vector database
            knowledge_graph: Optional knowledge graph
            embedding_model: Optional embedding model
            llm_provider: Optional LLM for synthesis
            max_hops: Maximum reasoning hops
            timeout_seconds: Timeout for reasoning
        """
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.max_hops = max_hops
        self.timeout_seconds = timeout_seconds
        
        # Initialize components
        self.decomposer = QuestionDecomposer(embedding_model)
        self.evidence_collector = EvidenceCollector(vector_store, knowledge_graph, None)
        self.executor = ReasoningChainExecutor(self.evidence_collector, llm_provider)
        self.aggregator = ConfidenceAggregator()
        self.explainer = ExplanationGenerator()
    
    def reason(
        self,
        query: str,
        kernel_ids: Optional[List[str]] = None,
        temporal_filter: Optional[Dict[str, Any]] = None,
        reasoning_type: ReasoningType = ReasoningType.CHAIN_OF_THOUGHT,
    ) -> MultiHopResult:
        """
        Execute multi-hop reasoning for a query
        
        Args:
            query: Complex query to reason about
            kernel_ids: Optional kernel IDs to search
            temporal_filter: Optional temporal constraints
            reasoning_type: Type of reasoning to use
            
        Returns:
            MultiHopResult with answer and reasoning trace
        """
        start_time = time.time()
        errors = []
        
        try:
            # Step 1: Decompose question
            sub_questions = self.decomposer.decompose(query)
            
            if len(sub_questions) > self.max_hops:
                errors.append(f"Query requires {len(sub_questions)} hops, max is {self.max_hops}")
                return MultiHopResult(
                    success=False,
                    answer="",
                    confidence=0.0,
                    trace=None,
                    sub_answers={},
                    errors=errors,
                    execution_time_ms=(time.time() - start_time) * 1000,
                )
            
            # Step 2: Execute reasoning chain
            steps, sub_answers = self.executor.execute_chain(sub_questions, reasoning_type)
            
            # Step 3: Collect evidence
            all_evidence = []
            for step in steps:
                for evidence_id in step.evidence_used:
                    ev = self._find_evidence(evidence_id)
                    if ev:
                        all_evidence.append(ev)
            
            # Step 4: Synthesize final answer
            final_answer = self._synthesize_answer(query, sub_answers, steps)
            
            # Step 5: Calculate confidence
            confidence = self.aggregator.aggregate(all_evidence, sub_answers, steps)
            
            # Step 6: Create trace
            trace = ReasoningTrace(
                trace_id=str(uuid.uuid4()),
                original_query=query,
                decomposed_questions=sub_questions,
                reasoning_steps=steps,
                evidence_collected=all_evidence,
                final_answer=final_answer,
                confidence_score=confidence,
                reasoning_type=reasoning_type,
                execution_time_ms=(time.time() - start_time) * 1000,
                metadata={
                    "kernel_ids": kernel_ids,
                    "temporal_filter": temporal_filter,
                },
            )
            
            return MultiHopResult(
                success=True,
                answer=final_answer,
                confidence=confidence,
                trace=trace,
                sub_answers=sub_answers,
                errors=[],
                execution_time_ms=(time.time() - start_time) * 1000,
            )
            
        except Exception as e:
            logger.error(f"Reasoning failed: {e}")
            return MultiHopResult(
                success=False,
                answer="",
                confidence=0.0,
                trace=None,
                sub_answers={},
                errors=[str(e)],
                execution_time_ms=(time.time() - start_time) * 1000,
            )
    
    def _find_evidence(self, evidence_id: str) -> Optional[Evidence]:
        """Find evidence by ID (placeholder)"""
        return None
    
    def _synthesize_answer(
        self,
        original_query: str,
        sub_answers: Dict[str, str],
        steps: List[ReasoningStep],
    ) -> str:
        """Synthesize final answer from sub-answers"""
        # Use LLM if available
        if self.llm_provider:
            context = "\n\n".join(
                f"Q: {step.question}\nA: {step.answer}"
                for step in steps
            )
            return self._llm_synthesize_final(original_query, context)
        
        # Fallback: concatenate answers
        return " | ".join(sub_answers.values())
    
    def _llm_synthesize_final(
        self,
        query: str,
        context: str,
    ) -> str:
        """Use LLM to synthesize final answer"""
        # Placeholder - would call actual LLM
        return f"Based on the analysis: {context[:500]}..."
    
    def explain_answer(self, result: MultiHopResult) -> str:
        """Generate explanation for answer"""
        if not result.trace:
            return "No reasoning trace available."
        
        return self.explainer.generate_explanation(result.trace)
