#!/usr/bin/env python3
"""
Causal Reasoning Support
=========================

Implements genuine causal reasoning capabilities:
- Distinguishes correlation from causation
- Supports counterfactual queries
- Identifies confounding variables
- Generates causal explanations
- Integrates with causal inference models

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
import logging

logger = logging.getLogger(__name__)


class CausalRelationType(Enum):
    """Types of causal relations"""
    DIRECT_CAUSE = "direct_cause"
    INDIRECT_CAUSE = "indirect_cause"
    ENABLING = "enabling"
    PREVENTING = "preventing"
    CONTRIBUTING = "contributing"
    CORRELATION = "correlation"  # Non-causal
    CONFOUNDING = "confounding"


class CausalQueryType(Enum):
    """Types of causal queries"""
    WHY = "why"  # What caused X?
    HOW = "how"  # How did X cause Y?
    WHAT_IF = "what_if"  # Counterfactual
    PREVENT = "prevent"  # How to prevent X?
    INCREASE = "increase"  # How to increase X?
    CONFOUNDERS = "confounders"  # What confounds X and Y?


@dataclass
class CausalNode:
    """Node in causal graph"""
    id: str
    name: str
    node_type: str  # event, factor, outcome
    properties: Dict[str, Any]
    temporal_bounds: Optional[Tuple[datetime, datetime]]
    created_at: datetime


@dataclass
class CausalEdge:
    """Edge in causal graph"""
    id: str
    source_id: str
    target_id: str
    relation_type: CausalRelationType
    strength: float  # 0-1
    confidence: float  # 0-1
    evidence: List[str]
    temporal_direction: str  # forward, backward, simultaneous
    metadata: Dict[str, Any]
    created_at: datetime


@dataclass
class CausalPath:
    """Path through causal graph"""
    nodes: List[CausalNode]
    edges: List[CausalEdge]
    total_strength: float
    total_confidence: float
    path_type: str  # direct, indirect, mediated


@dataclass
class CounterfactualScenario:
    """Counterfactual what-if scenario"""
    id: str
    description: str
    original_outcome: str
    hypothetical_outcome: str
    intervention: Dict[str, Any]
    probability_change: float
    assumptions: List[str]
    confidence: float


@dataclass
class CausalExplanation:
    """Complete causal explanation"""
    explanation_id: str
    query: str
    query_type: CausalQueryType
    direct_causes: List[CausalPath]
    indirect_causes: List[CausalPath]
    confounding_factors: List[CausalNode]
    counterfactuals: List[CounterfactualScenario]
    causal_mechanism: str
    confidence: float
    evidence_summary: str
    limitations: List[str]
    created_at: datetime


class CausalPatternExtractor:
    """
    Extracts causal patterns from text
    """
    
    CAUSE_PATTERNS = [
        # Explicit causal verbs
        (r"(\w+) (?:caused|leads? to|resulted in|triggered|induced|produced|generated|brought about) (\w+)", "explicit"),
        (r"(\w+) (?:is|are) (?:the|a) (?:cause|reason|driver|origin|source) of (\w+)", "explicit"),
        (r"(\w+) (?:is|are) responsible for (\w+)", "explicit"),
        (r"because of (\w+), (\w+) (?:occurred|happened|took place)", "explicit"),
        (r"due to (\w+), (\w+) (?:occurred|happened)", "explicit"),
        (r"(\w+) (?:led to|led to the|results in) (\w+)", "explicit"),
        
        # Implicit patterns
        (r"when (\w+) (?:increases?|decreases?|rises?|falls?), (\w+) (?:also|similarly|accordingly) (?:increases|decreases)", "correlational"),
        (r"higher (\w+) (?:is associated with|leads to|results in) higher (\w+)", "correlational"),
        (r"after (\w+), (\w+) typically (?:occurs|happens|follows)", "temporal"),
        (r"without (\w+), (\w+) cannot (?:happen|occur|exist)", "necessary"),
        (r"if (\w+) then (\w+)", "conditional"),
    ]
    
    COUNTERFACTUAL_PATTERNS = [
        (r"what if (\w+) (?:had not|did not|never) (\w+)", "absence"),
        (r"if (\w+) (?:had|were) (?:different|changed), (\w+) would (?:have been|be)", "intervention"),
        (r"but for (\w+), (\w+) would not have (\w+)", "necessity"),
        (r"suppose (\w+) (?:didn't|did not) (\w+)", "hypothetical"),
        (r"imagine a scenario where (\w+) (\w+)", "imagination"),
    ]
    
    def __init__(self):
        """Initialize pattern extractor"""
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self._cause_patterns = [
            (re.compile(p, re.IGNORECASE), t)
            for p, t in self.CAUSE_PATTERNS
        ]
        self._counterfactual_patterns = [
            (re.compile(p, re.IGNORECASE), t)
            for p, t in self.COUNTERFACTUAL_PATTERNS
        ]
    
    def extract_causal_relations(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract causal relations from text
        
        Args:
            text: Input text
            
        Returns:
            List of extracted causal relations
        """
        relations = []
        
        for pattern, relation_type in self._cause_patterns:
            for match in pattern.finditer(text):
                cause = match.group(1).strip()
                effect = match.group(2).strip()
                
                # Determine if causal or correlational
                is_causal = relation_type == "explicit"
                
                relations.append({
                    "cause": cause,
                    "effect": effect,
                    "relation_type": "causal" if is_causal else "correlational",
                    "confidence": 0.9 if is_causal else 0.6,
                    "pattern_type": relation_type,
                    "match_text": match.group(0),
                    "position": match.span(),
                })
        
        return relations
    
    def extract_counterfactuals(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract counterfactual patterns from text
        
        Args:
            text: Input text
            
        Returns:
            List of counterfactual patterns
        """
        counterfactuals = []
        
        for pattern, cf_type in self._counterfactual_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                
                if len(groups) >= 2:
                    counterfactuals.append({
                        "intervention": groups[0],
                        "outcome": groups[1] if len(groups) > 1 else None,
                        "counterfactual_type": cf_type,
                        "confidence": 0.8,
                        "match_text": match.group(0),
                        "position": match.span(),
                    })
        
        return counterfactuals
    
    def classify_causal_query(self, query: str) -> Tuple[CausalQueryType, float]:
        """
        Classify the type of causal query
        
        Args:
            query: User query
            
        Returns:
            (query_type, confidence)
        """
        query_lower = query.lower()
        
        # WHY queries
        if any(w in query_lower for w in ["why", "reason", "cause", "what caused", "what made"]):
            return CausalQueryType.WHY, 0.9
        
        # HOW queries
        if any(w in query_lower for w in ["how did", "how does", "how can", "mechanism"]):
            return CausalQueryType.HOW, 0.85
        
        # WHAT IF queries
        if any(w in query_lower for w in ["what if", "suppose", "imagine", "if then", "counterfactual"]):
            return CausalQueryType.WHAT_IF, 0.9
        
        # PREVENT queries
        if any(w in query_lower for w in ["prevent", "avoid", "stop", "block", "inhibit"]):
            return CausalQueryType.PREVENT, 0.85
        
        # INCREASE queries
        if any(w in query_lower for w in ["increase", "boost", "enhance", "improve", "promote"]):
            return CausalQueryType.INCREASE, 0.85
        
        # CONFOUNDERS queries
        if any(w in query_lower for w in ["confound", "factor", "variable", "affect"]):
            return CausalQueryType.CONFOUNDERS, 0.8
        
        return CausalQueryType.WHY, 0.5  # Default


class CausalGraphBuilder:
    """
    Builds and maintains causal graphs
    """
    
    def __init__(self):
        """Initialize graph builder"""
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: Dict[str, CausalEdge] = {}
        self._node_index: Dict[str, str] = {}  # name -> id
    
    def add_node(
        self,
        name: str,
        node_type: str = "event",
        properties: Optional[Dict[str, Any]] = None,
        temporal_bounds: Optional[Tuple[datetime, datetime]] = None,
    ) -> str:
        """Add a node to the causal graph"""
        node_id = self._get_node_id(name)
        
        if node_id in self.nodes:
            # Update existing node
            self.nodes[node_id].properties.update(properties or {})
            return node_id
        
        node = CausalNode(
            id=node_id,
            name=name,
            node_type=node_type,
            properties=properties or {},
            temporal_bounds=temporal_bounds,
            created_at=datetime.utcnow(),
        )
        
        self.nodes[node_id] = node
        self._node_index[name.lower()] = node_id
        
        return node_id
    
    def _get_node_id(self, name: str) -> str:
        """Get or create node ID"""
        name_lower = name.lower()
        if name_lower in self._node_index:
            return self._node_index[name_lower]
        return str(uuid.uuid4())
    
    def add_edge(
        self,
        source_name: str,
        target_name: str,
        relation_type: CausalRelationType,
        strength: float = 0.5,
        confidence: float = 0.5,
        evidence: Optional[List[str]] = None,
        temporal_direction: str = "forward",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Add an edge to the causal graph
        
        Returns:
            Edge ID if successful, None if nodes don't exist
        """
        source_id = self._get_node_id(source_name)
        target_id = self._get_node_id(target_name)
        
        # Ensure nodes exist
        if source_id not in self.nodes:
            self.add_node(source_name)
        if target_id not in self.nodes:
            self.add_node(target_name)
        
        edge_id = str(uuid.uuid4())
        
        edge = CausalEdge(
            id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            confidence=confidence,
            evidence=evidence or [],
            temporal_direction=temporal_direction,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
        )
        
        self.edges[edge_id] = edge
        return edge_id
    
    def build_from_relations(
        self,
        relations: List[Dict[str, Any]],
        min_confidence: float = 0.5,
    ) -> int:
        """
        Build graph from extracted relations
        
        Args:
            relations: List of causal relations
            min_confidence: Minimum confidence threshold
            
        Returns:
            Number of edges added
        """
        edges_added = 0
        
        for rel in relations:
            if rel.get("confidence", 0) < min_confidence:
                continue
            
            # Determine relation type
            if rel.get("relation_type") == "correlational":
                relation = CausalRelationType.CORRELATION
                strength = 0.3
            else:
                relation = CausalRelationType.DIRECT_CAUSE
                strength = 0.7
            
            edge_id = self.add_edge(
                source_name=rel["cause"],
                target_name=rel["effect"],
                relation_type=relation,
                strength=strength * rel.get("confidence", 0.5),
                confidence=rel.get("confidence", 0.5),
                evidence=[rel.get("match_text", "")],
                metadata={"pattern_type": rel.get("pattern_type", "")},
            )
            
            if edge_id:
                edges_added += 1
        
        return edges_added
    
    def find_paths(
        self,
        source_name: str,
        target_name: str,
        max_path_length: int = 5,
    ) -> List[CausalPath]:
        """
        Find causal paths between two nodes
        
        Args:
            source_name: Source node name
            target_name: Target node name
            max_path_length: Maximum path length
            
        Returns:
            List of causal paths
        """
        source_id = self._get_node_id(source_name)
        target_id = self._get_node_id(target_name)
        
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
        
        # BFS to find paths
        paths = []
        queue = deque([(source_id, [source_id], [])])
        visited = {source_id: [(source_id, [])]}
        
        while queue and len(paths) < 100:
            current, node_path, edge_path = queue.popleft()
            
            if current == target_id:
                # Convert to CausalPath
                nodes = [self.nodes[n] for n in node_path]
                edges = [self.edges[e] for e in edge_path]
                
                total_strength = sum(e.strength for e in edges) / max(len(edges), 1)
                total_confidence = sum(e.confidence for e in edges) / max(len(edges), 1)
                
                path_type = "direct" if len(edges) == 1 else "indirect"
                
                paths.append(CausalPath(
                    nodes=nodes,
                    edges=edges,
                    total_strength=total_strength,
                    total_confidence=total_confidence,
                    path_type=path_type,
                ))
                continue
            
            if len(node_path) >= max_path_length:
                continue
            
            # Find outgoing edges
            for edge_id, edge in self.edges.items():
                if edge.source_id == current and edge.target_id not in node_path:
                    new_node_path = node_path + [edge.target_id]
                    new_edge_path = edge_path + [edge_id]
                    
                    # Check visited
                    if edge.target_id in visited:
                        if all(edge.target_id in p for p in visited[edge.target_id]):
                            continue
                    
                    visited.setdefault(edge.target_id, []).append(new_node_path)
                    queue.append((edge.target_id, new_node_path, new_edge_path))
        
        # Sort by strength and confidence
        paths.sort(key=lambda p: (p.total_strength, p.total_confidence), reverse=True)
        
        return paths[:10]  # Return top 10 paths
    
    def identify_confounders(
        self,
        cause_name: str,
        effect_name: str,
    ) -> List[CausalNode]:
        """
        Identify potential confounding variables
        
        Args:
            cause_name: Potential cause
            effect_name: Potential effect
            
        Returns:
            List of potential confounders
        """
        cause_id = self._get_node_id(cause_name)
        effect_id = self._get_node_id(effect_name)
        
        if cause_id not in self.nodes or effect_id not in self.nodes:
            return []
        
        confounders = []
        
        # Find common predecessors
        cause_predecessors = self._get_predecessors(cause_id)
        effect_predecessors = self._get_predecessors(effect_id)
        
        common_predecessors = set(cause_predecessors) & set(effect_predecessors)
        
        for node_id in common_predecessors:
            # Check if it connects to both
            has_edge_to_cause = any(
                e.target_id == cause_id for e in self.edges.values()
            )
            has_edge_to_effect = any(
                e.target_id == effect_id for e in self.edges.values()
            )
            
            if has_edge_to_cause and has_edge_to_effect:
                node = self.nodes.get(node_id)
                if node:
                    confounders.append(node)
        
        return confounders
    
    def _get_predecessors(self, node_id: str) -> Set[str]:
        """Get all predecessors of a node"""
        predecessors = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            
            for edge in self.edges.values():
                if edge.target_id == current and edge.source_id not in predecessors:
                    predecessors.add(edge.source_id)
                    queue.append(edge.source_id)
        
        return predecessors
    
    def get_subgraph(self, node_name: str, radius: int = 2) -> Dict[str, Any]:
        """
        Get subgraph around a node
        
        Args:
            node_name: Center node name
            radius: Search radius
            
        Returns:
            Subgraph dictionary
        """
        center_id = self._get_node_id(node_name)
        
        if center_id not in self.nodes:
            return {"nodes": [], "edges": []}
        
        # BFS to find nodes within radius
        visited_nodes = {center_id}
        visited_edges = set()
        queue = deque([(center_id, 0)])
        
        while queue:
            current, dist = queue.popleft()
            
            if dist >= radius:
                continue
            
            for edge_id, edge in self.edges.items():
                if edge.source_id == current and edge.target_id not in visited_nodes:
                    visited_nodes.add(edge.target_id)
                    visited_edges.add(edge_id)
                    queue.append((edge.target_id, dist + 1))
                
                if edge.target_id == current and edge.source_id not in visited_nodes:
                    visited_nodes.add(edge.source_id)
                    visited_edges.add(edge_id)
                    queue.append((edge.source_id, dist + 1))
        
        return {
            "center": node_name,
            "nodes": [self.nodes[n].__dict__ for n in visited_nodes if n in self.nodes],
            "edges": [self.edges[e].__dict__ for e in visited_edges if e in self.edges],
        }
    
    def export_graph(self) -> Dict[str, Any]:
        """Export graph as dictionary"""
        return {
            "nodes": {
                nid: node.__dict__ 
                for nid, node in self.nodes.items()
            },
            "edges": {
                eid: edge.__dict__ 
                for eid, edge in self.edges.items()
            },
        }


class CounterfactualEngine:
    """
    Handles counterfactual reasoning
    """
    
    def __init__(self, causal_graph: CausalGraphBuilder):
        """
        Initialize counterfactual engine
        
        Args:
            causal_graph: Causal graph to use
        """
        self.graph = causal_graph
    
    def analyze_counterfactual(
        self,
        query: str,
        intervention: Dict[str, Any],
        original_outcome: str,
    ) -> CounterfactualScenario:
        """
        Analyze a counterfactual scenario
        
        Args:
            query: Original query
            intervention: What was changed
            original_outcome: What actually happened
            
        Returns:
            CounterfactualScenario with analysis
        """
        # Find paths from intervention to outcome
        paths = self.graph.find_paths(
            list(intervention.keys())[0] if intervention else "",
            original_outcome,
            max_path_length=5,
        )
        
        # Calculate effect
        total_effect = 0.0
        for path in paths:
            total_effect += path.total_strength * path.total_confidence
        
        # Determine probability change
        probability_change = min(total_effect, 1.0)
        
        # Generate assumptions
        assumptions = [
            "Causal graph accurately represents real-world relationships",
            "No unmeasured confounders",
            "Effect is linear and additive",
            "Temporal ordering is correct",
        ]
        
        return CounterfactualScenario(
            id=str(uuid.uuid4()),
            description=query,
            original_outcome=original_outcome,
            hypothetical_outcome=self._predict_outcome(intervention, paths),
            intervention=intervention,
            probability_change=probability_change,
            assumptions=assumptions,
            confidence=min(len(paths) * 0.1 + 0.3, 0.9),
        )
    
    def _predict_outcome(
        self,
        intervention: Dict[str, Any],
        paths: List[CausalPath],
    ) -> str:
        """Predict outcome given intervention"""
        if not paths:
            return "Unable to predict: no causal path found"
        
        # Simple heuristic: combine path effects
        effect = sum(p.total_strength for p in paths) / len(paths)
        
        if effect > 0.7:
            return "Would have been significantly affected"
        elif effect > 0.3:
            return "Would have been moderately affected"
        elif effect > 0:
            return "Would have been slightly affected"
        else:
            return "Would not have been substantially affected"
    
    def generate_counterfactual_queries(
        self,
        outcome: str,
        num_queries: int = 3,
    ) -> List[str]:
        """
        Generate counterfactual what-if queries
        
        Args:
            outcome: Event or outcome to query about
            num_queries: Number of queries to generate
            
        Returns:
            List of counterfactual queries
        """
        queries = [
            f"What if {outcome} had not occurred?",
            f"If {outcome} were different, what would change?",
            f"Suppose {outcome} was prevented - what would happen instead?",
            f"How would things be different if {outcome} never happened?",
            f"Counterfactual: {outcome} did not occur - analyze the effects",
        ]
        
        return queries[:num_queries]


class CausalReasoningEngine:
    """
    Main causal reasoning engine
    """
    
    def __init__(
        self,
        llm_provider=None,
        embedding_model=None,
    ):
        """
        Initialize causal reasoning engine
        
        Args:
            llm_provider: Optional LLM for explanation generation
            embedding_model: Optional embedding model
        """
        self.llm_provider = llm_provider
        self.embedding_model = embedding_model
        
        # Initialize components
        self.pattern_extractor = CausalPatternExtractor()
        self.graph_builder = CausalGraphBuilder()
        self.counterfactual_engine = CounterfactualEngine(self.graph_builder)
    
    def extract_and_build(self, text: str) -> Dict[str, Any]:
        """
        Extract causal relations and build graph
        
        Args:
            text: Input text
            
        Returns:
            Extraction summary
        """
        relations = self.pattern_extractor.extract_causal_relations(text)
        edges_added = self.graph_builder.build_from_relations(relations)
        
        return {
            "relations_extracted": len(relations),
            "causal_relations": [r for r in relations if r.get("relation_type") == "causal"],
            "correlational_relations": [r for r in relations if r.get("relation_type") == "correlational"],
            "edges_added": edges_added,
            "node_count": len(self.graph_builder.nodes),
            "edge_count": len(self.graph_builder.edges),
        }
    
    def explain_causality(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> CausalExplanation:
        """
        Generate causal explanation for a query
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            CausalExplanation
        """
        # Classify query type
        query_type, _ = self.pattern_extractor.classify_causal_query(query)
        
        # Extract entities
        cause, effect = self._extract_cause_effect(query)
        
        # Find causal paths
        direct_causes = []
        indirect_causes = []
        
        if cause and effect:
            all_paths = self.graph_builder.find_paths(cause, effect, max_path_length=5)
            
            for path in all_paths:
                if path.path_type == "direct":
                    direct_causes.append(path)
                else:
                    indirect_causes.append(path)
        
        # Identify confounders
        confounding_factors = []
        if cause and effect:
            confounding_factors = self.graph_builder.identify_confounders(cause, effect)
        
        # Generate counterfactuals
        counterfactuals = []
        if cause and effect:
            cf_query = f"What if {cause} had not caused {effect}?"
            counterfactual = self.counterfactual_engine.analyze_counterfactual(
                cf_query,
                {cause: True},
                effect,
            )
            counterfactuals.append(counterfactual)
        
        # Generate mechanism explanation
        causal_mechanism = self._generate_mechanism(cause, effect, direct_causes)
        
        # Calculate confidence
        confidence = self._calculate_confidence(direct_causes, confounding_factors)
        
        # Generate evidence summary
        evidence_summary = self._summarize_evidence(direct_causes, indirect_causes)
        
        # Identify limitations
        limitations = self._identify_limitations(direct_causes, indirect_causes, confounding_factors)
        
        return CausalExplanation(
            explanation_id=str(uuid.uuid4()),
            query=query,
            query_type=query_type,
            direct_causes=direct_causes[:5],
            indirect_causes=indirect_causes[:5],
            confounding_factors=confounding_factors[:5],
            counterfactuals=counterfactuals,
            causal_mechanism=causal_mechanism,
            confidence=confidence,
            evidence_summary=evidence_summary,
            limitations=limitations,
            created_at=datetime.utcnow(),
        )
    
    def _extract_cause_effect(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract cause and effect from query"""
        query_lower = query.lower()
        
        # Try patterns
        patterns = [
            (r"what (?:caused|causes) (\w+)", None, "effect"),
            (r"why did (\w+) (?:happen|occur)", None, "effect"),
            (r"what (?:is the |are the )?(\w+) (?:caused by|result of|due to)", None, "effect"),
            (r"(\w+) (?:caused|led to|resulted in) (\w+)", "cause", "effect"),
            (r"(\w+) (?:is the )?cause of (\w+)", "cause", "effect"),
        ]
        
        for pattern, ca, ef in patterns:
            match = re.search(pattern, query_lower)
            if match:
                groups = match.groups()
                
                cause = groups[0] if ca else None
                effect = groups[1] if ef else (groups[0] if ca is None else None)
                
                return cause, effect
        
        return None, None
    
    def _generate_mechanism(
        self,
        cause: Optional[str],
        effect: Optional[str],
        paths: List[CausalPath],
    ) -> str:
        """Generate causal mechanism description"""
        if not paths:
            return "No causal mechanism identified."
        
        mechanism_parts = []
        
        if cause and effect:
            mechanism_parts.append(f"{cause} affects {effect} through:")
            
            for i, path in enumerate(paths[:3], 1):
                if path.nodes:
                    intermediate = " → ".join(n.name for n in path.nodes[1:-1])
                    mechanism_parts.append(
                        f"{i}. Direct path" if not intermediate else
                        f"{i}. {intermediate}"
                    )
        
        return "\n".join(mechanism_parts) if mechanism_parts else "Insufficient evidence for causal mechanism."
    
    def _calculate_confidence(
        self,
        direct: List[CausalPath],
        confounders: List[Any],
    ) -> float:
        """Calculate confidence in causal explanation"""
        base_confidence = 0.5
        
        # Increase with more evidence
        base_confidence += min(len(direct) * 0.1, 0.3)
        
        # Decrease with confounders
        base_confidence -= min(len(confounders) * 0.05, 0.2)
        
        return min(max(base_confidence, 0.1), 0.95)
    
    def _summarize_evidence(
        self,
        direct: List[CausalPath],
        indirect: List[CausalPath],
    ) -> str:
        """Summarize evidence for causal claim"""
        parts = []
        
        if direct:
            parts.append(f"{len(direct)} direct causal path(s) identified")
        if indirect:
            parts.append(f"{len(indirect)} indirect path(s) identified")
        
        return "; ".join(parts) if parts else "Limited evidence available"
    
    def _identify_limitations(
        self,
        direct: List[CausalPath],
        indirect: List[CausalPath],
        confounders: List[Any],
    ) -> List[str]:
        """Identify limitations of causal analysis"""
        limitations = []
        
        if not direct and not indirect:
            limitations.append("No causal paths identified in knowledge base")
        
        if len(direct) < 2:
            limitations.append("Limited direct evidence for causal relationship")
        
        if confounders:
            limitations.append(f"Potential confounders identified: {len(confounders)}")
            limitations.append("Confounding variables may affect causal interpretation")
        
        limitations.append("Temporal precedence not verified")
        limitations.append("Correlation may not imply causation")
        
        return limitations
    
    def query_counterfactual(
        self,
        query: str,
    ) -> List[CounterfactualScenario]:
        """
        Answer counterfactual queries
        
        Args:
            query: What-if query
            
        Returns:
            List of counterfactual scenarios
        """
        # Extract intervention and outcome
        intervention_match = re.search(r"what if (\w+) (?:did not|had not) (\w+)", query.lower())
        
        if not intervention_match:
            return []
        
        intervention = {
            intervention_match.group(1): False,
            "negated": intervention_match.group(2),
        }
        outcome = intervention_match.group(2)
        
        return [
            self.counterfactual_engine.analyze_counterfactual(
                query,
                intervention,
                outcome,
            )
        ]
    
    def get_causal_subgraph(self, entity: str) -> Dict[str, Any]:
        """Get causal subgraph around an entity"""
        return self.graph_builder.get_subgraph(entity, radius=2)
    
    def export_causal_graph(self) -> Dict[str, Any]:
        """Export the causal graph"""
        return self.graph_builder.export_graph()
