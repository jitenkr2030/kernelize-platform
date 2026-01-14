"""
KERNELIZE Platform - Knowledge Graph Integration
=================================================

Structured knowledge representation enabling sophisticated
reasoning over kernel relationships and content.

Features:
- Automatic knowledge graph extraction from kernels
- Graph neural network reasoning over relationships
- Hybrid search combining text similarity with graph traversal
- Knowledge graph queries spanning multiple kernels

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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set
from collections import defaultdict
import math

logger = logging.getLogger(__name__)


class GraphNodeType(Enum):
    """Types of graph nodes"""
    ENTITY = "entity"
    CONCEPT = "concept"
    DOCUMENT = "document"
    EVENT = "event"
    RELATION = "relation"
    KERNEL = "kernel"
    QUERY = "query"


class GraphEdgeType(Enum):
    """Types of graph edges"""
    CONTAINS = "contains"
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CAUSES = "causes"
    SIMILAR_TO = "similar_to"
    REFERENCES = "references"
    DEPENDS_ON = "depends_on"
    MENTIONS = "mentions"


@dataclass
class GraphNode:
    """Node in knowledge graph"""
    node_id: str
    node_type: str
    label: str
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Embedding for similarity
    embedding: Optional[List[float]] = None
    
    # Metadata
    source_kernel: Optional[str] = None
    confidence: float = 1.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class GraphEdge:
    """Edge in knowledge graph"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    
    # Properties
    properties: Dict[str, Any] = field(default_factory=dict)
    
    # Weight for traversal
    weight: float = 1.0
    directed: bool = True
    
    # Metadata
    confidence: float = 1.0
    source_kernel: Optional[str] = None


@dataclass
class KnowledgeGraph:
    """Knowledge graph structure"""
    graph_id: str
    name: str
    
    # Components
    nodes: Dict[str, GraphNode] = field(default_factory=dict)
    edges: Dict[str, GraphEdge] = field(default_factory=dict)
    
    # Indexes
    node_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    label_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    kernel_index: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source_kernels: List[str] = field(default_factory=list)
    
    def add_node(self, node: GraphNode):
        """Add node to graph"""
        self.nodes[node.node_id] = node
        
        # Update indexes
        self.node_index[node.node_type].add(node.node_id)
        self.label_index[node.label.lower()].add(node.node_id)
        
        if node.source_kernel:
            self.kernel_index[node.source_kernel].add(node.node_id)
            if node.source_kernel not in self.source_kernels:
                self.source_kernels.append(node.source_kernel)
        
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def add_edge(self, edge: GraphEdge):
        """Add edge to graph"""
        self.edges[edge.edge_id] = edge
        
        # Ensure nodes exist
        if edge.source_id not in self.nodes:
            self.nodes[edge.source_id] = GraphNode(
                node_id=edge.source_id,
                node_type="unknown",
                label=edge.source_id
            )
        if edge.target_id not in self.nodes:
            self.nodes[edge.target_id] = GraphNode(
                node_id=edge.target_id,
                node_type="unknown",
                label=edge.target_id
            )
        
        self.updated_at = datetime.now(timezone.utc).isoformat()
    
    def get_nodes_by_type(self, node_type: str) -> List[GraphNode]:
        """Get nodes by type"""
        node_ids = self.node_index.get(node_type, set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_nodes_by_label(self, label: str) -> List[GraphNode]:
        """Get nodes by label (case-insensitive)"""
        node_ids = self.label_index.get(label.lower(), set())
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]
    
    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[str] = None
    ) -> List[Tuple[GraphNode, GraphEdge]]:
        """Get neighboring nodes"""
        neighbors = []
        
        for edge in self.edges.values():
            if edge.source_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    if edge.target_id in self.nodes:
                        neighbors.append((self.nodes[edge.target_id], edge))
            elif not edge.directed and edge.target_id == node_id:
                if edge_type is None or edge.edge_type == edge_type:
                    if edge.source_id in self.nodes:
                        neighbors.append((self.nodes[edge.source_id], edge))
        
        return neighbors
    
    def get_subgraph(
        self,
        center_node_id: str,
        depth: int = 2
    ) -> 'KnowledgeGraph':
        """Get subgraph around node"""
        subgraph = KnowledgeGraph(
            graph_id=f"subgraph_{center_node_id}",
            name=f"Subgraph of {center_node_id}"
        )
        
        # BFS to collect nodes
        visited = set()
        queue = [(center_node_id, 0)]
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_id in visited or current_depth > depth:
                continue
            
            visited.add(current_id)
            
            if current_id in self.nodes:
                subgraph.add_node(self.nodes[current_id])
            
            # Get neighbors
            for neighbor, edge in self.get_neighbors(current_id):
                subgraph.add_node(neighbor)
                subgraph.add_edge(edge)
                
                if neighbor.node_id not in visited:
                    queue.append((neighbor.node_id, current_depth + 1))
        
        return subgraph
    
    def to_dict(self) -> Dict[str, Any]:
        """Export graph to dictionary"""
        return {
            'graph_id': self.graph_id,
            'name': self.name,
            'node_count': len(self.nodes),
            'edge_count': len(self.edges),
            'nodes': [
                {
                    'id': n.node_id,
                    'type': n.node_type,
                    'label': n.label,
                    'properties': n.properties
                }
                for n in self.nodes.values()
            ],
            'edges': [
                {
                    'id': e.edge_id,
                    'source': e.source_id,
                    'target': e.target_id,
                    'type': e.edge_type,
                    'weight': e.weight
                }
                for e in self.edges.values()
            ]
        }


@dataclass
class GraphQueryResult:
    """Result of graph query"""
    query: str
    result_type: str
    
    # Results
    matched_nodes: List[GraphNode] = field(default_factory=list)
    matched_paths: List[List[GraphEdge]] = field(default_factory=list)
    
    # Scores
    relevance_scores: Dict[str, float] = field(default_factory=dict)
    total_score: float = 0.0
    
    # Metadata
    nodes_visited: int = 0
    edges_traversed: int = 0
    processing_time_ms: float = 0.0


class KnowledgeGraphExtractor:
    """
    Extracts knowledge graphs from text content.
    
    Identifies entities, relationships, and concepts
    to build structured knowledge representations.
    """
    
    def __init__(self):
        """Initialize knowledge graph extractor"""
        self._entity_patterns: Dict[str, List[Tuple[str, str]]] = {}
        self._relation_patterns: Dict[str, List[Tuple[str, str, str]]] = {}
        self._concept_patterns: Dict[str, str] = {}
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile entity and relation extraction patterns"""
        # Entity patterns (type -> [(pattern, description)])
        self._entity_patterns = {
            GraphNodeType.ENTITY.value: [
                (r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', "Named Entity"),
                (r'\b[A-Z]{2,}\b', "Acronym"),
            ],
            GraphNodeType.CONCEPT.value: [
                (r'\b(?:machine learning|artificial intelligence|data science)\b', "AI Concept"),
                (r'\b(?:neural network|deep learning|transformer)\b', "DL Concept"),
            ],
            GraphNodeType.EVENT.value: [
                (r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b', "Dated Event"),
                (r'\b\d{4}-\d{2}-\d{2}\b', "ISO Date"),
            ]
        }
        
        # Relation patterns (pattern -> (subject_type, relation, object_type))
        self._relation_patterns = {
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:is|are)\s+([A-Z][a-z]+)': (
                GraphNodeType.ENTITY.value,
                "is_a",
                GraphNodeType.CONCEPT.value
            ),
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:was|were|is|are)\s+located\s+in\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)': (
                GraphNodeType.ENTITY.value,
                "located_in",
                GraphNodeType.ENTITY.value
            ),
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:founded|created|established)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)': (
                GraphNodeType.ENTITY.value,
                "founded_by",
                GraphNodeType.ENTITY.value
            )
        }
    
    def extract(
        self,
        content: str,
        kernel_id: Optional[str] = None,
        min_confidence: float = 0.5
    ) -> KnowledgeGraph:
        """
        Extract knowledge graph from content
        
        Args:
            content: Text content to extract from
            kernel_id: Source kernel ID
            min_confidence: Minimum confidence threshold
            
        Returns:
            Extracted knowledge graph
        """
        graph = KnowledgeGraph(
            graph_id=str(uuid.uuid4()),
            name=f"Graph from {kernel_id or 'content'}"
        )
        
        # Extract entities
        entities = self._extract_entities(content)
        
        for entity in entities:
            if entity['confidence'] >= min_confidence:
                node = GraphNode(
                    node_id=str(uuid.uuid4()),
                    node_type=entity['type'],
                    label=entity['text'],
                    source_kernel=kernel_id,
                    confidence=entity['confidence'],
                    properties={
                        'position': entity.get('position', 0),
                        'description': entity.get('description', '')
                    }
                )
                graph.add_node(node)
        
        # Extract relations
        relations = self._extract_relations(content, entities)
        
        for relation in relations:
            if relation['confidence'] >= min_confidence:
                # Find or create nodes
                source_node = self._find_or_create_node(
                    graph, relation['subject'], GraphNodeType.ENTITY.value, kernel_id
                )
                target_node = self._find_or_create_node(
                    graph, relation['object'], GraphNodeType.ENTITY.value, kernel_id
                )
                
                edge = GraphEdge(
                    edge_id=str(uuid.uuid4()),
                    source_id=source_node.node_id,
                    target_id=target_node.node_id,
                    edge_type=relation['type'],
                    source_kernel=kernel_id,
                    confidence=relation['confidence']
                )
                graph.add_edge(edge)
        
        # Extract concepts
        concepts = self._extract_concepts(content)
        
        for concept in concepts:
            if concept['confidence'] >= min_confidence:
                node = GraphNode(
                    node_id=str(uuid.uuid4()),
                    node_type=GraphNodeType.CONCEPT.value,
                    label=concept['text'],
                    source_kernel=kernel_id,
                    confidence=concept['confidence']
                )
                graph.add_node(node)
        
        return graph
    
    def _extract_entities(self, content: str) -> List[Dict[str, Any]]:
        """Extract named entities"""
        entities = []
        
        for entity_type, patterns in self._entity_patterns.items():
            for pattern, description in patterns:
                for match in re.finditer(pattern, content):
                    # Calculate confidence based on context
                    confidence = self._calculate_entity_confidence(match, content, entity_type)
                    
                    entities.append({
                        'text': match.group(),
                        'type': entity_type,
                        'position': match.start(),
                        'description': description,
                        'confidence': confidence
                    })
        
        # Remove duplicates
        seen = set()
        unique_entities = []
        for entity in entities:
            key = (entity['text'].lower(), entity['type'])
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _calculate_entity_confidence(
        self,
        match: re.Match,
        content: str,
        entity_type: str
    ) -> float:
        """Calculate confidence score for entity"""
        base_confidence = 0.7
        
        # Check context
        context_window = content[max(0, match.start() - 50):match.end() + 50]
        
        # Capitalization patterns
        if re.match(r'^[A-Z]', match.group()):
            base_confidence += 0.1
        
        # Context indicators
        context_indicators = {
            GraphNodeType.ENTITY.value: ['the', 'a', 'an', 'called', 'named'],
            GraphNodeType.CONCEPT.value: ['of', 'in', 'for', 'the'],
            GraphNodeType.EVENT.value: ['on', 'at', 'during', 'when']
        }
        
        for indicator in context_indicators.get(entity_type, []):
            if indicator in context_window.lower():
                base_confidence += 0.05
        
        return min(0.95, base_confidence)
    
    def _extract_relations(
        self,
        content: str,
        entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract relations between entities"""
        relations = []
        
        for pattern, (subj_type, rel_type, obj_type) in self._relation_patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                subject = match.group(1)
                obj = match.group(2)
                
                # Verify entities exist
                if any(e['text'].lower() == subject.lower() for e in entities) or \
                   any(e['text'].lower() == obj.lower() for e in entities):
                    relations.append({
                        'subject': subject,
                        'object': obj,
                        'type': rel_type,
                        'confidence': 0.75
                    })
        
        return relations
    
    def _extract_concepts(self, content: str) -> List[Dict[str, Any]]:
        """Extract concepts"""
        concepts = []
        
        for concept_name, pattern in self._concept_patterns.items():
            for match in re.finditer(pattern, content, re.IGNORECASE):
                concepts.append({
                    'text': match.group(),
                    'type': GraphNodeType.CONCEPT.value,
                    'confidence': 0.8
                })
        
        return concepts
    
    def _find_or_create_node(
        self,
        graph: KnowledgeGraph,
        label: str,
        node_type: str,
        kernel_id: Optional[str]
    ) -> GraphNode:
        """Find existing node or create new one"""
        # Try to find existing
        existing = graph.get_nodes_by_label(label)
        if existing:
            return existing[0]
        
        # Create new node
        node = GraphNode(
            node_id=str(uuid.uuid4()),
            node_type=node_type,
            label=label,
            source_kernel=kernel_id,
            confidence=0.7
        )
        graph.add_node(node)
        
        return node


class GraphNeuralNetwork:
    """
    Graph Neural Network for reasoning over knowledge graphs.
    
    Performs node classification, link prediction, and
    graph-level reasoning over kernel relationships.
    """
    
    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_layers: List[int] = [64, 32]
    ):
        """
        Initialize graph neural network
        
        Args:
            embedding_dim: Node embedding dimension
            hidden_layers: Hidden layer sizes
        """
        self.embedding_dim = embedding_dim
        self.hidden_layers = hidden_layers
        
        # Learned embeddings
        self._node_embeddings: Dict[str, List[float]] = {}
        
        # GNN parameters (simplified)
        self._weights: List[Dict[str, List[List[float]]]] = []
    
    def compute_embeddings(self, graph: KnowledgeGraph) -> Dict[str, List[float]]:
        """
        Compute embeddings for all nodes in graph
        
        Args:
            graph: Knowledge graph
            
        Returns:
            Node ID -> embedding mapping
        """
        embeddings = {}
        
        for node_id, node in graph.nodes.items():
            # Generate embedding based on node properties
            embedding = self._generate_embedding(node)
            embeddings[node_id] = embedding
            self._node_embeddings[node_id] = embedding
        
        # Apply message passing (simplified)
        embeddings = self._message_passing(graph, embeddings)
        
        return embeddings
    
    def _generate_embedding(self, node: GraphNode) -> List[float]:
        """Generate embedding for node"""
        import random
        random.seed(hash(node.node_id))
        
        # Base embedding
        embedding = [random.uniform(-1, 1) for _ in range(self.embedding_dim)]
        
        # Modify based on node type
        type_offset = hash(node.node_type) % self.embedding_dim
        embedding[type_offset % self.embedding_dim] += 0.5
        
        # Modify based on label
        label_offset = sum(ord(c) for c in node.label) % self.embedding_dim
        embedding[label_offset % self.embedding_dim] += 0.3
        
        return embedding
    
    def _message_passing(
        self,
        graph: KnowledgeGraph,
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, List[float]]:
        """Perform message passing (simplified GNN)"""
        # Simple averaging of neighbor embeddings
        iterations = 2
        
        for _ in range(iterations):
            new_embeddings = {}
            
            for node_id, embedding in embeddings.items():
                # Get neighbor embeddings
                neighbor_data = graph.get_neighbors(node_id)
                if neighbor_data:
                    neighbors, _ = zip(*neighbor_data)
                else:
                    neighbors, _ = [], []
                
                if neighbors:
                    # Average neighbor embeddings
                    neighbor_embeds = [embeddings.get(n.node_id, self._generate_embedding(n)) 
                                       for n in neighbors]
                    
                    # Combine with own embedding
                    avg_neighbor = [
                        sum(e[i] for e in neighbor_embeds) / len(neighbor_embeds)
                        for i in range(self.embedding_dim)
                    ]
                    
                    # Weighted combination
                    combined = [
                        0.7 * embedding[i] + 0.3 * avg_neighbor[i]
                        for i in range(self.embedding_dim)
                    ]
                    new_embeddings[node_id] = combined
                else:
                    new_embeddings[node_id] = embedding
            
            embeddings = new_embeddings
        
        return embeddings
    
    def predict_link(
        self,
        graph: KnowledgeGraph,
        source_id: str,
        target_id: str
    ) -> Tuple[float, Optional[str]]:
        """
        Predict likelihood of link between nodes
        
        Args:
            graph: Knowledge graph
            source_id: Source node ID
            target_id: Target node ID
            
        Returns:
            (confidence, predicted edge type)
        """
        if source_id not in self._node_embeddings:
            self.compute_embeddings(graph)
        
        if source_id not in self._node_embeddings or target_id not in self._node_embeddings:
            return 0.0, None
        
        # Get embeddings
        src_emb = self._node_embeddings[source_id]
        tgt_emb = self._node_embeddings[target_id]
        
        # Calculate similarity and normalize to [0, 1]
        similarity = self._cosine_similarity(src_emb, tgt_emb)
        confidence = (similarity + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
        
        # Predict edge type based on context
        edge_type = self._predict_edge_type(graph, source_id, target_id)
        
        return confidence, edge_type
    
    def _cosine_similarity(
        self,
        vec1: List[float],
        vec2: List[float]
    ) -> float:
        """Calculate cosine similarity"""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 * norm2 == 0:
            return 0.0
        
        return dot / (norm1 * norm2)
    
    def _predict_edge_type(
        self,
        graph: KnowledgeGraph,
        source_id: str,
        target_id: str
    ) -> Optional[str]:
        """Predict edge type based on context"""
        # Get node types
        source_node = graph.nodes.get(source_id)
        target_node = graph.nodes.get(target_id)
        
        if not source_node or not target_node:
            return None
        
        source_type = source_node.node_type
        target_type = target_node.node_type
        
        # Predict edge type based on type combination
        type_to_edge = {
            (GraphNodeType.ENTITY.value, GraphNodeType.CONCEPT.value): GraphEdgeType.PART_OF.value,
            (GraphNodeType.ENTITY.value, GraphNodeType.ENTITY.value): GraphEdgeType.RELATED_TO.value,
            (GraphNodeType.CONCEPT.value, GraphNodeType.CONCEPT.value): GraphEdgeType.SIMILAR_TO.value,
            (GraphNodeType.DOCUMENT.value, GraphNodeType.ENTITY.value): GraphEdgeType.MENTIONS.value,
        }
        
        return type_to_edge.get((source_type, target_type))
    
    def classify_node(
        self,
        graph: KnowledgeGraph,
        node_id: str
    ) -> Dict[str, float]:
        """
        Classify node based on graph structure
        
        Args:
            graph: Knowledge graph
            node_id: Node to classify
            
        Returns:
            Type -> confidence mapping
        """
        if node_id not in self._node_embeddings:
            self.compute_embeddings(graph)
        
        # Get node neighbors
        neighbor_data = graph.get_neighbors(node_id)
        if neighbor_data:
            neighbors, _ = zip(*neighbor_data)
        else:
            neighbors, _ = [], []
        
        # Count neighbor types
        neighbor_types = defaultdict(int)
        for neighbor in neighbors:
            neighbor_types[neighbor.node_type] += 1
        
        # Classify based on majority neighbor type
        scores = {}
        for node_type in [et.value for et in GraphNodeType]:
            # Check direct neighbors
            direct_count = neighbor_types.get(node_type, 0)
            
            # Check 2-hop neighbors
            two_hop_count = 0
            for neighbor in neighbors:
                neighbor_data = graph.get_neighbors(neighbor.node_id)
                if neighbor_data:
                    sub_neighbors, _ = zip(*neighbor_data)
                else:
                    sub_neighbors, _ = [], []
                for sub_neighbor in sub_neighbors:
                    if sub_neighbor.node_type == node_type:
                        two_hop_count += 1
            
            scores[node_type] = (direct_count + two_hop_count * 0.5) / max(len(neighbors), 1)
        
        return scores


class HybridSearchEngine:
    """
    Hybrid search combining text similarity with graph traversal.
    
    Enables sophisticated queries that leverage both
    keyword matching and relationship reasoning.
    """
    
    def __init__(self):
        """Initialize hybrid search engine"""
        self._text_index: Dict[str, List[str]] = defaultdict(list)  # term -> doc_ids
        self._kernel_embeddings: Dict[str, List[float]] = {}
        
        # GNN for graph-based search
        self.gnn = GraphNeuralNetwork()
        
        # Connected graphs
        self._graphs: Dict[str, KnowledgeGraph] = {}
    
    def index_kernel(
        self,
        kernel_id: str,
        content: str,
        graph: Optional[KnowledgeGraph] = None
    ):
        """Index a kernel for search"""
        # Text indexing
        terms = self._extract_terms(content)
        for term in terms:
            self._text_index[term.lower()].append(kernel_id)
        
        # Store graph
        if graph:
            self._graphs[kernel_id] = graph
            
            # Compute embeddings
            self.gnn.compute_embeddings(graph)
    
    def _extract_terms(self, content: str) -> List[str]:
        """Extract search terms from content"""
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'of', 'in', 'for', 'on', 'with'}
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        terms = [w for w in words if w not in stopwords]
        
        return terms
    
    def search(
        self,
        query: str,
        kernel_ids: Optional[List[str]] = None,
        search_type: str = "hybrid",
        max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search
        
        Args:
            query: Search query
            kernel_ids: Optional kernel filter
            search_type: hybrid, text_only, graph_only
            max_results: Maximum results
            
        Returns:
            Search results with scores
        """
        start_time = time.time()
        
        # Text-based search
        text_results = self._text_search(query, kernel_ids)
        
        # Graph-based search
        graph_results = self._graph_search(query, kernel_ids)
        
        # Combine results
        combined = self._combine_results(
            text_results,
            graph_results,
            search_type
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return combined[:max_results]
    
    def _text_search(
        self,
        query: str,
        kernel_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Perform text-based search"""
        query_terms = self._extract_terms(query)
        scores = defaultdict(float)
        
        for term in query_terms:
            matching_kernels = self._text_index.get(term.lower(), [])
            
            for kernel_id in matching_kernels:
                if kernel_ids is None or kernel_id in kernel_ids:
                    scores[kernel_id] += 1.0
        
        # Normalize
        max_score = max(scores.values()) if scores else 1.0
        for kid in scores:
            scores[kid] = scores[kid] / max_score
        
        return dict(scores)
    
    def _graph_search(
        self,
        query: str,
        kernel_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Perform graph-based search"""
        scores = defaultdict(float)
        
        # Extract query entities
        extractor = KnowledgeGraphExtractor()
        query_graph = extractor.extract(query)
        query_nodes = list(query_graph.nodes.values())
        
        for kernel_id, graph in self._graphs.items():
            if kernel_ids is not None and kernel_id not in kernel_ids:
                continue
            
            # Compute graph similarity
            similarity = self._calculate_graph_similarity(query_graph, graph)
            scores[kernel_id] = similarity
        
        return dict(scores)
    
    def _calculate_graph_similarity(
        self,
        query_graph: KnowledgeGraph,
        target_graph: KnowledgeGraph
    ) -> float:
        """Calculate similarity between query and target graph"""
        if not target_graph.nodes:
            return 0.0
        
        # Node type overlap
        query_types = set(n.node_type for n in query_graph.nodes.values())
        target_types = set(n.node_type for n in target_graph.nodes.values())
        
        type_overlap = len(query_types & target_types)
        type_score = type_overlap / max(len(query_types), 1)
        
        # Label overlap
        query_labels = set(n.label.lower() for n in query_graph.nodes.values())
        target_labels = set(n.label.lower() for n in target_graph.nodes.values())
        
        label_overlap = len(query_labels & target_labels)
        label_score = label_overlap / max(len(query_labels), 1)
        
        # Combined score
        return (type_score * 0.4 + label_score * 0.6)
    
    def _combine_results(
        self,
        text_results: Dict[str, float],
        graph_results: Dict[str, float],
        search_type: str
    ) -> List[Dict[str, Any]]:
        """Combine text and graph results"""
        all_kernels = set(text_results.keys()) | set(graph_results.keys())
        
        combined = []
        
        for kernel_id in all_kernels:
            text_score = text_results.get(kernel_id, 0.0)
            graph_score = graph_results.get(kernel_id, 0.0)
            
            if search_type == "text_only":
                final_score = text_score
            elif search_type == "graph_only":
                final_score = graph_score
            else:  # hybrid
                final_score = text_score * 0.5 + graph_score * 0.5
            
            combined.append({
                'kernel_id': kernel_id,
                'text_score': text_score,
                'graph_score': graph_score,
                'combined_score': final_score
            })
        
        # Sort by combined score
        combined.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return combined


class KnowledgeGraphQuerier:
    """
    Query interface for knowledge graphs spanning multiple kernels.
    
    Supports complex queries that traverse relationships
    across kernel boundaries.
    """
    
    def __init__(self):
        """Initialize knowledge graph querier"""
        self._master_graph = KnowledgeGraph(
            graph_id="master",
            name="Master Knowledge Graph"
        )
        
        self._kernel_maps: Dict[str, KnowledgeGraph] = {}
    
    def register_kernel_graph(
        self,
        kernel_id: str,
        graph: KnowledgeGraph
    ):
        """Register a kernel's knowledge graph"""
        self._kernel_maps[kernel_id] = graph
        
        # Merge into master graph
        for node_id, node in graph.nodes.items():
            # Create unique node ID
            unique_id = f"{kernel_id}_{node_id}"
            merged_node = GraphNode(
                node_id=unique_id,
                node_type=node.node_type,
                label=node.label,
                source_kernel=kernel_id,
                confidence=node.confidence,
                properties=node.properties
            )
            self._master_graph.add_node(merged_node)
        
        for edge_id, edge in graph.edges.items():
            # Create unique edge
            unique_edge = GraphEdge(
                edge_id=f"{kernel_id}_{edge_id}",
                source_id=f"{kernel_id}_{edge.source_id}",
                target_id=f"{kernel_id}_{edge.target_id}",
                edge_type=edge.edge_type,
                source_kernel=kernel_id,
                confidence=edge.confidence,
                weight=edge.weight
            )
            self._master_graph.add_edge(unique_edge)
        
        # Add kernel containment edges
        kernel_node = GraphNode(
            node_id=f"kernel_{kernel_id}",
            node_type=GraphNodeType.KERNEL.value,
            label=kernel_id,
            source_kernel=kernel_id
        )
        self._master_graph.add_node(kernel_node)
        
        for node_id in graph.nodes.keys():
            kernel_edge = GraphEdge(
                edge_id=f"contains_{kernel_id}_{node_id}",
                source_id=kernel_node.node_id,
                target_id=f"{kernel_id}_{node_id}",
                edge_type=GraphEdgeType.CONTAINS.value,
                source_kernel=kernel_id,
                directed=True
            )
            self._master_graph.add_edge(kernel_edge)
        
        logger.info(f"Registered kernel graph: {kernel_id}")
    
    def query(
        self,
        query: str,
        kernel_ids: Optional[List[str]] = None
    ) -> GraphQueryResult:
        """
        Query across knowledge graphs
        
        Args:
            query: Query string
            kernel_ids: Optional kernel filter
            
        Returns:
            Query result
        """
        start_time = time.time()
        
        # Parse query
        query_parts = self._parse_query(query)
        
        # Build kernel filter
        if kernel_ids is None:
            kernel_ids = list(self._kernel_maps.keys())
        
        # Execute traversal
        matched_nodes = []
        matched_paths = []
        
        for kernel_id in kernel_ids:
            if kernel_id not in self._kernel_maps:
                continue
            
            kernel_graph = self._kernel_maps[kernel_id]
            
            # Find matching nodes
            for node_id, node in kernel_graph.nodes.items():
                if self._matches_query(node, query_parts):
                    matched_nodes.append(node)
                    
                    # Find paths to query context
                    paths = self._find_paths_to_context(
                        kernel_graph, node_id, query_parts
                    )
                    matched_paths.extend(paths)
        
        # Calculate scores
        relevance_scores = {}
        for node in matched_nodes:
            score = self._calculate_relevance(node, query_parts)
            relevance_scores[node.node_id] = score
        
        total_score = sum(relevance_scores.values()) / max(len(relevance_scores), 1)
        
        processing_time = (time.time() - start_time) * 1000
        
        return GraphQueryResult(
            query=query,
            result_type="hybrid",
            matched_nodes=matched_nodes,
            matched_paths=matched_paths,
            relevance_scores=relevance_scores,
            total_score=total_score,
            nodes_visited=len(matched_nodes),
            edges_traversed=len(matched_paths),
            processing_time_ms=processing_time
        )
    
    def _parse_query(self, query: str) -> Dict[str, Any]:
        """Parse query into components"""
        return {
            'entities': re.findall(r'\[([^\]]+)\]', query),
            'relations': re.findall(r'\(([^)]+)\)', query),
            'keywords': self._extract_keywords(query)
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'find', 'show', 'what'}
        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        return [w for w in words if w not in stopwords]
    
    def _matches_query(
        self,
        node: GraphNode,
        query_parts: Dict[str, Any]
    ) -> bool:
        """Check if node matches query"""
        # Check entity brackets
        for entity in query_parts.get('entities', []):
            if entity.lower() in node.label.lower():
                return True
        
        # Check keywords
        for keyword in query_parts.get('keywords', []):
            if keyword in node.label.lower():
                return True
        
        return False
    
    def _find_paths_to_context(
        self,
        graph: KnowledgeGraph,
        start_node_id: str,
        query_parts: Dict[str, Any]
    ) -> List[List[GraphEdge]]:
        """Find paths to query context"""
        paths = []
        
        # Simple BFS
        visited = {start_node_id}
        queue = [(start_node_id, 0, [])]
        
        while queue:
            current_id, current_depth, current_path = queue.pop(0)
            
            if current_depth > 3:  # Max depth
                continue
            
            for neighbor, edge in graph.get_neighbors(current_id):
                if neighbor.node_id not in visited:
                    new_path = current_path + [edge]
                    
                    # Check if this is a relevant path
                    if self._is_relevant_path(new_path, query_parts):
                        paths.append(new_path)
                    
                    visited.add(neighbor.node_id)
                    queue.append((neighbor.node_id, current_depth + 1, new_path))
        
        return paths
    
    def _is_relevant_path(
        self,
        path: List[GraphEdge],
        query_parts: Dict[str, Any]
    ) -> bool:
        """Check if path is relevant to query"""
        # Simplified: paths of length 1-3 are relevant
        return 1 <= len(path) <= 3
    
    def _calculate_relevance(
        self,
        node: GraphNode,
        query_parts: Dict[str, Any]
    ) -> float:
        """Calculate relevance score"""
        score = 0.0
        
        # Exact entity match
        for entity in query_parts.get('entities', []):
            if entity.lower() == node.label.lower():
                score = 1.0
                break
            elif entity.lower() in node.label.lower():
                score = max(score, 0.7)
        
        # Keyword match
        for keyword in query_parts.get('keywords', []):
            if keyword in node.label.lower():
                score = max(score, 0.5)
        
        # Type bonus
        if node.node_type == GraphNodeType.ENTITY.value:
            score *= 1.2
        
        return min(1.0, score)
    
    def traverse_relationship(
        self,
        start_kernel: str,
        start_entity: str,
        relation_type: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Traverse relationships starting from entity
        
        Args:
            start_kernel: Starting kernel
            start_entity: Starting entity label
            relation_type: Relation type to follow
            max_depth: Maximum traversal depth
            
        Returns:
            Found entities with paths
        """
        results = []
        
        if start_kernel not in self._kernel_maps:
            return results
        
        graph = self._kernel_maps[start_kernel]
        
        # Find starting node
        start_nodes = graph.get_nodes_by_label(start_entity)
        
        for start_node in start_nodes:
            # BFS traversal
            visited = {start_node.node_id}
            queue = [(start_node.node_id, 0, [])]
            
            while queue:
                current_id, depth, path = queue.pop(0)
                
                if depth > max_depth:
                    continue
                
                for neighbor, edge in graph.get_neighbors(current_id, relation_type):
                    if neighbor.node_id not in visited:
                        visited.add(neighbor.node_id)
                        
                        new_path = path + [(edge, neighbor)]
                        
                        results.append({
                            'entity': neighbor.label,
                            'type': neighbor.node_type,
                            'kernel': start_kernel,
                            'path_length': depth + 1,
                            'confidence': edge.confidence
                        })
                        
                        queue.append((neighbor.node_id, depth + 1, new_path))
        
        return results


class KernelGraphManager:
    """
    Manager for knowledge graphs across all kernels.
    
    Coordinates extraction, indexing, and querying
    across the kernel ecosystem.
    """
    
    def __init__(self):
        """Initialize kernel graph manager"""
        self.extractor = KnowledgeGraphExtractor()
        self.search_engine = HybridSearchEngine()
        self.querier = KnowledgeGraphQuerier()
        
        # Registry
        self._kernel_graphs: Dict[str, KnowledgeGraph] = {}
        self._kernel_content: Dict[str, str] = {}
    
    def register_kernel(
        self,
        kernel_id: str,
        content: str,
        extract_graph: bool = True
    ) -> KnowledgeGraph:
        """
        Register a kernel for graph management
        
        Args:
            kernel_id: Kernel identifier
            content: Kernel content
            extract_graph: Whether to extract graph
            
        Returns:
            Extracted knowledge graph
        """
        self._kernel_content[kernel_id] = content
        
        if extract_graph:
            graph = self.extractor.extract(content, kernel_id)
            self._kernel_graphs[kernel_id] = graph
            
            # Register with search engine
            self.search_engine.index_kernel(kernel_id, content, graph)
            
            # Register with querier
            self.querier.register_kernel_graph(kernel_id, graph)
            
            return graph
        
        return None
    
    def query_kernels(
        self,
        query: str,
        kernel_ids: Optional[List[str]] = None,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Query across kernels"""
        return self.search_engine.search(query, kernel_ids, search_type)
    
    def query_graph(
        self,
        query: str,
        kernel_ids: Optional[List[str]] = None
    ) -> GraphQueryResult:
        """Query knowledge graphs"""
        return self.querier.query(query, kernel_ids)
    
    def traverse_relationships(
        self,
        kernel_id: str,
        entity: str,
        relation: str,
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """Traverse entity relationships"""
        return self.querier.traverse_relationship(kernel_id, entity, relation, max_depth)
    
    def get_kernel_graph(self, kernel_id: str) -> Optional[KnowledgeGraph]:
        """Get kernel's knowledge graph"""
        return self._kernel_graphs.get(kernel_id)
    
    def list_kernels(self) -> List[str]:
        """List registered kernels"""
        return list(self._kernel_graphs.keys())
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics"""
        total_nodes = sum(len(g.nodes) for g in self._kernel_graphs.values())
        total_edges = sum(len(g.edges) for g in self._kernel_graphs.values())
        
        # Node type distribution
        type_counts = defaultdict(int)
        for graph in self._kernel_graphs.values():
            for node in graph.nodes.values():
                type_counts[node.node_type] += 1
        
        return {
            'total_kernels': len(self._kernel_graphs),
            'total_nodes': total_nodes,
            'total_edges': total_edges,
            'node_type_distribution': dict(type_counts),
            'avg_nodes_per_kernel': total_nodes / max(len(self._kernel_graphs), 1)
        }


# Singleton instance
_kernel_graph_manager: Optional[KernelGraphManager] = None


def get_kernel_graph_manager() -> KernelGraphManager:
    """Get kernel graph manager singleton"""
    global _kernel_graph_manager
    
    if _kernel_graph_manager is None:
        _kernel_graph_manager = KernelGraphManager()
    
    return _kernel_graph_manager


def init_knowledge_graph() -> KernelGraphManager:
    """Initialize knowledge graph system"""
    global _kernel_graph_manager
    
    _kernel_graph_manager = KernelGraphManager()
    
    return _kernel_graph_manager
