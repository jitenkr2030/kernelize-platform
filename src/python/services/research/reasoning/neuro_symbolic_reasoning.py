"""
KERNELIZE Platform - Neuro-Symbolic Reasoning Research
=======================================================

Advanced reasoning capabilities combining neural network pattern
recognition with symbolic reasoning for robust knowledge inference.

Features:
- Neural module networks for complex reasoning decomposition
- Symbolic knowledge representation with explicit reasoning chains
- Hybrid approaches using neural matching with symbolic verification
- Multi-step inference across kernel boundaries

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
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)


class ReasoningType(Enum):
    """Types of reasoning supported"""
    DEDUCTIVE = "deductive"       # General to specific
    INDUCTIVE = "inductive"       # Specific to general
    ABDUCTIVE = "abductive"       # Best explanation
    ANALOGICAL = "analogical"     # Similarity-based
    CAUSAL = "causal"             # Cause-effect relationships
    TEMPORAL = "temporal"         # Time-based reasoning
    COUNTERFACTUAL = "counterfactual"  # What-if scenarios


@dataclass
class InferenceResult:
    """Result of reasoning inference"""
    conclusion: str
    reasoning_type: str
    confidence: float
    supporting_evidence: List[str] = field(default_factory=list)
    reasoning_chain: List[Dict[str, Any]] = field(default_factory=list)
    intermediate_conclusions: List[str] = field(default_factory=list)
    
    # Quality metrics
    logical_consistency: float = 1.0
    factual_accuracy: float = 1.0
    completeness: float = 1.0
    
    # Metadata
    processing_time_ms: float = 0.0
    modules_used: List[str] = field(default_factory=list)
    rules_applied: List[str] = field(default_factory=list)


@dataclass
class SymbolicRule:
    """Symbolic reasoning rule"""
    rule_id: str
    name: str
    premise: str
    conclusion: str
    condition: Optional[str] = None
    
    # Rule metadata
    confidence: float = 1.0
    priority: int = 100
    domain: str = "general"
    
    # Applicability
    antecedents: List[str] = field(default_factory=list)
    consequents: List[str] = field(default_factory=list)
    
    # Source
    source: str = "manual"
    description: str = ""


@dataclass
class KnowledgeAssertion:
    """Atomic knowledge assertion"""
    assertion_id: str
    subject: str
    predicate: str
    object: str
    
    # Confidence and provenance
    confidence: float = 1.0
    source: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Context
    context: str = ""
    domain: str = "general"
    
    # Metadata
    derived: bool = False
    derivation_steps: List[str] = field(default_factory=list)


@dataclass
class NeuralModule:
    """Neural module for reasoning component"""
    module_id: str
    name: str
    module_type: str
    
    # Function
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    
    # Implementation
    neural_network: Any = None  # Placeholder for actual NN
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Capabilities
    reasoning_types: List[str] = field(default_factory=list)
    input_types: List[str] = field(default_factory=list)
    output_types: List[str] = field(default_factory=list)


@dataclass
class ReasoningStep:
    """Step in multi-step reasoning chain"""
    step_id: str
    step_number: int
    module_name: str
    action: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    
    # Intermediate result
    intermediate_conclusion: Optional[str] = None
    confidence: float = 1.0
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    provides_to: List[str] = field(default_factory=list)


class SymbolicKnowledgeBase:
    """
    Symbolic knowledge representation for explicit reasoning.
    
    Stores knowledge as assertions and rules supporting
    logical inference and reasoning chains.
    """
    
    def __init__(self, name: str = "default"):
        """
        Initialize symbolic knowledge base
        
        Args:
            name: Knowledge base name
        """
        self.name = name
        self._assertions: Dict[str, KnowledgeAssertion] = {}
        self._rules: Dict[str, SymbolicRule] = {}
        self._indexes: Dict[str, set] = defaultdict(set)
        
        # Inference engine
        self._forward_chains: Dict[str, List[str]] = defaultdict(list)  # antecedent -> rules
        self._backward_chains: Dict[str, List[str]] = defaultdict(list)  # consequent -> rules
    
    def add_assertion(self, assertion: KnowledgeAssertion):
        """Add knowledge assertion"""
        self._assertions[assertion.assertion_id] = assertion
        
        # Update indexes
        self._indexes[f"subject:{assertion.subject}"].add(assertion.assertion_id)
        self._indexes[f"predicate:{assertion.predicate}"].add(assertion.assertion_id)
        self._indexes[f"object:{assertion.object}"].add(assertion.assertion_id)
        self._indexes[f"domain:{assertion.domain}"].add(assertion.assertion_id)
    
    def add_rule(self, rule: SymbolicRule):
        """Add reasoning rule"""
        self._rules[rule.rule_id] = rule
        
        # Update chain indexes
        for antecedent in rule.antecedents:
            self._forward_chains[antecedent].append(rule.rule_id)
        for consequent in rule.consequents:
            self._backward_chains[consequent].append(rule.rule_id)
    
    def query(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object: Optional[str] = None,
        domain: Optional[str] = None
    ) -> List[KnowledgeAssertion]:
        """Query knowledge base"""
        results = set(self._assertions.keys())
        
        if subject:
            results &= self._indexes.get(f"subject:{subject}", set())
        if predicate:
            results &= self._indexes.get(f"predicate:{predicate}", set())
        if object:
            results &= self._indexes.get(f"object:{object}", set())
        if domain:
            results &= self._indexes.get(f"domain:{domain}", set())
        
        return [self._assertions[a_id] for a_id in results]
    
    def get_rules_for_conclusion(self, conclusion: str) -> List[SymbolicRule]:
        """Get rules that can derive a conclusion"""
        rule_ids = self._backward_chains.get(conclusion, [])
        return [self._rules[rid] for rid in rule_ids if rid in self._rules]
    
    def get_applicable_rules(self, premises: List[str]) -> List[SymbolicRule]:
        """Get rules applicable to current premises"""
        applicable = []
        
        for premise in premises:
            rule_ids = self._forward_chains.get(premise, [])
            for rid in rule_ids:
                if rid in self._rules:
                    applicable.append(self._rules[rid])
        
        # Sort by priority
        applicable.sort(key=lambda r: r.priority)
        
        return applicable
    
    def derive_conclusion(
        self,
        rule: SymbolicRule,
        matched_assertions: List[KnowledgeAssertion]
    ) -> Optional[KnowledgeAssertion]:
        """Derive new assertion from rule application"""
        # Create new assertion
        assertion = KnowledgeAssertion(
            assertion_id=str(uuid.uuid4()),
            subject=rule.conclusion,
            predicate="derived_from",
            object=rule.name,
            confidence=rule.confidence * min(a.confidence for a in matched_assertions),
            source=f"Rule: {rule.rule_id}",
            derived=True,
            derivation_steps=[a.assertion_id for a in matched_assertions]
        )
        
        return assertion
    
    def export(self) -> Dict[str, Any]:
        """Export knowledge base"""
        return {
            'name': self.name,
            'assertions': [
                {
                    'id': a.assertion_id,
                    'subject': a.subject,
                    'predicate': a.predicate,
                    'object': a.object,
                    'confidence': a.confidence,
                    'domain': a.domain
                }
                for a in self._assertions.values()
            ],
            'rules': [
                {
                    'id': r.rule_id,
                    'name': r.name,
                    'premise': r.premise,
                    'conclusion': r.conclusion,
                    'confidence': r.confidence,
                    'domain': r.domain
                }
                for r in self._rules.values()
            ]
        }


class NeuralModuleNetwork:
    """
    Neural module network for decomposing complex reasoning.
    
    Uses learned neural modules that can be composed
    for different reasoning tasks.
    """
    
    def __init__(self):
        """Initialize neural module network"""
        self._modules: Dict[str, NeuralModule] = {}
        self._compositions: Dict[str, List[str]] = {}  # task -> module sequence
        self._module_outputs: Dict[str, Any] = {}
        
        # Initialize default modules
        self._initialize_default_modules()
    
    def _initialize_default_modules(self):
        """Initialize default reasoning modules"""
        # Comparison module
        self.register_module(NeuralModule(
            module_id="compare",
            name="CompareModule",
            module_type="comparison",
            reasoning_types=[ReasoningType.ANALOGICAL.value],
            input_types=["text", "text"],
            output_types=["similarity_score", "differences"]
        ))
        
        # Classification module
        self.register_module(NeuralModule(
            module_id="classify",
            name="ClassifyModule",
            module_type="classification",
            reasoning_types=[ReasoningType.DEDUCTIVE.value],
            input_types=["text", "categories"],
            output_types=["category", "confidence"]
        ))
        
        # Extraction module
        self.register_module(NeuralModule(
            module_id="extract",
            name="ExtractModule",
            module_type="extraction",
            reasoning_types=[ReasoningType.INDUCTIVE.value],
            input_types=["text", "schema"],
            output_types=["entities", "relations"]
        ))
        
        # Inference module
        self.register_module(NeuralModule(
            module_id="infer",
            name="InferModule",
            module_type="inference",
            reasoning_types=[ReasoningType.DEDUCTIVE.value, ReasoningType.CAUSAL.value],
            input_types=["premises", "rules"],
            output_types=["conclusions", "confidence"]
        ))
        
        # Synthesis module
        self.register_module(NeuralModule(
            module_id="synthesize",
            name="SynthesizeModule",
            module_type="synthesis",
            reasoning_types=[ReasoningType.ABDUCTIVE.value],
            input_types=["evidence", "hypotheses"],
            output_types=["best_explanation", "supporting_evidence"]
        ))
        
        # Temporal reasoning module
        self.register_module(NeuralModule(
            module_id="temporal",
            name="TemporalModule",
            module_type="temporal",
            reasoning_types=[ReasoningType.TEMPORAL.value],
            input_types=["events", "temporal_relations"],
            output_types=["timeline", "causal_chain"]
        ))
    
    def register_module(self, module: NeuralModule):
        """Register a neural module"""
        self._modules[module.module_id] = module
        
        logger.info(f"Registered neural module: {module.name}")
    
    def compose_for_task(
        self,
        task_type: str,
        module_sequence: List[str]
    ):
        """Compose modules for a task type"""
        self._compositions[task_type] = module_sequence
    
    def execute_module(
        self,
        module_id: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a neural module
        
        In production, this would call actual neural networks.
        Here we simulate module execution.
        """
        if module_id not in self._modules:
            raise ValueError(f"Unknown module: {module_id}")
        
        module = self._modules[module_id]
        start_time = time.time()
        
        # Simulate module execution
        output = self._simulate_module_execution(module, input_data)
        
        processing_time = (time.time() - start_time) * 1000
        
        self._module_outputs[module_id] = output
        
        return output
    
    def _simulate_module_execution(
        self,
        module: NeuralModule,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate neural module execution"""
        module_type = module.module_type
        
        if module_type == "comparison":
            # Compare two inputs
            text1 = input_data.get("text1", "")
            text2 = input_data.get("text2", "")
            
            # Calculate similarity (simplified)
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1 & words2
            union = words1 | words2
            similarity = len(intersection) / max(len(union), 1)
            
            differences = list(words2 - words1)
            
            return {
                "similarity_score": similarity,
                "differences": differences[:5],
                "common_elements": list(intersection)[:5]
            }
        
        elif module_type == "classification":
            # Classify input
            text = input_data.get("text", "")
            categories = input_data.get("categories", [])
            
            # Simple keyword-based classification
            text_lower = text.lower()
            scores = {}
            
            for cat in categories:
                cat_lower = cat.lower()
                if cat_lower in text_lower:
                    scores[cat] = 0.8 + (text_lower.count(cat_lower) * 0.05)
                else:
                    scores[cat] = 0.3
            
            # Get best category
            best_cat = max(scores, key=scores.get) if scores else categories[0] if categories else "unknown"
            
            return {
                "category": best_cat,
                "confidence": min(1.0, scores.get(best_cat, 0.0)),
                "all_scores": scores
            }
        
        elif module_type == "extraction":
            # Extract entities/relations
            text = input_data.get("text", "")
            schema = input_data.get("schema", {})
            
            # Simple extraction
            entities = []
            relations = []
            
            # Extract capitalized phrases as entities
            for match in re.finditer(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text):
                entities.append({
                    "text": match.group(),
                    "position": match.start(),
                    "type": "entity"
                })
            
            return {
                "entities": entities[:10],
                "relations": relations[:5]
            }
        
        elif module_type == "inference":
            # Perform inference
            premises = input_data.get("premises", [])
            rules = input_data.get("rules", [])
            
            conclusions = []
            for premise in premises:
                for rule in rules:
                    if premise.lower() in rule.get("premise", "").lower():
                        conclusions.append(rule.get("conclusion", ""))
            
            return {
                "conclusions": list(set(conclusions))[:5],
                "confidence": 0.7 if conclusions else 0.3
            }
        
        elif module_type == "synthesis":
            # Synthesize explanation
            evidence = input_data.get("evidence", [])
            hypotheses = input_data.get("hypotheses", [])
            
            # Simple best explanation selection
            best = hypotheses[0] if hypotheses else "No hypothesis available"
            
            return {
                "best_explanation": best,
                "supporting_evidence": evidence[:3],
                "confidence": 0.6
            }
        
        elif module_type == "temporal":
            # Temporal reasoning
            events = input_data.get("events", [])
            
            # Sort by time (simplified)
            sorted_events = sorted(events, key=lambda e: e.get("time", ""))
            
            return {
                "timeline": sorted_events[:10],
                "causal_chain": [],
                "confidence": 0.7
            }
        
        return {"status": "completed"}
    
    def get_module(self, module_id: str) -> Optional[NeuralModule]:
        """Get module by ID"""
        return self._modules.get(module_id)
    
    def list_modules(self) -> List[Dict[str, str]]:
        """List all modules"""
        return [
            {"id": m.module_id, "name": m.name, "type": m.module_type}
            for m in self._modules.values()
        ]


class HybridReasoningEngine:
    """
    Hybrid reasoning engine combining neural and symbolic approaches.
    
    Uses neural networks for pattern recognition and fuzzy matching,
    while using symbolic reasoning for logical inference and verification.
    """
    
    def __init__(self):
        """Initialize hybrid reasoning engine"""
        self.symbolic_kb = SymbolicKnowledgeBase()
        self.neural_modules = NeuralModuleNetwork()
        
        # Configuration
        self.neural_threshold = 0.7
        self.symbolic_threshold = 0.9
        
        # Reasoning history
        self._reasoning_history: List[InferenceResult] = []
    
    def reason(
        self,
        query: str,
        context: str,
        reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE
    ) -> InferenceResult:
        """
        Perform hybrid reasoning on query
        
        Args:
            query: Question or reasoning task
            context: Supporting context
            reasoning_type: Type of reasoning to apply
            
        Returns:
            Inference result with conclusion and reasoning chain
        """
        start_time = time.time()
        
        # Step 1: Neural processing for pattern matching
        neural_result = self._neural_process(query, context, reasoning_type)
        
        # Step 2: Symbolic verification and inference
        symbolic_result = self._symbolic_verify(neural_result, reasoning_type)
        
        # Step 3: Combine results
        final_result = self._combine_results(neural_result, symbolic_result)
        
        # Step 4: Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(
            neural_result, symbolic_result, final_result
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create final result
        result = InferenceResult(
            conclusion=final_result.get("conclusion", ""),
            reasoning_type=reasoning_type.value,
            confidence=final_result.get("confidence", 0.5),
            supporting_evidence=final_result.get("evidence", []),
            reasoning_chain=reasoning_chain,
            intermediate_conclusions=final_result.get("intermediate", []),
            processing_time_ms=processing_time,
            modules_used=final_result.get("modules", []),
            rules_applied=final_result.get("rules", [])
        )
        
        self._reasoning_history.append(result)
        
        return result
    
    def _neural_process(
        self,
        query: str,
        context: str,
        reasoning_type: ReasoningType
    ) -> Dict[str, Any]:
        """Process using neural modules"""
        # Select appropriate modules based on reasoning type
        module_sequence = self._select_modules(reasoning_type)
        
        intermediate_results = []
        current_input = {"query": query, "context": context}
        
        for module_id in module_sequence:
            result = self.neural_modules.execute_module(module_id, current_input)
            
            intermediate_results.append({
                "module": module_id,
                "output": result
            })
            
            # Update input for next module
            current_input.update(result)
        
        return {
            "primary_conclusion": current_input.get("best_exclusion", current_input.get("category", "")),
            "confidence": current_input.get("confidence", 0.5),
            "intermediate": intermediate_results,
            "extracted_entities": current_input.get("entities", []),
            "similarity_score": current_input.get("similarity_score", 0.0)
        }
    
    def _select_modules(self, reasoning_type: ReasoningType) -> List[str]:
        """Select modules for reasoning type"""
        module_map = {
            ReasoningType.DEDUCTIVE: ["extract", "infer"],
            ReasoningType.INDUCTIVE: ["extract", "classify"],
            ReasoningType.ABDUCTIVE: ["extract", "synthesize"],
            ReasoningType.ANALOGICAL: ["compare", "extract"],
            ReasoningType.CAUSAL: ["extract", "infer"],
            ReasoningType.TEMPORAL: ["temporal", "extract"],
            ReasoningType.COUNTERFACTUAL: ["synthesize", "infer"]
        }
        
        return module_map.get(reasoning_type, ["extract", "infer"])
    
    def _symbolic_verify(
        self,
        neural_result: Dict[str, Any],
        reasoning_type: ReasoningType
    ) -> Dict[str, Any]:
        """Verify and extend using symbolic reasoning"""
        # Query knowledge base
        entities = neural_result.get("extracted_entities", [])
        conclusions = []
        applied_rules = []
        
        for entity in entities:
            entity_text = entity.get("text", "")
            
            # Get applicable rules
            rules = self.symbolic_kb.get_applicable_rules([entity_text])
            
            for rule in rules[:3]:  # Limit rules
                # Check if rule applies
                if self._check_rule_applicability(rule, neural_result):
                    applied_rules.append(rule.name)
                    
                    # Derive conclusion
                    assertion = self.symbolic_kb.derive_conclusion(
                        rule, [KnowledgeAssertion(
                            assertion_id=str(uuid.uuid4()),
                            subject=entity_text,
                            predicate="matched",
                            object=rule.premise
                        )]
                    )
                    
                    if assertion:
                        conclusions.append(assertion.object)
        
        return {
            "symbolic_conclusions": conclusions,
            "rules_applied": applied_rules,
            "verification_confidence": 0.9 if applied_rules else 0.5
        }
    
    def _check_rule_applicability(
        self,
        rule: SymbolicRule,
        neural_result: Dict[str, Any]
    ) -> bool:
        """Check if rule applies to neural result"""
        # Simple check - in production would be more sophisticated
        conclusion_lower = rule.conclusion.lower()
        entities = [e.get("text", "").lower() for e in neural_result.get("extracted_entities", [])]
        
        return any(entity in conclusion_lower for entity in entities)
    
    def _combine_results(
        self,
        neural: Dict[str, Any],
        symbolic: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine neural and symbolic results"""
        neural_conf = neural.get("confidence", 0.5)
        symbolic_conf = symbolic.get("verification_confidence", 0.5)
        
        # Weighted combination
        combined_conf = (neural_conf * 0.4 + symbolic_conf * 0.6)
        
        # Combine conclusions
        neural_conclusion = neural.get("primary_conclusion", "")
        symbolic_conclusions = symbolic.get("symbolic_conclusions", [])
        
        # Prefer symbolic if confident
        if symbolic_conf > self.symbolic_threshold and symbolic_conclusions:
            conclusion = symbolic_conclusions[0]
        else:
            conclusion = neural_conclusion or (symbolic_conclusions[0] if symbolic_conclusions else "")
        
        return {
            "conclusion": conclusion,
            "confidence": combined_conf,
            "evidence": neural.get("extracted_entities", []),
            "intermediate": [neural.get("primary_conclusion", "")],
            "modules": ["neural_modules"],
            "rules": symbolic.get("rules_applied", [])
        }
    
    def _generate_reasoning_chain(
        self,
        neural: Dict[str, Any],
        symbolic: Dict[str, Any],
        final: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate reasoning chain for explanation"""
        chain = []
        
        # Neural processing step
        chain.append({
            "step": 1,
            "type": "neural",
            "description": "Pattern matching and entity extraction",
            "modules_used": final.get("modules", []),
            "confidence": neural.get("confidence", 0.5)
        })
        
        # Symbolic verification step
        if symbolic.get("rules_applied"):
            chain.append({
                "step": 2,
                "type": "symbolic",
                "description": "Logical verification and inference",
                "rules_applied": symbolic.get("rules_applied", []),
                "confidence": symbolic.get("verification_confidence", 0.5)
            })
        
        # Final combination
        chain.append({
            "step": 3,
            "type": "hybrid",
            "description": "Result synthesis and confidence calculation",
            "final_confidence": final.get("confidence", 0.0)
        })
        
        return chain
    
    def add_knowledge(
        self,
        assertions: Optional[List[KnowledgeAssertion]] = None,
        rules: Optional[List[SymbolicRule]] = None
    ):
        """Add knowledge to the knowledge base"""
        if assertions:
            for assertion in assertions:
                self.symbolic_kb.add_assertion(assertion)
        
        if rules:
            for rule in rules:
                self.symbolic_kb.add_rule(rule)
    
    def get_reasoning_history(self) -> List[Dict[str, Any]]:
        """Get reasoning history"""
        return [
            {
                "conclusion": r.conclusion,
                "reasoning_type": r.reasoning_type,
                "confidence": r.confidence,
                "processing_time_ms": r.processing_time_ms
            }
            for r in self._reasoning_history
        ]


class MultiStepInferenceEngine:
    """
    Multi-step inference engine supporting reasoning across
    multiple kernel boundaries.
    
    Enables complex reasoning chains that span multiple
    documents or knowledge segments.
    """
    
    def __init__(self):
        """Initialize multi-step inference engine"""
        self._kernels: Dict[str, Dict[str, Any]] = {}  # kernel_id -> content
        self._inference_chains: List[List[ReasoningStep]] = []
        
        # Current chain
        self._current_chain: List[ReasoningStep] = []
    
    def register_kernel(
        self,
        kernel_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Register a knowledge kernel"""
        self._kernels[kernel_id] = {
            "content": content,
            "metadata": metadata or {},
            "assertions": []
        }
    
    def add_assertion_to_kernel(
        self,
        kernel_id: str,
        assertion: KnowledgeAssertion
    ):
        """Add assertion to kernel"""
        if kernel_id in self._kernels:
            self._kernels[kernel_id]["assertions"].append(assertion)
    
    def infer_across_kernels(
        self,
        kernel_ids: List[str],
        target_conclusion: str
    ) -> InferenceResult:
        """
        Perform inference across multiple kernels
        
        Args:
            kernel_ids: Kernels to reason across
            target_conclusion: Conclusion to derive
            
        Returns:
            Inference result
        """
        start_time = time.time()
        
        # Collect all assertions
        all_assertions = []
        for kid in kernel_ids:
            if kid in self._kernels:
                all_assertions.extend(self._kernels[kid]["assertions"])
        
        # Build inference chain
        chain = self._build_inference_chain(all_assertions, target_conclusion)
        
        # Execute chain
        conclusions = self._execute_chain(chain)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Create result
        result = InferenceResult(
            conclusion=conclusions[0] if conclusions else "",
            reasoning_type=ReasoningType.DEDUCTIVE.value,
            confidence=self._calculate_chain_confidence(chain),
            reasoning_chain=[s.to_dict() if hasattr(s, 'to_dict') else {"step": s} for s in chain],
            intermediate_conclusions=conclusions,
            processing_time_ms=processing_time
        )
        
        self._inference_chains.append(chain)
        
        return result
    
    def _build_inference_chain(
        self,
        assertions: List[KnowledgeAssertion],
        target: str
    ) -> List[ReasoningStep]:
        """Build inference chain towards target"""
        chain = []
        step_number = 0
        
        # Group assertions by subject
        by_subject = defaultdict(list)
        for a in assertions:
            by_subject[a.subject].append(a)
        
        # Create extraction steps
        for subject, subj_assertions in by_subject.items():
            step_number += 1
            chain.append(ReasoningStep(
                step_id=f"step_{step_number}",
                step_number=step_number,
                module_name="extract",
                action="extract_assertions",
                input_data={"subject": subject},
                output_data={"assertions": [a.__dict__ for a in subj_assertions]},
                intermediate_conclusion=f"Found {len(subj_assertions)} assertions about {subject}"
            ))
        
        # Create inference steps
        if chain:
            step_number += 1
            chain.append(ReasoningStep(
                step_id=f"step_{step_number}",
                step_number=step_number,
                module_name="infer",
                action="derive_conclusion",
                input_data={"previous_steps": len(chain)},
                output_data={"target": target},
                intermediate_conclusion=f"Deriving conclusions for {target}"
            ))
        
        return chain
    
    def _execute_chain(
        self,
        chain: List[ReasoningStep]
    ) -> List[str]:
        """Execute inference chain"""
        conclusions = []
        
        for step in chain:
            if step.action == "derive_conclusion":
                # Simulate conclusion
                if step.intermediate_conclusion:
                    conclusions.append(step.intermediate_conclusion)
        
        return conclusions
    
    def _calculate_chain_confidence(self, chain: List[ReasoningStep]) -> float:
        """Calculate confidence of inference chain"""
        if not chain:
            return 0.0
        
        # Average confidence of steps
        avg_conf = sum(s.confidence for s in chain) / len(chain)
        
        # Penalize long chains
        length_penalty = max(0.5, 1.0 - len(chain) * 0.05)
        
        return avg_conf * length_penalty
    
    def get_kernel_summary(self, kernel_id: str) -> Dict[str, Any]:
        """Get summary of kernel contents"""
        if kernel_id not in self._kernels:
            return {}
        
        kernel = self._kernels[kernel_id]
        
        return {
            "kernel_id": kernel_id,
            "content_length": len(kernel["content"]),
            "assertion_count": len(kernel["assertions"]),
            "subjects": list(set(a.subject for a in kernel["assertions"])),
            "metadata": kernel["metadata"]
        }
    
    def list_kernels(self) -> List[str]:
        """List registered kernels"""
        return list(self._kernels.keys())


class NeuroSymbolicReasoner:
    """
    Main neuro-symbolic reasoning orchestrator.
    
    Combines all reasoning capabilities:
    - Neural module networks
    - Symbolic knowledge bases
    - Hybrid reasoning engine
    - Multi-step inference
    """
    
    def __init__(self):
        """Initialize neuro-symbolic reasoner"""
        self.hybrid_engine = HybridReasoningEngine()
        self.multi_step_engine = MultiStepInferenceEngine()
        
        # Cache
        self._result_cache: Dict[str, InferenceResult] = {}
    
    def reason(
        self,
        query: str,
        context: str = "",
        reasoning_type: ReasoningType = ReasoningType.DEDUCTIVE,
        cache_result: bool = True
    ) -> InferenceResult:
        """
        Perform neuro-symbolic reasoning
        
        Args:
            query: Question or task
            context: Supporting context
            reasoning_type: Type of reasoning
            cache_result: Whether to cache result
            
        Returns:
            Inference result
        """
        # Check cache
        cache_key = f"{query}:{reasoning_type.value}"
        if cache_result and cache_key in self._result_cache:
            return self._result_cache[cache_key]
        
        # Perform reasoning
        result = self.hybrid_engine.reason(query, context, reasoning_type)
        
        # Cache if enabled
        if cache_result:
            self._result_cache[cache_key] = result
        
        return result
    
    def reason_across_kernels(
        self,
        kernel_ids: List[str],
        query: str
    ) -> InferenceResult:
        """
        Reason across multiple knowledge kernels
        
        Args:
            kernel_ids: Kernels to reason across
            query: Query to answer
            
        Returns:
            Inference result
        """
        return self.multi_step_engine.infer_across_kernels(kernel_ids, query)
    
    def add_knowledge(
        self,
        assertions: Optional[List[KnowledgeAssertion]] = None,
        rules: Optional[List[SymbolicRule]] = None,
        kernel_id: Optional[str] = None,
        kernel_content: Optional[str] = None
    ):
        """Add knowledge to the reasoner"""
        self.hybrid_engine.add_knowledge(assertions, rules)
        
        if kernel_id and kernel_content:
            self.multi_step_engine.register_kernel(kernel_id, kernel_content)
            
            if assertions:
                for assertion in assertions:
                    self.multi_step_engine.add_assertion_to_kernel(kernel_id, assertion)
    
    def explain_reasoning(
        self,
        query: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """Get detailed explanation of reasoning process"""
        result = self.reason(query, context)
        
        return {
            "query": query,
            "conclusion": result.conclusion,
            "reasoning_type": result.reasoning_type,
            "confidence": result.confidence,
            "reasoning_chain": result.reasoning_chain,
            "supporting_evidence": result.supporting_evidence,
            "modules_used": result.modules_used,
            "rules_applied": result.rules_applied,
            "processing_time_ms": result.processing_time_ms
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get reasoning statistics"""
        history = self.hybrid_engine.get_reasoning_history()
        
        return {
            "total_reasoning_sessions": len(history),
            "average_confidence": sum(r["confidence"] for r in history) / max(len(history), 1),
            "average_processing_time_ms": sum(r["processing_time_ms"] for r in history) / max(len(history), 1),
            "registered_kernels": self.multi_step_engine.list_kernels(),
            "cached_results": len(self._result_cache)
        }


# Singleton instance
_reasoner: Optional[NeuroSymbolicReasoner] = None


def get_neuro_symbolic_reasoner() -> NeuroSymbolicReasoner:
    """Get neuro-symbolic reasoner singleton"""
    global _reasoner
    
    if _reasoner is None:
        _reasoner = NeuroSymbolicReasoner()
    
    return _reasoner


def init_neuro_symbolic_reasoning() -> NeuroSymbolicReasoner:
    """Initialize neuro-symbolic reasoning system"""
    global _reasoner
    
    _reasoner = NeuroSymbolicReasoner()
    
    return _reasoner
