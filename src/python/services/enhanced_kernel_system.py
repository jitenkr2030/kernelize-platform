#!/usr/bin/env python3
"""
Enhanced Knowledge Kernel System - Integration Module
======================================================

This module provides an integrated interface that combines all Priority 2
capabilities: persistent storage, query understanding, multi-hop reasoning,
and causal reasoning.

Author: MiniMax Agent
"""

import sys
import os
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Add source directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from services.storage import (
    PostgreSQLStorageBackend,
    KernelMetadata,
    KernelVersion,
    KernelExporter,
    KernelImporter,
    ExportOptions,
    ImportOptions,
    ExportFormat,
)
from services.query import (
    QueryUnderstandingPipeline,
    QueryIntent,
    QueryRewrite,
)
from services.reasoning import (
    MultiHopReasoningEngine,
    CausalReasoningEngine,
    ReasoningType,
    CausalQueryType,
    MultiHopResult,
    CausalExplanation,
)


class SystemState(Enum):
    """System initialization state"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


@dataclass
class SystemConfig:
    """System configuration"""
    # Storage
    postgres_connection: str = "postgresql://localhost:5432/kernel_db"
    
    # Vector store
    vector_store_type: str = "inmemory"  # or "qdrant"
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    
    # Cache
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    
    # Reasoning
    max_reasoning_hops: int = 5
    reasoning_timeout: int = 30
    
    # Export
    export_path: str = "./exports"
    encryption_enabled: bool = False


class EnhancedKernelSystem:
    """
    Enhanced Knowledge Kernel System with Priority 2 capabilities
    
    Integrates:
    - Persistent PostgreSQL storage
    - Import/Export functionality
    - Natural language query understanding
    - Multi-hop reasoning
    - Causal reasoning
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the enhanced system
        
        Args:
            config: System configuration
        """
        self.config = config or SystemConfig()
        self.state = SystemState.INITIALIZING
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all system components"""
        try:
            # Storage backend
            self.storage = PostgreSQLStorageBackend(
                connection_string=self.config.postgres_connection,
            )
            
            # Query understanding pipeline
            self.query_pipeline = QueryUnderstandingPipeline()
            
            # Multi-hop reasoning engine
            self.reasoning_engine = MultiHopReasoningEngine(
                vector_store=None,  # Would be connected to actual vector store
                knowledge_graph=None,
                embedding_model=None,
                max_hops=self.config.max_reasoning_hops,
                timeout_seconds=self.config.reasoning_timeout,
            )
            
            # Causal reasoning engine
            self.causal_engine = CausalReasoningEngine()
            
            # Import/Export
            self.exporter = KernelExporter(self.storage)
            self.importer = KernelImporter(self.storage)
            
            self.state = SystemState.READY
            print("✓ Enhanced Kernel System initialized successfully")
            
        except Exception as e:
            self.state = SystemState.ERROR
            print(f"✗ Initialization failed: {e}")
            raise
    
    # ==================== Kernel Management ====================
    
    def create_kernel(
        self,
        name: str,
        description: str = "",
        domain: str = "general",
        owner_id: str = "default",
    ) -> str:
        """
        Create a new knowledge kernel
        
        Args:
            name: Kernel name
            description: Kernel description
            domain: Knowledge domain
            owner_id: Owner identifier
            
        Returns:
            Kernel ID
        """
        metadata = KernelMetadata(
            name=name,
            description=description,
            domain=domain,
            owner_id=owner_id,
        )
        
        kernel_id = self.storage.create_kernel(metadata)
        print(f"✓ Created kernel: {name} ({kernel_id})")
        
        return kernel_id
    
    def get_kernel(self, kernel_id: str) -> Optional[Dict[str, Any]]:
        """Get kernel information"""
        return self.storage.get_kernel(kernel_id)
    
    def list_kernels(
        self,
        owner_id: Optional[str] = None,
        domain: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """List kernels with optional filtering"""
        return self.storage.list_kernels(owner_id, domain, limit)
    
    # ==================== Version Control ====================
    
    def commit_kernel(
        self,
        kernel_id: str,
        commit_message: str,
        created_by: str = "system",
    ) -> str:
        """
        Create a new version of a kernel
        
        Args:
            kernel_id: Kernel to commit
            commit_message: Version message
            created_by: Author of commit
            
        Returns:
            Version ID
        """
        # Get current state hash
        kernel = self.storage.get_kernel(kernel_id)
        docs = self.storage.get_documents_by_kernel(kernel_id)
        
        # Create content hash
        content = json.dumps({
            "kernel": kernel,
            "documents": [d["id"] for d in docs],
        }).encode()
        content_hash = hashlib.sha256(content).hexdigest()
        
        # Get change summary
        change_summary = {
            "documents_added": len(docs),
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        version_id = self.storage.create_version(
            kernel_id=kernel_id,
            commit_message=commit_message,
            content_hash=content_hash,
            change_summary=change_summary,
            created_by=created_by,
        )
        
        print(f"✓ Created version {version_id[:8]} for kernel {kernel_id[:8]}")
        
        return version_id
    
    def get_version_history(self, kernel_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Get version history for a kernel"""
        return self.storage.get_version_history(kernel_id, limit)
    
    def rollback_kernel(self, kernel_id: str, version_id: str) -> bool:
        """Rollback kernel to a previous version"""
        return self.storage.rollback_kernel(kernel_id, version_id)
    
    # ==================== Import/Export ====================
    
    def export_kernel(
        self,
        kernel_id: str,
        format: str = "json",
        compress: bool = True,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Export a kernel to file
        
        Args:
            kernel_id: Kernel to export
            format: Export format (json, jsonld, ttl, owl)
            compress: Whether to compress output
            output_path: Output file path
            
        Returns:
            Output file path
        """
        export_format = ExportFormat(format)
        
        options = ExportOptions(
            format=export_format,
            compression=compress,
            include_versions=True,
            include_relationships=True,
        )
        
        if not output_path:
            output_path = os.path.join(
                self.config.export_path,
                f"kernel_{kernel_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
            )
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = self.exporter.export_kernel_to_file(
            kernel_id=kernel_id,
            file_path=output_path,
            options=options,
        )
        
        if success:
            print(f"✓ Exported kernel to: {output_path}")
            return output_path
        else:
            raise RuntimeError("Export failed")
    
    def import_kernel(
        self,
        file_path: str,
        merge_strategy: str = "create_new",
        validation_mode: str = "strict",
    ) -> str:
        """
        Import a kernel from file
        
        Args:
            file_path: Input file path
            merge_strategy: How to handle conflicts
            validation_mode: Validation strictness
            
        Returns:
            Imported kernel ID
        """
        options = ImportOptions(
            merge_strategy=merge_strategy,
            validation_mode=validation_mode,
        )
        
        result = self.importer.import_kernel_from_file(file_path, options)
        
        if result.success:
            print(f"✓ Imported kernel: {result.kernel_id}")
            if result.warnings:
                for warning in result.warnings:
                    print(f"  ⚠ {warning}")
            return result.kernel_id
        else:
            for error in result.errors:
                print(f"✗ Import error: {error}")
            raise RuntimeError(f"Import failed: {result.errors}")
    
    # ==================== Query Processing ====================
    
    def understand_query(self, query: str) -> QueryRewrite:
        """
        Process and understand a natural language query
        
        Args:
            query: User query
            
        Returns:
            QueryRewrite with parsed intent and expansions
        """
        rewrite = self.query_pipeline.process(query)
        
        print(f"\nQuery Understanding Results:")
        print(f"  Original: {rewrite.original_query}")
        print(f"  Intent: {rewrite.intent.value}")
        print(f"  Confidence: {rewrite.confidence:.1%}")
        print(f"  Suggested top_k: {rewrite.suggested_top_k}")
        
        if rewrite.synonyms_added:
            print(f"  Synonyms: {', '.join(rewrite.synonyms_added[:5])}")
        
        if rewrite.temporal_constraints:
            print(f"  Temporal: {[tc.original_text for tc in rewrite.temporal_constraints]}")
        
        return rewrite
    
    # ==================== Reasoning ====================
    
    def reason_about_query(
        self,
        query: str,
        reasoning_type: str = "chain_of_thought",
        kernel_ids: Optional[List[str]] = None,
    ) -> MultiHopResult:
        """
        Execute multi-hop reasoning for a query
        
        Args:
            query: Complex query
            reasoning_type: Type of reasoning
            kernel_ids: Kernels to search
            
        Returns:
            MultiHopResult with answer and reasoning trace
        """
        rt = ReasoningType(reasoning_type)
        
        result = self.reasoning_engine.reason(
            query=query,
            kernel_ids=kernel_ids,
            reasoning_type=rt,
        )
        
        print(f"\nMulti-Hop Reasoning Results:")
        print(f"  Success: {result.success}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  Execution time: {result.execution_time_ms:.0f}ms")
        
        if result.errors:
            print(f"  Errors: {result.errors}")
        
        if result.trace:
            print(f"  Reasoning steps: {len(result.trace.reasoning_steps)}")
        
        return result
    
    def analyze_causality(
        self,
        query: str,
        text_context: Optional[str] = None,
    ) -> CausalExplanation:
        """
        Analyze causal relationships
        
        Args:
            query: Causal query
            text_context: Optional text to analyze
            
        Returns:
            CausalExplanation
        """
        if text_context:
            self.causal_engine.extract_and_build(text_context)
        
        explanation = self.causal_engine.explain_causality(query)
        
        print(f"\nCausal Analysis Results:")
        print(f"  Query type: {explanation.query_type.value}")
        print(f"  Confidence: {explanation.confidence:.1%}")
        print(f"  Direct causes: {len(explanation.direct_causes)}")
        print(f"  Indirect causes: {len(explanation.indirect_causes)}")
        print(f"  Confounders: {len(explanation.confounding_factors)}")
        
        if explanation.limitations:
            print(f"  Limitations: {len(explanation.limitations)}")
        
        return explanation
    
    def query_counterfactual(
        self,
        what_if_query: str,
    ) -> List[Any]:
        """
        Answer counterfactual queries
        
        Args:
            what_if_query: What-if scenario
            
        Returns:
            List of counterfactual scenarios
        """
        scenarios = self.causal_engine.query_counterfactual(what_if_query)
        
        print(f"\nCounterfactual Analysis:")
        for scenario in scenarios[:3]:
            print(f"  Scenario: {scenario.description[:50]}...")
            print(f"  Effect: {scenario.hypothetical_outcome}")
            print(f"  Probability change: {scenario.probability_change:.1%}")
            print(f"  Confidence: {scenario.confidence:.1%}")
        
        return scenarios
    
    # ==================== Statistics ====================
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        storage_stats = self.storage.get_storage_stats()
        
        return {
            "storage": storage_stats,
            "system_state": self.state.value,
            "config": {
                "embedding_model": self.config.embedding_model,
                "max_reasoning_hops": self.config.max_reasoning_hops,
            },
        }


# Convenience function
def create_enhanced_system(config: Optional[SystemConfig] = None) -> EnhancedKernelSystem:
    """Create and initialize enhanced kernel system"""
    return EnhancedKernelSystem(config)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Knowledge Kernel System - Demo")
    print("=" * 70)
    
    try:
        # Initialize system
        system = create_enhanced_system()
        
        # 1. Create a kernel
        kernel_id = system.create_kernel(
            name="Machine Learning Knowledge Base",
            description="Comprehensive ML and AI knowledge",
            domain="artificial_intelligence",
        )
        
        # 2. Create a version
        version_id = system.commit_kernel(
            kernel_id=kernel_id,
            commit_message="Initial knowledge base creation",
            created_by="demo_user",
        )
        
        # 3. Understand a query
        print("\n" + "-" * 70)
        print("Query Understanding Demo")
        print("-" * 70)
        
        rewrite = system.understand_query(
            "What are the key factors that caused the rapid adoption of deep learning since 2012?"
        )
        
        # 4. Multi-hop reasoning
        print("\n" + "-" * 70)
        print("Multi-Hop Reasoning Demo")
        print("-" * 70)
        
        result = system.reason_about_query(
            query="Who developed the transformer architecture and what company acquired the organization where they worked?",
            reasoning_type="chain_of_thought",
        )
        
        # 5. Causal analysis
        print("\n" + "-" * 70)
        print("Causal Reasoning Demo")
        print("-" * 70)
        
        explanation = system.analyze_causality(
            query="Why did GPU usage explode in deep learning?",
        )
        
        # 6. Counterfactual query
        print("\n" + "-" * 70)
        print("Counterfactual Query Demo")
        print("-" * 70)
        
        scenarios = system.query_counterfactual(
            what_if_query="What if deep learning had not been invented?",
        )
        
        # 7. Export kernel
        print("\n" + "-" * 70)
        print("Export Demo")
        print("-" * 70)
        
        export_path = system.export_kernel(kernel_id, format="json")
        
        # Get stats
        print("\n" + "-" * 70)
        print("System Statistics")
        print("-" * 70)
        
        stats = system.get_system_stats()
        print(f"Storage stats: {stats['storage']}")
        
        print("\n" + "=" * 70)
        print("Demo completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
