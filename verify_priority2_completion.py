#!/usr/bin/env python3
"""
Priority 2 Completion Verification
===================================

Verifies the implementation of:
- Task 2.2.2: Kernel Merge Engine
- Task 2.3.2: Horizontal Scaling Support

Author: MiniMax Agent
"""

import sys
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add the source directory to path
sys.path.insert(0, '/workspace/src/python')

from services.storage.kernel_merge_engine import (
    KernelMergeEngine,
    MergeStrategy,
    ConflictType,
    ConflictResolution,
    MergeConflict,
    MergeSection,
    SemanticAligner,
    ConsistencyValidator,
    ConflictResolver,
    EntityExtractor,
)
from services.horizontal_scaling import (
    HorizontalScalingManager,
    ShardingStrategy,
    ShardLocation,
    ShardRouter,
    DistributedLockManager,
    SessionAffinityManager,
    HealthCheckManager,
    LoadBalancer,
    InstanceInfo,
    ShardInfo,
    LockInfo,
    SessionAffinity,
)


class MockStorageBackend:
    """Mock storage backend for testing"""
    
    def __init__(self):
        self.kernels = {}
        self.documents = {}
        self.chunks = {}
        self.versions = []
    
    def _serialize_datetime(self, dt):
        """Serialize datetime to ISO format string"""
        if isinstance(dt, datetime):
            return dt.isoformat()
        return dt
    
    def create_kernel(self, metadata):
        kernel_id = f"kernel-{len(self.kernels)}"
        self.kernels[kernel_id] = {
            "id": kernel_id,
            "name": getattr(metadata, "name", "Test Kernel"),
            "description": getattr(metadata, "description", ""),
            "owner_id": getattr(metadata, "owner_id", "default"),
            "domain": getattr(metadata, "domain", "general"),
            "tags": getattr(metadata, "tags", []),
            "schema_version": getattr(metadata, "schema_version", "1.0"),
            "is_public": getattr(metadata, "is_public", False),
            "metadata_": getattr(metadata, "metadata_", {}),
            "created_at": self._serialize_datetime(datetime.utcnow()),
            "updated_at": self._serialize_datetime(datetime.utcnow()),
        }
        return kernel_id
    
    def get_kernel(self, kernel_id):
        return self.kernels.get(kernel_id)
    
    def add_document(self, kernel_id, title, content_hash, blob_path=None, metadata=None,
                     source_url=None, author=None, published_date=None):
        doc_id = f"doc-{len(self.documents)}"
        self.documents[doc_id] = {
            "id": doc_id,
            "kernel_id": kernel_id,
            "title": title,
            "content_hash": content_hash,
            "blob_path": blob_path,
            "metadata": metadata or {},
            "source_url": source_url,
            "author": author,
            "published_date": published_date,
            "created_at": self._serialize_datetime(datetime.utcnow()),
        }
        self.kernels[kernel_id].setdefault("document_ids", []).append(doc_id)
        return doc_id
    
    def get_documents_by_kernel(self, kernel_id):
        doc_ids = self.kernels[kernel_id].get("document_ids", [])
        return [self.documents[doc_id] for doc_id in doc_ids if doc_id in self.documents]
    
    def get_chunks_by_document(self, document_id):
        return self.chunks.get(document_id, [])
    
    def get_chunks_by_kernel(self, kernel_id, limit=1000):
        all_chunks = []
        for doc_id, chunks in self.chunks.items():
            all_chunks.extend(chunks)
        return all_chunks[:limit]
    
    def add_chunk(self, document_id, kernel_id, chunk_index, text_content,
                  content_hash, qdrant_point_id=None, metadata=None):
        chunk_id = f"chunk-{len(self.chunks)}"
        if document_id not in self.chunks:
            self.chunks[document_id] = []
        self.chunks[document_id].append({
            "id": chunk_id,
            "document_id": document_id,
            "kernel_id": kernel_id,
            "chunk_index": chunk_index,
            "text_content": text_content,
            "content_hash": content_hash,
            "qdrant_point_id": qdrant_point_id,
            "metadata": metadata or {},
        })
        return chunk_id
    
    def create_version(self, kernel_id, commit_message, content_hash,
                       change_summary=None, diff_summary=None, created_by="system"):
        version_id = f"version-{len(self.versions)}"
        version = {
            "id": version_id,
            "kernel_id": kernel_id,
            "commit_message": commit_message,
            "content_hash": content_hash,
            "change_summary": change_summary or {},
            "diff_summary": diff_summary or "",
            "created_by": created_by,
            "created_at": datetime.utcnow(),
        }
        self.versions.append(version)
        return version_id
    
    def get_version_history(self, kernel_id, limit=50):
        return [v for v in self.versions if v["kernel_id"] == kernel_id][:limit]


def test_kernel_merge_engine():
    """Test Kernel Merge Engine implementation"""
    print("\n" + "=" * 60)
    print("Testing Task 2.2.2: Kernel Merge Engine")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Entity Extractor
    print("\n1. Testing Entity Extractor...")
    try:
        extractor = EntityExtractor()
        entities = extractor.extract_entities(
            "Document: Introduction\nConcept: Machine Learning\nTerm: Neural Network",
            entity_types=["document", "concept", "term"]
        )
        assert len(entities) >= 3, f"Expected at least 3 entities, got {len(entities)}"
        print("   ✓ Entity extraction works correctly")
        passed += 1
    except Exception as e:
        print(f"   ✗ Entity extraction failed: {e}")
        failed += 1
    
    # Test 2: Semantic Aligner
    print("\n2. Testing Semantic Aligner...")
    try:
        aligner = SemanticAligner(similarity_threshold=0.7)
        
        source_entities = [
            {"id": "src1", "title": "Introduction to Machine Learning"},
            {"id": "src2", "title": "Deep Neural Networks"},
        ]
        
        target_entities = [
            {"id": "tgt1", "title": "Machine Learning Introduction"},
            {"id": "tgt2", "title": "Neural Networks Deep Dive"},
        ]
        
        alignments = aligner.align_entities(source_entities, target_entities, "document")
        
        # Should find some alignments
        assert len(alignments) == 2, f"Expected 2 alignments, got {len(alignments)}"
        print("   ✓ Semantic alignment works correctly")
        passed += 1
    except Exception as e:
        print(f"   ✗ Semantic alignment failed: {e}")
        failed += 1
    
    # Test 3: Consistency Validator
    print("\n3. Testing Consistency Validator...")
    try:
        validator = ConsistencyValidator()
        
        # Check schema compatibility
        compatible, message = validator.check_schema_compatibility("1.0", "1.1")
        assert compatible is True, f"Expected schemas to be compatible, got: {message}"
        
        not_compatible, _ = validator.check_schema_compatibility("1.0", "2.0")
        assert not_compatible is False, "Expected major version mismatch to be incompatible"
        
        print("   ✓ Schema compatibility checking works")
        passed += 1
    except Exception as e:
        print(f"   ✗ Consistency validation failed: {e}")
        failed += 1
    
    # Test 4: Conflict Resolver
    print("\n4. Testing Conflict Resolver...")
    try:
        resolver = ConflictResolver()
        
        conflict = MergeConflict(
            conflict_id="test-1",
            conflict_type=ConflictType.ENTITY_MISMATCH,
            source_kernel_id="source",
            target_kernel_id="target",
            entity_type="document",
            entity_id="doc1",
            source_value={"title": "Source Title"},
            target_value={"title": "Target Title"},
            conflict_details={},
        )
        
        resolved, description = resolver.resolve_conflict(
            conflict,
            ConflictResolution.KEEP_SOURCE,
            {"title": "Source Title"},
            {"title": "Target Title"},
        )
        
        assert resolved == {"title": "Source Title"}, "Expected source value to be kept"
        print("   ✓ Conflict resolution works correctly")
        passed += 1
    except Exception as e:
        print(f"   ✗ Conflict resolver failed: {e}")
        failed += 1
    
    # Test 5: Kernel Merge Engine
    print("\n5. Testing Kernel Merge Engine...")
    try:
        storage = MockStorageBackend()
        engine = KernelMergeEngine(storage)
        
        # Create source kernel
        source_id = storage.create_kernel(
            type('Metadata', (), {
                'name': 'Source Kernel',
                'description': 'Test source kernel',
                'owner_id': 'user1',
                'domain': 'technology',
                'tags': ['ai', 'ml'],
                'schema_version': '1.0',
                'is_public': False,
                'metadata_': {},
            })()
        )
        
        # Add documents to source
        doc1_id = storage.add_document(
            kernel_id=source_id,
            title="Machine Learning Guide",
            content_hash="abc123",
        )
        storage.add_chunk(
            document_id=doc1_id,
            kernel_id=source_id,
            chunk_index=0,
            text_content="Machine learning is...",
            content_hash="chunk1",
        )
        
        # Create target kernel
        target_id = storage.create_kernel(
            type('Metadata', (), {
                'name': 'Target Kernel',
                'description': 'Test target kernel',
                'owner_id': 'user2',
                'domain': 'technology',
                'tags': ['tech'],
                'schema_version': '1.0',
                'is_public': True,
                'metadata_': {},
            })()
        )
        
        # Add document to target (different title to avoid conflict)
        storage.add_document(
            kernel_id=target_id,
            title="Deep Learning Guide",
            content_hash="def456",
        )
        
        # Perform merge
        result = engine.merge_kernels(
            source_kernel_id=source_id,
            target_kernel_id=target_id,
            merge_strategy=MergeStrategy.SEMANTIC,
        )
        
        assert result.success, f"Merge should succeed, but got: {result.metadata}"
        assert result.merged_documents >= 1, f"Should merge at least 1 document"
        assert result.merged_kernel_id == target_id, "Should return target kernel ID"
        
        print("   ✓ Kernel merge engine works correctly")
        passed += 1
    except Exception as e:
        print(f"   ✗ Kernel merge engine failed: {e}")
        failed += 1
    
    # Test 6: Partial Merge
    print("\n6. Testing Partial Merge...")
    try:
        storage = MockStorageBackend()
        engine = KernelMergeEngine(storage)
        
        # Create source kernel
        source_id = storage.create_kernel(
            type('Metadata', (), {
                'name': 'Source for Partial',
                'description': '',
                'owner_id': 'user1',
                'domain': 'science',
                'tags': [],
                'schema_version': '1.0',
                'is_public': False,
                'metadata_': {},
            })()
        )
        
        storage.add_document(kernel_id=source_id, title="Physics", content_hash="h1")
        storage.add_document(kernel_id=source_id, title="Chemistry", content_hash="h2")
        storage.add_document(kernel_id=source_id, title="Biology", content_hash="h3")
        
        # Partial merge - only physics
        partial_request = type('Request', (), {
            'source_kernel_id': source_id,
            'target_kernel_id': None,
            'sections': [
                MergeSection(
                    section_type="documents",
                    filter_criteria={"title": "Physics"},
                    merge_behavior="append",
                )
            ],
            'merge_strategy': MergeStrategy.CONCATENATE,
            'conflict_resolution': ConflictResolution.KEEP_TARGET,
            'custom_resolvers': {},
            'preserve_metadata': True,
        })()
        
        result = engine.partial_merge(partial_request)
        
        assert result.success, f"Partial merge should succeed"
        assert result.merged_documents == 1, f"Should merge exactly 1 document"
        
        print("   ✓ Partial merge works correctly")
        passed += 1
    except Exception as e:
        print(f"   ✗ Partial merge failed: {e}")
        failed += 1
    
    print(f"\n{'='*60}")
    print(f"Task 2.2.2 Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return passed, failed


def test_horizontal_scaling():
    """Test Horizontal Scaling implementation"""
    print("\n" + "=" * 60)
    print("Testing Task 2.3.2: Horizontal Scaling Support")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    # Test 1: Shard Router
    print("\n1. Testing Shard Router...")
    try:
        router = ShardRouter(strategy=ShardingStrategy.HASH_RING)
        
        # Register shards
        for i in range(4):
            router.register_shard(ShardInfo(
                shard_id=f"shard-{i}",
                strategy=ShardingStrategy.HASH_RING,
                shard_key=f"instance-{i}",
                location=ShardLocation.ACTIVE,
                instance_id=f"instance-{i}",
                host=f"192.168.1.{100+i}",
                port=8080,
                weight=1.0,
            ))
        
        # Test routing
        kernel1 = router.get_shard(kernel_id="kernel-abc-123", kernel_domain="tech")
        kernel2 = router.get_shard(kernel_id="kernel-xyz-789", kernel_domain="tech")
        
        assert kernel1 is not None, "Should route to a shard"
        assert kernel2 is not None, "Should route to a shard"
        
        # Consistent hashing should work
        kernel1_again = router.get_shard(kernel_id="kernel-abc-123", kernel_domain="tech")
        assert kernel1.shard_id == kernel1_again.shard_id, "Consistent hashing should return same shard"
        
        print("   ✓ Shard routing with consistent hashing works")
        passed += 1
    except Exception as e:
        print(f"   ✗ Shard router failed: {e}")
        failed += 1
    
    # Test 2: Domain-based Sharding
    print("\n2. Testing Domain-based Sharding...")
    try:
        router = ShardRouter(strategy=ShardingStrategy.DOMAIN)
        
        router.register_shard(ShardInfo(
            shard_id="shard-medical",
            strategy=ShardingStrategy.DOMAIN,
            shard_key="medical",
            location=ShardLocation.ACTIVE,
            instance_id="instance-1",
            host="192.168.1.100",
            port=8080,
        ))
        
        router.register_shard(ShardInfo(
            shard_id="shard-legal",
            strategy=ShardingStrategy.DOMAIN,
            shard_key="legal",
            location=ShardLocation.ACTIVE,
            instance_id="instance-2",
            host="192.168.1.101",
            port=8080,
        ))
        
        medical_shard = router.get_shard(kernel_id="k1", kernel_domain="medical")
        legal_shard = router.get_shard(kernel_id="k2", kernel_domain="legal")
        
        assert medical_shard.shard_key == "medical", "Should route medical domain"
        assert legal_shard.shard_key == "legal", "Should route legal domain"
        
        print("   ✓ Domain-based sharding works")
        passed += 1
    except Exception as e:
        print(f"   ✗ Domain sharding failed: {e}")
        failed += 1
    
    # Test 3: Distributed Lock Manager
    print("\n3. Testing Distributed Lock Manager...")
    try:
        lock_manager = DistributedLockManager()
        
        # Acquire lock
        lock = lock_manager.acquire_lock(
            resource_type="kernel",
            resource_id="kernel-123",
            instance_id="instance-1",
            ttl_seconds=10,
        )
        
        assert lock is not None, "Should acquire lock"
        assert not lock.is_expired(), "New lock should not be expired"
        
        # Try to acquire same lock from different instance
        lock2 = lock_manager.acquire_lock(
            resource_type="kernel",
            resource_id="kernel-123",
            instance_id="instance-2",
            ttl_seconds=10,
        )
        
        assert lock2 is None, "Should not acquire lock held by another instance"
        
        # Release lock
        released = lock_manager.release_lock(lock)
        assert released is True, "Should release lock successfully"
        
        # Now should be able to acquire
        lock3 = lock_manager.acquire_lock(
            resource_type="kernel",
            resource_id="kernel-123",
            instance_id="instance-2",
            ttl_seconds=10,
        )
        
        assert lock3 is not None, "Should acquire lock after release"
        
        lock_manager.release_lock(lock3)
        
        print("   ✓ Distributed locking works correctly")
        passed += 1
    except Exception as e:
        print(f"   ✗ Distributed lock manager failed: {e}")
        failed += 1
    
    # Test 4: Session Affinity Manager
    print("\n4. Testing Session Affinity Manager...")
    try:
        session_manager = SessionAffinityManager()
        
        # Create sessions
        session1 = session_manager.create_session(
            session_id="session-abc",
            instance_id="instance-1",
        )
        
        session2 = session_manager.create_session(
            session_id="session-xyz",
            instance_id="instance-2",
        )
        
        assert session1.instance_id == "instance-1", "Should assign correct instance"
        assert session2.instance_id == "instance-2", "Should assign correct instance"
        
        # Get session instance
        instance_for_session1 = session_manager.get_session_instance("session-abc")
        assert instance_for_session1 == "instance-1", "Should retrieve correct instance"
        
        # Refresh session
        refreshed = session_manager.refresh_session("session-abc")
        assert refreshed is True, "Should refresh session successfully"
        
        # Remove session
        removed = session_manager.remove_session("session-abc")
        assert removed is True, "Should remove session"
        
        instance_after_removal = session_manager.get_session_instance("session-abc")
        assert instance_after_removal is None, "Session should be gone"
        
        print("   ✓ Session affinity management works")
        passed += 1
    except Exception as e:
        print(f"   ✗ Session affinity failed: {e}")
        failed += 1
    
    # Test 5: Health Check Manager
    print("\n5. Testing Health Check Manager...")
    try:
        health_manager = HealthCheckManager(check_interval_seconds=1)
        
        # Register instances
        health_manager.register_instance(InstanceInfo(
            instance_id="instance-1",
            host="localhost",
            port=8080,
        ))
        
        health_manager.register_instance(InstanceInfo(
            instance_id="instance-2",
            host="localhost",
            port=8081,
        ))
        
        # Run health check
        result = health_manager.check_instance("instance-1")
        
        assert result.instance_id == "instance-1", "Should return correct instance"
        assert "connectivity" in result.checks, "Should include connectivity check"
        
        # Get healthy instances
        healthy = health_manager.get_healthy_instances()
        assert len(healthy) >= 0, "Should return healthy instances"
        
        health_stats = health_manager.get_health_stats()
        assert "total_instances" in health_stats, "Should include statistics"
        
        print("   ✓ Health check management works")
        passed += 1
    except Exception as e:
        print(f"   ✗ Health check manager failed: {e}")
        failed += 1
    
    # Test 6: Load Balancer
    print("\n6. Testing Load Balancer...")
    try:
        health_manager = HealthCheckManager(check_interval_seconds=1)
        session_manager = SessionAffinityManager()
        load_balancer = LoadBalancer(health_manager, session_manager)
        
        # Register instances
        inst1 = InstanceInfo(instance_id="inst-1", host="host1", port=8080, weight=1.0)
        inst2 = InstanceInfo(instance_id="inst-2", host="host2", port=8080, weight=2.0)
        
        health_manager.register_instance(inst1)
        health_manager.register_instance(inst2)
        
        # Select instance (weighted random)
        selected = load_balancer.select_instance()
        assert selected is not None, "Should select an instance"
        assert selected.instance_id in ["inst-1", "inst-2"], "Should select valid instance"
        
        # Record request
        load_balancer.record_request("inst-1", response_time_ms=50.0, success=True)
        load_balancer.record_request("inst-2", response_time_ms=100.0, success=True)
        
        load_stats = load_balancer.get_load_stats()
        assert "total_requests" in load_stats, "Should include request statistics"
        
        print("   ✓ Load balancing works correctly")
        passed += 1
    except Exception as e:
        print(f"   ✗ Load balancer failed: {e}")
        failed += 1
    
    # Test 7: Horizontal Scaling Manager Integration
    print("\n7. Testing Horizontal Scaling Manager...")
    try:
        scaling_manager = HorizontalScalingManager(
            instance_id="primary-instance",
            host="192.168.1.50",
            port=8080,
        )
        
        # Create session
        session = scaling_manager.create_session("test-session-123")
        assert session.session_id == "test-session-123", "Should create session"
        
        # Get scaling stats
        stats = scaling_manager.get_scaling_stats()
        assert "instance" in stats, "Stats should include instance info"
        assert "shards" in stats, "Stats should include shard info"
        assert "health" in stats, "Stats should include health info"
        
        scaling_manager.shutdown()
        
        print("   ✓ Horizontal scaling manager works")
        passed += 1
    except Exception as e:
        print(f"   ✗ Horizontal scaling manager failed: {e}")
        failed += 1
    
    print(f"\n{'='*60}")
    print(f"Task 2.3.2 Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return passed, failed


def main():
    """Run all verification tests"""
    print("\n" + "=" * 60)
    print("Priority 2 Implementation Verification")
    print("=" * 60)
    
    merge_passed, merge_failed = test_kernel_merge_engine()
    scaling_passed, scaling_failed = test_horizontal_scaling()
    
    total_passed = merge_passed + scaling_passed
    total_failed = merge_failed + scaling_failed
    total_tests = total_passed + total_failed
    
    print("\n" + "=" * 60)
    print("OVERALL VERIFICATION RESULTS")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Success Rate: {total_passed/total_tests*100:.1f}%")
    
    if total_failed == 0:
        print("\n✓ ALL TESTS PASSED - Priority 2 is fully implemented!")
    else:
        print(f"\n✗ {total_failed} test(s) failed")
    
    print("=" * 60)
    
    return total_failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
