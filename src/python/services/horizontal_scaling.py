#!/usr/bin/env python3
"""
Horizontal Scaling Support Module
==================================

Enables horizontal scaling for kernel operations with:
- Kernel sharding strategies (domain-based, date-based, content-hash-based)
- Distributed locking for concurrent kernel updates
- Session affinity for query consistency
- Load balancing with health checks
- Instance registration and discovery

Author: MiniMax Agent
"""

import json
import hashlib
import uuid
import time
import threading
import socket
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
from contextlib import contextmanager
import logging
import random

logger = logging.getLogger(__name__)


class ShardingStrategy(Enum):
    """Available sharding strategies"""
    DOMAIN = "domain"  # Shard by kernel domain
    DATE = "date"  # Shard by creation date
    CONTENT_HASH = "content_hash"  # Shard by content hash
    HASH_RING = "hash_ring"  # Consistent hashing
    GEOGRAPHIC = "geographic"  # Shard by geographic region
    CUSTOM = "custom"  # Custom sharding function


class ShardLocation(Enum):
    """Status of a shard"""
    ACTIVE = "active"  # Shard is active and serving requests
    MIGRATING = "migrating"  # Shard data is being migrated
    OFFLINE = "offline"  # Shard is temporarily offline
    DECOMMISSIONED = "decommissioned"  # Shard is being removed


@dataclass
class ShardInfo:
    """Information about a kernel shard"""
    shard_id: str
    strategy: ShardingStrategy
    shard_key: str  # The key used for routing (domain name, date, etc.)
    location: ShardLocation
    instance_id: str  # Instance serving this shard
    host: str
    port: int
    weight: float = 1.0  # Load balancing weight
    load_score: float = 0.0  # Current load (0-1)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "shard_id": self.shard_id,
            "strategy": self.strategy.value,
            "shard_key": self.shard_key,
            "location": self.location.value,
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "weight": self.weight,
            "load_score": self.load_score,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class InstanceInfo:
    """Information about a scaling instance"""
    instance_id: str
    host: str
    port: int
    is_leader: bool = False
    status: str = "healthy"
    capabilities: List[str] = field(default_factory=list)
    weight: float = 1.0  # Load balancing weight
    started_at: datetime = field(default_factory=datetime.now)
    last_heartbeat: datetime = field(default_factory=datetime.now)
    request_count: int = 0
    error_count: int = 0
    avg_response_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "host": self.host,
            "port": self.port,
            "is_leader": self.is_leader,
            "status": self.status,
            "capabilities": self.capabilities,
            "weight": self.weight,
            "started_at": self.started_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "request_count": self.request_count,
            "error_count": self.error_count,
            "avg_response_time_ms": self.avg_response_time_ms,
        }


@dataclass
class LockInfo:
    """Information about a distributed lock"""
    lock_id: str
    resource_type: str  # "kernel", "document", "chunk"
    resource_id: str
    instance_id: str
    acquired_at: datetime
    expires_at: datetime
    ttl_seconds: int
    is_extended: bool = False
    
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "lock_id": self.lock_id,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "instance_id": self.instance_id,
            "acquired_at": self.acquired_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "is_extended": self.is_extended,
        }


@dataclass
class SessionAffinity:
    """Session affinity configuration"""
    session_id: str
    instance_id: str
    created_at: datetime
    last_accessed: datetime
    ttl_seconds: int = 3600  # 1 hour default
    affinity_strength: float = 1.0  # 0-1, higher means stronger affinity
    
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.last_accessed + timedelta(seconds=self.ttl_seconds)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "instance_id": self.instance_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "ttl_seconds": self.ttl_seconds,
            "affinity_strength": self.affinity_strength,
        }


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    instance_id: str
    is_healthy: bool
    checks: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    latency_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "is_healthy": self.is_healthy,
            "checks": self.checks,
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": self.latency_ms,
        }


class ShardRouter:
    """
    Routes kernel operations to appropriate shards based on sharding strategy
    """
    
    def __init__(
        self,
        strategy: ShardingStrategy = ShardingStrategy.HASH_RING,
        shard_count: int = 16,
    ):
        """
        Initialize shard router
        
        Args:
            strategy: Default sharding strategy
            shard_count: Number of virtual shards for hash ring
        """
        self.strategy = strategy
        self.shard_count = shard_count
        
        # Shard registry
        self._shards: Dict[str, ShardInfo] = {}
        
        # Hash ring for consistent hashing
        self._hash_ring: Dict[int, str] = {}  # hash -> shard_id
        
        # Shard key to shard ID mapping
        self._key_to_shard: Dict[str, str] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def register_shard(self, shard: ShardInfo):
        """Register a new shard"""
        with self._lock:
            self._shards[shard.shard_id] = shard
            
            # Update key mapping
            self._key_to_shard[shard.shard_key] = shard.shard_id
            
            # Update hash ring
            if self.strategy == ShardingStrategy.HASH_RING:
                self._rebuild_hash_ring()
    
    def unregister_shard(self, shard_id: str):
        """Unregister a shard"""
        with self._lock:
            if shard_id in self._shards:
                shard = self._shards[shard_id]
                del self._shards[shard_id]
                del self._key_to_shard[shard.shard_key]
                
                if self.strategy == ShardingStrategy.HASH_RING:
                    self._rebuild_hash_ring()
    
    def _rebuild_hash_ring(self):
        """Rebuild the hash ring"""
        self._hash_ring.clear()
        
        for shard_id, shard in self._shards.items():
            if shard.location == ShardLocation.ACTIVE:
                # Create multiple virtual nodes for better distribution
                for i in range(self.shard_count):
                    key = f"{shard_id}:{i}"
                    hash_value = self._hash_key(key)
                    self._hash_ring[hash_value] = shard_id
    
    def _hash_key(self, key: str) -> int:
        """Hash a key to an integer"""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)
    
    def get_shard(
        self,
        kernel_id: str,
        kernel_domain: str = "general",
        kernel_date: Optional[datetime] = None,
        strategy: Optional[ShardingStrategy] = None,
    ) -> Optional[ShardInfo]:
        """
        Get the appropriate shard for a kernel
        
        Args:
            kernel_id: Kernel ID to route
            kernel_domain: Kernel domain for domain-based sharding
            kernel_date: Kernel creation date for date-based sharding
            strategy: Override sharding strategy
            
        Returns:
            ShardInfo or None if no suitable shard found
        """
        use_strategy = strategy or self.strategy
        
        with self._lock:
            if use_strategy == ShardingStrategy.DOMAIN:
                return self._get_shard_by_domain(kernel_domain)
            elif use_strategy == ShardingStrategy.DATE:
                return self._get_shard_by_date(kernel_date or datetime.utcnow())
            elif use_strategy == ShardingStrategy.CONTENT_HASH:
                return self._get_shard_by_content_hash(kernel_id)
            elif use_strategy == ShardingStrategy.HASH_RING:
                return self._get_shard_by_hash_ring(kernel_id)
            else:
                return self._get_shard_by_domain(kernel_domain)
    
    def _get_shard_by_domain(self, domain: str) -> Optional[ShardInfo]:
        """Get shard by domain"""
        if domain in self._key_to_shard:
            shard_id = self._key_to_shard[domain]
            return self._shards.get(shard_id)
        
        # Default to first available shard
        return self._get_first_available_shard()
    
    def _get_shard_by_date(self, date: datetime) -> Optional[ShardInfo]:
        """Get shard by date (monthly buckets)"""
        date_key = date.strftime("%Y-%m")  # e.g., "2024-01"
        return self._get_shard_by_domain(date_key)
    
    def _get_shard_by_content_hash(self, kernel_id: str) -> Optional[ShardInfo]:
        """Get shard by content hash"""
        # Use consistent hashing
        return self._get_shard_by_hash_ring(kernel_id)
    
    def _get_shard_by_hash_ring(self, key: str) -> Optional[ShardInfo]:
        """Get shard using consistent hashing"""
        if not self._hash_ring:
            return self._get_first_available_shard()
        
        key_hash = self._hash_key(key)
        
        # Find the first hash >= key_hash, or wrap around to the smallest
        sorted_hashes = sorted(self._hash_ring.keys())
        
        for h in sorted_hashes:
            if h >= key_hash:
                shard_id = self._hash_ring[h]
                return self._shards.get(shard_id)
        
        # Wrap around
        if sorted_hashes:
            first_hash = sorted_hashes[0]
            shard_id = self._hash_ring[first_hash]
            return self._shards.get(shard_id)
        
        return None
    
    def _get_first_available_shard(self) -> Optional[ShardInfo]:
        """Get the first available shard"""
        for shard in self._shards.values():
            if shard.location == ShardLocation.ACTIVE:
                return shard
        return None
    
    def get_all_shards(self, status: Optional[ShardLocation] = None) -> List[ShardInfo]:
        """Get all registered shards"""
        with self._lock:
            shards = list(self._shards.values())
            if status:
                shards = [s for s in shards if s.location == status]
            return shards
    
    def get_shard_distribution(self) -> Dict[str, int]:
        """Get distribution of kernels across shards"""
        distribution = defaultdict(int)
        with self._lock:
            for shard in self._shards.values():
                if shard.location == ShardLocation.ACTIVE:
                    distribution[shard.shard_key] += int(shard.weight * 100)
        return dict(distribution)


class DistributedLockManager:
    """
    Manages distributed locks for concurrent kernel operations
    """
    
    def __init__(
        self,
        redis_client=None,
        default_ttl_seconds: int = 30,
        heartbeat_interval: int = 10,
    ):
        """
        Initialize distributed lock manager
        
        Args:
            redis_client: Redis client for lock storage (None for in-memory)
            default_ttl_seconds: Default lock TTL
            heartbeat_interval: Lock heartbeat interval
        """
        self.redis = redis_client
        self.default_ttl = default_ttl_seconds
        self.heartbeat_interval = heartbeat_interval
        
        # In-memory lock storage (fallback when Redis unavailable)
        self._locks: Dict[str, LockInfo] = {}
        
        # Lock owners for tracking
        self._owner_locks: Dict[str, Set[str]] = defaultdict(set)
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Background thread for lock expiration
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_expired_locks)
        self._cleanup_thread.daemon = True
        self._cleanup_thread.start()
    
    def acquire_lock(
        self,
        resource_type: str,
        resource_id: str,
        instance_id: str,
        ttl_seconds: Optional[int] = None,
        blocking: bool = False,
        blocking_timeout_seconds: float = 5.0,
    ) -> Optional[LockInfo]:
        """
        Acquire a distributed lock
        
        Args:
            resource_type: Type of resource (kernel, document, chunk)
            resource_id: ID of the resource
            instance_id: ID of the instance acquiring the lock
            ttl_seconds: Lock TTL in seconds
            blocking: Whether to block waiting for the lock
            blocking_timeout_seconds: Maximum time to wait
            
        Returns:
            LockInfo if acquired, None if not available
        """
        lock_key = f"{resource_type}:{resource_id}"
        ttl = ttl_seconds or self.default_ttl
        
        start_time = time.time()
        
        while True:
            with self._lock:
                # Check if lock exists
                if lock_key in self._locks:
                    existing_lock = self._locks[lock_key]
                    
                    if existing_lock.is_expired():
                        # Lock expired, can acquire
                        return self._create_lock(
                            lock_key, resource_type, resource_id, instance_id, ttl
                        )
                    
                    # Lock exists and is valid
                    if blocking and time.time() - start_time < blocking_timeout_seconds:
                        time.sleep(0.1)  # Wait and retry
                        continue
                    else:
                        return None
                else:
                    # No lock exists
                    return self._create_lock(
                        lock_key, resource_type, resource_id, instance_id, ttl
                    )
    
    def _create_lock(
        self,
        lock_key: str,
        resource_type: str,
        resource_id: str,
        instance_id: str,
        ttl_seconds: int,
    ) -> LockInfo:
        """Create a new lock"""
        now = datetime.utcnow()
        lock = LockInfo(
            lock_id=str(uuid.uuid4()),
            resource_type=resource_type,
            resource_id=resource_id,
            instance_id=instance_id,
            acquired_at=now,
            expires_at=now + timedelta(seconds=ttl_seconds),
            ttl_seconds=ttl_seconds,
        )
        
        self._locks[lock_key] = lock
        self._owner_locks[instance_id].add(lock_key)
        
        logger.debug(f"Acquired lock {lock.lock_id} for {lock_key}")
        return lock
    
    def release_lock(self, lock: LockInfo) -> bool:
        """
        Release a lock
        
        Args:
            lock: Lock to release
            
        Returns:
            True if released, False if not owner
        """
        with self._lock:
            lock_key = f"{lock.resource_type}:{lock.resource_id}"
            
            if lock_key in self._locks:
                existing = self._locks[lock_key]
                if existing.lock_id == lock.lock_id:
                    del self._locks[lock_key]
                    self._owner_locks[lock.instance_id].discard(lock_key)
                    
                    logger.debug(f"Released lock {lock.lock_id}")
                    return True
            
            return False
    
    def extend_lock(
        self,
        lock: LockInfo,
        additional_seconds: int = 30,
    ) -> bool:
        """
        Extend a lock's TTL
        
        Args:
            lock: Lock to extend
            additional_seconds: Seconds to add to TTL
            
        Returns:
            True if extended, False if not owner or expired
        """
        with self._lock:
            lock_key = f"{lock.resource_type}:{lock.resource_id}"
            
            if lock_key in self._locks:
                existing = self._locks[lock_key]
                if existing.lock_id == lock.lock_id and not existing.is_expired():
                    existing.expires_at = datetime.utcnow() + timedelta(
                        seconds=additional_seconds
                    )
                    existing.is_extended = True
                    
                    logger.debug(f"Extended lock {lock.lock_id}")
                    return True
            
            return False
    
    def get_lock(self, resource_type: str, resource_id: str) -> Optional[LockInfo]:
        """Get lock info for a resource"""
        lock_key = f"{resource_type}:{resource_id}"
        with self._lock:
            return self._locks.get(lock_key)
    
    def get_locks_by_instance(self, instance_id: str) -> List[LockInfo]:
        """Get all locks held by an instance"""
        with self._lock:
            lock_keys = self._owner_locks.get(instance_id, set())
            return [self._locks[k] for k in lock_keys if k in self._locks]
    
    def release_instance_locks(self, instance_id: str):
        """Release all locks held by an instance (for cleanup)"""
        with self._lock:
            lock_keys = list(self._owner_locks.get(instance_id, set()))
            for lock_key in lock_keys:
                if lock_key in self._locks:
                    lock = self._locks[lock_key]
                    if lock.instance_id == instance_id:
                        del self._locks[lock_key]
            
            self._owner_locks[instance_id].clear()
    
    def _cleanup_expired_locks(self):
        """Background task to clean up expired locks"""
        while self._running:
            time.sleep(1)
            
            with self._lock:
                expired_keys = [
                    k for k, lock in self._locks.items()
                    if lock.is_expired()
                ]
                
                for key in expired_keys:
                    lock = self._locks[key]
                    self._owner_locks[lock.instance_id].discard(key)
                    del self._locks[key]
                    
                    logger.debug(f"Cleaned up expired lock {key}")
    
    def shutdown(self):
        """Shutdown the lock manager"""
        self._running = False
        if self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
    
    def get_lock_stats(self) -> Dict[str, Any]:
        """Get lock manager statistics"""
        with self._lock:
            active_locks = len(self._locks)
            instances_with_locks = len(self._owner_locks)
            
            expired_count = sum(
                1 for lock in self._locks.values() if lock.is_expired()
            )
            
            return {
                "active_locks": active_locks,
                "instances_with_locks": instances_with_locks,
                "expired_locks": expired_count,
            }


class SessionAffinityManager:
    """
    Manages session affinity for consistent routing
    """
    
    def __init__(
        self,
        default_ttl_seconds: int = 3600,
        max_sessions_per_instance: int = 10000,
    ):
        """
        Initialize session affinity manager
        
        Args:
            default_ttl_seconds: Default session TTL
            max_sessions_per_instance: Max sessions per instance
        """
        self.default_ttl = default_ttl_seconds
        self.max_sessions = max_sessions_per_instance
        
        # Session storage
        self._sessions: Dict[str, SessionAffinity] = {}
        
        # Session ID to instance mapping for quick lookup
        self._session_to_instance: Dict[str, str] = {}
        
        # Instance to session count
        self._instance_sessions: Dict[str, int] = defaultdict(int)
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def create_session(
        self,
        session_id: str,
        instance_id: str,
        ttl_seconds: Optional[int] = None,
    ) -> SessionAffinity:
        """
        Create a new session with affinity
        
        Args:
            session_id: Session identifier
            instance_id: Preferred instance
            ttl_seconds: Session TTL
            
        Returns:
            SessionAffinity object
        """
        with self._lock:
            # Check instance load
            if self._instance_sessions[instance_id] >= self.max_sessions:
                # Find least loaded instance
                instance_id = self._get_least_loaded_instance()
            
            session = SessionAffinity(
                session_id=session_id,
                instance_id=instance_id,
                created_at=datetime.utcnow(),
                last_accessed=datetime.utcnow(),
                ttl_seconds=ttl_seconds or self.default_ttl,
            )
            
            self._sessions[session_id] = session
            self._session_to_instance[session_id] = instance_id
            self._instance_sessions[instance_id] += 1
            
            return session
    
    def get_session_instance(self, session_id: str) -> Optional[str]:
        """
        Get the instance for a session
        
        Args:
            session_id: Session identifier
            
        Returns:
            Instance ID or None if session not found/expired
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session is None or session.is_expired():
                if session:
                    self._remove_session(session_id)
                return None
            
            # Update last accessed
            session.last_accessed = datetime.utcnow()
            
            return session.instance_id
    
    def refresh_session(self, session_id: str) -> bool:
        """Refresh a session's TTL"""
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session and not session.is_expired():
                session.last_accessed = datetime.utcnow()
                return True
            
            return False
    
    def remove_session(self, session_id: str) -> bool:
        """Remove a session"""
        with self._lock:
            return self._remove_session(session_id)
    
    def _remove_session(self, session_id: str) -> bool:
        """Internal session removal"""
        session = self._sessions.get(session_id)
        
        if session:
            instance_id = session.instance_id
            self._instance_sessions[instance_id] -= 1
            
            if self._instance_sessions[instance_id] <= 0:
                del self._instance_sessions[instance_id]
            
            del self._sessions[session_id]
            del self._session_to_instance[session_id]
            
            return True
        
        return False
    
    def _get_least_loaded_instance(self) -> str:
        """Get the least loaded instance"""
        if not self._instance_sessions:
            return "default"
        
        return min(
            self._instance_sessions.keys(),
            key=lambda i: self._instance_sessions[i]
        )
    
    def rebalance_sessions(
        self,
        instance_loads: Dict[str, float],
        max_migrations: int = 100,
    ) -> List[Tuple[str, str, str]]:
        """
        Rebalance sessions across instances
        
        Args:
            instance_loads: Current load scores for instances (0-1)
            max_migrations: Maximum number of session migrations
            
        Returns:
            List of (session_id, old_instance, new_instance)
        """
        migrations = []
        
        with self._lock:
            # Sort instances by load
            sorted_instances = sorted(
                instance_loads.items(),
                key=lambda x: x[1]
            )
            
            least_loaded = sorted_instances[0][0] if sorted_instances else None
            most_loaded = sorted_instances[-1][0] if sorted_instances else None
            
            if least_loaded and most_loaded and least_loaded != most_loaded:
                # Find sessions on most loaded instance
                for session_id, session in self._sessions.items():
                    if session.instance_id == most_loaded:
                        if len(migrations) >= max_migrations:
                            break
                        
                        old_instance = session.instance_id
                        session.instance_id = least_loaded
                        
                        self._instance_sessions[old_instance] -= 1
                        self._instance_sessions[least_loaded] += 1
                        
                        migrations.append((session_id, old_instance, least_loaded))
            
            return migrations
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get session manager statistics"""
        with self._lock:
            active_sessions = len(self._sessions)
            
            # Count sessions per instance
            instance_distribution = dict(self._instance_sessions)
            
            return {
                "active_sessions": active_sessions,
                "instance_distribution": instance_distribution,
                "avg_sessions_per_instance": (
                    active_sessions / len(instance_distribution)
                    if instance_distribution else 0
                ),
            }


class HealthCheckManager:
    """
    Manages health checks for scaling instances
    """
    
    def __init__(
        self,
        check_interval_seconds: int = 10,
        failure_threshold: int = 3,
        recovery_threshold: int = 2,
    ):
        """
        Initialize health check manager
        
        Args:
            check_interval_seconds: How often to run checks
            failure_threshold: Consecutive failures before marking unhealthy
            recovery_threshold: Consecutive successes before marking healthy
        """
        self.check_interval = check_interval_seconds
        self.failure_threshold = failure_threshold
        self.recovery_threshold = recovery_threshold
        
        # Instance health tracking
        self._health_status: Dict[str, Dict[str, Any]] = {}
        
        # Custom health check functions
        self._check_functions: Dict[str, Callable] = {}
        
        # Registered instances
        self._instances: Dict[str, InstanceInfo] = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Background thread
        self._running = True
        self._check_thread = threading.Thread(target=self._run_health_checks)
        self._check_thread.daemon = True
        self._check_thread.start()
    
    def register_instance(self, instance: InstanceInfo):
        """Register an instance for health checks"""
        with self._lock:
            self._instances[instance.instance_id] = instance
            
            self._health_status[instance.instance_id] = {
                "consecutive_failures": 0,
                "consecutive_successes": 1,
                "last_check": None,
                "last_status": "unknown",
            }
    
    def unregister_instance(self, instance_id: str):
        """Unregister an instance"""
        with self._lock:
            if instance_id in self._instances:
                del self._instances[instance_id]
            if instance_id in self._health_status:
                del self._health_status[instance_id]
    
    def add_check(self, check_name: str, check_function: Callable):
        """Add a custom health check function"""
        self._check_functions[check_name] = check_function
    
    def check_instance(self, instance_id: str) -> HealthCheckResult:
        """
        Run health checks on an instance
        
        Args:
            instance_id: Instance to check
            
        Returns:
            HealthCheckResult with check outcomes
        """
        start_time = time.time()
        
        with self._lock:
            instance = self._instances.get(instance_id)
            
            if not instance:
                return HealthCheckResult(
                    instance_id=instance_id,
                    is_healthy=False,
                    checks={"error": "Instance not registered"},
                    latency_ms=(time.time() - start_time) * 1000,
                )
        
        checks = {}
        is_healthy = True
        
        # Run registered check functions
        for check_name, check_func in self._check_functions.items():
            try:
                result = check_func(instance)
                checks[check_name] = result
                if not result.get("passed", True):
                    is_healthy = False
            except Exception as e:
                checks[check_name] = {"passed": False, "error": str(e)}
                is_healthy = False
        
        # Basic connectivity check
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((instance.host, instance.port))
            checks["connectivity"] = {"passed": result == 0, "port": instance.port}
            if result != 0:
                is_healthy = False
            sock.close()
        except Exception as e:
            checks["connectivity"] = {"passed": False, "error": str(e)}
            is_healthy = False
        
        # Update health status
        with self._lock:
            status = self._health_status.get(instance_id, {})
            
            if is_healthy:
                status["consecutive_successes"] = status.get("consecutive_successes", 0) + 1
                status["consecutive_failures"] = 0
                
                if status["consecutive_successes"] >= self.recovery_threshold:
                    status["last_status"] = "healthy"
                instance.status = "healthy" if status["last_status"] == "healthy" else "degraded"
            else:
                status["consecutive_failures"] = status.get("consecutive_failures", 0) + 1
                status["consecutive_successes"] = 0
                
                if status["consecutive_failures"] >= self.failure_threshold:
                    status["last_status"] = "unhealthy"
                instance.status = "unhealthy"
            
            status["last_check"] = datetime.utcnow().isoformat()
        
        return HealthCheckResult(
            instance_id=instance_id,
            is_healthy=is_healthy and status.get("last_status") != "unhealthy",
            checks=checks,
            latency_ms=(time.time() - start_time) * 1000,
        )
    
    def get_healthy_instances(self) -> List[InstanceInfo]:
        """Get all healthy instances"""
        with self._lock:
            return [
                inst for inst in self._instances.values()
                if inst.status in ("healthy", "degraded")
            ]
    
    def get_instance_health(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """Get health status for an instance"""
        with self._lock:
            return self._health_status.get(instance_id)
    
    def _run_health_checks(self):
        """Background task for running health checks"""
        while self._running:
            time.sleep(self.check_interval)
            
            with self._lock:
                instance_ids = list(self._instances.keys())
            
            for instance_id in instance_ids:
                try:
                    self.check_instance(instance_id)
                except Exception as e:
                    logger.error(f"Health check failed for {instance_id}: {e}")
    
    def shutdown(self):
        """Shutdown health check manager"""
        self._running = False
        if self._check_thread.is_alive():
            self._check_thread.join(timeout=5)
    
    def get_health_stats(self) -> Dict[str, Any]:
        """Get health check statistics"""
        with self._lock:
            total = len(self._instances)
            healthy = sum(1 for inst in self._instances.values() if inst.status == "healthy")
            degraded = sum(1 for inst in self._instances.values() if inst.status == "degraded")
            unhealthy = sum(1 for inst in self._instances.values() if inst.status == "unhealthy")
            
            return {
                "total_instances": total,
                "healthy": healthy,
                "degraded": degraded,
                "unhealthy": unhealthy,
                "availability": (healthy + degraded) / total if total > 0 else 0,
            }


class LoadBalancer:
    """
    Load balancer for distributing requests across instances
    """
    
    def __init__(
        self,
        health_manager: HealthCheckManager,
        session_manager: SessionAffinityManager,
        algorithm: str = "weighted_round_robin",
    ):
        """
        Initialize load balancer
        
        Args:
            health_manager: Health check manager
            session_manager: Session affinity manager
            algorithm: Load balancing algorithm
        """
        self.health_manager = health_manager
        self.session_manager = session_manager
        self.algorithm = algorithm
        
        # Round robin state
        self._rr_counters: Dict[str, int] = defaultdict(int)
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def select_instance(
        self,
        session_id: Optional[str] = None,
        preferred_instance: Optional[str] = None,
        kernel_domain: Optional[str] = None,
    ) -> Optional[InstanceInfo]:
        """
        Select an instance for request routing
        
        Args:
            session_id: Session for affinity
            preferred_instance: User-specified preference
            kernel_domain: Kernel domain for routing
            
        Returns:
            Selected instance or None
        """
        with self._lock:
            # Check session affinity first
            if session_id:
                instance_id = self.session_manager.get_session_instance(session_id)
                if instance_id:
                    instance = self._get_instance_by_id(instance_id)
                    if instance and instance.status != "unhealthy":
                        return instance
            
            # Check preferred instance
            if preferred_instance:
                instance = self._get_instance_by_id(preferred_instance)
                if instance and instance.status != "unhealthy":
                    return instance
            
            # Get healthy instances
            instances = self.health_manager.get_healthy_instances()
            
            if not instances:
                return None
            
            # Apply load balancing algorithm
            if self.algorithm == "weighted_round_robin":
                return self._weighted_round_robin(instances)
            elif self.algorithm == "least_connections":
                return self._least_connections(instances)
            elif self.algorithm == "weighted_random":
                return self._weighted_random(instances)
            elif self.algorithm == "ip_hash":
                return self._ip_hash(instances, kernel_domain or "")
            else:
                return instances[0]
    
    def _get_instance_by_id(self, instance_id: str) -> Optional[InstanceInfo]:
        """Get instance by ID from health manager"""
        with self._lock:
            return self.health_manager._instances.get(instance_id)
    
    def _weighted_round_robin(self, instances: List[InstanceInfo]) -> InstanceInfo:
        """Weighted round robin selection"""
        # Calculate total weight
        total_weight = sum(inst.weight for inst in instances)
        
        if total_weight == 0:
            return random.choice(instances)
        
        # Generate deterministic index based on counter
        self._rr_counters["global"] += 1
        counter = self._rr_counters["global"]
        
        # Find instance based on weight
        threshold = (counter % total_weight)
        cumulative = 0
        
        for inst in instances:
            cumulative += inst.weight
            if cumulative > threshold:
                return inst
        
        return instances[-1]
    
    def _least_connections(self, instances: List[InstanceInfo]) -> InstanceInfo:
        """Select instance with fewest connections"""
        return min(
            instances,
            key=lambda inst: inst.request_count,
        )
    
    def _weighted_random(self, instances: List[InstanceInfo]) -> InstanceInfo:
        """Random selection weighted by capacity"""
        weights = [inst.weight for inst in instances]
        total = sum(weights)
        
        if total == 0:
            return random.choice(instances)
        
        r = random.uniform(0, total)
        cumulative = 0
        
        for i, inst in enumerate(instances):
            cumulative += weights[i]
            if cumulative >= r:
                return inst
        
        return instances[-1]
    
    def _ip_hash(self, instances: List[InstanceInfo], key: str) -> InstanceInfo:
        """Consistent hashing based on IP or other key"""
        if not instances:
            return None
        
        hash_value = int(hashlib.md5(key.encode()).hexdigest(), 16)
        index = hash_value % len(instances)
        
        return instances[index]
    
    def record_request(
        self,
        instance_id: str,
        response_time_ms: float,
        success: bool,
    ):
        """Record request outcome for load metrics"""
        with self._lock:
            instance = self._get_instance_by_id(instance_id)
            
            if instance:
                instance.request_count += 1
                
                if not success:
                    instance.error_count += 1
                
                # Update moving average of response time
                if instance.avg_response_time_ms == 0:
                    instance.avg_response_time_ms = response_time_ms
                else:
                    instance.avg_response_time_ms = (
                        instance.avg_response_time_ms * 0.9 +
                        response_time_ms * 0.1
                    )
    
    def get_load_stats(self) -> Dict[str, Any]:
        """Get load balancing statistics"""
        with self._lock:
            instances = list(self.health_manager._instances.values())
            
            if not instances:
                return {"total_requests": 0, "error_rate": 0}
            
            total_requests = sum(inst.request_count for inst in instances)
            total_errors = sum(inst.error_count for inst in instances)
            
            return {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": total_errors / total_requests if total_requests > 0 else 0,
                "avg_response_time_ms": sum(
                    inst.avg_response_time_ms for inst in instances
                ) / len(instances),
            }


class HorizontalScalingManager:
    """
    Main horizontal scaling coordinator
    """
    
    def __init__(
        self,
        instance_id: str,
        host: str,
        port: int,
        storage_backend=None,
    ):
        """
        Initialize horizontal scaling manager
        
        Args:
            instance_id: Unique identifier for this instance
            host: Host address of this instance
            port: Port number of this instance
            storage_backend: Storage backend for coordination
        """
        self.instance_id = instance_id
        self.host = host
        self.port = port
        
        # Initialize components
        self.shard_router = ShardRouter()
        self.lock_manager = DistributedLockManager()
        self.session_manager = SessionAffinityManager()
        self.health_manager = HealthCheckManager()
        self.load_balancer = LoadBalancer(
            self.health_manager,
            self.session_manager,
        )
        
        # Register this instance
        self._register_instance()
        
        # Statistics
        self._stats = {
            "instance_id": instance_id,
            "start_time": datetime.utcnow(),
            "requests_handled": 0,
            "locks_acquired": 0,
            "sessions_created": 0,
        }
    
    def _register_instance(self):
        """Register this instance with the scaling system"""
        instance = InstanceInfo(
            instance_id=self.instance_id,
            host=self.host,
            port=self.port,
            capabilities=["kernel_operations", "query_processing", "reasoning"],
        )
        
        self.health_manager.register_instance(instance)
        self.shard_router.register_shard(ShardInfo(
            shard_id=f"shard-{self.instance_id}",
            strategy=ShardingStrategy.HASH_RING,
            shard_key=self.instance_id,
            location=ShardLocation.ACTIVE,
            instance_id=self.instance_id,
            host=self.host,
            port=self.port,
        ))
    
    @contextmanager
    def distributed_lock(
        self,
        resource_type: str,
        resource_id: str,
        ttl_seconds: int = 30,
    ):
        """
        Context manager for distributed locking
        
        Usage:
            with scaling_manager.distributed_lock("kernel", kernel_id):
                # Perform kernel operation
                pass
        """
        lock = self.lock_manager.acquire_lock(
            resource_type=resource_type,
            resource_id=resource_id,
            instance_id=self.instance_id,
            ttl_seconds=ttl_seconds,
            blocking=True,
        )
        
        try:
            if lock:
                self._stats["locks_acquired"] += 1
                yield lock
            else:
                raise RuntimeError(
                    f"Failed to acquire lock for {resource_type}:{resource_id}"
                )
        finally:
            if lock:
                self.lock_manager.release_lock(lock)
    
    def create_session(self, session_id: str) -> SessionAffinity:
        """Create a new session with affinity"""
        instance = self.load_balancer.select_instance()
        
        if instance:
            instance_id = instance.instance_id
        else:
            instance_id = self.instance_id
        
        self._stats["sessions_created"] += 1
        
        return self.session_manager.create_session(
            session_id=session_id,
            instance_id=instance_id,
        )
    
    def route_request(
        self,
        kernel_id: str,
        operation: str,
        session_id: Optional[str] = None,
    ) -> Tuple[Optional[InstanceInfo], Dict[str, Any]]:
        """
        Route a request to the appropriate instance
        
        Args:
            kernel_id: Target kernel ID
            operation: Operation type
            session_id: Session for affinity
            
        Returns:
            (selected_instance, routing_info)
        """
        # Get kernel info for routing
        kernel = None
        if hasattr(self, 'storage') and self.storage:
            try:
                kernel = self.storage.get_kernel(kernel_id)
            except Exception:
                pass
        
        kernel_domain = kernel.get("domain", "general") if kernel else "general"
        
        # Select instance
        instance = self.load_balancer.select_instance(
            session_id=session_id,
            kernel_domain=kernel_domain,
        )
        
        routing_info = {
            "kernel_id": kernel_id,
            "operation": operation,
            "routed_to": instance.instance_id if instance else None,
            "strategy": self.shard_router.strategy.value,
            "timestamp": datetime.utcnow().isoformat(),
        }
        
        return instance, routing_info
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics"""
        return {
            "instance": {
                "instance_id": self.instance_id,
                "host": self.host,
                "port": self.port,
                "uptime_seconds": (
                    datetime.utcnow() - self._stats["start_time"]
                ).total_seconds(),
                "requests_handled": self._stats["requests_handled"],
                "locks_acquired": self._stats["locks_acquired"],
                "sessions_created": self._stats["sessions_created"],
            },
            "shards": self.shard_router.get_shard_distribution(),
            "locks": self.lock_manager.get_lock_stats(),
            "sessions": self.session_manager.get_session_stats(),
            "health": self.health_manager.get_health_stats(),
            "load_balancing": self.load_balancer.get_load_stats(),
        }
    
    def shutdown(self):
        """Shutdown the scaling manager"""
        # Release all locks
        self.lock_manager.release_instance_locks(self.instance_id)
        
        # Shutdown components
        self.lock_manager.shutdown()
        self.health_manager.shutdown()
        
        # Unregister instance
        self.health_manager.unregister_instance(self.instance_id)
        self.shard_router.unregister_shard(f"shard-{self.instance_id}")
        
        logger.info(f"Horizontal scaling manager {self.instance_id} shutdown complete")
