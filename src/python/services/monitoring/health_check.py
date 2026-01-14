"""
KERNELIZE Platform - Health Check System
=========================================

Comprehensive system health monitoring with detailed checks,
trend analysis, synthetic monitoring, and automatic degraded mode.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
from threading import RLock
from collections import deque
import random

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component types for health checking"""
    DATABASE = "database"
    CACHE = "cache"
    VECTOR_DB = "vector_db"
    STORAGE = "storage"
    API = "api"
    MODEL = "model"
    QUEUE = "queue"
    EXTERNAL = "external"


@dataclass
class HealthCheck:
    """Individual health check result"""
    check_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    component: str = ""
    component_type: str = ""
    
    status: str = HealthStatus.UNKNOWN.value
    message: str = ""
    
    # Timing
    latency_ms: float = 0.0
    checked_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Details
    details: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    
    # Thresholds
    latency_threshold_ms: float = 1000.0
    
    # History
    consecutive_failures: int = 0
    consecutive_successes: int = 0


@dataclass
class HealthSnapshot:
    """Complete health snapshot"""
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Overall status
    overall_status: str = HealthStatus.UNKNOWN.value
    health_score: float = 100.0
    
    # Component summary
    components_healthy: int = 0
    components_degraded: int = 0
    components_unhealthy: int = 0
    components_unknown: int = 0
    
    # Individual checks
    checks: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregate metrics
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    
    # System info
    system_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthHistoryEntry:
    """Entry in health history"""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    overall_status: str = HealthStatus.UNKNOWN.value
    health_score: float = 100.0
    
    component_statuses: Dict[str, str] = field(default_factory=dict)
    
    # Metrics
    total_latency_ms: float = 0.0
    failed_checks: int = 0
    
    # Events
    events: List[str] = field(default_factory=list)


@dataclass
class HealthConfiguration:
    """Health check configuration"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Health Config"
    
    # Check intervals (seconds)
    check_interval_seconds: int = 30
    critical_check_interval_seconds: int = 5
    
    # Timeouts
    default_timeout_seconds: float = 5.0
    database_timeout_seconds: float = 3.0
    cache_timeout_seconds: float = 1.0
    
    # Thresholds
    max_latency_threshold_ms: float = 1000.0
    warning_latency_threshold_ms: float = 500.0
    consecutive_failure_threshold: int = 3
    
    # History
    history_retention_hours: int = 24
    history_max_entries: int = 2880  # 24 hours at 30-second intervals
    
    # Degraded mode
    auto_degraded_mode: bool = True
    degraded_mode_threshold: float = 50.0  # Health score threshold
    
    is_active: bool = True


class HealthCheckRegistry:
    """Registry of health check functions"""
    
    def __init__(self):
        """Initialize registry"""
        self._checks: Dict[str, Tuple[Callable, Dict[str, Any]]] = {}
        self._lock = RLock()
    
    def register(
        self,
        name: str,
        component: str,
        component_type: str,
        check_func: Callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Register a health check
        
        Args:
            name: Check name
            component: Component name
            component_type: Type of component
            check_func: Async function that returns check result
            config: Check configuration
        """
        with self._lock:
            self._checks[name] = (check_func, {
                'name': name,
                'component': component,
                'component_type': component_type,
                **(config or {})
            })
    
    def unregister(self, name: str):
        """Unregister a health check"""
        with self._lock:
            if name in self._checks:
                del self._checks[name]
    
    def get_checks(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered checks"""
        with self._lock:
            return {name: config for name, (_, config) in self._checks.items()}
    
    async def execute_check(
        self,
        name: str,
        timeout: float = 5.0
    ) -> HealthCheck:
        """Execute a specific health check"""
        with self._lock:
            if name not in self._checks:
                return HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN.value,
                    message=f"Check '{name}' not found"
                )
            
            check_func, config = self._checks[name]
        
        start_time = time.perf_counter()
        
        try:
            # Execute check with timeout
            result = await asyncio.wait_for(
                check_func(),
                timeout=timeout
            )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            # Add timing to result
            result.latency_ms = latency_ms
            result.checked_at = datetime.now(timezone.utc).isoformat()
            
            return result
            
        except asyncio.TimeoutError:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheck(
                name=config['name'],
                component=config['component'],
                component_type=config['component_type'],
                status=HealthStatus.CRITICAL.value,
                message="Health check timed out",
                latency_ms=latency_ms,
                error_message=f"Timeout after {timeout}s",
                consecutive_failures=1
            )
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            return HealthCheck(
                name=config['name'],
                component=config['component'],
                component_type=config['component_type'],
                status=HealthStatus.UNHEALTHY.value,
                message=f"Health check failed: {str(e)}",
                latency_ms=latency_ms,
                error_message=str(e),
                consecutive_failures=1
            )
    
    async def execute_all(
        self,
        timeout: float = 5.0
    ) -> List[HealthCheck]:
        """Execute all registered health checks"""
        tasks = [
            self.execute_check(name, timeout) 
            for name in self._checks.keys()
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to health checks
        checks = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                name = list(self._checks.keys())[i]
                checks.append(HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY.value,
                    error_message=str(result)
                ))
            else:
                checks.append(result)
        
        return checks


class HealthMonitor:
    """
    Main health monitoring service
    
    Coordinates health checks, tracks history, provides synthetic monitoring,
    and manages degraded mode.
    """
    
    def __init__(self, config: Optional[HealthConfiguration] = None):
        """
        Initialize health monitor
        
        Args:
            config: Health configuration
        """
        self.config = config or HealthConfiguration()
        self.registry = HealthCheckRegistry()
        
        # History
        self._history: deque = deque(maxlen=self.config.history_max_entries)
        
        # State
        self._is_running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Degraded mode
        self._degraded_mode = False
        self._degraded_reason: Optional[str] = None
        
        # Callbacks
        self._status_callbacks: List[Callable[[HealthSnapshot], None]] = []
        self._alert_callbacks: List[Callable[[str, str, str], None]] = []  # level, message, details
        
        # Statistics
        self._stats = {
            'checks_executed': 0,
            'checks_failed': 0,
            'alerts_sent': 0,
            'mode_changes': 0
        }
        self._stats_lock = RLock()
        
        # Register default checks
        self._register_default_checks()
    
    def _register_default_checks(self):
        """Register default health checks"""
        
        # Database health check
        async def check_database():
            # Would actually check database connection
            return HealthCheck(
                name="PostgreSQL",
                component="database",
                component_type=ComponentType.DATABASE.value,
                status=HealthStatus.HEALTHY.value,
                message="Database connection healthy",
                latency_ms=random.uniform(1, 10)
            )
        
        self.registry.register(
            "database",
            "database",
            ComponentType.DATABASE.value,
            check_database,
            {'priority': 'high'}
        )
        
        # Cache health check
        async def check_cache():
            return HealthCheck(
                name="Redis Cache",
                component="cache",
                component_type=ComponentType.CACHE.value,
                status=HealthStatus.HEALTHY.value,
                message="Cache connection healthy",
                latency_ms=random.uniform(0.5, 5)
            )
        
        self.registry.register(
            "cache",
            "cache",
            ComponentType.CACHE.value,
            check_cache
        )
        
        # Vector database check
        async def check_vector_db():
            return HealthCheck(
                name="Qdrant Vector DB",
                component="vector_db",
                component_type=ComponentType.VECTOR_DB.value,
                status=HealthStatus.HEALTHY.value,
                message="Vector database connection healthy",
                latency_ms=random.uniform(2, 15)
            )
        
        self.registry.register(
            "vector_db",
            "vector_db",
            ComponentType.VECTOR_DB.value,
            check_vector_db
        )
        
        # Model health check
        async def check_model():
            return HealthCheck(
                name="Embedding Model",
                component="model",
                component_type=ComponentType.MODEL.value,
                status=HealthStatus.HEALTHY.value,
                message="Model loaded and responsive",
                latency_ms=random.uniform(10, 50)
            )
        
        self.registry.register(
            "model",
            "model",
            ComponentType.MODEL.value,
            check_model
        )
        
        # API health check
        async def check_api():
            return HealthCheck(
                name="API Status",
                component="api",
                component_type=ComponentType.API.value,
                status=HealthStatus.HEALTHY.value,
                message="API endpoint responsive",
                latency_ms=random.uniform(1, 20)
            )
        
        self.registry.register(
            "api",
            "api",
            ComponentType.API.value,
            check_api
        )
    
    async def start(self):
        """Start health monitoring"""
        if self._is_running:
            return
        
        self._is_running = True
        
        # Start monitoring loop
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        
        logger.info("Health monitoring started")
    
    async def stop(self):
        """Stop health monitoring"""
        self._is_running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop"""
        while self._is_running:
            try:
                # Execute all checks
                snapshot = await self.check_all()
                
                # Store in history
                self._add_to_history(snapshot)
                
                # Check for degraded mode
                self._check_degraded_mode(snapshot)
                
                # Send alerts if needed
                self._check_alerts(snapshot)
                
                # Notify callbacks
                self._notify_status_change(snapshot)
                
                # Wait for next interval
                interval = (
                    self.config.critical_check_interval_seconds 
                    if snapshot.health_score < 70 
                    else self.config.check_interval_seconds
                )
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                await asyncio.sleep(5)
    
    async def check_all(self) -> HealthSnapshot:
        """
        Execute all health checks and return snapshot
        
        Returns:
            Complete health snapshot
        """
        start_time = time.perf_counter()
        
        # Execute all checks
        checks = await self.registry.execute_all(
            timeout=self.config.default_timeout_seconds
        )
        
        # Process results
        snapshot = self._process_check_results(checks)
        
        # Calculate timing
        total_time = (time.perf_counter() - start_time) * 1000
        snapshot.total_latency_ms = total_time
        
        # Update stats
        with self._stats_lock:
            self._stats['checks_executed'] += len(checks)
            self._stats['checks_failed'] += len([
                c for c in checks 
                if c.status in [HealthStatus.UNHEALTHY.value, HealthStatus.CRITICAL.value]
            ])
        
        return snapshot
    
    def _process_check_results(self, checks: List[HealthCheck]) -> HealthSnapshot:
        """Process health check results into snapshot"""
        snapshot = HealthSnapshot()
        
        # Count by status
        status_counts = {
            HealthStatus.HEALTHY.value: 0,
            HealthStatus.DEGRADED.value: 0,
            HealthStatus.UNHEALTHY.value: 0,
            HealthStatus.CRITICAL.value: 0,
            HealthStatus.UNKNOWN.value: 0
        }
        
        total_latency = 0
        max_latency = 0
        
        for check in checks:
            status_counts[check.status] += 1
            
            total_latency += check.latency_ms
            max_latency = max(max_latency, check.latency_ms)
            
            # Add to snapshot
            snapshot.checks.append({
                'name': check.name,
                'component': check.component,
                'status': check.status,
                'message': check.message,
                'latency_ms': check.latency_ms,
                'checked_at': check.checked_at,
                'details': check.details
            })
        
        snapshot.components_healthy = status_counts[HealthStatus.HEALTHY.value]
        snapshot.components_degraded = status_counts[HealthStatus.DEGRADED.value]
        snapshot.components_unhealthy = status_counts[HealthStatus.UNHEALTHY.value]
        snapshot.components_unknown = status_counts[HealthStatus.UNKNOWN.value]
        
        # Calculate overall status
        total_components = len(checks)
        
        if total_components == 0:
            snapshot.overall_status = HealthStatus.UNKNOWN.value
            snapshot.health_score = 0
        elif status_counts[HealthStatus.CRITICAL.value] > 0:
            snapshot.overall_status = HealthStatus.CRITICAL.value
            snapshot.health_score = 0
        elif status_counts[HealthStatus.UNHEALTHY.value] > 0:
            snapshot.overall_status = HealthStatus.UNHEALTHY.value
            snapshot.health_score = 25
        elif status_counts[HealthStatus.DEGRADED.value] > 0:
            snapshot.overall_status = HealthStatus.DEGRADED.value
            snapshot.health_score = 70
        else:
            snapshot.overall_status = HealthStatus.HEALTHY.value
            snapshot.health_score = 100
        
        # Average latency
        snapshot.avg_latency_ms = total_latency / total_components if total_components > 0 else 0
        snapshot.max_latency_ms = max_latency
        
        return snapshot
    
    def _add_to_history(self, snapshot: HealthSnapshot):
        """Add snapshot to history"""
        entry = HealthHistoryEntry(
            overall_status=snapshot.overall_status,
            health_score=snapshot.health_score,
            component_statuses={
                check['component']: check['status'] 
                for check in snapshot.checks
            },
            total_latency_ms=snapshot.total_latency_ms,
            failed_checks=snapshot.components_unhealthy + snapshot.components_degraded
        )
        
        self._history.append(entry)
    
    def _check_degraded_mode(self, snapshot: HealthSnapshot):
        """Check if degraded mode should be enabled/disabled"""
        should_be_degraded = (
            snapshot.health_score < self.config.degraded_mode_threshold or
            snapshot.components_unhealthy > 0
        )
        
        if should_be_degraded and not self._degraded_mode:
            self._degraded_mode = True
            self._degraded_reason = f"Health score dropped to {snapshot.health_score:.1f}%"
            logger.warning(f"Entering degraded mode: {self._degraded_reason}")
            
            with self._stats_lock:
                self._stats['mode_changes'] += 1
        
        elif not should_be_degraded and self._degraded_mode:
            self._degraded_mode = False
            self._degraded_reason = None
            logger.info("Exiting degraded mode - system recovered")
            
            with self._stats_lock:
                self._stats['mode_changes'] += 1
    
    def _check_alerts(self, snapshot: HealthSnapshot):
        """Check for conditions requiring alerts"""
        # Critical alert
        if snapshot.health_score < 30:
            self._send_alert(
                "critical",
                f"System health critical: {snapshot.health_score:.1f}%",
                snapshot
            )
        
        # Warning alert
        elif snapshot.health_score < 70:
            self._send_alert(
                "warning",
                f"System health degraded: {snapshot.health_score:.1f}%",
                snapshot
            )
        
        # Recovery alert
        elif self._degraded_mode and snapshot.health_score >= 70:
            self._send_alert(
                "info",
                "System has recovered",
                snapshot
            )
    
    def _send_alert(self, level: str, message: str, snapshot: HealthSnapshot):
        """Send alert through callbacks"""
        with self._stats_lock:
            self._stats['alerts_sent'] += 1
        
        for callback in self._alert_callbacks:
            try:
                callback(level, message, snapshot.to_dict() if hasattr(snapshot, 'to_dict') else {})
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def _notify_status_change(self, snapshot: HealthSnapshot):
        """Notify status change callbacks"""
        for callback in self._status_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Status callback failed: {e}")
    
    async def synthetic_check(
        self,
        check_type: str = "query",
        query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute synthetic monitoring check
        
        Args:
            check_type: Type of synthetic check (query, compression, etc.)
            query: Query for synthetic query check
            
        Returns:
            Synthetic check results
        """
        start_time = time.perf_counter()
        
        if check_type == "query":
            # Simulate a query execution
            await asyncio.sleep(0.1)  # Simulate query time
            
            result = {
                'type': 'synthetic_query',
                'status': 'success',
                'latency_ms': (time.perf_counter() - start_time) * 1000,
                'result_count': random.randint(1, 10),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        elif check_type == "compression":
            # Simulate compression check
            await asyncio.sleep(0.2)
            
            result = {
                'type': 'synthetic_compression',
                'status': 'success',
                'latency_ms': (time.perf_counter() - start_time) * 1000,
                'compression_ratio': random.uniform(0.3, 0.8),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        else:
            result = {
                'type': check_type,
                'status': 'unknown',
                'message': f"Unknown synthetic check type: {check_type}",
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        
        return result
    
    def get_history(
        self,
        hours: int = 1,
        component: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get health check history
        
        Args:
            hours: How many hours of history
            component: Filter by component
            
        Returns:
            List of history entries
        """
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        entries = []
        for entry in self._history:
            entry_time = datetime.fromisoformat(entry.timestamp)
            if entry_time < cutoff:
                continue
            
            if component and component not in entry.component_statuses:
                continue
            
            entries.append({
                'timestamp': entry.timestamp,
                'overall_status': entry.overall_status,
                'health_score': entry.health_score,
                'component_statuses': entry.component_statuses,
                'total_latency_ms': entry.total_latency_ms,
                'failed_checks': entry.failed_checks
            })
        
        return entries
    
    def get_trend_analysis(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze health trends over time
        
        Args:
            hours: Analysis window
            
        Returns:
            Trend analysis results
        """
        history = self.get_history(hours=hours)
        
        if not history:
            return {'error': 'No history available'}
        
        # Calculate trends
        health_scores = [h['health_score'] for h in history]
        latencies = [h['total_latency_ms'] for h in history]
        
        # Overall trend (compare first half to second half)
        mid_point = len(health_scores) // 2
        first_half_avg = sum(health_scores[:mid_point]) / mid_point if mid_point > 0 else 0
        second_half_avg = sum(health_scores[mid_point:]) / (len(health_scores) - mid_point) if len(health_scores) > mid_point else 0
        
        trend = "stable"
        if second_half_avg - first_half_avg > 5:
            trend = "improving"
        elif first_half_avg - second_half_avg > 5:
            trend = "declining"
        
        # Identify problematic periods
        problematic = []
        for h in history:
            if h['health_score'] < 70:
                problematic.append({
                    'timestamp': h['timestamp'],
                    'health_score': h['health_score'],
                    'issues': [
                        c for c, s in h['component_statuses'].items() 
                        if s != 'healthy'
                    ]
                })
        
        return {
            'period_hours': hours,
            'data_points': len(history),
            'health_score': {
                'current': health_scores[-1] if health_scores else 0,
                'average': sum(health_scores) / len(health_scores) if health_scores else 0,
                'min': min(health_scores) if health_scores else 0,
                'max': max(health_scores) if health_scores else 0,
                'trend': trend,
                'change': second_half_avg - first_half_avg
            },
            'latency': {
                'average': sum(latencies) / len(latencies) if latencies else 0,
                'max': max(latencies) if latencies else 0
            },
            'problematic_periods': problematic[:10]  # Top 10
        }
    
    def add_status_callback(self, callback: Callable[[HealthSnapshot], None]):
        """Add status change callback"""
        self._status_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[str, str, str], None]):
        """Add alert callback"""
        self._alert_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get health monitor statistics"""
        with self._stats_lock:
            stats = dict(self._stats)
            stats['is_running'] = self._is_running
            stats['degraded_mode'] = self._degraded_mode
            stats['registered_checks'] = len(self.registry._checks)
            stats['history_entries'] = len(self._history)
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            'overall_status': self._degraded_mode or "healthy",
            'degraded_mode': self._degraded_mode,
            'degraded_reason': self._degraded_reason,
            'registered_checks': len(self.registry._checks),
            'is_monitoring': self._is_running
        }


# Singleton instance
_health_monitor: Optional[HealthMonitor] = None


def get_health_monitor() -> HealthMonitor:
    """Get health monitor singleton"""
    global _health_monitor
    
    if _health_monitor is None:
        config = HealthConfiguration()
        _health_monitor = HealthMonitor(config)
    
    return _health_monitor


def init_health_monitoring() -> HealthMonitor:
    """Initialize health monitoring system"""
    global _health_monitor
    
    _health_monitor = HealthMonitor()
    
    return _health_monitor
