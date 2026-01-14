"""
KERNELIZE Platform - Operational Dashboards
============================================

Comprehensive monitoring views for operations.
Provides real-time metrics, capacity planning, business metrics,
and incident management.

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
from typing import Any, Callable, Dict, List, Optional, Tuple
from threading import RLock
from collections import deque
import threading

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert statuses"""
    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Metric:
    """Individual metric value"""
    name: str = ""
    value: float = 0.0
    metric_type: str = MetricType.GAUGE.value
    
    # Labels
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Context
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'value': self.value,
            'type': self.metric_type,
            'labels': self.labels,
            'timestamp': self.timestamp,
            'source': self.source
        }


@dataclass
class TimeSeriesPoint:
    """Single point in a time series"""
    timestamp: str = ""
    value: float = 0.0


@dataclass
class TimeSeries:
    """Time series data"""
    name: str = ""
    labels: Dict[str, str] = field(default_factory=dict)
    points: List[TimeSeriesPoint] = field(default_factory=list)
    
    # Statistics
    min_value: float = 0.0
    max_value: float = 0.0
    avg_value: float = 0.0
    latest_value: float = 0.0
    
    def add_point(self, value: float, timestamp: Optional[str] = None):
        """Add data point"""
        point = TimeSeriesPoint(
            timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
            value=value
        )
        self.points.append(point)
        
        # Update statistics
        if not self.points:
            self.min_value = value
            self.max_value = value
            self.avg_value = value
        else:
            self.min_value = min(self.min_value, value)
            self.max_value = max(self.max_value, value)
            self.avg_value = sum(p.value for p in self.points) / len(self.points)
        
        self.latest_value = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'name': self.name,
            'labels': self.labels,
            'points': [
                {'timestamp': p.timestamp, 'value': p.value}
                for p in self.points[-1000:]  # Limit points
            ],
            'statistics': {
                'min': self.min_value,
                'max': self.max_value,
                'avg': self.avg_value,
                'latest': self.latest_value
            }
        }


@dataclass
class AlertRule:
    """Alert rule configuration"""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Metric
    metric_name: str = ""
    metric_labels: Dict[str, str] = field(default_factory=dict)
    
    # Condition
    condition: str = ">"  # >, <, >=, <=, ==, !=
    threshold: float = 0.0
    duration_seconds: int = 60  # How long condition must persist
    
    # Evaluation
    evaluation_interval_seconds: int = 10
    lookback_period_seconds: int = 300
    
    # Actions
    severity: str = AlertSeverity.MEDIUM.value
    channels: List[str] = field(default_factory=list)  # email, slack, webhook
    
    # Status
    is_enabled: bool = True
    last_evaluated: Optional[str] = None
    last_triggered: Optional[str] = None
    
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Alert:
    """Active or historical alert"""
    alert_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    rule_id: str = ""
    rule_name: str = ""
    
    status: str = AlertStatus.FIRING.value
    severity: str = AlertSeverity.MEDIUM.value
    
    # Details
    metric_name: str = ""
    current_value: float = 0.0
    threshold: float = 0.0
    condition: str = ""
    
    # Labels
    labels: Dict[str, str] = field(default_factory=dict)
    
    # Timing
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ended_at: Optional[str] = None
    acknowledged_at: Optional[str] = None
    acknowledged_by: Optional[str] = None
    
    # Resolution
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None
    resolution: Optional[str] = None
    
    # Notes
    notes: List[Dict[str, str]] = field(default_factory=list)
    
    # Related
    related_alerts: List[str] = field(default_factory=list)
    
    def acknowledge(self, user_id: str):
        """Acknowledge the alert"""
        self.status = AlertStatus.ACKNOWLEDGED.value
        self.acknowledged_at = datetime.now(timezone.utc).isoformat()
        self.acknowledged_by = user_id
    
    def resolve(self, user_id: str, resolution: str):
        """Resolve the alert"""
        self.status = AlertStatus.RESOLVED.value
        self.resolved_at = datetime.now(timezone.utc).isoformat()
        self.resolved_by = user_id
        self.resolution = resolution
        self.ended_at = self.resolved_at
    
    def add_note(self, user_id: str, note: str):
        """Add note to alert"""
        self.notes.append({
            'user_id': user_id,
            'note': note,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })


@dataclass
class Incident:
    """Major incident tracking"""
    incident_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    
    severity: str = AlertSeverity.HIGH.value
    status: str = "open"  # open, investigating, identified, monitoring, closed
    
    # Timeline
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    detected_at: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    
    # Impact
    affected_services: List[str] = field(default_factory=list)
    affected_users: int = 0
    estimated_impact: str = ""
    
    # Assignment
    lead: Optional[str] = None
    team: List[str] = field(default_factory=list)
    
    # Communication
    updates: List[Dict[str, str]] = field(default_factory=list)
    
    # Root cause
    root_cause: Optional[str] = None
    remediation: Optional[str] = None
    prevention: Optional[str] = None
    
    def add_update(self, message: str, author: str, status: Optional[str] = None):
        """Add status update"""
        update = {
            'message': message,
            'author': author,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        if status:
            update['status'] = status
        self.updates.append(update)
        
        # Update incident status
        if status and status != self.status:
            old_status = self.status
            self.status = status
            
            # Track timing
            if status == "investigating" and not self.acknowledged_at:
                self.acknowledged_at = datetime.now(timezone.utc).isoformat()
            elif status == "closed" and not self.resolved_at:
                self.resolved_at = datetime.now(timezone.utc).isoformat()


class MetricsCollector:
    """Collects and stores metrics"""
    
    def __init__(self, retention_hours: int = 24):
        """
        Initialize metrics collector
        
        Args:
            retention_hours: How long to retain metrics
        """
        self.retention_hours = retention_hours
        
        # Time series storage
        self._series: Dict[str, TimeSeries] = {}
        self._lock = RLock()
        
        # Current values (for gauges)
        self._current_values: Dict[str, float] = {}
        
        # Metric callbacks
        self._collection_callbacks: List[Callable[[], List[Metric]]] = []
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: str = MetricType.GAUGE.value,
        labels: Optional[Dict[str, str]] = None,
        source: str = ""
    ):
        """
        Record a metric value
        
        Args:
            name: Metric name
            value: Metric value
            metric_type: Type of metric
            labels: Metric labels
            source: Source of metric
        """
        labels = labels or {}
        
        # Create series key
        series_key = self._make_series_key(name, labels)
        
        with self._lock:
            # Get or create series
            if series_key not in self._series:
                self._series[series_key] = TimeSeries(
                    name=name,
                    labels=labels
                )
            
            # Add point
            self._series[series_key].add_point(value)
            
            # Update current value
            if metric_type == MetricType.GAUGE.value:
                self._current_values[series_key] = value
        
        # Run collection callbacks
        for callback in self._collection_callbacks:
            try:
                callback()
            except:
                pass
    
    def _make_series_key(self, name: str, labels: Dict[str, str]) -> str:
        """Create unique key for time series"""
        label_str = ','.join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]" if label_str else name
    
    def get_series(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        duration: int = 3600  # 1 hour
    ) -> TimeSeries:
        """
        Get time series for a metric
        
        Args:
            name: Metric name
            labels: Filter by labels
            duration: How much history
            
        Returns:
            Time series data
        """
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=duration)
        
        with self._lock:
            # Find matching series
            if labels:
                target_key = self._make_series_key(name, labels)
                if target_key in self._series:
                    series = self._series[target_key]
                    # Filter old points
                    series.points = [
                        p for p in series.points
                        if datetime.fromisoformat(p.timestamp) >= cutoff
                    ]
                    return series
            
            # Aggregate all matching series
            aggregated = TimeSeries(name=name, labels=labels or {})
            
            for series_key, series in self._series.items():
                if series_key.startswith(name + '[') or series_key == name:
                    for point in series.points:
                        if datetime.fromisoformat(point.timestamp) >= cutoff:
                            aggregated.add_point(point.value, point.timestamp)
            
            return aggregated
    
    def get_current_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """Get current value of a metric"""
        series_key = self._make_series_key(name, labels or {})
        
        with self._lock:
            return self._current_values.get(series_key)
    
    def add_collection_callback(self, callback: Callable[[], List[Metric]]):
        """Add callback for automatic metric collection"""
        self._collection_callbacks.append(callback)
    
    def record_increment(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        amount: float = 1,
        source: str = ""
    ):
        """Increment a counter metric"""
        current = self.get_current_value(name, labels) or 0
        self.record_metric(name, current + amount, MetricType.COUNTER.value, labels, source)
    
    def record_decrement(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None,
        amount: float = 1,
        source: str = ""
    ):
        """Decrement a counter metric"""
        current = self.get_current_value(name, labels) or 0
        self.record_metric(name, max(0, current - amount), MetricType.COUNTER.value, labels, source)


class AlertManager:
    """Manages alerts and alert rules"""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """
        Initialize alert manager
        
        Args:
            metrics_collector: Metrics collector for evaluation
        """
        self._metrics = metrics_collector
        
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        
        self._lock = RLock()
        
        # Default rules
        self._init_default_rules()
        
        # Alert callbacks
        self._alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Start evaluation loop
        self._running = False
        self._evaluation_task = None
    
    def _init_default_rules(self):
        """Initialize default alert rules"""
        # High error rate
        self.add_rule(AlertRule(
            name="High Error Rate",
            description="Error rate exceeds 5%",
            metric_name="error_rate",
            condition=">",
            threshold=5.0,
            duration_seconds=60,
            severity=AlertSeverity.HIGH.value,
            channels=["slack", "email"]
        ))
        
        # High latency
        self.add_rule(AlertRule(
            name="High Latency",
            description="P99 latency exceeds 1 second",
            metric_name="query_latency_p99",
            condition=">",
            threshold=1000.0,
            duration_seconds=120,
            severity=AlertSeverity.MEDIUM.value,
            channels=["slack"]
        ))
        
        # Low cache hit rate
        self.add_rule(AlertRule(
            name="Low Cache Hit Rate",
            description="Cache hit rate below 80%",
            metric_name="cache_hit_rate",
            condition="<",
            threshold=80.0,
            duration_seconds=300,
            severity=AlertSeverity.LOW.value,
            channels=["slack"]
        ))
        
        # System health
        self.add_rule(AlertRule(
            name="System Unhealthy",
            description="System health score below 50",
            metric_name="health_score",
            condition="<",
            threshold=50.0,
            duration_seconds=60,
            severity=AlertSeverity.CRITICAL.value,
            channels=["slack", "email", "pagerduty"]
        ))
    
    def add_rule(self, rule: AlertRule):
        """Add alert rule"""
        with self._lock:
            self._rules[rule.rule_id] = rule
    
    def remove_rule(self, rule_id: str):
        """Remove alert rule"""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
    
    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules"""
        with self._lock:
            return list(self._rules.values())
    
    async def evaluate_rules(self):
        """Evaluate all alert rules"""
        with self._lock:
            rules = list(self._rules.values())
        
        for rule in rules:
            if not rule.is_enabled:
                continue
            
            try:
                # Get metric value
                value = self._metrics.get_current_value(
                    rule.metric_name,
                    rule.metric_labels
                )
                
                if value is None:
                    continue
                
                # Check condition
                triggered = self._check_condition(value, rule.condition, rule.threshold)
                
                rule.last_evaluated = datetime.now(timezone.utc).isoformat()
                
                if triggered:
                    await self._trigger_alert(rule, value)
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {e}")
    
    def _check_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Check if condition is met"""
        if condition == ">":
            return value > threshold
        elif condition == ">=":
            return value >= threshold
        elif condition == "<":
            return value < threshold
        elif condition == "<=":
            return value <= threshold
        elif condition == "==":
            return value == threshold
        elif condition == "!=":
            return value != threshold
        return False
    
    async def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert"""
        # Check if alert already exists
        with self._lock:
            for alert in self._alerts.values():
                if alert.rule_id == rule.rule_id:
                    if alert.status == AlertStatus.FIRING.value:
                        # Alert already firing
                        return
                    elif alert.status == AlertStatus.RESOLVED.value:
                        # Create new alert
                        break
                    else:
                        # Acknowledged alert
                        return
        
        # Create new alert
        alert = Alert(
            rule_id=rule.rule_id,
            rule_name=rule.name,
            severity=rule.severity,
            metric_name=rule.metric_name,
            current_value=current_value,
            threshold=rule.threshold,
            condition=rule.condition,
            labels=rule.metric_labels
        )
        
        with self._lock:
            self._alerts[alert.alert_id] = alert
        
        rule.last_triggered = datetime.now(timezone.utc).isoformat()
        
        # Notify callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def acknowledge_alert(self, alert_id: str, user_id: str) -> bool:
        """Acknowledge an alert"""
        with self._lock:
            if alert_id in self._alerts:
                self._alerts[alert_id].acknowledge(user_id)
                return True
        return False
    
    def resolve_alert(
        self,
        alert_id: str,
        user_id: str,
        resolution: str
    ) -> bool:
        """Resolve an alert"""
        with self._lock:
            if alert_id in self._alerts:
                self._alerts[alert_id].resolve(user_id, resolution)
                return True
        return False
    
    def get_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts"""
        with self._lock:
            alerts = list(self._alerts.values())
        
        # Filter
        if status:
            alerts = [a for a in alerts if a.status == status]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        # Sort by start time (newest first)
        alerts.sort(key=lambda a: a.started_at, reverse=True)
        
        return alerts[:limit]
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add callback for new alerts"""
        self._alert_callbacks.append(callback)
    
    def start(self):
        """Start alert evaluation"""
        if self._running:
            return
        
        self._running = True
        self._evaluation_task = asyncio.create_task(self._evaluation_loop())
    
    async def stop(self):
        """Stop alert evaluation"""
        self._running = False
        
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
    
    async def _evaluation_loop(self):
        """Background evaluation loop"""
        while self._running:
            try:
                await self.evaluate_rules()
            except Exception as e:
                logger.error(f"Error in alert evaluation loop: {e}")
            
            await asyncio.sleep(10)  # Evaluate every 10 seconds


class OperationalDashboard:
    """
    Main operational dashboard service
    
    Provides comprehensive monitoring views with real-time metrics,
    capacity planning, business metrics, and incident management.
    """
    
    def __init__(self):
        """Initialize operational dashboard"""
        self._metrics = MetricsCollector()
        self._alerts = AlertManager(self._metrics)
        
        # Business metrics
        self._kernels_created: int = 0
        self._queries_executed: int = 0
        self._compression_jobs: int = 0
        self._active_users: int = 0
        
        # Resource metrics
        self._cpu_usage: float = 0.0
        self._memory_usage: float = 0.0
        self._disk_usage: float = 0.0
        self._network_io: Dict[str, float] = field(default_factory=dict)
        
        # History
        self._metrics_history: deque = deque(maxlen=3600)  # 1 hour of 1-second samples
        
        # Lock
        self._lock = RLock()
        
        # Start background collection
        self._running = False
        self._collection_task = None
        
        # Initialize default metrics
        self._init_default_metrics()
    
    def _init_default_metrics(self):
        """Initialize default metrics"""
        # Query metrics
        self._metrics.record_metric("queries_total", 0, MetricType.COUNTER.value)
        self._metrics.record_metric("queries_active", 0, MetricType.GAUGE.value)
        self._metrics.record_metric("query_latency_avg", 0, MetricType.GAUGE.value)
        self._metrics.record_metric("query_latency_p99", 0, MetricType.GAUGE.value)
        self._metrics.record_metric("error_rate", 0, MetricType.GAUGE.value)
        
        # Cache metrics
        self._metrics.record_metric("cache_hits_total", 0, MetricType.COUNTER.value)
        self._metrics.record_metric("cache_misses_total", 0, MetricType.COUNTER.value)
        self._metrics.record_metric("cache_hit_rate", 0, MetricType.GAUGE.value)
        
        # Kernel metrics
        self._metrics.record_metric("kernels_total", 0, MetricType.COUNTER.value)
        self._metrics.record_metric("kernels_active", 0, MetricType.GAUGE.value)
        
        # System metrics
        self._metrics.record_metric("health_score", 100, MetricType.GAUGE.value)
        self._metrics.record_metric("cpu_usage", 0, MetricType.GAUGE.value)
        self._metrics.record_metric("memory_usage", 0, MetricType.GAUGE.value)
        self._metrics.record_metric("disk_usage", 0, MetricType.GAUGE.value)
    
    def start(self):
        """Start dashboard collection"""
        if self._running:
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        self._alerts.start()
        
        logger.info("Operational dashboard started")
    
    async def stop(self):
        """Stop dashboard collection"""
        self._running = False
        
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        await self._alerts.stop()
        
        logger.info("Operational dashboard stopped")
    
    async def _collection_loop(self):
        """Background metrics collection"""
        while self._running:
            try:
                # Collect system metrics (would use psutil in production)
                self._collect_system_metrics()
                
                # Collect derived metrics
                self._collect_derived_metrics()
                
                # Store in history
                self._store_history()
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in dashboard collection: {e}")
    
    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        # Would use psutil for actual metrics
        import random
        
        self._cpu_usage = random.uniform(20, 60)
        self._memory_usage = random.uniform(40, 70)
        self._disk_usage = random.uniform(30, 50)
        
        self._metrics.record_metric("cpu_usage", self._cpu_usage)
        self._metrics.record_metric("memory_usage", self._memory_usage)
        self._metrics.record_metric("disk_usage", self._disk_usage)
    
    def _collect_derived_metrics(self):
        """Calculate derived metrics"""
        # Calculate cache hit rate
        hits = self._metrics.get_current_value("cache_hits_total") or 0
        misses = self._metrics.get_current_value("cache_misses_total") or 0
        total = hits + misses
        
        if total > 0:
            hit_rate = (hits / total) * 100
            self._metrics.record_metric("cache_hit_rate", hit_rate)
        
        # Calculate error rate
        errors = self._metrics.get_current_value("errors_total") or 0
        queries = self._metrics.get_current_value("queries_total") or 0
        
        if queries > 0:
            error_rate = (errors / queries) * 100
            self._metrics.record_metric("error_rate", error_rate)
        
        # Health score
        health = self._calculate_health_score()
        self._metrics.record_metric("health_score", health)
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score"""
        score = 100.0
        
        # Subtract for high resource usage
        if self._cpu_usage > 80:
            score -= 20
        elif self._cpu_usage > 60:
            score -= 10
        
        if self._memory_usage > 85:
            score -= 25
        elif self._memory_usage > 70:
            score -= 15
        
        if self._disk_usage > 90:
            score -= 30
        elif self._disk_usage > 80:
            score -= 15
        
        # Check for active alerts
        firing_alerts = self._alerts.get_alerts(status=AlertStatus.FIRING.value)
        for alert in firing_alerts:
            if alert.severity == AlertSeverity.CRITICAL.value:
                score -= 25
            elif alert.severity == AlertSeverity.HIGH.value:
                score -= 15
            elif alert.severity == AlertSeverity.MEDIUM.value:
                score -= 10
            else:
                score -= 5
        
        return max(0, score)
    
    def _store_history(self):
        """Store current metrics in history"""
        with self._lock:
            self._metrics_history.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'queries_per_second': self._metrics.get_current_value("queries_per_second") or 0,
                'avg_latency': self._metrics.get_current_value("query_latency_avg") or 0,
                'error_rate': self._metrics.get_current_value("error_rate") or 0,
                'health_score': self._metrics.get_current_value("health_score") or 0,
                'cpu': self._cpu_usage,
                'memory': self._memory_usage
            })
    
    # Public API methods
    
    def record_query(self, latency_ms: float, success: bool = True):
        """Record query execution"""
        self._queries_executed += 1
        
        self._metrics.record_increment("queries_total")
        
        # Record latency
        self._metrics.record_metric("query_latency_avg", latency_ms)
        
        # P99 latency (simulated)
        p99 = latency_ms * 1.5 if success else latency_ms * 2
        self._metrics.record_metric("query_latency_p99", p99)
        
        if not success:
            self._metrics.record_increment("errors_total")
    
    def record_kernel_created(self):
        """Record new kernel creation"""
        self._kernels_created += 1
        self._metrics.record_increment("kernels_total")
    
    def record_compression_job(self):
        """Record compression job"""
        self._compression_jobs += 1
    
    def set_active_users(self, count: int):
        """Set number of active users"""
        self._active_users = count
        self._metrics.record_metric("kernels_active", count)
    
    def get_real_time_metrics(self, duration: int = 60) -> Dict[str, Any]:
        """
        Get real-time metrics for dashboard
        
        Args:
            duration: How many seconds of history
            
        Returns:
            Real-time metrics data
        """
        # Get time series
        queries = self._metrics.get_series("queries_per_second", duration=duration)
        latency = self._metrics.get_series("query_latency_avg", duration=duration)
        errors = self._metrics.get_series("error_rate", duration=duration)
        health = self._metrics.get_series("health_score", duration=duration)
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'queries': queries.to_dict(),
            'latency': latency.to_dict(),
            'error_rate': errors.to_dict(),
            'health_score': health.to_dict(),
            'current': {
                'queries_per_second': self._metrics.get_current_value("queries_per_second") or 0,
                'avg_latency_ms': self._metrics.get_current_value("query_latency_avg") or 0,
                'p99_latency_ms': self._metrics.get_current_value("query_latency_p99") or 0,
                'error_rate': self._metrics.get_current_value("error_rate") or 0,
                'health_score': self._metrics.get_current_value("health_score") or 0,
                'active_queries': self._metrics.get_current_value("queries_active") or 0,
                'cache_hit_rate': self._metrics.get_current_value("cache_hit_rate") or 0,
                'cpu_usage': self._cpu_usage,
                'memory_usage': self._memory_usage
            }
        }
    
    def get_capacity_metrics(self, duration: int = 86400) -> Dict[str, Any]:
        """
        Get capacity planning metrics
        
        Args:
            duration: How many seconds of history
            
        Returns:
            Capacity metrics data
        """
        # Resource utilization over time
        cpu = self._metrics.get_series("cpu_usage", duration=duration)
        memory = self._metrics.get_series("memory_usage", duration=duration)
        disk = self._metrics.get_series("disk_usage", duration=duration)
        
        # Trends
        cpu_trend = self._calculate_trend(cpu)
        memory_trend = self._calculate_trend(memory)
        
        # Capacity projections (simplified)
        projected_cpu = cpu.avg_value * 1.2  # 20% growth assumed
        projected_memory = memory.avg_value * 1.2
        
        # Days until capacity (assuming 90% threshold)
        cpu_days = self._project_days_until_threshold(
            cpu.latest_value, projected_cpu, 90
        )
        memory_days = self._project_days_until_threshold(
            memory.latest_value, projected_memory, 90
        )
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'cpu': cpu.to_dict(),
            'memory': memory.to_dict(),
            'disk': disk.to_dict(),
            'current_utilization': {
                'cpu_percent': round(self._cpu_usage, 1),
                'memory_percent': round(self._memory_usage, 1),
                'disk_percent': round(self._disk_usage, 1)
            },
            'projections': {
                'cpu_growth_rate': f"{((cpu_trend - 1) * 100):.1f}%/day" if cpu_trend else "stable",
                'memory_growth_rate': f"{((memory_trend - 1) * 100):.1f}%/day" if memory_trend else "stable",
                'days_until_cpu_limit': cpu_days,
                'days_until_memory_limit': memory_days
            },
            'recommendations': self._generate_capacity_recommendations(
                cpu, memory, disk
            )
        }
    
    def _calculate_trend(self, series: TimeSeries) -> float:
        """Calculate trend (growth rate)"""
        if len(series.points) < 2:
            return 1.0
        
        first_half_avg = sum(
            p.value for p in series.points[:len(series.points)//2]
        ) / (len(series.points)//2) if len(series.points) > 1 else series.points[0].value
        
        second_half_avg = sum(
            p.value for p in series.points[len(series.points)//2:]
        ) / (len(series.points) - len(series.points)//2)
        
        if first_half_avg == 0:
            return 1.0
        
        return second_half_avg / first_half_avg
    
    def _project_days_until_threshold(
        self,
        current: float,
        projected: float,
        threshold: float
    ) -> Optional[int]:
        """Project days until threshold is reached"""
        if projected <= current:
            return None
        
        daily_growth = projected - current
        remaining = threshold - current
        
        if daily_growth <= 0:
            return None
        
        return int(remaining / daily_growth)
    
    def _generate_capacity_recommendations(
        self,
        cpu: TimeSeries,
        memory: TimeSeries,
        disk: TimeSeries
    ) -> List[str]:
        """Generate capacity recommendations"""
        recommendations = []
        
        if cpu.max_value > 80:
            recommendations.append(
                "CPU utilization frequently exceeds 80%. Consider scaling up or optimizing workloads."
            )
        
        if memory.max_value > 85:
            recommendations.append(
                "Memory utilization frequently exceeds 85%. Consider adding memory or optimizing usage."
            )
        
        if disk.max_value > 80:
            recommendations.append(
                "Disk utilization frequently exceeds 80%. Consider archival or storage expansion."
            )
        
        if not recommendations:
            recommendations.append("System capacity is healthy. Continue monitoring.")
        
        return recommendations
    
    def get_business_metrics(self, duration: int = 86400) -> Dict[str, Any]:
        """
        Get business metrics
        
        Args:
            duration: How many seconds of history
            
        Returns:
            Business metrics data
        """
        # Query trends
        queries = self._metrics.get_series("queries_total", duration=duration)
        query_trend = queries.latest_value - queries.points[0].value if queries.points else 0
        
        # Kernel trends
        kernels = self._metrics.get_series("kernels_total", duration=duration)
        kernel_trend = kernels.latest_value - kernels.points[0].value if kernels.points else 0
        
        # Revenue (placeholder - would integrate with billing)
        estimated_revenue = kernel_trend * 0.50  # $0.50 per kernel
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'period_hours': duration // 3600,
            'kernels': {
                'total': self._metrics.get_current_value("kernels_total") or 0,
                'active': self._metrics.get_current_value("kernels_active") or 0,
                'created_period': kernel_trend,
                'trend': 'up' if kernel_trend > 0 else 'down'
            },
            'queries': {
                'total': self._queries_executed,
                'per_second': self._metrics.get_current_value("queries_per_second") or 0,
                'avg_latency_ms': self._metrics.get_current_value("query_latency_avg") or 0,
                'period': query_trend,
                'trend': 'up' if query_trend > 0 else 'down'
            },
            'compression': {
                'total_jobs': self._compression_jobs,
                'period_jobs': 0  # Would track period-specific
            },
            'users': {
                'active': self._active_users,
                'period_new': 0  # Would track new users
            },
            'revenue': {
                'estimated': estimated_revenue,
                'currency': 'USD',
                'period': estimated_revenue
            }
        }
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary"""
        firing = self._alerts.get_alerts(status=AlertStatus.FIRING.value)
        acknowledged = self._alerts.get_alerts(status=AlertStatus.ACKNOWLEDGED.value)
        resolved = self._alerts.get_alerts(status=AlertStatus.RESOLVED.value, limit=100)
        
        # Count by severity
        by_severity = {}
        for alert in firing:
            by_severity[alert.severity] = by_severity.get(alert.severity, 0) + 1
        
        return {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'firing': len(firing),
            'acknowledged': len(acknowledged),
            'resolved_24h': len([
                a for a in resolved
                if a.resolved_at and
                datetime.fromisoformat(a.resolved_at) > datetime.now(timezone.utc) - timedelta(hours=24)
            ]),
            'by_severity': by_severity,
            'recent_alerts': [
                {
                    'id': a.alert_id,
                    'rule': a.rule_name,
                    'severity': a.severity,
                    'started': a.started_at,
                    'current_value': a.current_value,
                    'threshold': a.threshold
                }
                for a in firing[:10]
            ]
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        return {
            'metrics_count': len(self._metrics._series),
            'active_alerts': len(self._alerts.get_alerts(status=AlertStatus.FIRING.value)),
            'alerts_24h': len(self._alerts.get_alerts(limit=1000)),
            'history_points': len(self._metrics_history)
        }


# Singleton instance
_dashboard: Optional[OperationalDashboard] = None


def get_operational_dashboard() -> OperationalDashboard:
    """Get operational dashboard singleton"""
    global _dashboard
    
    if _dashboard is None:
        _dashboard = OperationalDashboard()
    
    return _dashboard


def init_operational_dashboard() -> OperationalDashboard:
    """Initialize operational dashboard system"""
    global _dashboard
    
    _dashboard = OperationalDashboard()
    _dashboard.start()
    
    return _dashboard
