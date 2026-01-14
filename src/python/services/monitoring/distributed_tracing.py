"""
KERNELIZE Platform - Distributed Tracing System
================================================

Performance debugging across services with distributed tracing.
Implements trace propagation, span creation, Jaeger integration,
and intelligent sampling.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple
from threading import RLock
from contextlib import contextmanager
import asyncio
import random

logger = logging.getLogger(__name__)


class SpanKind(Enum):
    """Types of spans"""
    SERVER = "server"
    CLIENT = "client"
    PRODUCER = "producer"
    CONSUMER = "consumer"
    INTERNAL = "internal"


class SpanStatus(Enum):
    """Span status codes"""
    OK = "ok"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class SpanContext:
    """Trace context passed between services"""
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: Optional[str] = None
    is_sampled: bool = True
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to trace headers for propagation"""
        return {
            'traceparent': f"00-{self.trace_id[:32]}-{self.span_id[:16]}-01",
            'tracestate': ','.join([
                f"{k}={v}" for k, v in self.baggage.items()
            ]) if self.baggage else ''
        }
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> 'SpanContext':
        """Create from trace headers"""
        traceparent = headers.get('traceparent', '')
        
        if traceparent:
            parts = traceparent.split('-')
            if len(parts) >= 3:
                return cls(
                    trace_id=parts[1],
                    span_id=parts[2],
                    is_sampled=parts[0] == '00'
                )
        
        return cls()
    
    @classmethod
    def create_new(cls) -> 'SpanContext':
        """Create new trace context"""
        return cls(
            trace_id=uuid.uuid4().hex[:32],
            span_id=uuid.uuid4().hex[:16],
            is_sampled=random.random() > 0.1  # 10% sampling by default
        )


@dataclass
class Span:
    """Individual trace span"""
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    
    name: str = ""
    kind: str = SpanKind.INTERNAL.value
    status: str = SpanStatus.OK.value
    
    # Timing
    start_time: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    end_time: Optional[str] = None
    duration_ms: float = 0.0
    
    # Location
    service_name: str = "kernelize"
    component: str = ""
    
    # Attributes
    attributes: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # Events
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Relationships
    child_span_ids: List[str] = field(default_factory=list)
    
    # Error
    error_message: Optional[str] = None
    error_stack: Optional[str] = None
    
    def start(self, trace_context: Optional[SpanContext] = None):
        """Start the span"""
        self.start_time = datetime.now(timezone.utc).isoformat()
        
        if trace_context:
            self.trace_id = trace_context.trace_id
            self.parent_span_id = trace_context.span_id
    
    def finish(self):
        """Finish the span"""
        self.end_time = datetime.now(timezone.utc).isoformat()
        
        start = datetime.fromisoformat(self.start_time)
        end = datetime.fromisoformat(self.end_time)
        self.duration_ms = (end - start).total_seconds() * 1000
    
    def set_attribute(self, key: str, value: Any):
        """Set span attribute"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add event to span"""
        self.events.append({
            'name': name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'attributes': attributes or {}
        })
    
    def set_error(self, message: str, stack: Optional[str] = None):
        """Set span error"""
        self.status = SpanStatus.ERROR.value
        self.error_message = message
        self.error_stack = stack
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'span_id': self.span_id,
            'trace_id': self.trace_id,
            'parent_span_id': self.parent_span_id,
            'name': self.name,
            'kind': self.kind,
            'status': self.status,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_ms': self.duration_ms,
            'service_name': self.service_name,
            'component': self.component,
            'attributes': self.attributes,
            'tags': self.tags,
            'events': self.events,
            'child_span_ids': self.child_span_ids,
            'error_message': self.error_message,
            'error_stack': self.error_stack
        }
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class Trace:
    """Complete trace with multiple spans"""
    trace_id: str = ""
    spans: List[Span] = field(default_factory=list)
    
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    
    # Statistics
    total_spans: int = 0
    total_duration_ms: float = 0.0
    error_count: int = 0
    
    def add_span(self, span: Span):
        """Add span to trace"""
        self.spans.append(span)
        self.total_spans = len(self.spans)
        
        # Update statistics
        if span.duration_ms > self.total_duration_ms:
            self.total_duration_ms = span.duration_ms
        
        if span.status == SpanStatus.ERROR.value:
            self.error_count += 1
    
    def get_root_span(self) -> Optional[Span]:
        """Get the root span (no parent)"""
        parent_ids = {s.parent_span_id for s in self.spans if s.parent_span_id}
        for span in self.spans:
            if span.span_id not in parent_ids:
                return span
        return self.spans[0] if self.spans else None
    
    def get_spans_by_name(self, name: str) -> List[Span]:
        """Get spans by name"""
        return [s for s in self.spans if s.name == name]
    
    def get_spans_by_component(self, component: str) -> List[Span]:
        """Get spans by component"""
        return [s for s in self.spans if s.component == component]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'trace_id': self.trace_id,
            'spans': [s.to_dict() for s in self.spans],
            'created_at': self.created_at,
            'completed_at': self.completed_at,
            'total_spans': self.total_spans,
            'total_duration_ms': self.total_duration_ms,
            'error_count': self.error_count
        }


@dataclass
class TracingConfiguration:
    """Tracing configuration"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Tracing"
    
    # Sampling
    sampling_rate: float = 0.1  # 10% of traces
    sampling_minimum: int = 0   # Always sample at least this many
    sampling_based_on_parent: bool = True
    
    # Propagation
    propagation_format: str = "traceparent"  # w3c, b3, jaeger
    propagate_baggage: bool = True
    
    # Export
    export_enabled: bool = True
    export_batch_size: int = 100
    export_interval_seconds: int = 5
    
    # Jaeger (example)
    jaeger_enabled: bool = False
    jaeger_endpoint: Optional[str] = None
    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    
    # Local storage
    local_storage_enabled: bool = True
    local_storage_max_traces: int = 10000
    
    is_active: bool = True


class SpanManager:
    """Manages span lifecycle"""
    
    def __init__(self, config: Optional[TracingConfiguration] = None):
        """
        Initialize span manager
        
        Args:
            config: Tracing configuration
        """
        self.config = config or TracingConfiguration()
        self._active_spans: Dict[str, Span] = {}
        self._span_stack: List[str] = []  # Stack of active span IDs
        self._lock = RLock()
    
    def start_span(
        self,
        name: str,
        kind: str = SpanKind.INTERNAL.value,
        component: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        trace_context: Optional[SpanContext] = None
    ) -> Span:
        """
        Start a new span
        
        Args:
            name: Span name
            kind: Span kind
            component: Component name
            attributes: Initial attributes
            trace_context: Parent trace context
            
        Returns:
            Created span
        """
        span = Span(
            name=name,
            kind=kind,
            component=component,
            attributes=attributes or {},
            service_name="kernelize"
        )
        
        # Set parent
        if trace_context:
            span.trace_id = trace_context.trace_id
            span.parent_span_id = trace_context.span_id
            
            # Sample decision
            if self.config.sampling_based_on_parent:
                span.is_sampled = trace_context.is_sampled
            else:
                span.is_sampled = random.random() < self.config.sampling_rate
        
        # Start span
        span.start(trace_context)
        
        with self._lock:
            self._active_spans[span.span_id] = span
            self._span_stack.append(span.span_id)
        
        return span
    
    def finish_span(self, span: Span, result: Optional[Any] = None):
        """
        Finish a span
        
        Args:
            span: Span to finish
            result: Optional result to record
        """
        span.finish()
        
        if result is not None:
            try:
                span.set_attribute('result', str(result)[:1000])
            except:
                pass
        
        with self._lock:
            # Remove from active spans
            if span.span_id in self._active_spans:
                del self._active_spans[span.span_id]
            
            # Remove from stack
            if span.span_id in self._span_stack:
                self._span_stack.remove(span.span_id)
            
            # Update parent
            if span.parent_span_id and span.parent_span_id in self._active_spans:
                parent = self._active_spans[span.parent_span_id]
                parent.child_span_ids.append(span.span_id)
    
    def get_current_span(self) -> Optional[Span]:
        """Get the currently active span"""
        with self._lock:
            if self._span_stack:
                span_id = self._span_stack[-1]
                return self._active_spans.get(span_id)
        return None
    
    def get_active_span_count(self) -> int:
        """Get count of active spans"""
        with self._lock:
            return len(self._active_spans)


class TraceCollector:
    """Collects and exports traces"""
    
    def __init__(self, config: Optional[TracingConfiguration] = None):
        """
        Initialize trace collector
        
        Args:
            config: Tracing configuration
        """
        self.config = config or TracingConfiguration()
        
        # Local storage
        self._traces: Dict[str, Trace] = {}
        self._lock = RLock()
        
        # Export queue
        self._export_queue: List[Trace] = []
        self._export_lock = RLock()
        
        # Statistics
        self._stats = {
            'traces_collected': 0,
            'spans_collected': 0,
            'traces_exported': 0,
            'traces_sampled_out': 0
        }
        self._stats_lock = RLock()
    
    def collect_trace(self, trace: Trace) -> bool:
        """
        Collect a completed trace
        
        Args:
            trace: Trace to collect
            
        Returns:
            True if collected
        """
        # Check sampling
        if not self._should_sample(trace):
            with self._stats_lock:
                self._stats['traces_sampled_out'] += 1
            return False
        
        with self._lock:
            self._traces[trace.trace_id] = trace
            
            # Update stats
            with self._stats_lock:
                self._stats['traces_collected'] += 1
                self._stats['spans_collected'] += trace.total_spans
        
        # Check if should export
        if self.config.export_enabled:
            self._queue_for_export(trace)
        
        return True
    
    def _should_sample(self, trace: Trace) -> bool:
        """Determine if trace should be sampled"""
        # Always sample errors
        if trace.error_count > 0:
            return True
        
        # Check sampling rate
        return random.random() < self.config.sampling_rate
    
    def _queue_for_export(self, trace: Trace):
        """Queue trace for export"""
        with self._export_lock:
            self._export_queue.append(trace)
            
            # Check batch size
            if len(self._export_queue) >= self.config.export_batch_size:
                self._export_traces()
    
    def _export_traces(self):
        """Export queued traces"""
        with self._export_lock:
            traces = self._export_queue[:self.config.export_batch_size]
            self._export_queue = self._export_queue[self.config.export_batch_size:]
        
        # Export to configured backends
        if self.config.jaeger_enabled:
            self._export_to_jaeger(traces)
        
        with self._stats_lock:
            self._stats['traces_exported'] += len(traces)
    
    def _export_to_jaeger(self, traces: List[Trace]):
        """Export traces to Jaeger"""
        # Would implement actual Jaeger client integration here
        logger.debug(f"Would export {len(traces)} traces to Jaeger")
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get trace by ID"""
        with self._lock:
            return self._traces.get(trace_id)
    
    def get_traces(
        self,
        service_name: Optional[str] = None,
        span_name: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100
    ) -> List[Trace]:
        """
        Query traces
        
        Args:
            service_name: Filter by service
            span_name: Filter by span name
            start_time: Filter by start time
            end_time: Filter by end time
            limit: Maximum results
            
        Returns:
            Matching traces
        """
        with self._lock:
            traces = list(self._traces.values())
        
        # Apply filters
        if service_name:
            traces = [t for t in traces if any(
                s.service_name == service_name for s in t.spans
            )]
        
        if span_name:
            traces = [t for t in traces if any(
                s.name == span_name for s in t.spans
            )]
        
        if start_time:
            cutoff = datetime.fromisoformat(start_time)
            traces = [t for t in traces if datetime.fromisoformat(t.created_at) >= cutoff]
        
        if end_time:
            cutoff = datetime.fromisoformat(end_time)
            traces = [t for t in traces if datetime.fromisoformat(t.created_at) <= cutoff]
        
        # Sort by duration (longest first)
        traces.sort(key=lambda t: t.total_duration_ms, reverse=True)
        
        return traces[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collector statistics"""
        with self._stats_lock:
            stats = dict(self._stats)
        
        with self._lock:
            stats['stored_traces'] = len(self._traces)
            stats['queued_exports'] = len(self._export_queue)
        
        return stats


class DistributedTracer:
    """
    Main distributed tracing service
    
    Provides high-level API for trace creation, propagation,
    and collection with automatic context management.
    """
    
    def __init__(self, config: Optional[TracingConfiguration] = None):
        """
        Initialize distributed tracer
        
        Args:
            config: Tracing configuration
        """
        self.config = config or TracingConfiguration()
        
        self._span_manager = SpanManager(config)
        self._collector = TraceCollector(config)
        
        # Context storage
        self._context_key = 'kernelize_trace_context'
        
        # Active trace
        self._current_trace: Optional[Trace] = None
    
    @contextmanager
    async def start_trace(
        self,
        name: str,
        service_name: str = "kernelize",
        attributes: Optional[Dict[str, Any]] = None
    ):
        """
        Context manager for creating a trace
        
        Usage:
            async with tracer.start_trace("my_operation") as trace:
                # Create spans within the trace
                pass
        """
        # Create new trace context
        context = SpanContext.create_new()
        
        # Create trace
        trace = Trace(trace_id=context.trace_id)
        
        # Store previous context
        previous_context = self._get_context()
        
        try:
            # Set context
            self._set_context(context)
            
            # Start root span
            span = self._span_manager.start_span(
                name=name,
                kind=SpanKind.SERVER.value,
                component=service_name,
                attributes=attributes,
                trace_context=context
            )
            
            self._current_trace = trace
            
            yield trace
            
        finally:
            # Finish span
            if self._span_manager.get_current_span():
                self._span_manager.finish_span(
                    self._span_manager.get_current_span()
                )
            
            # Complete trace
            trace.completed_at = datetime.now(timezone.utc).isoformat()
            trace.total_duration_ms = sum(s.duration_ms for s in trace.spans)
            
            # Collect trace
            self._collector.collect_trace(trace)
            
            # Restore previous context
            self._set_context(previous_context)
            self._current_trace = None
    
    def start_span(
        self,
        name: str,
        kind: str = SpanKind.INTERNAL.value,
        component: str = "",
        attributes: Optional[Dict[str, Any]] = None
    ) -> Span:
        """
        Start a new span in current trace
        
        Args:
            name: Span name
            kind: Span kind
            component: Component name
            attributes: Span attributes
            
        Returns:
            Created span
        """
        context = self._get_context()
        
        span = self._span_manager.start_span(
            name=name,
            kind=kind,
            component=component,
            attributes=attributes,
            trace_context=context
        )
        
        # Add to current trace
        if self._current_trace:
            self._current_trace.add_span(span)
        
        return span
    
    def finish_span(self, span: Span, result: Optional[Any] = None):
        """
        Finish a span
        
        Args:
            span: Span to finish
            result: Optional result to record
        """
        self._span_manager.finish_span(span, result)
    
    def inject_context(self, carrier: Dict[str, str]) -> Dict[str, str]:
        """
        Inject trace context into carrier for propagation
        
        Args:
            carrier: Carrier dictionary (e.g., HTTP headers)
            
        Returns:
            Carrier with trace context
        """
        context = self._get_context()
        
        if self.config.propagation_format == "traceparent":
            headers = context.to_headers()
            carrier.update({k: v for k, v in headers.items() if v})
        
        return carrier
    
    def extract_context(self, carrier: Dict[str, str]) -> SpanContext:
        """
        Extract trace context from carrier
        
        Args:
            carrier: Carrier dictionary (e.g., HTTP headers)
            
        Returns:
            Extracted span context
        """
        if self.config.propagation_format == "traceparent":
            return SpanContext.from_headers(carrier)
        
        return SpanContext()
    
    def _get_context(self) -> Optional[SpanContext]:
        """Get current trace context"""
        # Would integrate with async local storage in production
        return None
    
    def _set_context(self, context: Optional[SpanContext]):
        """Set current trace context"""
        pass
    
    async def trace_operation(
        self,
        name: str,
        operation: Callable,
        kind: str = SpanKind.INTERNAL.value,
        component: str = "",
        attributes: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ):
        """
        Trace an operation
        
        Args:
            name: Operation name
            operation: Async or sync function to trace
            kind: Span kind
            component: Component name
            attributes: Span attributes
            *args, **kwargs: Arguments to operation
            
        Returns:
            Operation result
        """
        span = self.start_span(name, kind, component, attributes)
        
        try:
            result = await operation(*args, **kwargs)
            return result
        except Exception as e:
            span.set_error(str(e))
            raise
        finally:
            self.finish_span(span)
    
    def get_current_trace(self) -> Optional[Trace]:
        """Get current trace"""
        return self._current_trace
    
    def query_traces(
        self,
        service_name: Optional[str] = None,
        span_name: Optional[str] = None,
        limit: int = 100
    ) -> List[Trace]:
        """Query collected traces"""
        return self._collector.get_traces(
            service_name=service_name,
            span_name=span_name,
            limit=limit
        )
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get specific trace"""
        return self._collector.get_trace(trace_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        stats = self._collector.get_statistics()
        stats['active_spans'] = self._span_manager.get_active_span_count()
        return stats


# Decorator for automatic tracing
def traced(
    name: Optional[str] = None,
    kind: str = SpanKind.INTERNAL.value,
    component: str = ""
):
    """
    Decorator for automatic span creation
    
    Args:
        name: Span name (defaults to function name)
        kind: Span kind
        component: Component name
    """
    def decorator(func):
        span_name = name or func.__name__
        
        async def async_wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            span = tracer.start_span(
                name=span_name,
                kind=kind,
                component=component
            )
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                span.set_error(str(e))
                raise
            finally:
                tracer.finish_span(span)
        
        def sync_wrapper(*args, **kwargs):
            tracer = get_tracer()
            
            span = tracer.start_span(
                name=span_name,
                kind=kind,
                component=component
            )
            
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                span.set_error(str(e))
                raise
            finally:
                tracer.finish_span(span)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Singleton instance
_tracer: Optional[DistributedTracer] = None


def get_tracer() -> DistributedTracer:
    """Get distributed tracer singleton"""
    global _tracer
    
    if _tracer is None:
        _tracer = DistributedTracer()
    
    return _tracer


def init_tracing(
    config: Optional[TracingConfiguration] = None
) -> DistributedTracer:
    """Initialize tracing system"""
    global _tracer
    
    _tracer = DistributedTracer(config)
    
    return _tracer
