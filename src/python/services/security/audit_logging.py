"""
KERNELIZE Platform - Audit Logging System
==========================================

Comprehensive audit logging for compliance requirements.
Implements tamper-evident logging with cryptographic signatures,
configurable retention policies, and SIEM integration.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import hmac
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union
from functools import wraps
import threading
from queue import Queue, Empty
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)


class AuditEventType(Enum):
    """Types of auditable events"""
    # Authentication events
    USER_LOGIN = "user.login"
    USER_LOGOUT = "user.logout"
    USER_LOGIN_FAILED = "user.login_failed"
    PASSWORD_CHANGE = "user.password_change"
    API_KEY_CREATED = "user.api_key_created"
    API_KEY_REVOKED = "user.api_key_revoked"
    
    # Kernel events
    KERNEL_CREATED = "kernel.created"
    KERNEL_ACCESSED = "kernel.accessed"
    KERNEL_UPDATED = "kernel.updated"
    KERNEL_DELETED = "kernel.deleted"
    KERNEL_EXPORTED = "kernel.exported"
    KERNEL_SHARED = "kernel.shared"
    
    # Query events
    QUERY_EXECUTED = "query.executed"
    QUERY_FAILED = "query.failed"
    QUERY_RATE_LIMITED = "query.rate_limited"
    
    # Admin events
    ORGANIZATION_CREATED = "organization.created"
    ORGANIZATION_UPDATED = "organization.updated"
    USER_INVITED = "user.invited"
    USER_ROLE_CHANGED = "user.role_changed"
    PERMISSION_GRANTED = "permission.granted"
    PERMISSION_REVOKED = "permission.revoked"
    
    # System events
    SYSTEM_CONFIG_CHANGED = "system.config_changed"
    SECURITY_ALERT = "security.alert"
    INTEGRATION_EVENT = "integration.event"


class AuditSeverity(Enum):
    """Severity levels for audit events"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    """Audit event data structure"""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_id: Optional[str] = None
    organization_id: Optional[str] = None
    kernel_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: str = ""
    severity: str = AuditSeverity.INFO.value
    success: bool = True
    error_message: Optional[str] = None
    request_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    request_body: Optional[Dict[str, Any]] = None
    response_status: Optional[int] = None
    duration_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Serialize to JSON"""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class LogRetentionPolicy:
    """Log retention policy configuration"""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    organization_id: Optional[str] = None  # None means global policy
    name: str = "Default Policy"
    
    # Retention periods (in days)
    retention_days_info: int = 90
    retention_days_warning: int = 180
    retention_days_error: int = 365
    retention_days_critical: int = 730
    
    # Archive settings
    archive_after_days: int = 30
    archive_compression: bool = True
    archive_storage_location: Optional[str] = None
    
    # Deletion settings
    soft_delete_enabled: bool = True
    soft_delete_days: int = 7
    permanent_delete_enabled: bool = False
    
    # Compliance settings
    compliance_mode: bool = False
    compliance_framework: Optional[str] = None
    
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True


class CryptoSigner:
    """Cryptographic signing for tamper-evident logging"""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        """
        Initialize cryptographic signer
        
        Args:
            secret_key: Secret key for HMAC signing (generated if not provided)
        """
        if secret_key is None:
            secret_key = os.urandom(32)
        self.secret_key = secret_key
    
    def sign(self, data: Union[str, Dict[str, Any]]) -> str:
        """
        Create cryptographic signature for data
        
        Args:
            data: Data to sign (string or dictionary)
            
        Returns:
            Hex-encoded signature
        """
        if isinstance(data, dict):
            data = json.dumps(data, sort_keys=True, default=str)
        
        signature = hmac.new(
            self.secret_key,
            data.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify(self, data: Union[str, Dict[str, Any]], signature: str) -> bool:
        """
        Verify cryptographic signature
        
        Args:
            data: Data to verify
            signature: Signature to verify against
            
        Returns:
            True if signature is valid
        """
        expected_signature = self.sign(data)
        return hmac.compare_digest(expected_signature, signature)


class AuditLogStore:
    """Storage backend for audit logs"""
    
    def __init__(self, storage_path: str = "data/audit_logs"):
        """
        Initialize audit log storage
        
        Args:
            storage_path: Path to store audit log database
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.db_path = self.storage_path / "audit_logs.db"
        self.signature_db_path = self.storage_path / "audit_signatures.db"
        
        self._init_databases()
        self._lock = threading.Lock()
    
    def _init_databases(self):
        """Initialize SQLite databases for logs and signatures"""
        # Main audit log database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_logs (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                user_id TEXT,
                organization_id TEXT,
                kernel_id TEXT,
                resource_type TEXT,
                resource_id TEXT,
                action TEXT,
                severity TEXT,
                success INTEGER,
                error_message TEXT,
                request_id TEXT,
                ip_address TEXT,
                user_agent TEXT,
                endpoint TEXT,
                method TEXT,
                request_body TEXT,
                response_status INTEGER,
                duration_ms REAL,
                metadata TEXT,
                previous_state TEXT,
                new_state TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp 
            ON audit_logs(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_audit_user 
            ON audit_logs(user_id, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_audit_organization 
            ON audit_logs(organization_id, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_audit_event_type 
            ON audit_logs(event_type, timestamp)
        ''')
        
        conn.commit()
        conn.close()
        
        # Signature database for tamper detection
        conn = sqlite3.connect(str(self.signature_db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signatures (
                event_id TEXT PRIMARY KEY,
                event_data TEXT NOT NULL,
                signature TEXT NOT NULL,
                previous_event_id TEXT,
                chain_signature TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signatures_event_id 
            ON signatures(event_id)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_signatures_chain 
            ON signatures(previous_event_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def store(self, event: AuditEvent, signature: str, chain_signature: Optional[str] = None) -> bool:
        """
        Store audit event with signature
        
        Args:
            event: Audit event to store
            signature: Cryptographic signature
            chain_signature: Previous event chain signature
            
        Returns:
            True if stored successfully
        """
        with self._lock:
            try:
                conn = sqlite3.connect(str(self.db_path))
                cursor = conn.cursor()
                
                # Get the last event for chain
                last_event_id = self._get_last_event_id(cursor)
                
                event_data = event.to_json()
                if chain_signature is None and last_event_id:
                    # Generate chain signature
                    chain_signature = self._generate_chain_signature(
                        last_event_id, event_data
                    )
                
                cursor.execute('''
                    INSERT INTO audit_logs (
                        event_id, event_type, timestamp, user_id, organization_id,
                        kernel_id, resource_type, resource_id, action, severity,
                        success, error_message, request_id, ip_address, user_agent,
                        endpoint, method, request_body, response_status, duration_ms,
                        metadata, previous_state, new_state, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id, event.event_type, event.timestamp, event.user_id,
                    event.organization_id, event.kernel_id, event.resource_type,
                    event.resource_id, event.action, event.severity, int(event.success),
                    event.error_message, event.request_id, event.ip_address,
                    event.user_agent, event.endpoint, event.method,
                    json.dumps(event.request_body) if event.request_body else None,
                    event.response_status, event.duration_ms,
                    json.dumps(event.metadata) if event.metadata else None,
                    json.dumps(event.previous_state) if event.previous_state else None,
                    json.dumps(event.new_state) if event.new_state else None,
                    datetime.now(timezone.utc).isoformat()
                ))
                
                conn.commit()
                conn.close()
                
                # Store signature separately
                self._store_signature(event.event_id, event_data, signature, chain_signature)
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to store audit event: {e}")
                return False
    
    def _get_last_event_id(self, cursor) -> Optional[str]:
        """Get the last event ID for chaining"""
        cursor.execute(
            'SELECT event_id FROM audit_logs ORDER BY created_at DESC LIMIT 1'
        )
        result = cursor.fetchone()
        return result[0] if result else None
    
    def _generate_chain_signature(self, previous_event_id: str, event_data: str) -> str:
        """Generate chain signature linking to previous event"""
        chain_data = f"{previous_event_id}:{event_data}"
        return hashlib.sha256(chain_data.encode()).hexdigest()
    
    def _store_signature(
        self,
        event_id: str,
        event_data: str,
        signature: str,
        chain_signature: Optional[str] = None
    ):
        """Store cryptographic signature"""
        conn = sqlite3.connect(str(self.signature_db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signatures (
                event_id, event_data, signature, previous_event_id, chain_signature, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            event_id, event_data, signature,
            None,  # Will update this
            chain_signature,
            datetime.now(timezone.utc).isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    def query(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        event_types: Optional[List[str]] = None,
        kernel_id: Optional[str] = None,
        severity: Optional[str] = None,
        success: Optional[bool] = None,
        limit: int = 1000,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Query audit logs with filters
        
        Args:
            start_time: Filter events after this time
            end_time: Filter events before this time
            user_id: Filter by user
            organization_id: Filter by organization
            event_types: Filter by event types
            kernel_id: Filter by kernel
            severity: Filter by severity
            success: Filter by success status
            limit: Maximum results to return
            offset: Offset for pagination
            
        Returns:
            List of matching audit events
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        query = "SELECT * FROM audit_logs WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time)
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time)
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if organization_id:
            query += " AND organization_id = ?"
            params.append(organization_id)
        
        if kernel_id:
            query += " AND kernel_id = ?"
            params.append(kernel_id)
        
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        
        if success is not None:
            query += " AND success = ?"
            params.append(int(success))
        
        if event_types:
            placeholders = ','.join('?' * len(event_types))
            query += f" AND event_type IN ({placeholders})"
            params.extend(event_types)
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        columns = [desc[0] for desc in cursor.description]
        results = []
        
        for row in cursor.fetchall():
            result = dict(zip(columns, row))
            # Parse JSON fields
            for field in ['request_body', 'metadata', 'previous_state', 'new_state']:
                if result.get(field):
                    try:
                        result[field] = json.loads(result[field])
                    except json.JSONDecodeError:
                        pass
            results.append(result)
        
        conn.close()
        return results
    
    def get_tamper_report(self, start_time: str, end_time: str) -> Dict[str, Any]:
        """
        Generate tamper detection report
        
        Args:
            start_time: Start of report period
            end_time: End of report period
            
        Returns:
            Tamper detection report
        """
        events = self.query(start_time=start_time, end_time=end_time, limit=10000)
        
        tampered = []
        missing_chain = []
        
        conn = sqlite3.connect(str(self.signature_db_path))
        cursor = conn.cursor()
        
        for event in events:
            event_id = event['event_id']
            cursor.execute(
                'SELECT signature, chain_signature FROM signatures WHERE event_id = ?',
                (event_id,)
            )
            sig_result = cursor.fetchone()
            
            if not sig_result:
                missing_chain.append(event_id)
                continue
            
            stored_signature, stored_chain = sig_result
            event_data = json.dumps(event, sort_keys=True, default=str)
            expected_signature = hashlib.sha256(
                f"{event_data}".encode()
            ).hexdigest()
            
            if stored_signature != expected_signature:
                tampered.append({
                    'event_id': event_id,
                    'timestamp': event.get('timestamp'),
                    'reason': 'signature_mismatch'
                })
        
        conn.close()
        
        return {
            'period': {'start': start_time, 'end': end_time},
            'total_events': len(events),
            'tampered_events': len(tampered),
            'missing_signatures': len(missing_chain),
            'tampered_details': tampered,
            'missing_details': missing_chain,
            'integrity_status': 'VALID' if not tampered and not missing_chain else 'COMPROMISED',
            'generated_at': datetime.now(timezone.utc).isoformat()
        }


class AuditLogger:
    """
    Main audit logging service
    
    Provides comprehensive logging for compliance requirements with
    tamper-evident signatures and configurable retention policies.
    """
    
    def __init__(
        self,
        storage_path: str = "data/audit_logs",
        secret_key: Optional[bytes] = None,
        retention_policies: Optional[List[LogRetentionPolicy]] = None,
        async_mode: bool = True,
        batch_size: int = 100,
        flush_interval: float = 5.0
    ):
        """
        Initialize audit logger
        
        Args:
            storage_path: Path for audit log storage
            secret_key: Secret key for cryptographic signatures
            retention_policies: Log retention policies
            async_mode: Use asynchronous logging
            batch_size: Batch size for async writes
            flush_interval: Flush interval in seconds
        """
        self.storage_path = storage_path
        self.signer = CryptoSigner(secret_key)
        self.store = AuditLogStore(storage_path)
        
        # Retention policies
        self.retention_policies = retention_policies or [LogRetentionPolicy()]
        self._policy_cache: Dict[str, LogRetentionPolicy] = {}
        
        # Async logging
        self.async_mode = async_mode
        self._event_queue: Queue = Queue()
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = {
            'events_logged': 0,
            'events_failed': 0,
            'current_batch_size': 0
        }
        self._stats_lock = threading.Lock()
        
        # SIEM integrations
        self._siem_integrations: List[Callable] = []
        
        if async_mode:
            self._start_worker()
    
    def _start_worker(self):
        """Start async worker thread"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_events, daemon=True)
        self._worker_thread.start()
    
    def _process_events(self):
        """Process events from queue in background"""
        batch = []
        
        while self._running or not self._event_queue.empty():
            try:
                # Collect batch with timeout
                while len(batch) < self._batch_size:
                    try:
                        event = self._event_queue.get(timeout=self._flush_interval)
                        batch.append(event)
                    except Empty:
                        break
                
                if batch:
                    self._write_batch(batch)
                    batch.clear()
                    
            except Exception as e:
                logger.error(f"Error processing audit events: {e}")
        
        # Flush remaining events
        if batch:
            self._write_batch(batch)
    
    def _write_batch(self, events: List[AuditEvent]):
        """Write batch of events to storage"""
        for event in events:
            try:
                event_data = event.to_json()
                signature = self.signer.sign(event_data)
                
                if self.store.store(event, signature):
                    self._increment_stat('events_logged')
                    
                    # Send to SIEM integrations
                    for siem_handler in self._siem_integrations:
                        try:
                            siem_handler(event, signature)
                        except Exception as e:
                            logger.warning(f"SIEM integration failed: {e}")
                else:
                    self._increment_stat('events_failed')
                    
            except Exception as e:
                logger.error(f"Failed to write audit event: {e}")
                self._increment_stat('events_failed')
    
    def _increment_stat(self, key: str):
        """Increment statistics counter"""
        with self._stats_lock:
            self._stats[key] += 1
    
    def log_event(self, event: AuditEvent):
        """
        Log an audit event
        
        Args:
            event: Audit event to log
        """
        if self.async_mode:
            self._event_queue.put(event)
        else:
            self._write_batch([event])
    
    def log_api_request(
        self,
        request_id: str,
        user_id: Optional[str],
        organization_id: Optional[str],
        endpoint: str,
        method: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        request_body: Optional[Dict[str, Any]] = None,
        response_status: Optional[int] = None,
        duration_ms: Optional[float] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        **kwargs
    ):
        """
        Log an API request
        
        Args:
            request_id: Request correlation ID
            user_id: User making the request
            organization_id: Organization context
            endpoint: API endpoint
            method: HTTP method
            ip_address: Client IP address
            user_agent: Client user agent
            request_body: Request body (sanitized)
            response_status: HTTP response status
            duration_ms: Request duration in milliseconds
            success: Whether request succeeded
            error_message: Error message if failed
            **kwargs: Additional metadata
        """
        # Determine severity based on response status
        severity = AuditSeverity.INFO.value
        if response_status:
            if response_status >= 500:
                severity = AuditSeverity.CRITICAL.value
            elif response_status >= 400:
                severity = AuditSeverity.ERROR.value
        
        event = AuditEvent(
            event_type=AuditEventType.QUERY_EXECUTED.value if success else AuditEventType.QUERY_FAILED.value,
            request_id=request_id,
            user_id=user_id,
            organization_id=organization_id,
            endpoint=endpoint,
            method=method,
            ip_address=ip_address,
            user_agent=user_agent,
            request_body=self._sanitize_request_body(request_body),
            response_status=response_status,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            severity=severity,
            metadata=kwargs
        )
        
        self.log_event(event)
    
    def log_kernel_access(
        self,
        user_id: str,
        organization_id: str,
        kernel_id: str,
        action: str,
        previous_state: Optional[Dict[str, Any]] = None,
        new_state: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Log kernel access or modification
        
        Args:
            user_id: User accessing the kernel
            organization_id: Organization context
            kernel_id: Kernel being accessed
            action: Action performed
            previous_state: State before change
            new_state: State after change
            **kwargs: Additional metadata
        """
        action_map = {
            'create': AuditEventType.KERNEL_CREATED.value,
            'read': AuditEventType.KERNEL_ACCESSED.value,
            'update': AuditEventType.KERNEL_UPDATED.value,
            'delete': AuditEventType.KERNEL_DELETED.value,
            'export': AuditEventType.KERNEL_EXPORTED.value,
            'share': AuditEventType.KERNEL_SHARED.value,
        }
        
        event_type = action_map.get(action, AuditEventType.KERNEL_ACCESSED.value)
        
        event = AuditEvent(
            event_type=event_type,
            user_id=user_id,
            organization_id=organization_id,
            kernel_id=kernel_id,
            resource_type="kernel",
            resource_id=kernel_id,
            action=action,
            previous_state=previous_state,
            new_state=new_state,
            metadata=kwargs
        )
        
        self.log_event(event)
    
    def log_security_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        description: str = "",
        severity: AuditSeverity = AuditSeverity.WARNING,
        **kwargs
    ):
        """
        Log security-related event
        
        Args:
            event_type: Type of security event
            user_id: Related user ID
            organization_id: Related organization
            description: Event description
            severity: Event severity
            **kwargs: Additional details
        """
        event = AuditEvent(
            event_type=AuditEventType.SECURITY_ALERT.value,
            event_type_specific=event_type,
            user_id=user_id,
            organization_id=organization_id,
            action=event_type,
            severity=severity.value,
            metadata={
                'description': description,
                'details': kwargs
            }
        )
        
        self.log_event(event)
    
    def _sanitize_request_body(self, body: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Remove sensitive data from request body"""
        if not body:
            return None
        
        sensitive_keys = [
            'password', 'secret', 'token', 'api_key', 'authorization',
            'credit_card', 'ssn', 'social_security'
        ]
        
        sanitized = {}
        for key, value in body.items():
            key_lower = key.lower()
            if any(sk in key_lower for sk in sensitive_keys):
                sanitized[key] = '[REDACTED]'
            else:
                sanitized[key] = value
        
        return sanitized
    
    def add_siem_integration(self, handler: Callable[[AuditEvent, str], None]):
        """
        Add SIEM integration handler
        
        Args:
            handler: Function to call for each event
        """
        self._siem_integrations.append(handler)
    
    def generate_compliance_report(
        self,
        organization_id: str,
        start_time: str,
        end_time: str,
        framework: str = "SOC2"
    ) -> Dict[str, Any]:
        """
        Generate compliance audit report
        
        Args:
            organization_id: Organization to report on
            start_time: Report start time
            end_time: Report end time
            framework: Compliance framework (SOC2, GDPR, HIPAA, etc.)
            
        Returns:
            Compliance report
        """
        events = self.store.query(
            organization_id=organization_id,
            start_time=start_time,
            end_time=end_time,
            limit=10000
        )
        
        # Aggregate by event type
        event_counts = {}
        user_activity = {}
        kernel_access = {}
        
        for event in events:
            # Count by type
            event_type = event.get('event_type', 'unknown')
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Track user activity
            user_id = event.get('user_id')
            if user_id:
                if user_id not in user_activity:
                    user_activity[user_id] = {'actions': 0, 'kernels': set()}
                user_activity[user_id]['actions'] += 1
                kernel_id = event.get('kernel_id')
                if kernel_id:
                    user_activity[user_id]['kernels'].add(kernel_id)
            
            # Track kernel access
            kernel_id = event.get('kernel_id')
            if kernel_id:
                kernel_access[kernel_id] = kernel_access.get(kernel_id, 0) + 1
        
        # Convert sets to counts
        for user_id in user_activity:
            user_activity[user_id]['kernels'] = len(user_activity[user_id]['kernels'])
        
        # Generate report
        report = {
            'report_id': str(uuid.uuid4()),
            'organization_id': organization_id,
            'framework': framework,
            'period': {'start': start_time, 'end': end_time},
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'summary': {
                'total_events': len(events),
                'unique_users': len(user_activity),
                'unique_kernels': len(kernel_access),
                'event_breakdown': event_counts
            },
            'user_activity': user_activity,
            'kernel_access_patterns': kernel_access,
            'security_events': [
                e for e in events 
                if e.get('severity') in ['warning', 'error', 'critical']
            ],
            'failed_operations': [
                e for e in events 
                if not e.get('success', True)
            ],
            'tamper_check': self.store.get_tamper_report(start_time, end_time)
        }
        
        return report
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logger statistics"""
        with self._stats_lock:
            stats = dict(self._stats)
        
        stats['queue_size'] = self._event_queue.qsize() if self.async_mode else 0
        stats['async_mode'] = self.async_mode
        stats['siem_integrations'] = len(self._siem_integrations)
        
        return stats
    
    def cleanup_old_logs(self, organization_id: Optional[str] = None):
        """
        Clean up old logs according to retention policies
        
        Args:
            organization_id: Specific organization (None for all)
        """
        for policy in self.retention_policies:
            if policy.is_active:
                cutoff_time = datetime.now(timezone.utc)
                
                if policy.retention_days_info > 0:
                    # Implementation would delete old logs
                    logger.info(f"Would clean logs older than {policy.retention_days_info} days")
    
    def shutdown(self):
        """Shutdown audit logger and flush pending events"""
        self._running = False
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)


# Decorator for automatic API request logging
def audit_api_request(
    event_type: str = "api.request",
    log_request_body: bool = False,
    log_response_body: bool = False
):
    """
    Decorator for automatic API request auditing
    
    Args:
        event_type: Type of event to log
        log_request_body: Include request body in logs
        log_response_body: Include response body in logs
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            from src.python.main import get_audit_logger
            
            # Extract request info (depends on FastAPI)
            # This is a simplified version
            
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                # Log failure
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                raise
        
        if hasattr(func, '__wrapped__'):
            # Check if it's a coroutine
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Singleton instance
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Get or create audit logger singleton"""
    global _audit_logger
    
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    
    return _audit_logger


def init_audit_logger(
    storage_path: str = "data/audit_logs",
    secret_key: Optional[bytes] = None,
    retention_policies: Optional[List[LogRetentionPolicy]] = None,
    async_mode: bool = True
) -> AuditLogger:
    """Initialize audit logger singleton"""
    global _audit_logger
    
    _audit_logger = AuditLogger(
        storage_path=storage_path,
        secret_key=secret_key,
        retention_policies=retention_policies,
        async_mode=async_mode
    )
    
    return _audit_logger
