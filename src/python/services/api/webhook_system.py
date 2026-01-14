"""
Webhook System for Kernel Ecosystem

This module provides a complete webhook infrastructure for event-driven
notifications within the Kernel ecosystem. It supports event subscription,
payload construction, delivery mechanisms, and retry logic.

Key Components:
- Event Types: Enumeration of all possible system events
- Webhook Manager: Registration and management of webhook endpoints
- Payload Builder: Constructs event payloads with proper formatting
- Delivery Service: Handles webhook delivery with retry logic
- Signature Handler: Manages request signing and verification
"""

import hashlib
import hmac
import json
import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebhookEventType(Enum):
    """
    Enumeration of all event types that can trigger webhooks.
    
    Events are organized by category: Kernel, Fine-tuning, Marketplace,
    and System events. Each event type includes a description of when
    it fires and what data it includes.
    """
    # Kernel lifecycle events
    KERNEL_CREATED = "kernel.created"
    KERNEL_UPDATED = "kernel.updated"
    KERNEL_DELETED = "kernel.deleted"
    KERNEL_DEPLOYED = "kernel.deployed"
    KERNEL_STATUS_CHANGED = "kernel.status_changed"
    
    # Kernel version events
    VERSION_CREATED = "kernel.version.created"
    VERSION_ACTIVATED = "kernel.version.activated"
    VERSION_ARCHIVED = "kernel.version.archived"
    
    # Query and execution events
    QUERY_EXECUTED = "kernel.query.executed"
    QUERY_FAILED = "kernel.query.failed"
    REASONING_COMPLETED = "kernel.reasoning.completed"
    
    # Fine-tuning events
    FINE_TUNING_STARTED = "fine_tuning.started"
    FINE_TUNING_PROGRESS = "fine_tuning.progress"
    FINE_TUNING_COMPLETED = "fine_tuning.completed"
    FINE_TUNING_FAILED = "fine_tuning.failed"
    FINE_TUNING_CANCELLED = "fine_tuning.cancelled"
    
    # Knowledge pack events
    KNOWLEDGE_PACK_CREATED = "knowledge_pack.created"
    KNOWLEDGE_PACK_EXPORTED = "knowledge_pack.exported"
    KNOWLEDGE_PACK_IMPORTED = "knowledge_pack.importmed"
    KNOWLEDGE_PACK_DELETED = "knowledge_pack.deleted"
    
    # Marketplace events
    LISTING_CREATED = "marketplace.listing.created"
    LISTING_UPDATED = "marketplace.listing.updated"
    LISTING_APPROVED = "marketplace.listing.approved"
    LISTING_REJECTED = "marketplace.listing.rejected"
    LISTING_FEATURED = "marketplace.listing.featured"
    PURCHASE_COMPLETED = "marketplace.purchase.completed"
    PURCHASE_REFUNDED = "marketplace.purchase.refunded"
    REVIEW_CREATED = "marketplace.review.created"
    
    # Context optimization events
    CONTEXT_OPTIMIZED = "context.optimized"
    CONTEXT_COMPRESSION_COMPLETED = "context.compression.completed"
    
    # System events
    SYSTEM_ERROR = "system.error"
    SYSTEM_MAINTENANCE = "system.maintenance"
    RATE_LIMIT_EXCEEDED = "system.rate_limit_exceeded"
    
    @classmethod
    def get_category(cls, event_type: 'WebhookEventType') -> str:
        """Get the category for an event type."""
        prefix = event_type.value.split('.')[0]
        categories = {
            "kernel": "Kernel Events",
            "fine_tuning": "Fine-Tuning Events",
            "knowledge_pack": "Knowledge Pack Events",
            "marketplace": "Marketplace Events",
            "context": "Context Events",
            "system": "System Events"
        }
        return categories.get(prefix, "Other Events")
    
    @classmethod
    def get_description(cls, event_type: 'WebhookEventType') -> str:
        """Get a human description for an event type."""
        descriptions = {
            cls.KERNEL_CREATED: "A new kernel has been created",
            cls.KERNEL_UPDATED: "A kernel's configuration has been updated",
            cls.KERNEL_DELETED: "A kernel has been permanently deleted",
            cls.KERNEL_DEPLOYED: "A kernel has been deployed for use",
            cls.KERNEL_STATUS_CHANGED: "A kernel's status has changed",
            cls.VERSION_CREATED: "A new version of a kernel has been created",
            cls.VERSION_ACTIVATED: "A kernel version has been activated",
            cls.VERSION_ARCHIVED: "A kernel version has been archived",
            cls.QUERY_EXECUTED: "A query has been executed on a kernel",
            cls.QUERY_FAILED: "A query execution has failed",
            cls.REASONING_COMPLETED: "Reasoning process completed",
            cls.FINE_TUNING_STARTED: "Fine-tuning job has started",
            cls.FINE_TUNING_PROGRESS: "Fine-tuning job progress update",
            cls.FINE_TUNING_COMPLETED: "Fine-tuning job completed successfully",
            cls.FINE_TUNING_FAILED: "Fine-tuning job has failed",
            cls.FINE_TUNING_CANCELLED: "Fine-tuning job was cancelled",
            cls.KNOWLEDGE_PACK_CREATED: "A new knowledge pack has been created",
            cls.KNOWLEDGE_PACK_EXPORTED: "A knowledge pack has been exported",
            cls.KNOWLEDGE_PACK_IMPORTED: "A knowledge pack has been imported",
            cls.KNOWLEDGE_PACK_DELETED: "A knowledge pack has been deleted",
            cls.LISTING_CREATED: "A new marketplace listing has been created",
            cls.LISTING_UPDATED: "A marketplace listing has been updated",
            cls.LISTING_APPROVED: "A marketplace listing has been approved",
            cls.LISTING_REJECTED: "A marketplace listing has been rejected",
            cls.LISTING_FEATURED: "A marketplace listing has been featured",
            cls.PURCHASE_COMPLETED: "A purchase has been completed",
            cls.PURCHASE_REFUNDED: "A purchase has been refunded",
            cls.REVIEW_CREATED: "A new review has been posted",
            cls.CONTEXT_OPTIMIZED: "Context optimization completed",
            cls.CONTEXT_COMPRESSION_COMPLETED: "Context compression completed",
            cls.SYSTEM_ERROR: "A system error has occurred",
            cls.SYSTEM_MAINTENANCE: "System maintenance notification",
            cls.RATE_LIMIT_EXCEEDED: "Rate limit has been exceeded"
        }
        return descriptions.get(event_type, "Unknown event")


@dataclass
class WebhookDeliveryAttempt:
    """Represents a single delivery attempt for a webhook."""
    attempt_number: int
    timestamp: datetime
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    duration_ms: float = 0.0
    success: bool = False
    error_message: Optional[str] = None


@dataclass
class WebhookDelivery:
    """Represents the complete delivery record for a webhook event."""
    delivery_id: str
    webhook_id: str
    event_type: WebhookEventType
    payload: Dict[str, Any]
    status: str = "pending"  # pending, delivering, completed, failed, exhausted
    attempts: List[WebhookDeliveryAttempt] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    next_retry_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    event_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize delivery record to dictionary."""
        return {
            "delivery_id": self.delivery_id,
            "webhook_id": self.webhook_id,
            "event_type": self.event_type.value,
            "payload": self.payload,
            "status": self.status,
            "attempts": [
                {
                    "attempt_number": a.attempt_number,
                    "timestamp": a.timestamp.isoformat(),
                    "status_code": a.status_code,
                    "response_body": a.response_body,
                    "duration_ms": a.duration_ms,
                    "success": a.success,
                    "error_message": a.error_message
                }
                for a in self.attempts
            ],
            "created_at": self.created_at.isoformat(),
            "next_retry_at": self.next_retry_at.isoformat() if self.next_retry_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "event_id": self.event_id
        }


@dataclass
class Webhook:
    """
    Represents a registered webhook endpoint.
    
    A webhook is registered with a URL, subscribed to specific event types,
    and optionally secured with a signing secret for verification.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    url: str = ""
    events: Set[WebhookEventType] = field(default_factory=set)
    secret: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True
    failure_count: int = 0
    last_delivery_at: Optional[datetime] = None
    last_status_code: Optional[int] = None
    description: str = ""
    owner_id: Optional[str] = None
    
    def __post_init__(self):
        """Validate and normalize webhook configuration."""
        if self.url:
            parsed = urlparse(self.url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid webhook URL: {self.url}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize webhook to dictionary."""
        return {
            "id": self.id,
            "url": self.url,
            "events": [e.value for e in self.events],
            "secret_provided": self.secret is not None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "is_active": self.is_active,
            "failure_count": self.failure_count,
            "last_delivery_at": self.last_delivery_at.isoformat() if self.last_delivery_at else None,
            "last_status_code": self.last_status_code,
            "description": self.description,
            "owner_id": self.owner_id
        }


@dataclass
class WebhookEvent:
    """
    Represents an event to be delivered via webhooks.
    
    Contains all information needed to construct payloads and deliver
    to subscribed endpoints.
    """
    event_type: WebhookEventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""  # Entity that triggered the event
    source_id: str = ""  # ID of the source entity
    data: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None  # Associated user, if any
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_payload(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """
        Construct the webhook payload for this event.
        
        Args:
            include_sensitive: Whether to include sensitive data
            
        Returns:
            Dictionary payload for webhook delivery
        """
        payload = {
            "id": self.event_id,
            "type": self.event_type.value,
            "created": self.timestamp.isoformat(),
            "source": {
                "type": self.source,
                "id": self.source_id
            },
            "data": self.data.copy()
        }
        
        # Add metadata if present
        if self.metadata:
            payload["metadata"] = self.metadata.copy()
        
        # Add user context if available
        if self.user_id:
            payload["user"] = {"id": self.user_id}
        
        # Remove sensitive data if not requested
        if not include_sensitive:
            sensitive_keys = ["password", "secret", "token", "api_key", "credit_card"]
            for key in list(payload["data"].keys()):
                for sensitive in sensitive_keys:
                    if sensitive in key.lower():
                        payload["data"].pop(key, None)
                        break
        
        return payload
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize event to dictionary."""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "source_id": self.source_id,
            "data": self.data,
            "user_id": self.user_id,
            "metadata": self.metadata
        }


class SignatureHandler:
    """
    Handles webhook request signing and verification.
    
    Uses HMAC-SHA256 for creating and verifying request signatures,
    ensuring that webhook payloads are authentic and haven't been
    tampered with.
    """
    
    @staticmethod
    def generate_signature(payload: str, secret: str) -> str:
        """
        Generate HMAC-SHA256 signature for a payload.
        
        Args:
            payload: JSON payload string
            secret: Webhook signing secret
            
        Returns:
            Hex-encoded signature
        """
        if not secret:
            return ""
        signature = hmac.new(
            secret.encode('utf-8'),
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"
    
    @staticmethod
    def verify_signature(payload: str, signature: str, secret: str) -> bool:
        """
        Verify a webhook request signature.
        
        Args:
            payload: Raw request body
            signature: Provided signature header
            secret: Webhook signing secret
            
        Returns:
            True if signature is valid
        """
        if not secret or not signature:
            return False
        
        # Extract signature from header format
        if signature.startswith("sha256="):
            expected_sig = signature[7:]
        else:
            expected_sig = signature
        
        # Calculate actual signature
        actual_sig = SignatureHandler.generate_signature(payload, secret)
        
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected_sig, actual_sig[7:])
    
    @staticmethod
    def generate_secret(length: int = 32) -> str:
        """
        Generate a secure random secret.
        
        Args:
            length: Length of the secret in bytes
            
        Returns:
            Hex-encoded secret
        """
        return uuid.uuid4().hex + uuid.uuid4().hex


class PayloadBuilder:
    """
    Constructs webhook payloads with proper formatting and metadata.
    
    Handles payload serialization, custom headers, and ensures
    consistent payload structure across different event types.
    """
    
    # Common headers for all webhook requests
    COMMON_HEADERS = {
        "Content-Type": "application/json",
        "X-Webhook-Version": "1.0"
    }
    
    @classmethod
    def build_headers(
        cls,
        event: WebhookEvent,
        signature: str,
        delivery_id: str
    ) -> Dict[str, str]:
        """
        Build headers for a webhook request.
        
        Args:
            event: The triggering event
            signature: Request signature
            delivery_id: Unique delivery identifier
            
        Returns:
            Dictionary of HTTP headers
        """
        headers = cls.COMMON_HEADERS.copy()
        headers.update({
            "X-Webhook-ID": delivery_id,
            "X-Webhook-Event": event.event_type.value,
            "X-Webhook-Timestamp": event.timestamp.isoformat(),
            "X-Webhook-Signature": signature,
            "X-Webhook-Delivery": delivery_id
        })
        return headers
    
    @classmethod
    def build_payload(
        cls,
        event: WebhookEvent,
        delivery_id: str,
        include_signature: bool = False,
        secret: str = None
    ) -> tuple:
        """
        Build complete payload for webhook delivery.
        
        Args:
            event: The triggering event
            delivery_id: Unique delivery identifier
            include_signature: Whether to include signature in payload
            secret: Webhook secret for signature
            
        Returns:
            Tuple of (payload_dict, payload_string, signature)
        """
        payload_dict = event.to_payload()
        
        # Add delivery metadata
        payload_dict["delivery"] = {
            "id": delivery_id,
            "timestamp": datetime.now().isoformat()
        }
        
        # Serialize to JSON
        payload_string = json.dumps(payload_dict, default=str)
        
        # Generate signature
        signature = ""
        if secret:
            signature = SignatureHandler.generate_signature(payload_string, secret)
        
        return payload_dict, payload_string, signature
    
    @classmethod
    def format_error_payload(
        cls,
        delivery_id: str,
        error_message: str,
        event: WebhookEvent = None
    ) -> Dict[str, Any]:
        """Format an error payload for failed deliveries."""
        return {
            "error": {
                "delivery_id": delivery_id,
                "message": error_message,
                "timestamp": datetime.now().isoformat(),
                "event_type": event.event_type.value if event else None
            }
        }


class WebhookDeliveryService:
    """
    Handles the actual delivery of webhook events.
    
    Manages HTTP requests to webhook endpoints, handles retries,
    and records delivery status. Supports both synchronous and
    asynchronous delivery modes.
    """
    
    # Retry configuration
    MAX_RETRIES = 5
    RETRY_INTERVALS = [1, 5, 30, 300, 3600]  # Seconds between retries
    
    # HTTP timeout
    REQUEST_TIMEOUT = 30
    
    # Success status codes
    SUCCESS_CODES = {200, 201, 202, 204}
    
    def __init__(
        self,
        session: requests.Session = None,
        max_concurrent: int = 10
    ):
        """
        Initialize the delivery service.
        
        Args:
            session: Optional requests session for connection pooling
            max_concurrent: Maximum concurrent deliveries
        """
        self.session = session or requests.Session()
        self._delivery_queue: queue.Queue = queue.Queue()
        self._delivery_thread: Optional[threading.Thread] = None
        self._running = False
        self._max_concurrent = max_concurrent
        self._semaphore = threading.Semaphore(max_concurrent)
        self._delivery_history: List[WebhookDelivery] = []
        self._lock = threading.Lock()
        
        # Statistics
        self._stats = {
            "total_deliveries": 0,
            "successful_deliveries": 0,
            "failed_deliveries": 0,
            "total_attempts": 0
        }
    
    def start(self):
        """Start the delivery worker thread."""
        if self._running:
            return
        
        self._running = True
        self._delivery_thread = threading.Thread(target=self._process_deliveries, daemon=True)
        self._delivery_thread.start()
        logger.info("Webhook delivery service started")
    
    def stop(self):
        """Stop the delivery service gracefully."""
        self._running = False
        if self._delivery_thread:
            self._delivery_thread.join(timeout=5)
        logger.info("Webhook delivery service stopped")
    
    def deliver(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        delivery_id: str,
        blocking: bool = False
    ) -> WebhookDelivery:
        """
        Queue a webhook delivery.
        
        Args:
            webhook: Target webhook endpoint
            event: Event to deliver
            delivery_id: Unique delivery identifier
            blocking: Whether to wait for delivery to complete
            
        Returns:
            WebhookDelivery record
        """
        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            webhook_id=webhook.id,
            event_type=event.event_type,
            payload=event.to_payload(),
            event_id=event.event_id
        )
        
        if blocking:
            self._deliver_sync(webhook, event, delivery)
        else:
            self._delivery_queue.put((webhook, event, delivery))
        
        return delivery
    
    def deliver_now(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        delivery_id: str
    ) -> WebhookDelivery:
        """
        Immediately deliver a webhook (synchronous).
        
        Args:
            webhook: Target webhook endpoint
            event: Event to deliver
            delivery_id: Unique delivery identifier
            
        Returns:
            WebhookDelivery record with attempt details
        """
        delivery = WebhookDelivery(
            delivery_id=delivery_id,
            webhook_id=webhook.id,
            event_type=event.event_type,
            payload=event.to_payload(),
            event_id=event.event_id
        )
        self._deliver_sync(webhook, event, delivery)
        return delivery
    
    def _process_deliveries(self):
        """Background worker for processing delivery queue."""
        while self._running:
            try:
                webhook, event, delivery = self._delivery_queue.get(timeout=1)
                with self._semaphore:
                    self._deliver_sync(webhook, event, delivery)
                self._delivery_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing delivery: {e}")
    
    def _deliver_sync(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        delivery: WebhookDelivery
    ):
        """
        Synchronously execute webhook delivery with retries.
        
        Args:
            webhook: Target webhook endpoint
            event: Event to deliver
            delivery: Delivery record to update
        """
        start_time = time.time()
        
        # Build payload
        payload_dict, payload_string, signature = PayloadBuilder.build_payload(
            event, delivery.delivery_id, secret=webhook.secret
        )
        headers = PayloadBuilder.build_headers(event, signature, delivery.delivery_id)
        
        # Attempt delivery with retries
        for attempt in range(self.MAX_RETRIES):
            attempt_record = WebhookDeliveryAttempt(
                attempt_number=attempt + 1,
                timestamp=datetime.now()
            )
            
            try:
                response = self.session.post(
                    webhook.url,
                    data=payload_string,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT
                )
                
                attempt_record.status_code = response.status_code
                attempt_record.response_body = response.text[:1000]  # Limit response size
                attempt_record.duration_ms = (time.time() - start_time) * 1000
                
                if response.status_code in self.SUCCESS_CODES:
                    attempt_record.success = True
                    delivery.status = "completed"
                    delivery.completed_at = datetime.now()
                    self._record_success(webhook, response.status_code)
                    break
                else:
                    attempt_record.success = False
                    attempt_record.error_message = f"HTTP {response.status_code}"
                    self._record_failure(webhook, response.status_code)
                    
            except requests.exceptions.Timeout:
                attempt_record.error_message = "Request timeout"
                attempt_record.duration_ms = self.REQUEST_TIMEOUT * 1000
                self._record_failure(webhook, None)
                
            except requests.exceptions.RequestException as e:
                attempt_record.error_message = str(e)
                self._record_failure(webhook, None)
            
            delivery.attempts.append(attempt_record)
            self._stats["total_attempts"] += 1
            
            # Calculate next retry time
            if attempt < self.MAX_RETRIES - 1:
                wait_time = self.RETRY_INTERVALS[attempt]
                delivery.next_retry_at = datetime.now() + timedelta(seconds=wait_time)
                time.sleep(wait_time)
        else:
            # All retries exhausted
            delivery.status = "exhausted"
            delivery.completed_at = datetime.now()
        
        # Record delivery
        with self._lock:
            self._delivery_history.append(delivery)
            self._stats["total_deliveries"] += 1
            if delivery.status == "completed":
                self._stats["successful_deliveries"] += 1
            else:
                self._stats["failed_deliveries"] += 1
    
    def _record_success(self, webhook: Webhook, status_code: int):
        """Record a successful delivery."""
        webhook.failure_count = 0
        webhook.last_delivery_at = datetime.now()
        webhook.last_status_code = status_code
    
    def _record_failure(self, webhook: Webhook, status_code: Optional[int]):
        """Record a failed delivery attempt."""
        webhook.failure_count += 1
        webhook.last_delivery_at = datetime.now()
        webhook.last_status_code = status_code
        
        # Deactivate webhook after too many failures
        if webhook.failure_count >= 10:
            webhook.is_active = False
            logger.warning(f"Webhook {webhook.id} deactivated due to repeated failures")
    
    def get_stats(self) -> Dict[str, int]:
        """Get delivery statistics."""
        return self._stats.copy()
    
    def get_recent_deliveries(self, limit: int = 100) -> List[WebhookDelivery]:
        """Get recent delivery records."""
        with self._lock:
            return self._delivery_history[-limit:]


class WebhookManager:
    """
    Manages webhook registrations and event subscriptions.
    
    Provides high-level operations for creating, updating, and
    deleting webhooks, as well as dispatching events to all
    subscribed endpoints.
    """
    
    def __init__(
        self,
        storage_backend: Dict[str, Webhook] = None,
        delivery_service: WebhookDeliveryService = None
    ):
        """
        Initialize the webhook manager.
        
        Args:
            storage_backend: Dict-like storage for webhook persistence
            delivery_service: Service for webhook delivery
        """
        self._storage = storage_backend or {}
        self._delivery_service = delivery_service or WebhookDeliveryService()
        self._event_subscribers: Dict[WebhookEventType, Set[str]] = {
            event: set() for event in WebhookEventType
        }
        self._lock = threading.Lock()
        
        # Build subscriber index
        for webhook in self._storage.values():
            for event in webhook.events:
                self._event_subscribers[event].add(webhook.id)
        
        # Start delivery service
        self._delivery_service.start()
    
    def register_webhook(
        self,
        url: str,
        events: List[WebhookEventType],
        secret: str = None,
        description: str = "",
        owner_id: str = None
    ) -> Webhook:
        """
        Register a new webhook endpoint.
        
        Args:
            url: Target URL for webhook delivery
            events: List of event types to subscribe to
            secret: Optional signing secret
            description: Human-readable description
            owner_id: ID of the webhook owner
            
        Returns:
            Created Webhook object
            
        Raises:
            ValueError: If URL is invalid or webhook already exists
        """
        # Generate secret if not provided
        if not secret:
            secret = SignatureHandler.generate_secret()
        
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(f"Invalid webhook URL: {url}")
        
        webhook = Webhook(
            url=url,
            events=set(events),
            secret=secret,
            description=description,
            owner_id=owner_id
        )
        
        # Store webhook
        with self._lock:
            self._storage[webhook.id] = webhook
            
            # Update subscriber index
            for event in events:
                self._event_subscribers[event].add(webhook.id)
        
        logger.info(f"Registered webhook {webhook.id} for {len(events)} event types")
        return webhook
    
    def update_webhook(
        self,
        webhook_id: str,
        url: str = None,
        events: List[WebhookEventType] = None,
        description: str = None,
        is_active: bool = None
    ) -> Webhook:
        """
        Update an existing webhook.
        
        Args:
            webhook_id: ID of webhook to update
            url: New URL (if changing)
            events: New event subscriptions (if changing)
            description: New description (if changing)
            is_active: New active status (if changing)
            
        Returns:
            Updated Webhook object
            
        Raises:
            KeyError: If webhook not found
        """
        with self._lock:
            if webhook_id not in self._storage:
                raise KeyError(f"Webhook not found: {webhook_id}")
            
            webhook = self._storage[webhook_id]
            
            # Update URL
            if url:
                parsed = urlparse(url)
                if not parsed.scheme or not parsed.netloc:
                    raise ValueError(f"Invalid webhook URL: {url}")
                webhook.url = url
            
            # Update events
            if events:
                # Remove from old event subscriptions
                for event in webhook.events:
                    self._event_subscribers[event].discard(webhook_id)
                # Add to new event subscriptions
                webhook.events = set(events)
                for event in events:
                    self._event_subscribers[event].add(webhook_id)
            
            # Update other fields
            if description is not None:
                webhook.description = description
            if is_active is not None:
                webhook.is_active = is_active
            
            webhook.updated_at = datetime.now()
        
        logger.info(f"Updated webhook {webhook_id}")
        return webhook
    
    def delete_webhook(self, webhook_id: str) -> bool:
        """
        Delete a webhook registration.
        
        Args:
            webhook_id: ID of webhook to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if webhook_id not in self._storage:
                return False
            
            webhook = self._storage.pop(webhook_id)
            
            # Remove from subscriber index
            for event in webhook.events:
                self._event_subscribers[event].discard(webhook_id)
        
        logger.info(f"Deleted webhook {webhook_id}")
        return True
    
    def get_webhook(self, webhook_id: str) -> Optional[Webhook]:
        """Get a webhook by ID."""
        return self._storage.get(webhook_id)
    
    def list_webhooks(
        self,
        owner_id: str = None,
        is_active: bool = None,
        event_type: WebhookEventType = None
    ) -> List[Webhook]:
        """
        List webhooks with optional filtering.
        
        Args:
            owner_id: Filter by owner
            is_active: Filter by active status
            event_type: Filter by subscribed event type
            
        Returns:
            List of matching webhooks
        """
        result = list(self._storage.values())
        
        if owner_id:
            result = [w for w in result if w.owner_id == owner_id]
        
        if is_active is not None:
            result = [w for w in result if w.is_active == is_active]
        
        if event_type:
            webhook_ids = self._event_subscribers.get(event_type, set())
            result = [w for w in result if w.id in webhook_ids]
        
        return result
    
    def get_subscribers(self, event_type: WebhookEventType) -> List[Webhook]:
        """
        Get all webhooks subscribed to an event type.
        
        Args:
            event_type: Event type to query
            
        Returns:
            List of subscribed webhooks
        """
        with self._lock:
            webhook_ids = self._event_subscribers.get(event_type, set())
            return [
                self._storage[wid]
                for wid in webhook_ids
                if wid in self._storage and self._storage[wid].is_active
            ]
    
    def dispatch_event(
        self,
        event: WebhookEvent,
        blocking: bool = False
    ) -> List[WebhookDelivery]:
        """
        Dispatch an event to all subscribed webhooks.
        
        Args:
            event: Event to dispatch
            blocking: Whether to wait for all deliveries to complete
            
        Returns:
            List of delivery records
        """
        subscribers = self.get_subscribers(event.event_type)
        deliveries = []
        
        for webhook in subscribers:
            delivery_id = str(uuid.uuid4())
            delivery = self._delivery_service.deliver(
                webhook, event, delivery_id, blocking=blocking
            )
            deliveries.append(delivery)
        
        logger.info(
            f"Dispatched {event.event_type.value} to {len(deliveries)} subscribers"
        )
        return deliveries
    
    def dispatch_event_sync(
        self,
        event: WebhookEvent
    ) -> List[WebhookDelivery]:
        """
        Dispatch an event synchronously to all subscribers.
        
        Args:
            event: Event to dispatch
            
        Returns:
            List of delivery records
        """
        subscribers = self.get_subscribers(event.event_type)
        deliveries = []
        
        for webhook in subscribers:
            delivery_id = str(uuid.uuid4())
            delivery = self._delivery_service.deliver_now(
                webhook, event, delivery_id
            )
            deliveries.append(delivery)
        
        return deliveries
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get webhook system statistics."""
        webhooks = list(self._storage.values())
        active_webhooks = [w for w in webhooks if w.is_active]
        
        return {
            "total_webhooks": len(webhooks),
            "active_webhooks": len(active_webhooks),
            "inactive_webhooks": len(webhooks) - len(active_webhooks),
            "total_subscriptions": sum(len(w.events) for w in webhooks),
            "delivery_stats": self._delivery_service.get_stats(),
            "event_counts": {
                event.value: len(subscribers)
                for event, subscribers in self._event_subscribers.items()
                if subscribers
            }
        }


class WebhookEventBuilder:
    """
    Utility class for constructing webhook events.
    
    Provides convenient methods for creating events of various
    types with properly structured data.
    """
    
    @classmethod
    def kernel_event(
        cls,
        event_type: WebhookEventType,
        kernel_id: str,
        kernel_data: Dict[str, Any],
        user_id: str = None
    ) -> WebhookEvent:
        """Create a kernel lifecycle event."""
        return WebhookEvent(
            event_type=event_type,
            source="kernel",
            source_id=kernel_id,
            data={
                "kernel": kernel_data,
                "changes": kernel_data.get("changes", {})
            },
            user_id=user_id
        )
    
    @classmethod
    def fine_tuning_event(
        cls,
        event_type: WebhookEventType,
        job_id: str,
        job_data: Dict[str, Any],
        user_id: str = None
    ) -> WebhookEvent:
        """Create a fine-tuning job event."""
        return WebhookEvent(
            event_type=event_type,
            source="fine_tuning",
            source_id=job_id,
            data={
                "job": job_data,
                "progress": job_data.get("progress", 0),
                "metrics": job_data.get("metrics", {})
            },
            user_id=user_id
        )
    
    @classmethod
    def marketplace_event(
        cls,
        event_type: WebhookEventType,
        listing_id: str,
        listing_data: Dict[str, Any],
        user_id: str = None
    ) -> WebhookEvent:
        """Create a marketplace event."""
        return WebhookEvent(
            event_type=event_type,
            source="marketplace",
            source_id=listing_id,
            data={
                "listing": listing_data,
                "purchase": listing_data.get("purchase", {}),
                "review": listing_data.get("review", {})
            },
            user_id=user_id
        )
    
    @classmethod
    def system_event(
        cls,
        event_type: WebhookEventType,
        message: str,
        details: Dict[str, Any] = None
    ) -> WebhookEvent:
        """Create a system event."""
        return WebhookEvent(
            event_type=event_type,
            source="system",
            source_id="system",
            data={
                "message": message,
                "details": details or {}
            }
        )


# Event dispatcher singleton for easy integration
_event_dispatcher: Optional[WebhookManager] = None


def get_event_dispatcher() -> WebhookManager:
    """Get or create the global event dispatcher."""
    global _event_dispatcher
    if _event_dispatcher is None:
        _event_dispatcher = WebhookManager()
    return _event_dispatcher


def dispatch_kernel_event(
    event_type: WebhookEventType,
    kernel_id: str,
    kernel_data: Dict[str, Any],
    user_id: str = None
):
    """Convenience function to dispatch a kernel event."""
    event = WebhookEventBuilder.kernel_event(
        event_type, kernel_id, kernel_data, user_id
    )
    get_event_dispatcher().dispatch_event(event)


def dispatch_fine_tuning_event(
    event_type: WebhookEventType,
    job_id: str,
    job_data: Dict[str, Any],
    user_id: str = None
):
    """Convenience function to dispatch a fine-tuning event."""
    event = WebhookEventBuilder.fine_tuning_event(
        event_type, job_id, job_data, user_id
    )
    get_event_dispatcher().dispatch_event(event)


def dispatch_marketplace_event(
    event_type: WebhookEventType,
    listing_id: str,
    listing_data: Dict[str, Any],
    user_id: str = None
):
    """Convenience function to dispatch a marketplace event."""
    event = WebhookEventBuilder.marketplace_event(
        event_type, listing_id, listing_data, user_id
    )
    get_event_dispatcher().dispatch_event(event)


# Example usage and testing
if __name__ == "__main__":
    # Example webhook registration
    manager = WebhookManager()
    
    # Register a webhook
    webhook = manager.register_webhook(
        url="https://example.com/webhook",
        events=[
            WebhookEventType.KERNEL_CREATED,
            WebhookEventType.FINE_TUNING_COMPLETED,
            WebhookEventType.PURCHASE_COMPLETED
        ],
        description="My application webhook",
        owner_id="user_123"
    )
    
    print(f"Created webhook: {webhook.id}")
    print(f"Secret: {webhook.secret}")  # Show secret for configuration
    
    # Dispatch an event
    event = WebhookEventBuilder.kernel_event(
        WebhookEventType.KERNEL_CREATED,
        kernel_id="kernel_456",
        kernel_data={
            "name": "Test Kernel",
            "type": "reasoning",
            "version": "1.0.0"
        },
        user_id="user_123"
    )
    
    deliveries = manager.dispatch_event_sync(event)
    print(f"Dispatched to {len(deliveries)} subscribers")
    
    # Get statistics
    stats = manager.get_statistics()
    print(f"\nWebhook Statistics:")
    print(f"  Total webhooks: {stats['total_webhooks']}")
    print(f"  Active webhooks: {stats['active_webhooks']}")
    print(f"  Delivery stats: {stats['delivery_stats']}")
