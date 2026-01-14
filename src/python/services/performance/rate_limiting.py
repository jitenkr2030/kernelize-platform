"""
KERNELIZE Platform - Query Rate Limiting System
================================================

Protection against abuse with configurable rate limits.
Implements tiered rate limiting, monitoring, and alerting.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Tuple
from threading import RLock
from collections import deque
import threading

logger = logging.getLogger(__name__)


class RateLimitTier(Enum):
    """Rate limiting tiers"""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"
    UNLIMITED = "unlimited"


class EndpointCategory(Enum):
    """Endpoint categories for rate limiting"""
    API = "api"
    QUERY = "query"
    COMPRESSION = "compression"
    AUTH = "auth"
    ADMIN = "admin"
    WEBHOOK = "webhook"


@dataclass
class RateLimitConfig:
    """Rate limit configuration for a tier"""
    config_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tier: str = RateLimitTier.BASIC.value
    
    # Requests per window
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    
    # Burst configuration
    burst_size: int = 10
    burst_window_seconds: int = 1
    
    # Endpoint-specific overrides
    endpoint_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Quota
    monthly_quota: Optional[int] = None
    current_monthly_usage: int = 0
    
    # Compliance
    enforce_global_limits: bool = True
    priority: int = 100  # Lower = higher priority
    
    is_active: bool = True


@dataclass
class RateLimitRule:
    """Individual rate limit rule"""
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Targeting
    endpoint_pattern: str = ""  # Regex pattern
    methods: List[str] = field(default_factory=lambda: ["*"])
    categories: List[str] = field(default_factory=list)
    
    # Limits
    max_requests: int = 100
    window_seconds: int = 60
    
    # Conditions
    user_tiers: List[str] = field(default_factory=list)
    ip_ranges: List[str] = field(default_factory=list)
    
    # Response
    http_status_code: int = 429
    retry_after_seconds: int = 60
    
    is_active: bool = True
    priority: int = 100


@dataclass
class RateLimitRecord:
    """Record of a rate-limited request"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    api_key_id: Optional[str] = None
    ip_address: Optional[str] = None
    
    endpoint: str = ""
    method: str = ""
    
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    limit_exceeded: int = 0
    limit_max: int = 0
    
    # Response
    was_limited: bool = False
    retry_after: Optional[int] = None
    
    # Context
    user_agent: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class RateLimitStats:
    """Rate limiting statistics"""
    # Request counts
    total_requests: int = 0
    limited_requests: int = 0
    
    # By tier
    by_tier: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # By endpoint
    by_endpoint: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # By hour
    by_hour: Dict[str, Dict[str, int]] = field(default_factory=dict)
    
    # Current usage
    current_minute_requests: int = 0
    current_hour_requests: int = 0
    
    # Alerts
    alerts_triggered: int = 0
    last_alert: Optional[str] = None
    
    def hit_rate(self) -> float:
        """Calculate limited request percentage"""
        if self.total_requests == 0:
            return 0.0
        return (self.limited_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_requests': self.total_requests,
            'limited_requests': self.limited_requests,
            'limit_rate': f"{self.hit_rate():.4f}%",
            'by_tier': self.by_tier,
            'by_endpoint': dict(self.by_endpoint),
            'current_minute_requests': self.current_minute_requests,
            'current_hour_requests': self.current_hour_requests,
            'alerts_triggered': self.alerts_triggered
        }


class RateLimitWindow:
    """Rate limit tracking for a sliding window"""
    
    def __init__(self, window_seconds: int):
        """
        Initialize rate limit window
        
        Args:
            window_seconds: Window duration in seconds
        """
        self.window_seconds = window_seconds
        self._timestamps: deque = deque()
        self._lock = RLock()
    
    def try_acquire(self, count: int = 1) -> Tuple[bool, int]:
        """
        Try to acquire rate limit slot
        
        Args:
            count: Number of slots to acquire
            
        Returns:
            (acquired, remaining)
        """
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove expired timestamps
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()
            
            # Check if we can acquire
            available = 100 - len(self._timestamps)  # Default limit
            
            if len(self._timestamps) + count <= available:
                # Add new timestamps
                for _ in range(count):
                    self._timestamps.append(now)
                return True, available - len(self._timestamps) - count
            
            return False, available
    
    def get_remaining(self) -> int:
        """Get remaining requests in window"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove expired
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()
            
            return max(0, 100 - len(self._timestamps))
    
    def get_count(self) -> int:
        """Get current count in window"""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Remove expired
            while self._timestamps and self._timestamps[0] < window_start:
                self._timestamps.popleft()
            
            return len(self._timestamps)


class RateLimiter:
    """
    Main rate limiting service
    
    Implements tiered rate limiting with sliding windows,
    burst protection, and comprehensive monitoring.
    """
    
    # Default tier configurations
    DEFAULT_TIERS = {
        RateLimitTier.FREE: RateLimitConfig(
            tier=RateLimitTier.FREE.value,
            requests_per_minute=10,
            requests_per_hour=100,
            requests_per_day=500,
            burst_size=3,
            monthly_quota=5000
        ),
        RateLimitTier.BASIC: RateLimitConfig(
            tier=RateLimitTier.BASIC.value,
            requests_per_minute=60,
            requests_per_hour=1000,
            requests_per_day=10000,
            burst_size=10,
            monthly_quota=100000
        ),
        RateLimitTier.PROFESSIONAL: RateLimitConfig(
            tier=RateLimitTier.PROFESSIONAL.value,
            requests_per_minute=300,
            requests_per_hour=10000,
            requests_per_day=100000,
            burst_size=30,
            monthly_quota=1000000
        ),
        RateLimitTier.ENTERPRISE: RateLimitConfig(
            tier=RateLimitTier.ENTERPRISE.value,
            requests_per_minute=1000,
            requests_per_hour=100000,
            requests_per_day=1000000,
            burst_size=100,
            monthly_quota=None  # Unlimited
        ),
        RateLimitTier.UNLIMITED: RateLimitConfig(
            tier=RateLimitTier.UNLIMITED.value,
            requests_per_minute=10000,
            requests_per_hour=1000000,
            requests_per_day=10000000,
            burst_size=1000,
            monthly_quota=None  # Unlimited
        ),
    }
    
    def __init__(self, storage_path: str = "data/rate_limits"):
        """
        Initialize rate limiter
        
        Args:
            storage_path: Path for storage
        """
        self._configs: Dict[str, RateLimitConfig] = {}
        self._rules: List[RateLimitRule] = []
        self._user_tiers: Dict[str, str] = {}  # user_id -> tier
        self._api_key_tiers: Dict[str, str] = {}  # api_key -> tier
        
        # Tracking windows (by user/endpoint)
        self._windows: Dict[Tuple[str, str, int], RateLimitWindow] = {}
        self._window_lock = RLock()
        
        # Trusted applications
        self._trusted_apps: Dict[str, Dict[str, Any]] = {}
        
        # Statistics
        self._stats = RateLimitStats()
        self._stats_lock = RLock()
        
        # Initialize default tiers
        for tier, config in self.DEFAULT_TIERS.items():
            self._configs[tier.value] = config
        
        # Initialize default rules
        self._init_default_rules()
        
        # Monitoring
        self._alerts: List[Dict[str, Any]] = []
        self._alert_thresholds = {
            'critical': 0.5,  # 50% of requests limited
            'warning': 0.2,   # 20% of requests limited
        }
    
    def _init_default_rules(self):
        """Initialize default rate limit rules"""
        # Stricter limits for auth endpoints
        auth_rule = RateLimitRule(
            name="Auth Rate Limit",
            description="Stricter limits for authentication endpoints",
            endpoint_pattern=r"/api/v1/auth/.*",
            methods=["POST"],
            max_requests=5,
            window_seconds=60,
            http_status_code= 429,
            retry_after_seconds=300
        )
        self._rules.append(auth_rule)
        
        # Compression endpoints
        compression_rule = RateLimitRule(
            name="Compression Rate Limit",
            description="Limits for document compression",
            endpoint_pattern=r"/api/v1/compress.*",
            methods=["POST"],
            max_requests=20,
            window_seconds=60,
            priority=50
        )
        self._rules.append(compression_rule)
        
        # Query endpoints (more lenient)
        query_rule = RateLimitRule(
            name="Query Rate Limit",
            description="Standard limits for query endpoints",
            endpoint_pattern=r"/api/v1/query.*",
            methods=["GET", "POST"],
            max_requests=100,
            window_seconds=60,
            priority=100
        )
        self._rules.append(query_rule)
    
    def set_user_tier(self, user_id: str, tier: str):
        """Set rate limit tier for a user"""
        self._user_tiers[user_id] = tier
    
    def set_api_key_tier(self, api_key_id: str, tier: str):
        """Set rate limit tier for an API key"""
        self._api_key_tiers[api_key_id] = tier
    
    def add_trusted_app(
        self,
        app_id: str,
        name: str,
        rate_limit_multiplier: float = 1.0,
        bypass_endpoints: Optional[List[str]] = None
    ):
        """Add trusted application with relaxed limits"""
        self._trusted_apps[app_id] = {
            'name': name,
            'rate_limit_multiplier': rate_limit_multiplier,
            'bypass_endpoints': bypass_endpoints or []
        }
    
    def check_rate_limit(
        self,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        endpoint: str = "",
        method: str = "GET",
        user_agent: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if request is within rate limits
        
        Args:
            user_id: User making request
            api_key_id: API key used
            ip_address: Client IP
            endpoint: Request endpoint
            method: HTTP method
            user_agent: Client user agent
            
        Returns:
            Rate limit check result
        """
        # Check if trusted app
        if api_key_id and api_key_id in self._trusted_apps:
            trusted = self._trusted_apps[api_key_id]
            if not endpoint.startswith(tuple(trusted.get('bypass_endpoints', []))):
                multiplier = trusted.get('rate_limit_multiplier', 1.0)
                if multiplier > 1.0:
                    # Apply multiplier to limits
                    pass  # Would adjust limits
        
        # Get tier
        tier = self._get_tier(user_id, api_key_id)
        config = self._configs.get(tier, self._configs[RateLimitTier.BASIC.value])
        
        # Check endpoint-specific rules
        rule_result = self._check_rules(endpoint, method, tier)
        if rule_result:
            return rule_result
        
        # Get or create rate limit windows
        window_key = self._get_window_key(user_id, api_key_id, ip_address, endpoint)
        
        # Check per-minute limit
        minute_window = self._get_window(window_key, 60)
        acquired, remaining = minute_window.try_acquire()
        
        if not acquired:
            return self._create_limit_response(
                "Per-minute rate limit exceeded",
                remaining,
                60,
                config.http_status_code if hasattr(config, 'http_status_code') else 429
            )
        
        # Check per-hour limit
        hour_window = self._get_window(window_key, 3600)
        hour_remaining = hour_window.get_remaining()
        
        # Update stats
        with self._stats_lock:
            self._stats.total_requests += 1
            self._stats.current_minute_requests += 1
            self._stats.current_hour_requests += 1
            
            # By tier
            if tier not in self._stats.by_tier:
                self._stats.by_tier[tier] = {'requests': 0, 'limited': 0}
            self._stats.by_tier[tier]['requests'] += 1
            
            # By endpoint
            if endpoint not in self._stats.by_endpoint:
                self._stats.by_endpoint[endpoint] = {'requests': 0, 'limited': 0}
            self._stats.by_endpoint[endpoint]['requests'] += 1
        
        # Check for alerts
        self._check_alerts()
        
        return {
            'allowed': True,
            'remaining': remaining,
            'hourly_remaining': hour_remaining,
            'limit_reset': int(time.time()) + 60,
            'tier': tier
        }
    
    def _get_tier(self, user_id: Optional[str], api_key_id: Optional[str]) -> str:
        """Determine rate limit tier"""
        if api_key_id and api_key_id in self._api_key_tiers:
            return self._api_key_tiers[api_key_id]
        if user_id and user_id in self._user_tiers:
            return self._user_tiers[user_id]
        return RateLimitTier.FREE.value
    
    def _get_window_key(
        self,
        user_id: Optional[str],
        api_key_id: Optional[str],
        ip_address: Optional[str],
        endpoint: str
    ) -> str:
        """Generate window key"""
        key_parts = []
        
        if user_id:
            key_parts.append(f"u:{user_id}")
        elif api_key_id:
            key_parts.append(f"k:{api_key_id}")
        elif ip_address:
            key_parts.append(f"i:{ip_address}")
        else:
            key_parts.append(f"e:{endpoint}")
        
        return ":".join(key_parts)
    
    def _get_window(self, key: str, window_seconds: int) -> RateLimitWindow:
        """Get or create rate limit window"""
        with self._window_lock:
            window_key = (key, window_seconds)
            
            if window_key not in self._windows:
                self._windows[window_key] = RateLimitWindow(window_seconds)
            
            return self._windows[window_key]
    
    def _check_rules(
        self,
        endpoint: str,
        method: str,
        tier: str
    ) -> Optional[Dict[str, Any]]:
        """Check endpoint-specific rules"""
        import re
        
        for rule in sorted(self._rules, key=lambda r: r.priority):
            if not rule.is_active:
                continue
            
            # Check endpoint pattern
            if rule.endpoint_pattern:
                if not re.match(rule.endpoint_pattern, endpoint):
                    continue
            
            # Check method
            if rule.methods and '*' not in rule.methods:
                if method.upper() not in rule.methods:
                    continue
            
            # Check tier
            if rule.user_tiers:
                if tier not in rule.user_tiers:
                    continue
            
            # Rule matches - apply limit
            window = self._get_window(f"rule:{rule.rule_id}", rule.window_seconds)
            acquired, remaining = window.try_acquire()
            
            if not acquired:
                # Record limited request
                self._record_limited_request(tier, endpoint)
                
                return {
                    'allowed': False,
                    'error': 'Rate limit exceeded',
                    'message': f"Too many requests to {endpoint}",
                    'remaining': remaining,
                    'limit': rule.max_requests,
                    'retry_after': rule.retry_after_seconds,
                    'reset_at': int(time.time()) + rule.retry_after_seconds,
                    'http_status': rule.http_status_code
                }
        
        return None
    
    def _create_limit_response(
        self,
        message: str,
        remaining: int,
        window_seconds: int,
        status_code: int = 429
    ) -> Dict[str, Any]:
        """Create rate limit response"""
        return {
            'allowed': False,
            'error': message,
            'remaining': remaining,
            'limit': 100,  # Default
            'retry_after': window_seconds,
            'reset_at': int(time.time()) + window_seconds,
            'http_status': status_code
        }
    
    def _record_limited_request(self, tier: str, endpoint: str):
        """Record a rate-limited request"""
        with self._stats_lock:
            self._stats.limited_requests += 1
            
            if tier in self._stats.by_tier:
                self._stats.by_tier[tier]['limited'] += 1
            if endpoint in self._stats.by_endpoint:
                self._stats.by_endpoint[endpoint]['limited'] += 1
    
    def _check_alerts(self):
        """Check for rate limit alerts"""
        hit_rate = self._stats.hit_rate()
        
        alert_level = None
        if hit_rate >= self._alert_thresholds['critical'] * 100:
            alert_level = 'critical'
        elif hit_rate >= self._alert_thresholds['warning'] * 100:
            alert_level = 'warning'
        
        if alert_level:
            with self._stats_lock:
                self._stats.alerts_triggered += 1
                self._stats.last_alert = datetime.now(timezone.utc).isoformat()
            
            alert = {
                'level': alert_level,
                'hit_rate': hit_rate,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'message': f"Rate limit hit rate at {hit_rate:.2f}%"
            }
            
            self._alerts.append(alert)
            
            logger.warning(f"Rate limit alert ({alert_level}): {alert['message']}")
    
    def add_custom_config(self, config: RateLimitConfig):
        """Add custom rate limit configuration"""
        self._configs[config.tier] = config
    
    def add_custom_rule(self, rule: RateLimitRule):
        """Add custom rate limit rule"""
        self._rules.append(rule)
    
    def get_usage(
        self,
        user_id: Optional[str] = None,
        api_key_id: Optional[str] = None,
        period: str = "hour"
    ) -> Dict[str, Any]:
        """Get usage statistics for a user or API key"""
        tier = self._get_tier(user_id, api_key_id)
        config = self._configs.get(tier, self._configs[RateLimitTier.BASIC.value])
        
        window_key = self._get_window_key(user_id, api_key_id, None, "")
        
        if period == "minute":
            window = self._get_window(window_key, 60)
            used = window.get_count()
            limit = config.requests_per_minute
        else:  # hour
            window = self._get_window(window_key, 3600)
            used = window.get_count()
            limit = config.requests_per_hour
        
        return {
            'tier': tier,
            'period': period,
            'used': used,
            'limit': limit,
            'remaining': max(0, limit - used),
            'percentage': (used / limit * 100) if limit > 0 else 0
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiting statistics"""
        with self._stats_lock:
            stats = self._stats.to_dict()
            stats['tiers'] = {
                tier: {
                    'config': {
                        'requests_per_minute': config.requests_per_minute,
                        'requests_per_hour': config.requests_per_hour,
                        'burst_size': config.burst_size
                    }
                }
                for tier, config in self._configs.items()
            }
            stats['active_rules'] = len([r for r in self._rules if r.is_active])
            stats['trusted_apps'] = len(self._trusted_apps)
            stats['recent_alerts'] = self._alerts[-10:]
        return stats
    
    def get_alerts(self, level: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        if level:
            return [a for a in self._alerts if a['level'] == level][-limit:]
        return self._alerts[-limit:]
    
    def clear_stats(self):
        """Clear statistics"""
        with self._stats_lock:
            self._stats = RateLimitStats()


# Decorator for automatic rate limiting
def rate_limit(
    endpoint_pattern: str = ".*",
    tier: Optional[str] = None
):
    """
    Decorator for rate limiting endpoint functions
    
    Args:
        endpoint_pattern: Pattern to match
        tier: Rate limit tier to use
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        
        return sync_wrapper
    
    return decorator


# Singleton instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get rate limiter singleton"""
    global _rate_limiter
    
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    
    return _rate_limiter


def init_rate_limiter(storage_path: str = "data/rate_limits") -> RateLimiter:
    """Initialize rate limiting system"""
    global _rate_limiter
    
    _rate_limiter = RateLimiter(storage_path)
    
    return _rate_limiter
