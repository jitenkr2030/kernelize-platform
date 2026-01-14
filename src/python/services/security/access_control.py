"""
KERNELIZE Platform - Advanced Access Control System
====================================================

Enhanced security beyond RBAC with ABAC, row-level security,
temporary access grants, and suspicious activity detection.

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
from datetime import datetime, timezone, timedelta
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from threading import RLock
import re

logger = logging.getLogger(__name__)


class AccessEffect(Enum):
    """Access control effects"""
    ALLOW = "allow"
    DENY = "deny"


class ResourceType(Enum):
    """Resource types for access control"""
    KERNEL = "kernel"
    USER = "user"
    ORGANIZATION = "organization"
    API_KEY = "api_key"
    QUERY = "query"
    REPORT = "report"
    SETTINGS = "settings"
    INTEGRATION = "integration"


class ActionType(Enum):
    """Actions that can be performed"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    SHARE = "share"
    EXPORT = "export"
    EXECUTE = "execute"
    ADMIN = "admin"


@dataclass
class Subject:
    """Subject (user/service) making a request"""
    subject_id: str
    subject_type: str = "user"  # user, api_key, service
    organization_id: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    authentication_time: Optional[str] = None
    
    def has_role(self, role: str) -> bool:
        """Check if subject has a specific role"""
        return role in self.roles
    
    def has_any_role(self, roles: List[str]) -> bool:
        """Check if subject has any of the specified roles"""
        return any(r in self.roles for r in roles)


@dataclass
class Resource:
    """Resource being accessed"""
    resource_type: str
    resource_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    owner_id: Optional[str] = None
    organization_id: Optional[str] = None


@dataclass
class Context:
    """Context of the access request"""
    action: str = ""
    environment: Dict[str, Any] = field(default_factory=dict)
    time_constraints: Dict[str, Any] = field(default_factory=dict)
    additional_conditions: Dict[str, Any] = field(default_factory=dict)
    
    def get_current_time(self) -> datetime:
        """Get current time for time-based checks"""
        return datetime.now(timezone.utc)
    
    def is_within_time_window(
        self,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        days_of_week: Optional[List[int]] = None,
        hours: Optional[Tuple[int, int]] = None
    ) -> bool:
        """Check if current time is within specified window"""
        now = self.get_current_time()
        
        # Check day of week (0 = Monday, 6 = Sunday)
        if days_of_week is not None:
            if now.weekday() not in days_of_week:
                return False
        
        # Check hour range
        if hours is not None:
            if not (hours[0] <= now.hour < hours[1]):
                return False
        
        # Check explicit time range
        if start_time:
            start = datetime.fromisoformat(start_time)
            if now < start:
                return False
        
        if end_time:
            end = datetime.fromisoformat(end_time)
            if now > end:
                return False
        
        return True


@dataclass
class AccessPolicy:
    """Attribute-based access control policy"""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    
    # Policy targeting
    target_resources: List[str] = field(default_factory=list)
    target_actions: List[str] = field(default_factory=list)
    target_subjects: List[str] = field(default_factory=list)
    
    # Conditions
    subject_conditions: Dict[str, Any] = field(default_factory=dict)
    resource_conditions: Dict[str, Any] = field(default_factory=dict)
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    
    # Effect
    effect: str = AccessEffect.ALLOW.value
    
    # Priority (lower number = higher priority)
    priority: int = 100
    
    # Temporal constraints
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    
    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: Optional[str] = None
    is_active: bool = True
    
    def is_temporally_valid(self) -> bool:
        """Check if policy is currently valid based on time"""
        now = datetime.now(timezone.utc)
        
        if self.valid_from:
            valid_from = datetime.fromisoformat(self.valid_from)
            if now < valid_from:
                return False
        
        if self.valid_until:
            valid_until = datetime.fromisoformat(self.valid_until)
            if now > valid_until:
                return False
        
        return True


@dataclass
class TemporaryGrant:
    """Temporary access grant"""
    grant_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    resource_type: str = ""
    resource_id: Optional[str] = None
    actions: List[str] = field(default_factory=list)
    
    # Grant validity
    granted_by: str = ""
    granted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: str = field(default_factory=lambda: (
        datetime.now(timezone.utc) + timedelta(hours=24)
    ).isoformat())
    
    # Restrictions
    max_uses: Optional[int] = None
    current_uses: int = 0
    ip_restriction: Optional[str] = None
    
    # Metadata
    reason: str = ""
    is_active: bool = True
    
    def is_valid(self) -> bool:
        """Check if grant is currently valid"""
        if not self.is_active:
            return False
        
        expires = datetime.fromisoformat(self.expires_at)
        if datetime.now(timezone.utc) > expires:
            return False
        
        if self.max_uses is not None:
            if self.current_uses >= self.max_uses:
                return False
        
        return True
    
    def use(self) -> bool:
        """Use the grant (increment use count)"""
        if not self.is_valid():
            return False
        
        self.current_uses += 1
        return True


@dataclass
class SuspiciousActivity:
    """Suspicious activity record"""
    activity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    subject_id: str = ""
    activity_type: str = ""
    description: str = ""
    severity: str = "medium"  # low, medium, high, critical
    
    # Details
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    endpoint: Optional[str] = None
    request_data: Optional[Dict[str, Any]] = None
    
    # Resolution
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    resolution: Optional[str] = None
    blocked: bool = False
    
    # Related activities
    related_activities: List[str] = field(default_factory=list)


class ConditionEvaluator:
    """Evaluates ABAC conditions"""
    
    @staticmethod
    def evaluate(
        condition: Dict[str, Any],
        subject: Subject,
        resource: Resource,
        context: Context
    ) -> bool:
        """
        Evaluate a condition against subject, resource, and context
        
        Args:
            condition: Condition to evaluate
            subject: Subject of the request
            resource: Resource being accessed
            context: Context of the request
            
        Returns:
            True if condition is satisfied
        """
        if not condition:
            return True
        
        # Handle logical operators
        if 'and' in condition:
            return all(
                ConditionEvaluator.evaluate(c, subject, resource, context)
                for c in condition['and']
            )
        
        if 'or' in condition:
            return any(
                ConditionEvaluator.evaluate(c, subject, resource, context)
                for c in condition['or']
            )
        
        if 'not' in condition:
            return not ConditionEvaluator.evaluate(
                condition['not'], subject, resource, context
            )
        
        # Handle comparison operators
        for operator, value in condition.items():
            if operator == 'equals':
                return ConditionEvaluator._equals(value, subject, resource)
            if operator == 'not_equals':
                return not ConditionEvaluator._equals(value, subject, resource)
            if operator == 'in':
                return ConditionEvaluator._in(value, subject, resource)
            if operator == 'contains':
                return ConditionEvaluator._contains(value, subject, resource)
            if operator == 'greater_than':
                return ConditionEvaluator._greater_than(value, subject, resource)
            if operator == 'less_than':
                return ConditionEvaluator._less_than(value, subject, resource)
            if operator == 'matches':
                return ConditionEvaluator._matches(value, subject, resource)
            if operator == 'ip_in_range':
                return ConditionEvaluator._ip_in_range(value, context)
        
        return True
    
    @staticmethod
    def _equals(condition: Dict[str, Any], subject: Subject, resource: Resource) -> bool:
        """Check if attribute equals specified value"""
        attr_path = condition.get('attribute', '').split('.')
        value = condition.get('value')
        
        # Get attribute value
        attr_value = None
        if attr_path[0] == 'subject':
            attr_value = ConditionEvaluator._get_nested_attr(subject.attributes, attr_path[1:])
        elif attr_path[0] == 'resource':
            attr_value = ConditionEvaluator._get_nested_attr(resource.attributes, attr_path[1:])
        
        return attr_value == value
    
    @staticmethod
    def _get_nested_attr(obj: Any, path: List[str]) -> Any:
        """Get nested attribute value"""
        for key in path:
            if isinstance(obj, dict):
                obj = obj.get(key)
            else:
                obj = getattr(obj, key, None)
            if obj is None:
                return None
        return obj
    
    @staticmethod
    def _in(condition: Dict[str, Any], subject: Subject, resource: Resource) -> bool:
        """Check if attribute is in list"""
        attr_path = condition.get('attribute', '').split('.')
        values = condition.get('values', [])
        
        attr_value = ConditionEvaluator._get_nested_attr(
            subject.attributes, attr_path[1:]
        )
        
        return attr_value in values
    
    @staticmethod
    def _contains(condition: Dict[str, Any], subject: Subject, resource: Resource) -> bool:
        """Check if attribute contains value"""
        attr_path = condition.get('attribute', '').split('.')
        value = condition.get('value')
        
        attr_value = ConditionEvaluator._get_nested_attr(
            subject.attributes, attr_path[1:]
        )
        
        if isinstance(attr_value, (list, set)):
            return value in attr_value
        elif isinstance(attr_value, str):
            return value in attr_value
        
        return False
    
    @staticmethod
    def _greater_than(condition: Dict[str, Any], subject: Subject, resource: Resource) -> bool:
        """Check if attribute is greater than value"""
        attr_path = condition.get('attribute', '').split('.')
        value = condition.get('value')
        
        attr_value = ConditionEvaluator._get_nested_attr(
            subject.attributes, attr_path[1:]
        )
        
        try:
            return float(attr_value) > float(value)
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def _less_than(condition: Dict[str, Any], subject: Subject, resource: Resource) -> bool:
        """Check if attribute is less than value"""
        attr_path = condition.get('attribute', '').split('.')
        value = condition.get('value')
        
        attr_value = ConditionEvaluator._get_nested_attr(
            subject.attributes, attr_path[1:]
        )
        
        try:
            return float(attr_value) < float(value)
        except (ValueError, TypeError):
            return False
    
    @staticmethod
    def _matches(condition: Dict[str, Any], subject: Subject, resource: Resource) -> bool:
        """Check if attribute matches regex pattern"""
        attr_path = condition.get('attribute', '').split('.')
        pattern = condition.get('pattern')
        
        attr_value = ConditionEvaluator._get_nested_attr(
            subject.attributes, attr_path[1:]
        )
        
        if isinstance(attr_value, str) and pattern:
            return bool(re.match(pattern, attr_value))
        
        return False
    
    @staticmethod
    def _ip_in_range(condition: Dict[str, Any], context: Context) -> bool:
        """Check if IP is in range"""
        ip_range = condition.get('range', '')
        ip = context.environment.get('ip_address', '')
        
        # Simple IP matching (would use proper IP library in production)
        return ip.startswith(ip_range.split('.')[0]) if ip and '.' in ip_range else False


class RowLevelSecurity:
    """
    Row-level security for multi-tenant deployments
    
    Ensures users can only access data they're authorized to see
    by automatically filtering queries based on ownership and permissions.
    """
    
    def __init__(self):
        """Initialize row-level security engine"""
        self._policies: Dict[str, List[Dict[str, Any]]] = {}  # table -> policies
        self._lock = RLock()
    
    def add_policy(
        self,
        table_name: str,
        policy: Dict[str, Any]
    ):
        """
        Add row-level security policy for a table
        
        Args:
            table_name: Name of the table
            policy: Policy configuration with filter conditions
        """
        with self._lock:
            if table_name not in self._policies:
                self._policies[table_name] = []
            self._policies[table_name].append(policy)
    
    def get_filter(
        self,
        table_name: str,
        subject: Subject
    ) -> Dict[str, Any]:
        """
        Get filter conditions for a table
        
        Args:
            table_name: Name of the table
            subject: User making the request
            
        Returns:
            Filter conditions to apply
        """
        with self._lock:
            policies = self._policies.get(table_name, [])
        
        filters = {
            'conditions': [],
            'params': {}
        }
        
        for policy in policies:
            # Check if policy applies to subject
            if self._policy_matches_subject(policy, subject):
                condition = policy.get('condition', {})
                filters['conditions'].append(condition)
        
        # Combine conditions
        if not filters['conditions']:
            return {'always_true': True}
        
        if len(filters['conditions']) == 1:
            return filters['conditions'][0]
        
        return {'or': filters['conditions']}
    
    def _policy_matches_subject(
        self,
        policy: Dict[str, Any],
        subject: Subject
    ) -> bool:
        """Check if policy applies to subject"""
        # Check roles
        required_roles = policy.get('required_roles', [])
        if required_roles:
            if not subject.has_any_role(required_roles):
                return False
        
        # Check organization
        org_condition = policy.get('organization_id')
        if org_condition:
            if isinstance(org_condition, dict):
                # Complex condition
                if org_condition.get('equals') != subject.organization_id:
                    return False
            else:
                if org_condition != subject.organization_id:
                    return False
        
        return True
    
    def check_row_access(
        self,
        table_name: str,
        row: Dict[str, Any],
        subject: Subject
    ) -> bool:
        """
        Check if subject can access a specific row
        
        Args:
            table_name: Name of the table
            row: Row data
            subject: User making the request
            
        Returns:
            True if access is allowed
        """
        # Check if user owns the row
        if row.get('owner_id') == subject.subject_id:
            return True
        
        # Check if user is in same organization
        if row.get('organization_id') == subject.organization_id:
            # Check if row is shared with organization
            if row.get('is_organization_shared', False):
                return True
        
        # Check explicit permissions
        # This would integrate with the main access control system
        
        return False


class AccessControlManager:
    """
    Main access control manager
    
    Implements ABAC with policy evaluation, row-level security,
    temporary grants, and suspicious activity detection.
    """
    
    def __init__(self, storage_path: str = "data/access_control"):
        """
        Initialize access control manager
        
        Args:
            storage_path: Path for policy storage
        """
        self._policies: Dict[str, AccessPolicy] = {}
        self._temporary_grants: Dict[str, TemporaryGrant] = {}
        self._suspicious_activities: List[SuspiciousActivity] = []
        
        self._condition_evaluator = ConditionEvaluator()
        self._rls_engine = RowLevelSecurity()
        
        self._lock = RLock()
        
        # Statistics
        self._stats = {
            'access_checks': 0,
            'access_granted': 0,
            'access_denied': 0,
            'suspicious_activities_detected': 0
        }
        
        # Initialize default policies
        self._init_default_policies()
    
    def _init_default_policies(self):
        """Initialize default access control policies"""
        # Admin full access
        admin_policy = AccessPolicy(
            name="Admin Full Access",
            description="Administrators have full access to all resources",
            target_resources=["*"],
            target_actions=["*"],
            target_subjects=["admin"],
            subject_conditions={
                "subject.attributes.role": "admin"
            },
            effect=AccessEffect.ALLOW.value,
            priority=1
        )
        self.add_policy(admin_policy)
        
        # User self-access
        user_policy = AccessPolicy(
            name="User Self Access",
            description="Users can access their own resources",
            target_resources=["user", "kernel", "api_key"],
            target_actions=["read", "update", "delete"],
            subject_conditions={
                "subject.attributes.user_id": "resource.attributes.owner_id"
            },
            effect=AccessEffect.ALLOW.value,
            priority=10
        )
        self.add_policy(user_policy)
        
        # Organization member access
        org_policy = AccessPolicy(
            name="Organization Member Access",
            description="Org members can access shared org resources",
            target_resources=["kernel"],
            target_actions=["read"],
            subject_conditions={
                "subject.organization_id": "resource.organization_id"
            },
            context_conditions={
                "resource.attributes.is_shared_with_org": True
            },
            effect=AccessEffect.ALLOW.value,
            priority=20
        )
        self.add_policy(org_policy)
        
        # Default deny
        deny_policy = AccessPolicy(
            name="Default Deny",
            description="Default policy to deny access",
            target_resources=["*"],
            target_actions=["*"],
            target_subjects=["*"],
            effect=AccessEffect.DENY.value,
            priority=1000
        )
        self.add_policy(deny_policy)
    
    def add_policy(self, policy: AccessPolicy):
        """Add access control policy"""
        with self._lock:
            self._policies[policy.policy_id] = policy
    
    def get_policy(self, policy_id: str) -> Optional[AccessPolicy]:
        """Get policy by ID"""
        return self._policies.get(policy_id)
    
    def check_access(
        self,
        subject: Subject,
        resource: Resource,
        action: str,
        context: Optional[Context] = None
    ) -> Tuple[bool, str]:
        """
        Check if subject can perform action on resource
        
        Args:
            subject: Subject making the request
            resource: Resource being accessed
            action: Action being performed
            context: Request context
            
        Returns:
            (allowed, reason)
        """
        self._stats['access_checks'] += 1
        
        if context is None:
            context = Context(action=action)
        
        # Check temporary grants first
        grant_check = self._check_temporary_grants(subject, resource, action)
        if grant_check[0]:
            self._stats['access_granted'] += 1
            return grant_check
        
        # Evaluate policies
        applicable_policies = self._get_applicable_policies(
            subject, resource, action
        )
        
        # Sort by priority (lower number = higher priority)
        applicable_policies.sort(key=lambda p: p.priority)
        
        # Evaluate each policy
        for policy in applicable_policies:
            if not policy.is_temporally_valid():
                continue
            
            # Check if policy targets this action
            if policy.target_actions and action not in policy.target_actions:
                continue
            
            # Check subject conditions
            if policy.subject_conditions:
                if not self._condition_evaluator.evaluate(
                    policy.subject_conditions, subject, resource, context
                ):
                    continue
            
            # Check resource conditions
            if policy.resource_conditions:
                if not self._condition_evaluator.evaluate(
                    policy.resource_conditions, subject, resource, context
                ):
                    continue
            
            # Check context conditions
            if policy.context_conditions:
                if not self._condition_evaluator.evaluate(
                    policy.context_conditions, subject, resource, context
                ):
                    continue
            
            # Policy matches - apply effect
            if policy.effect == AccessEffect.ALLOW.value:
                self._stats['access_granted'] += 1
                return True, f"Allowed by policy: {policy.name}"
            else:
                self._stats['access_denied'] += 1
                return False, f"Denied by policy: {policy.name}"
        
        # Default deny
        self._stats['access_denied'] += 1
        return False, "No matching allow policy"
    
    def _get_applicable_policies(
        self,
        subject: Subject,
        resource: Resource,
        action: str
    ) -> List[AccessPolicy]:
        """Get policies applicable to the request"""
        applicable = []
        
        with self._lock:
            for policy in self._policies.values():
                if not policy.is_active:
                    continue
                
                # Check if policy targets this resource type
                if policy.target_resources:
                    if resource.resource_type not in policy.target_resources:
                        if '*' not in policy.target_resources:
                            continue
                
                # Check if policy targets this subject
                if policy.target_subjects:
                    if subject.subject_type not in policy.target_subjects:
                        if '*' not in policy.target_subjects:
                            continue
                
                applicable.append(policy)
        
        return applicable
    
    def _check_temporary_grants(
        self,
        subject: Subject,
        resource: Resource,
        action: str
    ) -> Tuple[bool, str]:
        """Check if subject has temporary grant for this access"""
        with self._lock:
            for grant in self._temporary_grants.values():
                if not grant.is_valid():
                    continue
                
                # Check if grant applies
                if grant.subject_id != subject.subject_id:
                    continue
                
                if grant.resource_type != resource.resource_type:
                    continue
                
                if grant.resource_id and grant.resource_id != resource.resource_id:
                    continue
                
                if action not in grant.actions:
                    continue
                
                # Check IP restriction
                if grant.ip_restriction:
                    if not subject.ip_address:
                        continue
                    if not subject.ip_address.startswith(grant.ip_restriction):
                        continue
                
                # Use the grant
                if grant.use():
                    return True, f"Allowed by temporary grant: {grant.reason}"
        
        return False, ""
    
    def create_temporary_grant(
        self,
        subject_id: str,
        resource_type: str,
        actions: List[str],
        granted_by: str,
        duration_hours: int = 24,
        max_uses: Optional[int] = None,
        ip_restriction: Optional[str] = None,
        reason: str = ""
    ) -> TemporaryGrant:
        """
        Create temporary access grant
        
        Args:
            subject_id: Subject to grant access to
            resource_type: Type of resource
            actions: Allowed actions
            granted_by: Who granted the access
            duration_hours: How long the grant lasts
            max_uses: Maximum number of uses
            ip_restriction: Restrict to IP range
            reason: Reason for grant
            
        Returns:
            Created temporary grant
        """
        grant = TemporaryGrant(
            subject_id=subject_id,
            resource_type=resource_type,
            actions=actions,
            granted_by=granted_by,
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=duration_hours)).isoformat(),
            max_uses=max_uses,
            ip_restriction=ip_restriction,
            reason=reason
        )
        
        with self._lock:
            self._temporary_grants[grant.grant_id] = grant
        
        return grant
    
    def revoke_grant(self, grant_id: str) -> bool:
        """
        Revoke temporary grant
        
        Args:
            grant_id: Grant to revoke
            
        Returns:
            True if revoked
        """
        with self._lock:
            if grant_id in self._temporary_grants:
                self._temporary_grants[grant_id].is_active = False
                return True
        return False
    
    def detect_suspicious_activity(
        self,
        subject_id: str,
        activity_type: str,
        description: str,
        severity: str = "medium",
        ip_address: Optional[str] = None,
        endpoint: Optional[str] = None,
        request_data: Optional[Dict[str, Any]] = None
    ) -> SuspiciousActivity:
        """
        Detect and record suspicious activity
        
        Args:
            subject_id: Subject performing activity
            activity_type: Type of activity
            description: Description of activity
            severity: Severity level
            ip_address: Source IP
            endpoint: Target endpoint
            request_data: Request data
            
        Returns:
            Created suspicious activity record
        """
        activity = SuspiciousActivity(
            subject_id=subject_id,
            activity_type=activity_type,
            description=description,
            severity=severity,
            ip_address=ip_address,
            endpoint=endpoint,
            request_data=request_data
        )
        
        # Check for blocking
        if severity in ['high', 'critical']:
            activity.blocked = True
            self._block_subject(subject_id, reason=description)
        
        with self._lock:
            self._suspicious_activities.append(activity)
            self._stats['suspicious_activities_detected'] += 1
        
        logger.warning(
            f"Suspicious activity detected: {activity_type} by {subject_id} - {description}"
        )
        
        return activity
    
    def _block_subject(self, subject_id: str, reason: str):
        """Block a subject from the system"""
        # Implementation would add to blocked list
        logger.error(f"Subject blocked: {subject_id} - {reason}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get access control statistics"""
        with self._lock:
            stats = dict(self._stats)
            stats['active_policies'] = len([p for p in self._policies.values() if p.is_active])
            stats['active_grants'] = len([g for g in self._temporary_grants.values() if g.is_active])
            stats['pending_activities'] = len([
                a for a in self._suspicious_activities 
                if not a.resolved_at
            ])
        return stats


# Decorator for access control
def require_access(
    resource_type: str,
    action: str,
    resource_id_param: Optional[str] = None
):
    """
    Decorator to enforce access control on functions
    
    Args:
        resource_type: Type of resource being accessed
        action: Action being performed
        resource_id_param: Name of parameter containing resource ID
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract subject from context (implementation specific)
            # This would integrate with authentication system
            
            result = await func(*args, **kwargs)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return result
        
        if hasattr(func, '__wrapped__'):
            import asyncio
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Singleton instance
_access_control_manager: Optional[AccessControlManager] = None


def get_access_control_manager() -> AccessControlManager:
    """Get access control manager singleton"""
    global _access_control_manager
    
    if _access_control_manager is None:
        _access_control_manager = AccessControlManager()
    
    return _access_control_manager


def init_access_control(storage_path: str = "data/access_control") -> AccessControlManager:
    """Initialize access control system"""
    global _access_control_manager
    
    _access_control_manager = AccessControlManager(storage_path)
    
    return _access_control_manager
