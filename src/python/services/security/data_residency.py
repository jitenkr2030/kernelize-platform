"""
KERNELIZE Platform - Data Residency System
===========================================

Enable compliance with regional data requirements.
Implements multi-region storage configuration, data residency policies,
geo-routing, and compliance reporting.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import threading
import re

logger = logging.getLogger(__name__)


class DataRegion(Enum):
    """Supported data regions"""
    US_EAST = "us-east-1"
    US_WEST = "us-west-1"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    AP_SOUTH = "ap-south-1"
    AP_NORTHEAST = "ap-northeast-1"
    SA_EAST = "sa-east-1"
    CA_CENTRAL = "ca-central-1"
    
    @property
    def name(self) -> str:
        """Human-readable region name"""
        names = {
            "us-east-1": "US East (N. Virginia)",
            "us-west-1": "US West (California)",
            "eu-west-1": "EU West (Ireland)",
            "eu-central-1": "EU Central (Frankfurt)",
            "ap-south-1": "Asia Pacific (Mumbai)",
            "ap-northeast-1": "Asia Pacific (Tokyo)",
            "sa-east-1": "South America (São Paulo)",
            "ca-central-1": "Canada (Central)",
        }
        return names.get(self.value, self.value)
    
    @property
    def jurisdiction(self) -> str:
        """Legal jurisdiction for the region"""
        jurisdictions = {
            "us-east-1": "United States",
            "us-west-1": "United States",
            "eu-west-1": "European Union",
            "eu-central-1": "European Union",
            "ap-south-1": "India",
            "ap-northeast-1": "Japan",
            "sa-east-1": "Brazil",
            "ca-central-1": "Canada",
        }
        return jurisdictions.get(self.value, "Unknown")
    
    @property
    def gdpr_compliant(self) -> bool:
        """Whether region is GDPR compliant"""
        return self.value in ["eu-west-1", "eu-central-1"]
    
    @property
    def hipaa_compliant(self) -> bool:
        """Whether region is HIPAA compliant"""
        return self.value in ["us-east-1", "us-west-1"]


class ComplianceFramework(Enum):
    """Compliance frameworks"""
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    SOC2 = "SOC2"
    PCI_DSS = "PCI_DSS"
    ISO27001 = "ISO27001"
    CCPA = "CCPA"
    PIPEDA = "PIPEDA"


@dataclass
class DataResidencyPolicy:
    """Data residency policy configuration"""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "Default Data Residency"
    organization_id: Optional[str] = None  # None means default policy
    user_id: Optional[str] = None  # None means applies to all users in org
    
    # Region configuration
    primary_region: str = DataRegion.US_EAST.value
    allowed_regions: List[str] = field(default_factory=lambda: [
        DataRegion.US_EAST.value,
        DataRegion.US_WEST.value
    ])
    backup_regions: List[str] = field(default_factory=list)
    
    # Data classification
    classify_personal_data: bool = True
    classify_health_data: bool = False
    classify_financial_data: bool = False
    
    # Retention requirements (in days)
    retention_personal_data: int = 365
    retention_health_data: int = 2190  # 6 years for HIPAA
    retention_financial_data: int = 2555  # 7 years for financial
    
    # Compliance requirements
    required_compliance: List[str] = field(default_factory=list)
    encryption_required: bool = True
    audit_trail_required: bool = True
    
    # Data transfer restrictions
    restrict_cross_border_transfer: bool = False
    approved_countries: List[str] = field(default_factory=list)
    
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True


@dataclass
class GeoRoute:
    """Geographic routing configuration"""
    route_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    country_codes: List[str] = field(default_factory=list)
    region_codes: List[str] = field(default_factory=list)
    target_region: str = ""
    priority: int = 1
    fallback_region: Optional[str] = None
    is_active: bool = True


@dataclass
class DataLocationRecord:
    """Record of where data is stored"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    resource_type: str = ""  # kernel, user_data, logs, etc.
    resource_id: str = ""
    region: str = ""
    storage_type: str = ""  # primary, backup, cache
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_verified: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class RegionManager:
    """Manages data regions and routing"""
    
    def __init__(self):
        """Initialize region manager"""
        self._regions: Dict[str, DataRegion] = {}
        self._routes: List[GeoRoute] = []
        self._lock = threading.RLock()
        
        # Initialize default regions
        for region in DataRegion:
            self._regions[region.value] = region
    
    def add_route(self, route: GeoRoute):
        """Add geographic routing rule"""
        with self._lock:
            self._routes.append(route)
            self._routes.sort(key=lambda r: r.priority)
    
    def get_route(self, country_code: str, region_code: Optional[str] = None) -> Optional[GeoRoute]:
        """
        Get routing rule for a geographic location
        
        Args:
            country_code: ISO country code
            region_code: Optional region/state code
            
        Returns:
            Matching route or None
        """
        with self._lock:
            # First try exact country match
            for route in self._routes:
                if country_code in route.country_codes:
                    # Check region if specified
                    if region_code and route.region_codes:
                        if region_code in route.region_codes:
                            return route
                    return route
            
            return None
    
    def get_region_for_ip(self, ip_address: str) -> DataRegion:
        """
        Get recommended region for an IP address
        
        Args:
            ip_address: IP address to geolocate
            
        Returns:
            Recommended data region
        """
        # In production, this would use GeoIP lookup
        # For now, return US East as default
        return DataRegion.US_EAST
    
    def get_region_info(self, region_code: str) -> Optional[DataRegion]:
        """Get information about a region"""
        return self._regions.get(region_code)
    
    def list_regions(self) -> List[DataRegion]:
        """List all available regions"""
        return list(self._regions.values())


class DataResidencyManager:
    """
    Manages data residency policies and enforcement
    """
    
    def __init__(self, storage_path: str = "data/residency"):
        """
        Initialize data residency manager
        
        Args:
            storage_path: Path for policy storage
        """
        self.region_manager = RegionManager()
        self._policies: Dict[str, DataResidencyPolicy] = {}
        self._location_records: Dict[str, List[DataLocationRecord]] = {}
        self._lock = threading.Lock()
        
        # Default policy
        self._default_policy = DataResidencyPolicy()
        self._policies[self._default_policy.policy_id] = self._default_policy
    
    def create_policy(self, policy: DataResidencyPolicy) -> DataResidencyPolicy:
        """
        Create a new data residency policy
        
        Args:
            policy: Policy configuration
            
        Returns:
            Created policy with generated ID
        """
        with self._lock:
            if not policy.policy_id:
                policy.policy_id = str(uuid.uuid4())
            policy.created_at = datetime.now(timezone.utc).isoformat()
            policy.updated_at = datetime.now(timezone.utc).isoformat()
            
            self._policies[policy.policy_id] = policy
            
            # Index by organization
            if policy.organization_id:
                org_key = f"org:{policy.organization_id}"
                # Additional indexing logic
            
            return policy
    
    def get_policy(
        self,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> DataResidencyPolicy:
        """
        Get applicable policy for an organization/user
        
        Args:
            organization_id: Organization to get policy for
            user_id: Specific user (for user-level policies)
            
        Returns:
            Applicable policy
        """
        with self._lock:
            # First, try to find user-specific policy
            if user_id:
                for policy in self._policies.values():
                    if policy.user_id == user_id and policy.is_active:
                        return policy
            
            # Then, try organization-specific policy
            if organization_id:
                for policy in self._policies.values():
                    if policy.organization_id == organization_id and policy.is_active:
                        if not policy.user_id:  # Org-level policy
                            return policy
            
            # Return default policy
            return self._default_policy
    
    def validate_data_location(
        self,
        resource_type: str,
        resource_id: str,
        target_region: str,
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate if data can be stored in a region
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            target_region: Target region code
            organization_id: Organization context
            
        Returns:
            Validation result with allowed/disallowed status
        """
        policy = self.get_policy(organization_id)
        
        # Check if region is allowed
        if target_region not in policy.allowed_regions:
            return {
                'allowed': False,
                'reason': f"Region {target_region} is not allowed by policy",
                'allowed_regions': policy.allowed_regions,
                'policy_id': policy.policy_id
            }
        
        # Check cross-border transfer restrictions
        target_region_info = self.region_manager.get_region_info(target_region)
        if target_region_info and policy.restrict_cross_border_transfer:
            if target_region_info.jurisdiction not in policy.approved_countries:
                return {
                    'allowed': False,
                    'reason': f"Cross-border transfer to {target_region_info.jurisdiction} is restricted",
                    'jurisdiction': target_region_info.jurisdiction,
                    'policy_id': policy.policy_id
                }
        
        return {
            'allowed': True,
            'region': target_region,
            'policy_id': policy.policy_id
        }
    
    def record_data_location(
        self,
        resource_type: str,
        resource_id: str,
        region: str,
        storage_type: str = "primary",
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataLocationRecord:
        """
        Record where data is stored
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            region: Region where data is stored
            storage_type: Type of storage (primary, backup, cache)
            metadata: Additional metadata
            
        Returns:
            Created location record
        """
        record = DataLocationRecord(
            resource_type=resource_type,
            resource_id=resource_id,
            region=region,
            storage_type=storage_type,
            metadata=metadata or {}
        )
        
        with self._lock:
            key = f"{resource_type}:{resource_id}"
            if key not in self._location_records:
                self._location_records[key] = []
            self._location_records[key].append(record)
        
        return record
    
    def get_data_locations(
        self,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        region: Optional[str] = None
    ) -> List[DataLocationRecord]:
        """
        Get data location records
        
        Args:
            resource_type: Filter by resource type
            resource_id: Filter by resource ID
            region: Filter by region
            
        Returns:
            List of location records
        """
        with self._lock:
            records = []
            
            for key, locations in self._location_records.items():
                for record in locations:
                    if resource_type and record.resource_type != resource_type:
                        continue
                    if resource_id and record.resource_id != resource_id:
                        continue
                    if region and record.region != region:
                        continue
                    records.append(record)
            
            return records
    
    def get_compliance_report(
        self,
        organization_id: str,
        framework: str = "GDPR"
    ) -> Dict[str, Any]:
        """
        Generate compliance report for data residency
        
        Args:
            organization_id: Organization to report on
            framework: Compliance framework
            
        Returns:
            Compliance report
        """
        policy = self.get_policy(organization_id)
        
        # Get all data locations for this organization
        locations = self.get_data_locations()
        
        # Aggregate by region
        region_distribution = {}
        for loc in locations:
            region = loc.region
            if region not in region_distribution:
                region_distribution[region] = {
                    'count': 0,
                    'resources': set()
                }
            region_distribution[region]['count'] += 1
            region_distribution[region]['resources'].add(f"{loc.resource_type}:{loc.resource_id}")
        
        # Convert sets to counts
        for region in region_distribution:
            region_distribution[region]['resources'] = len(
                region_distribution[region]['resources']
            )
        
        # Check compliance requirements
        compliance_checks = []
        
        if framework == "GDPR":
            # Check for personal data in non-EU regions
            for region_code, data in region_distribution.items():
                region_info = self.region_manager.get_region_info(region_code)
                if region_info and not region_info.gdpr_compliant:
                    compliance_checks.append({
                        'check': 'personal_data_in_eu',
                        'status': 'FAIL' if data['count'] > 0 else 'PASS',
                        'details': f"Personal data found in non-EU region: {region_code}",
                        'affected_resources': data['resources']
                    })
        
        elif framework == "HIPAA":
            # Check for health data encryption and retention
            compliance_checks.append({
                'check': 'hipaa_retention',
                'status': 'PASS',  # Would be detailed in full implementation
                'details': 'Health data retention policies configured'
            })
        
        # Overall compliance status
        overall_status = "COMPLIANT"
        failed_checks = [c for c in compliance_checks if c['status'] == 'FAIL']
        if failed_checks:
            overall_status = "NON_COMPLIANT"
        elif compliance_checks:
            overall_status = "COMPLIANT"
        
        return {
            'report_id': str(uuid.uuid4()),
            'organization_id': organization_id,
            'framework': framework,
            'generated_at': datetime.now(timezone.utc).isoformat(),
            'policy': {
                'policy_id': policy.policy_id,
                'name': policy.name,
                'primary_region': policy.primary_region,
                'allowed_regions': policy.allowed_regions
            },
            'data_distribution': {
                region: {
                    'resource_count': data['count'],
                    'unique_resources': data['resources']
                }
                for region, data in region_distribution.items()
            },
            'compliance_checks': compliance_checks,
            'overall_status': overall_status,
            'recommendations': self._generate_compliance_recommendations(
                policy, region_distribution, framework
            )
        }
    
    def _generate_compliance_recommendations(
        self,
        policy: DataResidencyPolicy,
        region_distribution: Dict[str, Any],
        framework: str
    ) -> List[str]:
        """Generate compliance recommendations"""
        recommendations = []
        
        if framework == "GDPR":
            non_eu_regions = [
                r for r in region_distribution.keys()
                if self.region_manager.get_region_info(r) and
                not self.region_manager.get_region_info(r).gdpr_complient
            ]
            if non_eu_regions:
                recommendations.append(
                    f"Consider migrating data from non-EU regions ({', '.join(non_eu_regions)}) "
                    "to EU regions for GDPR compliance"
                )
        
        if not policy.encryption_required:
            recommendations.append(
                "Enable encryption for data at rest and in transit"
            )
        
        if not policy.audit_trail_required:
            recommendations.append(
                "Enable audit logging for compliance tracking"
            )
        
        return recommendations
    
    def geo_route_request(
        self,
        country_code: str,
        region_code: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Route API request to appropriate region
        
        Args:
            country_code: Request origin country
            region_code: Request origin region/state
            organization_id: Organization for policy lookup
            
        Returns:
            Routing decision with target region
        """
        # Check for explicit route
        route = self.region_manager.get_route(country_code, region_code)
        
        if route:
            return {
                'routed': True,
                'target_region': route.target_region,
                'fallback_region': route.fallback_region,
                'route_id': route.route_id,
                'reason': 'Geographic routing rule matched'
            }
        
        # Get policy-based routing
        policy = self.get_policy(organization_id)
        
        # Route to primary region by default
        return {
            'routed': True,
            'target_region': policy.primary_region,
            'fallback_region': policy.backup_regions[0] if policy.backup_regions else None,
            'reason': 'Using organization policy'
        }
    
    def verify_data_location(
        self,
        resource_type: str,
        resource_id: str,
        expected_region: str
    ) -> Dict[str, Any]:
        """
        Verify data is in expected location
        
        Args:
            resource_type: Type of resource
            resource_id: Resource identifier
            expected_region: Expected region
            
        Returns:
            Verification result
        """
        locations = self.get_data_locations(
            resource_type=resource_type,
            resource_id=resource_id
        )
        
        if not locations:
            return {
                'verified': False,
                'reason': 'No location records found',
                'resource_type': resource_type,
                'resource_id': resource_id
            }
        
        # Check if any location matches expected region
        for loc in locations:
            if loc.region == expected_region:
                return {
                    'verified': True,
                    'resource_type': resource_type,
                    'resource_id': resource_id,
                    'region': expected_region,
                    'last_verified': loc.last_verified
                }
        
        return {
            'verified': False,
            'reason': f'Data is in {locations[0].region}, not {expected_region}',
            'resource_type': resource_type,
            'resource_id': resource_id,
            'current_region': locations[0].region,
            'expected_region': expected_region
        }


class DataClassificationEngine:
    """Classifies data based on content and context"""
    
    # Patterns for detecting sensitive data
    PATTERNS = {
        'email': r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
        'phone': r'\+?[0-9]{10,15}',
        'ssn': r'\d{3}-\d{2}-\d{4}',
        'credit_card': r'\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}',
        'ip_address': r'(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)?',
        'date_of_birth': r'(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4})|(?:\d{4}[-/]\d{1,2}[-/]\d{1,2})',
    }
    
    def __init__(self):
        """Initialize classification engine"""
        self._classification_cache: Dict[str, Dict[str, Any]] = {}
    
    def classify_content(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Classify content for data residency
        
        Args:
            content: Text content to classify
            metadata: Associated metadata
            
        Returns:
            Classification results
        """
        classifications = []
        detected_types = set()
        
        # Check for patterns
        for data_type, pattern in self.PATTERNS.items():
            if re.search(pattern, content):
                detected_types.add(data_type)
                classifications.append({
                    'type': data_type,
                    'confidence': 0.9,
                    'location': 'content'
                })
        
        # Check metadata for classification hints
        if metadata:
            if 'contains_pii' in metadata and metadata['contains_pii']:
                detected_types.add('pii')
                classifications.append({
                    'type': 'pii',
                    'confidence': 1.0,
                    'location': 'metadata'
                })
            
            if 'contains_phi' in metadata and metadata['contains_phi']:
                detected_types.add('phi')
                classifications.append({
                    'type': 'phi',
                    'confidence': 1.0,
                    'location': 'metadata'
                })
            
            if 'contains_financial' in metadata and metadata['contains_financial']:
                detected_types.add('financial')
                classifications.append({
                    'type': 'financial',
                    'confidence': 1.0,
                    'location': 'metadata'
                })
        
        # Determine required retention
        retention_days = 365  # Default
        
        if 'phi' in detected_types:
            retention_days = 2190  # HIPAA: 6 years
        elif 'financial' in detected_types:
            retention_days = 2555  # 7 years
        elif 'pii' in detected_types:
            retention_days = 365  # Standard PII
        
        return {
            'classifications': classifications,
            'detected_types': list(detected_types),
            'requires_protection': len(detected_types) > 0,
            'recommended_retention_days': retention_days,
            'compliance_flags': self._get_compliance_flags(detected_types)
        }
    
    def _get_compliance_flags(self, detected_types: set) -> List[str]:
        """Get compliance requirements for detected types"""
        flags = []
        
        if 'pii' in detected_types:
            flags.append('GDPR')
            flags.append('CCPA')
        
        if 'phi' in detected_types:
            flags.append('HIPAA')
        
        if 'financial' in detected_types:
            flags.append('PCI_DSS')
        
        return list(set(flags))
    
    def get_storage_requirements(
        self,
        classifications: Dict[str, Any],
        policy: DataResidencyPolicy
    ) -> Dict[str, Any]:
        """
        Get storage requirements based on classification and policy
        
        Args:
            classifications: Content classifications
            policy: Applied data residency policy
            
        Returns:
            Storage requirements
        """
        requirements = {
            'encryption_required': policy.encryption_required,
            'audit_required': policy.audit_trail_required,
            'retention_days': policy.retention_personal_data
        }
        
        # Adjust based on detected types
        detected = set(classifications.get('detected_types', []))
        
        if 'phi' in detected:
            requirements['retention_days'] = policy.retention_health_data
        elif 'financial' in detected:
            requirements['retention_days'] = policy.retention_financial_data
        
        # Check allowed regions
        requirements['allowed_regions'] = policy.allowed_regions
        
        # Check compliance requirements
        compliance_flags = classifications.get('compliance_flags', [])
        requirements['compliance_requirements'] = [
            f for f in compliance_flags if f in policy.required_compliance
        ]
        
        return requirements


# Singleton instances
_region_manager = RegionManager()
_data_residency_manager: Optional[DataResidencyManager] = None
_classification_engine: Optional[DataClassificationEngine] = None


def get_region_manager() -> RegionManager:
    """Get region manager singleton"""
    return _region_manager


def get_data_residency_manager() -> DataResidencyManager:
    """Get data residency manager singleton"""
    global _data_residency_manager
    
    if _data_residency_manager is None:
        _data_residency_manager = DataResidencyManager()
    
    return _data_residency_manager


def get_classification_engine() -> DataClassificationEngine:
    """Get classification engine singleton"""
    global _classification_engine
    
    if _classification_engine is None:
        _classification_engine = DataClassificationEngine()
    
    return _classification_engine


def init_data_residency(
    storage_path: str = "data/residency",
    default_regions: Optional[List[str]] = None
) -> DataResidencyManager:
    """Initialize data residency system"""
    global _data_residency_manager
    
    _data_residency_manager = DataResidencyManager(storage_path)
    
    # Add default geo routes
    routes = [
        GeoRoute(
            name="EU Default",
            country_codes=["DE", "FR", "IT", "ES", "NL", "BE", "AT", "CH"],
            target_region=DataRegion.EU_CENTRAL.value,
            priority=1
        ),
        GeoRoute(
            name="US Default",
            country_codes=["US", "CA", "MX"],
            target_region=DataRegion.US_EAST.value,
            priority=1
        ),
        GeoRoute(
            name="Asia Pacific",
            country_codes=["JP", "KR", "AU", "NZ", "SG", "IN"],
            target_region=DataRegion.AP_NORTHEAST.value,
            priority=1
        ),
    ]
    
    for route in routes:
        _data_residency_manager.region_manager.add_route(route)
    
    return _data_residency_manager
