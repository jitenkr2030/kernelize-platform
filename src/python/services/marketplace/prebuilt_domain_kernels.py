#!/usr/bin/env python3
"""
Pre-Built Domain Kernels

This module provides a comprehensive catalog of ready-to-use kernels
for specific domains. Each domain kernel includes optimized configurations,
specialized prompts, domain-specific constraints, and appropriate
capabilities for immediate deployment.

Domain Kernels Available:
- Medical/Healthcare
- Legal/Corporate
- Financial/Trading
- Scientific Research
- Customer Support
- Code Development
- Creative Writing
- Document Analysis
- Translation/Localization
- Data Science

Author: MiniMax Agent
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path


class DomainCategory(Enum):
    """
    Enumeration of supported domain categories for pre-built kernels.
    Each category represents a specialized field with unique requirements
    and optimization strategies.
    """
    MEDICAL = "medical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    SCIENTIFIC = "scientific"
    CUSTOMER_SUPPORT = "customer_support"
    CODE_DEVELOPMENT = "code_development"
    CREATIVE_WRITING = "creative_writing"
    DOCUMENT_ANALYSIS = "document_analysis"
    TRANSLATION = "translation"
    DATA_SCIENCE = "data_science"
    GENERAL = "general"


class KernelComplexity(Enum):
    """
    Complexity levels for domain kernels based on use case requirements.
    """
    LIGHT = "light"         # Simple queries, fast response
    STANDARD = "standard"   # Balanced capability and speed
    ADVANCED = "advanced"   # Complex reasoning, deeper analysis
    EXPERT = "expert"       # Maximum capability, longer context


class KernelTargetAudience(Enum):
    """
    Target audience classification for domain kernels.
    """
    BEGINNER = "beginner"           # Easy to use, guided interactions
    INTERMEDIATE = "intermediate"   # Standard business use
    PROFESSIONAL = "professional"   # Professional/technical users
    EXPERT = "expert"               # Advanced professional users
    ENTERPRISE = "enterprise"       # Large organization deployment


@dataclass
class DomainKernelMetadata:
    """
    Metadata describing a pre-built domain kernel.
    """
    kernel_id: str
    name: str
    description: str
    domain: DomainCategory
    complexity: KernelComplexity
    target_audience: KernelTargetAudience
    
    # Version and compatibility
    version: str = "1.0.0"
    min_kernel_version: str = "1.0.0"
    
    # Statistics
    average_response_time_ms: float = 0.0
    success_rate: float = 0.95
    user_rating: float = 0.0
    usage_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_validated: datetime = field(default_factory=datetime.now)
    
    # Categorization
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Quality indicators
    is_featured: bool = False
    is_verified: bool = False
    certification_level: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary."""
        return {
            "kernel_id": self.kernel_id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain.value,
            "complexity": self.complexity.value,
            "target_audience": self.target_audience.value,
            "version": self.version,
            "statistics": {
                "avg_response_time_ms": self.average_response_time_ms,
                "success_rate": self.success_rate,
                "user_rating": self.user_rating,
                "usage_count": self.usage_count
            },
            "tags": self.tags,
            "keywords": self.keywords,
            "quality": {
                "is_featured": self.is_featured,
                "is_verified": self.is_verified,
                "certification_level": self.certification_level
            },
            "timestamps": {
                "created_at": self.created_at.isoformat(),
                "updated_at": self.updated_at.isoformat(),
                "last_validated": self.last_validated.isoformat()
            }
        }


@dataclass
class DomainSpecificConfiguration:
    """
    Domain-specific configuration for kernel behavior.
    """
    # Response configuration
    response_style: str = "professional"  # formal, casual, technical, creative
    detail_level: str = "comprehensive"   # brief, standard, comprehensive
    include_references: bool = True
    include_confidence_scores: bool = True
    
    # Context handling
    max_context_length: int = 4096
    prefer_recent_information: bool = True
    information_freshness_days: int = 30
    
    # Safety and compliance
    require_human_review: bool = False
    compliance_frameworks: List[str] = field(default_factory=list)
    disclaimer_required: bool = False
    disclaimer_text: str = ""
    
    # Domain-specific settings
    domain_settings: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "response_style": self.response_style,
            "detail_level": self.detail_level,
            "include_references": self.include_references,
            "include_confidence_scores": self.include_confidence_scores,
            "max_context_length": self.max_context_length,
            "prefer_recent_information": self.prefer_recent_information,
            "information_freshness_days": self.information_freshness_days,
            "require_human_review": self.require_human_review,
            "compliance_frameworks": self.compliance_frameworks,
            "disclaimer_required": self.disclaimer_required,
            "disclaimer_text": self.disclaimer_text,
            "domain_settings": self.domain_settings
        }


@dataclass
class PreBuiltDomainKernel:
    """
    Complete pre-built domain kernel specification.
    
    This class encapsulates all components needed to deploy a specialized
    kernel for a specific domain, including metadata, system prompts,
    constraints, capabilities, and integration settings.
    """
    # Core identification
    kernel_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    domain: DomainCategory = DomainCategory.GENERAL
    
    # Metadata
    metadata: DomainKernelMetadata = None
    
    # System prompt configuration
    system_prompt: str = ""
    system_prompt_variables: List[str] = field(default_factory=list)
    chain_of_thought_enabled: bool = False
    
    # Domain-specific configuration
    config: DomainSpecificConfiguration = None
    
    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    
    # Constraints
    behavioral_constraints: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=list)
    compliance_constraints: List[str] = field(default_factory=list)
    
    # Example prompts
    example_queries: List[Dict[str, str]] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)
    
    # Integration settings
    required_integrations: List[str] = field(default_factory=list)
    optional_integrations: List[str] = field(default_factory=list)
    
    # Deployment metadata
    deployment_template: Dict[str, Any] = field(default_factory=dict)
    estimated_cost_per_1k_tokens: float = 0.01
    
    def __post_init__(self):
        """Initialize default values."""
        if self.metadata is None:
            self.metadata = DomainKernelMetadata(
                kernel_id=self.kernel_id,
                name=self.name,
                description=self.description,
                domain=self.domain,
                complexity=KernelComplexity.STANDARD,
                target_audience=KernelTargetAudience.INTERMEDIATE
            )
        
        if self.config is None:
            self.config = DomainSpecificConfiguration()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize kernel to dictionary."""
        return {
            "kernel_id": self.kernel_id,
            "name": self.name,
            "description": self.description,
            "domain": self.domain.value,
            "metadata": self.metadata.to_dict(),
            "system_prompt": {
                "template": self.system_prompt,
                "variables": self.system_prompt_variables,
                "chain_of_thought_enabled": self.chain_of_thought_enabled
            },
            "configuration": self.config.to_dict(),
            "capabilities": self.capabilities,
            "limitations": self.limitations,
            "constraints": {
                "behavioral": self.behavioral_constraints,
                "safety": self.safety_constraints,
                "compliance": self.compliance_constraints
            },
            "examples": {
                "queries": self.example_queries,
                "expected_outputs": self.expected_outputs
            },
            "integrations": {
                "required": self.required_integrations,
                "optional": self.optional_integrations
            },
            "deployment": self.deployment_template,
            "cost_estimate": {
                "per_1k_tokens": self.estimated_cost_per_1k_tokens
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PreBuiltDomainKernel':
        """Deserialize kernel from dictionary."""
        kernel = cls(
            kernel_id=data.get("kernel_id", str(uuid.uuid4())),
            name=data.get("name", ""),
            description=data.get("description", ""),
            domain=DomainCategory(data.get("domain", "general")),
            system_prompt=data.get("system_prompt", ""),
            system_prompt_variables=data.get("system_prompt_variables", []),
            chain_of_thought_enabled=data.get("chain_of_thought_enabled", False),
            capabilities=data.get("capabilities", []),
            limitations=data.get("limitations", []),
            behavioral_constraints=data.get("behavioral_constraints", []),
            safety_constraints=data.get("safety_constraints", []),
            compliance_constraints=data.get("compliance_constraints", []),
            example_queries=data.get("example_queries", []),
            expected_outputs=data.get("expected_outputs", []),
            required_integrations=data.get("required_integrations", []),
            optional_integrations=data.get("optional_integrations", []),
            deployment_template=data.get("deployment_template", {}),
            estimated_cost_per_1k_tokens=data.get("estimated_cost_per_1k_tokens", 0.01)
        )
        
        if "metadata" in data:
            m = data["metadata"]
            kernel.metadata = DomainKernelMetadata(
                kernel_id=m.get("kernel_id", kernel.kernel_id),
                name=m.get("name", kernel.name),
                description=m.get("description", kernel.description),
                domain=DomainCategory(m.get("domain", "general")),
                complexity=KernelComplexity(m.get("complexity", "standard")),
                target_audience=KernelTargetAudience(m.get("target_audience", "intermediate")),
                version=m.get("version", "1.0.0"),
                tags=m.get("tags", []),
                keywords=m.get("keywords", []),
                is_featured=m.get("is_featured", False),
                is_verified=m.get("is_verified", False),
                certification_level=m.get("certification_level", "standard")
            )
        
        return kernel


class DomainKernelCatalog:
    """
    Catalog manager for pre-built domain kernels.
    
    Provides search, filtering, and retrieval capabilities for all
    available pre-built kernels across different domains.
    """
    
    def __init__(self):
        """Initialize the kernel catalog with all pre-built kernels."""
        self._kernels: Dict[str, PreBuiltDomainKernel] = {}
        self._index_by_domain: Dict[DomainCategory, List[str]] = {}
        self._index_by_tag: Dict[str, List[str]] = {}
        self._index_by_complexity: Dict[KernelComplexity, List[str]] = {}
        
        # Initialize the catalog with all domain kernels
        self._initialize_catalog()
    
    def _initialize_catalog(self):
        """Initialize the catalog with all pre-built domain kernels."""
        kernels = self._create_all_domain_kernels()
        
        for kernel in kernels:
            self.register_kernel(kernel)
    
    def register_kernel(self, kernel: PreBuiltDomainKernel):
        """Register a kernel in the catalog."""
        self._kernels[kernel.kernel_id] = kernel
        
        # Index by domain
        if kernel.domain not in self._index_by_domain:
            self._index_by_domain[kernel.domain] = []
        self._index_by_domain[kernel.domain].append(kernel.kernel_id)
        
        # Index by tags
        for tag in kernel.metadata.tags:
            if tag not in self._index_by_tag:
                self._index_by_tag[tag] = []
            self._index_by_tag[tag].append(kernel.kernel_id)
        
        # Index by complexity
        if kernel.metadata.complexity not in self._index_by_complexity:
            self._index_by_complexity[kernel.metadata.complexity] = []
        self._index_by_complexity[kernel.metadata.complexity].append(kernel.kernel_id)
    
    def get_kernel(self, kernel_id: str) -> Optional[PreBuiltDomainKernel]:
        """Retrieve a kernel by ID."""
        return self._kernels.get(kernel_id)
    
    def search_kernels(
        self,
        query: str = None,
        domain: DomainCategory = None,
        complexity: KernelComplexity = None,
        target_audience: KernelTargetAudience = None,
        tags: List[str] = None,
        is_featured: bool = None,
        is_verified: bool = None,
        min_rating: float = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[PreBuiltDomainKernel]:
        """
        Search for kernels matching criteria.
        
        Args:
            query: Text search in name/description
            domain: Filter by domain category
            complexity: Filter by complexity level
            target_audience: Filter by target audience
            tags: Filter by tags
            is_featured: Filter featured kernels
            is_verified: Filter verified kernels
            min_rating: Minimum user rating
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of matching kernels
        """
        results = list(self._kernels.values())
        
        # Apply text search
        if query:
            query_lower = query.lower()
            results = [
                k for k in results
                if query_lower in k.name.lower() or
                   query_lower in k.description.lower() or
                   any(query_lower in tag for tag in k.metadata.tags)
            ]
        
        # Apply domain filter
        if domain:
            results = [k for k in results if k.domain == domain]
        
        # Apply complexity filter
        if complexity:
            results = [k for k in results if k.metadata.complexity == complexity]
        
        # Apply target audience filter
        if target_audience:
            results = [k for k in results if k.metadata.target_audience == target_audience]
        
        # Apply tags filter
        if tags:
            results = [
                k for k in results
                if any(tag in k.metadata.tags for tag in tags)
            ]
        
        # Apply featured filter
        if is_featured is not None:
            results = [k for k in results if k.metadata.is_featured == is_featured]
        
        # Apply verified filter
        if is_verified is not None:
            results = [k for k in results if k.metadata.is_verified == is_verified]
        
        # Apply rating filter
        if min_rating is not None:
            results = [k for k in results if k.metadata.user_rating >= min_rating]
        
        # Sort by rating and usage
        results.sort(
            key=lambda k: (k.metadata.user_rating, k.metadata.usage_count),
            reverse=True
        )
        
        return results[offset:offset + limit]
    
    def get_kernels_by_domain(self, domain: DomainCategory) -> List[PreBuiltDomainKernel]:
        """Get all kernels for a specific domain."""
        kernel_ids = self._index_by_domain.get(domain, [])
        return [self._kernels[kid] for kid in kernel_ids if kid in self._kernels]
    
    def get_featured_kernels(self, limit: int = 10) -> List[PreBuiltDomainKernel]:
        """Get featured kernels for homepage display."""
        featured = [
            k for k in self._kernels.values()
            if k.metadata.is_featured
        ]
        return sorted(featured, key=lambda k: k.metadata.user_rating, reverse=True)[:limit]
    
    def get_recommended_kernels(
        self,
        user_domain: DomainCategory = None,
        complexity: KernelComplexity = None,
        limit: int = 5
    ) -> List[PreBuiltDomainKernel]:
        """Get recommended kernels based on user preferences."""
        results = list(self._kernels.values())
        
        if user_domain:
            # Boost kernels matching user's domain
            for k in results:
                if k.domain == user_domain:
                    k.metadata.usage_count += 100
        
        if complexity:
            results = [k for k in results if k.metadata.complexity == complexity]
        
        # Sort by weighted score
        results.sort(
            key=lambda k: (
                k.metadata.user_rating * 0.4 +
                k.metadata.success_rate * 0.3 +
                (k.metadata.usage_count / 1000) * 0.3
            ),
            reverse=True
        )
        
        return results[:limit]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get catalog statistics."""
        domains = {}
        for domain, kernel_ids in self._index_by_domain.items():
            kernels = [self._kernels[kid] for kid in kernel_ids if kid in self._kernels]
            domains[domain.value] = {
                "count": len(kernels),
                "featured": sum(1 for k in kernels if k.metadata.is_featured),
                "verified": sum(1 for k in kernels if k.metadata.is_verified),
                "avg_rating": sum(k.metadata.user_rating for k in kernels) / max(1, len(kernels))
            }
        
        return {
            "total_kernels": len(self._kernels),
            "by_domain": domains,
            "total_domains": len(self._index_by_domain),
            "featured_count": sum(1 for k in self._kernels.values() if k.metadata.is_featured)
        }
    
    def _create_all_domain_kernels(self) -> List[PreBuiltDomainKernel]:
        """Create all pre-built domain kernels."""
        return [
            self._create_medical_kernel(),
            self._create_legal_kernel(),
            self._create_financial_kernel(),
            self._create_scientific_kernel(),
            self._create_customer_support_kernel(),
            self._create_code_development_kernel(),
            self._create_creative_writing_kernel(),
            self._create_document_analysis_kernel(),
            self._create_translation_kernel(),
            self._create_data_science_kernel(),
        ]
    
    def _create_medical_kernel(self) -> PreBuiltDomainKernel:
        """Create Medical/Healthcare domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-medical-001",
            name="Medical Assistant",
            description="Specialized kernel for medical information, clinical documentation, and healthcare support. Provides accurate, citation-backed medical knowledge with appropriate disclaimers.",
            domain=DomainCategory.MEDICAL,
            system_prompt="""You are a Medical Assistant AI specialized in providing healthcare information and support. You must always:

1. Base responses on current medical knowledge and clinical guidelines
2. Include citations and references for medical claims
3. Clearly state when information is not a substitute for professional medical advice
4. Recommend consulting healthcare providers for diagnosis and treatment
5. Maintain patient confidentiality and HIPAA compliance
6. Use appropriate medical terminology while remaining accessible
7. Acknowledge uncertainty and limitations in medical knowledge

When discussing medical conditions, treatments, or medications:
- Provide evidence-based information from reliable sources
- Include relevant ICD-10 codes when appropriate
- Suggest appropriate clinical guidelines (e.g., CDC, WHO, AMA)
- Flag emergency situations requiring immediate attention""",
            system_prompt_variables=["patient_context", "clinical_notes", "medication_list"],
            chain_of_thought_enabled=True,
            config=DomainSpecificConfiguration(
                response_style="professional",
                detail_level="comprehensive",
                include_references=True,
                include_confidence_scores=True,
                max_context_length=8192,
                require_human_review=True,
                compliance_frameworks=["HIPAA", "HITECH"],
                disclaimer_required=True,
                disclaimer_text="This information is for educational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.",
                domain_settings={
                    "auto_icd_coding": True,
                    "drug_interaction_check": True,
                    "clinical_guideline_refs": True
                }
            ),
            capabilities=[
                "medical_qa",
                "clinical_documentation",
                "medication_information",
                "symptom_analysis",
                "icd_coding_assistance",
                "treatment_guidelines",
                "patient_education"
            ],
            limitations=[
                "Cannot provide definitive diagnoses",
                "Cannot prescribe medications",
                "Cannot replace in-person medical examination",
                "Limited to English language queries"
            ],
            behavioral_constraints=[
                "Always recommend consulting healthcare professionals",
                "Never provide diagnosis without clear disclaimer",
                "Maintain professional medical tone",
                "Prioritize patient safety in all responses"
            ],
            safety_constraints=[
                "Flag emergency symptoms for immediate care",
                "Do not provide instructions for self-harm or harm to others",
                "Protect patient privacy in all interactions",
                "Report adverse drug reactions when identified"
            ],
            compliance_constraints=[
                "HIPAA compliance for patient data",
                "FDA guidelines for medication information",
                "State medical board regulations",
                "Telemedicine regulations when applicable"
            ],
            example_queries=[
                {"query": "What are the current treatment guidelines for Type 2 diabetes?", "type": "clinical_guidance"},
                {"query": "Explain the mechanism of action for statins", "type": "medication_info"},
                {"query": "What are warning signs of heart attack in women?", "type": "patient_education"}
            ],
            expected_outputs=[
                "Evidence-based information with citations",
                "Relevant clinical guidelines references",
                "Appropriate disclaimers",
                "Recommendation to consult healthcare provider"
            ],
            required_integrations=["medical_database", "drug_database"],
            optional_integrations=["emr_system", "telemedicine"],
            deployment_template={
                "recommended_instance": "enterprise",
                "max_concurrent_requests": 100,
                "timeout_seconds": 30,
                "memory_recommendation": "16GB"
            },
            estimated_cost_per_1k_tokens=0.015,
            metadata=DomainKernelMetadata(
                kernel_id="domain-medical-001",
                name="Medical Assistant",
                description="Healthcare support with clinical documentation and medication information",
                domain=DomainCategory.MEDICAL,
                complexity=KernelComplexity.ADVANCED,
                target_audience=KernelTargetAudience.PROFESSIONAL,
                tags=["healthcare", "medical", "clinical", "patient-support"],
                keywords=["medical", "healthcare", "clinical", "diagnosis", "treatment", "medication"],
                is_featured=True,
                is_verified=True,
                certification_level="expert",
                average_response_time_ms=2500,
                success_rate=0.97,
                user_rating=4.7
            )
        )
    
    def _create_legal_kernel(self) -> PreBuiltDomainKernel:
        """Create Legal/Corporate domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-legal-001",
            name="Legal Research Assistant",
            description="Specialized kernel for legal research, contract analysis, and corporate compliance. Provides citation-backed legal information with appropriate disclaimers.",
            domain=DomainCategory.LEGAL,
            system_prompt="""You are a Legal Research Assistant AI specialized in providing legal information and support. You must always:

1. Base responses on legal statutes, case law, and regulations
2. Include citations and references for legal claims
3. Clearly state jurisdiction and applicability of legal information
4. Recommend consulting licensed attorneys for legal advice
5. Maintain attorney-client privilege when applicable
6. Use appropriate legal terminology with clear explanations
7. Distinguish between legal information and legal advice

When discussing legal matters:
- Cite relevant statutes, regulations, and case law
- Specify applicable jurisdiction and effective dates
- Note any pending legislation or recent changes
- Distinguish between federal, state, and local laws""",
            system_prompt_variables=["jurisdiction", "case_facts", "relevant_documents"],
            chain_of_thought_enabled=True,
            config=DomainSpecificConfiguration(
                response_style="formal",
                detail_level="comprehensive",
                include_references=True,
                include_confidence_scores=True,
                max_context_length=8192,
                require_human_review=True,
                compliance_frameworks=["GDPR", "CCPA", "SOX"],
                disclaimer_required=True,
                disclaimer_text="This information is for educational purposes only and does not constitute legal advice. Consult a licensed attorney for specific legal matters.",
                domain_settings={
                    "jurisdiction_detection": True,
                    "case_law_references": True,
                    "statute_citations": True
                }
            ),
            capabilities=[
                "legal_qa",
                "contract_analysis",
                "statute_research",
                "case_law_search",
                "compliance_check",
                "jurisdictional_guidance",
                "regulatory_parsing"
            ],
            limitations=[
                "Cannot provide definitive legal advice",
                "Cannot represent clients in legal matters",
                "Cannot establish attorney-client relationship",
                "Limited to general legal information"
            ],
            behavioral_constraints=[
                "Always recommend consulting licensed attorneys",
                "Clearly distinguish information from advice",
                "Maintain professional legal tone",
                "Respect attorney-client privilege"
            ],
            safety_constraints=[
                "Do not provide instructions for illegal activities",
                "Do not help evade legal obligations",
                "Report illegal activity when required by law",
                "Protect confidential legal information"
            ],
            compliance_constraints=[
                "Attorney ethics rules",
                "Jurisdiction-specific practice of law rules",
                "Legal professional privilege requirements",
                "Data protection for legal information"
            ],
            example_queries=[
                {"query": "What are the key provisions of GDPR for US companies?", "type": "regulatory"},
                {"query": "Analyze this contract clause for potential risks", "type": "contract_review"},
                {"query": "What is the statute of limitations for breach of contract in California?", "type": "statutory_research"}
            ],
            required_integrations=["legal_database", "case_law_database"],
            optional_integrations=["document_management", "clm_system"],
            deployment_template={
                "recommended_instance": "enterprise",
                "max_concurrent_requests": 100,
                "timeout_seconds": 30,
                "memory_recommendation": "16GB"
            },
            estimated_cost_per_1k_tokens=0.015,
            metadata=DomainKernelMetadata(
                kernel_id="domain-legal-001",
                name="Legal Research Assistant",
                description="Legal research with contract analysis and compliance guidance",
                domain=DomainCategory.LEGAL,
                complexity=KernelComplexity.EXPERT,
                target_audience=KernelTargetAudience.EXPERT,
                tags=["legal", "corporate", "compliance", "research"],
                keywords=["legal", "contract", "compliance", "jurisdiction", "regulation", "case law"],
                is_featured=True,
                is_verified=True,
                certification_level="expert",
                average_response_time_ms=3000,
                success_rate=0.96,
                user_rating=4.6
            )
        )
    
    def _create_financial_kernel(self) -> PreBuiltDomainKernel:
        """Create Financial/Trading domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-financial-001",
            name="Financial Analyst",
            description="Specialized kernel for financial analysis, investment research, and trading support. Provides data-driven financial insights with appropriate risk disclosures.",
            domain=DomainCategory.FINANCIAL,
            system_prompt="""You are a Financial Analyst AI specialized in providing financial information and support. You must always:

1. Base responses on verified financial data and market information
2. Include relevant metrics, ratios, and financial indicators
3. Clearly disclose risks associated with financial decisions
4. Recommend consulting financial advisors for investment decisions
5. Maintain objectivity and avoid financial conflicts of interest
6. Use appropriate financial terminology with clear explanations
7. Distinguish between factual information and opinions/predictions

When discussing financial matters:
- Cite sources for financial data and market information
- Present both opportunities and risks clearly
- Note past performance does not guarantee future results
- Include relevant regulatory disclosures""",
            system_prompt_variables=["market_data", "financial_statements", "portfolio_context"],
            chain_of_thought_enabled=True,
            config=DomainSpecificConfiguration(
                response_style="professional",
                detail_level="comprehensive",
                include_references=True,
                include_confidence_scores=True,
                max_context_length=8192,
                require_human_review=True,
                compliance_frameworks=["SEC", "FINRA", "MiFID II"],
                disclaimer_required=True,
                disclaimer_text="Investments involve risk. Past performance does not guarantee future results. Consult a financial advisor before making investment decisions.",
                domain_settings={
                    "real_time_data": False,
                    "risk_metrics": True,
                    "regulatory_disclosures": True
                }
            ),
            capabilities=[
                "financial_qa",
                "investment_analysis",
                "portfolio_review",
                "market_research",
                "risk_assessment",
                "financial_statement_analysis",
                "regulatory_filing_review"
            ],
            limitations=[
                "Cannot provide personalized investment advice",
                "Cannot predict market movements",
                "Cannot guarantee investment returns",
                "Limited to publicly available information"
            ],
            behavioral_constraints=[
                "Always disclose risks and limitations",
                "Maintain objective, unbiased analysis",
                "Recommend consulting financial advisors",
                "Avoid market timing recommendations"
            ],
            safety_constraints=[
                "Do not recommend specific securities",
                "Do not guarantee investment returns",
                "Do not facilitate financial fraud",
                "Protect non-public financial information"
            ],
            compliance_constraints=[
                "SEC regulations for investment recommendations",
                "FINRA rules for financial advice",
                "Anti-money laundering requirements",
                "Insider trading prohibitions"
            ],
            example_queries=[
                {"query": "Analyze the financial health of Company X based on their latest 10-K", "type": "financial_analysis"},
                {"query": "What are the key metrics for evaluating a technology company?", "type": "investment_research"},
                {"query": "Explain the risks of high-frequency trading strategies", "type": "risk_assessment"}
            ],
            required_integrations=["financial_data_api"],
            optional_integrations=["trading_platform", "portfolio_manager"],
            deployment_template={
                "recommended_instance": "enterprise",
                "max_concurrent_requests": 150,
                "timeout_seconds": 25,
                "memory_recommendation": "16GB"
            },
            estimated_cost_per_1k_tokens=0.018,
            metadata=DomainKernelMetadata(
                kernel_id="domain-financial-001",
                name="Financial Analyst",
                description="Investment analysis with portfolio review and market research",
                domain=DomainCategory.FINANCIAL,
                complexity=KernelComplexity.ADVANCED,
                target_audience=KernelTargetAudience.PROFESSIONAL,
                tags=["finance", "investment", "trading", "analysis"],
                keywords=["financial", "investment", "portfolio", "trading", "market", "analysis"],
                is_featured=True,
                is_verified=True,
                certification_level="expert",
                average_response_time_ms=2000,
                success_rate=0.97,
                user_rating=4.8
            )
        )
    
    def _create_scientific_kernel(self) -> PreBuiltDomainKernel:
        """Create Scientific Research domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-scientific-001",
            name="Scientific Research Assistant",
            description="Specialized kernel for scientific research, literature review, and technical analysis. Provides citation-backed scientific information.",
            domain=DomainCategory.SCIENTIFIC,
            system_prompt="""You are a Scientific Research Assistant AI specialized in providing scientific information and support. You must always:

1. Base responses on peer-reviewed research and scientific literature
2. Include citations and references for scientific claims
3. Clearly distinguish between established science and hypotheses
4. Acknowledge scientific uncertainty and ongoing research
5. Use appropriate scientific terminology with clear explanations
6. Follow scientific methodology principles
7. Maintain objectivity and avoid scientific bias

When discussing scientific topics:
- Cite peer-reviewed sources and publications
- Note study limitations and confidence intervals
- Distinguish correlation from causation
- Present information with appropriate certainty levels""",
            system_prompt_variables=["research_topic", "literature_context", "methodology_notes"],
            chain_of_thought_enabled=True,
            config=DomainSpecificConfiguration(
                response_style="technical",
                detail_level="comprehensive",
                include_references=True,
                include_confidence_scores=True,
                max_context_length=8192,
                require_human_review=False,
                compliance_frameworks=[],
                disclaimer_required=False,
                domain_settings={
                    "peer_review_check": True,
                    "methodology_analysis": True,
                    "statistical_significance": True
                }
            ),
            capabilities=[
                "scientific_qa",
                "literature_review",
                "methodology_analysis",
                "hypothesis_evaluation",
                "data_interpretation",
                "reference_management",
                "technical_writing_assistance"
            ],
            limitations=[
                "Cannot access real-time experimental data",
                "Cannot perform actual scientific experiments",
                "Cannot provide definitive scientific consensus",
                "Limited to published literature"
            ],
            behavioral_constraints=[
                "Present scientific information objectively",
                "Acknowledge uncertainty and limitations",
                "Follow scientific ethics principles",
                "Maintain research integrity"
            ],
            safety_constraints=[
                "Do not provide instructions for dangerous experiments",
                "Do not facilitate scientific misconduct",
                "Report research ethics violations",
                "Protect confidential research data"
            ],
            compliance_constraints=[
                "Research ethics requirements",
                "Peer review standards",
                "Scientific integrity guidelines",
                "Data sharing policies"
            ],
            example_queries=[
                {"query": "What does the latest research say about CRISPR gene editing for genetic disorders?", "type": "literature_review"},
                {"query": "Explain the methodology used in this clinical trial", "type": "methodology_analysis"},
                {"query": "What are the current theories about dark matter?", "type": "scientific_research"}
            ],
            required_integrations=["scientific_database", "pubmed"],
            optional_integrations=["research_management", "citation_manager"],
            deployment_template={
                "recommended_instance": "standard",
                "max_concurrent_requests": 80,
                "timeout_seconds": 35,
                "memory_recommendation": "12GB"
            },
            estimated_cost_per_1k_tokens=0.012,
            metadata=DomainKernelMetadata(
                kernel_id="domain-scientific-001",
                name="Scientific Research Assistant",
                description="Research support with literature review and methodology analysis",
                domain=DomainCategory.SCIENTIFIC,
                complexity=KernelComplexity.ADVANCED,
                target_audience=KernelTargetAudience.PROFESSIONAL,
                tags=["science", "research", "academic", "technical"],
                keywords=["research", "scientific", "methodology", "analysis", "literature"],
                is_featured=False,
                is_verified=True,
                certification_level="expert",
                average_response_time_ms=2800,
                success_rate=0.98,
                user_rating=4.7
            )
        )
    
    def _create_customer_support_kernel(self) -> PreBuiltDomainKernel:
        """Create Customer Support domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-support-001",
            name="Customer Support Agent",
            description="Specialized kernel for customer service, support ticket handling, and customer communication. Provides empathetic, efficient customer support.",
            domain=DomainCategory.CUSTOMER_SUPPORT,
            system_prompt="""You are a Customer Support AI specialized in providing customer service and support. You must always:

1. Respond with empathy and understanding to customer concerns
2. Use clear, friendly, and professional language
3. Focus on resolving customer issues efficiently
4. Escalate complex issues to human agents when appropriate
5. Maintain consistent service quality across interactions
6. Follow company support protocols and procedures
7. Document interactions accurately for follow-up

When handling support requests:
- Acknowledge the customer's issue and feelings
- Provide clear steps for resolution when possible
- Set realistic expectations for resolution times
- Offer additional help and follow-up options""",
            system_prompt_variables=["customer_history", "product_context", "issue_details"],
            chain_of_thought_enabled=False,
            config=DomainSpecificConfiguration(
                response_style="friendly",
                detail_level="standard",
                include_references=False,
                include_confidence_scores=False,
                max_context_length=4096,
                require_human_review=False,
                compliance_frameworks=["SOC 2"],
                disclaimer_required=False,
                domain_settings={
                    "escalation_triggers": True,
                    "sentiment_analysis": True,
                    "sla_tracking": True
                }
            ),
            capabilities=[
                "customer_qa",
                "ticket_routing",
                "troubleshooting",
                "faq_responses",
                "complaint_resolution",
                "product_information",
                "order_status_inquiry"
            ],
            limitations=[
                "Cannot access customer-specific data without authentication",
                "Cannot process refunds or financial transactions",
                "Cannot override company policies",
                "Cannot access real-time inventory"
            ],
            behavioral_constraints=[
                "Maintain empathetic and patient tone",
                "Focus on customer satisfaction",
                "Escalate appropriately when needed",
                "Follow company brand guidelines"
            ],
            safety_constraints=[
                "Do not share confidential company information",
                "Do not make promises beyond scope",
                "Do not engage with abusive customers",
                "Protect customer privacy in all interactions"
            ],
            compliance_constraints=[
                "Data protection for customer information",
                "PCI DSS for payment information",
                "Customer service quality standards",
                "Accessibility requirements"
            ],
            example_queries=[
                {"query": "How do I reset my password?", "type": "faq"},
                {"query": "My order hasn't arrived yet. Can you check the status?", "type": "order_inquiry"},
                {"query": "I'm having trouble with the mobile app. Can you help?", "type": "troubleshooting"}
            ],
            required_integrations=["crm_system", "knowledge_base"],
            optional_integrations=["ticketing_system", "order_management"],
            deployment_template={
                "recommended_instance": "standard",
                "max_concurrent_requests": 500,
                "timeout_seconds": 15,
                "memory_recommendation": "8GB"
            },
            estimated_cost_per_1k_tokens=0.008,
            metadata=DomainKernelMetadata(
                kernel_id="domain-support-001",
                name="Customer Support Agent",
                description="Customer service with ticket handling and troubleshooting",
                domain=DomainCategory.CUSTOMER_SUPPORT,
                complexity=KernelComplexity.LIGHT,
                target_audience=KernelTargetAudience.BEGINNER,
                tags=["support", "customer service", "helpdesk", "service"],
                keywords=["support", "customer", "help", "service", "ticket", "resolution"],
                is_featured=True,
                is_verified=True,
                certification_level="standard",
                average_response_time_ms=1500,
                success_rate=0.95,
                user_rating=4.5
            )
        )
    
    def _create_code_development_kernel(self) -> PreBuiltDomainKernel:
        """Create Code Development domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-code-001",
            name="Code Development Assistant",
            description="Specialized kernel for software development, code generation, and technical documentation. Provides programming assistance with best practices.",
            domain=DomainCategory.CODE_DEVELOPMENT,
            system_prompt="""You are a Code Development Assistant AI specialized in providing programming support. You must always:

1. Provide accurate, working code solutions
2. Follow best practices and coding standards
3. Include appropriate comments and documentation
4. Consider security implications in code
5. Explain code functionality clearly
6. Suggest optimizations and improvements
7. Reference official documentation when relevant

When providing code:
- Write clean, readable, and maintainable code
- Include error handling and edge cases
- Consider performance and scalability
- Follow language-specific conventions
- Provide usage examples and test cases""",
            system_prompt_variables=["codebase_context", "tech_stack", "requirements"],
            chain_of_thought_enabled=True,
            config=DomainSpecificConfiguration(
                response_style="technical",
                detail_level="comprehensive",
                include_references=True,
                include_confidence_scores=True,
                max_context_length=8192,
                require_human_review=False,
                compliance_frameworks=[],
                disclaimer_required=False,
                domain_settings={
                    "security_scan": True,
                    "performance_tips": True,
                    "documentation_generation": True
                }
            ),
            capabilities=[
                "code_generation",
                "code_review",
                "debugging_assistance",
                "architecture_design",
                "documentation",
                "api_integration",
                "testing_guidance"
            ],
            limitations=[
                "Cannot execute code directly",
                "Cannot access private repositories without authorization",
                "Cannot guarantee code works in all environments",
                "Limited to knowledge cutoff date"
            ],
            behavioral_constraints=[
                "Provide accurate and helpful code",
                "Consider edge cases and error handling",
                "Follow security best practices",
                "Suggest code improvements when relevant"
            ],
            safety_constraints=[
                "Do not generate malicious code",
                "Do not help with security exploits",
                "Do not bypass access controls",
                "Protect intellectual property"
            ],
            compliance_constraints=[
                "Open source license compliance",
                "Security coding standards",
                "Accessibility requirements",
                "Privacy by design principles"
            ],
            example_queries=[
                {"query": "Write a Python function to process a CSV file and return aggregated statistics", "type": "code_generation"},
                {"query": "Review this code snippet for potential security vulnerabilities", "type": "code_review"},
                {"query": "Design a microservices architecture for an e-commerce platform", "type": "architecture_design"}
            ],
            required_integrations=["code_repository"],
            optional_integrations=["ci_cd_pipeline", "testing_framework"],
            deployment_template={
                "recommended_instance": "standard",
                "max_concurrent_requests": 200,
                "timeout_seconds": 25,
                "memory_recommendation": "12GB"
            },
            estimated_cost_per_1k_tokens=0.012,
            metadata=DomainKernelMetadata(
                kernel_id="domain-code-001",
                name="Code Development Assistant",
                description="Programming support with code generation and review",
                domain=DomainCategory.CODE_DEVELOPMENT,
                complexity=KernelComplexity.ADVANCED,
                target_audience=KernelTargetAudience.INTERMEDIATE,
                tags=["development", "programming", "coding", "software"],
                keywords=["code", "programming", "development", "software", "engineering", "api"],
                is_featured=True,
                is_verified=True,
                certification_level="expert",
                average_response_time_ms=2000,
                success_rate=0.96,
                user_rating=4.8
            )
        )
    
    def _create_creative_writing_kernel(self) -> PreBuiltDomainKernel:
        """Create Creative Writing domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-creative-001",
            name="Creative Writing Assistant",
            description="Specialized kernel for creative writing, content creation, and storytelling. Provides imaginative, engaging creative content.",
            domain=DomainCategory.CREATIVE_WRITING,
            system_prompt="""You are a Creative Writing Assistant AI specialized in providing creative content and storytelling support. You must always:

1. Create engaging and original creative content
2. Adapt writing style to match requested genres and tones
3. Develop compelling characters, plots, and narratives
4. Use vivid language and descriptive imagery
5. Maintain consistency in fictional worlds
6. Respect creative direction and constraints
7. Encourage and support creative exploration

When providing creative content:
- Match the requested style and tone
- Develop original ideas and perspectives
- Create engaging narratives and characters
- Use language creatively and effectively""",
            system_prompt_variables=["genre", "tone", "target_audience", "creative_constraints"],
            chain_of_thought_enabled=False,
            config=DomainSpecificConfiguration(
                response_style="creative",
                detail_level="comprehensive",
                include_references=False,
                include_confidence_scores=False,
                max_context_length=8192,
                require_human_review=False,
                compliance_frameworks=[],
                disclaimer_required=False,
                domain_settings={
                    "genre_adaptation": True,
                    "tone_consistency": True,
                    "character_development": True
                }
            ),
            capabilities=[
                "story_generation",
                "content_creation",
                "dialogue_writing",
                "world_building",
                "editing_assistance",
                "brainstorming",
                "content_adaptation"
            ],
            limitations=[
                "Cannot replicate copyrighted material",
                "Cannot guarantee publication-quality output",
                "Cannot replace human creativity entirely",
                "Limited to language model knowledge"
            ],
            behavioral_constraints=[
                "Respect intellectual property rights",
                "Support creative exploration",
                "Adapt to creative feedback",
                "Maintain originality in suggestions"
            ],
            safety_constraints=[
                "Do not generate harmful content",
                "Do not create content promoting illegal activities",
                "Do not generate explicit content unless requested",
                "Do not plagiarize existing works"
            ],
            compliance_constraints=[
                "Copyright and fair use considerations",
                "Content moderation policies",
                "Brand safety guidelines",
                "Accessibility for generated content"
            ],
            example_queries=[
                {"query": "Write a short story about a time traveler who accidentally changes history", "type": "story_generation"},
                {"query": "Create marketing copy for a new sustainable fashion brand", "type": "content_creation"},
                {"query": "Develop dialogue between two characters meeting after 20 years", "type": "dialogue_writing"}
            ],
            required_integrations=[],
            optional_integrations=["content_management", "publishing_platform"],
            deployment_template={
                "recommended_instance": "standard",
                "max_concurrent_requests": 100,
                "timeout_seconds": 20,
                "memory_recommendation": "8GB"
            },
            estimated_cost_per_1k_tokens=0.010,
            metadata=DomainKernelMetadata(
                kernel_id="domain-creative-001",
                name="Creative Writing Assistant",
                description="Creative content with story generation and content creation",
                domain=DomainCategory.CREATIVE_WRITING,
                complexity=KernelComplexity.STANDARD,
                target_audience=KernelTargetAudience.BEGINNER,
                tags=["creative", "writing", "content", "storytelling"],
                keywords=["creative", "writing", "story", "content", "marketing", "copywriting"],
                is_featured=False,
                is_verified=False,
                certification_level="standard",
                average_response_time_ms=2500,
                success_rate=0.94,
                user_rating=4.4
            )
        )
    
    def _create_document_analysis_kernel(self) -> PreBuiltDomainKernel:
        """Create Document Analysis domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-document-001",
            name="Document Analysis Assistant",
            description="Specialized kernel for document processing, information extraction, and content summarization. Provides efficient document understanding.",
            domain=DomainCategory.DOCUMENT_ANALYSIS,
            system_prompt="""You are a Document Analysis Assistant AI specialized in processing and analyzing documents. You must always:

1. Extract key information accurately from documents
2. Identify document structure and organization
3. Summarize content concisely and accurately
4. Flag important sections and key findings
5. Maintain objectivity in analysis
6. Respect document confidentiality
7. Provide structured outputs when appropriate

When analyzing documents:
- Identify document type and purpose
- Extract relevant information and key points
- Summarize without losing important details
- Note any inconsistencies or gaps""",
            system_prompt_variables=["document_type", "analysis_goals", "extraction_targets"],
            chain_of_thought_enabled=True,
            config=DomainSpecificConfiguration(
                response_style="professional",
                detail_level="standard",
                include_references=False,
                include_confidence_scores=True,
                max_context_length=16384,
                require_human_review=False,
                compliance_frameworks=[],
                disclaimer_required=False,
                domain_settings={
                    "entity_extraction": True,
                    "key_point_summary": True,
                    "sentiment_analysis": True,
                    "document_classification": True
                }
            ),
            capabilities=[
                "document_summarization",
                "information_extraction",
                "entity_recognition",
                "key_point_identification",
                "document_classification",
                "comparison_analysis",
                "trend_detection"
            ],
            limitations=[
                "Cannot access password-protected documents",
                "Cannot verify accuracy of document claims",
                "Cannot process heavily formatted documents well",
                "Limited by context window size"
            ],
            behavioral_constraints=[
                "Maintain objectivity in analysis",
                "Respect document confidentiality",
                "Provide accurate extractions",
                "Acknowledge analysis limitations"
            ],
            safety_constraints=[
                "Do not share confidential document content",
                "Do not modify document originals",
                "Do not access unauthorized documents",
                "Protect sensitive information in analysis"
            ],
            compliance_constraints=[
                "Data protection for confidential documents",
                "Privacy requirements for personal data",
                "Document retention policies",
                "Access control requirements"
            ],
            example_queries=[
                {"query": "Summarize this 50-page research report into key findings", "type": "summarization"},
                {"query": "Extract all mentioned entities and dates from this contract", "type": "entity_extraction"},
                {"query": "Compare the key differences between these three policy documents", "type": "comparison"}
            ],
            required_integrations=["document_storage"],
            optional_integrations=["ocr_system", "extraction_engine"],
            deployment_template={
                "recommended_instance": "enterprise",
                "max_concurrent_requests": 150,
                "timeout_seconds": 30,
                "memory_recommendation": "16GB"
            },
            estimated_cost_per_1k_tokens=0.010,
            metadata=DomainKernelMetadata(
                kernel_id="domain-document-001",
                name="Document Analysis Assistant",
                description="Document processing with summarization and information extraction",
                domain=DomainCategory.DOCUMENT_ANALYSIS,
                complexity=KernelComplexity.STANDARD,
                target_audience=KernelTargetAudience.INTERMEDIATE,
                tags=["document", "analysis", "extraction", "summarization"],
                keywords=["document", "analysis", "summary", "extraction", "processing"],
                is_featured=False,
                is_verified=True,
                certification_level="standard",
                average_response_time_ms=3000,
                success_rate=0.95,
                user_rating=4.5
            )
        )
    
    def _create_translation_kernel(self) -> PreBuiltDomainKernel:
        """Create Translation/Localization domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-translation-001",
            name="Translation and Localization Assistant",
            description="Specialized kernel for language translation, localization, and cross-cultural communication. Provides accurate, context-aware translations.",
            domain=DomainCategory.TRANSLATION,
            system_prompt="""You are a Translation and Localization Assistant AI specialized in providing language services. You must always:

1. Provide accurate translations that preserve meaning
2. Consider cultural context and nuances
3. Adapt content for target audiences
4. Maintain appropriate tone and style
5. Preserve technical terminology accuracy
6. Handle idioms and cultural references appropriately
7. Follow localization best practices

When translating or localizing:
- Preserve original meaning and intent
- Adapt cultural references for target audience
- Maintain consistent terminology
- Consider formal/informal register
- Preserve formatting and structure when possible""",
            system_prompt_variables=["source_language", "target_language", "content_type", "cultural_context"],
            chain_of_thought_enabled=False,
            config=DomainSpecificConfiguration(
                response_style="professional",
                detail_level="comprehensive",
                include_references=False,
                include_confidence_scores=True,
                max_context_length=8192,
                require_human_review=False,
                compliance_frameworks=[],
                disclaimer_required=False,
                domain_settings={
                    "preserve_formatting": True,
                    "cultural_adaptation": True,
                    "terminology_consistency": True
                }
            ),
            capabilities=[
                "language_translation",
                "cultural_adaptation",
                "localization",
                "transcreation",
                "multilingual_qa",
                "glossary_management",
                "quality_assurance"
            ],
            limitations=[
                "Cannot guarantee human-level nuance",
                "Cannot handle highly specialized technical translations",
                "Cannot replace professional translators for critical documents",
                "Limited to supported language pairs"
            ],
            behavioral_constraints=[
                "Provide accurate translations",
                "Acknowledge translation limitations",
                "Suggest human review for critical content",
                "Maintain terminology consistency"
            ],
            safety_constraints=[
                "Do not translate harmful or illegal content",
                "Do not facilitate communication in illegal activities",
                "Protect confidential translation content",
                "Respect language access requirements"
            ],
            compliance_constraints=[
                "Localization industry standards",
                "Quality assurance requirements",
                "Privacy for translated content",
                "Accessibility in multiple languages"
            ],
            example_queries=[
                {"query": "Translate this product description from English to Spanish", "type": "translation"},
                {"query": "Adapt this marketing campaign for a Japanese audience", "type": "localization"},
                {"query": "Translate and localize this user interface for French-Canadian users", "type": "ui_localization"}
            ],
            required_integrations=["translation_memory"],
            optional_integrations=["terminology_database", "quality_check"],
            deployment_template={
                "recommended_instance": "standard",
                "max_concurrent_requests": 200,
                "timeout_seconds": 20,
                "memory_recommendation": "8GB"
            },
            estimated_cost_per_1k_tokens=0.010,
            metadata=DomainKernelMetadata(
                kernel_id="domain-translation-001",
                name="Translation and Localization Assistant",
                description="Multilingual support with translation and localization",
                domain=DomainCategory.TRANSLATION,
                complexity=KernelComplexity.STANDARD,
                target_audience=KernelTargetAudience.INTERMEDIATE,
                tags=["translation", "localization", "multilingual", "language"],
                keywords=["translation", "localization", "language", "multilingual", "internationalization"],
                is_featured=False,
                is_verified=False,
                certification_level="standard",
                average_response_time_ms=2000,
                success_rate=0.94,
                user_rating=4.3
            )
        )
    
    def _create_data_science_kernel(self) -> PreBuiltDomainKernel:
        """Create Data Science domain kernel."""
        return PreBuiltDomainKernel(
            kernel_id="domain-datascience-001",
            name="Data Science Assistant",
            description="Specialized kernel for data analysis, statistical modeling, and machine learning. Provides data-driven insights and technical guidance.",
            domain=DomainCategory.DATA_SCIENCE,
            system_prompt="""You are a Data Science Assistant AI specialized in providing data analysis and machine learning support. You must always:

1. Base recommendations on statistical principles and best practices
2. Explain statistical concepts clearly and accurately
3. Consider data quality and preprocessing requirements
4. Recommend appropriate analytical methods
5. Interpret results with appropriate caveats
6. Suggest validation and testing approaches
7. Follow reproducible research practices

When providing data science guidance:
- Consider data characteristics and limitations
- Suggest appropriate statistical methods
- Explain model assumptions and requirements
- Interpret results with uncertainty quantification
- Recommend validation strategies""",
            system_prompt_variables=["data_description", "analysis_objectives", "constraints"],
            chain_of_thought_enabled=True,
            config=DomainSpecificConfiguration(
                response_style="technical",
                detail_level="comprehensive",
                include_references=True,
                include_confidence_scores=True,
                max_context_length=8192,
                require_human_review=False,
                compliance_frameworks=[],
                disclaimer_required=False,
                domain_settings={
                    "statistical_significance": True,
                    "model_interpretability": True,
                    "data_quality_checks": True
                }
            ),
            capabilities=[
                "statistical_analysis",
                "machine_learning_guidance",
                "data_visualization",
                "feature_engineering",
                "model_selection",
                "a_b_testing",
                "time_series_analysis"
            ],
            limitations=[
                "Cannot access or process actual datasets directly",
                "Cannot guarantee model performance without testing",
                "Cannot replace domain expertise",
                "Limited to general statistical knowledge"
            ],
            behavioral_constraints=[
                "Provide statistically sound recommendations",
                "Acknowledge uncertainty in predictions",
                "Recommend validation and testing",
                "Follow ethical AI practices"
            ],
            safety_constraints=[
                "Do not facilitate discriminatory algorithms",
                "Do not recommend unethical data practices",
                "Do not bypass data privacy requirements",
                "Protect sensitive data in examples"
            ],
            compliance_constraints=[
                "Data privacy regulations",
                "Algorithmic accountability",
                "Fairness and bias considerations",
                "Reproducibility requirements"
            ],
            example_queries=[
                {"query": "What machine learning approach would you recommend for predicting customer churn?", "type": "model_selection"},
                {"query": "Explain how to interpret feature importance in random forest models", "type": "model_interpretation"},
                {"query": "Design an A/B test for comparing two website layouts", "type": "experimental_design"}
            ],
            required_integrations=["data_platform"],
            optional_integrations=["ml_platform", "visualization_library"],
            deployment_template={
                "recommended_instance": "standard",
                "max_concurrent_requests": 100,
                "timeout_seconds": 30,
                "memory_recommendation": "12GB"
            },
            estimated_cost_per_1k_tokens=0.014,
            metadata=DomainKernelMetadata(
                kernel_id="domain-datascience-001",
                name="Data Science Assistant",
                description="Analytics support with machine learning and statistical guidance",
                domain=DomainCategory.DATA_SCIENCE,
                complexity=KernelComplexity.ADVANCED,
                target_audience=KernelTargetAudience.PROFESSIONAL,
                tags=["data science", "machine learning", "analytics", "statistics"],
                keywords=["data", "machine learning", "analytics", "statistics", "modeling", "ai"],
                is_featured=True,
                is_verified=True,
                certification_level="expert",
                average_response_time_ms=2500,
                success_rate=0.97,
                user_rating=4.7
            )
        )


# Convenience function to get the kernel catalog
def get_domain_kernel_catalog() -> DomainKernelCatalog:
    """Get the pre-built domain kernel catalog."""
    return DomainKernelCatalog()


# Function to get a specific kernel
def get_domain_kernel(kernel_id: str) -> Optional[PreBuiltDomainKernel]:
    """Retrieve a specific domain kernel by ID."""
    catalog = get_domain_kernel_catalog()
    return catalog.get_kernel(kernel_id)


# Function to list all kernel IDs
def list_domain_kernel_ids() -> List[str]:
    """List all available domain kernel IDs."""
    catalog = get_domain_kernel_catalog()
    return list(catalog._kernels.keys())


# Export for external use
__all__ = [
    "DomainCategory",
    "KernelComplexity",
    "KernelTargetAudience",
    "PreBuiltDomainKernel",
    "DomainKernelCatalog",
    "DomainKernelMetadata",
    "DomainSpecificConfiguration",
    "get_domain_kernel_catalog",
    "get_domain_kernel",
    "list_domain_kernel_ids"
]


# Example usage and testing
if __name__ == "__main__":
    # Initialize the catalog
    catalog = get_domain_kernel_catalog()
    
    # Print statistics
    stats = catalog.get_statistics()
    print("Domain Kernel Catalog Statistics:")
    print(f"Total Kernels: {stats['total_kernels']}")
    print(f"Total Domains: {stats['total_domains']}")
    print(f"Featured Kernels: {stats['featured_count']}")
    print()
    
    # Print kernels by domain
    print("Kernels by Domain:")
    for domain, info in stats["by_domain"].items():
        print(f"  {domain}: {info['count']} kernels (rating: {info['avg_rating']:.2f})")
    print()
    
    # Get featured kernels
    featured = catalog.get_featured_kernels(3)
    print("Featured Kernels:")
    for kernel in featured:
        print(f"  - {kernel.name} ({kernel.domain.value}) - Rating: {kernel.metadata.user_rating}")
    print()
    
    # Search for code development kernel
    results = catalog.search_kernels(domain=DomainCategory.CODE_DEVELOPMENT)
    print(f"Code Development Kernels: {len(results)}")
    for k in results:
        print(f"  - {k.name}: {k.description[:100]}...")
