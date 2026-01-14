"""
Kernel Marketplace Infrastructure

This module provides the complete marketplace infrastructure for buying,
selling, and distributing kernels. It includes listing management, purchase
processing, licensing mechanisms, and seller dashboards.

Key Components:
- Listing Management: Create, update, and manage marketplace listings
- Purchase Processing: Handle transactions and payments
- Licensing System: Manage usage licenses and entitlements
- Seller Dashboard: Analytics and seller tools
- Review System: User reviews and ratings
- Category Management: Organize kernels by category
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PricingModel(Enum):
    """
    Pricing models available for marketplace listings.
    
    Different models support different business models from free kernels
    to subscription-based services.
    """
    FREE = "free"
    CREDIT_BASED = "credit_based"
    SUBSCRIPTION = "subscription"
    ONE_TIME = "one_time"
    NEGOTIATED = "negotiated"
    
    @classmethod
    def get_display_name(cls, model: 'PricingModel') -> str:
        """Get human-readable display name."""
        names = {
            cls.FREE: "Free",
            cls.CREDIT_BASED: "Credits",
            cls.SUBSCRIPTION: "Subscription",
            cls.ONE_TIME: "One-time Purchase",
            cls.NEGOTIATED: "Contact Seller"
        }
        return names.get(model, model.value)


class ListingStatus(Enum):
    """
    Status of a marketplace listing.
    
    Listings go through a lifecycle from draft to active to archived.
    """
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"
    SOLD_OUT = "sold_out"


class PurchaseStatus(Enum):
    """
    Status of a purchase transaction.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"


class LicenseType(Enum):
    """
    Types of licenses available for kernels.
    
    Defines what rights the purchaser has after buying a kernel.
    """
    PERSONAL = "personal"  # Single user, personal use only
    COMMERCIAL = "commercial"  # Commercial use allowed
    ENTERPRISE = "enterprise"  # Multi-user, enterprise features
    EXTENDED = "extended"  # Full rights, including redistribution
    EVALUATION = "evaluation"  # Time-limited evaluation license
    
    @classmethod
    def get_default_price(cls, license_type: 'LicenseType', base_price: float) -> float:
        """Get default price multiplier for license type."""
        multipliers = {
            cls.PERSONAL: 1.0,
            cls.COMMERCIAL: 2.5,
            cls.ENTERPRISE: 5.0,
            cls.EXTENDED: 10.0,
            cls.EVALUATION: 0.3
        }
        return base_price * multipliers.get(license_type, 1.0)


class KernelCategory(Enum):
    """
    Categories for organizing marketplace listings.
    
    Categories help buyers discover relevant kernels and enable
    category-specific browsing and filtering.
    """
    REASONING = "reasoning"
    EXTRACTION = "extraction"
    GENERATION = "generation"
    CLASSIFICATION = "classification"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    QUESTION_ANSWERING = "question_answering"
    CODE_GENERATION = "code_generation"
    DATA_ANALYSIS = "data_analysis"
    CREATIVE_WRITING = "creative_writing"
    CUSTOM = "custom"
    
    @classmethod
    def get_display_name(cls, category: 'KernelCategory') -> str:
        """Get human-readable display name."""
        names = {
            cls.REASONING: "Reasoning & Logic",
            cls.EXTRACTION: "Data Extraction",
            cls.GENERATION: "Content Generation",
            cls.CLASSIFICATION: "Classification",
            cls.TRANSLATION: "Translation",
            cls.SUMMARIZATION: "Summarization",
            cls.QUESTION_ANSWERING: "Question Answering",
            cls.CODE_GENERATION: "Code Generation",
            cls.DATA_ANALYSIS: "Data Analysis",
            cls.CREATIVE_WRITING: "Creative Writing",
            cls.CUSTOM: "Custom Solutions"
        }
        return names.get(category, category.value)


@dataclass
class License:
    """
    Usage license for a purchased kernel.
    
    Defines the rights and restrictions for using a purchased kernel.
    """
    license_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kernel_id: str = ""
    listing_id: str = ""
    purchaser_id: str = ""
    license_type: LicenseType = LicenseType.PERSONAL
    purchase_id: str = ""
    issued_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    max_users: int = 1
    max_queries: Optional[int] = None
    current_queries: int = 0
    features: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    is_active: bool = True
    transfer_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if license is currently valid."""
        if not self.is_active:
            return False
        if self.expires_at and datetime.now() > self.expires_at:
            return False
        if self.max_queries is not None and self.current_queries >= self.max_queries:
            return False
        return True
    
    def can_use(self, user_id: str = None) -> Tuple[bool, str]:
        """
        Check if license can be used.
        
        Returns:
            Tuple of (can_use, reason)
        """
        if not self.is_valid():
            return False, "License is not valid"
        
        if self.max_users > 0:
            # Check concurrent usage (simplified)
            pass
        
        return True, ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize license to dictionary."""
        return {
            "license_id": self.license_id,
            "kernel_id": self.kernel_id,
            "listing_id": self.listing_id,
            "purchaser_id": self.purchaser_id,
            "license_type": self.license_type.value,
            "purchase_id": self.purchase_id,
            "issued_at": self.issued_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "max_users": self.max_users,
            "max_queries": self.max_queries,
            "current_queries": self.current_queries,
            "features": self.features,
            "restrictions": self.restrictions,
            "is_active": self.is_active,
            "transfer_count": self.transfer_count,
            "metadata": self.metadata
        }


@dataclass
class Listing:
    """
    Marketplace listing for a kernel.
    
    Contains all information about a kernel offered in the marketplace,
    including pricing, description, and metadata.
    """
    listing_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    kernel_id: str = ""
    seller_id: str = ""
    title: str = ""
    description: str = ""
    short_description: str = ""
    category: KernelCategory = KernelCategory.CUSTOM
    pricing_model: PricingModel = PricingModel.FREE
    base_price: float = 0.0
    currency: str = "USD"
    
    # Subscription details
    subscription_period: str = "monthly"  # monthly, yearly
    subscription_price: float = 0.0
    
    # Credits model
    credits_per_query: int = 1
    credits_pack_price: float = 0.0
    credits_pack_size: int = 1000
    
    # License options
    available_licenses: List[LicenseType] = field(
        default_factory=lambda: [LicenseType.PERSONAL]
    )
    
    # Media and documentation
    screenshots: List[str] = field(default_factory=list)
    demo_url: Optional[str] = None
    documentation_url: Optional[str] = None
    video_url: Optional[str] = None
    
    # Search and discovery
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    
    # Status and moderation
    status: ListingStatus = ListingStatus.DRAFT
    rejection_reason: Optional[str] = None
    moderation_notes: str = ""
    
    # Statistics
    view_count: int = 0
    download_count: int = 0
    purchase_count: int = 0
    average_rating: float = 0.0
    review_count: int = 0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    published_at: Optional[datetime] = None
    featured_until: Optional[datetime] = None
    
    # Featured and promoted
    is_featured: bool = False
    is_promoted: bool = False
    promotion_level: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize listing to dictionary."""
        return {
            "listing_id": self.listing_id,
            "kernel_id": self.kernel_id,
            "seller_id": self.seller_id,
            "title": self.title,
            "description": self.description,
            "short_description": self.short_description,
            "category": self.category.value,
            "pricing_model": self.pricing_model.value,
            "base_price": self.base_price,
            "currency": self.currency,
            "subscription_details": {
                "period": self.subscription_period,
                "price": self.subscription_price
            },
            "credits_details": {
                "per_query": self.credits_per_query,
                "pack_price": self.credits_pack_price,
                "pack_size": self.credits_pack_size
            },
            "available_licenses": [l.value for l in self.available_licenses],
            "media": {
                "screenshots": self.screenshots,
                "demo_url": self.demo_url,
                "documentation_url": self.documentation_url,
                "video_url": self.video_url
            },
            "discovery": {
                "tags": self.tags,
                "keywords": self.keywords,
                "capabilities": self.capabilities
            },
            "status": self.status.value,
            "stats": {
                "views": self.view_count,
                "downloads": self.download_count,
                "purchases": self.purchase_count,
                "rating": self.average_rating,
                "reviews": self.review_count
            },
            "featured": self.is_featured,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    def get_price_for_license(self, license_type: LicenseType) -> float:
        """Get price for a specific license type."""
        return LicenseType.get_default_price(license_type, self.base_price)


@dataclass
class Purchase:
    """
    Record of a marketplace purchase.
    
    Tracks the complete purchase transaction including payment,
    delivery, and any subsequent actions like refunds.
    """
    purchase_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    listing_id: str = ""
    kernel_id: str = ""
    buyer_id: str = ""
    seller_id: str = ""
    
    # Transaction details
    status: PurchaseStatus = PurchaseStatus.PENDING
    license_type: LicenseType = LicenseType.PERSONAL
    
    # Pricing
    price: float = 0.0
    currency: str = "USD"
    discount: float = 0.0
    final_price: float = 0.0
    
    # Payment info
    payment_method: str = ""
    payment_reference: Optional[str] = None
    transaction_id: Optional[str] = None
    
    # Delivery
    license_id: Optional[str] = None
    download_url: Optional[str] = None
    delivered_at: Optional[datetime] = None
    
    # Timeline
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    refunded_at: Optional[datetime] = None
    
    # Refund info
    refund_reason: Optional[str] = None
    refund_amount: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize purchase to dictionary."""
        return {
            "purchase_id": self.purchase_id,
            "listing_id": self.listing_id,
            "kernel_id": self.kernel_id,
            "buyer_id": self.buyer_id,
            "seller_id": self.seller_id,
            "status": self.status.value,
            "license_type": self.license_type.value,
            "pricing": {
                "price": self.price,
                "currency": self.currency,
                "discount": self.discount,
                "final_price": self.final_price
            },
            "payment": {
                "method": self.payment_method,
                "reference": self.payment_reference,
                "transaction_id": self.transaction_id
            },
            "delivery": {
                "license_id": self.license_id,
                "download_url": self.download_url,
                "delivered_at": self.delivered_at.isoformat() if self.delivered_at else None
            },
            "refund": {
                "reason": self.refund_reason,
                "amount": self.refund_amount,
                "refunded_at": self.refunded_at.isoformat() if self.refunded_at else None
            },
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }


@dataclass
class Review:
    """
    User review of a marketplace listing.
    
    Allows purchasers to share their experience and rate kernels.
    """
    review_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    listing_id: str = ""
    kernel_id: str = ""
    user_id: str = ""
    
    # Rating (1-5)
    rating: int = 5
    title: str = ""
    content: str = ""
    
    # Pros and cons
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    
    # Use case
    use_case: str = ""
    use_duration: str = ""
    
    # Verification
    is_verified_purchase: bool = False
    is_verified_owner: bool = False
    
    # Engagement
    helpful_count: int = 0
    helpful_votes: Dict[str, bool] = field(default_factory=dict)  # user_id -> voted
    response_count: int = 0
    
    # Moderation
    is_approved: bool = True
    is_featured: bool = False
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize review to dictionary."""
        return {
            "review_id": self.review_id,
            "listing_id": self.listing_id,
            "kernel_id": self.kernel_id,
            "user_id": self.user_id,
            "rating": self.rating,
            "title": self.title,
            "content": self.content,
            "pros": self.pros,
            "cons": self.cons,
            "use_case": self.use_case,
            "verification": {
                "is_verified_purchase": self.is_verified_purchase,
                "is_verified_owner": self.is_verified_owner
            },
            "engagement": {
                "helpful_count": self.helpful_count,
                "response_count": self.response_count
            },
            "moderation": {
                "is_approved": self.is_approved,
                "is_featured": self.is_featured
            },
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


@dataclass
class SellerAnalytics:
    """
    Analytics data for a seller.
    
    Provides insights into listing performance, revenue, and trends.
    """
    seller_id: str = ""
    period_start: datetime = None
    period_end: datetime = None
    
    # Revenue metrics
    total_revenue: float = 0.0
    gross_revenue: float = 0.0
    net_revenue: float = 0.0
    platform_fee: float = 0.0
    refunds_total: float = 0.0
    
    # Sales metrics
    total_sales: int = 0
    new_customers: int = 0
    returning_customers: int = 0
    
    # Listing metrics
    total_listings: int = 0
    active_listings: int = 0
    total_views: int = 0
    total_downloads: int = 0
    
    # Rating metrics
    average_rating: float = 0.0
    total_reviews: int = 0
    
    # Top performing
    top_listings: List[Dict[str, Any]] = field(default_factory=list)
    
    # Trends
    revenue_by_day: Dict[str, float] = field(default_factory=dict)
    sales_by_category: Dict[str, int] = field(default_factory=dict)


class MarketplaceManager:
    """
    Main marketplace management class.
    
    Handles all marketplace operations including listings, purchases,
    reviews, and seller management.
    """
    
    # Platform fee percentage
    PLATFORM_FEE_RATE = 0.15  # 15% platform fee
    
    def __init__(self):
        """Initialize the marketplace manager."""
        self._listings: Dict[str, Listing] = {}
        self._purchases: Dict[str, Purchase] = {}
        self._licenses: Dict[str, License] = {}
        self._reviews: Dict[str, Review] = {}
        self._seller_analytics: Dict[str, SellerAnalytics] = {}
        
        # Seller accounts
        self._sellers: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Marketplace manager initialized")
    
    # Listing Management
    def create_listing(
        self,
        kernel_id: str,
        seller_id: str,
        title: str,
        description: str = "",
        category: KernelCategory = KernelCategory.CUSTOM,
        pricing_model: PricingModel = PricingModel.FREE,
        base_price: float = 0.0,
        tags: List[str] = None,
        **kwargs
    ) -> Listing:
        """
        Create a new marketplace listing.
        
        Args:
            kernel_id: ID of the kernel to list
            seller_id: ID of the seller creating the listing
            title: Listing title
            description: Full description
            category: Kernel category
            pricing_model: Pricing model to use
            base_price: Base price for the kernel
            tags: Tags for discovery
            
        Returns:
            Created Listing object
        """
        listing = Listing(
            kernel_id=kernel_id,
            seller_id=seller_id,
            title=title,
            description=description,
            category=category,
            pricing_model=pricing_model,
            base_price=base_price,
            tags=tags or [],
            status=ListingStatus.DRAFT,
            **kwargs
        )
        
        self._listings[listing.listing_id] = listing
        
        # Update seller stats
        self._update_seller_stat(seller_id, "listings_count", 1)
        
        logger.info(f"Created listing {listing.listing_id} for kernel {kernel_id}")
        return listing
    
    def update_listing(
        self,
        listing_id: str,
        **updates
    ) -> Listing:
        """
        Update an existing listing.
        
        Args:
            listing_id: ID of listing to update
            **updates: Fields to update
            
        Returns:
            Updated Listing object
            
        Raises:
            KeyError: If listing not found
        """
        if listing_id not in self._listings:
            raise KeyError(f"Listing not found: {listing_id}")
        
        listing = self._listings[listing_id]
        
        # Update allowed fields
        allowed_fields = {
            "title", "description", "short_description", "category",
            "pricing_model", "base_price", "subscription_price",
            "credits_pack_price", "credits_pack_size", "available_licenses",
            "screenshots", "demo_url", "documentation_url", "video_url",
            "tags", "keywords", "capabilities", "status", "is_featured"
        }
        
        for field_name, value in updates.items():
            if field_name in allowed_fields:
                setattr(listing, field_name, value)
        
        listing.updated_at = datetime.now()
        
        # If publishing, set published_at
        if listing.status == ListingStatus.ACTIVE and not listing.published_at:
            listing.published_at = datetime.now()
        
        logger.info(f"Updated listing {listing_id}")
        return listing
    
    def submit_for_review(self, listing_id: str) -> Listing:
        """Submit a listing for moderation review."""
        if listing_id not in self._listings:
            raise KeyError(f"Listing not found: {listing_id}")
        
        listing = self._listings[listing_id]
        listing.status = ListingStatus.PENDING_REVIEW
        listing.updated_at = datetime.now()
        
        logger.info(f"Listing {listing_id} submitted for review")
        return listing
    
    def approve_listing(self, listing_id: str, notes: str = "") -> Listing:
        """Approve a listing for publication."""
        if listing_id not in self._listings:
            raise KeyError(f"Listing not found: {listing_id}")
        
        listing = self._listings[listing_id]
        listing.status = ListingStatus.APPROVED
        listing.moderation_notes = notes
        listing.updated_at = datetime.now()
        
        logger.info(f"Listing {listing_id} approved")
        return listing
    
    def reject_listing(self, listing_id: str, reason: str) -> Listing:
        """Reject a listing with reason."""
        if listing_id not in self._listings:
            raise KeyError(f"Listing not found: {listing_id}")
        
        listing = self._listings[listing_id]
        listing.status = ListingStatus.REJECTED
        listing.rejection_reason = reason
        listing.updated_at = datetime.now()
        
        logger.info(f"Listing {listing_id} rejected: {reason}")
        return listing
    
    def publish_listing(self, listing_id: str) -> Listing:
        """Publish an approved listing."""
        if listing_id not in self._listings:
            raise KeyError(f"Listing not found: {listing_id}")
        
        listing = self._listings[listing_id]
        if listing.status != ListingStatus.APPROVED:
            raise ValueError(f"Listing must be approved before publishing")
        
        listing.status = ListingStatus.ACTIVE
        listing.published_at = datetime.now()
        listing.updated_at = datetime.now()
        
        logger.info(f"Listing {listing_id} published")
        return listing
    
    def archive_listing(self, listing_id: str) -> Listing:
        """Archive a listing (hide from marketplace)."""
        if listing_id not in self._listings:
            raise KeyError(f"Listing not found: {listing_id}")
        
        listing = self._listings[listing_id]
        listing.status = ListingStatus.ARCHIVED
        listing.updated_at = datetime.now()
        
        logger.info(f"Listing {listing_id} archived")
        return listing
    
    def get_listing(self, listing_id: str) -> Optional[Listing]:
        """Get a listing by ID."""
        return self._listings.get(listing_id)
    
    def search_listings(
        self,
        query: str = None,
        category: KernelCategory = None,
        min_price: float = None,
        max_price: float = None,
        min_rating: float = None,
        sort_by: str = "rating",
        limit: int = 20,
        offset: int = 0
    ) -> List[Listing]:
        """
        Search marketplace listings.
        
        Args:
            query: Search query string
            category: Filter by category
            min_price: Minimum price filter
            max_price: Maximum price filter
            min_rating: Minimum rating filter
            sort_by: Sort field (rating, price, newest, popular)
            limit: Result limit
            offset: Result offset
            
        Returns:
            List of matching listings
        """
        results = [
            l for l in self._listings.values()
            if l.status == ListingStatus.ACTIVE
        ]
        
        # Apply filters
        if query:
            query_lower = query.lower()
            results = [
                l for l in results
                if query_lower in l.title.lower() or
                   query_lower in l.description.lower() or
                   any(query_lower in tag for tag in l.tags)
            ]
        
        if category:
            results = [l for l in results if l.category == category]
        
        if min_price is not None:
            results = [l for l in results if l.base_price >= min_price]
        
        if max_price is not None:
            results = [l for l in results if l.base_price <= max_price]
        
        if min_rating is not None:
            results = [l for l in results if l.average_rating >= min_rating]
        
        # Apply sorting
        if sort_by == "rating":
            results.sort(key=lambda l: l.average_rating, reverse=True)
        elif sort_by == "price_low":
            results.sort(key=lambda l: l.base_price)
        elif sort_by == "price_high":
            results.sort(key=lambda l: l.base_price, reverse=True)
        elif sort_by == "newest":
            results.sort(key=lambda l: l.created_at, reverse=True)
        elif sort_by == "popular":
            results.sort(key=lambda l: l.download_count, reverse=True)
        
        # Increment view counts
        for listing in results[offset:offset + limit]:
            listing.view_count += 1
        
        return results[offset:offset + limit]
    
    def get_featured_listings(self, limit: int = 10) -> List[Listing]:
        """Get featured listings for homepage."""
        featured = [
            l for l in self._listings.values()
            if l.status == ListingStatus.ACTIVE and l.is_featured
        ]
        return sorted(featured, key=lambda l: l.featured_until or datetime.min, reverse=True)[:limit]
    
    # Purchase Processing
    def create_purchase(
        self,
        listing_id: str,
        buyer_id: str,
        license_type: LicenseType = LicenseType.PERSONAL,
        payment_method: str = "credit_card",
        **kwargs
    ) -> Purchase:
        """
        Create a new purchase transaction.
        
        Args:
            listing_id: ID of listing to purchase
            buyer_id: ID of the buyer
            license_type: Type of license to purchase
            payment_method: Payment method used
            
        Returns:
            Created Purchase object
        """
        if listing_id not in self._listings:
            raise KeyError(f"Listing not found: {listing_id}")
        
        listing = self._listings[listing_id]
        
        # Calculate price
        price = listing.get_price_for_license(license_type)
        
        purchase = Purchase(
            listing_id=listing_id,
            kernel_id=listing.kernel_id,
            buyer_id=buyer_id,
            seller_id=listing.seller_id,
            license_type=license_type,
            price=price,
            final_price=price,
            payment_method=payment_method,
            status=PurchaseStatus.PENDING,
            **kwargs
        )
        
        self._purchases[purchase.purchase_id] = purchase
        
        logger.info(f"Created purchase {purchase.purchase_id} for listing {listing_id}")
        return purchase
    
    def complete_purchase(self, purchase_id: str, transaction_id: str) -> Purchase:
        """
        Complete a purchase transaction.
        
        Args:
            purchase_id: ID of the purchase
            transaction_id: Payment processor transaction ID
            
        Returns:
            Updated Purchase object
        """
        if purchase_id not in self._purchases:
            raise KeyError(f"Purchase not found: {purchase_id}")
        
        purchase = self._purchases[purchase_id]
        purchase.status = PurchaseStatus.COMPLETED
        purchase.transaction_id = transaction_id
        purchase.completed_at = datetime.now()
        
        # Create license for the buyer
        license = self._create_license_from_purchase(purchase)
        purchase.license_id = license.license_id
        
        # Update listing stats
        listing = self._listings.get(purchase.listing_id)
        if listing:
            listing.purchase_count += 1
            listing.download_count += 1
        
        # Update seller stats
        self._update_seller_stat(
            purchase.seller_id,
            "revenue",
            purchase.final_price * (1 - self.PLATFORM_FEE_RATE)
        )
        self._update_seller_stat(purchase.seller_id, "sales_count", 1)
        
        logger.info(f"Completed purchase {purchase_id}")
        return purchase
    
    def refund_purchase(
        self,
        purchase_id: str,
        reason: str,
        amount: float = None
    ) -> Purchase:
        """
        Process a refund for a purchase.
        
        Args:
            purchase_id: ID of the purchase to refund
            reason: Reason for refund
            amount: Refund amount (full if not specified)
            
        Returns:
            Updated Purchase object
        """
        if purchase_id not in self._purchases:
            raise KeyError(f"Purchase not found: {purchase_id}")
        
        purchase = self._purchases[purchase_id]
        
        # Update purchase status
        purchase.status = PurchaseStatus.REFUNDED
        purchase.refund_reason = reason
        purchase.refund_amount = amount or purchase.final_price
        purchase.refunded_at = datetime.now()
        
        # Deactivate license
        if purchase.license_id and purchase.license_id in self._licenses:
            self._licenses[purchase.license_id].is_active = False
        
        # Update seller stats
        self._update_seller_stat(
            purchase.seller_id,
            "refunds",
            purchase.refund_amount
        )
        
        logger.info(f"Refunded purchase {purchase_id}: {reason}")
        return purchase
    
    def _create_license_from_purchase(self, purchase: Purchase) -> License:
        """Create a license record from a completed purchase."""
        listing = self._listings.get(purchase.listing_id)
        
        license = License(
            kernel_id=purchase.kernel_id,
            listing_id=purchase.listing_id,
            purchaser_id=purchase.buyer_id,
            license_type=purchase.license_type,
            purchase_id=purchase.purchase_id,
            max_users=1 if purchase.license_type == LicenseType.PERSONAL else 10,
            features=self._get_license_features(purchase.license_type),
            restrictions=self._get_license_restrictions(purchase.license_type)
        )
        
        self._licenses[license.license_id] = license
        return license
    
    def _get_license_features(self, license_type: LicenseType) -> List[str]:
        """Get features included with a license type."""
        features = {
            LicenseType.PERSONAL: ["personal_use", "non_commercial"],
            LicenseType.COMMERCIAL: ["personal_use", "commercial_use", "support"],
            LicenseType.ENTERPRISE: [
                "personal_use", "commercial_use", "support",
                "multi_user", "priority_queue", "dedicated_resources"
            ],
            LicenseType.EXTENDED: [
                "personal_use", "commercial_use", "support",
                "multi_user", "redistribution", "custom_development",
                "white_label", "source_access"
            ],
            LicenseType.EVALUATION: ["personal_use", "time_limited", "limited_support"]
        }
        return features.get(license_type, [])
    
    def _get_license_restrictions(self, license_type: LicenseType) -> List[str]:
        """Get restrictions for a license type."""
        restrictions = {
            LicenseType.PERSONAL: ["no_commercial_use", "no_redistribution"],
            LicenseType.COMMERCIAL: ["no_redistribution"],
            LicenseType.ENTERPRISE: [],
            LicenseType.EXTENDED: [],
            LicenseType.EVALUATION: ["time_limited", "no_redistribution"]
        }
        return restrictions.get(license_type, [])
    
    def get_purchase(self, purchase_id: str) -> Optional[Purchase]:
        """Get a purchase by ID."""
        return self._purchases.get(purchase_id)
    
    def get_user_purchases(self, user_id: str) -> List[Purchase]:
        """Get all purchases by a user."""
        return [
            p for p in self._purchases.values()
            if p.buyer_id == user_id
        ]
    
    # License Management
    def get_license(self, license_id: str) -> Optional[License]:
        """Get a license by ID."""
        return self._licenses.get(license_id)
    
    def get_user_licenses(self, user_id: str) -> List[License]:
        """Get all licenses for a user."""
        return [
            l for l in self._licenses.values()
            if l.purchaser_id == user_id and l.is_active
        ]
    
    def validate_license(self, license_id: str, user_id: str = None) -> Tuple[bool, str]:
        """Validate a license for use."""
        license = self._licenses.get(license_id)
        if not license:
            return False, "License not found"
        
        return license.can_use(user_id)
    
    def increment_license_usage(self, license_id: str) -> bool:
        """Increment usage counter for a license."""
        license = self._licenses.get(license_id)
        if not license:
            return False
        
        if license.max_queries is not None:
            license.current_queries += 1
            if license.current_queries >= license.max_queries:
                license.is_active = False
        
        return True
    
    # Review Management
    def create_review(
        self,
        listing_id: str,
        kernel_id: str,
        user_id: str,
        rating: int,
        content: str,
        title: str = "",
        pros: List[str] = None,
        cons: List[str] = None,
        **kwargs
    ) -> Review:
        """
        Create a review for a listing.
        
        Args:
            listing_id: ID of the listing being reviewed
            kernel_id: ID of the kernel being reviewed
            user_id: ID of the reviewer
            rating: Rating (1-5)
            content: Review content
            title: Review title
            pros: List of positive points
            cons: List of negative points
            
        Returns:
            Created Review object
        """
        # Verify user has purchased the kernel
        has_purchase = any(
            p.listing_id == listing_id and p.buyer_id == user_id
            for p in self._purchases.values()
        )
        
        review = Review(
            listing_id=listing_id,
            kernel_id=kernel_id,
            user_id=user_id,
            rating=rating,
            content=content,
            title=title,
            pros=pros or [],
            cons=cons or [],
            is_verified_purchase=has_purchase,
            **kwargs
        )
        
        self._reviews[review.review_id] = review
        
        # Update listing rating
        self._update_listing_rating(listing_id)
        
        logger.info(f"Created review {review.review_id} for listing {listing_id}")
        return review
    
    def _update_listing_rating(self, listing_id: str):
        """Update the average rating for a listing."""
        listing = self._listings.get(listing_id)
        if not listing:
            return
        
        reviews = [
            r for r in self._reviews.values()
            if r.listing_id == listing_id and r.is_approved
        ]
        
        if reviews:
            listing.average_rating = sum(r.rating for r in reviews) / len(reviews)
            listing.review_count = len(reviews)
    
    def get_listing_reviews(
        self,
        listing_id: str,
        min_rating: int = None,
        limit: int = 20,
        offset: int = 0
    ) -> List[Review]:
        """Get reviews for a listing."""
        reviews = [
            r for r in self._reviews.values()
            if r.listing_id == listing_id and r.is_approved
        ]
        
        if min_rating is not None:
            reviews = [r for r in reviews if r.rating >= min_rating]
        
        reviews.sort(key=lambda r: r.created_at, reverse=True)
        return reviews[offset:offset + limit]
    
    def vote_review_helpful(self, review_id: str, user_id: str) -> Review:
        """Mark a review as helpful."""
        if review_id not in self._reviews:
            raise KeyError(f"Review not found: {review_id}")
        
        review = self._reviews[review_id]
        
        if user_id not in review.helpful_votes:
            review.helpful_votes[user_id] = True
            review.helpful_count += 1
        
        return review
    
    # Seller Management
    def register_seller(self, user_id: str, **kwargs) -> Dict[str, Any]:
        """Register a seller account."""
        if user_id not in self._sellers:
            self._sellers[user_id] = {
                "user_id": user_id,
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "stats": {
                    "revenue": 0.0,
                    "sales_count": 0,
                    "listings_count": 0,
                    "refunds": 0.0
                },
                "settings": {},
                **kwargs
            }
        return self._sellers[user_id]
    
    def _update_seller_stat(self, seller_id: str, stat: str, value: float):
        """Update a seller statistic."""
        if seller_id in self._sellers:
            self._sellers[seller_id]["stats"][stat] = (
                self._sellers[seller_id]["stats"].get(stat, 0) + value
            )
    
    def get_seller_analytics(
        self,
        seller_id: str,
        period_days: int = 30
    ) -> SellerAnalytics:
        """Get analytics for a seller."""
        period_end = datetime.now()
        period_start = period_end - timedelta(days=period_days)
        
        analytics = SellerAnalytics(
            seller_id=seller_id,
            period_start=period_start,
            period_end=period_end
        )
        
        # Get seller's purchases
        seller_purchases = [
            p for p in self._purchases.values()
            if p.seller_id == seller_id and p.completed_at
        ]
        
        # Calculate revenue
        analytics.gross_revenue = sum(p.final_price for p in seller_purchases)
        analytics.platform_fee = analytics.gross_revenue * self.PLATFORM_FEE_RATE
        analytics.net_revenue = analytics.gross_revenue - analytics.platform_fee
        analytics.total_revenue = analytics.net_revenue
        
        # Calculate refunds
        refunded = [p for p in seller_purchases if p.status == PurchaseStatus.REFUNDED]
        analytics.refunds_total = sum(p.refund_amount for p in refunded)
        analytics.total_revenue -= analytics.refunds_total
        
        # Sales metrics
        analytics.total_sales = len(seller_purchases)
        
        # Get seller's listings
        seller_listings = [
            l for l in self._listings.values()
            if l.seller_id == seller_id
        ]
        analytics.total_listings = len(seller_listings)
        analytics.active_listings = len([
            l for l in seller_listings if l.status == ListingStatus.ACTIVE
        ])
        analytics.total_views = sum(l.view_count for l in seller_listings)
        analytics.total_downloads = sum(l.download_count for l in seller_listings)
        
        # Calculate average rating
        reviews = []
        for listing in seller_listings:
            reviews.extend(self.get_listing_reviews(listing.listing_id))
        
        if reviews:
            analytics.average_rating = sum(r.rating for r in reviews) / len(reviews)
            analytics.total_reviews = len(reviews)
        
        # Top listings
        analytics.top_listings = [
            {
                "listing_id": l.listing_id,
                "title": l.title,
                "downloads": l.download_count,
                "revenue": l.download_count * l.base_price * (1 - self.PLATFORM_FEE_RATE)
            }
            for l in sorted(seller_listings, key=lambda x: x.download_count, reverse=True)[:5]
        ]
        
        return analytics
    
    def get_seller_payout_info(self, seller_id: str) -> Dict[str, Any]:
        """Get payout information for a seller."""
        analytics = self.get_seller_analytics(seller_id)
        
        return {
            "seller_id": seller_id,
            "available_balance": analytics.net_revenue - analytics.refunds_total,
            "pending_balance": sum(
                p.final_price * (1 - self.PLATFORM_FEE_RATE)
                for p in self._purchases.values()
                if p.seller_id == seller_id and p.status == PurchaseStatus.PENDING
            ),
            "total_earnings": analytics.total_revenue,
            "platform_fees_paid": analytics.platform_fee,
            "last_payout": None  # Would track actual payouts
        }


# Convenience functions
def create_marketplace() -> MarketplaceManager:
    """Create a new marketplace instance."""
    return MarketplaceManager()


def format_price(amount: float, currency: str = "USD") -> str:
    """Format a price for display."""
    return f"{currency} {amount:.2f}"


# Example usage
if __name__ == "__main__":
    marketplace = create_marketplace()
    
    # Register a seller
    seller = marketplace.register_seller(
        user_id="seller_001",
        name="AI Solutions Inc.",
        email="contact@aisolutions.com"
    )
    print(f"Registered seller: {seller['user_id']}")
    
    # Create a listing
    listing = marketplace.create_listing(
        kernel_id="kernel_reasoning_v1",
        seller_id="seller_001",
        title="Advanced Reasoning Kernel",
        description="A powerful reasoning kernel for complex problem solving.",
        category=KernelCategory.REASONING,
        pricing_model=PricingModel.ONE_TIME,
        base_price=99.99,
        tags=["reasoning", "problem-solving", "logic"]
    )
    print(f"Created listing: {listing.listing_id}")
    
    # Approve and publish
    listing = marketplace.approve_listing(listing.listing_id, "Looks good!")
    listing = marketplace.publish_listing(listing.listing_id)
    print(f"Published listing: {listing.listing_id}")
    
    # Create a purchase
    purchase = marketplace.create_purchase(
        listing_id=listing.listing_id,
        buyer_id="user_001",
        license_type=LicenseType.COMMERCIAL
    )
    print(f"Created purchase: {purchase.purchase_id}")
    
    # Complete the purchase
    purchase = marketplace.complete_purchase(purchase.purchase_id, "txn_12345")
    print(f"Completed purchase with license: {purchase.license_id}")
    
    # Create a review
    review = marketplace.create_review(
        listing_id=listing.listing_id,
        kernel_id=listing.kernel_id,
        user_id="user_001",
        rating=5,
        title="Excellent kernel!",
        content="This kernel solved all my reasoning problems quickly and accurately.",
        pros=["Fast", "Accurate", "Easy to use"],
        cons=["Could use more documentation"]
    )
    print(f"Created review: {review.review_id}")
    
    # Get seller analytics
    analytics = marketplace.get_seller_analytics("seller_001")
    print(f"\nSeller Analytics:")
    print(f"  Total Revenue: {format_price(analytics.total_revenue)}")
    print(f"  Platform Fees: {format_price(analytics.platform_fee)}")
    print(f"  Total Sales: {analytics.total_sales}")
    print(f"  Average Rating: {analytics.average_rating:.2f}")
    
    # Search listings
    results = marketplace.search_listings(query="reasoning")
    print(f"\nSearch results for 'reasoning': {len(results)} listings")
