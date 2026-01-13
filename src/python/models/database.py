"""
KERNELIZE Platform - Database Models
======================================

This module defines all SQLAlchemy ORM models for the KERNELIZE Knowledge
Compression Platform. These models represent the core data structures
for users, kernels, compression jobs, and analytics.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..core.database import Base


class User(Base):
    """
    User Model
    
    Stores user account information, authentication credentials, and account status.
    Supports both email/password authentication and API key authentication.
    """
    
    __tablename__ = "users"
    
    # Primary key and basic information
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # Account status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # User roles and permissions
    role: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    plan_tier: Mapped[str] = mapped_column(String(50), default="free", nullable=False)
    
    # Usage limits
    monthly_quota: Mapped[int] = mapped_column(Integer, default=1000, nullable=False)
    monthly_usage: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    
    # Relationships
    api_keys: Mapped[List["APIKey"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    kernels: Mapped[List["Kernel"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    compression_jobs: Mapped[List["CompressionJob"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email}, role={self.role})>"


class APIKey(Base):
    """
    API Key Model
    
    API keys for programmatic access to the KERNELIZE platform.
    Supports multiple keys, expiration dates, and usage statistics.
    """
    
    __tablename__ = "api_keys"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Key information
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Key status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Usage statistics
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # Relationship
    user: Mapped["User"] = relationship(back_populates="api_keys")
    
    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name}, prefix={self.key_prefix})>"


class Kernel(Base):
    """
    Knowledge Kernel Model
    
    Stores compressed knowledge kernels, including original content, semantic embeddings,
    metadata, and compression statistics.
    """
    
    __tablename__ = "kernels"
    
    # Primary key and identifier
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    kernel_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Content information
    original_content: Mapped[str] = mapped_column(Text, nullable=False)
    compressed_content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), default="text", nullable=False)
    
    # Semantic embeddings
    embedding_vector: Mapped[Optional[List[float]]] = mapped_column(JSONB, nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Compression statistics
    original_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    compressed_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    compression_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Metadata
    domain: Mapped[str] = mapped_column(String(100), default="general", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    tags: Mapped[List[str]] = mapped_column(JSONB, default=list, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    
    # Entities and relationships
    entities: Mapped[List[dict]] = mapped_column(JSONB, default=list, nullable=False)
    relationships: Mapped[List[dict]] = mapped_column(JSONB, default=list, nullable=False)
    causal_chains: Mapped[List[dict]] = mapped_column(JSONB, default=list, nullable=False)
    
    # Indexing
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    search_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
    
    # Relationships
    user: Mapped["User"] = relationship(back_populates="kernels")
    queries: Mapped[List["QueryLog"]] = relationship(
        back_populates="kernel",
        cascade="all, delete-orphan",
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_kernel_domain", "domain"),
        Index("idx_kernel_created_at", "created_at"),
        Index("idx_kernel_user_id", "user_id"),
        Index("idx_kernel_embedding", "embedding_vector", postgresql_using="ivfflat"),
    )
    
    def __repr__(self) -> str:
        return f"<Kernel(id={self.id}, kernel_id={self.kernel_id}, ratio={self.compression_ratio:.2f})>"


class CompressionJob(Base):
    """
    Compression Job Model
    
    Tracks asynchronous compression job execution status, progress, and results.
    Supports multimodal content compression (text, image, audio, video).
    """
    
    __tablename__ = "compression_jobs"
    
    # Primary key and identifier
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[str] = mapped_column(String(100), unique=True, nullable=False, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Job configuration
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    compression_level: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    domain: Mapped[str] = mapped_column(String(100), default="general", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    
    # Input and output
    input_source: Mapped[str] = mapped_column(String(500), nullable=False)
    input_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    output_kernel_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # Job status
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
    )  # pending, processing, completed, failed
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Performance statistics
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    peak_memory_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationship
    user: Mapped["User"] = relationship(back_populates="compression_jobs")
    
    def __repr__(self) -> str:
        return f"<CompressionJob(id={self.id}, job_id={self.job_id}, status={self.status})>"


class QueryLog(Base):
    """
    Query Log Model
    
    Records all kernel query operations for analysis, billing, and performance optimization.
    """
    
    __tablename__ = "query_logs"
    
    # Primary key
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    kernel_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("kernels.id", ondelete="CASCADE"),
        nullable=False,
    )
    
    # Query information
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(50), nullable=False)  # semantic, exact, fuzzy, hybrid
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Result statistics
    results_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    relevance_scores: Mapped[List[float]] = mapped_column(JSONB, default=list, nullable=False)
    
    # Authentication information
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    api_key_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("api_keys.id", ondelete="SET NULL"),
        nullable=True,
    )
    
    # Client information
    client_ip: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # Relationship
    kernel: Mapped["Kernel"] = relationship(back_populates="queries")
    
    # Indexes
    __table_args__ = (
        Index("idx_query_kernel_id", "kernel_id"),
        Index("idx_query_created_at", "created_at"),
        Index("idx_query_user_id", "user_id"),
    )
    
    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, query_type={self.query_type}, response_time={self.response_time_ms}ms)>"


class AnalyticsEvent(Base):
    """
    Analytics Event Model
    
    Collects platform usage events for product analytics, performance monitoring, and business intelligence.
    """
    
    __tablename__ = "analytics_events"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    
    event_data: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_analytics_event_type", "event_type"),
        Index("idx_analytics_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<AnalyticsEvent(id={self.id}, type={self.event_type})>"


# Export all models
__all__ = [
    "User",
    "APIKey",
    "Kernel",
    "CompressionJob",
    "QueryLog",
    "AnalyticsEvent",
]
