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
    用户模型
    
    存储用户账户信息、认证凭据和账户状态。
    支持邮箱密码认证和API密钥认证两种方式。
    """
    
    __tablename__ = "users"
    
    # 主键和基本信息
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    
    # 账户状态
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    is_verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    
    # 用户角色和权限
    role: Mapped[str] = mapped_column(String(50), default="user", nullable=False)
    plan_tier: Mapped[str] = mapped_column(String(50), default="free", nullable=False)
    
    # 使用限制
    monthly_quota: Mapped[int] = mapped_column(Integer, default=1000, nullable=False)
    monthly_usage: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # 时间戳
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
    
    # 关系
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
    API密钥模型
    
    用于程序化访问KERNELIZE平台的API密钥。
    支持多个密钥、过期日期和使用统计。
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
    
    # 密钥信息
    key_hash: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    key_prefix: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # 密钥状态
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # 使用统计
    request_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_used_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # 关系
    user: Mapped["User"] = relationship(back_populates="api_keys")
    
    def __repr__(self) -> str:
        return f"<APIKey(id={self.id}, name={self.name}, prefix={self.key_prefix})>"


class Kernel(Base):
    """
    知识内核模型
    
    存储压缩后的知识内核，包含原始内容、语义嵌入、
    元数据和压缩统计信息。
    """
    
    __tablename__ = "kernels"
    
    # 主键和标识
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
    
    # 内容信息
    original_content: Mapped[str] = mapped_column(Text, nullable=False)
    compressed_content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), default="text", nullable=False)
    
    # 语义嵌入
    embedding_vector: Mapped[Optional[List[float]]] = mapped_column(JSONB, nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # 压缩统计
    original_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    compressed_size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    compression_ratio: Mapped[float] = mapped_column(Float, nullable=False)
    
    # 元数据
    domain: Mapped[str] = mapped_column(String(100), default="general", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    tags: Mapped[List[str]] = mapped_column(JSONB, default=list, nullable=False)
    metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    
    # 实体和关系
    entities: Mapped[List[dict]] = mapped_column(JSONB, default=list, nullable=False)
    relationships: Mapped[List[dict]] = mapped_column(JSONB, default=list, nullable=False)
    causal_chains: Mapped[List[dict]] = mapped_column(JSONB, default=list, nullable=False)
    
    # 索引
    is_indexed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    search_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    
    # 时间戳
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
    
    # 关系
    user: Mapped["User"] = relationship(back_populates="kernels")
    queries: Mapped[List["QueryLog"]] = relationship(
        back_populates="kernel",
        cascade="all, delete-orphan",
    )
    
    # 索引
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
    压缩任务模型
    
    跟踪异步压缩任务的执行状态、进度和结果。
    支持多模态内容压缩（文本、图像、音频、视频）。
    """
    
    __tablename__ = "compression_jobs"
    
    # 主键和标识
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
    
    # 任务配置
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    compression_level: Mapped[int] = mapped_column(Integer, default=5, nullable=False)
    domain: Mapped[str] = mapped_column(String(100), default="general", nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)
    
    # 输入输出
    input_source: Mapped[str] = mapped_column(String(500), nullable=False)
    input_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    output_kernel_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    
    # 任务状态
    status: Mapped[str] = mapped_column(
        String(20),
        default="pending",
        nullable=False,
    )  # pending, processing, completed, failed
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # 性能统计
    processing_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    peak_memory_mb: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # 关系
    user: Mapped["User"] = relationship(back_populates="compression_jobs")
    
    def __repr__(self) -> str:
        return f"<CompressionJob(id={self.id}, job_id={self.job_id}, status={self.status})>"


class QueryLog(Base):
    """
    查询日志模型
    
    记录所有内核查询操作，用于分析、计费和性能优化。
    """
    
    __tablename__ = "query_logs"
    
    # 主键
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
    
    # 查询信息
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(50), nullable=False)  # semantic, exact, fuzzy, hybrid
    response_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # 结果统计
    results_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    relevance_scores: Mapped[List[float]] = mapped_column(JSONB, default=list, nullable=False)
    
    # 认证信息
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
    
    # 客户端信息
    client_ip: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # 时间戳
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    
    # 关系
    kernel: Mapped["Kernel"] = relationship(back_populates="queries")
    
    # 索引
    __table_args__ = (
        Index("idx_query_kernel_id", "kernel_id"),
        Index("idx_query_created_at", "created_at"),
        Index("idx_query_user_id", "user_id"),
    )
    
    def __repr__(self) -> str:
        return f"<QueryLog(id={self.id}, query_type={self.query_type}, response_time={self.response_time_ms}ms)>"


class AnalyticsEvent(Base):
    """
    分析事件模型
    
    收集平台使用事件，用于产品分析、性能监控和业务智能。
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
    
    # 索引
    __table_args__ = (
        Index("idx_analytics_event_type", "event_type"),
        Index("idx_analytics_created_at", "created_at"),
    )
    
    def __repr__(self) -> str:
        return f"<AnalyticsEvent(id={self.id}, type={self.event_type})>"


# 导出所有模型
__all__ = [
    "User",
    "APIKey",
    "Kernel",
    "CompressionJob",
    "QueryLog",
    "AnalyticsEvent",
]
