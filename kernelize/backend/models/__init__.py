"""
KERNELIZE Platform - Models Package
====================================

This package contains all database models and Pydantic schemas for the
KERNELIZE Knowledge Compression Platform.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

from .database import (
    User,
    APIKey,
    Kernel,
    CompressionJob,
    QueryLog,
    AnalyticsEvent,
)

from .schemas import (
    UserCreate,
    UserLogin,
    UserResponse,
    APIKeyCreate,
    APIKeyResponse,
    APIKeyFullResponse,
    CompressionRequest,
    CompressionResponse,
    BatchCompressionRequest,
    BatchCompressionResponse,
    ImageCompressionRequest,
    ImageCompressionResponse,
    AudioCompressionRequest,
    AudioCompressionResponse,
    VideoCompressionRequest,
    VideoCompressionResponse,
    DocumentCompressionRequest,
    DocumentCompressionResponse,
    QueryRequest,
    QueryResult,
    QueryResponse,
    MergeRequest,
    MergeResponse,
    DistillRequest,
    DistillResponse,
    TaskStatus,
    UsageStats,
    PerformanceMetrics,
    HealthResponse,
    ErrorResponse,
    Token,
    TokenRefresh,
    TokenPayload,
)

__all__ = [
    # Database Models
    "User",
    "APIKey",
    "Kernel",
    "CompressionJob",
    "QueryLog",
    "AnalyticsEvent",
    
    # Request/Response Schemas
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "APIKeyCreate",
    "APIKeyResponse",
    "APIKeyFullResponse",
    "CompressionRequest",
    "CompressionResponse",
    "BatchCompressionRequest",
    "BatchCompressionResponse",
    "ImageCompressionRequest",
    "ImageCompressionResponse",
    "AudioCompressionRequest",
    "AudioCompressionResponse",
    "VideoCompressionRequest",
    "VideoCompressionResponse",
    "DocumentCompressionRequest",
    "DocumentCompressionResponse",
    "QueryRequest",
    "QueryResult",
    "QueryResponse",
    "MergeRequest",
    "MergeResponse",
    "DistillRequest",
    "DistillResponse",
    "TaskStatus",
    "UsageStats",
    "PerformanceMetrics",
    "HealthResponse",
    "ErrorResponse",
    "Token",
    "TokenRefresh",
    "TokenPayload",
]
