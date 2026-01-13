"""
KERNELIZE Platform - Pydantic Schemas
======================================

This module defines all Pydantic models for API request/response validation.
These schemas provide data validation, serialization, and documentation
for the KERNELIZE Knowledge Compression API.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ==================== User-related Schemas ====================

class UserCreate(BaseModel):
    """User registration request"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password")
    confirm_password: str = Field(..., description="Confirm password")
    
    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if v != info.data.get("password"):
            raise ValueError("Passwords do not match")
        return v


class UserLogin(BaseModel):
    """User login request"""
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class UserResponse(BaseModel):
    """User response"""
    id: str = Field(..., description="User ID")
    email: str = Field(..., description="User email")
    role: str = Field(..., description="User role")
    plan_tier: str = Field(..., description="Plan tier")
    is_verified: bool = Field(..., description="Whether verified")
    created_at: datetime = Field(..., description="Creation time")
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """User update request"""
    email: Optional[EmailStr] = Field(None, description="New email address")
    role: Optional[str] = Field(None, description="User role")
    plan_tier: Optional[str] = Field(None, description="Plan tier")


# ==================== API Key Schemas ====================

class APIKeyCreate(BaseModel):
    """API key creation request"""
    name: str = Field(..., min_length=1, max_length=100, description="Key name")
    description: Optional[str] = Field(None, max_length=500, description="Key description")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")


class APIKeyResponse(BaseModel):
    """API key response (does not include full key)"""
    id: str = Field(..., description="Key ID")
    name: str = Field(..., description="Key name")
    description: Optional[str] = Field(None, description="Key description")
    key_prefix: str = Field(..., description="Key prefix")
    is_active: bool = Field(..., description="Whether active")
    expires_at: Optional[datetime] = Field(None, description="Expiration time")
    request_count: int = Field(..., description="Request count")
    last_used_at: Optional[datetime] = Field(None, description="Last used time")
    created_at: datetime = Field(..., description="Creation time")
    
    class Config:
        from_attributes = True


class APIKeyFullResponse(BaseModel):
    """Full API key response (returned only once upon creation)"""
    id: str = Field(..., description="Key ID")
    key: str = Field(..., description="Full API key (please keep it safe)")
    name: str = Field(..., description="Key name")
    created_at: datetime = Field(..., description="Creation time")


# ==================== Compression-related Schemas ====================

class CompressionRequest(BaseModel):
    """Compression request"""
    content: str = Field(..., description="Knowledge content to compress")
    domain: str = Field(default="general", max_length=100, description="Knowledge domain")
    language: str = Field(default="en", max_length=10, description="Language")
    compression_level: int = Field(default=5, ge=1, le=10, description="Compression level")
    extract_entities: bool = Field(default=True, description="Whether to extract entities")
    extract_relationships: bool = Field(default=True, description="Whether to extract relationships")
    extract_causality: bool = Field(default=True, description="Whether to extract causal relationships")
    generate_embedding: bool = Field(default=True, description="Whether to generate embedding vector")


class CompressionResponse(BaseModel):
    """Compression response"""
    kernel_id: str = Field(..., description="Kernel ID")
    original_size: int = Field(..., description="Original size (bytes)")
    compressed_size: int = Field(..., description="Compressed size (bytes)")
    compression_ratio: float = Field(..., description="Compression ratio")
    compressed_content: str = Field(..., description="Compressed content")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted relationships")
    causal_chains: List[Dict[str, Any]] = Field(default_factory=list, description="Causal chains")
    embedding_model: Optional[str] = Field(None, description="Embedding model name")
    processing_time_ms: int = Field(..., description="Processing time (milliseconds)")
    created_at: datetime = Field(..., description="Creation time")


class BatchCompressionRequest(BaseModel):
    """Batch compression request"""
    items: List[CompressionRequest] = Field(..., min_length=1, max_length=100, description="Compression item list")


class BatchCompressionResponse(BaseModel):
    """Batch compression response"""
    results: List[CompressionResponse] = Field(..., description="Compression result list")
    total_original_size: int = Field(..., description="Total original size")
    total_compressed_size: int = Field(..., description="Total compressed size")
    average_compression_ratio: float = Field(..., description="Average compression ratio")
    processing_time_ms: int = Field(..., description="Total processing time")


# ==================== Multimodal Compression Schemas ====================

class ImageCompressionRequest(BaseModel):
    """Image compression request"""
    image_data: str = Field(..., description="Base64-encoded image data")
    image_format: str = Field(default="jpeg", description="Image format (jpeg/png/webp)")
    compression_level: int = Field(default=5, ge=1, le=10, description="Compression level")
    extract_text: bool = Field(default=True, description="Whether to extract text (OCR)")
    generate_caption: bool = Field(default=True, description="Whether to generate description")
    generate_embedding: bool = Field(default=True, description="Whether to generate embedding vector")


class ImageCompressionResponse(BaseModel):
    """Image compression response"""
    kernel_id: str = Field(..., description="Kernel ID")
    original_size: int = Field(..., description="Original size")
    compressed_size: int = Field(..., description="Compressed size")
    compression_ratio: float = Field(..., description="Compression ratio")
    extracted_text: Optional[str] = Field(None, description="OCR extracted text")
    generated_caption: Optional[str] = Field(None, description="Generated description")
    embedding_vector: Optional[List[float]] = Field(None, description="Embedding vector")
    processing_time_ms: int = Field(..., description="Processing time")


class AudioCompressionRequest(BaseModel):
    """Audio compression request"""
    audio_data: str = Field(..., description="Base64-encoded audio data")
    audio_format: str = Field(default="wav", description="Audio format (wav/mp3/flac)")
    transcription_language: str = Field(default="en", description="Transcription language")
    transcribe: bool = Field(default=True, description="Whether to transcribe")
    generate_summary: bool = Field(default=True, description="Whether to generate summary")
    analyze_features: bool = Field(default=True, description="Whether to analyze audio features")


class AudioCompressionResponse(BaseModel):
    """Audio compression response"""
    kernel_id: str = Field(..., description="Kernel ID")
    original_size: int = Field(..., description="Original size")
    compressed_size: int = Field(..., description="Compressed size")
    compression_ratio: float = Field(..., description="Compression ratio")
    transcription: Optional[str] = Field(None, description="Speech transcription text")
    summary: Optional[str] = Field(None, description="Audio summary")
    audio_features: Optional[Dict[str, Any]] = Field(None, description="Audio features")
    processing_time_ms: int = Field(..., description="Processing time")


class VideoCompressionRequest(BaseModel):
    """Video compression request"""
    video_data: str = Field(..., description="Base64-encoded video data")
    video_format: str = Field(default="mp4", description="Video format (mp4/avi/mov)")
    extract_keyframes: bool = Field(default=True, description="Whether to extract keyframes")
    generate_transcription: bool = Field(default=True, description="Whether to generate transcription")
    scene_detection: bool = Field(default=True, description="Whether to perform scene detection")


class VideoCompressionResponse(BaseModel):
    """Video compression response"""
    kernel_id: str = Field(..., description="Kernel ID")
    original_size: int = Field(..., description="Original size")
    compressed_size: int = Field(..., description="Compressed size")
    compression_ratio: float = Field(..., description="Compression ratio")
    scenes: List[Dict[str, Any]] = Field(default_factory=list, description="Detected scenes")
    keyframes: List[str] = Field(default_factory=list, description="Keyframes (Base64-encoded)")
    transcription: Optional[str] = Field(None, description="Video transcription")
    processing_time_ms: int = Field(..., description="Processing time")


class DocumentCompressionRequest(BaseModel):
    """Document compression request"""
    document_data: str = Field(..., description="Base64-encoded document data")
    document_format: str = Field(default="pdf", description="Document format (pdf/docx/pptx/txt)")
    extract_images: bool = Field(default=True, description="Whether to extract images")
    extract_tables: bool = Field(default=True, description="Whether to extract tables")
    generate_summary: bool = Field(default=True, description="Whether to generate summary")
    extract_entities: bool = Field(default=True, description="Whether to extract entities")


class DocumentCompressionResponse(BaseModel):
    """Document compression response"""
    kernel_id: str = Field(..., description="Kernel ID")
    original_size: int = Field(..., description="Original size")
    compressed_size: int = Field(..., description="Compressed size")
    compression_ratio: float = Field(..., description="Compression ratio")
    content: str = Field(..., description="Extracted text content")
    summary: Optional[str] = Field(None, description="Document summary")
    images: List[str] = Field(default_factory=list, description="Extracted images (Base64-encoded)")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted tables")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="Extracted entities")
    processing_time_ms: int = Field(..., description="Processing time")


# ==================== Query-related Schemas ====================

class QueryRequest(BaseModel):
    """Query request"""
    query: str = Field(..., description="Query content")
    kernel_ids: Optional[List[str]] = Field(None, description="Kernel ID list to query")
    query_type: str = Field(default="semantic", description="Query type (semantic/exact/fuzzy/hybrid)")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity")


class QueryResult(BaseModel):
    """Query result"""
    kernel_id: str = Field(..., description="Kernel ID")
    kernel_id_short: str = Field(..., description="Kernel ID (short)")
    content: str = Field(..., description="Kernel content")
    similarity_score: float = Field(..., description="Similarity score")
    rank: int = Field(..., description="Rank")
    highlights: List[str] = Field(default_factory=list, description="Matching segments")


class QueryResponse(BaseModel):
    """Query response"""
    results: List[QueryResult] = Field(..., description="Query results")
    total_found: int = Field(..., description="Total results found")
    query_time_ms: int = Field(..., description="Query time (milliseconds)")
    query_type: str = Field(..., description="Query type used")


class MergeRequest(BaseModel):
    """Merge request"""
    kernel_ids: List[str] = Field(..., min_length=2, max_length=50, description="Kernel ID list to merge")
    merge_strategy: str = Field(default="concatenate", description="Merge strategy (concatenate/semantic)")
    resolve_conflicts: bool = Field(default=True, description="Whether to resolve conflicts")


class MergeResponse(BaseModel):
    """Merge response"""
    new_kernel_id: str = Field(..., description="New kernel ID")
    source_kernel_ids: List[str] = Field(..., description="Source kernel ID list")
    merged_content: str = Field(..., description="Merged content")
    conflicts_resolved: int = Field(default=0, description="Number of conflicts resolved")
    compression_ratio: float = Field(..., description="Compression ratio")


class DistillRequest(BaseModel):
    """Distill request"""
    kernel_id: str = Field(..., description="Kernel ID to distill")
    target_model: str = Field(default="llama-2-7b", description="Target model")
    distillation_level: int = Field(default=5, ge=1, le=10, description="Distillation level")


class DistillResponse(BaseModel):
    """Distill response"""
    kernel_id: str = Field(..., description="Source kernel ID")
    distilled_kernel_id: str = Field(..., description="Distilled kernel ID")
    target_model: str = Field(..., description="Target model")
    original_size: int = Field(..., description="Original size")
    distilled_size: int = Field(..., description="Distilled size")
    distillation_ratio: float = Field(..., description="Distillation ratio")
    processing_time_ms: int = Field(..., description="Processing time")


# ==================== Task-related Schemas ====================

class TaskStatus(BaseModel):
    """Task status"""
    job_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    progress: int = Field(default=0, description="Progress (0-100)")
    created_at: datetime = Field(..., description="Creation time")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    error_message: Optional[str] = Field(None, description="Error message")


# ==================== Statistics-related Schemas ====================

class UsageStats(BaseModel):
    """Usage statistics"""
    total_kernels: int = Field(..., description="Total kernels")
    total_queries: int = Field(..., description="Total queries")
    total_compressed_bytes: int = Field(..., description="Total compressed bytes")
    total_original_bytes: int = Field(..., description="Total original bytes")
    average_compression_ratio: float = Field(..., description="Average compression ratio")
    average_query_time_ms: float = Field(..., description="Average query time")


class PerformanceMetrics(BaseModel):
    """Performance metrics"""
    uptime_seconds: int = Field(..., description="Uptime")
    total_requests: int = Field(..., description="Total requests")
    successful_requests: int = Field(..., description="Successful requests")
    failed_requests: int = Field(..., description="Failed requests")
    average_response_time_ms: float = Field(..., description="Average response time")
    requests_per_minute: float = Field(..., description="Requests per minute")
    active_connections: int = Field(..., description="Active connections")


# ==================== Common Schemas ====================

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    database: str = Field(..., description="Database status")
    cache: str = Field(..., description="Cache status")
    timestamp: datetime = Field(..., description="Check time")


class ErrorResponse(BaseModel):
    """Error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error time")


class PaginationParams(BaseModel):
    """Pagination parameters"""
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Page size")


class PaginatedResponse(BaseModel):
    """Paginated response"""
    items: List[Any] = Field(..., description="Data items")
    total: int = Field(..., description="Total count")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")
    total_pages: int = Field(..., description="Total pages")
    has_next: bool = Field(..., description="Has next page")
    has_previous: bool = Field(..., description="Has previous page")


# ==================== Token Schemas ====================

class Token(BaseModel):
    """Access token response"""
    access_token: str = Field(..., description="Access token")
    refresh_token: str = Field(..., description="Refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Expiration time (seconds)")


class TokenRefresh(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(..., description="Refresh token")


class TokenPayload(BaseModel):
    """Token payload"""
    sub: str = Field(..., description="User ID")
    email: Optional[str] = Field(None, description="User email")
    role: str = Field(default="user", description="User role")
    exp: Optional[int] = Field(None, description="Expiration timestamp")
