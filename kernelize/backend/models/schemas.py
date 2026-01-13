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


# ==================== 用户相关 Schema ====================

class UserCreate(BaseModel):
    """用户注册请求"""
    email: EmailStr = Field(..., description="用户邮箱地址")
    password: str = Field(..., min_length=8, max_length=128, description="用户密码")
    confirm_password: str = Field(..., description="确认密码")
    
    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if v != info.data.get("password"):
            raise ValueError("两次输入的密码不匹配")
        return v


class UserLogin(BaseModel):
    """用户登录请求"""
    email: EmailStr = Field(..., description="用户邮箱地址")
    password: str = Field(..., description="用户密码")


class UserResponse(BaseModel):
    """用户响应"""
    id: str = Field(..., description="用户ID")
    email: str = Field(..., description="用户邮箱")
    role: str = Field(..., description="用户角色")
    plan_tier: str = Field(..., description="套餐等级")
    is_verified: bool = Field(..., description="是否已验证")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        from_attributes = True


class UserUpdate(BaseModel):
    """用户更新请求"""
    email: Optional[EmailStr] = Field(None, description="新邮箱地址")
    role: Optional[str] = Field(None, description="用户角色")
    plan_tier: Optional[str] = Field(None, description="套餐等级")


# ==================== API密钥 Schema ====================

class APIKeyCreate(BaseModel):
    """API密钥创建请求"""
    name: str = Field(..., min_length=1, max_length=100, description="密钥名称")
    description: Optional[str] = Field(None, max_length=500, description="密钥描述")
    expires_at: Optional[datetime] = Field(None, description="过期时间")


class APIKeyResponse(BaseModel):
    """API密钥响应（不包含完整密钥）"""
    id: str = Field(..., description="密钥ID")
    name: str = Field(..., description="密钥名称")
    description: Optional[str] = Field(None, description="密钥描述")
    key_prefix: str = Field(..., description="密钥前缀")
    is_active: bool = Field(..., description="是否激活")
    expires_at: Optional[datetime] = Field(None, description="过期时间")
    request_count: int = Field(..., description="请求次数")
    last_used_at: Optional[datetime] = Field(None, description="最后使用时间")
    created_at: datetime = Field(..., description="创建时间")
    
    class Config:
        from_attributes = True


class APIKeyFullResponse(BaseModel):
    """完整API密钥响应（只在创建时返回一次）"""
    id: str = Field(..., description="密钥ID")
    key: str = Field(..., description="完整API密钥（请妥善保管）")
    name: str = Field(..., description="密钥名称")
    created_at: datetime = Field(..., description="创建时间")


# ==================== 压缩相关 Schema ====================

class CompressionRequest(BaseModel):
    """压缩请求"""
    content: str = Field(..., description="要压缩的知识内容")
    domain: str = Field(default="general", max_length=100, description="知识领域")
    language: str = Field(default="en", max_length=10, description="语言")
    compression_level: int = Field(default=5, ge=1, le=10, description="压缩级别")
    extract_entities: bool = Field(default=True, description="是否提取实体")
    extract_relationships: bool = Field(default=True, description="是否提取关系")
    extract_causality: bool = Field(default=True, description="是否提取因果关系")
    generate_embedding: bool = Field(default=True, description="是否生成嵌入向量")


class CompressionResponse(BaseModel):
    """压缩响应"""
    kernel_id: str = Field(..., description="内核ID")
    original_size: int = Field(..., description="原始大小(字节)")
    compressed_size: int = Field(..., description="压缩后大小(字节)")
    compression_ratio: float = Field(..., description="压缩比")
    compressed_content: str = Field(..., description="压缩后的内容")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="提取的实体")
    relationships: List[Dict[str, Any]] = Field(default_factory=list, description="提取的关系")
    causal_chains: List[Dict[str, Any]] = Field(default_factory=list, description="因果链")
    embedding_model: Optional[str] = Field(None, description="嵌入模型名称")
    processing_time_ms: int = Field(..., description="处理时间(毫秒)")
    created_at: datetime = Field(..., description="创建时间")


class BatchCompressionRequest(BaseModel):
    """批量压缩请求"""
    items: List[CompressionRequest] = Field(..., min_length=1, max_length=100, description="压缩项目列表")


class BatchCompressionResponse(BaseModel):
    """批量压缩响应"""
    results: List[CompressionResponse] = Field(..., description="压缩结果列表")
    total_original_size: int = Field(..., description="总原始大小")
    total_compressed_size: int = Field(..., description="总压缩后大小")
    average_compression_ratio: float = Field(..., description="平均压缩比")
    processing_time_ms: int = Field(..., description="总处理时间")


# ==================== 多模态压缩 Schema ====================

class ImageCompressionRequest(BaseModel):
    """图像压缩请求"""
    image_data: str = Field(..., description="Base64编码的图像数据")
    image_format: str = Field(default="jpeg", description="图像格式(jpeg/png/webp)")
    compression_level: int = Field(default=5, ge=1, le=10, description="压缩级别")
    extract_text: bool = Field(default=True, description="是否提取文本(OCR)")
    generate_caption: bool = Field(default=True, description="是否生成描述")
    generate_embedding: bool = Field(default=True, description="是否生成嵌入向量")


class ImageCompressionResponse(BaseModel):
    """图像压缩响应"""
    kernel_id: str = Field(..., description="内核ID")
    original_size: int = Field(..., description="原始大小")
    compressed_size: int = Field(..., description="压缩后大小")
    compression_ratio: float = Field(..., description="压缩比")
    extracted_text: Optional[str] = Field(None, description="OCR提取的文本")
    generated_caption: Optional[str] = Field(None, description="生成的描述")
    embedding_vector: Optional[List[float]] = Field(None, description="嵌入向量")
    processing_time_ms: int = Field(..., description="处理时间")


class AudioCompressionRequest(BaseModel):
    """音频压缩请求"""
    audio_data: str = Field(..., description="Base64编码的音频数据")
    audio_format: str = Field(default="wav", description="音频格式(wav/mp3/flac)")
    transcription_language: str = Field(default="en", description="转录语言")
    transcribe: bool = Field(default=True, description="是否转录")
    generate_summary: bool = Field(default=True, description="是否生成摘要")
    analyze_features: bool = Field(default=True, description="是否分析音频特征")


class AudioCompressionResponse(BaseModel):
    """音频压缩响应"""
    kernel_id: str = Field(..., description="内核ID")
    original_size: int = Field(..., description="原始大小")
    compressed_size: int = Field(..., description="压缩后大小")
    compression_ratio: float = Field(..., description="压缩比")
    transcription: Optional[str] = Field(None, description="语音转录文本")
    summary: Optional[str] = Field(None, description="音频摘要")
    audio_features: Optional[Dict[str, Any]] = Field(None, description="音频特征")
    processing_time_ms: int = Field(..., description="处理时间")


class VideoCompressionRequest(BaseModel):
    """视频压缩请求"""
    video_data: str = Field(..., description="Base64编码的视频数据")
    video_format: str = Field(default="mp4", description="视频格式(mp4/avi/mov)")
    extract_keyframes: bool = Field(default=True, description="是否提取关键帧")
    generate_transcription: bool = Field(default=True, description="是否生成转录")
    scene_detection: bool = Field(default=True, description="是否进行场景检测")


class VideoCompressionResponse(BaseModel):
    """视频压缩响应"""
    kernel_id: str = Field(..., description="内核ID")
    original_size: int = Field(..., description="原始大小")
    compressed_size: int = Field(..., description="压缩后大小")
    compression_ratio: float = Field(..., description="压缩比")
    scenes: List[Dict[str, Any]] = Field(default_factory=list, description="检测到的场景")
    keyframes: List[str] = Field(default_factory=list, description="关键帧Base64编码")
    transcription: Optional[str] = Field(None, description="视频转录")
    processing_time_ms: int = Field(..., description="处理时间")


class DocumentCompressionRequest(BaseModel):
    """文档压缩请求"""
    document_data: str = Field(..., description="Base64编码的文档数据")
    document_format: str = Field(default="pdf", description="文档格式(pdf/docx/pptx/txt)")
    extract_images: bool = Field(default=True, description="是否提取图像")
    extract_tables: bool = Field(default=True, description="是否提取表格")
    generate_summary: bool = Field(default=True, description="是否生成摘要")
    extract_entities: bool = Field(default=True, description="是否提取实体")


class DocumentCompressionResponse(BaseModel):
    """文档压缩响应"""
    kernel_id: str = Field(..., description="内核ID")
    original_size: int = Field(..., description="原始大小")
    compressed_size: int = Field(..., description="压缩后大小")
    compression_ratio: float = Field(..., description="压缩比")
    content: str = Field(..., description="提取的文本内容")
    summary: Optional[str] = Field(None, description="文档摘要")
    images: List[str] = Field(default_factory=list, description="提取的图像Base64编码")
    tables: List[Dict[str, Any]] = Field(default_factory=list, description="提取的表格")
    entities: List[Dict[str, Any]] = Field(default_factory=list, description="提取的实体")
    processing_time_ms: int = Field(..., description="处理时间")


# ==================== 查询相关 Schema ====================

class QueryRequest(BaseModel):
    """查询请求"""
    query: str = Field(..., description="查询内容")
    kernel_ids: Optional[List[str]] = Field(None, description="要查询的内核ID列表")
    query_type: str = Field(default="semantic", description="查询类型(semantic/exact/fuzzy/hybrid)")
    top_k: int = Field(default=10, ge=1, le=100, description="返回结果数量")
    min_similarity: float = Field(default=0.7, ge=0.0, le=1.0, description="最小相似度")


class QueryResult(BaseModel):
    """查询结果"""
    kernel_id: str = Field(..., description="内核ID")
    kernel_id_short: str = Field(..., description="内核ID(短)")
    content: str = Field(..., description="内核内容")
    similarity_score: float = Field(..., description="相似度分数")
    rank: int = Field(..., description="排名")
    highlights: List[str] = Field(default_factory=list, description="匹配片段")


class QueryResponse(BaseModel):
    """查询响应"""
    results: List[QueryResult] = Field(..., description="查询结果")
    total_found: int = Field(..., description="找到的结果总数")
    query_time_ms: int = Field(..., description="查询时间(毫秒)")
    query_type: str = Field(..., description="使用的查询类型")


class MergeRequest(BaseModel):
    """合并请求"""
    kernel_ids: List[str] = Field(..., min_length=2, max_length=50, description="要合并的内核ID列表")
    merge_strategy: str = Field(default="concatenate", description="合并策略(concatenate/semantic)")
    resolve_conflicts: bool = Field(default=True, description="是否解决冲突")


class MergeResponse(BaseModel):
    """合并响应"""
    new_kernel_id: str = Field(..., description="新内核ID")
    source_kernel_ids: List[str] = Field(..., description="源内核ID列表")
    merged_content: str = Field(..., description="合并后的内容")
    conflicts_resolved: int = Field(default=0, description="解决的冲突数量")
    compression_ratio: float = Field(..., description="压缩比")


class DistillRequest(BaseModel):
    """蒸馏请求"""
    kernel_id: str = Field(..., description="要蒸馏的内核ID")
    target_model: str = Field(default="llama-2-7b", description="目标模型")
    distillation_level: int = Field(default=5, ge=1, le=10, description="蒸馏级别")


class DistillResponse(BaseModel):
    """蒸馏响应"""
    kernel_id: str = Field(..., description="源内核ID")
    distilled_kernel_id: str = Field(..., description="蒸馏后内核ID")
    target_model: str = Field(..., description="目标模型")
    original_size: int = Field(..., description="原始大小")
    distilled_size: int = Field(..., description="蒸馏后大小")
    distillation_ratio: float = Field(..., description="蒸馏比")
    processing_time_ms: int = Field(..., description="处理时间")


# ==================== 任务相关 Schema ====================

class TaskStatus(BaseModel):
    """任务状态"""
    job_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    progress: int = Field(default=0, description="进度(0-100)")
    created_at: datetime = Field(..., description="创建时间")
    started_at: Optional[datetime] = Field(None, description="开始时间")
    completed_at: Optional[datetime] = Field(None, description="完成时间")
    error_message: Optional[str] = Field(None, description="错误信息")


# ==================== 统计相关 Schema ====================

class UsageStats(BaseModel):
    """使用统计"""
    total_kernels: int = Field(..., description="内核总数")
    total_queries: int = Field(..., description="查询总数")
    total_compressed_bytes: int = Field(..., description="总压缩字节数")
    total_original_bytes: int = Field(..., description="总原始字节数")
    average_compression_ratio: float = Field(..., description="平均压缩比")
    average_query_time_ms: float = Field(..., description="平均查询时间")


class PerformanceMetrics(BaseModel):
    """性能指标"""
    uptime_seconds: int = Field(..., description="运行时间")
    total_requests: int = Field(..., description="总请求数")
    successful_requests: int = Field(..., description="成功请求数")
    failed_requests: int = Field(..., description="失败请求数")
    average_response_time_ms: float = Field(..., description="平均响应时间")
    requests_per_minute: float = Field(..., description="每分钟请求数")
    active_connections: int = Field(..., description="活跃连接数")


# ==================== 通用 Schema ====================

class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    version: str = Field(..., description="API版本")
    database: str = Field(..., description="数据库状态")
    cache: str = Field(..., description="缓存状态")
    timestamp: datetime = Field(..., description="检查时间")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="错误时间")


class PaginationParams(BaseModel):
    """分页参数"""
    page: int = Field(default=1, ge=1, description="页码")
    page_size: int = Field(default=20, ge=1, le=100, description="每页数量")


class PaginatedResponse(BaseModel):
    """分页响应"""
    items: List[Any] = Field(..., description="数据项")
    total: int = Field(..., description="总数")
    page: int = Field(..., description="当前页码")
    page_size: int = Field(..., description="每页数量")
    total_pages: int = Field(..., description="总页数")
    has_next: bool = Field(..., description="是否有下一页")
    has_previous: bool = Field(..., description="是否有上一页")


# ==================== 令牌 Schema ====================

class Token(BaseModel):
    """访问令牌响应"""
    access_token: str = Field(..., description="访问令牌")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")
    expires_in: int = Field(..., description="过期时间(秒)")


class TokenRefresh(BaseModel):
    """令牌刷新请求"""
    refresh_token: str = Field(..., description="刷新令牌")


class TokenPayload(BaseModel):
    """令牌载荷"""
    sub: str = Field(..., description="用户ID")
    email: Optional[str] = Field(None, description="用户邮箱")
    role: str = Field(default="user", description="用户角色")
    exp: Optional[int] = Field(None, description="过期时间戳")
