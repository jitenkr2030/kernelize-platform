"""
KERNELIZE Platform - Main Application
=======================================

This is the main FastAPI application for the KERNELIZE Knowledge
Compression Infrastructure. It provides a comprehensive API for
knowledge compression, semantic search, and kernel management.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from pydantic import ValidationError

from .core.config import settings
from .core.database import db_manager, init_db, close_db
from .core.security import security_manager
from .models.schemas import (
    CompressionRequest, CompressionResponse,
    QueryRequest, QueryResponse, QueryResult,
    HealthResponse, ErrorResponse, Token,
    UserCreate, UserLogin, UserResponse,
    APIKeyCreate, APIKeyResponse, APIKeyFullResponse,
)

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format=settings.logging.format,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Prometheus指标
REQUEST_COUNT = Counter(
    'kernelize_requests_total',
    'Total number of requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'kernelize_request_duration_seconds',
    'Request latency in seconds',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)
ACTIVE_CONNECTIONS = Gauge(
    'kernelize_active_connections',
    'Number of active connections'
)
COMPRESSION_RATIO = Histogram(
    'kernelize_compression_ratio',
    'Compression ratio distribution',
    buckets=[1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Starting KERNELIZE Platform...")
    
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
    
    ACTIVE_CONNECTIONS.set(0)
    logger.info("KERNELIZE Platform started successfully")
    
    yield
    
    # 关闭时
    logger.info("Shutting down KERNELIZE Platform...")
    await close_db()
    logger.info("KERNELIZE Platform shut down complete")


# 创建FastAPI应用
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    docs_url=settings.api.docs_url,
    redoc_url=settings.api.redoc_url,
    openapi_url=settings.api.openapi_url,
    lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 中间件：请求跟踪
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """跟踪请求并记录指标"""
    ACTIVE_CONNECTIONS.inc()
    
    start_time = datetime.utcnow()
    
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise
    finally:
        ACTIVE_CONNECTIONS.dec()
    
    # 记录指标
    endpoint = request.url.path
    method = request.method
    
    REQUEST_COUNT.labels(
        method=method,
        endpoint=endpoint,
        status=str(status_code)
    ).inc()
    
    latency = (datetime.utcnow() - start_time).total_seconds()
    REQUEST_LATENCY.labels(
        method=method,
        endpoint=endpoint
    ).observe(latency)
    
    return response


# 异常处理
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """处理验证错误"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "请求数据验证失败",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """处理HTTP错误"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "message": str(exc.detail),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理通用错误"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "服务器内部错误",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# 依赖注入
async def get_current_user(request: Request) -> Dict[str, Any]:
    """获取当前用户"""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="未提供认证令牌")
    
    token = auth_header.split(" ")[1]
    payload = security_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="无效或过期的令牌")
    
    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role", "user"),
    }


# ==================== 健康检查端点 ====================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    健康检查端点
    
    返回服务状态、版本信息和各组件健康状况。
    """
    # 检查数据库
    try:
        db_healthy = await db_manager.health_check()
    except Exception:
        db_healthy = False
    
    return HealthResponse(
        status="healthy" if db_healthy else "degraded",
        version=settings.api.version,
        database="healthy" if db_healthy else "unhealthy",
        cache="healthy",
        timestamp=datetime.utcnow(),
    )


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus指标端点"""
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ==================== 认证端点 ====================

@app.post("/auth/register", response_model=Dict[str, Any], tags=["Authentication"])
async def register(user_data: UserCreate):
    """
    用户注册
    
    创建新用户账户。
    """
    logger.info(f"User registration attempt: {user_data.email}")
    
    # 验证密码强度
    is_valid, message = security_manager.validate_password_strength(user_data.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # 检查密码是否匹配
    if user_data.password != user_data.confirm_password:
        raise HTTPException(status_code=400, detail="两次输入的密码不匹配")
    
    # 创建用户（实际实现需要数据库操作）
    user_id = str(user_data.email.split("@")[0])
    
    # 生成令牌
    access_token = security_manager.create_access_token(user_id, user_data.email)
    refresh_token = security_manager.create_refresh_token(user_id)
    
    logger.info(f"User registered successfully: {user_data.email}")
    
    return {
        "message": "用户注册成功",
        "user_id": user_id,
        "token": Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.security.access_token_expire_minutes * 60,
        ).model_dump(),
    }


@app.post("/auth/login", response_model=Dict[str, Any], tags=["Authentication"])
async def login(credentials: UserLogin):
    """
    用户登录
    
    验证用户凭据并返回访问令牌。
    """
    logger.info(f"Login attempt: {credentials.email}")
    
    # 验证用户（实际实现需要数据库查询）
    # 这里简化处理
    user_id = str(credentials.email.split("@")[0])
    
    # 生成令牌
    access_token = security_manager.create_access_token(user_id, credentials.email)
    refresh_token = security_manager.create_refresh_token(user_id)
    
    logger.info(f"User logged in successfully: {credentials.email}")
    
    return {
        "message": "登录成功",
        "user_id": user_id,
        "token": Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.security.access_token_expire_minutes * 60,
        ).model_dump(),
    }


# ==================== 压缩端点 ====================

@app.post("/v2/compress", response_model=CompressionResponse, tags=["Compression"])
async def compress_knowledge(
    request: CompressionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    压缩知识内容
    
    将原始知识内容压缩为语义内核，支持多种压缩级别和选项。
    
    - **content**: 要压缩的知识内容
    - **domain**: 知识领域（general, healthcare, finance, legal等）
    - **compression_level**: 压缩级别（1-10）
    - **extract_entities**: 是否提取实体
    - **extract_relationships**: 是否提取关系
    - **extract_causality**: 是否提取因果链
    """
    from .services.compression_engine import compression_engine
    
    logger.info(f"Compression request from user: {current_user['user_id']}")
    
    # 执行压缩
    result = compression_engine.compress(
        content=request.content,
        domain=request.domain,
        language=request.language,
        compression_level=request.compression_level,
        extract_entities=request.extract_entities,
        extract_relationships=request.extract_relationships,
        extract_causality=request.extract_causality,
        generate_embedding=request.generate_embedding,
    )
    
    # 记录压缩比
    COMPRESSION_RATIO.observe(result.compression_ratio)
    
    return CompressionResponse(
        kernel_id=result.kernel_id,
        original_size=result.original_size,
        compressed_size=result.compressed_size,
        compression_ratio=result.compression_ratio,
        compressed_content=result.compressed_content,
        entities=result.entities,
        relationships=result.relationships,
        causal_chains=result.causal_chains,
        embedding_model=result.embedding_model,
        processing_time_ms=result.processing_time_ms,
        created_at=datetime.utcnow(),
    )


@app.post("/v2/compress/batch", tags=["Compression"])
async def batch_compress(
    requests: list[CompressionRequest],
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    批量压缩
    
    同时压缩多个知识内容。
    """
    from .services.compression_engine import compression_engine
    
    logger.info(f"Batch compression request: {len(requests)} items")
    
    results = []
    total_original = 0
    total_compressed = 0
    
    for request in requests:
        result = compression_engine.compress(
            content=request.content,
            domain=request.domain,
            language=request.language,
            compression_level=request.compression_level,
            extract_entities=request.extract_entities,
            extract_relationships=request.extract_relationships,
            extract_causality=request.extract_causality,
        )
        
        results.append(CompressionResponse(
            kernel_id=result.kernel_id,
            original_size=result.original_size,
            compressed_size=result.compressed_size,
            compression_ratio=result.compression_ratio,
            compressed_content=result.compressed_content,
            entities=result.entities,
            relationships=result.relationships,
            causal_chains=result.causal_chains,
            embedding_model=result.embedding_model,
            processing_time_ms=result.processing_time_ms,
            created_at=datetime.utcnow(),
        ))
        
        total_original += result.original_size
        total_compressed += result.compressed_size
    
    avg_ratio = total_original / total_compressed if total_compressed > 0 else 1.0
    
    return {
        "results": [r.model_dump() for r in results],
        "total_original_size": total_original,
        "total_compressed_size": total_compressed,
        "average_compression_ratio": avg_ratio,
    }


# ==================== 查询端点 ====================

@app.post("/v2/kernels/{kernel_id}/query", response_model=QueryResponse, tags=["Query"])
async def query_kernel(
    kernel_id: str,
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    查询知识内核
    
    在指定的内核中执行语义搜索。
    """
    from .services.query_engine import query_engine, QueryType
    
    logger.info(f"Query request for kernel: {kernel_id}")
    
    results, metrics = query_engine.query(
        query_text=request.query,
        kernel_ids=[kernel_id],
        query_type=QueryType(request.query_type),
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )
    
    return QueryResponse(
        results=[
            QueryResult(
                kernel_id=r.kernel_id,
                content=r.content,
                similarity_score=r.similarity_score,
                rank=r.rank,
                highlights=r.highlights,
            ).model_dump()
            for r in results
        ],
        total_found=metrics.total_results,
        query_time_ms=metrics.query_time_ms,
        query_type=request.query_type,
    )


@app.post("/v2/query", response_model=QueryResponse, tags=["Query"])
async def query_all(
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    全局查询
    
    在所有知识内核中执行语义搜索。
    """
    from .services.query_engine import query_engine, QueryType
    
    logger.info(f"Global query request: {request.query[:100]}...")
    
    results, metrics = query_engine.query(
        query_text=request.query,
        kernel_ids=request.kernel_ids,
        query_type=QueryType(request.query_type),
        top_k=request.top_k,
        min_similarity=request.min_similarity,
    )
    
    return QueryResponse(
        results=[
            {
                "kernel_id": r.kernel_id,
                "content": r.content,
                "similarity_score": r.similarity_score,
                "rank": r.rank,
                "highlights": r.highlights,
            }
            for r in results
        ],
        total_found=metrics.total_results,
        query_time_ms=metrics.query_time_ms,
        query_type=request.query_type,
    )


# ==================== 内核管理端点 ====================

@app.get("/v2/kernels/{kernel_id}", tags=["Kernels"])
async def get_kernel(
    kernel_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    获取知识内核
    
    根据ID检索知识内核的详细信息。
    """
    # 实际实现需要从数据库获取
    return {
        "kernel_id": kernel_id,
        "status": "available",
        "created_at": datetime.utcnow().isoformat(),
    }


@app.delete("/v2/kernels/{kernel_id}", tags=["Kernels"])
async def delete_kernel(
    kernel_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    删除知识内核
    
    根据ID删除指定的知识内核。
    """
    from .services.query_engine import query_engine
    
    success = query_engine.delete_kernel(kernel_id)
    
    if success:
        return {"message": "内核删除成功", "kernel_id": kernel_id}
    else:
        raise HTTPException(status_code=404, detail="内核不存在")


# ==================== 统计端点 ====================

@app.get("/v2/stats/usage", tags=["Analytics"])
async def get_usage_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    获取使用统计
    
    返回当前用户的使用统计信息。
    """
    from .services.query_engine import query_engine
    
    stats = query_engine.get_stats()
    
    return {
        "total_kernels": stats["indexed_kernels"],
        "total_queries": 0,
        "cache_stats": stats["cache_stats"],
    }


@app.get("/v2/stats/performance", tags=["Analytics"])
async def get_performance_stats():
    """
    获取性能统计
    
    返回系统性能指标。
    """
    return {
        "uptime_seconds": 0,
        "total_requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "average_response_time_ms": 0,
        "requests_per_minute": 0,
        "active_connections": 0,
    }


# ==================== 根端点 ====================

@app.get("/", tags=["Root"])
async def root():
    """
    API根端点
    
    返回API基本信息和使用指南。
    """
    return {
        "name": settings.api.title,
        "version": settings.api.version,
        "description": settings.api.description,
        "docs": settings.api.docs_url,
        "health": "/health",
        "metrics": "/metrics",
    }


# 启动应用（用于本地开发）
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
    )
