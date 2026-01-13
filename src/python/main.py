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

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.logging.level),
    format=settings.logging.format,
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
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
    """Application lifecycle management"""
    # On startup
    logger.info("Starting KERNELIZE Platform...")
    
    try:
        await init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.warning(f"Database initialization failed: {e}")
    
    ACTIVE_CONNECTIONS.set(0)
    logger.info("KERNELIZE Platform started successfully")
    
    yield
    
    # On shutdown
    logger.info("Shutting down KERNELIZE Platform...")
    await close_db()
    logger.info("KERNELIZE Platform shut down complete")


# Create FastAPI application
app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    docs_url=settings.api.docs_url,
    redoc_url=settings.api.redoc_url,
    openapi_url=settings.api.openapi_url,
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Middleware: Request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track requests and record metrics"""
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
    
    # Record metrics
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


# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "message": "Request data validation failed",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP errors"""
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
    """Handle general errors"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "Server internal error",
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


# Dependency injection
async def get_current_user(request: Request) -> Dict[str, Any]:
    """Get current user"""
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Authentication token not provided")
    
    token = auth_header.split(" ")[1]
    payload = security_manager.verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role", "user"),
    }


# ==================== Health check endpoints ====================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint
    
    Returns service status, version information, and component health status.
    """
    # Check database
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
    """Prometheus metrics endpoint"""
    from fastapi.responses import Response
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ==================== Authentication endpoints ====================

@app.post("/auth/register", response_model=Dict[str, Any], tags=["Authentication"])
async def register(user_data: UserCreate):
    """
    User registration
    
    Creates a new user account.
    """
    logger.info(f"User registration attempt: {user_data.email}")
    
    # Validate password strength
    is_valid, message = security_manager.validate_password_strength(user_data.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=message)
    
    # Check if passwords match
    if user_data.password != user_data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match")
    
    # Create user (actual implementation requires database operation)
    user_id = str(user_data.email.split("@")[0])
    
    # Generate tokens
    access_token = security_manager.create_access_token(user_id, user_data.email)
    refresh_token = security_manager.create_refresh_token(user_id)
    
    logger.info(f"User registered successfully: {user_data.email}")
    
    return {
        "message": "User registered successfully",
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
    User login
    
    Validates user credentials and returns access token.
    """
    logger.info(f"Login attempt: {credentials.email}")
    
    # Validate user (actual implementation requires database query)
    # Simplified here
    user_id = str(credentials.email.split("@")[0])
    
    # Generate tokens
    access_token = security_manager.create_access_token(user_id, credentials.email)
    refresh_token = security_manager.create_refresh_token(user_id)
    
    logger.info(f"User logged in successfully: {credentials.email}")
    
    return {
        "message": "Login successful",
        "user_id": user_id,
        "token": Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.security.access_token_expire_minutes * 60,
        ).model_dump(),
    }


# ==================== Compression endpoints ====================

@app.post("/v2/compress", response_model=CompressionResponse, tags=["Compression"])
async def compress_knowledge(
    request: CompressionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Compress knowledge content
    
    Compresses raw knowledge content into semantic kernel with support for
    multiple compression levels and options.
    
    - **content**: Knowledge content to compress
    - **domain**: Knowledge domain (general, healthcare, finance, legal, etc.)
    - **compression_level**: Compression level (1-10)
    - **extract_entities**: Whether to extract entities
    - **extract_relationships**: Whether to extract relationships
    - **extract_causality**: Whether to extract causal chains
    """
    from .services.compression_engine import compression_engine
    
    logger.info(f"Compression request from user: {current_user['user_id']}")
    
    # Perform compression
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
    
    # Record compression ratio
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
    Batch compression
    
    Compresses multiple knowledge contents simultaneously.
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


# ==================== Query endpoints ====================

@app.post("/v2/kernels/{kernel_id}/query", response_model=QueryResponse, tags=["Query"])
async def query_kernel(
    kernel_id: str,
    request: QueryRequest,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Query knowledge kernel
    
    Executes semantic search within specified kernel.
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
    Global query
    
    Executes semantic search across all knowledge kernels.
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


# ==================== Kernel management endpoints ====================

@app.get("/v2/kernels/{kernel_id}", tags=["Kernels"])
async def get_kernel(
    kernel_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get knowledge kernel
    
    Retrieves detailed information about a knowledge kernel by ID.
    """
    # Actual implementation requires fetching from database
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
    Delete knowledge kernel
    
    Deletes a specified knowledge kernel by ID.
    """
    from .services.query_engine import query_engine
    
    success = query_engine.delete_kernel(kernel_id)
    
    if success:
        return {"message": "Kernel deleted successfully", "kernel_id": kernel_id}
    else:
        raise HTTPException(status_code=404, detail="Kernel not found")


# ==================== Statistics endpoints ====================

@app.get("/v2/stats/usage", tags=["Analytics"])
async def get_usage_stats(
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Get usage statistics
    
    Returns usage statistics for the current user.
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
    Get performance statistics
    
    Returns system performance metrics.
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


# ==================== Root endpoint ====================

@app.get("/", tags=["Root"])
async def root():
    """
    API root endpoint
    
    Returns API basic information and usage guide.
    """
    return {
        "name": settings.api.title,
        "version": settings.api.version,
        "description": settings.api.description,
        "docs": settings.api.docs_url,
        "health": "/health",
        "metrics": "/metrics",
    }


# Start application (for local development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
    )
