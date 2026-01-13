"""
KERNELIZE Platform - Core Configuration
=========================================

This module provides centralized configuration management for the KERNELIZE
Knowledge Compression Infrastructure. All settings are loaded from environment
variables with sensible defaults for development and production environments.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class DatabaseConfig(BaseModel):
    """Database configuration settings"""
    host: str = Field(default="localhost", description="Database host address")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    name: str = Field(default="kernelize", description="Database name")
    user: str = Field(default="kernelize", description="Database username")
    password: str = Field(default="kernelize_secure", description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Maximum overflow connections")
    
    @property
    def async_url(self) -> str:
        """Get async database connection URL"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        """Get sync database connection URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseModel):
    """Redis configuration settings"""
    host: str = Field(default="localhost", description="Redis host address")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis port")
    db: int = Field(default=0, ge=0, le=15, description="Redis database number")
    password: Optional[str] = Field(default=None, description="Redis password")
    
    @property
    def url(self) -> str:
        """Get Redis connection URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class SecurityConfig(BaseModel):
    """Security configuration settings"""
    secret_key: str = Field(default="kernelize-secret-key-change-in-production", description="JWT secret key")
    algorithm: str = Field(default="HS256", description="JWT algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=1, description="Access token expiry time")
    refresh_token_expire_days: int = Field(default=7, ge=1, description="Refresh token expiry days")
    bcrypt_rounds: int = Field(default=12, ge=4, le=31, description="Bcrypt encryption rounds")
    api_key_header: str = Field(default="X-API-Key", description="API key header name")


class CompressionConfig(BaseModel):
    """Compression engine configuration"""
    default_compression_level: int = Field(default=5, ge=1, le=10, description="Default compression level")
    max_input_size_mb: int = Field(default=100, ge=1, description="Maximum input size (MB)")
    max_output_size_mb: int = Field(default=10, ge=1, description="Maximum output size (MB)")
    semantic_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="Semantic similarity threshold")
    cache_enabled: bool = Field(default=True, description="Enable compression cache")
    cache_ttl_hours: int = Field(default=24, ge=1, description="Cache expiry time (hours)")


class ModelConfig(BaseModel):
    """AI model configuration"""
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model name")
    embedding_dimensions: int = Field(default=384, description="Embedding vector dimensions")
    max_sequence_length: int = Field(default=512, description="Maximum sequence length")
    device: str = Field(default="cpu", description="Compute device (cpu/cuda)")
    batch_size: int = Field(default=32, ge=1, description="Batch processing size")


class APIConfig(BaseModel):
    """API configuration settings"""
    title: str = Field(default="KERNELIZE Knowledge Compression API", description="API title")
    description: str = Field(default="World's first Knowledge Compression Infrastructure API", description="API description")
    version: str = Field(default="1.0.0", description="API version")
    docs_url: str = Field(default="/docs", description="API documentation URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc documentation URL")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI schema URL")
    rate_limit_requests: int = Field(default=1000, ge=1, description="Rate limit requests")
    rate_limit_window_seconds: int = Field(default=60, ge=1, description="Rate limit window (seconds)")


class LoggingConfig(BaseModel):
    """Logging configuration settings"""
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(default="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s", description="Log format")
    file_path: Optional[str] = Field(default=None, description="Log file path")
    max_file_size_mb: int = Field(default=100, ge=1, description="Maximum log file size (MB)")
    backup_count: int = Field(default=5, ge=1, description="Log backup count")


class MonitoringConfig(BaseModel):
    """Monitoring configuration settings"""
    enabled: bool = Field(default=True, description="Enable monitoring")
    metrics_port: int = Field(default=9090, ge=1, le=65535, description="Metrics port")
    health_check_interval: int = Field(default=30, ge=1, description="Health check interval (seconds)")
    alert_webhook_url: Optional[str] = Field(default=None, description="Alert webhook URL")


class Settings(BaseModel):
    """Main settings class"""
    environment: str = Field(default="development", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # Allow environment variable overrides
    class Config:
        env_prefix = "KERNELIZE_"
        env_nested_delimiter = "__"
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            """Parse environment variable values"""
            if field_name in ["secret_key", "api_key_header"]:
                return raw_val
            if field_name in ["debug", "cache_enabled", "enabled"]:
                return raw_val.lower() in ("true", "1", "yes")
            if field_name in ["port", "pool_size", "max_overflow"]:
                return int(raw_val)
            if field_name in ["semantic_threshold"]:
                return float(raw_val)
            return raw_val


@lru_cache()
def get_settings() -> Settings:
    """Get global settings instance (singleton pattern)"""
    return Settings()


# Global settings instance
settings = get_settings()
