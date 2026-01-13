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
    """数据库配置"""
    host: str = Field(default="localhost", description="数据库主机地址")
    port: int = Field(default=5432, ge=1, le=65535, description="数据库端口")
    name: str = Field(default="kernelize", description="数据库名称")
    user: str = Field(default="kernelize", description="数据库用户名")
    password: str = Field(default="kernelize_secure", description="数据库密码")
    pool_size: int = Field(default=10, ge=1, le=100, description="连接池大小")
    max_overflow: int = Field(default=20, ge=0, le=100, description="最大溢出连接数")
    
    @property
    def async_url(self) -> str:
        """获取异步数据库连接URL"""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
    
    @property
    def sync_url(self) -> str:
        """获取同步数据库连接URL"""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class RedisConfig(BaseModel):
    """Redis配置"""
    host: str = Field(default="localhost", description="Redis主机地址")
    port: int = Field(default=6379, ge=1, le=65535, description="Redis端口")
    db: int = Field(default=0, ge=0, le=15, description="Redis数据库编号")
    password: Optional[str] = Field(default=None, description="Redis密码")
    
    @property
    def url(self) -> str:
        """获取Redis连接URL"""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"


class SecurityConfig(BaseModel):
    """安全配置"""
    secret_key: str = Field(default="kernelize-secret-key-change-in-production", description="JWT密钥")
    algorithm: str = Field(default="HS256", description="JWT算法")
    access_token_expire_minutes: int = Field(default=30, ge=1, description="访问令牌过期时间")
    refresh_token_expire_days: int = Field(default=7, ge=1, description="刷新令牌过期时间")
    bcrypt_rounds: int = Field(default=12, ge=4, le=31, description="Bcrypt加密轮数")
    api_key_header: str = Field(default="X-API-Key", description="API密钥请求头名称")


class CompressionConfig(BaseModel):
    """压缩引擎配置"""
    default_compression_level: int = Field(default=5, ge=1, le=10, description="默认压缩级别")
    max_input_size_mb: int = Field(default=100, ge=1, description="最大输入大小(MB)")
    max_output_size_mb: int = Field(default=10, ge=1, description="最大输出大小(MB)")
    semantic_threshold: float = Field(default=0.85, ge=0.0, le=1.0, description="语义相似度阈值")
    cache_enabled: bool = Field(default=True, description="是否启用压缩缓存")
    cache_ttl_hours: int = Field(default=24, ge=1, description="缓存过期时间(小时)")


class ModelConfig(BaseModel):
    """AI模型配置"""
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="嵌入模型名称")
    embedding_dimensions: int = Field(default=384, description="嵌入向量维度")
    max_sequence_length: int = Field(default=512, description="最大序列长度")
    device: str = Field(default="cpu", description="运行设备(cpu/cuda)")
    batch_size: int = Field(default=32, ge=1, description="批处理大小")


class APIConfig(BaseModel):
    """API配置"""
    title: str = Field(default="KERNELIZE Knowledge Compression API", description="API标题")
    description: str = Field(default="World's first Knowledge Compression Infrastructure API", description="API描述")
    version: str = Field(default="1.0.0", description="API版本")
    docs_url: str = Field(default="/docs", description="API文档URL")
    redoc_url: str = Field(default="/redoc", description="ReDoc文档URL")
    openapi_url: str = Field(default="/openapi.json", description="OpenAPI模式URL")
    rate_limit_requests: int = Field(default=1000, ge=1, description="速率限制请求数")
    rate_limit_window_seconds: int = Field(default=60, ge=1, description="速率限制时间窗口(秒)")


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(default="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s", description="日志格式")
    file_path: Optional[str] = Field(default=None, description="日志文件路径")
    max_file_size_mb: int = Field(default=100, ge=1, description="最大日志文件大小(MB)")
    backup_count: int = Field(default=5, ge=1, description="日志备份数量")


class MonitoringConfig(BaseModel):
    """监控配置"""
    enabled: bool = Field(default=True, description="是否启用监控")
    metrics_port: int = Field(default=9090, ge=1, le=65535, description="指标端口")
    health_check_interval: int = Field(default=30, ge=1, description="健康检查间隔(秒)")
    alert_webhook_url: Optional[str] = Field(default=None, description="告警Webhook URL")


class Settings(BaseModel):
    """主设置类"""
    environment: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    compression: CompressionConfig = Field(default_factory=CompressionConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
    # 允许从环境变量覆盖设置
    class Config:
        env_prefix = "KERNELIZE_"
        env_nested_delimiter = "__"
        
        @classmethod
        def parse_env_var(cls, field_name: str, raw_val: str) -> Any:
            """解析环境变量值"""
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
    """获取全局设置实例（单例模式）"""
    return Settings()


# 全局设置实例
settings = get_settings()
