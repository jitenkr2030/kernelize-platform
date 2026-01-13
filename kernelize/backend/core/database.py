"""
KERNELIZE Platform - Database Connection Manager
==================================================

This module provides async database connection management for PostgreSQL
using SQLAlchemy. It implements connection pooling, session management,
and health checking for production deployments.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import AsyncAdaptedQueuePool, QueuePool

from .config import settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy声明基类"""
    pass


class DatabaseManager:
    """
    数据库连接管理器
    
    提供异步数据库连接池管理、会话创建、健康检查等功能。
    支持开发和生产环境的不同配置需求。
    """
    
    def __init__(self):
        self._async_engine = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        初始化数据库引擎和连接池
        
        根据环境配置创建适当的引擎实例，配置连接池参数，
        并设置必要的数据库事件监听器。
        """
        if self._initialized:
            logger.warning("Database manager already initialized")
            return
        
        db_config = settings.database
        
        # 创建异步引擎
        self._async_engine = create_async_engine(
            url=db_config.async_url,
            poolclass=AsyncAdaptedQueuePool if settings.environment == "production" else None,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.debug,
        )
        
        # 创建同步引擎（用于迁移和工具）
        self._sync_engine = create_engine(
            url=db_config.sync_url,
            poolclass=QueuePool,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.debug,
        )
        
        # 创建异步会话工厂
        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        # 创建同步会话工厂
        self._sync_session_factory = sessionmaker(
            bind=self._sync_engine,
            autocommit=False,
            autoflush=False,
        )
        
        # 设置SQLAlchemy事件监听器
        self._setup_event_listeners()
        
        self._initialized = True
        logger.info("Database manager initialized successfully")
    
    def _setup_event_listeners(self) -> None:
        """设置数据库事件监听器"""
        
        @event.listens_for(self._sync_engine, "connect")
        def set_session_vars(dbapi_connection, connection_record):
            """设置连接会话变量"""
            if settings.environment == "production":
                cursor = dbapi_connection.cursor()
                cursor.execute("SET statement_timeout = '60s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '60000'")
                cursor.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        获取异步数据库会话
        
        使用上下文管理器自动处理会话的创建和关闭，
        确保资源正确释放。
        """
        if not self._initialized:
            self.initialize()
        
        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    def get_sync_session(self) -> Session:
        """
        获取同步数据库会话
        
        注意：调用者负责手动提交和关闭会话。
        主要用于迁移脚本和管理工具。
        """
        if not self._initialized:
            self.initialize()
        
        return self._sync_session_factory()
    
    async def health_check(self) -> bool:
        """
        执行数据库健康检查
        
        尝试执行简单查询验证数据库连接是否正常。
        返回True表示健康，False表示存在连接问题。
        """
        if not self._initialized:
            return False
        
        try:
            async with self.get_async_session() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def close(self) -> None:
        """
        关闭所有数据库连接
        
        释放连接池资源，确保优雅关闭。
        """
        if self._async_engine:
            await self._async_engine.dispose()
        
        if self._sync_engine:
            self._sync_engine.dispose()
        
        self._initialized = False
        logger.info("Database connections closed")
    
    @property
    def is_initialized(self) -> bool:
        """检查数据库管理器是否已初始化"""
        return self._initialized


# 全局数据库管理器实例
db_manager = DatabaseManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI依赖注入函数
    
    提供异步数据库会话给API端点使用。
    符合FastAPI的依赖注入模式。
    """
    async with db_manager.get_async_session() as session:
        yield session


async def init_db() -> None:
    """初始化数据库表结构"""
    from ..models.database import Base
    
    if not db_manager.is_initialized:
        db_manager.initialize()
    
    async with db_manager.get_async_session() as session:
        await session.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """关闭数据库连接"""
    await db_manager.close()
