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
    """SQLAlchemy declarative base class"""
    pass


class DatabaseManager:
    """
    Database connection manager
    
    Provides async database connection pool management, session creation,
    and health checking for production deployments.
    """
    
    def __init__(self):
        self._async_engine = None
        self._sync_engine = None
        self._async_session_factory = None
        self._sync_session_factory = None
        self._initialized = False
    
    def initialize(self) -> None:
        """
        Initialize database engines and connection pools
        
        Creates appropriate engine instances based on environment configuration,
        configures connection pool parameters, and sets up database event listeners.
        """
        if self._initialized:
            logger.warning("Database manager already initialized")
            return
        
        db_config = settings.database
        
        # Create async engine
        self._async_engine = create_async_engine(
            url=db_config.async_url,
            poolclass=AsyncAdaptedQueuePool if settings.environment == "production" else None,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.debug,
        )
        
        # Create sync engine (for migrations and tools)
        self._sync_engine = create_engine(
            url=db_config.sync_url,
            poolclass=QueuePool,
            pool_size=db_config.pool_size,
            max_overflow=db_config.max_overflow,
            pool_pre_ping=True,
            pool_recycle=3600,
            echo=settings.debug,
        )
        
        # Create async session factory
        self._async_session_factory = async_sessionmaker(
            bind=self._async_engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autocommit=False,
            autoflush=False,
        )
        
        # Create sync session factory
        self._sync_session_factory = sessionmaker(
            bind=self._sync_engine,
            autocommit=False,
            autoflush=False,
        )
        
        # Set up SQLAlchemy event listeners
        self._setup_event_listeners()
        
        self._initialized = True
        logger.info("Database manager initialized successfully")
    
    def _setup_event_listeners(self) -> None:
        """Set up database event listeners"""
        
        @event.listens_for(self._sync_engine, "connect")
        def set_session_vars(dbapi_connection, connection_record):
            """Set connection session variables"""
            if settings.environment == "production":
                cursor = dbapi_connection.cursor()
                cursor.execute("SET statement_timeout = '60s'")
                cursor.execute("SET idle_in_transaction_session_timeout = '60000'")
                cursor.close()
    
    @asynccontextmanager
    async def get_async_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get async database session
        
        Use context manager to automatically handle session creation and closure,
        ensuring proper resource release.
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
        Get sync database session
        
        Note: Caller is responsible for manually committing and closing the session.
        Mainly used for migration scripts and management tools.
        """
        if not self._initialized:
            self.initialize()
        
        return self._sync_session_factory()
    
    async def health_check(self) -> bool:
        """
        Perform database health check
        
        Attempt to execute a simple query to verify database connectivity.
        Returns True if healthy, False if connection issues exist.
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
        Close all database connections
        
        Release connection pool resources to ensure graceful shutdown.
        """
        if self._async_engine:
            await self._async_engine.dispose()
        
        if self._sync_engine:
            self._sync_engine.dispose()
        
        self._initialized = False
        logger.info("Database connections closed")
    
    @property
    def is_initialized(self) -> bool:
        """Check if database manager is initialized"""
        return self._initialized


# Global database manager instance
db_manager = DatabaseManager()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency injection function
    
    Provide async database sessions for API endpoint usage.
    Follows FastAPI's dependency injection pattern.
    """
    async with db_manager.get_async_session() as session:
        yield session


async def init_db() -> None:
    """Initialize database table structure"""
    from ..models.database import Base
    
    if not db_manager.is_initialized:
        db_manager.initialize()
    
    async with db_manager.get_async_session() as session:
        await session.run_sync(Base.metadata.create_all)


async def close_db() -> None:
    """Close database connections"""
    await db_manager.close()
