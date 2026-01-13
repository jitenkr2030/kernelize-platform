"""
KERNELIZE Platform - Backend Package
=====================================

The KERNELIZE Knowledge Compression Infrastructure backend provides
a comprehensive API for compressing, storing, searching, and reasoning
over knowledge with unprecedented efficiency.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0

Features:
- Semantic Knowledge Compression (100×-10,000×)
- Sub-millisecond Semantic Search
- Multi-Modal Processing (images, video, audio, documents)
- Enterprise-Grade Security (JWT, API Keys, RBAC)
- Production Monitoring (Prometheus, Health Checks)

Quick Start:
    from kernelize.backend.main import app
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""

from .main import app
from .core.config import settings, get_settings
from .core.database import db_manager, get_db, init_db, close_db
from .core.security import security_manager
from .models.database import User, APIKey, Kernel, CompressionJob, QueryLog
from .services.compression_engine import compression_engine
from .services.query_engine import query_engine

__version__ = "1.0.0"

__all__ = [
    # Application
    "app",
    
    # Configuration
    "settings",
    "get_settings",
    
    # Database
    "db_manager",
    "get_db",
    "init_db",
    "close_db",
    
    # Security
    "security_manager",
    
    # Models
    "User",
    "APIKey",
    "Kernel",
    "CompressionJob",
    "QueryLog",
    
    # Services
    "compression_engine",
    "query_engine",
]
