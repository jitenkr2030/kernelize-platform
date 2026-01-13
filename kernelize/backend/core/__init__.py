"""
KERNELIZE Platform - Core Module
=================================

This module contains the core infrastructure components including configuration
management, database connection handling, and security implementations.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

from .config import settings, get_settings

__all__ = ["settings", "get_settings"]
