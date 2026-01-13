"""
KERNELIZE Platform - Security Module
======================================

This module provides comprehensive security functionality including JWT
authentication, API key management, password hashing, and access control.
Designed for enterprise-grade security requirements.

Author: KERNELIZE Team
Version: 1.0.0
License: Apache-2.0
"""

import hashlib
import logging
import secrets
import time
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from .config import settings

logger = logging.getLogger(__name__)

# 密码加密上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class SecurityManager:
    """
    安全管理器
    
    提供JWT令牌生成和验证、密码哈希、API密钥管理等安全功能。
    """
    
    def __init__(self):
        self.secret_key = settings.security.secret_key
        self.algorithm = settings.security.algorithm
        self.access_token_expire = timedelta(minutes=settings.security.access_token_expire_minutes)
        self.refresh_token_expire = timedelta(days=settings.security.refresh_token_expire_days)
        self.bcrypt_rounds = settings.security.bcrypt_rounds
    
    # ==================== 密码处理 ====================
    
    def hash_password(self, password: str) -> str:
        """
        哈希密码
        
        使用bcrypt算法对密码进行安全哈希，支持配置加密轮数。
        """
        return pwd_context.hash(password, rounds=self.bcrypt_rounds)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """
        验证密码
        
        将明文密码与哈希密码进行比对。
        """
        return pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> tuple[bool, str]:
        """
        验证密码强度
        
        检查密码是否满足最低安全要求：
        - 至少8个字符
        - 包含小写字母
        - 包含大写字母
        - 包含数字
        - 包含特殊字符
        """
        if len(password) < 8:
            return False, "密码长度至少8个字符"
        
        if not any(c.islower() for c in password):
            return False, "密码必须包含小写字母"
        
        if not any(c.isupper() for c in password):
            return False, "密码必须包含大写字母"
        
        if not any(c.isdigit() for c in password):
            return False, "密码必须包含数字"
        
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        if not any(c in special_chars for c in password):
            return False, "密码必须包含特殊字符"
        
        return True, "密码强度合格"
    
    # ==================== JWT令牌 ====================
    
    def create_access_token(
        self,
        user_id: str,
        email: Optional[str] = None,
        role: str = "user",
        additional_claims: Optional[dict] = None,
    ) -> str:
        """
        创建访问令牌
        
        生成JWT访问令牌，包含用户身份信息和自定义声明。
        """
        expire = datetime.utcnow() + self.access_token_expire
        
        payload = {
            "sub": user_id,
            "email": email,
            "role": role,
            "type": "access",
            "exp": expire,
            "iat": datetime.utcnow(),
        }
        
        if additional_claims:
            payload.update(additional_claims)
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def create_refresh_token(self, user_id: str) -> str:
        """
        创建刷新令牌
        
        生成JWT刷新令牌，用于获取新的访问令牌。
        """
        expire = datetime.utcnow() + self.refresh_token_expire
        
        payload = {
            "sub": user_id,
            "type": "refresh",
            "exp": expire,
            "iat": datetime.utcnow(),
        }
        
        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return token
    
    def verify_token(self, token: str) -> Optional[dict]:
        """
        验证令牌
        
        解码并验证JWT令牌，返回载荷或None（如果无效）。
        """
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            logger.warning(f"Token verification failed: {e}")
            return None
    
    def decode_token(self, token: str) -> Optional[dict]:
        """
        解码令牌（不验证过期时间）
        
        仅解码JWT令牌内容，不验证签名和过期。
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                options={"verify_exp": False},
            )
            return payload
        except JWTError:
            return None
    
    def get_token_expiration(self, token: str) -> Optional[datetime]:
        """
        获取令牌过期时间
        """
        payload = self.decode_token(token)
        if payload and "exp" in payload:
            return datetime.fromtimestamp(payload["exp"])
        return None
    
    # ==================== API密钥 ====================
    
    def generate_api_key(self) -> tuple[str, str, str]:
        """
        生成API密钥
        
        创建新的API密钥，返回完整密钥、前缀和哈希值。
        格式: kz_<random32>_<timestamp>
        """
        random_part = secrets.token_urlsafe(24)
        timestamp = int(time.time())
        key_string = f"kz_{random_part}_{timestamp}"
        
        key_hash = self.hash_api_key(key_string)
        key_prefix = f"kz_{random_part[:8]}"
        
        return key_string, key_prefix, key_hash
    
    def hash_api_key(self, api_key: str) -> str:
        """
        哈希API密钥
        
        使用SHA-256对API密钥进行单向哈希，用于存储。
        """
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def verify_api_key(self, api_key: str, stored_hash: str) -> bool:
        """
        验证API密钥
        
        将提供的API密钥与存储的哈希值进行比对。
        """
        return self.hash_api_key(api_key) == stored_hash
    
    def rotate_api_key(self, current_key: str) -> tuple[str, str, str]:
        """
        轮换API密钥
        
        生成新的API密钥，同时保持与旧密钥的关联（用于过渡期）。
        """
        return self.generate_api_key()
    
    # ==================== 访问控制 ====================
    
    def check_role_permission(
        self,
        user_role: str,
        required_role: str,
        hierarchical: bool = True,
    ) -> bool:
        """
        检查角色权限
        
        验证用户角色是否满足所需的角色权限。
        支持层级角色检查（如admin包含user的权限）。
        """
        role_hierarchy = {
            "superadmin": ["superadmin", "admin", "moderator", "user", "viewer"],
            "admin": ["admin", "moderator", "user", "viewer"],
            "moderator": ["moderator", "user", "viewer"],
            "user": ["user", "viewer"],
            "viewer": ["viewer"],
        }
        
        if hierarchical:
            allowed_roles = role_hierarchy.get(user_role, [user_role])
            return required_role in allowed_roles
        else:
            return user_role == required_role
    
    def create_api_key_hash(self) -> str:
        """
        创建API密钥哈希
        
        生成安全的随机字符串用于API密钥存储。
        """
        return secrets.token_hex(32)
    
    def mask_api_key(self, api_key: str) -> str:
        """
        遮蔽API密钥
        
        返回格式化的API密钥，仅显示前8个字符。
        """
        if len(api_key) <= 12:
            return "*" * len(api_key)
        return f"{api_key[:8]}...{api_key[-4:]}"
    
    # ==================== 安全审计 ====================
    
    def create_audit_entry(
        self,
        user_id: str,
        action: str,
        resource: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> dict:
        """
        创建审计日志条目
        
        生成标准化的安全审计日志条目。
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "details": details or {},
        }


# 全局安全管理器实例
security_manager = SecurityManager()


# 便捷函数
def hash_password(password: str) -> str:
    """快捷密码哈希函数"""
    return security_manager.hash_password(password)


def verify_password(plain: str, hashed: str) -> bool:
    """快捷密码验证函数"""
    return security_manager.verify_password(plain, hashed)


def create_access_token(
    user_id: str,
    email: Optional[str] = None,
    role: str = "user",
) -> str:
    """快捷访问令牌创建函数"""
    return security_manager.create_access_token(user_id, email, role)


def create_refresh_token(user_id: str) -> str:
    """快捷刷新令牌创建函数"""
    return security_manager.create_refresh_token(user_id)


def verify_token(token: str) -> Optional[dict]:
    """快捷令牌验证函数"""
    return security_manager.verify_token(token)
