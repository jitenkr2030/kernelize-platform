/**
 * KERNELIZE Platform - Core Backend
 * Licensed under the Business Source License 1.1 (BSL 1.1)
 * 
 * Copyright (c) 2026 KERNELIZE Platform. All rights reserved.
 * 
 * See LICENSE-CORE in the project root for license information.
 * See LICENSE-SDK for SDK and tool licensing terms.
 */

/**
 * KERNELIZE Platform - Core Backend
 * Licensed under the Business Source License 1.1 (BSL 1.1)
 * 
 * Copyright (c) 2026 KERNELIZE Platform. All rights reserved.
 * 
 * See LICENSE-CORE in the project root for license information.
 * See LICENSE-SDK for SDK and tool licensing terms.
 */

import { Request, Response } from 'express';
import crypto from 'crypto';
import { v4 as uuidv4 } from 'uuid';
import jwt from 'jsonwebtoken';
import winston from 'winston';

interface User {
  id: string;
  username: string;
  email: string;
  roles: string[];
  permissions: string[];
  ssoProvider?: string;
  ssoId?: string;
  isActive: boolean;
  lastLogin?: Date;
  createdAt: Date;
  updatedAt: Date;
  metadata: {
    department?: string;
    manager?: string;
    location?: string;
  };
}

interface Role {
  id: string;
  name: string;
  description: string;
  permissions: string[];
  isSystem: boolean;
  createdAt: Date;
  updatedAt: Date;
}

interface Permission {
  id: string;
  name: string;
  description: string;
  resource: string;
  action: string;
  conditions?: any;
}

interface Session {
  id: string;
  userId: string;
  token: string;
  refreshToken: string;
  expiresAt: Date;
  refreshExpiresAt: Date;
  ipAddress: string;
  userAgent: string;
  isActive: boolean;
  createdAt: Date;
}

interface SSOAuthConfig {
  provider: string;
  clientId: string;
  clientSecret: string;
  authorizationUrl: string;
  tokenUrl: string;
  userInfoUrl: string;
  scopes: string[];
  redirectUri: string;
}

interface EncryptionKey {
  id: string;
  keyId: string;
  keyVersion: number;
  key: string;
  algorithm: string;
  isActive: boolean;
  createdAt: Date;
  expiresAt?: Date;
  rotationScheduled: boolean;
}

interface ComplianceRecord {
  id: string;
  type: 'GDPR' | 'HIPAA' | 'SOC2' | 'ISO27001';
  category: 'data_processing' | 'data_retention' | 'data_deletion' | 'access_control' | 'audit_log';
  action: string;
  userId: string;
  resourceId: string;
  timestamp: Date;
  details: any;
  retentionUntil?: Date;
}

class EnterpriseSecurityService {
  private users: Map<string, User> = new Map();
  private roles: Map<string, Role> = new Map();
  private permissions: Map<string, Permission> = new Map();
  private sessions: Map<string, Session> = new Map();
  private encryptionKeys: Map<string, EncryptionKey> = new Map();
  private complianceRecords: Map<string, ComplianceRecord[]> = new Map();
  private ssoConfigs: Map<string, SSOAuthConfig> = new Map();
  private logger: winston.Logger;

  constructor(logger: winston.Logger) {
    this.logger = logger;
    this.initializeSecurity();
  }

  private initializeSecurity(): void {
    this.initializePermissions();
    this.initializeRoles();
    this.initializeDefaultUsers();
    this.initializeEncryptionKeys();
    this.initializeSSOConfigs();
  }

  private initializePermissions(): void {
    const permissions: Omit<Permission, 'id'>[] = [
      { name: 'data_pipeline_create', description: 'Create data pipelines', resource: 'data_pipeline', action: 'create' },
      { name: 'data_pipeline_read', description: 'Read data pipelines', resource: 'data_pipeline', action: 'read' },
      { name: 'data_pipeline_update', description: 'Update data pipelines', resource: 'data_pipeline', action: 'update' },
      { name: 'data_pipeline_delete', description: 'Delete data pipelines', resource: 'data_pipeline', action: 'delete' },
      
      { name: 'cloud_storage_read', description: 'Read cloud storage', resource: 'cloud_storage', action: 'read' },
      { name: 'cloud_storage_write', description: 'Write cloud storage', resource: 'cloud_storage', action: 'write' },
      { name: 'cloud_storage_delete', description: 'Delete cloud storage', resource: 'cloud_storage', action: 'delete' },
      
      { name: 'serverless_deploy', description: 'Deploy serverless functions', resource: 'serverless', action: 'deploy' },
      { name: 'serverless_invoke', description: 'Invoke serverless functions', resource: 'serverless', action: 'invoke' },
      
      { name: 'analytics_read', description: 'Read analytics data', resource: 'analytics', action: 'read' },
      { name: 'analytics_write', description: 'Write analytics data', resource: 'analytics', action: 'write' },
      { name: 'analytics_export', description: 'Export analytics data', resource: 'analytics', action: 'export' },
      
      { name: 'admin_user_manage', description: 'Manage users', resource: 'admin', action: 'user_manage' },
      { name: 'admin_role_manage', description: 'Manage roles', resource: 'admin', action: 'role_manage' },
      { name: 'admin_system_config', description: 'Configure system', resource: 'admin', action: 'system_config' },
      { name: 'admin_audit_logs', description: 'View audit logs', resource: 'admin', action: 'audit_logs' },
      
      { name: 'compliance_gdpr_read', description: 'Read GDPR compliance data', resource: 'compliance', action: 'gdpr_read' },
      { name: 'compliance_hipaa_read', description: 'Read HIPAA compliance data', resource: 'compliance', action: 'hipaa_read' },
      { name: 'compliance_export', description: 'Export compliance data', resource: 'compliance', action: 'export' },
      
      { name: 'security_encryption_manage', description: 'Manage encryption keys', resource: 'security', action: 'encryption_manage' },
      { name: 'security_sso_manage', description: 'Manage SSO configuration', resource: 'security', action: 'sso_manage' },
      { name: 'security_audit_read', description: 'Read security audit logs', resource: 'security', action: 'audit_read' }
    ];

    permissions.forEach(perm => {
      const permission: Permission = {
        id: uuidv4(),
        ...perm
      };
      this.permissions.set(permission.name, permission);
    });
  }

  private initializeRoles(): void {
    const roles: Omit<Role, 'id'>[] = [
      {
        name: 'admin',
        description: 'System administrator with full access',
        permissions: Array.from(this.permissions.keys()),
        isSystem: true
      },
      {
        name: 'data_engineer',
        description: 'Data engineer with pipeline and storage access',
        permissions: [
          'data_pipeline_create', 'data_pipeline_read', 'data_pipeline_update', 'data_pipeline_delete',
          'cloud_storage_read', 'cloud_storage_write', 'cloud_storage_delete',
          'serverless_deploy', 'serverless_invoke',
          'analytics_read', 'analytics_write'
        ],
        isSystem: false
      },
      {
        name: 'analyst',
        description: 'Business analyst with read access to data and analytics',
        permissions: [
          'data_pipeline_read',
          'cloud_storage_read',
          'analytics_read', 'analytics_export'
        ],
        isSystem: false
      },
      {
        name: 'developer',
        description: 'Developer with deployment and analytics access',
        permissions: [
          'data_pipeline_read', 'data_pipeline_update',
          'cloud_storage_read', 'cloud_storage_write',
          'serverless_deploy', 'serverless_invoke',
          'analytics_read'
        ],
        isSystem: false
      },
      {
        name: 'compliance_officer',
        description: 'Compliance officer with audit and compliance access',
        permissions: [
          'data_pipeline_read',
          'cloud_storage_read',
          'analytics_read',
          'compliance_gdpr_read', 'compliance_hipaa_read', 'compliance_export',
          'admin_audit_logs', 'security_audit_read'
        ],
        isSystem: false
      },
      {
        name: 'viewer',
        description: 'Read-only access to basic data and analytics',
        permissions: [
          'data_pipeline_read',
          'cloud_storage_read',
          'analytics_read'
        ],
        isSystem: false
      }
    ];

    roles.forEach(role => {
      const roleObj: Role = {
        id: uuidv4(),
        ...role,
        createdAt: new Date(),
        updatedAt: new Date()
      };
      this.roles.set(roleObj.name, roleObj);
    });
  }

  private initializeDefaultUsers(): void {
    const defaultUsers: Omit<User, 'id'>[] = [
      {
        username: 'admin',
        email: 'admin@kernelize.com',
        roles: ['admin'],
        permissions: Array.from(this.permissions.keys()),
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: {
          department: 'IT',
          location: 'Headquarters'
        }
      },
      {
        username: 'data_engineer_1',
        email: 'data.engineer@kernelize.com',
        roles: ['data_engineer'],
        permissions: [],
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: {
          department: 'Engineering',
          location: 'San Francisco'
        }
      },
      {
        username: 'analyst_1',
        email: 'analyst@kernelize.com',
        roles: ['analyst'],
        permissions: [],
        isActive: true,
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: {
          department: 'Business',
          location: 'New York'
        }
      }
    ];

    defaultUsers.forEach(user => {
      const userObj: User = {
        id: uuidv4(),
        ...user
      };
      this.users.set(userObj.id, userObj);
    });
  }

  private initializeEncryptionKeys(): void {
    const keyId = `kernelize-key-${Date.now()}`;
    const encryptionKey: EncryptionKey = {
      id: uuidv4(),
      keyId,
      keyVersion: 1,
      key: crypto.randomBytes(32).toString('hex'),
      algorithm: 'AES-256-GCM',
      isActive: true,
      createdAt: new Date(),
      rotationScheduled: false
    };
    this.encryptionKeys.set(encryptionKey.id, encryptionKey);
  }

  private initializeSSOConfigs(): void {
    const configs: SSOAuthConfig[] = [
      {
        provider: 'google',
        clientId: process.env.GOOGLE_CLIENT_ID || 'google-client-id',
        clientSecret: process.env.GOOGLE_CLIENT_SECRET || 'google-client-secret',
        authorizationUrl: 'https://accounts.google.com/o/oauth2/v2/auth',
        tokenUrl: 'https://oauth2.googleapis.com/token',
        userInfoUrl: 'https://www.googleapis.com/oauth2/v2/userinfo',
        scopes: ['openid', 'email', 'profile'],
        redirectUri: process.env.GOOGLE_REDIRECT_URI || 'http://localhost:3000/auth/google/callback'
      },
      {
        provider: 'microsoft',
        clientId: process.env.MICROSOFT_CLIENT_ID || 'microsoft-client-id',
        clientSecret: process.env.MICROSOFT_CLIENT_SECRET || 'microsoft-client-secret',
        authorizationUrl: 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
        tokenUrl: 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
        userInfoUrl: 'https://graph.microsoft.com/v1.0/me',
        scopes: ['openid', 'email', 'profile', 'User.Read'],
        redirectUri: process.env.MICROSOFT_REDIRECT_URI || 'http://localhost:3000/auth/microsoft/callback'
      }
    ];

    configs.forEach(config => {
      this.ssoConfigs.set(config.provider, config);
    });
  }

  // Authentication Methods
  async authenticate(username: string, password: string, req: Request): Promise<{ user: User; tokens: any } | null> {
    try {
      const user = Array.from(this.users.values()).find(u => u.username === username && u.isActive);
      
      if (!user) {
        this.logger.warn('Authentication failed: user not found', { username });
        return null;
      }

      // In production, verify password hash
      if (password !== 'password') { // Simplified for demo
        this.logger.warn('Authentication failed: invalid password', { username });
        return null;
      }

      // Generate tokens
      const tokens = this.generateTokens(user);
      
      // Create session
      const session: Session = {
        id: uuidv4(),
        userId: user.id,
        token: tokens.accessToken,
        refreshToken: tokens.refreshToken,
        expiresAt: new Date(Date.now() + 15 * 60 * 1000), // 15 minutes
        refreshExpiresAt: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days
        ipAddress: req.ip || req.connection.remoteAddress || 'unknown',
        userAgent: req.get('User-Agent') || 'unknown',
        isActive: true,
        createdAt: new Date()
      };
      
      this.sessions.set(session.id, session);
      
      // Update user last login
      user.lastLogin = new Date();
      user.updatedAt = new Date();
      
      this.logger.info('User authenticated successfully', { 
        userId: user.id, 
        username: user.username,
        ip: session.ipAddress 
      });
      
      return { user, tokens };
    } catch (error) {
      this.logger.error('Authentication error', { error, username });
      return null;
    }
  }

  public generateTokens(user: User): any {
    const payload = {
      sub: user.id,
      username: user.username,
      email: user.email,
      roles: user.roles,
      permissions: user.permissions
    };

    const accessToken = jwt.sign(payload, process.env.JWT_SECRET || 'default-secret', {
      expiresIn: '15m'
    });

    const refreshToken = jwt.sign(payload, process.env.JWT_REFRESH_SECRET || 'default-refresh-secret', {
      expiresIn: '7d'
    });

    return { accessToken, refreshToken };
  }

  async validateToken(token: string): Promise<User | null> {
    try {
      const decoded = jwt.verify(token, process.env.JWT_SECRET || 'default-secret') as any;
      const user = this.users.get(decoded.sub);
      
      if (!user || !user.isActive) {
        return null;
      }
      
      return user;
    } catch (error) {
      return null;
    }
  }

  // SSO Methods
  async initiateSSOAuth(provider: string, state: string): Promise<string | null> {
    try {
      const config = this.ssoConfigs.get(provider);
      if (!config) {
        return null;
      }

      const authUrl = `${config.authorizationUrl}?` +
        `client_id=${config.clientId}&` +
        `redirect_uri=${encodeURIComponent(config.redirectUri)}&` +
        `response_type=code&` +
        `scope=${encodeURIComponent(config.scopes.join(' '))}&` +
        `state=${state}`;

      return authUrl;
    } catch (error) {
      this.logger.error('SSO initiation error', { error, provider });
      return null;
    }
  }

  async handleSSOCallback(provider: string, code: string, req: Request): Promise<{ user: User; tokens: any } | null> {
    try {
      const config = this.ssoConfigs.get(provider);
      if (!config) {
        return null;
      }

      // In production, exchange code for tokens with provider
      // For demo, simulate user creation/fetch
      const ssoUser = {
        id: `sso_${provider}_${Date.now()}`,
        username: `user_${provider}`,
        email: `user@${provider}.com`,
        provider,
        ssoId: `provider_id_${Date.now()}`
      };

      let user = Array.from(this.users.values()).find(u => u.ssoId === ssoUser.ssoId);
      
      if (!user) {
        user = {
          id: uuidv4(),
          username: ssoUser.username,
          email: ssoUser.email,
          roles: ['viewer'],
          permissions: [],
          ssoProvider: provider,
          ssoId: ssoUser.ssoId,
          isActive: true,
          createdAt: new Date(),
          updatedAt: new Date(),
          metadata: {
            location: 'SSO'
          }
        };
        this.users.set(user.id, user);
      }

      const tokens = this.generateTokens(user);
      
      this.logger.info('SSO authentication successful', { 
        userId: user.id, 
        provider,
        ip: req.ip 
      });
      
      return { user, tokens };
    } catch (error) {
      this.logger.error('SSO callback error', { error, provider });
      return null;
    }
  }

  // Authorization Methods
  async checkPermission(userId: string, permission: string): Promise<boolean> {
    try {
      const user = this.users.get(userId);
      if (!user) {
        return false;
      }

      // Check direct permissions
      if (user.permissions.includes(permission)) {
        return true;
      }

      // Check role-based permissions
      for (const roleName of user.roles) {
        const role = this.roles.get(roleName);
        if (role && role.permissions.includes(permission)) {
          return true;
        }
      }

      return false;
    } catch (error) {
      this.logger.error('Permission check error', { error, userId, permission });
      return false;
    }
  }

  async getUserRoles(userId: string): Promise<string[]> {
    const user = this.users.get(userId);
    return user ? user.roles : [];
  }

  async assignRole(userId: string, roleName: string): Promise<boolean> {
    try {
      const user = this.users.get(userId);
      const role = this.roles.get(roleName);
      
      if (!user || !role) {
        return false;
      }

      if (!user.roles.includes(roleName)) {
        user.roles.push(roleName);
        user.updatedAt = new Date();
      }

      this.logger.info('Role assigned', { userId, roleName });
      return true;
    } catch (error) {
      this.logger.error('Role assignment error', { error, userId, roleName });
      return false;
    }
  }

  async removeRole(userId: string, roleName: string): Promise<boolean> {
    try {
      const user = this.users.get(userId);
      if (!user) {
        return false;
      }

      user.roles = user.roles.filter(r => r !== roleName);
      user.updatedAt = new Date();

      this.logger.info('Role removed', { userId, roleName });
      return true;
    } catch (error) {
      this.logger.error('Role removal error', { error, userId, roleName });
      return false;
    }
  }

  // Encryption Methods
  async encryptData(data: string, keyId?: string): Promise<{ encryptedData: string; iv: string; tag: string }> {
    try {
      const key = this.getActiveEncryptionKey(keyId);
      if (!key) {
        throw new Error('No active encryption key found');
      }

      const keyBuffer = Buffer.from(key.key, 'hex');
      const iv = crypto.randomBytes(16);
      const cipher = crypto.createCipher(key.algorithm, keyBuffer);
      cipher.setAAD(Buffer.from('kernelize-aad'));

      let encrypted = cipher.update(data, 'utf8', 'hex');
      encrypted += cipher.final('hex');
      
      const tag = cipher.getAuthTag();

      return {
        encryptedData: encrypted,
        iv: iv.toString('hex'),
        tag: tag.toString('hex')
      };
    } catch (error) {
      this.logger.error('Encryption error', { error });
      throw error;
    }
  }

  async decryptData(encryptedData: string, iv: string, tag: string, keyId?: string): Promise<string> {
    try {
      const key = this.getActiveEncryptionKey(keyId);
      if (!key) {
        throw new Error('No active encryption key found');
      }

      const keyBuffer = Buffer.from(key.key, 'hex');
      const ivBuffer = Buffer.from(iv, 'hex');
      const tagBuffer = Buffer.from(tag, 'hex');

      const decipher = crypto.createDecipher(key.algorithm, keyBuffer);
      decipher.setAAD(Buffer.from('kernelize-aad'));
      decipher.setAuthTag(tagBuffer);

      let decrypted = decipher.update(encryptedData, 'hex', 'utf8');
      decrypted += decipher.final('utf8');

      return decrypted;
    } catch (error) {
      this.logger.error('Decryption error', { error });
      throw error;
    }
  }

  private getActiveEncryptionKey(keyId?: string): EncryptionKey | null {
    if (keyId) {
      return this.encryptionKeys.get(keyId) || null;
    }
    
    // Return the most recent active key
    const activeKeys = Array.from(this.encryptionKeys.values())
      .filter(key => key.isActive)
      .sort((a, b) => b.keyVersion - a.keyVersion);
    
    return activeKeys.length > 0 ? activeKeys[0] : null;
  }

  async rotateEncryptionKey(): Promise<string> {
    try {
      const currentKey = this.getActiveEncryptionKey();
      if (currentKey) {
        currentKey.isActive = false;
      }

      const newKeyId = `kernelize-key-${Date.now()}`;
      const newKey: EncryptionKey = {
        id: uuidv4(),
        keyId: newKeyId,
        keyVersion: (currentKey?.keyVersion || 0) + 1,
        key: crypto.randomBytes(32).toString('hex'),
        algorithm: 'AES-256-GCM',
        isActive: true,
        createdAt: new Date(),
        rotationScheduled: false
      };

      this.encryptionKeys.set(newKey.id, newKey);
      
      this.logger.info('Encryption key rotated', { 
        keyId: newKeyId, 
        version: newKey.keyVersion 
      });
      
      return newKeyId;
    } catch (error) {
      this.logger.error('Key rotation error', { error });
      throw error;
    }
  }

  // Compliance Methods
  async logComplianceEvent(type: 'GDPR' | 'HIPAA' | 'SOC2' | 'ISO27001', category: string, action: string, userId: string, resourceId: string, details: any): Promise<void> {
    try {
      const record: ComplianceRecord = {
        id: uuidv4(),
        type,
        category: category as any,
        action,
        userId,
        resourceId,
        timestamp: new Date(),
        details,
        retentionUntil: this.calculateRetentionDate(type, category)
      };

      if (!this.complianceRecords.has(type)) {
        this.complianceRecords.set(type, []);
      }
      this.complianceRecords.get(type)!.push(record);

      this.logger.info('Compliance event logged', { type, category, userId, resourceId });
    } catch (error) {
      this.logger.error('Compliance logging error', { error, type, category });
    }
  }

  private calculateRetentionDate(type: string, category: string): Date {
    const retentionDays = {
      'GDPR:data_processing': 2555, // 7 years
      'GDPR:data_retention': 2555,
      'GDPR:data_deletion': 2555,
      'GDPR:access_control': 2555,
      'GDPR:audit_log': 2555,
      'HIPAA:data_processing': 3653, // 10 years
      'HIPAA:data_retention': 3653,
      'HIPAA:data_deletion': 3653,
      'HIPAA:access_control': 2555,
      'HIPAA:audit_log': 2555
    };

    const key = `${type}:${category}`;
    const days = retentionDays[key] || 2555; // Default 7 years
    
    return new Date(Date.now() + days * 24 * 60 * 60 * 1000);
  }

  async getComplianceRecords(type: 'GDPR' | 'HIPAA' | 'SOC2' | 'ISO27001', userId?: string, startDate?: Date, endDate?: Date): Promise<ComplianceRecord[]> {
    try {
      const records = this.complianceRecords.get(type) || [];
      
      return records.filter(record => {
        if (userId && record.userId !== userId) return false;
        if (startDate && record.timestamp < startDate) return false;
        if (endDate && record.timestamp > endDate) return false;
        return true;
      });
    } catch (error) {
      this.logger.error('Compliance records retrieval error', { error, type });
      return [];
    }
  }

  async exportComplianceData(type: 'GDPR' | 'HIPAA' | 'SOC2' | 'ISO27001', format: 'json' | 'csv' | 'xml' = 'json'): Promise<any> {
    try {
      const records = this.complianceRecords.get(type) || [];
 the      
      // Log export event
      await this.logComplianceEvent(type, 'audit_log', 'data_export', 'system', 'compliance_export', {
        exportFormat: format,
        recordCount: records.length,
        exportedBy: 'system'
      });

      return {
        type,
        exportDate: new Date().toISOString(),
        recordCount: records.length,
        format,
        data: records
      };
    } catch (error) {
      this.logger.error('Compliance export error', { error, type });
      throw error;
    }
  }

  // User Management
  async createUser(userData: Partial<User>): Promise<User> {
    try {
      const user: User = {
        id: uuidv4(),
        username: userData.username || '',
        email: userData.email || '',
        roles: userData.roles || ['viewer'],
        permissions: userData.permissions || [],
        isActive: userData.isActive !== undefined ? userData.isActive : true,
        createdAt: new Date(),
        updatedAt: new Date(),
        metadata: userData.metadata || {}
      };

      this.users.set(user.id, user);
      
      this.logger.info('User created', { userId: user.id, username: user.username });
      return user;
    } catch (error) {
      this.logger.error('User creation error', { error });
      throw error;
    }
  }

  async getUser(userId: string): Promise<User | null> {
    return this.users.get(userId) || null;
  }

  async getAllUsers(): Promise<User[]> {
    return Array.from(this.users.values());
  }

  async updateUser(userId: string, updates: Partial<User>): Promise<User | null> {
    try {
      const user = this.users.get(userId);
      if (!user) {
        return null;
      }

      Object.assign(user, updates, { updatedAt: new Date() });
      
      this.logger.info('User updated', { userId, updates: Object.keys(updates) });
      return user;
    } catch (error) {
      this.logger.error('User update error', { error, userId });
      throw error;
    }
  }

  async deleteUser(userId: string): Promise<boolean> {
    try {
      const user = this.users.get(userId);
      if (!user) {
        return false;
      }

      user.isActive = false;
      user.updatedAt = new Date();
      
      // Invalidate all sessions for this user
      Array.from(this.sessions.values())
        .filter(session => session.userId === userId)
        .forEach(session => {
          session.isActive = false;
        });

      this.logger.info('User deactivated', { userId });
      return true;
    } catch (error) {
      this.logger.error('User deletion error', { error, userId });
      throw error;
    }
  }

  // Session Management
  async invalidateSession(sessionId: string): Promise<boolean> {
    try {
      const session = this.sessions.get(sessionId);
      if (!session) {
        return false;
      }

      session.isActive = false;
      this.logger.info('Session invalidated', { sessionId });
      return true;
    } catch (error) {
      this.logger.error('Session invalidation error', { error, sessionId });
      return false;
    }
  }

  async cleanupExpiredSessions(): Promise<number> {
    try {
      const now = new Date();
      let cleanedCount = 0;

      Array.from(this.sessions.entries()).forEach(([id, session]) => {
        if (session.expiresAt < now || !session.isActive) {
          this.sessions.delete(id);
          cleanedCount++;
        }
      });

      this.logger.info('Expired sessions cleaned up', { cleanedCount });
      return cleanedCount;
    } catch (error) {
      this.logger.error('Session cleanup error', { error });
      return 0;
    }
  }

  // Audit and Logging
  async logSecurityEvent(event: string, userId: string, details: any): Promise<void> {
    try {
      await this.logComplianceEvent('SOC2', 'audit_log', event, userId, 'security_event', details);
      
      this.logger.warn('Security event', { 
        event, 
        userId, 
        details,
        timestamp: new Date().toISOString() 
      });
    } catch (error) {
      this.logger.error('Security event logging error', { error, event });
    }
  }

  // Security Status
  async getSecurityStatus(): Promise<any> {
    const activeUsers = Array.from(this.users.values()).filter(u => u.isActive).length;
    const activeSessions = Array.from(this.sessions.values()).filter(s => s.isActive).length;
    const activeKeys = Array.from(this.encryptionKeys.values()).filter(k => k.isActive).length;

    return {
      timestamp: new Date().toISOString(),
      users: {
        total: this.users.size,
        active: activeUsers,
        inactive: this.users.size - activeUsers
      },
      sessions: {
        total: this.sessions.size,
        active: activeSessions
      },
      encryption: {
        activeKeys,
        lastRotation: Array.from(this.encryptionKeys.values())
          .filter(k => k.isActive)
          .sort((a, b) => b.keyVersion - a.keyVersion)[0]?.createdAt
      },
      compliance: {
        gdpr: this.complianceRecords.get('GDPR')?.length || 0,
        hipaa: this.complianceRecords.get('HIPAA')?.length || 0,
        soc2: this.complianceRecords.get('SOC2')?.length || 0,
        iso27001: this.complianceRecords.get('ISO27001')?.length || 0
      },
      sso: {
        configured: this.ssoConfigs.size,
        active: Array.from(this.ssoConfigs.values()).filter(c => c.clientId !== 'default-client-id').length
      }
    };
  }
}

export default EnterpriseSecurityService;