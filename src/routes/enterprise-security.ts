/**
 * KERNELIZE Platform - Core Backend
 * Licensed under the Business Source License 1.1 (BSL 1.1)
 * 
 * Copyright (c) 2026 KERNELIZE Platform. All rights reserved.
 * 
 * See LICENSE-CORE in the project root for license information.
 * See LICENSE-SDK for SDK and tool licensing terms.
 */

import { Router, Request, Response } from 'express';
import EnterpriseSecurityService from '../services/security/enterprise-security-service';
import jwt from 'jsonwebtoken';

const router = Router();
const securityService = new EnterpriseSecurityService((req: any, res: any, next: any) => next()); // Simplified logger

// Authentication middleware
const authenticateToken = async (req: Request, res: Response, next: any) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ error: 'Access token required' });
  }

  try {
    const user = await securityService.validateToken(token);
    if (!user) {
      return res.status(403).json({ error: 'Invalid or expired token' });
    }
    (req as any).user = user;
    next();
  } catch (error) {
    return res.status(403).json({ error: 'Invalid token' });
  }
};

// Permission middleware
const requirePermission = (permission: string) => {
  return async (req: Request, res: Response, next: any) => {
    const user = (req as any).user;
    if (!user) {
      return res.status(401).json({ error: 'User not authenticated' });
    }

    const hasPermission = await securityService.checkPermission(user.id, permission);
    if (!hasPermission) {
      return res.status(403).json({ error: `Permission denied: ${permission}` });
    }

    next();
  };
};

// Authentication Routes
router.post('/auth/login', async (req, res) => {
  try {
    const { username, password } = req.body;
    
    if (!username || !password) {
      return res.status(400).json({ error: 'Username and password required' });
    }

    const result = await securityService.authenticate(username, password, req);
    
    if (!result) {
      return res.status(401).json({ error: 'Invalid credentials' });
    }

    res.json({
      user: {
        id: result.user.id,
        username: result.user.username,
        email: result.user.email,
        roles: result.user.roles,
        permissions: result.user.permissions
      },
      tokens: result.tokens
    });
  } catch (error) {
    console.error('Login error:', error);
    res.status(500).json({ error: 'Authentication failed' });
  }
});

router.post('/auth/refresh', async (req, res) => {
  try {
    const { refreshToken } = req.body;
    
    if (!refreshToken) {
      return res.status(400).json({ error: 'Refresh token required' });
    }

    const decoded = jwt.verify(refreshToken, process.env.JWT_REFRESH_SECRET || 'default-refresh-secret') as any;
    const user = await securityService.validateToken(decoded.sub);
    
    if (!user) {
      return res.status(403).json({ error: 'Invalid refresh token' });
    }

    const tokens = (securityService as any).generateTokens(user);
    res.json(tokens);
  } catch (error) {
    res.status(403).json({ error: 'Invalid refresh token' });
  }
});

router.post('/auth/logout', authenticateToken, async (req, res) => {
  try {
    const authHeader = req.headers['authorization'];
    const token = authHeader?.split(' ')[1];
    
    if (token) {
      // In production, add token to blacklist
      await securityService.logSecurityEvent('logout', (req as any).user.id, {
        timestamp: new Date(),
        ip: req.ip,
        userAgent: req.get('User-Agent')
      });
    }

    res.json({ message: 'Logged out successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Logout failed' });
  }
});

// SSO Routes
router.get('/auth/sso/:provider', async (req, res) => {
  try {
    const { provider } = req.params;
    const { state } = req.query;
    
    const authUrl = await securityService.initiateSSOAuth(provider, state as string);
    
    if (!authUrl) {
      return res.status(400).json({ error: 'Invalid SSO provider' });
    }

    res.json({ authUrl });
  } catch (error) {
    res.status(500).json({ error: 'SSO initiation failed' });
  }
});

router.post('/auth/sso/:provider/callback', async (req, res) => {
  try {
    const { provider } = req.params;
    const { code, state } = req.body;
    
    const result = await securityService.handleSSOCallback(provider, code, req);
    
    if (!result) {
      return res.status(401).json({ error: 'SSO authentication failed' });
    }

    res.json({
      user: {
        id: result.user.id,
        username: result.user.username,
        email: result.user.email,
        roles: result.user.roles,
        permissions: result.user.permissions
      },
      tokens: result.tokens
    });
  } catch (error) {
    res.status(500).json({ error: 'SSO callback failed' });
  }
});

// User Management Routes
router.get('/users', authenticateToken, requirePermission('admin_user_manage'), async (req, res) => {
  try {
    const users = await securityService.getAllUsers();
    res.json(users.map(user => ({
      id: user.id,
      username: user.username,
      email: user.email,
      roles: user.roles,
      isActive: user.isActive,
      lastLogin: user.lastLogin,
      createdAt: user.createdAt,
      metadata: user.metadata
    })));
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve users' });
  }
});

router.post('/users', authenticateToken, requirePermission('admin_user_manage'), async (req, res) => {
  try {
    const user = await securityService.createUser(req.body);
    res.status(201).json({
      id: user.id,
      username: user.username,
      email: user.email,
      roles: user.roles,
      isActive: user.isActive,
      metadata: user.metadata
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to create user' });
  }
});

router.get('/users/:userId', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    const user = await securityService.getUser(userId);
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    // Users can only view their own profile unless they have admin permissions
    const currentUser = (req as any).user;
    if (user.id !== currentUser.id && !await securityService.checkPermission(currentUser.id, 'admin_user_manage')) {
      return res.status(403).json({ error: 'Access denied' });
    }

    res.json({
      id: user.id,
      username: user.username,
      email: user.email,
      roles: user.roles,
      isActive: user.isActive,
      lastLogin: user.lastLogin,
      createdAt: user.createdAt,
      metadata: user.metadata
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve user' });
  }
});

router.put('/users/:userId', authenticateToken, async (req, res) => {
  try {
    const { userId } = req.params;
    const currentUser = (req as any).user;
    
    // Users can only update their own profile unless they have admin permissions
    if (userId !== currentUser.id && !await securityService.checkPermission(currentUser.id, 'admin_user_manage')) {
      return res.status(403).json({ error: 'Access denied' });
    }

    const user = await securityService.updateUser(userId, req.body);
    
    if (!user) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json({
      id: user.id,
      username: user.username,
      email: user.email,
      roles: user.roles,
      isActive: user.isActive,
      lastLogin: user.lastLogin,
      createdAt: user.createdAt,
      metadata: user.metadata
    });
  } catch (error) {
    res.status(500).json({ error: 'Failed to update user' });
  }
});

router.delete('/users/:userId', authenticateToken, requirePermission('admin_user_manage'), async (req, res) => {
  try {
    const { userId } = req.params;
    const success = await securityService.deleteUser(userId);
    
    if (!success) {
      return res.status(404).json({ error: 'User not found' });
    }

    res.json({ message: 'User deactivated successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to delete user' });
  }
});

// Role Management Routes
router.get('/roles', authenticateToken, requirePermission('admin_role_manage'), async (req, res) => {
  try {
    // This would be implemented to return all roles
    res.json([]);
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve roles' });
  }
});

router.post('/users/:userId/roles', authenticateToken, requirePermission('admin_role_manage'), async (req, res) => {
  try {
    const { userId } = req.params;
    const { roleName } = req.body;
    
    const success = await securityService.assignRole(userId, roleName);
    
    if (!success) {
      return res.status(400).json({ error: 'Failed to assign role' });
    }

    res.json({ message: 'Role assigned successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to assign role' });
  }
});

router.delete('/users/:userId/roles/:roleName', authenticateToken, requirePermission('admin_role_manage'), async (req, res) => {
  try {
    const { userId, roleName } = req.params;
    
    const success = await securityService.removeRole(userId, roleName);
    
    if (!success) {
      return res.status(400).json({ error: 'Failed to remove role' });
    }

    res.json({ message: 'Role removed successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to remove role' });
  }
});

// Encryption Routes
router.post('/encryption/encrypt', authenticateToken, requirePermission('security_encryption_manage'), async (req, res) => {
  try {
    const { data, keyId } = req.body;
    
    if (!data) {
      return res.status(400).json({ error: 'Data to encrypt is required' });
    }

    const result = await securityService.encryptData(data, keyId);
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: 'Encryption failed' });
  }
});

router.post('/encryption/decrypt', authenticateToken, requirePermission('security_encryption_manage'), async (req, res) => {
  try {
    const { encryptedData, iv, tag, keyId } = req.body;
    
    if (!encryptedData || !iv || !tag) {
      return res.status(400).json({ error: 'Encrypted data, IV, and tag are required' });
    }

    const result = await securityService.decryptData(encryptedData, iv, tag, keyId);
    res.json({ decryptedData: result });
  } catch (error) {
    res.status(500).json({ error: 'Decryption failed' });
  }
});

router.post('/encryption/rotate-key', authenticateToken, requirePermission('security_encryption_manage'), async (req, res) => {
  try {
    const newKeyId = await securityService.rotateEncryptionKey();
    res.json({ newKeyId, message: 'Encryption key rotated successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Key rotation failed' });
  }
});

// Compliance Routes
router.post('/compliance/log', authenticateToken, async (req, res) => {
  try {
    const { type, category, action, userId, resourceId, details } = req.body;
    
    if (!type || !category || !action || !userId || !resourceId) {
      return res.status(400).json({ error: 'Missing required compliance event fields' });
    }

    await securityService.logComplianceEvent(type, category, action, userId, resourceId, details);
    res.json({ message: 'Compliance event logged successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to log compliance event' });
  }
});

router.get('/compliance/:type', authenticateToken, requirePermission('compliance_gdpr_read'), async (req, res) => {
  try {
    const { type } = req.params;
    const { userId, startDate, endDate } = req.query;
    
    const records = await securityService.getComplianceRecords(
      type as any,
      userId as string,
      startDate ? new Date(startDate as string) : undefined,
      endDate ? new Date(endDate as string) : undefined
    );

    res.json(records);
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve compliance records' });
  }
});

router.get('/compliance/:type/export', authenticateToken, requirePermission('compliance_export'), async (req, res) => {
  try {
    const { type } = req.params;
    const { format } = req.query;
    
    const exportData = await securityService.exportComplianceData(
      type as any,
      (format as any) || 'json'
    );

    res.json(exportData);
  } catch (error) {
    res.status(500).json({ error: 'Failed to export compliance data' });
  }
});

// Security Status and Audit Routes
router.get('/security/status', authenticateToken, requirePermission('admin_system_config'), async (req, res) => {
  try {
    const status = await securityService.getSecurityStatus();
    res.json(status);
  } catch (error) {
    res.status(500).json({ error: 'Failed to retrieve security status' });
  }
});

router.post('/security/log-event', authenticateToken, async (req, res) => {
  try {
    const { event, details } = req.body;
    const user = (req as any).user;
    
    if (!event) {
      return res.status(400).json({ error: 'Event type is required' });
    }

    await securityService.logSecurityEvent(event, user.id, {
      ...details,
      ip: req.ip,
      userAgent: req.get('User-Agent'),
      timestamp: new Date()
    });

    res.json({ message: 'Security event logged successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to log security event' });
  }
});

router.post('/security/cleanup-sessions', authenticateToken, requirePermission('admin_system_config'), async (req, res) => {
  try {
    const cleanedCount = await securityService.cleanupExpiredSessions();
    res.json({ cleanedCount, message: 'Expired sessions cleaned up successfully' });
  } catch (error) {
    res.status(500).json({ error: 'Failed to cleanup sessions' });
  }
});

// Health Check
router.get('/health', async (req, res) => {
  res.json({
    status: 'healthy',
    service: 'enterprise-security',
    timestamp: new Date().toISOString(),
    version: '1.0.0'
  });
});

export default router;