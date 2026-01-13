/**
 * KERNELIZE Platform - Plugin Management System
 * Comprehensive plugin installation, activation, monitoring, and lifecycle management
 */

import { BasePlugin, PluginFactory, PluginLifecycleManager } from '../plugins/base-plugin.js';
import {
  PluginMetadata,
  PluginSource,
  InstallationResult,
  PluginHealth,
  PluginStatus,
  PluginUpdate,
  PluginRepository,
  SandboxConfig
} from '../plugins/plugin-types.js';
import * as fs from 'fs/promises';
import * as path from 'path';
import crypto from 'crypto';

interface PluginInstallation {
  pluginId: string;
  version: string;
  installedAt: Date;
  source: PluginSource;
  signature?: string;
  checksum?: string;
  status: PluginStatus;
  health: PluginHealth;
  config: Record<string, any>;
  permissions: any[];
}

interface PluginMetrics {
  pluginId: string;
  executionCount: number;
  averageExecutionTime: number;
  errorRate: number;
  memoryUsage: number;
  cpuUsage: number;
  lastExecution: Date;
  uptime: number;
}

export class PluginManager {
  private lifecycleManager: PluginLifecycleManager;
  private pluginRegistry: Map<string, PluginInstallation> = new Map();
  private repositories: Map<string, PluginRepository> = new Map();
  private metrics: Map<string, PluginMetrics> = new Map();
  private sandboxConfig: SandboxConfig;
  private pluginDirectory: string;

  constructor(pluginDirectory: string = './plugins', sandboxConfig: SandboxConfig = {
    enabled: false,
    networkIsolation: false,
    fileSystemAccess: 'read-only',
    processIsolation: false,
    timeLimits: { execution: 30000, idle: 600000 },
    resourceLimits: { memory: 256, cpu: 1, disk: 1024 }
  }) {
    this.pluginDirectory = pluginDirectory;
    this.sandboxConfig = sandboxConfig;
    this.lifecycleManager = new PluginLifecycleManager();
  }

  /**
   * Plugin Repository Management
   */
  async addRepository(repository: PluginRepository): Promise<void> {
    this.repositories.set(repository.id, repository);
    await this.syncRepository(repository);
  }

  async removeRepository(repositoryId: string): Promise<void> {
    this.repositories.delete(repositoryId);
  }

  async syncRepository(repository: PluginRepository): Promise<void> {
    try {
      // In real implementation, fetch plugin list from repository
      console.log(`Syncing repository: ${repository.name}`);
      
      // Mock plugin list
      const mockPlugins: PluginMetadata[] = [
        {
          id: 'snowflake-connector',
          name: 'Snowflake Connector',
          version: '1.0.0',
          description: 'Snowflake data warehouse integration',
          author: 'KERNELIZE Team',
          category: 'connector' as any,
          keywords: ['snowflake', 'data-warehouse'],
          license: 'MIT',
          createdAt: new Date(),
          updatedAt: new Date(),
          downloads: 0,
          rating: 0,
          compatibility: {
            minPlatformVersion: '1.0.0',
            supportedNodes: ['api', 'data-pipeline']
          }
        }
      ];

      repository.plugins = mockPlugins;
      repository.lastSync = new Date();
      
    } catch (error) {
      console.error(`Failed to sync repository ${repository.id}:`, error);
      throw error;
    }
  }

  async searchPlugins(query: string, category?: string): Promise<PluginMetadata[]> {
    const results: PluginMetadata[] = [];
    
    for (const repository of this.repositories.values()) {
      if (!repository.enabled) continue;
      
      for (const plugin of repository.plugins) {
        const matchesQuery = plugin.name.toLowerCase().includes(query.toLowerCase()) ||
                           plugin.description.toLowerCase().includes(query.toLowerCase()) ||
                           plugin.keywords.some(keyword => keyword.toLowerCase().includes(query.toLowerCase()));
        
        const matchesCategory = !category || plugin.category === category;
        
        if (matchesQuery && matchesCategory) {
          results.push(plugin);
        }
      }
    }
    
    return results.sort((a, b) => b.downloads - a.downloads);
  }

  /**
   * Plugin Installation
   */
  async installPlugin(source: PluginSource, config: Record<string, any> = {}): Promise<InstallationResult> {
    try {
      console.log(`Installing plugin from: ${source.location}`);
      
      // Download plugin package
      const pluginPackage = await this.downloadPlugin(source);
      
      // Verify signature if provided
      if (source.signature) {
        await this.verifySignature(pluginPackage, source.signature);
      }
      
      // Verify checksum
      if (source.checksum) {
        await this.verifyChecksum(pluginPackage, source.checksum);
      }
      
      // Extract and validate plugin
      const extractedPath = await this.extractPlugin(pluginPackage);
      const pluginMetadata = await this.loadPluginMetadata(extractedPath);
      
      // Check dependencies
      await this.checkDependencies(pluginMetadata);
      
      // Install plugin files
      const installationPath = path.join(this.pluginDirectory, pluginMetadata.id);
      await this.installPluginFiles(extractedPath, installationPath);
      
      // Create installation record
      const installation: PluginInstallation = {
        pluginId: pluginMetadata.id,
        version: pluginMetadata.version,
        installedAt: new Date(),
        source,
        signature: source.signature,
        checksum: source.checksum,
        status: PluginStatus.INSTALLED,
        health: this.createInitialHealth(),
        config,
        permissions: config.permissions || []
      };
      
      this.pluginRegistry.set(pluginMetadata.id, installation);
      
      // Load plugin class
      const pluginClass = await this.loadPluginClass(installationPath);
      PluginFactory.registerPlugin(pluginMetadata.id, pluginClass);
      
      console.log(`Plugin ${pluginMetadata.id} installed successfully`);
      
      return {
        success: true,
        pluginId: pluginMetadata.id,
        version: pluginMetadata.version,
        installedAt: new Date()
      };
      
    } catch (error) {
      console.error('Plugin installation failed:', error);
      return {
        success: false,
        pluginId: '',
        version: '',
        installedAt: new Date(),
        errors: [error.message]
      };
    }
  }

  async downloadPlugin(source: PluginSource): Promise<Buffer> {
    // Mock implementation - in real implementation would download from URL/file
    console.log(`Downloading plugin from: ${source.location}`);
    
    // For demonstration, return mock package
    return Buffer.from(JSON.stringify({
      metadata: { name: 'mock-plugin', version: '1.0.0' },
      code: 'module.exports = class MockPlugin {}'
    }));
  }

  async verifySignature(pluginPackage: Buffer, signature: string): Promise<void> {
    // Mock signature verification
    console.log('Verifying plugin signature...');
    
    // In real implementation, would verify digital signature
    if (!signature.startsWith('sig-')) {
      throw new Error('Invalid plugin signature');
    }
  }

  async verifyChecksum(pluginPackage: Buffer, expectedChecksum: string): Promise<void> {
    const actualChecksum = crypto.createHash('sha256').update(pluginPackage).digest('hex');
    
    if (actualChecksum !== expectedChecksum) {
      throw new Error(`Checksum mismatch. Expected: ${expectedChecksum}, Got: ${actualChecksum}`);
    }
  }

  async extractPlugin(pluginPackage: Buffer): Promise<string> {
    // Mock extraction
    const extractPath = path.join(this.pluginDirectory, 'temp', `extracted-${Date.now()}`);
    await fs.mkdir(extractPath, { recursive: true });
    
    // In real implementation, would extract tar.gz or zip file
    console.log(`Plugin extracted to: ${extractPath}`);
    
    return extractPath;
  }

  async loadPluginMetadata(pluginPath: string): Promise<PluginMetadata> {
    const metadataPath = path.join(pluginPath, 'plugin.json');
    
    try {
      const metadataContent = await fs.readFile(metadataPath, 'utf-8');
      return JSON.parse(metadataContent);
    } catch (error) {
      throw new Error(`Failed to load plugin metadata: ${error.message}`);
    }
  }

  async checkDependencies(pluginMetadata: PluginMetadata): Promise<void> {
    // Check if dependencies are installed
    for (const dependency of pluginMetadata.compatibility.supportedNodes) {
      if (!this.pluginRegistry.has(dependency)) {
        console.warn(`Dependency ${dependency} not found for plugin ${pluginMetadata.id}`);
      }
    }
  }

  async installPluginFiles(sourcePath: string, targetPath: string): Promise<void> {
    // Copy plugin files to permanent location
    await fs.mkdir(targetPath, { recursive: true });
    
    // In real implementation, would copy all plugin files
    console.log(`Plugin files installed to: ${targetPath}`);
  }

  async loadPluginClass(pluginPath: string): Promise<typeof BasePlugin> {
    // Mock plugin class loading
    // In real implementation, would dynamically import the plugin class
    return BasePlugin as any;
  }

  private createInitialHealth(): PluginHealth {
    return {
      status: 'healthy',
      uptime: 0,
      performance: {
        responseTime: 0,
        throughput: 0,
        errorRate: 0,
        memoryUsage: 0,
        cpuUsage: 0,
        activeConnections: 0
      },
      errors: {
        totalErrors: 0,
        errorRate: 0
      },
      lastCheck: new Date(),
      checks: []
    };
  }

  /**
   * Plugin Activation/Deactivation
   */
  async activatePlugin(pluginId: string, config?: Record<string, any>): Promise<void> {
    const installation = this.pluginRegistry.get(pluginId);
    if (!installation) {
      throw new Error(`Plugin not found: ${pluginId}`);
    }

    try {
      // Update configuration
      if (config) {
        installation.config = { ...installation.config, ...config };
      }

      // Create plugin instance
      const metadata = await this.getPluginMetadata(pluginId);
      const plugin = PluginFactory.createPlugin(pluginId, metadata, installation.config);
      
      // Initialize plugin
      await this.lifecycleManager.installPlugin(plugin);
      
      // Update status
      installation.status = PluginStatus.ACTIVE;
      installation.health = plugin.getHealth();
      
      console.log(`Plugin ${pluginId} activated successfully`);
      
    } catch (error) {
      installation.status = PluginStatus.ERROR;
      console.error(`Failed to activate plugin ${pluginId}:`, error);
      throw error;
    }
  }

  async deactivatePlugin(pluginId: string): Promise<void> {
    const installation = this.pluginRegistry.get(pluginId);
    if (!installation) {
      throw new Error(`Plugin not found: ${pluginId}`);
    }

    try {
      await this.lifecycleManager.deactivatePlugin(pluginId);
      
      installation.status = PluginStatus.INACTIVE;
      console.log(`Plugin ${pluginId} deactivated successfully`);
      
    } catch (error) {
      console.error(`Failed to deactivate plugin ${pluginId}:`, error);
      throw error;
    }
  }

  async uninstallPlugin(pluginId: string): Promise<void> {
    const installation = this.pluginRegistry.get(pluginId);
    if (!installation) {
      throw new Error(`Plugin not found: ${pluginId}`);
    }

    try {
      // Deactivate if active
      if (installation.status === PluginStatus.ACTIVE) {
        await this.deactivatePlugin(pluginId);
      }

      // Remove plugin files
      const pluginPath = path.join(this.pluginDirectory, pluginId);
      await fs.rm(pluginPath, { recursive: true, force: true });
      
      // Remove from registry
      this.pluginRegistry.delete(pluginId);
      this.metrics.delete(pluginId);
      
      console.log(`Plugin ${pluginId} uninstalled successfully`);
      
    } catch (error) {
      console.error(`Failed to uninstall plugin ${pluginId}:`, error);
      throw error;
    }
  }

  /**
   * Plugin Monitoring
   */
  async getPluginHealth(pluginId: string): Promise<PluginHealth | null> {
    const installation = this.pluginRegistry.get(pluginId);
    if (!installation) {
      return null;
    }

    try {
      const plugin = this.lifecycleManager.getPlugin(pluginId);
      if (plugin) {
        installation.health = plugin.getHealth();
        this.updateMetrics(pluginId, plugin);
      }
      
      return installation.health;
      
    } catch (error) {
      console.error(`Failed to get health for plugin ${pluginId}:`, error);
      return null;
    }
  }

  async performHealthCheck(): Promise<Map<string, PluginHealth>> {
    const healthMap = new Map<string, PluginHealth>();
    
    for (const [pluginId, installation] of this.pluginRegistry) {
      try {
        const health = await this.getPluginHealth(pluginId);
        if (health) {
          healthMap.set(pluginId, health);
        }
      } catch (error) {
        console.error(`Health check failed for plugin ${pluginId}:`, error);
        
        healthMap.set(pluginId, {
          status: 'unhealthy',
          uptime: 0,
          performance: {
            responseTime: 0,
            throughput: 0,
            errorRate: 1,
            memoryUsage: 0,
            cpuUsage: 0,
            activeConnections: 0
          },
          errors: {
            totalErrors: 1,
            errorRate: 1,
            lastError: error as Error
          },
          lastCheck: new Date(),
          checks: []
        });
      }
    }
    
    return healthMap;
  }

  private updateMetrics(pluginId: string, plugin: BasePlugin): void {
    const stats = plugin.getStats();
    
    this.metrics.set(pluginId, {
      pluginId,
      executionCount: stats.executions,
      averageExecutionTime: stats.uptime / Math.max(stats.executions, 1),
      errorRate: stats.errorRate,
      memoryUsage: 0, // Would be collected from plugin
      cpuUsage: 0, // Would be collected from plugin
      lastExecution: stats.lastActivity,
      uptime: stats.uptime
    });
  }

  async getPluginMetrics(pluginId: string): Promise<PluginMetrics | null> {
    return this.metrics.get(pluginId) || null;
  }

  async getAllMetrics(): Promise<PluginMetrics[]> {
    return Array.from(this.metrics.values());
  }

  /**
   * Plugin Updates
   */
  async checkForUpdates(): Promise<PluginUpdate[]> {
    const updates: PluginUpdate[] = [];
    
    for (const installation of this.pluginRegistry.values()) {
      try {
        const update = await this.checkPluginUpdate(installation);
        if (update) {
          updates.push(update);
        }
      } catch (error) {
        console.error(`Failed to check update for plugin ${installation.pluginId}:`, error);
      }
    }
    
    return updates;
  }

  private async checkPluginUpdate(installation: PluginInstallation): Promise<PluginUpdate | null> {
    // Mock update check
    const availableVersion = '1.1.0';
    
    if (availableVersion !== installation.version) {
      return {
        pluginId: installation.pluginId,
        currentVersion: installation.version,
        availableVersion,
        releaseNotes: 'Bug fixes and performance improvements',
        breaking: false,
        securityFix: false,
        autoUpdate: false
      };
    }
    
    return null;
  }

  async updatePlugin(pluginId: string, version?: string): Promise<InstallationResult> {
    const installation = this.pluginRegistry.get(pluginId);
    if (!installation) {
      throw new Error(`Plugin not found: ${pluginId}`);
    }

    try {
      // Find update
      const updates = await this.checkForUpdates();
      const update = updates.find(u => u.pluginId === pluginId);
      
      if (!update) {
        throw new Error('No update available');
      }

      const targetVersion = version || update.availableVersion;
      
      // Download new version
      const newSource: PluginSource = {
        ...installation.source,
        version: targetVersion
      };
      
      // Backup current configuration
      const backupConfig = { ...installation.config };
      
      try {
        // Install new version
        const result = await this.installPlugin(newSource, backupConfig);
        
        if (result.success) {
          // Deactivate old version
          await this.deactivatePlugin(pluginId);
          
          // Activate new version
          await this.activatePlugin(pluginId, backupConfig);
          
          console.log(`Plugin ${pluginId} updated from ${installation.version} to ${targetVersion}`);
        }
        
        return result;
        
      } catch (error) {
        // Rollback on failure
        console.error(`Update failed, rolling back: ${error.message}`);
        await this.activatePlugin(pluginId, backupConfig);
        throw error;
      }
      
    } catch (error) {
      console.error(`Failed to update plugin ${pluginId}:`, error);
      return {
        success: false,
        pluginId,
        version: '',
        installedAt: new Date(),
        errors: [error.message]
      };
    }
  }

  /**
   * Utility Methods
   */
  async getPluginMetadata(pluginId: string): Promise<PluginMetadata> {
    const pluginPath = path.join(this.pluginDirectory, pluginId);
    const metadataPath = path.join(pluginPath, 'plugin.json');
    
    const metadataContent = await fs.readFile(metadataPath, 'utf-8');
    return JSON.parse(metadataContent);
  }

  listInstalledPlugins(): PluginInstallation[] {
    return Array.from(this.pluginRegistry.values());
  }

  listActivePlugins(): PluginInstallation[] {
    return Array.from(this.pluginRegistry.values())
      .filter(installation => installation.status === PluginStatus.ACTIVE);
  }

  async getPluginStatus(pluginId: string): Promise<PluginStatus | null> {
    const installation = this.pluginRegistry.get(pluginId);
    return installation ? installation.status : null;
  }

  async updatePluginConfig(pluginId: string, config: Record<string, any>): Promise<void> {
    const installation = this.pluginRegistry.get(pluginId);
    if (!installation) {
      throw new Error(`Plugin not found: ${pluginId}`);
    }

    installation.config = { ...installation.config, ...config };
    
    // Restart plugin if active
    if (installation.status === PluginStatus.ACTIVE) {
      await this.deactivatePlugin(pluginId);
      await this.activatePlugin(pluginId, installation.config);
    }
  }

  async exportConfiguration(): Promise<any> {
    return {
      repositories: Array.from(this.repositories.values()),
      installedPlugins: this.listInstalledPlugins(),
      metrics: this.getAllMetrics(),
      sandboxConfig: this.sandboxConfig
    };
  }

  async importConfiguration(config: any): Promise<void> {
    // Import repositories
    if (config.repositories) {
      for (const repo of config.repositories) {
        await this.addRepository(repo);
      }
    }

    // Import sandbox configuration
    if (config.sandboxConfig) {
      this.sandboxConfig = config.sandboxConfig;
    }
  }
}