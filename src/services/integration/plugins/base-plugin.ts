/**
 * KERNELIZE Platform - Plugin Base Interface
 * Core plugin interface and base class implementation
 */

import {
  Plugin,
  PluginConfig,
  PluginMetadata,
  PluginStatus,
  ValidationResult,
  PluginHealth,
  Permission,
  ResourceLimits,
  CompressedData,
  CompressionStats
} from './plugin-types.js';

export abstract class BasePlugin implements Plugin {
  public readonly id: string;
  public readonly name: string;
  public readonly version: string;
  public readonly description: string;
  public readonly author: string;
  public readonly category: any;
  public readonly dependencies: string[];
  public readonly permissions: Permission[];
  public readonly config: PluginConfig;
  
  protected status: PluginStatus = PluginStatus.INSTALLED;
  protected health: PluginHealth;
  protected lastActivity: Date = new Date();
  protected executionCount: number = 0;
  protected errorCount: number = 0;
  protected startTime: Date = new Date();

  constructor(metadata: PluginMetadata, config: PluginConfig = {}) {
    this.id = metadata.id;
    this.name = metadata.name;
    this.version = metadata.version;
    this.description = metadata.description;
    this.author = metadata.author;
    this.category = metadata.category;
    this.dependencies = metadata.compatibility.supportedNodes;
    this.permissions = config.permissions || [];
    this.config = config;
    
    this.health = {
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

  abstract async initialize(config: PluginConfig): Promise<void>;
  abstract async execute(input: any): Promise<any>;
  abstract async shutdown(): Promise<void>;
  abstract validate(): ValidationResult;

  public getMetadata(): PluginMetadata {
    return {
      id: this.id,
      name: this.name,
      version: this.version,
      description: this.description,
      author: this.author,
      category: this.category,
      keywords: [],
      license: 'MIT',
      createdAt: this.startTime,
      updatedAt: new Date(),
      downloads: 0,
      rating: 0,
      compatibility: {
        minPlatformVersion: '1.0.0',
        supportedNodes: this.dependencies
      }
    };
  }

  public getHealth(): PluginHealth {
    this.health.uptime = Date.now() - this.startTime.getTime();
    this.health.lastCheck = new Date();
    this.health.errors.totalErrors = this.errorCount;
    this.health.errors.errorRate = this.executionCount > 0 ? this.errorCount / this.executionCount : 0;
    
    return this.health;
  }

  public getStatus(): PluginStatus {
    return this.status;
  }

  public setStatus(status: PluginStatus): void {
    this.status = status;
    this.lastActivity = new Date();
  }

  public async executeWithMonitoring(input: any): Promise<any> {
    const startTime = Date.now();
    
    try {
      this.executionCount++;
      this.setStatus(PluginStatus.ACTIVE);
      
      const result = await this.execute(input);
      
      const executionTime = Date.now() - startTime;
      this.updatePerformanceMetrics(executionTime, true);
      
      this.setStatus(PluginStatus.ACTIVE);
      return result;
      
    } catch (error) {
      this.errorCount++;
      this.updatePerformanceMetrics(Date.now() - startTime, false);
      throw error;
    }
  }

  private updatePerformanceMetrics(executionTime: number, success: boolean): void {
    const currentHealth = this.health;
    
    currentHealth.performance.responseTime = executionTime;
    currentHealth.performance.errorRate = success ? 0 : 1;
    
    if (!success) {
      currentHealth.errors.lastError = new Error('Execution failed');
    }
  }

  public checkPermissions(requiredPermissions: Permission[]): boolean {
    return requiredPermissions.every(req => 
      this.permissions.some(perm => 
        perm.resource === req.resource && 
        req.actions.every(action => perm.actions.includes(action))
      )
    );
  }

  public validateConfig(config: PluginConfig): ValidationResult {
    const errors: any[] = [];
    const warnings: any[] = [];

    // Basic validation
    if (!config) {
      errors.push({
        field: 'config',
        code: 'REQUIRED',
        message: 'Configuration is required'
      });
    }

    // Permission validation
    if (config.permissions && !Array.isArray(config.permissions)) {
      errors.push({
        field: 'permissions',
        code: 'INVALID_TYPE',
        message: 'Permissions must be an array'
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  public getStats() {
    return {
      uptime: Date.now() - this.startTime.getTime(),
      executions: this.executionCount,
      errors: this.errorCount,
      errorRate: this.executionCount > 0 ? this.errorCount / this.executionCount : 0,
      lastActivity: this.lastActivity,
      status: this.status
    };
  }
}

export class PluginFactory {
  private static plugins: Map<string, typeof BasePlugin> = new Map();

  public static registerPlugin(id: string, pluginClass: typeof BasePlugin): void {
    this.plugins.set(id, pluginClass);
  }

  public static createPlugin(id: string, metadata: PluginMetadata, config: PluginConfig): BasePlugin {
    const PluginClass = this.plugins.get(id);
    
    if (!PluginClass) {
      throw new Error(`Plugin class not found: ${id}`);
    }

    return new PluginClass(metadata, config);
  }

  public static getRegisteredPlugins(): string[] {
    return Array.from(this.plugins.keys());
  }
}

export class PluginLifecycleManager {
  private plugins: Map<string, BasePlugin> = new Map();

  public async installPlugin(plugin: BasePlugin): Promise<void> {
    plugin.setStatus(PluginStatus.INSTALLED);
    await plugin.initialize(plugin.config);
    plugin.setStatus(PluginStatus.ACTIVE);
    
    this.plugins.set(plugin.id, plugin);
  }

  public async activatePlugin(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      throw new Error(`Plugin not found: ${pluginId}`);
    }

    plugin.setStatus(PluginStatus.ACTIVE);
  }

  public async deactivatePlugin(pluginId: string): Promise<void> {
    const plugin = this.plugins.get(pluginId);
    if (!plugin) {
      throw new Error(`Plugin not found: ${pluginId}`);
    }

    await plugin.shutdown();
    plugin.setStatus(PluginStatus.INACTIVE);
  }

  public getPlugin(pluginId: string): BasePlugin | undefined {
    return this.plugins.get(pluginId);
  }

  public getAllPlugins(): BasePlugin[] {
    return Array.from(this.plugins.values());
  }

  public async healthCheck(): Promise<Map<string, PluginHealth>> {
    const healthMap = new Map<string, PluginHealth>();
    
    for (const [id, plugin] of this.plugins) {
      try {
        const health = plugin.getHealth();
        healthMap.set(id, health);
      } catch (error) {
        healthMap.set(id, {
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
}