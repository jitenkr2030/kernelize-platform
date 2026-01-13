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

import { Logger } from 'winston';
import { EventEmitter } from 'events';

export class CDNIntegrationService extends EventEmitter {
  private logger: Logger;
  private providers: Map<string, CDNProvider> = new Map();
  private distributions: Map<string, CDNDistribution> = new Map();
  private cacheRules: Map<string, CacheRule[]> = new Map();

  constructor(logger: Logger) {
    super();
    this.logger = logger;
    this.initializeProviders();
  }

  /**
   * Register CDN provider
   */
  registerProvider(name: string, provider: CDNProvider): void {
    this.providers.set(name, provider);
    this.logger.info('CDN provider registered', { provider: name });
  }

  /**
   * Create CDN distribution
   */
  async createDistribution(params: CreateDistributionParams): Promise<DistributionResult> {
    const {
      provider,
      originDomain,
      domainName,
      certificateArn,
      comment,
      defaultCacheBehavior,
      cacheBehaviors = [],
      priceClass = 'PriceClass_100',
      geoRestriction,
      logging
    } = params;

    this.logger.info('Creating CDN distribution', {
      provider,
      originDomain,
      domainName,
      priceClass
    });

    const providerInstance = this.providers.get(provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${provider}`);
    }

    try {
      const distributionConfig: CDNDistribution = {
        id: this.generateDistributionId(),
        provider,
        originDomain,
        domainName,
        certificateArn,
        comment,
        status: 'creating',
        enabled: true,
        priceClass,
        geoRestriction,
        logging,
        defaultCacheBehavior,
        cacheBehaviors,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      const result = await providerInstance.createDistribution(distributionConfig);

      // Store distribution info
      this.distributions.set(result.distributionId, {
        ...distributionConfig,
        arn: result.arn,
        status: 'creating'
      });

      this.emit('distribution_creating', {
        distributionId: result.distributionId,
        provider,
        domainName
      });

      this.logger.info('CDN distribution creation initiated', {
        distributionId: result.distributionId,
        provider,
        domainName,
        arn: result.arn
      });

      return result;

    } catch (error) {
      this.logger.error('CDN distribution creation failed', {
        provider,
        originDomain,
        domainName,
        error: error as Error
      });

      this.emit('distribution_error', {
        provider,
        domainName,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Update CDN distribution
   */
  async updateDistribution(params: UpdateDistributionParams): Promise<UpdateResult> {
    const { distributionId, config } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    this.logger.info('Updating CDN distribution', {
      distributionId,
      provider: distribution.provider,
      domainName: distribution.domainName
    });

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      const result = await providerInstance.updateDistribution(distributionId, config);

      // Update stored distribution
      const updatedDistribution = {
        ...distribution,
        ...config,
        updatedAt: new Date()
      };
      this.distributions.set(distributionId, updatedDistribution);

      this.emit('distribution_updated', {
        distributionId,
        provider: distribution.provider,
        domainName: distribution.domainName
      });

      this.logger.info('CDN distribution updated successfully', {
        distributionId,
        provider: distribution.provider,
        domainName: distribution.domainName
      });

      return result;

    } catch (error) {
      this.logger.error('CDN distribution update failed', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Delete CDN distribution
   */
  async deleteDistribution(params: DeleteDistributionParams): Promise<void> {
    const { distributionId } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    this.logger.info('Deleting CDN distribution', {
      distributionId,
      provider: distribution.provider,
      domainName: distribution.domainName
    });

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      await providerInstance.deleteDistribution(distributionId);

      // Remove from storage
      this.distributions.delete(distributionId);
      this.cacheRules.delete(distributionId);

      this.emit('distribution_deleted', {
        distributionId,
        provider: distribution.provider,
        domainName: distribution.domainName
      });

      this.logger.info('CDN distribution deleted successfully', {
        distributionId,
        provider: distribution.provider,
        domainName: distribution.domainName
      });

    } catch (error) {
      this.logger.error('CDN distribution deletion failed', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Invalidate cache
   */
  async invalidateCache(params: InvalidateParams): Promise<InvalidationResult> {
    const { distributionId, paths } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    this.logger.info('Invalidating CDN cache', {
      distributionId,
      pathCount: paths.length,
      paths: paths.slice(0, 5) // Log first 5 paths
    });

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      const result = await providerInstance.invalidateCache(distributionId, {
        paths,
        callerReference: this.generateCallerReference()
      });

      this.emit('cache_invalidated', {
        distributionId,
        provider: distribution.provider,
        invalidationId: result.invalidationId,
        pathCount: paths.length
      });

      this.logger.info('CDN cache invalidation completed', {
        distributionId,
        invalidationId: result.invalidationId,
        pathCount: paths.length
      });

      return result;

    } catch (error) {
      this.logger.error('CDN cache invalidation failed', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Purge cache completely
   */
  async purgeCache(params: PurgeParams): Promise<PurgeResult> {
    const { distributionId } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    this.logger.info('Purging CDN cache', {
      distributionId,
      provider: distribution.provider,
      domainName: distribution.domainName
    });

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      const result = await providerInstance.purgeCache(distributionId);

      this.emit('cache_purged', {
        distributionId,
        provider: distribution.provider,
        purgeId: result.purgeId
      });

      this.logger.info('CDN cache purge completed', {
        distributionId,
        purgeId: result.purgeId,
        provider: distribution.provider
      });

      return result;

    } catch (error) {
      this.logger.error('CDN cache purge failed', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Create cache rules
   */
  async createCacheRules(params: CacheRulesParams): Promise<void> {
    const { distributionId, rules } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    this.logger.info('Creating CDN cache rules', {
      distributionId,
      ruleCount: rules.length,
      provider: distribution.provider
    });

    // Validate cache rules
    for (const rule of rules) {
      if (!rule.pathPattern || !rule.cacheBehavior) {
        throw new Error('Cache rule must have pathPattern and cacheBehavior');
      }
    }

    // Store cache rules
    this.cacheRules.set(distributionId, rules);

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      await providerInstance.updateCacheBehaviors(distributionId, {
        defaultCacheBehavior: distribution.defaultCacheBehavior,
        cacheBehaviors: rules
      });

      this.emit('cache_rules_updated', {
        distributionId,
        provider: distribution.provider,
        ruleCount: rules.length
      });

      this.logger.info('CDN cache rules updated successfully', {
        distributionId,
        provider: distribution.provider,
        ruleCount: rules.length
      });

    } catch (error) {
      this.logger.error('CDN cache rules update failed', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Get distribution status
   */
  async getDistributionStatus(distributionId: string): Promise<DistributionStatus> {
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      const status = await providerInstance.getDistributionStatus(distributionId);
      
      // Update stored distribution status
      const updatedDistribution = {
        ...distribution,
        status: status.status,
        lastModifiedTime: status.lastModifiedTime,
        enabled: status.enabled
      };
      this.distributions.set(distributionId, updatedDistribution);

      return status;

    } catch (error) {
      this.logger.error('Failed to get distribution status', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Get cache statistics
   */
  async getCacheStats(params: CacheStatsParams): Promise<CacheStatistics> {
    const { distributionId, startTime, endTime } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      const stats = await providerInstance.getCacheStatistics(distributionId, {
        startTime,
        endTime
      });

      return stats;

    } catch (error) {
      this.logger.error('Failed to get cache statistics', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Get distribution metrics
   */
  async getDistributionMetrics(params: MetricsParams): Promise<DistributionMetrics> {
    const { distributionId, startTime, endTime, metrics = [] } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      const metricsData = await providerInstance.getDistributionMetrics(distributionId, {
        startTime,
        endTime,
        metrics
      });

      return metricsData;

    } catch (error) {
      this.logger.error('Failed to get distribution metrics', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Create custom error page
   */
  async createErrorPage(params: ErrorPageParams): Promise<ErrorPageResult> {
    const { distributionId, errorCode, responsePagePath, responseCode, errorCachingMinTTL } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    this.logger.info('Creating CDN error page', {
      distributionId,
      errorCode,
      responsePagePath,
      responseCode
    });

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      const result = await providerInstance.createErrorPage(distributionId, {
        errorCode,
        responsePagePath,
        responseCode,
        errorCachingMinTTL
      });

      this.emit('error_page_created', {
        distributionId,
        errorCode,
        responsePagePath
      });

      this.logger.info('CDN error page created successfully', {
        distributionId,
        errorCode,
        responsePagePath
      });

      return result;

    } catch (error) {
      this.logger.error('CDN error page creation failed', {
        distributionId,
        errorCode,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Enable/disable distribution
   */
  async toggleDistribution(params: ToggleParams): Promise<void> {
    const { distributionId, enabled } = params;
    
    const distribution = this.distributions.get(distributionId);
    if (!distribution) {
      throw new Error(`Distribution not found: ${distributionId}`);
    }

    this.logger.info('Toggling CDN distribution', {
      distributionId,
      enabled,
      provider: distribution.provider
    });

    const providerInstance = this.providers.get(distribution.provider);
    if (!providerInstance) {
      throw new Error(`CDN provider not registered: ${distribution.provider}`);
    }

    try {
      await providerInstance.toggleDistribution(distributionId, { enabled });

      // Update stored distribution
      const updatedDistribution = {
        ...distribution,
        enabled,
        updatedAt: new Date()
      };
      this.distributions.set(distributionId, updatedDistribution);

      this.emit('distribution_toggled', {
        distributionId,
        provider: distribution.provider,
        enabled
      });

      this.logger.info('CDN distribution toggled successfully', {
        distributionId,
        provider: distribution.provider,
        enabled
      });

    } catch (error) {
      this.logger.error('CDN distribution toggle failed', {
        distributionId,
        provider: distribution.provider,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * List all distributions
   */
  getDistributions(): CDNDistribution[] {
    return Array.from(this.distributions.values());
  }

  /**
   * Get distribution by ID
   */
  getDistribution(distributionId: string): CDNDistribution | null {
    return this.distributions.get(distributionId) || null;
  }

  /**
   * Get cache rules for distribution
   */
  getCacheRules(distributionId: string): CacheRule[] {
    return this.cacheRules.get(distributionId) || [];
  }

  /**
   * Get CDN statistics
   */
  getStatistics(): CDNStatistics {
    const providers = new Set(this.distributions.values().map(d => d.provider));
    const totalDistributions = this.distributions.size;
    const activeDistributions = Array.from(this.distributions.values())
      .filter(d => d.status === 'deployed' && d.enabled).length;

    const providerStats = new Map<string, number>();
    for (const provider of providers) {
      providerStats.set(provider, 
        Array.from(this.distributions.values()).filter(d => d.provider === provider).length
      );
    }

    return {
      totalDistributions,
      activeDistributions,
      totalProviders: providers.size,
      providerStats: Object.fromEntries(providerStats)
    };
  }

  private generateDistributionId(): string {
    return `cdn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private generateCallerReference(): string {
    return `ref_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  private initializeProviders(): void {
    // Register built-in CDN providers
    // this.registerProvider('aws_cloudfront', new AWSCloudFrontProvider());
    // this.registerProvider('azure_cdn', new AzureCDNProvider());
    // this.registerProvider('google_cloud_cdn', new GoogleCloudCDNProvider());
    // this.registerProvider('cloudflare', new CloudflareProvider());
    // this.registerProvider('fastly', new FastlyProvider());
    
    this.logger.info('CDN providers initialized');
  }
}

// CDN Provider Interface
export interface CDNProvider {
  createDistribution(config: CDNDistribution): Promise<DistributionResult>;
  updateDistribution(distributionId: string, config: any): Promise<UpdateResult>;
  deleteDistribution(distributionId: string): Promise<void>;
  getDistributionStatus(distributionId: string): Promise<DistributionStatus>;
  invalidateCache(distributionId: string, params: InvalidationParams): Promise<InvalidationResult>;
  purgeCache(distributionId: string): Promise<PurgeResult>;
  updateCacheBehaviors(distributionId: string, params: CacheBehaviorsParams): Promise<void>;
  getCacheStatistics(distributionId: string, params: CacheStatsParams): Promise<CacheStatistics>;
  getDistributionMetrics(distributionId: string, params: MetricsParams): Promise<DistributionMetrics>;
  createErrorPage(distributionId: string, params: ErrorPageCreationParams): Promise<ErrorPageResult>;
  toggleDistribution(distributionId: string, params: ToggleParams): Promise<void>;
}

// Core Types
export interface CDNDistribution {
  id: string;
  provider: string;
  originDomain: string;
  domainName: string;
  certificateArn?: string;
  comment?: string;
  status: 'creating' | 'deployed' | 'in_progress' | 'disabled' | 'error';
  enabled: boolean;
  priceClass: string;
  geoRestriction?: GeoRestriction;
  logging?: LoggingConfig;
  defaultCacheBehavior: CacheBehavior;
  cacheBehaviors: CacheBehavior[];
  arn?: string;
  lastModifiedTime?: Date;
  createdAt: Date;
  updatedAt: Date;
}

export interface CacheRule {
  pathPattern: string;
  cacheBehavior: CacheBehavior;
  priority: number;
  targetOriginId: string;
  viewerProtocolPolicy: 'redirect-to-https' | 'allow-all';
  trustedSigners?: string[];
  minTTL: number;
  defaultTTL: number;
  maxTTL: number;
}

export interface CacheBehavior {
  targetOriginId: string;
  viewerProtocolPolicy: 'redirect-to-https' | 'allow-all';
  allowedMethods: string[];
  cachedMethods: string[];
  compress: boolean;
  minTTL: number;
  defaultTTL: number;
  maxTTL: number;
  trustedSigners?: string[];
  forwardedValues: ForwardedValues;
}

export interface ForwardedValues {
  queryString: boolean;
  queryStringCacheKeys: string[];
  headers: string[];
  cookies: CookiePreference;
}

export interface CookiePreference {
  forward: 'none' | 'whitelist' | 'all';
  whitelistedNames?: string[];
}

export interface GeoRestriction {
  restrictionType: 'none' | 'blacklist' | 'whitelist';
  locations: string[];
}

// Parameter Types
export interface CreateDistributionParams {
  provider: string;
  originDomain: string;
  domainName: string;
  certificateArn?: string;
  comment?: string;
  defaultCacheBehavior: CacheBehavior;
  cacheBehaviors?: CacheBehavior[];
  priceClass?: string;
  geoRestriction?: GeoRestriction;
  logging?: LoggingConfig;
}

export interface UpdateDistributionParams {
  distributionId: string;
  config: Partial<CDNDistribution>;
}

export interface DeleteDistributionParams {
  distributionId: string;
}

export interface InvalidateParams {
  distributionId: string;
  paths: string[];
}

export interface PurgeParams {
  distributionId: string;
}

export interface CacheRulesParams {
  distributionId: string;
  rules: CacheRule[];
}

export interface CacheStatsParams {
  distributionId: string;
  startTime: Date;
  endTime: Date;
}

export interface MetricsParams {
  distributionId: string;
  startTime: Date;
  endTime: Date;
  metrics?: string[];
}

export interface ErrorPageParams {
  distributionId: string;
  errorCode: number;
  responsePagePath: string;
  responseCode?: number;
  errorCachingMinTTL?: number;
}

export interface ToggleParams {
  distributionId: string;
  enabled: boolean;
}

// Result Types
export interface DistributionResult {
  distributionId: string;
  arn: string;
  domainName: string;
  status: string;
  enabled: boolean;
}

export interface UpdateResult {
  distributionId: string;
  etag: string;
  lastModifiedTime: Date;
}

export interface DistributionStatus {
  status: string;
  lastModifiedTime: Date;
  enabled: boolean;
  inProgressInvalidationBatches?: number;
}

export interface InvalidationResult {
  invalidationId: string;
  status: string;
  createTime: Date;
}

export interface PurgeResult {
  purgeId: string;
  status: string;
  createTime: Date;
}

export interface CacheStatistics {
  requests: number;
  hits: number;
  misses: number;
  hitRate: number;
  bandwidth: number;
  errorRate: number;
}

export interface DistributionMetrics {
  requests: MetricData[];
  bandwidth: MetricData[];
  errors: MetricData[];
  cacheHitRate: MetricData[];
}

export interface MetricData {
  timestamp: Date;
  value: number;
  unit: string;
}

export interface ErrorPageResult {
  errorPageId: string;
  status: string;
}

export interface LoggingConfig {
  enabled: boolean;
  bucket?: string;
  prefix?: string;
  includeCookies?: boolean;
}

export interface CDNStatistics {
  totalDistributions: number;
  activeDistributions: number;
  totalProviders: number;
  providerStats: Record<string, number>;
}

// Provider-specific parameter types
export interface InvalidationParams {
  paths: string[];
  callerReference: string;
}

export interface CacheBehaviorsParams {
  defaultCacheBehavior: CacheBehavior;
  cacheBehaviors: CacheRule[];
}

export interface CacheStatsParams {
  startTime: Date;
  endTime: Date;
}

export interface MetricsParams {
  startTime: Date;
  endTime: Date;
  metrics: string[];
}

export interface ErrorPageCreationParams {
  errorCode: number;
  responsePagePath: string;
  responseCode?: number;
  errorCachingMinTTL?: number;
}

export interface ToggleParams {
  enabled: boolean;
}