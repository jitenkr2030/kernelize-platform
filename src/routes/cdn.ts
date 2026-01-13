/**
 * KERNELIZE Platform - Core Backend
 * Licensed under the Business Source License 1.1 (BSL 1.1)
 * 
 * Copyright (c) 2026 KERNELIZE Platform. All rights reserved.
 * 
 * See LICENSE-CORE in the project root for license information.
 * See LICENSE-SDK for SDK and tool licensing terms.
 */

import { Router } from 'express';
import { CDNIntegrationService } from '../services/cdn/cdn-integration-service';
import { Logger } from 'winston';
import {
  CreateDistributionParams,
  UpdateDistributionParams,
  DeleteDistributionParams,
  InvalidateParams,
  PurgeParams,
  CacheRulesParams,
  CacheStatsParams,
  MetricsParams,
  ErrorPageParams,
  ToggleParams
} from '../services/data-pipeline/types';

export function createCDNRoutes(
  cdnService: CDNIntegrationService,
  logger: Logger
): Router {
  const router = Router();

  // Distribution Management
  router.post('/distributions', async (req, res) => {
    try {
      const params: CreateDistributionParams = req.body;
      const result = await cdnService.createDistribution(params);
      
      logger.info('CDN distribution created', { 
        domainName: params.domainName,
        provider: params.provider 
      });
      res.status(201).json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('CDN distribution creation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.put('/distributions/:distributionId', async (req, res) => {
    try {
      const { distributionId } = req.params;
      const params: UpdateDistributionParams = { distributionId, config: req.body };
      const result = await cdnService.updateDistribution(params);
      
      logger.info('CDN distribution updated', { distributionId });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('CDN distribution update failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.delete('/distributions/:distributionId', async (req, res) => {
    try {
      const params: DeleteDistributionParams = { distributionId: req.params.distributionId };
      await cdnService.deleteDistribution(params);
      
      logger.info('CDN distribution deleted', { 
        distributionId: req.params.distributionId 
      });
      res.json({ 
        success: true, 
        message: 'Distribution deleted successfully' 
      });
    } catch (error) {
      logger.error('CDN distribution deletion failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/distributions', async (req, res) => {
    try {
      const distributions = cdnService.getDistributions();
      res.json({ 
        success: true, 
        data: { distributions, count: distributions.length } 
      });
    } catch (error) {
      logger.error('Failed to get distributions', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/distributions/:distributionId', async (req, res) => {
    try {
      const distribution = cdnService.getDistribution(req.params.distributionId);
      if (!distribution) {
        return res.status(404).json({ 
          success: false, 
          error: 'Distribution not found' 
        });
      }
      res.json({ 
        success: true, 
        data: distribution 
      });
    } catch (error) {
      logger.error('Failed to get distribution', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Distribution Status
  router.get('/distributions/:distributionId/status', async (req, res) => {
    try {
      const status = await cdnService.getDistributionStatus(req.params.distributionId);
      res.json({ 
        success: true, 
        data: status 
      });
    } catch (error) {
      logger.error('Failed to get distribution status', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Cache Management
  router.post('/distributions/:distributionId/invalidate', async (req, res) => {
    try {
      const params: InvalidateParams = { 
        distributionId: req.params.distributionId,
        paths: req.body.paths 
      };
      const result = await cdnService.invalidateCache(params);
      
      logger.info('CDN cache invalidation requested', { 
        distributionId: req.params.distributionId,
        pathCount: params.paths.length
      });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('CDN cache invalidation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/distributions/:distributionId/purge', async (req, res) => {
    try {
      const params: PurgeParams = { distributionId: req.params.distributionId };
      const result = await cdnService.purgeCache(params);
      
      logger.info('CDN cache purge requested', { 
        distributionId: req.params.distributionId 
      });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('CDN cache purge failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Cache Rules Management
  router.post('/distributions/:distributionId/cache-rules', async (req, res) => {
    try {
      const params: CacheRulesParams = { 
        distributionId: req.params.distributionId,
        rules: req.body.rules 
      };
      await cdnService.createCacheRules(params);
      
      logger.info('CDN cache rules created', { 
        distributionId: req.params.distributionId,
        ruleCount: params.rules.length
      });
      res.json({ 
        success: true, 
        message: 'Cache rules created successfully' 
      });
    } catch (error) {
      logger.error('CDN cache rules creation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/distributions/:distributionId/cache-rules', async (req, res) => {
    try {
      const rules = cdnService.getCacheRules(req.params.distributionId);
      res.json({ 
        success: true, 
        data: { rules, count: rules.length } 
      });
    } catch (error) {
      logger.error('Failed to get cache rules', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Cache Statistics and Metrics
  router.get('/distributions/:distributionId/cache-stats', async (req, res) => {
    try {
      const params: CacheStatsParams = {
        distributionId: req.params.distributionId,
        startTime: new Date(req.query.startTime as string),
        endTime: new Date(req.query.endTime as string)
      };
      const stats = await cdnService.getCacheStats(params);
      
      res.json({ 
        success: true, 
        data: stats 
      });
    } catch (error) {
      logger.error('Failed to get cache statistics', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/distributions/:distributionId/metrics', async (req, res) => {
    try {
      const params: MetricsParams = {
        distributionId: req.params.distributionId,
        startTime: new Date(req.query.startTime as string),
        endTime: new Date(req.query.endTime as string),
        metrics: req.query.metrics ? (req.query.metrics as string).split(',') : []
      };
      const metrics = await cdnService.getDistributionMetrics(params);
      
      res.json({ 
        success: true, 
        data: metrics 
      });
    } catch (error) {
      logger.error('Failed to get distribution metrics', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Error Pages
  router.post('/distributions/:distributionId/error-pages', async (req, res) => {
    try {
      const params: ErrorPageParams = { 
        distributionId: req.params.distributionId,
        ...req.body 
      };
      const result = await cdnService.createErrorPage(params);
      
      logger.info('CDN error page created', { 
        distributionId: req.params.distributionId,
        errorCode: params.errorCode
      });
      res.status(201).json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('CDN error page creation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Distribution Control
  router.post('/distributions/:distributionId/toggle', async (req, res) => {
    try {
      const params: ToggleParams = { 
        distributionId: req.params.distributionId,
        enabled: req.body.enabled 
      };
      await cdnService.toggleDistribution(params);
      
      logger.info('CDN distribution toggled', { 
        distributionId: req.params.distributionId,
        enabled: params.enabled
      });
      res.json({ 
        success: true, 
        message: `Distribution ${params.enabled ? 'enabled' : 'disabled'} successfully` 
      });
    } catch (error) {
      logger.error('CDN distribution toggle failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Statistics
  router.get('/stats', async (req, res) => {
    try {
      const stats = cdnService.getStatistics();
      res.json({ 
        success: true, 
        data: stats 
      });
    } catch (error) {
      logger.error('Failed to get CDN statistics', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  return router;
}