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
import { DataPipelineService } from '../services/data-pipeline/data-pipeline-service';
import { Logger } from 'winston';
import { 
  PipelineConfig, 
  ExecutionOptions, 
  ImportOptions, 
  ExportOptions,
  ValidationRule,
  SchemaDefinition,
  PipelineStatus,
  PipelineStats 
} from '../services/data-pipeline/types';

export function createDataPipelineRoutes(
  pipelineService: DataPipelineService,
  logger: Logger
): Router {
  const router = Router();

  // Pipeline Management
  router.post('/pipelines', async (req, res) => {
    try {
      const config: PipelineConfig = req.body;
      const pipelineId = await pipelineService.createPipeline(config);
      
      logger.info('Pipeline created', { pipelineId, name: config.name });
      res.status(201).json({ 
        success: true, 
        data: { pipelineId } 
      });
    } catch (error) {
      logger.error('Pipeline creation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/pipelines', async (req, res) => {
    try {
      const stats = await pipelineService.getPipelineStats();
      res.json({ 
        success: true, 
        data: stats 
      });
    } catch (error) {
      logger.error('Failed to get pipeline stats', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Pipeline Execution
  router.post('/pipelines/:pipelineId/execute', async (req, res) => {
    try {
      const { pipelineId } = req.params;
      const options: ExecutionOptions = req.body;
      
      const jobId = await pipelineService.executePipeline(pipelineId, options);
      
      logger.info('Pipeline execution started', { pipelineId, jobId });
      res.status(202).json({ 
        success: true, 
        data: { jobId } 
      });
    } catch (error) {
      logger.error('Pipeline execution failed', { 
        pipelineId: req.params.pipelineId, 
        error 
      });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/executions/:jobId/status', async (req, res) => {
    try {
      const { jobId } = req.params;
      const status: PipelineStatus = await pipelineService.getPipelineStatus(jobId);
      
      res.json({ 
        success: true, 
        data: status 
      });
    } catch (error) {
      logger.error('Failed to get execution status', { 
        jobId: req.params.jobId, 
        error 
      });
      res.status(404).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.delete('/executions/:jobId', async (req, res) => {
    try {
      const { jobId } = req.params;
      await pipelineService.cancelPipeline(jobId);
      
      logger.info('Pipeline execution cancelled', { jobId });
      res.json({ 
        success: true, 
        message: 'Pipeline execution cancelled' 
      });
    } catch (error) {
      logger.error('Failed to cancel pipeline execution', { 
        jobId: req.params.jobId, 
        error 
      });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Data Import/Export
  router.post('/import', async (req, res) => {
    try {
      const { source, target, options } = req.body;
      const importOptions: ImportOptions = options || {};
      
      const jobId = await pipelineService.importData(source, target, importOptions);
      
      logger.info('Data import started', { jobId, source: source.type, target: target.type });
      res.status(202).json({ 
        success: true, 
        data: { jobId } 
      });
    } catch (error) {
      logger.error('Data import failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/export', async (req, res) => {
    try {
      const { source, target, options } = req.body;
      const exportOptions: ExportOptions = options || {};
      
      const jobId = await pipelineService.exportData(source, target, exportOptions);
      
      logger.info('Data export started', { jobId, source: source.type, target: target.type });
      res.status(202).json({ 
        success: true, 
        data: { jobId } 
      });
    } catch (error) {
      logger.error('Data export failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Data Validation
  router.post('/validate', async (req, res) => {
    try {
      const { data, rules } = req.body;
      const validationRules: ValidationRule[] = rules;
      
      const result = await pipelineService.validateData(data, validationRules);
      
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('Data validation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Schema Management
  router.post('/schemas', async (req, res) => {
    try {
      const { name, schema } = req.body;
      const schemaDefinition: SchemaDefinition = schema;
      
      const schemaId = await pipelineService.createSchema(name, schemaDefinition);
      
      logger.info('Schema created', { schemaId, name });
      res.status(201).json({ 
        success: true, 
        data: { schemaId } 
      });
    } catch (error) {
      logger.error('Schema creation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.put('/schemas/:name', async (req, res) => {
    try {
      const { name } = req.params;
      const { schema, description } = req.body;
      const schemaDefinition: SchemaDefinition = schema;
      
      await pipelineService.updateSchema(name, schemaDefinition, description);
      
      logger.info('Schema updated', { name });
      res.json({ 
        success: true, 
        message: 'Schema updated successfully' 
      });
    } catch (error) {
      logger.error('Schema update failed', { name, error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/schemas/:name/validate', async (req, res) => {
    try {
      const { name } = req.params;
      const { data } = req.body;
      
      const isValid = await pipelineService.validateDataAgainstSchema(data, name);
      
      res.json({ 
        success: true, 
        data: { isValid } 
      });
    } catch (error) {
      logger.error('Schema validation failed', { 
        schemaName: req.params.name, 
        error 
      });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Real-time Event Handling
  pipelineService.on('pipeline_progress', (data) => {
    logger.debug('Pipeline progress', data);
    // In a real implementation, you would emit this to WebSocket clients
  });

  pipelineService.on('pipeline_complete', (data) => {
    logger.info('Pipeline completed', data);
    // Handle completion events
  });

  pipelineService.on('pipeline_error', (data) => {
    logger.error('Pipeline error', data);
    // Handle error events
  });

  return router;
}