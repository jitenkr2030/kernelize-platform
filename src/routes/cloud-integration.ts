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
import { CloudStorageService } from '../services/cloud/cloud-storage-service';
import { ServerlessDeploymentService } from '../services/serverless/serverless-deployment-service';
import { Logger } from 'winston';
import {
  CloudConfig,
  UploadParams,
  DownloadParams,
  DeleteParams,
  ListParams,
  MetadataParams,
  CopyParams,
  PresignedUrlParams,
  SyncParams,
  CreateBucketParams,
  DeleteBucketParams,
  DeployFunctionParams,
  UpdateFunctionParams,
  InvokeParams,
  DeleteParams as ServerlessDeleteParams,
  LogsParams,
  MetricsParams,
  ApiGatewayParams,
  MultiEnvironmentParams
} from '../services/data-pipeline/types';

export function createCloudIntegrationRoutes(
  cloudStorage: CloudStorageService,
  serverlessDeployment: ServerlessDeploymentService,
  logger: Logger
): Router {
  const router = Router();

  // Cloud Storage Management
  router.post('/storage/configure', async (req, res) => {
    try {
      const config: CloudConfig = req.body;
      await cloudStorage.configure(config);
      
      logger.info('Cloud storage configured', { provider: config.provider });
      res.json({ 
        success: true, 
        message: 'Cloud storage configured successfully' 
      });
    } catch (error) {
      logger.error('Cloud storage configuration failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // File Operations
  router.post('/storage/upload', async (req, res) => {
    try {
      const params: UploadParams = req.body;
      const result = await cloudStorage.uploadFile(params);
      
      logger.info('File uploaded to cloud storage', { 
        provider: params.provider, 
        bucket: params.bucket, 
        key: params.key 
      });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('File upload failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/storage/download', async (req, res) => {
    try {
      const params: DownloadParams = req.body;
      const result = await cloudStorage.downloadFile(params);
      
      logger.info('File downloaded from cloud storage', { 
        provider: params.provider, 
        bucket: params.bucket, 
        key: params.key 
      });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('File download failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.delete('/storage/files', async (req, res) => {
    try {
      const params: DeleteParams = req.body;
      await cloudStorage.deleteFile(params);
      
      logger.info('File deleted from cloud storage', { 
        provider: params.provider, 
        bucket: params.bucket, 
        key: params.key 
      });
      res.json({ 
        success: true, 
        message: 'File deleted successfully' 
      });
    } catch (error) {
      logger.error('File deletion failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/storage/list', async (req, res) => {
    try {
      const params: ListParams = req.body;
      const files = await cloudStorage.listFiles(params);
      
      res.json({ 
        success: true, 
        data: { files, count: files.length } 
      });
    } catch (error) {
      logger.error('File listing failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/storage/metadata', async (req, res) => {
    try {
      const params: MetadataParams = req.body;
      const metadata = await cloudStorage.getFileMetadata(params);
      
      res.json({ 
        success: true, 
        data: metadata 
      });
    } catch (error) {
      logger.error('Failed to get file metadata', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/storage/copy', async (req, res) => {
    try {
      const params: CopyParams = req.body;
      const result = await cloudStorage.copyFile(params);
      
      logger.info('File copied in cloud storage', { 
        provider: params.provider,
        sourceBucket: params.sourceBucket,
        sourceKey: params.sourceKey,
        destBucket: params.destBucket,
        destKey: params.destKey
      });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('File copy failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/storage/presigned-url', async (req, res) => {
    try {
      const params: PresignedUrlParams = req.body;
      const url = await cloudStorage.generatePresignedUrl(params);
      
      res.json({ 
        success: true, 
        data: { url } 
      });
    } catch (error) {
      logger.error('Failed to generate presigned URL', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/storage/sync', async (req, res) => {
    try {
      const params: SyncParams = req.body;
      const result = await cloudStorage.syncFiles(params);
      
      logger.info('Cloud storage sync completed', { 
        provider: params.provider,
        direction: params.direction,
        filesProcessed: result.filesProcessed
      });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('Cloud storage sync failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Bucket Management
  router.post('/storage/buckets', async (req, res) => {
    try {
      const params: CreateBucketParams = req.body;
      await cloudStorage.createBucket(params);
      
      logger.info('Cloud storage bucket created', { 
        provider: params.provider, 
        bucket: params.bucket 
      });
      res.status(201).json({ 
        success: true, 
        message: 'Bucket created successfully' 
      });
    } catch (error) {
      logger.error('Bucket creation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.delete('/storage/buckets/:provider/:bucket', async (req, res) => {
    try {
      const { provider, bucket } = req.params;
      await cloudStorage.deleteBucket({ provider, bucket });
      
      logger.info('Cloud storage bucket deleted', { provider, bucket });
      res.json({ 
        success: true, 
        message: 'Bucket deleted successfully' 
      });
    } catch (error) {
      logger.error('Bucket deletion failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/storage/stats/:provider', async (req, res) => {
    try {
      const { provider } = req.params;
      const stats = await cloudStorage.getStorageStats(provider);
      
      res.json({ 
        success: true, 
        data: stats 
      });
    } catch (error) {
      logger.error('Failed to get storage stats', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Serverless Deployment
  router.post('/serverless/deploy', async (req, res) => {
    try {
      const params: DeployFunctionParams = req.body;
      const result = await serverlessDeployment.deployFunction(params);
      
      logger.info('Serverless function deployed', { 
        functionName: params.functionName,
        platform: params.platform
      });
      res.status(201).json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('Serverless deployment failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.put('/serverless/functions/:deploymentId', async (req, res) => {
    try {
      const { deploymentId } = req.params;
      const params: UpdateFunctionParams = { deploymentId, ...req.body };
      const result = await serverlessDeployment.updateFunction(params);
      
      logger.info('Serverless function updated', { deploymentId });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('Serverless function update failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.post('/serverless/invoke', async (req, res) => {
    try {
      const params: InvokeParams = req.body;
      const result = await serverlessDeployment.invokeFunction(params);
      
      logger.info('Serverless function invoked', { 
        deploymentId: params.deploymentId 
      });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('Serverless function invocation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.delete('/serverless/functions/:deploymentId', async (req, res) => {
    try {
      const params: ServerlessDeleteParams = { deploymentId: req.params.deploymentId };
      await serverlessDeployment.deleteFunction(params);
      
      logger.info('Serverless function deleted', { 
        deploymentId: req.params.deploymentId 
      });
      res.json({ 
        success: true, 
        message: 'Function deleted successfully' 
      });
    } catch (error) {
      logger.error('Serverless function deletion failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/serverless/deployments', async (req, res) => {
    try {
      const deployments = serverlessDeployment.getDeployments();
      res.json({ 
        success: true, 
        data: { deployments, count: deployments.length } 
      });
    } catch (error) {
      logger.error('Failed to get deployments', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/serverless/deployments/:deploymentId', async (req, res) => {
    try {
      const deployment = serverlessDeployment.getDeployment(req.params.deploymentId);
      if (!deployment) {
        return res.status(404).json({ 
          success: false, 
          error: 'Deployment not found' 
        });
      }
      res.json({ 
        success: true, 
        data: deployment 
      });
    } catch (error) {
      logger.error('Failed to get deployment', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/serverless/functions/platform/:platform', async (req, res) => {
    try {
      const { platform } = req.params;
      const functions = serverlessDeployment.getFunctionsByPlatform(platform);
      res.json({ 
        success: true, 
        data: { functions, count: functions.length } 
      });
    } catch (error) {
      logger.error('Failed to get functions by platform', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/serverless/logs', async (req, res) => {
    try {
      const params: LogsParams = { 
        deploymentId: req.query.deploymentId as string,
        startTime: req.query.startTime ? new Date(req.query.startTime as string) : undefined,
        endTime: req.query.endTime ? new Date(req.query.endTime as string) : undefined,
        limit: req.query.limit ? parseInt(req.query.limit as string) : undefined
      };
      const logs = await serverlessDeployment.getFunctionLogs(params);
      
      res.json({ 
        success: true, 
        data: { logs, count: logs.length } 
      });
    } catch (error) {
      logger.error('Failed to get function logs', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  router.get('/serverless/metrics', async (req, res) => {
    try {
      const params: MetricsParams = {
        deploymentId: req.query.deploymentId as string,
        startTime: new Date(req.query.startTime as string),
        endTime: new Date(req.query.endTime as string)
      };
      const metrics = await serverlessDeployment.getFunctionMetrics(params);
      
      res.json({ 
        success: true, 
        data: metrics 
      });
    } catch (error) {
      logger.error('Failed to get function metrics', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // API Gateway Management
  router.post('/serverless/api-gateway', async (req, res) => {
    try {
      const params: ApiGatewayParams = req.body;
      const result = await serverlessDeployment.createApiGateway(params);
      
      logger.info('API Gateway created', { 
        deploymentId: params.deploymentId,
        apiId: result.apiId
      });
      res.status(201).json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('API Gateway creation failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Multi-environment Deployment
  router.post('/serverless/multi-environment', async (req, res) => {
    try {
      const params: MultiEnvironmentParams = req.body;
      const result = await serverlessDeployment.deployToEnvironments(params);
      
      logger.info('Multi-environment deployment completed', {
        functionName: params.functionName,
        totalEnvironments: result.totalEnvironments,
        successfulDeployments: result.successfulDeployments,
        failedDeployments: result.failedDeployments
      });
      res.json({ 
        success: true, 
        data: result 
      });
    } catch (error) {
      logger.error('Multi-environment deployment failed', { error });
      res.status(400).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  // Statistics
  router.get('/serverless/stats', async (req, res) => {
    try {
      const stats = serverlessDeployment.getStatistics();
      res.json({ 
        success: true, 
        data: stats 
      });
    } catch (error) {
      logger.error('Failed to get serverless stats', { error });
      res.status(500).json({ 
        success: false, 
        error: (error as Error).message 
      });
    }
  });

  return router;
}