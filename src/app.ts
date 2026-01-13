/**
 * KERNELIZE Platform - Core Backend
 * Licensed under the Business Source License 1.1 (BSL 1.1)
 * 
 * Copyright (c) 2026 KERNELIZE Platform. All rights reserved.
 * 
 * See LICENSE-CORE in the project root for license information.
 * See LICENSE-SDK for SDK and tool licensing terms.
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import compression from 'compression';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import winston from 'winston';
import { DataPipelineService } from './services/data-pipeline/data-pipeline-service';
import { ETLEngine } from './services/data-pipeline/etl-engine';
import { DataValidator } from './services/data-pipeline/data-validator';
import { SchemaManager } from './services/data-pipeline/schema-manager';
import { DataReader } from './services/data-pipeline/data-reader';
import { DataWriter } from './services/data-pipeline/data-writer';
import { DataTransformer } from './services/data-pipeline/data-transformer';
import { BatchProcessor } from './services/data-pipeline/batch-processor';
import { CloudStorageService } from './services/cloud/cloud-storage-service';
import { ServerlessDeploymentService } from './services/serverless/serverless-deployment-service';
import { CDNIntegrationService } from './services/cdn/cdn-integration-service';
import { CompressionService } from './services/compression/compression-service';
import CompressionAnalyticsService from './services/analytics/compression-analytics-service';
import BusinessIntelligenceService from './services/analytics/business-intelligence-service';
import EnterpriseSecurityService from './services/security/enterprise-security-service';
import HighAvailabilityService from './services/high-availability/high-availability-service';
import { createDataPipelineRoutes } from './routes/data-pipeline';
import { createCloudIntegrationRoutes } from './routes/cloud-integration';
import { createCDNRoutes } from './routes/cdn';
import analyticsRoutes from './routes/analytics';
import businessIntelligenceRoutes from './routes/business-intelligence';
import enterpriseSecurityRoutes from './routes/enterprise-security';
import highAvailabilityRoutes from './routes/high-availability';
import { IntegrationEcosystemRoutes } from './routes/integration';
import { ServiceConfig } from './services/data-pipeline/types';

// Initialize logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'kernelize-backend' },
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' })
  ]
});

// Initialize services
const compressionService = new CompressionService(logger);
const dataValidator = new DataValidator(logger);
const schemaManager = new SchemaManager(logger);
const dataReader = new DataReader(logger);
const dataWriter = new DataWriter(logger);
const dataTransformer = new DataTransformer(logger);
const batchProcessor = new BatchProcessor(logger);
const etlEngine = new ETLEngine(
  logger,
  dataReader,
  dataWriter,
  dataTransformer,
  compressionService,
  batchProcessor
);

const pipelineService = new DataPipelineService(
  logger,
  etlEngine,
  dataValidator,
  schemaManager,
  new CloudStorageService(logger) // Will be properly initialized later
);

const cloudStorageService = new CloudStorageService(logger);
const serverlessDeploymentService = new ServerlessDeploymentService(logger);
const cdnService = new CDNIntegrationService(logger);
const analyticsService = new CompressionAnalyticsService();
const businessIntelligenceService = new BusinessIntelligenceService();
const securityService = new EnterpriseSecurityService(logger);
const highAvailabilityService = new HighAvailabilityService(logger);

// Create Express app
const app = express();
const server = createServer(app);

// Configure middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "wss:", "ws:"],
      fontSrc: ["'self'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"]
    }
  },
  crossOriginEmbedderPolicy: false
}));

app.use(cors({
  origin: process.env.FRONTEND_URL || 'http://localhost:3000',
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

app.use(compression());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
  message: {
    error: 'Too many requests from this IP, please try again later.'
  },
  standardHeaders: true,
  legacyHeaders: false
});
app.use(limiter);

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    version: process.env.npm_package_version || '1.0.0',
    services: {
      dataPipeline: 'available',
      cloudStorage: 'available',
      serverless: 'available',
      cdn: 'available',
      analytics: 'available',
      businessIntelligence: 'available',
      security: 'available',
      highAvailability: 'available',
      integrationEcosystem: 'available'
    }
  });
});

// API Routes
const apiRouter = express.Router();

// Data Pipeline Routes
apiRouter.use('/data-pipeline', createDataPipelineRoutes(pipelineService, logger));

// Cloud Integration Routes
apiRouter.use('/cloud', createCloudIntegrationRoutes(
  cloudStorageService,
  serverlessDeploymentService,
  logger
));

// CDN Routes
apiRouter.use('/cdn', createCDNRoutes(cdnService, logger));

// Analytics Routes
apiRouter.use('/analytics', analyticsRoutes);

// Business Intelligence Routes
apiRouter.use('/business-intelligence', businessIntelligenceRoutes);

// Enterprise Security Routes
apiRouter.use('/security', enterpriseSecurityRoutes);

// High Availability Routes
apiRouter.use('/high-availability', highAvailabilityRoutes);

// Integration Ecosystem Routes
const integrationRoutes = new IntegrationEcosystemRoutes();
apiRouter.use('/integration', integrationRoutes.getRouter());

// Version info
apiRouter.get('/version', (req, res) => {
  res.json({
    version: '1.0.0',
    buildDate: process.env.BUILD_DATE || new Date().toISOString(),
    environment: process.env.NODE_ENV || 'development',
    features: {
      dataPipeline: true,
      etl: true,
      validation: true,
      schemaManagement: true,
      cloudStorage: true,
      serverlessDeployment: true,
      cdnIntegration: true,
      compressionAnalytics: true,
      businessIntelligence: true,
      usagePatternAnalysis: true,
      costAnalysis: true,
      performanceBenchmarking: true,
      trendAnalysis: true,
      ssoAuthentication: true,
      roleBasedAccessControl: true,
      dataEncryption: true,
      complianceMonitoring: true,
      multiRegionDeployment: true,
      automatedFailover: true,
      loadBalancingOptimization: true,
      disasterRecovery: true,
      integrationEcosystem: true,
      pluginArchitecture: true,
      thirdPartyConnectors: true,
      workflowAutomation: true,
      apiIntegration: true,
      securityCompliance: true,
      performanceOptimization: true
    }
  });
});

app.use('/api/v1', apiRouter);

// WebSocket for real-time updates
const wss = new WebSocketServer({ server });

wss.on('connection', (ws) => {
  logger.info('WebSocket client connected');
  
  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message.toString());
      handleWebSocketMessage(ws, data);
    } catch (error) {
      logger.error('WebSocket message parsing failed', { error });
    }
  });

  ws.on('close', () => {
    logger.info('WebSocket client disconnected');
  });

  // Send initial connection message
  ws.send(JSON.stringify({
    type: 'connection',
    message: 'Connected to KERNELIZE real-time updates',
    timestamp: new Date().toISOString()
  }));
});

// Handle WebSocket messages
function handleWebSocketMessage(ws: any, data: any) {
  switch (data.type) {
    case 'subscribe':
      // Subscribe to pipeline updates
      ws.subscriptions = ws.subscriptions || new Set();
      ws.subscriptions.add(data.channel);
      ws.send(JSON.stringify({
        type: 'subscribed',
        channel: data.channel,
        timestamp: new Date().toISOString()
      }));
      break;
    
    case 'unsubscribe':
      if (ws.subscriptions) {
        ws.subscriptions.delete(data.channel);
      }
      ws.send(JSON.stringify({
        type: 'unsubscribed',
        channel: data.channel,
        timestamp: new Date().toISOString()
      }));
      break;
    
    default:
      ws.send(JSON.stringify({
        type: 'error',
        message: 'Unknown message type',
        timestamp: new Date().toISOString()
      }));
  }
}

// Broadcast real-time updates
function broadcastUpdate(channel: string, data: any) {
  wss.clients.forEach((client: any) => {
    if (client.readyState === 1 && client.subscriptions && client.subscriptions.has(channel)) {
      client.send(JSON.stringify({
        type: 'update',
        channel,
        data,
        timestamp: new Date().toISOString()
      }));
    }
  });
}

// Set up real-time event listeners
pipelineService.on('pipeline_progress', (data) => {
  broadcastUpdate('pipeline_progress', data);
});

pipelineService.on('pipeline_complete', (data) => {
  broadcastUpdate('pipeline_complete', data);
});

pipelineService.on('pipeline_error', (data) => {
  broadcastUpdate('pipeline_error', data);
});

serverlessDeploymentService.on('function_deployed', (data) => {
  broadcastUpdate('serverless_deployment', data);
});

serverlessDeploymentService.on('function_invoked', (data) => {
  broadcastUpdate('serverless_invocation', data);
});

cdnService.on('distribution_creating', (data) => {
  broadcastUpdate('cdn_distribution', data);
});

cdnService.on('cache_invalidated', (data) => {
  broadcastUpdate('cdn_cache', data);
});

// Analytics real-time updates
analyticsService.addWebSocketClient = (ws: any) => {
  wss.clients.add(ws);
  ws.on('close', () => {
    wss.clients.delete(ws);
  });
};

// Business Intelligence real-time updates
businessIntelligenceService.addWebSocketClient = (ws: any) => {
  wss.clients.add(ws);
  ws.on('close', () => {
    wss.clients.delete(ws);
  });
};

// Security real-time updates
securityService.addWebSocketClient = (ws: any) => {
  wss.clients.add(ws);
  ws.on('close', () => {
    wss.clients.delete(ws);
  });
};

// High Availability real-time updates
highAvailabilityService.addWebSocketClient = (ws: any) => {
  wss.clients.add(ws);
  ws.on('close', () => {
    wss.clients.delete(ws);
  });
};

// Error handling middleware
app.use((error: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
  logger.error('Unhandled error', {
    error: error.message,
    stack: error.stack,
    url: req.url,
    method: req.method,
    ip: req.ip,
    userAgent: req.get('User-Agent')
  });

  res.status(500).json({
    success: false,
    error: process.env.NODE_ENV === 'production' 
      ? 'Internal server error' 
      : error.message,
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    success: false,
    error: 'Route not found',
    path: req.originalUrl,
    method: req.method,
    timestamp: new Date().toISOString()
  });
});

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

// Start server
const PORT = process.env.PORT || 8000;
const HOST = process.env.HOST || '0.0.0.0';

server.listen(PORT, HOST, () => {
  logger.info(`KERNELIZE Backend API Server running on http://${HOST}:${PORT}`);
  logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
  logger.info(`API Documentation: http://${HOST}:${PORT}/api/v1`);
  logger.info(`Health Check: http://${HOST}:${PORT}/health`);
  logger.info(`WebSocket Server: ws://${HOST}:${PORT}`);
});

// Export services for testing
export {
  pipelineService,
  cloudStorageService,
  serverlessDeploymentService,
  cdnService,
  analyticsService,
  businessIntelligenceService,
  securityService,
  highAvailabilityService,
  integrationRoutes,
  compressionService,
  dataValidator,
  schemaManager,
  dataReader,
  dataWriter,
  dataTransformer,
  batchProcessor,
  etlEngine
};

export default app;