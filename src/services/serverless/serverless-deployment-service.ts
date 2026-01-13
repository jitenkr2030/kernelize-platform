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

export class ServerlessDeploymentService extends EventEmitter {
  private logger: Logger;
  private platforms: Map<string, ServerlessPlatform> = new Map();
  private deployments: Map<string, Deployment> = new Map();
  private functions: Map<string, ServerlessFunction> = new Map();

  constructor(logger: Logger) {
    super();
    this.logger = logger;
    this.initializePlatforms();
  }

  /**
   * Register serverless platform
   */
  registerPlatform(name: string, platform: ServerlessPlatform): void {
    this.platforms.set(name, platform);
    this.logger.info('Serverless platform registered', { platform: name });
  }

  /**
   * Deploy serverless function
   */
  async deployFunction(params: DeployFunctionParams): Promise<DeploymentResult> {
    const { 
      platform, 
      functionName, 
      code, 
      runtime, 
      handler, 
      memorySize = 256,
      timeout = 30,
      environment = {},
      layers = [],
      tags = {}
    } = params;

    this.logger.info('Deploying serverless function', {
      platform,
      functionName,
      runtime,
      memorySize,
      timeout
    });

    const platformInstance = this.platforms.get(platform);
    if (!platformInstance) {
      throw new Error(`Platform not registered: ${platform}`);
    }

    try {
      // Create function configuration
      const functionConfig: ServerlessFunction = {
        name: functionName,
        platform,
        runtime,
        handler,
        memorySize,
        timeout,
        environment,
        layers,
        tags,
        code,
        createdAt: new Date(),
        updatedAt: new Date()
      };

      // Deploy function
      const deploymentResult = await platformInstance.deployFunction(functionConfig);

      // Store deployment info
      const deployment: Deployment = {
        id: deploymentResult.deploymentId,
        functionName,
        platform,
        status: 'deployed',
        createdAt: new Date(),
        version: deploymentResult.version,
        arn: deploymentResult.arn,
        url: deploymentResult.url,
        logs: []
      };

      this.deployments.set(deployment.id, deployment);
      this.functions.set(functionName, functionConfig);

      this.emit('function_deployed', {
        deploymentId: deployment.id,
        functionName,
        platform,
        arn: deployment.arn,
        url: deployment.url
      });

      this.logger.info('Function deployed successfully', {
        deploymentId: deployment.id,
        functionName,
        platform,
        arn: deployment.arn,
        url: deployment.url
      });

      return {
        deploymentId: deployment.id,
        functionName,
        platform,
        status: 'deployed',
        arn: deployment.arn,
        url: deployment.url,
        version: deployment.version,
        logs: deployment.logs
      };

    } catch (error) {
      this.logger.error('Function deployment failed', {
        platform,
        functionName,
        error: error as Error
      });

      this.emit('deployment_error', {
        functionName,
        platform,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Update serverless function
   */
  async updateFunction(params: UpdateFunctionParams): Promise<UpdateResult> {
    const { deploymentId, functionName, code, runtime, handler, memorySize, timeout, environment } = params;
    
    this.logger.info('Updating serverless function', {
      deploymentId,
      functionName
    });

    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    const platformInstance = this.platforms.get(deployment.platform);
    if (!platformInstance) {
      throw new Error(`Platform not registered: ${deployment.platform}`);
    }

    try {
      const existingFunction = this.functions.get(functionName);
      if (!existingFunction) {
        throw new Error(`Function not found: ${functionName}`);
      }

      // Update function configuration
      const updatedFunction: ServerlessFunction = {
        ...existingFunction,
        runtime: runtime || existingFunction.runtime,
        handler: handler || existingFunction.handler,
        memorySize: memorySize || existingFunction.memorySize,
        timeout: timeout || existingFunction.timeout,
        environment: environment || existingFunction.environment,
        code: code || existingFunction.code,
        updatedAt: new Date()
      };

      const updateResult = await platformInstance.updateFunction(deploymentId, updatedFunction);

      // Update stored function
      this.functions.set(functionName, updatedFunction);
      
      // Update deployment
      deployment.version = updateResult.version;
      deployment.updatedAt = new Date();

      this.emit('function_updated', {
        deploymentId,
        functionName,
        platform: deployment.platform,
        version: updateResult.version
      });

      this.logger.info('Function updated successfully', {
        deploymentId,
        functionName,
        version: updateResult.version
      });

      return {
        deploymentId,
        functionName,
        status: 'updated',
        version: updateResult.version,
        previousVersion: updateResult.previousVersion
      };

    } catch (error) {
      this.logger.error('Function update failed', {
        deploymentId,
        functionName,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Invoke serverless function
   */
  async invokeFunction(params: InvokeParams): Promise<InvocationResult> {
    const { deploymentId, payload, async = false } = params;
    
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    this.logger.info('Invoking serverless function', {
      deploymentId,
      functionName: deployment.functionName,
      async
    });

    const platformInstance = this.platforms.get(deployment.platform);
    if (!platformInstance) {
      throw new Error(`Platform not registered: ${deployment.platform}`);
    }

    try {
      const result = await platformInstance.invokeFunction(deploymentId, {
        payload,
        async
      });

      this.emit('function_invoked', {
        deploymentId,
        functionName: deployment.functionName,
        requestId: result.requestId,
        async
      });

      this.logger.info('Function invoked successfully', {
        deploymentId,
        functionName: deployment.functionName,
        requestId: result.requestId,
        async
      });

      return result;

    } catch (error) {
      this.logger.error('Function invocation failed', {
        deploymentId,
        functionName: deployment.functionName,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Delete serverless function
   */
  async deleteFunction(params: DeleteParams): Promise<void> {
    const { deploymentId } = params;
    
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    this.logger.info('Deleting serverless function', {
      deploymentId,
      functionName: deployment.functionName
    });

    const platformInstance = this.platforms.get(deployment.platform);
    if (!platformInstance) {
      throw new Error(`Platform not registered: ${deployment.platform}`);
    }

    try {
      await platformInstance.deleteFunction(deploymentId);

      // Clean up stored data
      this.deployments.delete(deploymentId);
      this.functions.delete(deployment.functionName);

      this.emit('function_deleted', {
        deploymentId,
        functionName: deployment.functionName,
        platform: deployment.platform
      });

      this.logger.info('Function deleted successfully', {
        deploymentId,
        functionName: deployment.functionName
      });

    } catch (error) {
      this.logger.error('Function deletion failed', {
        deploymentId,
        functionName: deployment.functionName,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Get function logs
   */
  async getFunctionLogs(params: LogsParams): Promise<FunctionLog[]> {
    const { deploymentId, startTime, endTime, limit = 100 } = params;
    
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    const platformInstance = this.platforms.get(deployment.platform);
    if (!platformInstance) {
      throw new Error(`Platform not registered: ${deployment.platform}`);
    }

    try {
      const logs = await platformInstance.getLogs(deploymentId, {
        startTime,
        endTime,
        limit
      });

      return logs;

    } catch (error) {
      this.logger.error('Failed to get function logs', {
        deploymentId,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Get function metrics
   */
  async getFunctionMetrics(params: MetricsParams): Promise<FunctionMetrics> {
    const { deploymentId, startTime, endTime } = params;
    
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    const platformInstance = this.platforms.get(deployment.platform);
    if (!platformInstance) {
      throw new Error(`Platform not registered: ${deployment.platform}`);
    }

    try {
      const metrics = await platformInstance.getMetrics(deploymentId, {
        startTime,
        endTime
      });

      return metrics;

    } catch (error) {
      this.logger.error('Failed to get function metrics', {
        deploymentId,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * List deployments
   */
  getDeployments(): Deployment[] {
    return Array.from(this.deployments.values());
  }

  /**
   * Get deployment by ID
   */
  getDeployment(deploymentId: string): Deployment | null {
    return this.deployments.get(deploymentId) || null;
  }

  /**
   * List functions by platform
   */
  getFunctionsByPlatform(platform: string): ServerlessFunction[] {
    return Array.from(this.functions.values()).filter(func => func.platform === platform);
  }

  /**
   * Create API Gateway for function
   */
  async createApiGateway(params: ApiGatewayParams): Promise<ApiGatewayResult> {
    const { deploymentId, paths, methods, authentication } = params;
    
    const deployment = this.deployments.get(deploymentId);
    if (!deployment) {
      throw new Error(`Deployment not found: ${deploymentId}`);
    }

    const platformInstance = this.platforms.get(deployment.platform);
    if (!platformInstance) {
      throw new Error(`Platform not registered: ${deployment.platform}`);
    }

    try {
      const result = await platformInstance.createApiGateway(deploymentId, {
        paths,
        methods,
        authentication
      });

      this.emit('api_gateway_created', {
        deploymentId,
        apiId: result.apiId,
        url: result.url
      });

      this.logger.info('API Gateway created successfully', {
        deploymentId,
        apiId: result.apiId,
        url: result.url
      });

      return result;

    } catch (error) {
      this.logger.error('API Gateway creation failed', {
        deploymentId,
        error: error as Error
      });

      throw error;
    }
  }

  /**
   * Deploy to multiple environments
   */
  async deployToEnvironments(params: MultiEnvironmentParams): Promise<MultiEnvironmentResult> {
    const { functionName, code, environments, runtime, handler } = params;
    
    this.logger.info('Deploying to multiple environments', {
      functionName,
      environmentCount: environments.length
    });

    const results: EnvironmentDeployment[] = [];
    const errors: DeploymentError[] = [];

    for (const env of environments) {
      try {
        const deployParams: DeployFunctionParams = {
          platform: env.platform,
          functionName: `${functionName}-${env.name}`,
          code,
          runtime,
          handler,
          memorySize: env.memorySize,
          timeout: env.timeout,
          environment: { ...env.environment, ENVIRONMENT: env.name },
          layers: env.layers,
          tags: { ...env.tags, environment: env.name }
        };

        const result = await this.deployFunction(deployParams);
        
        results.push({
          environment: env.name,
          platform: env.platform,
          deploymentId: result.deploymentId,
          status: 'deployed',
          url: result.url
        });

      } catch (error) {
        errors.push({
          environment: env.name,
          platform: env.platform,
          error: error as Error
        });

        this.logger.error('Multi-environment deployment failed for environment', {
          environment: env.name,
          platform: env.platform,
          error: error as Error
        });
      }
    }

    return {
      functionName,
      totalEnvironments: environments.length,
      successfulDeployments: results.length,
      failedDeployments: errors.length,
      results,
      errors
    };
  }

  /**
   * Get deployment statistics
   */
  getStatistics(): ServerlessStatistics {
    const platforms = new Set(this.functions.values().map(f => f.platform));
    const totalFunctions = this.functions.size;
    const totalDeployments = this.deployments.size;
    
    const platformStats = new Map<string, number>();
    for (const platform of platforms) {
      platformStats.set(platform, this.getFunctionsByPlatform(platform).length);
    }

    return {
      totalFunctions,
      totalDeployments,
      totalPlatforms: platforms.size,
      platformStats: Object.fromEntries(platformStats)
    };
  }

  private initializePlatforms(): void {
    // Register built-in platforms
    // this.registerPlatform('aws_lambda', new AWSLambdaPlatform());
    // this.registerPlatform('azure_functions', new AzureFunctionsPlatform());
    // this.registerPlatform('google_cloud_functions', new GoogleCloudFunctionsPlatform());
    // this.registerPlatform('vercel', new VercelPlatform());
    // this.registerPlatform('netlify', new NetlifyPlatform());
    
    this.logger.info('Serverless platforms initialized');
  }
}

// Serverless Platform Interface
export interface ServerlessPlatform {
  deployFunction(functionConfig: ServerlessFunction): Promise<DeploymentResult>;
  updateFunction(deploymentId: string, functionConfig: ServerlessFunction): Promise<UpdateResult>;
  invokeFunction(deploymentId: string, params: InvocationParams): Promise<InvocationResult>;
  deleteFunction(deploymentId: string): Promise<void>;
  getLogs(deploymentId: string, params: LogsParams): Promise<FunctionLog[]>;
  getMetrics(deploymentId: string, params: MetricsParams): Promise<FunctionMetrics>;
  createApiGateway(deploymentId: string, params: ApiGatewayCreationParams): Promise<ApiGatewayResult>;
}

// Core Types
export interface ServerlessFunction {
  name: string;
  platform: string;
  runtime: string;
  handler: string;
  memorySize: number;
  timeout: number;
  environment: Record<string, string>;
  layers: string[];
  tags: Record<string, string>;
  code: string | Buffer;
  createdAt: Date;
  updatedAt: Date;
}

export interface Deployment {
  id: string;
  functionName: string;
  platform: string;
  status: 'deploying' | 'deployed' | 'updating' | 'failed' | 'deleted';
  createdAt: Date;
  updatedAt?: Date;
  version?: string;
  arn?: string;
  url?: string;
  logs: DeploymentLog[];
}

export interface DeploymentLog {
  timestamp: Date;
  level: 'info' | 'warn' | 'error';
  message: string;
  metadata?: any;
}

// Parameter Types
export interface DeployFunctionParams {
  platform: string;
  functionName: string;
  code: string | Buffer;
  runtime: string;
  handler: string;
  memorySize?: number;
  timeout?: number;
  environment?: Record<string, string>;
  layers?: string[];
  tags?: Record<string, string>;
}

export interface UpdateFunctionParams {
  deploymentId: string;
  functionName: string;
  code?: string | Buffer;
  runtime?: string;
  handler?: string;
  memorySize?: number;
  timeout?: number;
  environment?: Record<string, string>;
}

export interface InvokeParams {
  deploymentId: string;
  payload?: any;
  async?: boolean;
}

export interface DeleteParams {
  deploymentId: string;
}

export interface LogsParams {
  deploymentId: string;
  startTime?: Date;
  endTime?: Date;
  limit?: number;
}

export interface MetricsParams {
  deploymentId: string;
  startTime: Date;
  endTime: Date;
}

export interface ApiGatewayParams {
  deploymentId: string;
  paths: ApiPath[];
  methods: string[];
  authentication?: ApiAuthentication;
}

export interface MultiEnvironmentParams {
  functionName: string;
  code: string | Buffer;
  runtime: string;
  handler: string;
  environments: DeploymentEnvironment[];
}

export interface DeploymentEnvironment {
  name: string;
  platform: string;
  memorySize?: number;
  timeout?: number;
  environment?: Record<string, string>;
  layers?: string[];
  tags?: Record<string, string>;
}

export interface ApiPath {
  path: string;
  methods: string[];
  auth?: boolean;
}

export interface ApiAuthentication {
  type: 'none' | 'api_key' | 'jwt' | 'oauth';
  config?: any;
}

// Result Types
export interface DeploymentResult {
  deploymentId: string;
  version: string;
  arn: string;
  url: string;
}

export interface UpdateResult {
  version: string;
  previousVersion: string;
}

export interface InvocationResult {
  requestId: string;
  statusCode: number;
  response?: any;
  logs?: string[];
  executionTime: number;
  billedDuration?: number;
}

export interface FunctionLog {
  timestamp: Date;
  level: 'info' | 'warn' | 'error';
  message: string;
  requestId?: string;
  metadata?: any;
}

export interface FunctionMetrics {
  invocations: number;
  errors: number;
  averageDuration: number;
  averageMemory: number;
  coldStarts: number;
  cost?: number;
}

export interface ApiGatewayResult {
  apiId: string;
  url: string;
  paths: string[];
  methods: string[];
}

export interface MultiEnvironmentResult {
  functionName: string;
  totalEnvironments: number;
  successfulDeployments: number;
  failedDeployments: number;
  results: EnvironmentDeployment[];
  errors: DeploymentError[];
}

export interface EnvironmentDeployment {
  environment: string;
  platform: string;
  deploymentId: string;
  status: 'deployed' | 'failed';
  url?: string;
}

export interface DeploymentError {
  environment: string;
  platform: string;
  error: Error;
}

export interface ServerlessStatistics {
  totalFunctions: number;
  totalDeployments: number;
  totalPlatforms: number;
  platformStats: Record<string, number>;
}

// Platform-specific parameter types
export interface InvocationParams {
  payload?: any;
  async: boolean;
}

export interface LogsParams {
  startTime?: Date;
  endTime?: Date;
  limit: number;
}

export interface MetricsParams {
  startTime: Date;
  endTime: Date;
}

export interface ApiGatewayCreationParams {
  paths: ApiPath[];
  methods: string[];
  authentication?: ApiAuthentication;
}

export interface ApiAuthentication {
  type: 'none' | 'api_key' | 'jwt' | 'oauth';
  config?: any;
}