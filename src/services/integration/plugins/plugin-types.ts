/**
 * KERNELIZE Platform - Plugin Architecture Core
 * Extensible framework for third-party integrations and custom algorithms
 */

export enum PluginCategory {
  DATA_PROCESSOR = 'data-processor',
  COMPRESSOR = 'compressor',
  CONNECTOR = 'connector',
  VISUALIZER = 'visualizer',
  ANALYTICS = 'analytics',
  WORKFLOW = 'workflow',
  SECURITY = 'security',
  MONITORING = 'monitoring',
  HEALTHCARE = 'healthcare',
  FINANCE = 'finance',
  MANUFACTURING = 'manufacturing',
  RETAIL = 'retail'
}

export enum PluginStatus {
  INSTALLED = 'installed',
  ACTIVE = 'active',
  INACTIVE = 'inactive',
  ERROR = 'error',
  UPDATING = 'updating'
}

export enum CompressionSpeed {
  FASTEST = 'fastest',
  FAST = 'fast',
  BALANCED = 'balanced',
  SMALLEST = 'smallest'
}

export interface PluginMetadata {
  id: string;
  name: string;
  version: string;
  description: string;
  author: string;
  category: PluginCategory;
  homepage?: string;
  repository?: string;
  keywords: string[];
  license: string;
  createdAt: Date;
  updatedAt: Date;
  downloads: number;
  rating: number;
  compatibility: {
    minPlatformVersion: string;
    supportedNodes: string[];
  };
}

export interface PluginConfig {
  [key: string]: any;
}

export interface Permission {
  resource: string;
  actions: string[];
  conditions?: Record<string, any>;
}

export interface ResourceLimits {
  maxMemory: number;
  maxCPU: number;
  maxExecutionTime: number;
  maxConcurrentExecutions: number;
}

export interface ValidationResult {
  valid: boolean;
  errors: ValidationError[];
  warnings: ValidationWarning[];
}

export interface ValidationError {
  field: string;
  code: string;
  message: string;
  details?: any;
}

export interface ValidationWarning {
  field: string;
  code: string;
  message: string;
  suggestion?: string;
}

export interface CompressionStats {
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  compressionTime: number;
  decompressionTime: number;
  memoryUsed: number;
}

export interface CompressedData {
  data: Buffer;
  metadata: {
    algorithm: string;
    version: string;
    originalSize: number;
    compressedSize: number;
    timestamp: Date;
    checksum: string;
  };
}

export interface PerformanceMetrics {
  responseTime: number;
  throughput: number;
  errorRate: number;
  memoryUsage: number;
  cpuUsage: number;
  activeConnections: number;
}

export interface ErrorStats {
  totalErrors: number;
  errorRate: number;
  lastError?: Error;
  errorTypes: Record<string, number>;
}

export interface PluginHealth {
  status: 'healthy' | 'degraded' | 'unhealthy';
  uptime: number;
  performance: PerformanceMetrics;
  errors: ErrorStats;
  lastCheck: Date;
  checks: HealthCheck[];
}

export interface HealthCheck {
  name: string;
  status: 'pass' | 'fail' | 'warning';
  message: string;
  duration: number;
  timestamp: Date;
}

export interface PluginSource {
  type: 'registry' | 'url' | 'file' | 'git';
  location: string;
  version?: string;
  checksum?: string;
  signature?: string;
}

export interface InstallationResult {
  success: boolean;
  pluginId: string;
  version: string;
  installedAt: Date;
  errors?: string[];
  warnings?: string[];
}

export interface PluginEvent {
  type: string;
  pluginId: string;
  timestamp: Date;
  data: any;
  source: string;
}

export interface WorkflowContext {
  workflowId: string;
  executionId: string;
  userId: string;
  variables: Record<string, any>;
  state: Record<string, any>;
  startedAt: Date;
}

export interface ExecutionResult {
  success: boolean;
  executionId: string;
  startedAt: Date;
  completedAt: Date;
  duration: number;
  output: any;
  errors?: ExecutionError[];
  logs: ExecutionLog[];
}

export interface ExecutionError {
  step: string;
  message: string;
  code: string;
  stack?: string;
  timestamp: Date;
}

export interface ExecutionLog {
  level: 'debug' | 'info' | 'warn' | 'error';
  message: string;
  timestamp: Date;
  metadata?: Record<string, any>;
}

export interface CronExpression {
  expression: string;
  timezone?: string;
  enabled: boolean;
  lastRun?: Date;
  nextRun?: Date;
}

export interface ConnectorConfig {
  baseURL?: string;
  authentication: {
    type: 'none' | 'basic' | 'bearer' | 'oauth2' | 'api-key';
    credentials: Record<string, any>;
  };
  timeout: number;
  retries: number;
  rateLimit: {
    requests: number;
    window: number; // milliseconds
  };
  headers: Record<string, string>;
}

export interface PluginRepository {
  id: string;
  name: string;
  url: string;
  type: 'official' | 'community' | 'enterprise';
  enabled: boolean;
  priority: number;
  lastSync: Date;
  plugins: PluginMetadata[];
}

export interface PluginUpdate {
  pluginId: string;
  currentVersion: string;
  availableVersion: string;
  releaseNotes: string;
  breaking: boolean;
  securityFix: boolean;
  autoUpdate: boolean;
}

export interface PluginDependency {
  pluginId: string;
  version: string;
  required: boolean;
  optional: boolean;
}

export interface PluginSignature {
  algorithm: string;
  signature: string;
  timestamp: Date;
  issuer: string;
}

export interface SandboxConfig {
  enabled: boolean;
  networkIsolation: boolean;
  fileSystemAccess: 'none' | 'read-only' | 'read-write';
  processIsolation: boolean;
  timeLimits: {
    execution: number;
    idle: number;
  };
  resourceLimits: {
    memory: number;
    cpu: number;
    disk: number;
  };
}