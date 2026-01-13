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

// Data Pipeline Types
export interface DataSource {
  type: 'file' | 'database' | 'api' | 'cloud_storage' | 'stream';
  path?: string;
  connection?: DatabaseConnection | APIConnection;
  query?: string;
  table?: string;
  url?: string;
  method?: string;
  headers?: Record<string, string>;
  params?: Record<string, any>;
  provider?: string;
  bucket?: string;
  key?: string;
  region?: string;
  stream?: any;
  format?: string;
}

export interface DataTarget {
  type: 'file' | 'database' | 'api' | 'cloud_storage' | 'stream';
  path?: string;
  connection?: DatabaseConnection | APIConnection;
  table?: string;
  url?: string;
  method?: string;
  headers?: Record<string, string>;
  provider?: string;
  bucket?: string;
  key?: string;
  region?: string;
  mode?: 'insert' | 'upsert' | 'update' | 'append';
  stream?: any;
  format?: string;
}

export interface PipelineConfig {
  id: string;
  name: string;
  description?: string;
  source: DataSource;
  target: DataTarget;
  transformations: Transformation[];
  validation?: ValidationConfig;
  schedule?: ScheduleConfig;
  triggers: TriggerType[];
  schema?: string;
  metadata?: Record<string, any>;
}

export interface Transformation {
  name: string;
  type: string;
  parameters?: Record<string, any>;
  condition?: string;
  enabled: boolean;
}

export interface ValidationConfig {
  enabled: boolean;
  rules?: ValidationRule[];
  schema?: string;
  threshold?: number;
  strict?: boolean;
}

export interface ValidationRule {
  field: string;
  type: 'required' | 'type' | 'range' | 'pattern' | 'length' | 'email' | 'phone' | 'date' | 'custom' | 'schema_type';
  value?: any;
  min?: number;
  max?: number;
  pattern?: string;
  message?: string;
  validator?: string | ((value: any) => boolean);
}

export interface ScheduleConfig {
  type: 'cron' | 'interval' | 'manual';
  expression?: string;
  interval?: number;
  timezone?: string;
  enabled: boolean;
}

export type TriggerType = 'manual' | 'webhook' | 'schedule' | 'file_change' | 'api_call';

export interface ETLResult {
  jobId: string;
  pipelineId: string;
  status: 'completed' | 'failed' | 'cancelled';
  inputRecordCount: number;
  outputRecordCount: number;
  validationResult: ValidationResult;
  writeResult: WriteResult;
  compressionResult?: CompressionResult;
  executionTime: number;
  metadata: Record<string, any>;
}

export interface ValidationResult {
  isValid: boolean;
  totalRecords: number;
  validRecords: number;
  invalidRecords: number;
  errors: ValidationError[];
  warnings: ValidationWarning[];
  statistics: Record<string, FieldStatistics>;
}

export interface ValidationError {
  field: string;
  rule: string;
  value: any;
  message: string;
  severity: 'error' | 'warning';
  recordIndex?: number;
}

export interface ValidationWarning {
  field: string;
  message: string;
  severity: 'warning';
  recordIndex?: number;
}

export interface FieldStatistics {
  totalCount: number;
  nullCount: number;
  uniqueCount: number;
  minValue: any;
  maxValue: any;
  averageValue: any;
  validCount: number;
  invalidCount: number;
  nullPercentage: number;
  validPercentage: number;
  invalidPercentage: number;
}

export interface WriteResult {
  outputPath: string;
  format: string;
  recordCount: number;
  fileSize?: number;
  batchCount?: number;
  results?: any[];
}

export interface CompressionResult {
  algorithm: string;
  level: number;
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  outputPath: string;
  executionTime: number;
}

// Schema Management Types
export interface SchemaDefinition {
  fields: SchemaField[];
  constraints: SchemaConstraint[];
  metadata?: SchemaMetadata;
}

export interface SchemaField {
  name: string;
  type: 'string' | 'number' | 'boolean' | 'date' | 'array' | 'object';
  nullable: boolean;
  default?: any;
  description?: string;
  constraints?: FieldConstraint[];
  validation?: ValidationRule[];
}

export interface SchemaConstraint {
  type: 'unique' | 'foreign_key' | 'check' | 'not_null' | 'primary_key';
  field: string;
  value?: any;
  referencedTable?: string;
  referencedField?: string;
  checkExpression?: string;
}

export interface FieldConstraint {
  type: string;
  value?: any;
  message?: string;
}

export interface SchemaMetadata {
  version: string;
  createdAt: Date;
  updatedAt: Date;
  createdBy?: string;
  description?: string;
  tags?: string[];
  lineage?: SchemaLineage;
}

export interface SchemaLineage {
  source: string;
  transformations: string[];
  dependencies: string[];
}

export interface SchemaVersion {
  version: string;
  schema: SchemaDefinition;
  createdAt: Date;
  createdBy: string;
  description: string;
  changes: SchemaChange[];
}

export interface SchemaChange {
  type: 'field_added' | 'field_removed' | 'field_modified' | 'constraint_added' | 'constraint_removed';
  field?: string;
  oldValue?: any;
  newValue?: any;
  description: string;
}

// Database Connection Types
export interface DatabaseConnection {
  type: 'mysql' | 'postgresql' | 'oracle' | 'sqlserver' | 'mongodb' | 'redis' | 'elasticsearch';
  host: string;
  port: number;
  database: string;
  username: string;
  password: string;
  ssl?: boolean;
  poolSize?: number;
  timeout?: number;
  options?: Record<string, any>;
}

export interface APIConnection {
  baseUrl: string;
  authentication: APIAuthentication;
  headers?: Record<string, string>;
  timeout?: number;
  retry?: number;
  rateLimit?: RateLimit;
}

export interface APIAuthentication {
  type: 'none' | 'basic' | 'bearer' | 'oauth2' | 'apikey';
  credentials?: {
    username?: string;
    password?: string;
    token?: string;
    apiKey?: string;
    clientId?: string;
    clientSecret?: string;
    accessToken?: string;
    refreshToken?: string;
  };
}

export interface RateLimit {
  requests: number;
  period: 'second' | 'minute' | 'hour' | 'day';
  burst?: number;
}

// Data Quality Types
export interface DataQualityReport {
  overallScore: number;
  dimensions: QualityDimension[];
  issues: QualityIssue[];
  recommendations: Recommendation[];
  metadata: QualityMetadata;
}

export interface QualityDimension {
  name: string;
  score: number;
  weight: number;
  description: string;
  metrics: QualityMetric[];
}

export interface QualityMetric {
  name: string;
  value: number;
  threshold: number;
  status: 'pass' | 'warning' | 'fail';
  description: string;
}

export interface QualityIssue {
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: string;
  description: string;
  affectedRecords: number;
  fields: string[];
  recommendation: string;
}

export interface Recommendation {
  priority: 'high' | 'medium' | 'low';
  category: string;
  title: string;
  description: string;
  impact: string;
  effort: 'low' | 'medium' | 'high';
}

export interface QualityMetadata {
  profileId: string;
  datasetName: string;
  generatedAt: Date;
  profileVersion: string;
  totalRecords: number;
  fields: string[];
}

// Data Lineage Types
export interface DataLineage {
  datasetId: string;
  sources: LineageSource[];
  transformations: LineageTransformation[];
  targets: LineageTarget[];
  metadata: LineageMetadata;
}

export interface LineageSource {
  datasetId: string;
  sourceType: 'table' | 'file' | 'api' | 'stream';
  connection: string;
  lastAccessed: Date;
  frequency: string;
}

export interface LineageTransformation {
  transformationId: string;
  type: string;
  description: string;
  inputs: string[];
  outputs: string[];
  createdAt: Date;
  createdBy: string;
}

export interface LineageTarget {
  datasetId: string;
  targetType: 'table' | 'file' | 'api' | 'stream';
  connection: string;
  lastUpdated: Date;
  frequency: string;
}

export interface LineageMetadata {
  createdAt: Date;
  updatedAt: Date;
  version: string;
  createdBy: string;
  tags: string[];
}

// Data Catalog Types
export interface DataCatalog {
  id: string;
  name: string;
  description: string;
  datasets: CatalogDataset[];
  schemas: CatalogSchema[];
  lineage: DataLineage[];
  quality: DataQualityReport[];
  metadata: CatalogMetadata;
}

export interface CatalogDataset {
  id: string;
  name: string;
  description: string;
  datasetType: 'table' | 'file' | 'api' | 'stream';
  schema: string;
  source: DataSource;
  tags: string[];
  owners: string[];
  stakeholders: string[];
  sensitivityLevel: 'public' | 'internal' | 'confidential' | 'restricted';
  retentionPeriod?: number;
  lastModified: Date;
  createdAt: Date;
  metadata: DatasetMetadata;
}

export interface CatalogSchema {
  id: string;
  name: string;
  description: string;
  definition: SchemaDefinition;
  versions: SchemaVersion[];
  usage: SchemaUsage[];
  createdAt: Date;
  updatedAt: Date;
}

export interface SchemaUsage {
  datasetId: string;
  firstUsed: Date;
  lastUsed: Date;
  usageCount: number;
}

export interface DatasetMetadata {
  rowCount?: number;
  columnCount?: number;
  fileSize?: number;
  format: string;
  compression?: string;
  partitionColumns?: string[];
  sampleData?: any[];
  statistics?: DatasetStatistics;
}

export interface DatasetStatistics {
  completeness: number;
  uniqueness: number;
  validity: number;
  consistency: number;
  timeliness: number;
  accuracy: number;
}

export interface CatalogMetadata {
  version: string;
  createdAt: Date;
  updatedAt: Date;
  totalDatasets: number;
  totalSchemas: number;
  totalLineageRecords: number;
}

// Integration Types
export interface IntegrationConfig {
  id: string;
  name: string;
  type: 'database' | 'api' | 'file' | 'cloud' | 'streaming';
  provider: string;
  connection: DatabaseConnection | APIConnection | FileConnection | CloudConnection;
  settings: IntegrationSettings;
  schedule?: ScheduleConfig;
  enabled: boolean;
  metadata: IntegrationMetadata;
}

export interface FileConnection {
  type: 'local' | 'ftp' | 'sftp' | 's3' | 'gcs' | 'azure_blob';
  path: string;
  format: 'csv' | 'json' | 'parquet' | 'avro' | 'excel';
  encoding?: string;
  delimiter?: string;
  hasHeaders?: boolean;
  credentials?: Record<string, any>;
}

export interface CloudConnection {
  provider: 'aws' | 'gcp' | 'azure';
  region: string;
  bucket: string;
  credentials: CloudCredentials;
  settings: CloudSettings;
}

export interface CloudCredentials {
  accessKeyId?: string;
  secretAccessKey?: string;
  serviceAccountKey?: string;
  connectionString?: string;
  sasToken?: string;
}

export interface CloudSettings {
  serverSideEncryption?: boolean;
  storageClass?: string;
  lifecycle?: LifecycleRule[];
}

export interface LifecycleRule {
  id: string;
  status: 'Enabled' | 'Disabled';
  filter?: LifecycleFilter;
  expiration?: LifecycleExpiration;
  transition?: LifecycleTransition[];
}

export interface LifecycleFilter {
  prefix?: string;
  tags?: Record<string, string>;
}

export interface LifecycleExpiration {
  days?: number;
  date?: Date;
}

export interface LifecycleTransition {
  days?: number;
  date?: Date;
  storageClass: string;
}

export interface IntegrationSettings {
  batchSize?: number;
  timeout?: number;
  retryAttempts?: number;
  retryDelay?: number;
  compression?: boolean;
  encryption?: boolean;
  monitoring?: MonitoringConfig;
}

export interface MonitoringConfig {
  enabled: boolean;
  alerts: AlertConfig[];
  metrics: MetricConfig[];
  logging: LoggingConfig;
}

export interface AlertConfig {
  type: 'email' | 'webhook' | 'slack';
  recipients: string[];
  threshold: number;
  condition: 'greater_than' | 'less_than' | 'equals';
  enabled: boolean;
}

export interface MetricConfig {
  name: string;
  type: 'counter' | 'gauge' | 'histogram';
  labels?: string[];
  enabled: boolean;
}

export interface IntegrationMetadata {
  createdAt: Date;
  updatedAt: Date;
  createdBy: string;
  lastRun?: Date;
  totalRuns: number;
  successfulRuns: number;
  failedRuns: number;
  tags: string[];
}

// Performance Types
export interface PerformanceMetrics {
  throughput: ThroughputMetrics;
  latency: LatencyMetrics;
  resource: ResourceMetrics;
  reliability: ReliabilityMetrics;
}

export interface ThroughputMetrics {
  recordsPerSecond: number;
  bytesPerSecond: number;
  filesPerSecond: number;
  operationsPerSecond: number;
  peakThroughput: number;
  averageThroughput: number;
}

export interface LatencyMetrics {
  averageLatency: number;
  p50Latency: number;
  p95Latency: number;
  p99Latency: number;
  maxLatency: number;
  minLatency: number;
}

export interface ResourceMetrics {
  cpuUsage: number;
  memoryUsage: number;
  diskUsage: number;
  networkIO: NetworkIOMetrics;
}

export interface NetworkIOMetrics {
  bytesRead: number;
  bytesWritten: number;
  packetsRead: number;
  packetsWritten: number;
}

export interface ReliabilityMetrics {
  uptime: number;
  availability: number;
  mttr: number; // Mean Time To Recovery
  mtbf: number; // Mean Time Between Failures
  errorRate: number;
  successRate: number;
}

// Monitoring Types
export interface MonitoringConfig {
  enabled: boolean;
  metrics: MonitoringMetric[];
  alerts: MonitoringAlert[];
  dashboards: MonitoringDashboard[];
  retention: MonitoringRetention;
}

export interface MonitoringMetric {
  name: string;
  type: 'counter' | 'gauge' | 'histogram' | 'summary';
  labels?: string[];
  description: string;
  unit?: string;
  enabled: boolean;
}

export interface MonitoringAlert {
  name: string;
  condition: AlertCondition;
  notification: NotificationConfig;
  enabled: boolean;
  description: string;
}

export interface AlertCondition {
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'ne' | 'gte' | 'lte';
  threshold: number;
  duration?: number;
}

export interface NotificationConfig {
  type: 'email' | 'webhook' | 'slack' | 'pagerduty';
  recipients: string[];
  template: string;
  severity: 'critical' | 'warning' | 'info';
}

export interface MonitoringDashboard {
  id: string;
  name: string;
  description: string;
  widgets: DashboardWidget[];
  filters: DashboardFilter[];
  refreshInterval: number;
  layout: DashboardLayout;
}

export interface DashboardWidget {
  id: string;
  type: 'chart' | 'metric' | 'table' | 'log';
  title: string;
  position: WidgetPosition;
  config: WidgetConfig;
}

export interface WidgetPosition {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface WidgetConfig {
  metric: string;
  chartType: 'line' | 'bar' | 'pie' | 'heatmap';
  timeRange: string;
  aggregation?: string;
  filters?: Record<string, any>;
}

export interface DashboardFilter {
  name: string;
  type: 'select' | 'range' | 'date' | 'multiselect';
  options?: any[];
  defaultValue?: any;
  enabled: boolean;
}

export interface DashboardLayout {
  columns: number;
  rows: number;
  gap: number;
  responsive: boolean;
}

export interface MonitoringRetention {
  metrics: number; // days
  logs: number; // days
  alerts: number; // days
  dashboards: number; // days
}

// Common Utility Types
export interface PaginatedResult<T> {
  items: T[];
  total: number;
  page: number;
  limit: number;
  hasNext: boolean;
  hasPrevious: boolean;
}

export interface SortConfig {
  field: string;
  direction: 'asc' | 'desc';
}

export interface FilterConfig {
  field: string;
  operator: 'eq' | 'ne' | 'gt' | 'gte' | 'lt' | 'lte' | 'in' | 'nin' | 'like' | 'between';
  value: any;
}

export interface SearchResult<T> {
  items: T[];
  total: number;
  took: number;
  highlights?: Record<string, string[]>;
  facets?: Record<string, FacetResult>;
}

export interface FacetResult {
  field: string;
  values: FacetValue[];
}

export interface FacetValue {
  value: string;
  count: number;
  selected?: boolean;
}

// Error Types
export interface ServiceError {
  code: string;
  message: string;
  details?: any;
  timestamp: Date;
  requestId?: string;
  stack?: string;
}

export interface ValidationErrorDetail {
  field: string;
  code: string;
  message: string;
  value?: any;
}

// Configuration Types
export interface ServiceConfig {
  dataPipeline: DataPipelineConfig;
  cloud: CloudConfig;
  serverless: ServerlessConfig;
  cdn: CDNConfig;
  monitoring: MonitoringConfig;
  logging: LoggingConfig;
  security: SecurityConfig;
}

export interface DataPipelineConfig {
  maxConcurrentJobs: number;
  defaultBatchSize: number;
  defaultTimeout: number;
  enableValidation: boolean;
  enableCompression: boolean;
  tempDirectory: string;
}

export interface CloudConfig {
  defaultRegion: string;
  credentials: Record<string, any>;
  endpoints: Record<string, string>;
  retryConfig: RetryConfig;
}

export interface RetryConfig {
  maxAttempts: number;
  backoffMultiplier: number;
  maxBackoff: number;
  initialDelay: number;
}

export interface ServerlessConfig {
  defaultRuntime: string;
  defaultMemorySize: number;
  defaultTimeout: number;
  maxMemorySize: number;
  maxTimeout: number;
  supportedRuntimes: string[];
}

export interface CDNConfig {
  defaultProvider: string;
  defaultTTL: number;
  enableCompression: boolean;
  enableGzip: boolean;
  enableBrotli: boolean;
}

export interface SecurityConfig {
  encryptionEnabled: boolean;
  sslRequired: boolean;
  apiKeyRotation: number;
  accessTokenExpiry: number;
  refreshTokenExpiry: number;
}

export interface LoggingConfig {
  level: 'debug' | 'info' | 'warn' | 'error';
  format: 'json' | 'text';
  outputs: LogOutput[];
  retention: LogRetention;
}

export interface LogOutput {
  type: 'console' | 'file' | 'database' | 'elasticsearch' | 'cloudwatch';
  config: Record<string, any>;
  enabled: boolean;
}

export interface LogRetention {
  days: number;
  archiveEnabled: boolean;
  archiveLocation?: string;
}