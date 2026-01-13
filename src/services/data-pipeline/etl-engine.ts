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
import { DataSource, DataTarget, Transformation, ETLResult } from './types';
import { DataReader } from './data-reader';
import { DataWriter } from './data-writer';
import { DataTransformer } from './data-transformer';
import { CompressionService } from '../compression/compression-service';
import { BatchProcessor } from './batch-processor';

export class ETLEngine extends EventEmitter {
  private logger: Logger;
  private dataReader: DataReader;
  private dataWriter: DataWriter;
  private dataTransformer: DataTransformer;
  private compressionService: CompressionService;
  private batchProcessor: BatchProcessor;
  private activeJobs: Map<string, ETLJob> = new Map();

  constructor(
    logger: Logger,
    dataReader: DataReader,
    dataWriter: DataWriter,
    dataTransformer: DataTransformer,
    compressionService: CompressionService,
    batchProcessor: BatchProcessor
  ) {
    super();
    this.logger = logger;
    this.dataReader = dataReader;
    this.dataWriter = dataWriter;
    this.dataTransformer = dataTransformer;
    this.compressionService = compressionService;
    this.batchProcessor = batchProcessor;
  }

  /**
   * Execute ETL pipeline
   */
  async execute(pipelineId: string, options: any = {}): Promise<ETLResult> {
    this.logger.info('Starting ETL execution', { pipelineId, options });

    const job: ETLJob = {
      id: this.generateJobId(),
      pipelineId,
      status: 'running',
      startedAt: new Date(),
      options
    };

    this.activeJobs.set(job.id, job);

    try {
      // Emit progress event
      this.emit('progress', job.id, 0, 'Initializing');

      // Step 1: Read data from source
      this.emit('progress', job.id, 10, 'Reading data');
      const sourceData = await this.readSourceData(job);
      this.logger.info('Source data read completed', { 
        jobId: job.id, 
        recordCount: sourceData.length 
      });

      // Step 2: Apply transformations
      this.emit('progress', job.id, 30, 'Transforming data');
      const transformedData = await this.applyTransformations(sourceData, job);
      this.logger.info('Data transformation completed', { 
        jobId: job.id, 
        recordCount: transformedData.length 
      });

      // Step 3: Validate data
      this.emit('progress', job.id, 60, 'Validating data');
      const validationResult = await this.validateData(transformedData, job);
      
      if (!validationResult.isValid && !job.options.skipValidation) {
        throw new Error(`Data validation failed: ${validationResult.errors.join(', ')}`);
      }

      // Step 4: Write data to target
      this.emit('progress', job.id, 80, 'Writing data');
      const writeResult = await this.writeTargetData(transformedData, job);
      
      // Step 5: Generate compression (if enabled)
      let compressionResult = null;
      if (job.options.compressOutput) {
        this.emit('progress', job.id, 90, 'Compressing output');
        compressionResult = await this.compressOutput(writeResult.outputPath, job);
      }

      // Step 6: Complete job
      this.emit('progress', job.id, 100, 'Completed');
      job.status = 'completed';
      job.completedAt = new Date();

      const result: ETLResult = {
        jobId: job.id,
        pipelineId,
        status: 'completed',
        inputRecordCount: sourceData.length,
        outputRecordCount: transformedData.length,
        validationResult,
        writeResult,
        compressionResult,
        executionTime: job.completedAt.getTime() - job.startedAt.getTime(),
        metadata: {
          startedAt: job.startedAt,
          completedAt: job.completedAt,
          options: job.options
        }
      };

      this.emit('complete', job.id, result);
      return result;

    } catch (error) {
      job.status = 'failed';
      job.completedAt = new Date();
      job.error = error as Error;

      this.logger.error('ETL execution failed', { 
        jobId: job.id, 
        error: error as Error,
        stack: (error as Error).stack 
      });

      this.emit('error', job.id, error as Error);
      throw error;
    } finally {
      this.activeJobs.delete(job.id);
    }
  }

  /**
   * Cancel ETL job
   */
  async cancel(jobId: string): Promise<void> {
    const job = this.activeJobs.get(jobId);
    if (job) {
      job.status = 'cancelled';
      this.activeJobs.delete(jobId);
      this.logger.info('ETL job cancelled', { jobId });
    }
  }

  /**
   * Get job status
   */
  getJobStatus(jobId: string): ETLJob | null {
    return this.activeJobs.get(jobId) || null;
  }

  private async readSourceData(job: ETLJob): Promise<any[]> {
    const pipelineConfig = await this.getPipelineConfig(job.pipelineId);
    
    return await this.dataReader.read(
      pipelineConfig.source,
      job.options
    );
  }

  private async applyTransformations(data: any[], job: ETLJob): Promise<any[]> {
    const pipelineConfig = await this.getPipelineConfig(job.pipelineId);
    
    let transformedData = data;

    for (const transformation of pipelineConfig.transformations) {
      this.logger.debug('Applying transformation', { 
        jobId: job.id, 
        transformation: transformation.name 
      });

      transformedData = await this.dataTransformer.apply(
        transformedData,
        transformation,
        job.options
      );

      // Emit progress update
      const progress = 30 + (30 * (pipelineConfig.transformations.indexOf(transformation) + 1) / pipelineConfig.transformations.length);
      this.emit('progress', job.id, progress, `Applying ${transformation.name}`);
    }

    return transformedData;
  }

  private async validateData(data: any[], job: ETLJob): Promise<any> {
    const pipelineConfig = await this.getPipelineConfig(job.pipelineId);
    
    if (!pipelineConfig.validation?.enabled) {
      return { isValid: true, errors: [] };
    }

    // Implement data validation logic
    const errors = [];
    
    // Check required fields
    for (const rule of pipelineConfig.validation.rules || []) {
      const invalidRecords = data.filter(record => !this.isValidField(record, rule));
      if (invalidRecords.length > 0) {
        errors.push(`${rule.field}: ${invalidRecords.length} records failed validation`);
      }
    }

    return {
      isValid: errors.length === 0,
      errors,
      validRecordCount: data.length - errors.length,
      invalidRecordCount: errors.length
    };
  }

  private async writeTargetData(data: any[], job: ETLJob): Promise<any> {
    const pipelineConfig = await this.getPipelineConfig(job.pipelineId);
    
    const writeResult = await this.dataWriter.write(
      pipelineConfig.target,
      data,
      job.options
    );

    return writeResult;
  }

  private async compressOutput(outputPath: string, job: ETLJob): Promise<any> {
    const compressionConfig = {
      level: 6,
      algorithm: 'gzip',
      chunkSize: 1024 * 1024 // 1MB chunks
    };

    return await this.compressionService.compressFile(
      outputPath,
      `${outputPath}.gz`,
      compressionConfig
    );
  }

  private isValidField(record: any, rule: any): boolean {
    const value = record[rule.field];
    
    switch (rule.type) {
      case 'required':
        return value !== null && value !== undefined;
      
      case 'type':
        return typeof value === rule.value;
      
      case 'range':
        return value >= rule.min && value <= rule.max;
      
      case 'pattern':
        return new RegExp(rule.pattern).test(value);
      
      case 'custom':
        return rule.validator ? rule.validator(value) : true;
      
      default:
        return true;
    }
  }

  private async getPipelineConfig(pipelineId: string): Promise<any> {
    // This would fetch from a database or configuration store
    // For now, return a mock configuration
    return {
      id: pipelineId,
      source: { type: 'file', path: '/tmp/input.csv' },
      target: { type: 'file', path: '/tmp/output.csv' },
      transformations: [],
      validation: { enabled: false, rules: [] }
    };
  }

  private generateJobId(): string {
    return `etl_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

export interface ETLJob {
  id: string;
  pipelineId: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  startedAt: Date;
  completedAt?: Date;
  error?: Error;
  options: any;
}