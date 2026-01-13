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

export class BatchProcessor extends EventEmitter {
  private logger: Logger;
  private activeBatches: Map<string, BatchJob> = new Map();
  private maxConcurrentBatches: number = 5;
  private defaultBatchSize: number = 1000;

  constructor(logger: Logger) {
    super();
    this.logger = logger;
  }

  /**
   * Process data in batches
   */
  async processBatch<T, R>(
    data: T[],
    processor: BatchProcessorFunction<T, R>,
    options: BatchProcessingOptions = {}
  ): Promise<BatchResult<R>> {
    const {
      batchSize = this.defaultBatchSize,
      maxConcurrent = this.maxConcurrentBatches,
      onProgress,
      onBatchComplete,
      stopOnError = false
    } = options;

    const jobId = this.generateJobId();
    const totalBatches = Math.ceil(data.length / batchSize);
    
    this.logger.info('Starting batch processing', { 
      jobId, 
      totalRecords: data.length,
      batchSize,
      totalBatches,
      maxConcurrent 
    });

    const job: BatchJob = {
      id: jobId,
      status: 'running',
      totalRecords: data.length,
      processedRecords: 0,
      successfulBatches: 0,
      failedBatches: 0,
      startedAt: new Date()
    };

    this.activeBatches.set(jobId, job);

    try {
      const batches = this.createBatches(data, batchSize);
      const results: R[] = [];
      const errors: BatchError[] = [];

      // Process batches with concurrency control
      const batchPromises: Promise<void>[] = [];
      const batchQueue: Promise<any>[] = [];

      for (let i = 0; i < batches.length; i++) {
        const batch = batches[i];
        const batchIndex = i;

        // Create batch processor
        const processBatch = async (): Promise<void> => {
          try {
            this.logger.debug('Processing batch', { jobId, batchIndex, batchSize: batch.length });
            
            // Update progress
            job.processedRecords += batch.length;
            const progress = (job.processedRecords / job.totalRecords) * 100;
            
            this.emit('progress', {
              jobId,
              batchIndex,
              totalBatches,
              processedRecords: job.processedRecords,
              totalRecords: job.totalRecords,
              progress
            });

            // Emit progress callback
            if (onProgress) {
              onProgress({
                jobId,
                batchIndex,
                totalBatches,
                processedRecords: job.processedRecords,
                totalRecords: job.totalRecords,
                progress
              });
            }

            // Process the batch
            const batchResult = await processor(batch, batchIndex);
            results.push(...batchResult);

            job.successfulBatches++;
            
            this.emit('batch_complete', {
              jobId,
              batchIndex,
              batchSize: batch.length,
              resultCount: batchResult.length
            });

            if (onBatchComplete) {
              onBatchComplete({
                jobId,
                batchIndex,
                batchSize: batch.length,
                resultCount: batchResult.length
              });
            }

          } catch (error) {
            const batchError: BatchError = {
              jobId,
              batchIndex,
              batchSize: batch.length,
              error: error as Error,
              timestamp: new Date()
            };

            errors.push(batchError);
            job.failedBatches++;

            this.logger.error('Batch processing failed', { 
              jobId, 
              batchIndex, 
              error: error as Error 
            });

            this.emit('batch_error', batchError);

            if (stopOnError) {
              throw error;
            }
          }
        };

        // Add to queue with concurrency control
        if (batchQueue.length >= maxConcurrent) {
          await Promise.race(batchQueue);
        }

        const batchPromise = processBatch();
        batchQueue.push(batchPromise);
        batchPromises.push(batchPromise);

        // Remove completed promises from queue
        Promise.resolve(batchPromise).finally(() => {
          const index = batchQueue.indexOf(batchPromise);
          if (index > -1) {
            batchQueue.splice(index, 1);
          }
        });
      }

      // Wait for all batches to complete
      await Promise.allSettled(batchPromises);

      // Complete job
      job.status = errors.length === 0 ? 'completed' : 'completed_with_errors';
      job.completedAt = new Date();

      const result: BatchResult<R> = {
        jobId,
        status: job.status,
        totalRecords: job.totalRecords,
        processedRecords: job.processedRecords,
        successfulBatches: job.successfulBatches,
        failedBatches: job.failedBatches,
        results,
        errors: errors.length > 0 ? errors : undefined,
        executionTime: job.completedAt.getTime() - job.startedAt.getTime(),
        averageBatchTime: job.successfulBatches > 0 ? 
          (job.completedAt.getTime() - job.startedAt.getTime()) / job.successfulBatches : 0
      };

      this.emit('job_complete', result);
      
      this.logger.info('Batch processing completed', {
        jobId,
        status: job.status,
        totalRecords: job.totalRecords,
        processedRecords: job.processedRecords,
        successfulBatches: job.successfulBatches,
        failedBatches: job.failedBatches,
        executionTime: result.executionTime
      });

      return result;

    } catch (error) {
      job.status = 'failed';
      job.completedAt = new Date();
      job.error = error as Error;

      this.logger.error('Batch processing failed', { jobId, error: error as Error });

      this.emit('job_error', { jobId, error: error as Error });
      throw error;

    } finally {
      this.activeBatches.delete(jobId);
    }
  }

  /**
   * Stream processing for very large datasets
   */
  async processStream<T, R>(
    dataStream: AsyncIterable<T>,
    processor: StreamProcessorFunction<T, R>,
    options: StreamProcessingOptions = {}
  ): Promise<StreamResult<R>> {
    const {
      batchSize = this.defaultBatchSize,
      maxConcurrent = this.maxConcurrentBatches,
      bufferSize = batchSize * 10
    } = options;

    const jobId = this.generateJobId();
    
    this.logger.info('Starting stream processing', { 
      jobId, 
      batchSize,
      bufferSize,
      maxConcurrent 
    });

    const job: BatchJob = {
      id: jobId,
      status: 'running',
      totalRecords: 0,
      processedRecords: 0,
      successfulBatches: 0,
      failedBatches: 0,
      startedAt: new Date()
    };

    this.activeBatches.set(jobId, job);

    try {
      const results: R[] = [];
      const batchBuffer: T[] = [];
      let batchIndex = 0;
      let processedCount = 0;

      for await (const item of dataStream) {
        batchBuffer.push(item);
        job.totalRecords++;

        if (batchBuffer.length >= batchSize) {
          const batch = [...batchBuffer];
          batchBuffer.length = 0;

          // Process batch
          const batchResult = await processor(batch, batchIndex);
          results.push(...batchResult);

          batchIndex++;
          processedCount += batch.length;
          job.processedRecords = processedCount;
          job.successfulBatches++;

          // Emit progress
          const progress = job.totalRecords > 0 ? (processedCount / job.totalRecords) * 100 : 0;
          this.emit('progress', {
            jobId,
            batchIndex,
            processedRecords: processedCount,
            totalRecords: job.totalRecords,
            progress
          });
        }
      }

      // Process remaining items in buffer
      if (batchBuffer.length > 0) {
        const batchResult = await processor(batchBuffer, batchIndex);
        results.push(...batchResult);
        job.successfulBatches++;
      }

      job.status = 'completed';
      job.completedAt = new Date();

      const result: StreamResult<R> = {
        jobId,
        status: job.status,
        totalRecords: job.totalRecords,
        processedRecords: job.processedRecords,
        successfulBatches: job.successfulBatches,
        failedBatches: job.failedBatches,
        results,
        executionTime: job.completedAt.getTime() - job.startedAt.getTime()
      };

      this.logger.info('Stream processing completed', {
        jobId,
        totalRecords: job.totalRecords,
        processedRecords: job.processedRecords,
        executionTime: result.executionTime
      });

      return result;

    } catch (error) {
      job.status = 'failed';
      job.completedAt = new Date();
      job.error = error as Error;

      this.logger.error('Stream processing failed', { jobId, error: error as Error });
      throw error;

    } finally {
      this.activeBatches.delete(jobId);
    }
  }

  /**
   * Cancel batch processing job
   */
  async cancelJob(jobId: string): Promise<void> {
    const job = this.activeBatches.get(jobId);
    if (job) {
      job.status = 'cancelled';
      this.activeBatches.delete(jobId);
      
      this.logger.info('Batch job cancelled', { jobId });
      this.emit('job_cancelled', { jobId });
    }
  }

  /**
   * Get job status
   */
  getJobStatus(jobId: string): BatchJob | null {
    return this.activeBatches.get(jobId) || null;
  }

  /**
   * List active jobs
   */
  getActiveJobs(): BatchJob[] {
    return Array.from(this.activeBatches.values());
  }

  /**
   * Pause/resume batch processing
   */
  async pauseJob(jobId: string): Promise<void> {
    const job = this.activeBatches.get(jobId);
    if (job) {
      job.status = 'paused';
      this.logger.info('Batch job paused', { jobId });
      this.emit('job_paused', { jobId });
    }
  }

  async resumeJob(jobId: string): Promise<void> {
    const job = this.activeBatches.get(jobId);
    if (job) {
      job.status = 'running';
      this.logger.info('Batch job resumed', { jobId });
      this.emit('job_resumed', { jobId });
    }
  }

  /**
   * Get processing statistics
   */
  getStatistics(): BatchStatistics {
    const activeJobs = this.activeBatches.size;
    const totalProcessed = Array.from(this.activeBatches.values())
      .reduce((sum, job) => sum + job.processedRecords, 0);

    return {
      activeJobs,
      totalProcessed,
      maxConcurrentBatches: this.maxConcurrentBatches,
      defaultBatchSize: this.defaultBatchSize
    };
  }

  private createBatches<T>(data: T[], batchSize: number): T[][] {
    const batches: T[][] = [];
    for (let i = 0; i < data.length; i += batchSize) {
      batches.push(data.slice(i, i + batchSize));
    }
    return batches;
  }

  private generateJobId(): string {
    return `batch_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
}

// Types and Interfaces
export interface BatchJob {
  id: string;
  status: 'running' | 'paused' | 'completed' | 'failed' | 'cancelled' | 'completed_with_errors';
  totalRecords: number;
  processedRecords: number;
  successfulBatches: number;
  failedBatches: number;
  startedAt: Date;
  completedAt?: Date;
  error?: Error;
}

export interface BatchResult<T> {
  jobId: string;
  status: string;
  totalRecords: number;
  processedRecords: number;
  successfulBatches: number;
  failedBatches: number;
  results: T[];
  errors?: BatchError[];
  executionTime: number;
  averageBatchTime: number;
}

export interface StreamResult<T> {
  jobId: string;
  status: string;
  totalRecords: number;
  processedRecords: number;
  successfulBatches: number;
  failedBatches: number;
  results: T[];
  executionTime: number;
}

export interface BatchError {
  jobId: string;
  batchIndex: number;
  batchSize: number;
  error: Error;
  timestamp: Date;
}

export interface BatchProcessingOptions {
  batchSize?: number;
  maxConcurrent?: number;
  onProgress?: (progress: BatchProgress) => void;
  onBatchComplete?: (batch: BatchComplete) => void;
  stopOnError?: boolean;
}

export interface StreamProcessingOptions {
  batchSize?: number;
  maxConcurrent?: number;
  bufferSize?: number;
}

export interface BatchProgress {
  jobId: string;
  batchIndex: number;
  totalBatches: number;
  processedRecords: number;
  totalRecords: number;
  progress: number;
}

export interface BatchComplete {
  jobId: string;
  batchIndex: number;
  batchSize: number;
  resultCount: number;
}

export interface BatchStatistics {
  activeJobs: number;
  totalProcessed: number;
  maxConcurrentBatches: number;
  defaultBatchSize: number;
}

// Function Types
export type BatchProcessorFunction<T, R> = (batch: T[], batchIndex: number) => Promise<R[]>;
export type StreamProcessorFunction<T, R> = (batch: T[], batchIndex: number) => Promise<R[]>;