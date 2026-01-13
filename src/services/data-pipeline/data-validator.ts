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
import { DataValidationResult, ValidationRule, SchemaDefinition } from './types';

export class DataValidator {
  private logger: Logger;
  private defaultRules: Map<string, ValidationRule[]> = new Map();

  constructor(logger: Logger) {
    this.logger = logger;
    this.initializeDefaultRules();
  }

  /**
   * Validate data against validation rules
   */
  async validate(data: any[], rules: ValidationRule[]): Promise<DataValidationResult> {
    this.logger.info('Starting data validation', { 
      recordCount: data.length, 
      ruleCount: rules.length 
    });

    const result: DataValidationResult = {
      isValid: true,
      totalRecords: data.length,
      validRecords: 0,
      invalidRecords: 0,
      errors: [],
      warnings: [],
      statistics: {}
    };

    // Track field statistics
    const fieldStats: Map<string, FieldStats> = new Map();

    for (let i = 0; i < data.length; i++) {
      const record = data[i];
      const recordErrors: ValidationError[] = [];
      const recordWarnings: ValidationWarning[] = [];

      // Apply each validation rule
      for (const rule of rules) {
        const fieldValue = record[rule.field];
        
        try {
          const ruleResult = await this.validateField(fieldValue, rule);
          
          if (!ruleResult.isValid) {
            recordErrors.push({
              field: rule.field,
              rule: rule.type,
              value: fieldValue,
              message: ruleResult.message || `${rule.field} failed ${rule.type} validation`,
              severity: 'error'
            });
          }

          // Track field statistics
          this.updateFieldStats(fieldStats, rule.field, fieldValue, ruleResult.isValid);

        } catch (error) {
          this.logger.warn('Validation rule execution failed', { 
            rule: rule.type, 
            field: rule.field, 
            error: error as Error 
          });
          
          recordErrors.push({
            field: rule.field,
            rule: rule.type,
            value: fieldValue,
            message: `Validation rule error: ${(error as Error).message}`,
            severity: 'error'
          });
        }
      }

      if (recordErrors.length === 0) {
        result.validRecords++;
      } else {
        result.invalidRecords++;
        result.isValid = false;
        result.errors.push(...recordErrors);
      }

      // Add record warnings if any
      if (recordWarnings.length > 0) {
        result.warnings.push(...recordWarnings);
      }
    }

    // Generate field statistics
    result.statistics = this.generateFieldStatistics(fieldStats);

    this.logger.info('Data validation completed', {
      isValid: result.isValid,
      validRecords: result.validRecords,
      invalidRecords: result.invalidRecords,
      errorCount: result.errors.length
    });

    return result;
  }

  /**
   * Validate data against schema
   */
  async validateAgainstSchema(data: any[], schema: SchemaDefinition): Promise<DataValidationResult> {
    this.logger.info('Starting schema validation', { 
      recordCount: data.length, 
      fieldCount: schema.fields.length 
    });

    const rules: ValidationRule[] = [];

    // Generate validation rules from schema
    for (const field of schema.fields) {
      // Required field validation
      if (!field.nullable) {
        rules.push({
          field: field.name,
          type: 'required',
          message: `${field.name} is required`
        });
      }

      // Type validation
      rules.push({
        field: field.name,
        type: 'schema_type',
        value: field.type,
        message: `${field.name} must be of type ${field.type}`
      });

      // Check constraints
      for (const constraint of schema.constraints) {
        if (constraint.field === field.name) {
          rules.push({
            field: field.name,
            type: constraint.type,
            value: constraint.value,
            message: `${field.name} violates ${constraint.type} constraint`
          });
        }
      }
    }

    return await this.validate(data, rules);
  }

  /**
   * Add default validation rules for common data types
   */
  addDefaultRules(dataType: string, rules: ValidationRule[]): void {
    this.defaultRules.set(dataType, rules);
  }

  /**
   * Get default validation rules for data type
   */
  getDefaultRules(dataType: string): ValidationRule[] {
    return this.defaultRules.get(dataType) || [];
  }

  /**
   * Validate email addresses
   */
  validateEmail(email: string): boolean {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }

  /**
   * Validate phone numbers
   */
  validatePhone(phone: string): boolean {
    const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
    return phoneRegex.test(phone.replace(/\s+/g, ''));
  }

  /**
   * Validate dates
   */
  validateDate(date: string | Date, format?: string): boolean {
    const parsed = new Date(date);
    return !isNaN(parsed.getTime());
  }

  /**
   * Validate numeric ranges
   */
  validateRange(value: number, min?: number, max?: number): boolean {
    if (min !== undefined && value < min) return false;
    if (max !== undefined && value > max) return false;
    return true;
  }

  /**
   * Validate string length
   */
  validateLength(value: string, min?: number, max?: number): boolean {
    if (min !== undefined && value.length < min) return false;
    if (max !== undefined && value.length > max) return false;
    return true;
  }

  /**
   * Validate against patterns
   */
  validatePattern(value: string, pattern: string | RegExp): boolean {
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
    return regex.test(value);
  }

  /**
   * Validate custom business rules
   */
  async validateCustomRule(value: any, rule: any): Promise<boolean> {
    if (typeof rule.validator === 'function') {
      return await rule.validator(value);
    }
    
    if (typeof rule.validator === 'string') {
      // Support for predefined validation functions
      switch (rule.validator) {
        case 'email':
          return this.validateEmail(value);
        case 'phone':
          return this.validatePhone(value);
        case 'date':
          return this.validateDate(value);
        default:
          return true;
      }
    }

    return true;
  }

  private async validateField(value: any, rule: ValidationRule): Promise<ValidationResult> {
    switch (rule.type) {
      case 'required':
        return {
          isValid: value !== null && value !== undefined && value !== '',
          message: rule.message || `${rule.field} is required`
        };

      case 'type':
        return {
          isValid: this.validateType(value, rule.value),
          message: rule.message || `${rule.field} must be of type ${rule.value}`
        };

      case 'range':
        return {
          isValid: this.validateRange(value, rule.min, rule.max),
          message: rule.message || `${rule.field} must be between ${rule.min} and ${rule.max}`
        };

      case 'pattern':
        return {
          isValid: this.validatePattern(value, rule.pattern),
          message: rule.message || `${rule.field} must match pattern ${rule.pattern}`
        };

      case 'length':
        return {
          isValid: this.validateLength(value, rule.min, rule.max),
          message: rule.message || `${rule.field} length must be between ${rule.min} and ${rule.max}`
        };

      case 'email':
        return {
          isValid: this.validateEmail(value),
          message: rule.message || `${rule.field} must be a valid email`
        };

      case 'phone':
        return {
          isValid: this.validatePhone(value),
          message: rule.message || `${rule.field} must be a valid phone number`
        };

      case 'date':
        return {
          isValid: this.validateDate(value),
          message: rule.message || `${rule.field} must be a valid date`
        };

      case 'custom':
        return {
          isValid: await this.validateCustomRule(value, rule),
          message: rule.message || `${rule.field} failed custom validation`
        };

      case 'schema_type':
        return {
          isValid: this.validateSchemaType(value, rule.value),
          message: rule.message || `${rule.field} must be of type ${rule.value}`
        };

      default:
        return {
          isValid: true,
          message: 'Unknown validation rule type'
        };
    }
  }

  private validateType(value: any, expectedType: string): boolean {
    if (value === null || value === undefined) return true; // Let required handle null checks
    
    switch (expectedType) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number' && !isNaN(value);
      case 'boolean':
        return typeof value === 'boolean';
      case 'date':
        return value instanceof Date || this.validateDate(value);
      case 'array':
        return Array.isArray(value);
      case 'object':
        return typeof value === 'object' && !Array.isArray(value);
      default:
        return true;
    }
  }

  private validateSchemaType(value: any, expectedType: string): boolean {
    return this.validateType(value, expectedType);
  }

  private updateFieldStats(fieldStats: Map<string, FieldStats>, fieldName: string, value: any, isValid: boolean): void {
    let stats = fieldStats.get(fieldName);
    if (!stats) {
      stats = {
        totalCount: 0,
        nullCount: 0,
        uniqueCount: 0,
        minValue: null,
        maxValue: null,
        averageValue: null,
        validCount: 0,
        invalidCount: 0
      };
      fieldStats.set(fieldName, stats);
    }

    stats.totalCount++;
    
    if (value === null || value === undefined) {
      stats.nullCount++;
    }

    if (isValid) {
      stats.validCount++;
    } else {
      stats.invalidCount++;
    }

    // Update numeric statistics if applicable
    if (typeof value === 'number' && !isNaN(value)) {
      if (stats.minValue === null || value < stats.minValue) {
        stats.minValue = value;
      }
      if (stats.maxValue === null || value > stats.maxValue) {
        stats.maxValue = value;
      }
    }
  }

  private generateFieldStatistics(fieldStats: Map<string, FieldStats>): Record<string, FieldStatistics> {
    const statistics: Record<string, FieldStatistics> = {};

    for (const [fieldName, stats] of fieldStats) {
      statistics[fieldName] = {
        ...stats,
        nullPercentage: (stats.nullCount / stats.totalCount) * 100,
        validPercentage: (stats.validCount / stats.totalCount) * 100,
        invalidPercentage: (stats.invalidCount / stats.totalCount) * 100
      };
    }

    return statistics;
  }

  private initializeDefaultRules(): void {
    // Add common validation rules for different data types
    this.defaultRules.set('user', [
      { field: 'email', type: 'email', message: 'Invalid email format' },
      { field: 'age', type: 'range', min: 0, max: 150, message: 'Age must be between 0 and 150' }
    ]);

    this.defaultRules.set('product', [
      { field: 'price', type: 'range', min: 0, message: 'Price must be positive' },
      { field: 'sku', type: 'pattern', pattern: '^[A-Z0-9]{8}$', message: 'SKU must be 8 alphanumeric characters' }
    ]);
  }
}

interface ValidationResult {
  isValid: boolean;
  message: string;
}

interface ValidationError {
  field: string;
  rule: string;
  value: any;
  message: string;
  severity: 'error' | 'warning';
}

interface ValidationWarning {
  field: string;
  message: string;
  severity: 'warning';
}

interface FieldStats {
  totalCount: number;
  nullCount: number;
  uniqueCount: number;
  minValue: any;
  maxValue: any;
  averageValue: any;
  validCount: number;
  invalidCount: number;
}

interface FieldStatistics extends FieldStats {
  nullPercentage: number;
  validPercentage: number;
  invalidPercentage: number;
}