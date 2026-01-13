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
import { Transformation } from './types';

export class DataTransformer {
  private logger: Logger;
  private builtInTransformations: Map<string, TransformationFunction> = new Map();

  constructor(logger: Logger) {
    this.logger = logger;
    this.initializeBuiltInTransformations();
  }

  /**
   * Apply transformation to data
   */
  async apply(data: any[], transformation: Transformation, options: any = {}): Promise<any[]> {
    this.logger.debug('Applying transformation', { 
      name: transformation.name, 
      type: transformation.type,
      recordCount: data.length 
    });

    const transformFunction = this.builtInTransformations.get(transformation.type);
    if (!transformFunction) {
      throw new Error(`Unknown transformation type: ${transformation.type}`);
    }

    try {
      const result = await transformFunction(data, transformation.parameters || {}, options);
      
      this.logger.debug('Transformation applied successfully', { 
        name: transformation.name, 
        inputCount: data.length,
        outputCount: result.length 
      });

      return result;

    } catch (error) {
      this.logger.error('Transformation failed', { 
        name: transformation.name, 
        type: transformation.type, 
        error 
      });
      throw error;
    }
  }

  /**
   * Register custom transformation
   */
  registerTransformation(type: string, fn: TransformationFunction): void {
    this.builtInTransformations.set(type, fn);
    this.logger.info('Custom transformation registered', { type });
  }

  /**
   * Get available transformations
   */
  getAvailableTransformations(): string[] {
    return Array.from(this.builtInTransformations.keys());
  }

  /**
   * Validate transformation parameters
   */
  validateTransformation(type: string, parameters: any): boolean {
    const transformFunction = this.builtInTransformations.get(type);
    if (!transformFunction) {
      return false;
    }

    // Basic parameter validation - in a real implementation, 
    // this would be more comprehensive
    return true;
  }

  /**
   * Apply multiple transformations in sequence
   */
  async applySequence(data: any[], transformations: Transformation[], options: any = {}): Promise<any[]> {
    this.logger.info('Applying transformation sequence', { 
      transformationCount: transformations.length,
      recordCount: data.length 
    });

    let transformedData = data;

    for (let i = 0; i < transformations.length; i++) {
      const transformation = transformations[i];
      this.logger.debug('Applying transformation in sequence', { 
        index: i,
        name: transformation.name,
        type: transformation.type 
      });

      transformedData = await this.apply(transformedData, transformation, options);
    }

    this.logger.info('Transformation sequence completed', {
      finalRecordCount: transformedData.length
    });

    return transformedData;
  }

  private initializeBuiltInTransformations(): void {
    // Field operations
    this.builtInTransformations.set('select_fields', this.selectFields);
    this.builtInTransformations.set('rename_field', this.renameField);
    this.builtInTransformations.set('add_field', this.addField);
    this.builtInTransformations.set('remove_field', this.removeField);
    this.builtInTransformations.set('calculate_field', this.calculateField);

    // Data type conversions
    this.builtInTransformations.set('convert_type', this.convertType);
    this.builtInTransformations.set('parse_date', this.parseDate);
    this.builtInTransformations.set('format_date', this.formatDate);
    this.builtInTransformations.set('normalize_string', this.normalizeString);

    // Filtering and sorting
    this.builtInTransformations.set('filter', this.filterData);
    this.builtInTransformations.set('sort', this.sortData);
    this.builtInTransformations.set('deduplicate', this.deduplicate);
    this.builtInTransformations.set('sample', this.sampleData);

    // Aggregations
    this.builtInTransformations.set('group_by', this.groupBy);
    this.builtInTransformations.set('aggregate', this.aggregate);
    this.builtInTransformations.set('pivot', this.pivot);

    // Text processing
    this.builtInTransformations.set('text_replace', this.textReplace);
    this.builtInTransformations.set('text_extract', this.textExtract);
    this.builtInTransformations.set('text_split', this.textSplit);

    // Advanced operations
    this.builtInTransformations.set('join', this.joinData);
    this.builtInTransformations.set('union', this.unionData);
    this.builtInTransformations.set('split', this.splitData);
  }

  // Field Operations
  private selectFields = async (data: any[], params: any): Promise<any[]> => {
    const { fields } = params;
    return data.map(record => {
      const selected: any = {};
      for (const field of fields) {
        selected[field] = record[field];
      }
      return selected;
    });
  };

  private renameField = async (data: any[], params: any): Promise<any[]> => {
    const { oldField, newField } = params;
    return data.map(record => {
      const renamed = { ...record };
      if (renamed.hasOwnProperty(oldField)) {
        renamed[newField] = renamed[oldField];
        delete renamed[oldField];
      }
      return renamed;
    });
  };

  private addField = async (data: any[], params: any): Promise<any[]> => {
    const { field, value, expression } = params;
    return data.map(record => {
      const newRecord = { ...record };
      newRecord[field] = expression ? this.evaluateExpression(expression, record) : value;
      return newRecord;
    });
  };

  private removeField = async (data: any[], params: any): Promise<any[]> => {
    const { fields } = params;
    return data.map(record => {
      const cleaned = { ...record };
      for (const field of fields) {
        delete cleaned[field];
      }
      return cleaned;
    });
  };

  private calculateField = async (data: any[], params: any): Promise<any[]> => {
    const { targetField, expression, fields } = params;
    return data.map(record => {
      const newRecord = { ...record };
      const context = fields ? 
        fields.reduce((ctx: any, field: string) => ({ ...ctx, [field]: record[field] }), {}) : 
        record;
      newRecord[targetField] = this.evaluateExpression(expression, context);
      return newRecord;
    });
  };

  // Data Type Conversions
  private convertType = async (data: any[], params: any): Promise<any[]> => {
    const { field, targetType, format } = params;
    return data.map(record => {
      const converted = { ...record };
      const value = record[field];
      
      switch (targetType) {
        case 'number':
          converted[field] = Number(value);
          break;
        case 'string':
          converted[field] = String(value);
          break;
        case 'boolean':
          converted[field] = Boolean(value);
          break;
        case 'date':
          converted[field] = new Date(value);
          break;
        default:
          converted[field] = value;
      }
      
      return converted;
    });
  };

  private parseDate = async (data: any[], params: any): Promise<any[]> => {
    const { field, inputFormat, outputFormat } = params;
    return data.map(record => {
      const parsed = { ...record };
      const date = new Date(record[field]);
      parsed[field] = outputFormat ? this.formatDateString(date, outputFormat) : date;
      return parsed;
    });
  };

  private formatDate = async (data: any[], params: any): Promise<any[]> => {
    const { field, format } = params;
    return data.map(record => {
      const formatted = { ...record };
      const date = new Date(record[field]);
      formatted[field] = this.formatDateString(date, format);
      return formatted;
    });
  };

  private normalizeString = async (data: any[], params: any): Promise<any[]> => {
    const { field, operation = 'lowercase' } = params;
    return data.map(record => {
      const normalized = { ...record };
      const value = String(record[field]);
      
      switch (operation) {
        case 'lowercase':
          normalized[field] = value.toLowerCase();
          break;
        case 'uppercase':
          normalized[field] = value.toUpperCase();
          break;
        case 'trim':
          normalized[field] = value.trim();
          break;
        case 'slug':
          normalized[field] = value.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
          break;
        default:
          normalized[field] = value;
      }
      
      return normalized;
    });
  };

  // Filtering and Sorting
  private filterData = async (data: any[], params: any): Promise<any[]> => {
    const { field, operator, value } = params;
    return data.filter(record => {
      const fieldValue = record[field];
      
      switch (operator) {
        case 'equals':
          return fieldValue === value;
        case 'not_equals':
          return fieldValue !== value;
        case 'greater_than':
          return fieldValue > value;
        case 'less_than':
          return fieldValue < value;
        case 'contains':
          return String(fieldValue).includes(value);
        case 'starts_with':
          return String(fieldValue).startsWith(value);
        case 'ends_with':
          return String(fieldValue).endsWith(value);
        case 'regex':
          return new RegExp(value).test(String(fieldValue));
        default:
          return true;
      }
    });
  };

  private sortData = async (data: any[], params: any): Promise<any[]> => {
    const { field, order = 'asc' } = params;
    return data.sort((a, b) => {
      const aVal = a[field];
      const bVal = b[field];
      
      if (order === 'desc') {
        return bVal > aVal ? 1 : -1;
      } else {
        return aVal > bVal ? 1 : -1;
      }
    });
  };

  private deduplicate = async (data: any[], params: any): Promise<any[]> => {
    const { fields } = params;
    const seen = new Set();
    
    return data.filter(record => {
      const key = fields ? 
        fields.map((f: string) => record[f]).join('|') : 
        JSON.stringify(record);
      
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  };

  private sampleData = async (data: any[], params: any): Promise<any[]> => {
    const { count, percentage } = params;
    
    if (count) {
      const shuffled = [...data].sort(() => 0.5 - Math.random());
      return shuffled.slice(0, count);
    } else if (percentage) {
      const sampleSize = Math.floor(data.length * (percentage / 100));
      const shuffled = [...data].sort(() => 0.5 - Math.random());
      return shuffled.slice(0, sampleSize);
    }
    
    return data;
  };

  // Aggregations
  private groupBy = async (data: any[], params: any): Promise<any[]> => {
    const { groupField, aggregateField, operation = 'sum' } = params;
    const groups: Map<string, any[]> = new Map();
    
    // Group data
    data.forEach(record => {
      const key = record[groupField];
      if (!groups.has(key)) {
        groups.set(key, []);
      }
      groups.get(key)!.push(record);
    });
    
    // Aggregate each group
    const result: any[] = [];
    for (const [key, groupData] of groups) {
      const aggregated: any = { [groupField]: key };
      
      if (aggregateField) {
        const values = groupData.map(record => record[aggregateField]);
        switch (operation) {
          case 'sum':
            aggregated[`${aggregateField}_${operation}`] = values.reduce((a, b) => a + b, 0);
            break;
          case 'avg':
            aggregated[`${aggregateField}_${operation}`] = values.reduce((a, b) => a + b, 0) / values.length;
            break;
          case 'min':
            aggregated[`${aggregateField}_${operation}`] = Math.min(...values);
            break;
          case 'max':
            aggregated[`${aggregateField}_${operation}`] = Math.max(...values);
            break;
          case 'count':
            aggregated[`${aggregateField}_${operation}`] = values.length;
            break;
        }
      }
      
      result.push(aggregated);
    }
    
    return result;
  };

  private aggregate = async (data: any[], params: any): Promise<any[]> => {
    // Simplified aggregation - in a real implementation, this would be more comprehensive
    const { operations } = params;
    const result: any = {};
    
    for (const op of operations) {
      const { field, operation } = op;
      const values = data.map(record => record[field]).filter(v => v !== undefined && v !== null);
      
      switch (operation) {
        case 'sum':
          result[`${field}_sum`] = values.reduce((a, b) => a + b, 0);
          break;
        case 'avg':
          result[`${field}_avg`] = values.reduce((a, b) => a + b, 0) / values.length;
          break;
        case 'count':
          result[`${field}_count`] = values.length;
          break;
      }
    }
    
    return [result];
  };

  private pivot = async (data: any[], params: any): Promise<any[]> => {
    // Simplified pivot - in a real implementation, this would be more flexible
    const { indexField, columnField, valueField, operation = 'sum' } = params;
    const pivotData: Map<string, any> = new Map();
    
    data.forEach(record => {
      const index = record[indexField];
      const column = record[columnField];
      const value = record[valueField];
      
      if (!pivotData.has(index)) {
        pivotData.set(index, { [indexField]: index });
      }
      
      const row = pivotData.get(index)!;
      row[column] = (row[column] || 0) + value;
    });
    
    return Array.from(pivotData.values());
  };

  // Text Processing
  private textReplace = async (data: any[], params: any): Promise<any[]> => {
    const { field, pattern, replacement, flags = 'g' } = params;
    const regex = new RegExp(pattern, flags);
    
    return data.map(record => {
      const replaced = { ...record };
      replaced[field] = String(record[field]).replace(regex, replacement);
      return replaced;
    });
  };

  private textExtract = async (data: any[], params: any): Promise<any[]> => {
    const { field, pattern, group = 0 } = params;
    const regex = new RegExp(pattern);
    
    return data.map(record => {
      const match = String(record[field]).match(regex);
      const extracted = { ...record };
      extracted[`${field}_extracted`] = match ? match[group] : null;
      return extracted;
    });
  };

  private textSplit = async (data: any[], params: any): Promise<any[]> => {
    const { field, delimiter, maxParts, targetField } = params;
    
    return data.map(record => {
      const parts = String(record[field]).split(delimiter).slice(0, maxParts);
      const splitData = { ...record };
      
      parts.forEach((part, index) => {
        splitData[`${targetField || field}_part_${index + 1}`] = part;
      });
      
      return splitData;
    });
  };

  // Advanced Operations
  private joinData = async (data: any[], params: any): Promise<any[]> => {
    // This would require another dataset - simplified implementation
    const { leftField, rightField, joinType = 'inner' } = params;
    
    // In a real implementation, this would perform actual joins
    return data; // Simplified
  };

  private unionData = async (data: any[], params: any): Promise<any[]> => {
    // This would combine multiple datasets - simplified implementation
    return data; // Simplified
  };

  private splitData = async (data: any[], params: any): Promise<any[]> => {
    const { field, delimiter } = params;
    const splitRecords: any[] = [];
    
    data.forEach(record => {
      const values = String(record[field]).split(delimiter);
      values.forEach(value => {
        splitRecords.push({ ...record, [field]: value });
      });
    });
    
    return splitRecords;
  };

  // Helper Methods
  private evaluateExpression(expression: string, context: any): any {
    // Simple expression evaluator - in a real implementation, use a proper expression parser
    try {
      // Create a safe evaluation context
      const func = new Function(...Object.keys(context), `return ${expression}`);
      return func(...Object.values(context));
    } catch (error) {
      this.logger.warn('Expression evaluation failed', { expression, error });
      return null;
    }
  }

  private formatDateString(date: Date, format: string): string {
    // Simple date formatting - in a real implementation, use a proper date library
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    const seconds = String(date.getSeconds()).padStart(2, '0');
    
    return format
      .replace('YYYY', year.toString())
      .replace('MM', month)
      .replace('DD', day)
      .replace('HH', hours)
      .replace('mm', minutes)
      .replace('ss', seconds);
  }
}

type TransformationFunction = (data: any[], parameters: any, options: any) => Promise<any[]>;