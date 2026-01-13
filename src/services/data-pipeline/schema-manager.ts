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
import { SchemaDefinition, SchemaField, SchemaConstraint, SchemaVersion } from './types';

export class SchemaManager {
  private logger: Logger;
  private schemas: Map<string, SchemaDefinition> = new Map();
  private schemaVersions: Map<string, SchemaVersion[]> = new Map();
  private schemaHistory: Map<string, SchemaHistoryEntry[]> = new Map();

  constructor(logger: Logger) {
    this.logger = logger;
    this.initializeBuiltInSchemas();
  }

  /**
   * Create a new schema
   */
  async createSchema(name: string, schema: SchemaDefinition): Promise<string> {
    this.logger.info('Creating new schema', { schemaName: name, fieldCount: schema.fields.length });

    // Validate schema
    await this.validateSchema(schema);

    // Check if schema already exists
    if (this.schemas.has(name)) {
      throw new Error(`Schema already exists: ${name}`);
    }

    // Store schema
    this.schemas.set(name, schema);

    // Initialize version tracking
    const version: SchemaVersion = {
      version: '1.0.0',
      schema,
      createdAt: new Date(),
      createdBy: 'system',
      description: `Initial version of ${name} schema`
    };

    this.schemaVersions.set(name, [version]);

    // Track history
    this.addToHistory(name, {
      action: 'create',
      version: '1.0.0',
      timestamp: new Date(),
      user: 'system',
      description: 'Schema created'
    });

    this.logger.info('Schema created successfully', { schemaName: name, version: '1.0.0' });
    return name;
  }

  /**
   * Update an existing schema
   */
  async updateSchema(name: string, newSchema: SchemaDefinition, description?: string): Promise<void> {
    this.logger.info('Updating schema', { schemaName: name, description });

    const existingSchema = this.schemas.get(name);
    if (!existingSchema) {
      throw new Error(`Schema not found: ${name}`);
    }

    // Validate new schema
    await this.validateSchema(newSchema);

    // Determine version bump type
    const versionBump = this.determineVersionBump(existingSchema, newSchema);
    const newVersion = this.bumpVersion(this.getLatestVersion(name), versionBump);

    // Update schema
    this.schemas.set(name, newSchema);

    // Add version
    const version: SchemaVersion = {
      version: newVersion,
      schema: newSchema,
      createdAt: new Date(),
      createdBy: 'system',
      description: description || `Updated ${name} schema`
    };

    const versions = this.schemaVersions.get(name) || [];
    versions.push(version);
    this.schemaVersions.set(name, versions);

    // Track history
    this.addToHistory(name, {
      action: 'update',
      version: newVersion,
      timestamp: new Date(),
      user: 'system',
      description: description || 'Schema updated'
    });

    this.logger.info('Schema updated successfully', { 
      schemaName: name, 
      oldVersion: this.getLatestVersion(name),
      newVersion 
    });
  }

  /**
   * Get schema by name
   */
  async getSchema(name: string): Promise<SchemaDefinition | null> {
    const schema = this.schemas.get(name);
    if (!schema) {
      this.logger.warn('Schema not found', { schemaName: name });
      return null;
    }
    return schema;
  }

  /**
   * Get schema version
   */
  async getSchemaVersion(name: string, version?: string): Promise<SchemaVersion | null> {
    const versions = this.schemaVersions.get(name);
    if (!versions) {
      return null;
    }

    if (version) {
      return versions.find(v => v.version === version) || null;
    }

    return versions[versions.length - 1] || null;
  }

  /**
   * Get all schema versions
   */
  async getSchemaVersions(name: string): Promise<SchemaVersion[]> {
    return this.schemaVersions.get(name) || [];
  }

  /**
   * Get schema history
   */
  async getSchemaHistory(name: string): Promise<SchemaHistoryEntry[]> {
    return this.schemaHistory.get(name) || [];
  }

  /**
   * Delete schema
   */
  async deleteSchema(name: string): Promise<void> {
    this.logger.info('Deleting schema', { schemaName: name });

    if (!this.schemas.has(name)) {
      throw new Error(`Schema not found: ${name}`);
    }

    // Add to history before deletion
    this.addToHistory(name, {
      action: 'delete',
      version: this.getLatestVersion(name),
      timestamp: new Date(),
      user: 'system',
      description: 'Schema deleted'
    });

    // Remove schema and associated data
    this);
    this.schema.schemas.delete(nameVersions.delete(name);
    this.schemaHistory.delete(name);

    this.logger.info('Schema deleted successfully', { schemaName: name });
  }

  /**
   * List all schemas
   */
  async listSchemas(): Promise<SchemaSummary[]> {
    const summaries: SchemaSummary[] = [];

    for (const [name, schema] of this.schemas) {
      const versions = this.schemaVersions.get(name) || [];
      const latestVersion = versions[versions.length - 1];

      summaries.push({
        name,
        latestVersion: latestVersion?.version || '1.0.0',
        fieldCount: schema.fields.length,
        constraintCount: schema.constraints.length,
        createdAt: latestVersion?.createdAt,
        updatedAt: latestVersion?.createdAt,
        description: latestVersion?.description
      });
    }

    return summaries;
  }

  /**
   * Validate data against schema
   */
  async validateData(data: any[], schemaName: string): Promise<boolean> {
    const schema = await this.getSchema(schemaName);
    if (!schema) {
      throw new Error(`Schema not found: ${schemaName}`);
    }

    this.logger.info('Validating data against schema', { 
      schemaName, 
      recordCount: data.length 
    });

    for (const record of data) {
      // Check required fields
      for (const field of schema.fields) {
        if (!field.nullable && (record[field.name] === null || record[field.name] === undefined)) {
          this.logger.warn('Data validation failed', { 
            schemaName, 
            field: field.name, 
            reason: 'required field missing' 
          });
          return false;
        }

        // Check field types
        if (record[field.name] !== null && record[field.name] !== undefined) {
          if (!this.validateFieldType(record[field.name], field.type)) {
            this.logger.warn('Data validation failed', { 
              schemaName, 
              field: field.name, 
              expectedType: field.type,
              actualType: typeof record[field.name] 
            });
            return false;
          }
        }
      }

      // Check constraints
      for (const constraint of schema.constraints) {
        if (!this.validateConstraint(record, constraint)) {
          this.logger.warn('Data validation failed', { 
            schemaName, 
            constraint: constraint.type, 
            field: constraint.field 
          });
          return false;
        }
      }
    }

    return true;
  }

  /**
   * Compare two schemas
   */
  async compareSchemas(name1: string, name2: string): Promise<SchemaComparison> {
    const schema1 = await this.getSchema(name1);
    const schema2 = await this.getSchema(name2);

    if (!schema1 || !schema2) {
      throw new Error('One or both schemas not found');
    }

    const comparison: SchemaComparison = {
      schemasCompared: [name1, name2],
      differences: [],
      breakingChanges: [],
      compatibleChanges: []
    };

    // Compare fields
    const fields1 = new Map(schema1.fields.map(f => [f.name, f]));
    const fields2 = new Map(schema2.fields.map(f => [f.name, f]));

    // Find added fields
    for (const [fieldName, field] of fields2) {
      if (!fields1.has(fieldName)) {
        comparison.differences.push({
          type: 'field_added',
          field: fieldName,
          description: `Field ${fieldName} was added`
        });
      }
    }

    // Find removed fields
    for (const [fieldName, field] of fields1) {
      if (!fields2.has(fieldName)) {
        comparison.differences.push({
          type: 'field_removed',
          field: fieldName,
          description: `Field ${fieldName} was removed`
        });
        comparison.breakingChanges.push({
          type: 'field_removed',
          field: fieldName,
          severity: 'breaking',
          description: `Field ${fieldName} was removed`
        });
      }
    }

    // Find modified fields
    for (const [fieldName, field1] of fields1) {
      const field2 = fields2.get(fieldName);
      if (field2) {
        const changes = this.compareFields(field1, field2);
        comparison.differences.push(...changes);

        // Check for breaking changes
        for (const change of changes) {
          if (this.isBreakingChange(change)) {
            comparison.breakingChanges.push(change);
          } else {
            comparison.compatibleChanges.push(change);
          }
        }
      }
    }

    return comparison;
  }

  /**
   * Generate schema from data sample
   */
  async generateSchemaFromData(data: any[], name: string): Promise<SchemaDefinition> {
    this.logger.info('Generating schema from data sample', { 
      schemaName: name, 
      recordCount: data.length 
    });

    const fieldMap: Map<string, any[]> = new Map();

    // Collect samples for each field
    for (const record of data) {
      for (const [fieldName, value] of Object.entries(record)) {
        if (!fieldMap.has(fieldName)) {
          fieldMap.set(fieldName, []);
        }
        fieldMap.get(fieldName)!.push(value);
      }
    }

    // Infer field types and properties
    const fields: SchemaField[] = [];
    for (const [fieldName, samples] of fieldMap) {
      const fieldType = this.inferFieldType(samples);
      const isNullable = samples.some(s => s === null || s === undefined);

      fields.push({
        name: fieldName,
        type: fieldType,
        nullable: isNullable,
        description: `Auto-generated field ${fieldName}`
      });
    }

    const schema: SchemaDefinition = {
      fields,
      constraints: []
    };

    return schema;
  }

  private async validateSchema(schema: SchemaDefinition): Promise<void> {
    if (!schema.fields || schema.fields.length === 0) {
      throw new Error('Schema must have at least one field');
    }

    // Validate field names are unique
    const fieldNames = schema.fields.map(f => f.name);
    const uniqueFieldNames = new Set(fieldNames);
    if (fieldNames.length !== uniqueFieldNames.size) {
      throw new Error('Field names must be unique');
    }

    // Validate field types
    for (const field of schema.fields) {
      if (!field.name || !field.type) {
        throw new Error('Field name and type are required');
      }

      if (!['string', 'number', 'boolean', 'date', 'array', 'object'].includes(field.type)) {
        throw new Error(`Invalid field type: ${field.type}`);
      }
    }

    // Validate constraints
    for (const constraint of schema.constraints) {
      if (!constraint.type || !constraint.field) {
        throw new Error('Constraint type and field are required');
      }

      if (!['unique', 'foreign_key', 'check', 'not_null'].includes(constraint.type)) {
        throw new Error(`Invalid constraint type: ${constraint.type}`);
      }
    }
  }

  private validateFieldType(value: any, expectedType: string): boolean {
    switch (expectedType) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number' && !isNaN(value);
      case 'boolean':
        return typeof value === 'boolean';
      case 'date':
        return value instanceof Date || (!isNaN(new Date(value).getTime()));
      case 'array':
        return Array.isArray(value);
      case 'object':
        return typeof value === 'object' && !Array.isArray(value);
      default:
        return true;
    }
  }

  private validateConstraint(record: any, constraint: SchemaConstraint): boolean {
    const value = record[constraint.field];

    switch (constraint.type) {
      case 'not_null':
        return value !== null && value !== undefined;
      
      case 'unique':
        // This would require checking against other records
        return true; // Simplified for this example
      
      case 'check':
        // Custom validation logic would go here
        return true;
      
      case 'foreign_key':
        // Foreign key validation would go here
        return true;
      
      default:
        return true;
    }
  }

  private compareFields(field1: SchemaField, field2: SchemaField): SchemaDifference[] {
    const differences: SchemaDifference[] = [];

    if (field1.type !== field2.type) {
      differences.push({
        type: 'field_type_changed',
        field: field1.name,
        oldValue: field1.type,
        newValue: field2.type,
        description: `Field type changed from ${field1.type} to ${field2.type}`
      });
    }

    if (field1.nullable !== field2.nullable) {
      differences.push({
        type: 'field_nullability_changed',
        field: field1.name,
        oldValue: field1.nullable,
        newValue: field2.nullable,
        description: `Field nullability changed from ${field1.nullable} to ${field2.nullable}`
      });
    }

    return differences;
  }

  private isBreakingChange(change: SchemaDifference): boolean {
    return change.type === 'field_removed' || 
           change.type === 'field_type_changed' ||
           change.type === 'field_nullability_changed';
  }

  private inferFieldType(samples: any[]): SchemaField['type'] {
    // Remove null/undefined values for type inference
    const validSamples = samples.filter(s => s !== null && s !== undefined);
    
    if (validSamples.length === 0) {
      return 'string'; // Default to string for empty fields
    }

    // Check types in order of specificity
    const sample = validSamples[0];
    
    if (sample instanceof Date) return 'date';
    if (Array.isArray(sample)) return 'array';
    if (typeof sample === 'object') return 'object';
    if (typeof sample === 'boolean') return 'boolean';
    if (typeof sample === 'number') return 'number';
    
    return 'string';
  }

  private determineVersionBump(oldSchema: SchemaDefinition, newSchema: SchemaDefinition): 'major' | 'minor' | 'patch' {
    // Check for breaking changes
    const oldFields = new Map(oldSchema.fields.map(f => [f.name, f]));
    const newFields = new Map(newSchema.fields.map(f => [f.name, f]));

    // Removed fields are breaking changes
    for (const [fieldName] of oldFields) {
      if (!newFields.has(fieldName)) {
        return 'major';
      }
    }

    // Type changes or nullability changes are breaking changes
    for (const [fieldName, oldField] of oldFields) {
      const newField = newFields.get(fieldName);
      if (newField && (oldField.type !== newField.type || oldField.nullable !== newField.nullable)) {
        return 'major';
      }
    }

    // Added fields are minor changes
    for (const [fieldName] of newFields) {
      if (!oldFields.has(fieldName)) {
        return 'minor';
      }
    }

    // Other changes are patch changes
    return 'patch';
  }

  private bumpVersion(currentVersion: string, bumpType: 'major' | 'minor' | 'patch'): string {
    const [major, minor, patch] = currentVersion.split('.').map(Number);
    
    switch (bumpType) {
      case 'major':
        return `${major + 1}.0.0`;
      case 'minor':
        return `${major}.${minor + 1}.0`;
      case 'patch':
        return `${major}.${minor}.${patch + 1}`;
      default:
        return currentVersion;
    }
  }

  private getLatestVersion(schemaName: string): string {
    const versions = this.schemaVersions.get(schemaName);
    if (!versions || versions.length === 0) {
      return '1.0.0';
    }
    return versions[versions.length - 1].version;
  }

  private addToHistory(schemaName: string, entry: SchemaHistoryEntry): void {
    const history = this.schemaHistory.get(schemaName) || [];
    history.push(entry);
    this.schemaHistory.set(schemaName, history);
  }

  private initializeBuiltInSchemas(): void {
    // Initialize with some common schemas
    const commonSchemas = [
      {
        name: 'user',
        schema: {
          fields: [
            { name: 'id', type: 'string', nullable: false, description: 'User ID' },
            { name: 'email', type: 'string', nullable: false, description: 'User email' },
            { name: 'firstName', type: 'string', nullable: false, description: 'First name' },
            { name: 'lastName', type: 'string', nullable: false, description: 'Last name' },
            { name: 'createdAt', type: 'date', nullable: false, description: 'Creation date' }
          ],
          constraints: [
            { type: 'unique', field: 'email' }
          ]
        }
      }
    ];

    for (const { name, schema } of commonSchemas) {
      this.schemas.set(name, schema);
      this.schemaVersions.set(name, [{
        version: '1.0.0',
        schema,
        createdAt: new Date(),
        createdBy: 'system',
        description: `Built-in ${name} schema`
      }]);
    }
  }
}

interface SchemaSummary {
  name: string;
  latestVersion: string;
  fieldCount: number;
  constraintCount: number;
  createdAt?: Date;
  updatedAt?: Date;
  description?: string;
}

interface SchemaHistoryEntry {
  action: 'create' | 'update' | 'delete';
  version: string;
  timestamp: Date;
  user: string;
  description: string;
}

interface SchemaComparison {
  schemasCompared: string[];
  differences: SchemaDifference[];
  breakingChanges: SchemaDifference[];
  compatibleChanges: SchemaDifference[];
}

interface SchemaDifference {
  type: string;
  field?: string;
  oldValue?: any;
  newValue?: any;
  description: string;
  severity?: 'breaking' | 'compatible';
}