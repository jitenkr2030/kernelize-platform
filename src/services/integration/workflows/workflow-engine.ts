/**
 * KERNELIZE Platform - Workflow Automation Engine
 * Comprehensive workflow orchestration with multi-step processes, conditional logic, and automation
 */

import { BasePlugin } from '../plugins/base-plugin.js';
import {
  PluginMetadata,
  PluginCategory,
  PluginConfig,
  ValidationResult,
  WorkflowContext,
  ExecutionResult,
  ExecutionError,
  ExecutionLog,
  CronExpression
} from '../plugins/plugin-types.js';

interface WorkflowStep {
  id: string;
  name: string;
  type: 'task' | 'condition' | 'parallel' | 'loop' | 'sub-workflow';
  pluginId: string;
  action: string;
  config: Record<string, any>;
  conditions?: WorkflowCondition[];
  next?: string[];
  onError?: 'fail' | 'continue' | 'retry';
  retry?: RetryConfig;
  timeout?: number;
  dependencies?: string[];
}

interface WorkflowCondition {
  field: string;
  operator: 'equals' | 'not_equals' | 'greater_than' | 'less_than' | 'contains' | 'exists';
  value: any;
  logicalOperator?: 'AND' | 'OR';
}

interface RetryConfig {
  maxAttempts: number;
  backoffMultiplier: number;
  initialDelay: number;
  maxDelay: number;
}

interface WorkflowDefinition {
  id: string;
  name: string;
  description: string;
  version: string;
  triggers: WorkflowTrigger[];
  steps: WorkflowStep[];
  variables: Record<string, any>;
  errorHandling: ErrorHandlingConfig;
  timeout: number;
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
}

interface WorkflowTrigger {
  type: 'schedule' | 'webhook' | 'event' | 'manual';
  config: any;
  enabled: boolean;
}

interface ErrorHandlingConfig {
  strategy: 'fail-fast' | 'continue' | 'compensate';
  maxRetries: number;
  compensationSteps?: string[];
}

interface WorkflowExecution {
  workflowId: string;
  executionId: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'timeout';
  startedAt: Date;
  completedAt?: Date;
  currentStep?: string;
  context: WorkflowContext;
  results: Map<string, any>;
  errors: ExecutionError[];
  logs: ExecutionLog[];
}

export class WorkflowAutomationEngine extends BasePlugin {
  private workflows: Map<string, WorkflowDefinition> = new Map();
  private executions: Map<string, WorkflowExecution> = new Map();
  private scheduledExecutions: Map<string, CronJob> = new Map();
  private pluginManager: any; // PluginManager instance

  constructor() {
    const metadata: PluginMetadata = {
      id: 'workflow-automation',
      name: 'Workflow Automation Engine',
      version: '1.0.0',
      description: 'Comprehensive workflow automation and orchestration engine',
      author: 'KERNELIZE Team',
      category: PluginCategory.WORKFLOW,
      keywords: ['workflow', 'automation', 'orchestration', 'pipeline'],
      license: 'MIT',
      createdAt: new Date(),
      updatedAt: new Date(),
      downloads: 0,
      rating: 0,
      compatibility: {
        minPlatformVersion: '1.0.0',
        supportedNodes: ['api', 'data-pipeline', 'analytics']
      }
    };

    super(metadata);
  }

  async initialize(config: PluginConfig): Promise<void> {
    this.pluginManager = config.pluginManager;
    this.loadExistingWorkflows();
  }

  async execute(input: any): Promise<any> {
    const { action, params } = input;

    switch (action) {
      case 'create_workflow':
        return await this.createWorkflow(params.workflow);
      
      case 'execute_workflow':
        return await this.executeWorkflow(params.workflowId, params.context);
      
      case 'execute_async':
        return await this.executeWorkflowAsync(params.workflowId, params.context);
      
      case 'cancel_execution':
        return await this.cancelExecution(params.executionId);
      
      case 'get_workflow':
        return await this.getWorkflow(params.workflowId);
      
      case 'list_workflows':
        return await this.listWorkflows(params.filter);
      
      case 'update_workflow':
        return await this.updateWorkflow(params.workflowId, params.workflow);
      
      case 'delete_workflow':
        return await this.deleteWorkflow(params.workflowId);
      
      case 'get_execution_status':
        return await this.getExecutionStatus(params.executionId);
      
      case 'list_executions':
        return await this.listExecutions(params.workflowId, params.status, params.limit);
      
      case 'schedule_workflow':
        return await this.scheduleWorkflow(params.workflowId, params.schedule);
      
      case 'unschedule_workflow':
        return await this.unscheduleWorkflow(params.scheduleId);
      
      case 'trigger_webhook':
        return await this.triggerWebhook(params.workflowId, params.payload);
      
      case 'validate_workflow':
        return await this.validateWorkflow(params.workflow);
      
      case 'export_workflows':
        return await this.exportWorkflows(params.workflowIds);
      
      case 'import_workflows':
        return await this.importWorkflows(params.workflows);
      
      default:
        throw new Error(`Unknown action: ${action}`);
    }
  }

  async createWorkflow(workflow: Partial<WorkflowDefinition>): Promise<WorkflowDefinition> {
    const fullWorkflow: WorkflowDefinition = {
      id: workflow.id || `workflow-${Date.now()}`,
      name: workflow.name || 'Untitled Workflow',
      description: workflow.description || '',
      version: workflow.version || '1.0.0',
      triggers: workflow.triggers || [],
      steps: workflow.steps || [],
      variables: workflow.variables || {},
      errorHandling: workflow.errorHandling || {
        strategy: 'fail-fast',
        maxRetries: 3
      },
      timeout: workflow.timeout || 3600000, // 1 hour
      tags: workflow.tags || [],
      createdAt: new Date(),
      updatedAt: new Date()
    };

    // Validate workflow
    const validation = await this.validateWorkflowDefinition(fullWorkflow);
    if (!validation.valid) {
      throw new Error(`Invalid workflow: ${validation.errors.map(e => e.message).join(', ')}`);
    }

    this.workflows.set(fullWorkflow.id, fullWorkflow);
    console.log(`Workflow ${fullWorkflow.id} created successfully`);
    
    return fullWorkflow;
  }

  async executeWorkflow(workflowId: string, context: WorkflowContext): Promise<ExecutionResult> {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) {
      throw new Error(`Workflow not found: ${workflowId}`);
    }

    const execution: WorkflowExecution = {
      workflowId,
      executionId: `exec-${Date.now()}`,
      status: 'pending',
      startedAt: new Date(),
      context,
      results: new Map(),
      errors: [],
      logs: []
    };

    this.executions.set(execution.executionId, execution);
    
    try {
      execution.status = 'running';
      const result = await this.executeWorkflowInternal(workflow, execution);
      execution.status = 'completed';
      execution.completedAt = new Date();
      
      return result;
      
    } catch (error) {
      execution.status = 'failed';
      execution.completedAt = new Date();
      execution.errors.push({
        step: 'workflow',
        message: error.message,
        code: 'WORKFLOW_FAILED',
        timestamp: new Date()
      });
      
      throw error;
    }
  }

  async executeWorkflowAsync(workflowId: string, context: WorkflowContext): Promise<string> {
    // Execute workflow in background
    setImmediate(async () => {
      try {
        await this.executeWorkflow(workflowId, context);
      } catch (error) {
        console.error(`Async workflow execution failed: ${error.message}`);
      }
    });
    
    return `exec-${Date.now()}`;
  }

  private async executeWorkflowInternal(workflow: WorkflowDefinition, execution: WorkflowExecution): Promise<ExecutionResult> {
    const startTime = Date.now();
    const executedSteps = new Set<string>();
    const stepResults = new Map<string, any>();

    try {
      // Build dependency graph
      const dependencyGraph = this.buildDependencyGraph(workflow.steps);
      
      // Execute steps in dependency order
      const executionOrder = this.topologicalSort(dependencyGraph);
      
      for (const stepId of executionOrder) {
        if (executedSteps.has(stepId)) continue;
        
        const step = workflow.steps.find(s => s.id === stepId);
        if (!step) continue;

        execution.currentStep = stepId;
        
        // Check conditions
        if (step.conditions && !this.evaluateConditions(step.conditions, stepResults)) {
          this.log(execution, 'info', `Step ${stepId} skipped due to conditions`);
          continue;
        }

        try {
          const result = await this.executeStep(step, execution, stepResults);
          stepResults.set(stepId, result);
          executedSteps.add(stepId);
          
          this.log(execution, 'info', `Step ${stepId} completed successfully`);
          
        } catch (error) {
          await this.handleStepError(step, error, execution, stepResults);
        }
      }

      const duration = Date.now() - startTime;
      
      return {
        success: execution.errors.length === 0,
        executionId: execution.executionId,
        startedAt: execution.startedAt,
        completedAt: new Date(),
        duration,
        output: Object.fromEntries(stepResults),
        errors: execution.errors,
        logs: execution.logs
      };
      
    } catch (error) {
      execution.errors.push({
        step: 'workflow',
        message: error.message,
        code: 'WORKFLOW_FAILED',
        timestamp: new Date()
      });
      
      throw error;
    }
  }

  private async executeStep(step: WorkflowStep, execution: WorkflowExecution, stepResults: Map<string, any>): Promise<any> {
    const plugin = this.pluginManager?.getPlugin(step.pluginId);
    if (!plugin) {
      throw new Error(`Plugin not found: ${step.pluginId}`);
    }

    // Merge step config with workflow variables and previous results
    const input = {
      action: step.action,
      params: {
        ...step.config,
        workflowId: execution.workflowId,
        executionId: execution.executionId,
        variables: execution.context.variables,
        previousResults: Object.fromEntries(stepResults),
        stepId: step.id
      }
    };

    // Execute with timeout
    const timeout = step.timeout || 300000; // 5 minutes default
    const result = await this.executeWithTimeout(
      () => plugin.executeWithMonitoring(input),
      timeout
    );

    return result;
  }

  private async executeWithTimeout<T>(operation: () => Promise<T>, timeoutMs: number): Promise<T> {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        reject(new Error(`Operation timed out after ${timeoutMs}ms`));
      }, timeoutMs);

      operation()
        .then(result => {
          clearTimeout(timeout);
          resolve(result);
        })
        .catch(error => {
          clearTimeout(timeout);
          reject(error);
        });
    });
  }

  private async handleStepError(step: WorkflowStep, error: Error, execution: WorkflowExecution, stepResults: Map<string, any>): Promise<void> {
    const errorEntry: ExecutionError = {
      step: step.id,
      message: error.message,
      code: 'STEP_FAILED',
      stack: error.stack,
      timestamp: new Date()
    };

    execution.errors.push(errorEntry);
    
    switch (step.onError || 'fail') {
      case 'continue':
        this.log(execution, 'warn', `Step ${step.id} failed but continuing: ${error.message}`);
        break;
        
      case 'retry':
        await this.retryStep(step, error, execution, stepResults);
        break;
        
      case 'fail':
      default:
        this.log(execution, 'error', `Step ${step.id} failed: ${error.message}`);
        if (execution.context.variables['fail-fast'] !== false) {
          throw error;
        }
        break;
    }
  }

  private async retryStep(step: WorkflowStep, error: Error, execution: WorkflowExecution, stepResults: Map<string, any>): Promise<void> {
    const retryConfig = step.retry || { maxAttempts: 3, backoffMultiplier: 2, initialDelay: 1000, maxDelay: 60000 };
    let attempt = 1;
    let delay = retryConfig.initialDelay;

    while (attempt <= retryConfig.maxAttempts) {
      try {
        this.log(execution, 'info', `Retrying step ${step.id}, attempt ${attempt}/${retryConfig.maxAttempts}`);
        
        const result = await this.executeStep(step, execution, stepResults);
        stepResults.set(step.id, result);
        return;
        
      } catch (retryError) {
        attempt++;
        if (attempt > retryConfig.maxAttempts) {
          this.log(execution, 'error', `Step ${step.id} failed after ${retryConfig.maxAttempts} attempts`);
          throw retryError;
        }
        
        await this.sleep(delay);
        delay = Math.min(delay * retryConfig.backoffMultiplier, retryConfig.maxDelay);
      }
    }
  }

  private buildDependencyGraph(steps: WorkflowStep[]): Map<string, string[]> {
    const graph = new Map<string, string[]>();
    
    for (const step of steps) {
      graph.set(step.id, []);
      
      if (step.dependencies) {
        for (const dep of step.dependencies) {
          if (!graph.has(dep)) {
            graph.set(dep, []);
          }
          graph.get(dep)!.push(step.id);
        }
      }
    }
    
    return graph;
  }

  private topologicalSort(graph: Map<string, string[]>): string[] {
    const visited = new Set<string>();
    const stack: string[] = [];

    const visit = (node: string) => {
      if (visited.has(node)) return;
      visited.add(node);

      const dependencies = graph.get(node) || [];
      for (const dep of dependencies) {
        visit(dep);
      }

      stack.push(node);
    };

    for (const node of graph.keys()) {
      visit(node);
    }

    return stack.reverse();
  }

  private evaluateConditions(conditions: WorkflowCondition[], stepResults: Map<string, any>): boolean {
    if (conditions.length === 0) return true;

    let result = true;
    let currentOperator = 'AND';

    for (const condition of conditions) {
      const value = this.getConditionValue(condition.field, stepResults);
      const conditionResult = this.evaluateCondition(value, condition.operator, condition.value);

      if (currentOperator === 'AND') {
        result = result && conditionResult;
      } else {
        result = result || conditionResult;
      }

      currentOperator = condition.logicalOperator || 'AND';
    }

    return result;
  }

  private getConditionValue(field: string, stepResults: Map<string, any>): any {
    if (field.includes('.')) {
      const [stepId, property] = field.split('.');
      const stepResult = stepResults.get(stepId);
      return stepResult ? stepResult[property] : undefined;
    }
    
    return stepResults.get(field);
  }

  private evaluateCondition(value: any, operator: string, expected: any): boolean {
    switch (operator) {
      case 'equals':
        return value === expected;
      case 'not_equals':
        return value !== expected;
      case 'greater_than':
        return value > expected;
      case 'less_than':
        return value < expected;
      case 'contains':
        return String(value).includes(String(expected));
      case 'exists':
        return value !== undefined && value !== null;
      default:
        return false;
    }
  }

  private log(execution: WorkflowExecution, level: 'debug' | 'info' | 'warn' | 'error', message: string, metadata?: Record<string, any>): void {
    const logEntry: ExecutionLog = {
      level,
      message,
      timestamp: new Date(),
      metadata: {
        step: execution.currentStep,
        executionId: execution.executionId,
        ...metadata
      }
    };

    execution.logs.push(logEntry);
    console.log(`[${level.toUpperCase()}] ${message}`);
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  async cancelExecution(executionId: string): Promise<void> {
    const execution = this.executions.get(executionId);
    if (!execution) {
      throw new Error(`Execution not found: ${executionId}`);
    }

    execution.status = 'cancelled';
    execution.completedAt = new Date();
    
    console.log(`Execution ${executionId} cancelled`);
  }

  async getWorkflow(workflowId: string): Promise<WorkflowDefinition | null> {
    return this.workflows.get(workflowId) || null;
  }

  async listWorkflows(filter?: { tags?: string[], status?: string }): Promise<WorkflowDefinition[]> {
    let workflows = Array.from(this.workflows.values());

    if (filter?.tags) {
      workflows = workflows.filter(w => 
        filter.tags!.some(tag => w.tags.includes(tag))
      );
    }

    return workflows;
  }

  async updateWorkflow(workflowId: string, updates: Partial<WorkflowDefinition>): Promise<WorkflowDefinition> {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) {
      throw new Error(`Workflow not found: ${workflowId}`);
    }

    const updatedWorkflow = {
      ...workflow,
      ...updates,
      id: workflowId, // Preserve ID
      updatedAt: new Date()
    };

    const validation = await this.validateWorkflowDefinition(updatedWorkflow);
    if (!validation.valid) {
      throw new Error(`Invalid workflow update: ${validation.errors.map(e => e.message).join(', ')}`);
    }

    this.workflows.set(workflowId, updatedWorkflow);
    return updatedWorkflow;
  }

  async deleteWorkflow(workflowId: string): Promise<void> {
    if (!this.workflows.has(workflowId)) {
      throw new Error(`Workflow not found: ${workflowId}`);
    }

    this.workflows.delete(workflowId);
    console.log(`Workflow ${workflowId} deleted`);
  }

  async getExecutionStatus(executionId: string): Promise<WorkflowExecution | null> {
    return this.executions.get(executionId) || null;
  }

  async listExecutions(workflowId?: string, status?: string, limit: number = 100): Promise<WorkflowExecution[]> {
    let executions = Array.from(this.executions.values());

    if (workflowId) {
      executions = executions.filter(e => e.workflowId === workflowId);
    }

    if (status) {
      executions = executions.filter(e => e.status === status);
    }

    return executions
      .sort((a, b) => b.startedAt.getTime() - a.startedAt.getTime())
      .slice(0, limit);
  }

  async scheduleWorkflow(workflowId: string, schedule: CronExpression): Promise<string> {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) {
      throw new Error(`Workflow not found: ${workflowId}`);
    }

    const scheduleId = `schedule-${Date.now()}`;
    
    // In real implementation, would use cron library
    console.log(`Scheduling workflow ${workflowId} with expression: ${schedule.expression}`);
    
    const cronJob: CronJob = {
      id: scheduleId,
      workflowId,
      expression: schedule,
      nextRun: new Date(Date.now() + 60000), // Next minute for demo
      lastRun: schedule.lastRun,
      enabled: schedule.enabled
    };

    this.scheduledExecutions.set(scheduleId, cronJob);
    return scheduleId;
  }

  async unscheduleWorkflow(scheduleId: string): Promise<void> {
    this.scheduledExecutions.delete(scheduleId);
    console.log(`Workflow schedule ${scheduleId} removed`);
  }

  async triggerWebhook(workflowId: string, payload: any): Promise<string> {
    const workflow = this.workflows.get(workflowId);
    if (!workflow) {
      throw new Error(`Workflow not found: ${workflowId}`);
    }

    const context: WorkflowContext = {
      workflowId,
      executionId: `webhook-${Date.now()}`,
      userId: 'webhook-trigger',
      variables: { payload },
      state: {},
      startedAt: new Date()
    };

    return await this.executeWorkflowAsync(workflowId, context);
  }

  async validateWorkflow(workflow: Partial<WorkflowDefinition>): Promise<ValidationResult> {
    const fullWorkflow: WorkflowDefinition = {
      id: workflow.id || `temp-${Date.now()}`,
      name: workflow.name || 'Temporary',
      description: workflow.description || '',
      version: workflow.version || '1.0.0',
      triggers: workflow.triggers || [],
      steps: workflow.steps || [],
      variables: workflow.variables || {},
      errorHandling: workflow.errorHandling || { strategy: 'fail-fast', maxRetries: 3 },
      timeout: workflow.timeout || 3600000,
      tags: workflow.tags || [],
      createdAt: new Date(),
      updatedAt: new Date()
    };

    return await this.validateWorkflowDefinition(fullWorkflow);
  }

  private async validateWorkflowDefinition(workflow: WorkflowDefinition): Promise<ValidationResult> {
    const errors: any[] = [];
    const warnings: any[] = [];

    // Basic validation
    if (!workflow.name) {
      errors.push({ field: 'name', code: 'REQUIRED', message: 'Workflow name is required' });
    }

    if (!workflow.steps || workflow.steps.length === 0) {
      errors.push({ field: 'steps', code: 'REQUIRED', message: 'Workflow must have at least one step' });
    }

    // Validate steps
    const stepIds = new Set<string>();
    for (const step of workflow.steps) {
      if (!step.id) {
        errors.push({ field: 'step.id', code: 'REQUIRED', message: 'Step ID is required' });
      } else if (stepIds.has(step.id)) {
        errors.push({ field: 'step.id', code: 'DUPLICATE', message: `Step ID ${step.id} is duplicated` });
      } else {
        stepIds.add(step.id);
      }

      if (!step.name) {
        errors.push({ field: 'step.name', code: 'REQUIRED', message: 'Step name is required' });
      }

      if (!step.pluginId) {
        errors.push({ field: 'step.pluginId', code: 'REQUIRED', message: 'Step plugin ID is required' });
      }

      if (!step.action) {
        errors.push({ field: 'step.action', code: 'REQUIRED', message: 'Step action is required' });
      }
    }

    // Validate dependencies
    for (const step of workflow.steps) {
      if (step.dependencies) {
        for (const dep of step.dependencies) {
          if (!stepIds.has(dep)) {
            errors.push({ 
              field: 'step.dependencies', 
              code: 'INVALID', 
              message: `Dependency ${dep} not found for step ${step.id}` 
            });
          }
        }
      }
    }

    // Check for circular dependencies
    try {
      const dependencyGraph = this.buildDependencyGraph(workflow.steps);
      this.topologicalSort(dependencyGraph);
    } catch (error) {
      errors.push({ 
        field: 'steps', 
        code: 'CIRCULAR_DEPENDENCY', 
        message: 'Workflow has circular dependencies' 
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  async exportWorkflows(workflowIds?: string[]): Promise<any> {
    const workflows = workflowIds 
      ? workflowIds.map(id => this.workflows.get(id)).filter(Boolean)
      : Array.from(this.workflows.values());

    return {
      version: '1.0.0',
      exportedAt: new Date().toISOString(),
      workflows: workflows.map(w => ({
        ...w,
        executionStats: this.getWorkflowStats(w.id)
      }))
    };
  }

  async importWorkflows(importData: any): Promise<string[]> {
    const importedIds: string[] = [];

    for (const workflowData of importData.workflows) {
      try {
        const workflow = await this.createWorkflow(workflowData);
        importedIds.push(workflow.id);
      } catch (error) {
        console.error(`Failed to import workflow ${workflowData.name}:`, error);
      }
    }

    return importedIds;
  }

  private getWorkflowStats(workflowId: string): any {
    const executions = Array.from(this.executions.values())
      .filter(e => e.workflowId === workflowId);

    const completed = executions.filter(e => e.status === 'completed');
    const failed = executions.filter(e => e.status === 'failed');

    return {
      totalExecutions: executions.length,
      successfulExecutions: completed.length,
      failedExecutions: failed.length,
      successRate: executions.length > 0 ? (completed.length / executions.length * 100).toFixed(2) : 0,
      averageDuration: completed.length > 0 
        ? (completed.reduce((sum, e) => sum + (e.completedAt!.getTime() - e.startedAt.getTime()), 0) / completed.length)
        : 0
    };
  }

  private loadExistingWorkflows(): void {
    // In real implementation, would load from persistent storage
    console.log('Loading existing workflows...');
  }

  async shutdown(): Promise<void> {
    // Cancel all running executions
    for (const execution of this.executions.values()) {
      if (execution.status === 'running') {
        await this.cancelExecution(execution.executionId);
      }
    }

    // Clear scheduled executions
    this.scheduledExecutions.clear();
  }

  validate(): ValidationResult {
    const errors: any[] = [];
    const warnings: any[] = [];

    if (!this.pluginManager) {
      warnings.push({
        field: 'pluginManager',
        code: 'MISSING',
        message: 'Plugin manager not configured'
      });
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  getStats() {
    const baseStats = super.getStats();
    return {
      ...baseStats,
      workflowsCount: this.workflows.size,
      activeExecutions: Array.from(this.executions.values()).filter(e => e.status === 'running').length,
      scheduledExecutions: this.scheduledExecutions.size
    };
  }
}

interface CronJob {
  id: string;
  workflowId: string;
  expression: CronExpression;
  nextRun: Date;
  lastRun?: Date;
  enabled: boolean;
}