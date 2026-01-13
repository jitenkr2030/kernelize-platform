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

import { Request, Response } from 'express';
import WebSocket from 'ws';
import { v4 as uuidv4 } from 'uuid';

interface Dashboard {
  id: string;
  name: string;
  type: 'executive' | 'operational' | 'technical' | 'custom';
  widgets: DashboardWidget[];
  layout: DashboardLayout;
  permissions: {
    owner: string;
    viewers: string[];
    editors: string[];
  };
  createdAt: Date;
  updatedAt: Date;
  isPublic: boolean;
}

interface DashboardWidget {
  id: string;
  type: 'chart' | 'metric' | 'table' | 'gauge' | 'heatmap';
  title: string;
  dataSource: string;
  configuration: any;
  position: { x: number; y: number; width: number; height: number };
  refreshInterval: number; // seconds
}

interface DashboardLayout {
  columns: number;
  rows: number;
  responsive: boolean;
}

interface CostAnalysis {
  id: string;
  userId: string;
  period: {
    start: Date;
    end: Date;
  };
  breakdown: {
    storage: CostBreakdown;
    bandwidth: CostBreakdown;
    processing: CostBreakdown;
    requests: CostBreakdown;
  };
  trends: CostTrend[];
  projections: {
    next30Days: number;
    next90Days: number;
    nextYear: number;
  };
  optimizations: CostOptimization[];
  totalCost: number;
  budget?: {
    allocated: number;
    spent: number;
    remaining: number;
    alerts: BudgetAlert[];
  };
}

interface CostBreakdown {
  amount: number;
  unit: string;
  unitCost: number;
  quantity: number;
  growth: number; // percentage
  category: string;
}

interface CostTrend {
  date: Date;
  amount: number;
  category: string;
}

interface CostOptimization {
  id: string;
  type: 'storage' | 'bandwidth' | 'processing' | 'requests';
  description: string;
  potentialSavings: number;
  effort: 'low' | 'medium' | 'high';
  impact: 'low' | 'medium' | 'high';
  implemented: boolean;
  confidence: number;
}

interface BudgetAlert {
  threshold: number;
  message: string;
  severity: 'info' | 'warning' | 'critical';
  triggered: boolean;
}

interface PerformanceBenchmark {
  id: string;
  category: 'compression' | 'decompression' | 'batch_processing' | 'real_time';
  metric: string;
  currentValue: number;
  targetValue: number;
  industryAverage: number;
  bestInClass: number;
  percentile: number;
  trend: 'improving' | 'declining' | 'stable';
  lastUpdated: Date;
}

interface TrendAnalysis {
  id: string;
  type: 'usage' | 'performance' | 'cost' | 'quality';
  timeframe: string;
  dataPoints: TrendDataPoint[];
  insights: TrendInsight[];
  predictions: TrendPrediction[];
  anomalies: TrendAnomaly[];
}

interface TrendDataPoint {
  timestamp: Date;
  value: number;
  metadata?: any;
}

interface TrendInsight {
  type: 'pattern' | 'correlation' | 'seasonality' | 'outlier';
  description: string;
  confidence: number;
  impact: 'positive' | 'negative' | 'neutral';
}

interface TrendPrediction {
  timeframe: string;
  predictedValue: number;
  confidence: number;
  factors: string[];
}

interface TrendAnomaly {
  timestamp: Date;
  value: number;
  expectedValue: number;
  deviation: number;
  severity: 'low' | 'medium' | 'high';
  reason: string;
}

class BusinessIntelligenceService {
  private dashboards: Map<string, Dashboard[]> = new Map();
  private costAnalyses: Map<string, CostAnalysis[]> = new Map();
  private benchmarks: Map<string, PerformanceBenchmark[]> = new Map();
  private trendAnalyses: Map<string, TrendAnalysis[]> = new Map();
  private wsClients: Set<WebSocket> = new Set();

  constructor() {
    this.initializeBusinessIntelligence();
  }

  private initializeBusinessIntelligence(): void {
    this.generateSampleDashboards();
    this.generateSampleCostAnalyses();
    this.generateSampleBenchmarks();
    this.generateSampleTrendAnalyses();
  }

  private generateSampleDashboards(): void {
    const dashboardTemplates = [
      {
        name: 'Executive Overview',
        type: 'executive' as const,
        widgets: this.getExecutiveWidgets()
      },
      {
        name: 'Operations Dashboard',
        type: 'operational' as const,
        widgets: this.getOperationalWidgets()
      },
      {
        name: 'Technical Performance',
        type: 'technical' as const,
        widgets: this.getTechnicalWidgets()
      },
      {
        name: 'Cost Management',
        type: 'custom' as const,
        widgets: this.getCostManagementWidgets()
      }
    ];

    dashboardTemplates.forEach(template => {
      const dashboard: Dashboard = {
        id: uuidv4(),
        name: template.name,
        type: template.type,
        widgets: template.widgets,
        layout: {
          columns: 12,
          rows: 8,
          responsive: true
        },
        permissions: {
          owner: 'admin',
          viewers: [],
          editors: []
        },
        createdAt: new Date(),
        updatedAt: new Date(),
        isPublic: false
      };

      if (!this.dashboards.has('admin')) {
        this.dashboards.set('admin', []);
      }
      this.dashboards.get('admin')!.push(dashboard);
    });
  }

  private getExecutiveWidgets(): DashboardWidget[] {
    return [
      {
        id: uuidv4(),
        type: 'metric',
        title: 'Total Cost Savings',
        dataSource: 'cost_analysis',
        configuration: { format: 'currency', period: '30d' },
        position: { x: 0, y: 0, width: 3, height: 2 },
        refreshInterval: 3600
      },
      {
        id: uuidv4(),
        type: 'chart',
        title: 'Monthly ROI Trend',
        dataSource: 'roi_trends',
        configuration: { chartType: 'line', timeRange: '12m' },
        position: { x: 3, y: 0, width: 6, height: 4 },
        refreshInterval: 3600
      },
      {
        id: uuidv4(),
        type: 'gauge',
        title: 'System Efficiency',
        dataSource: 'performance_benchmarks',
        configuration: { metric: 'overall_efficiency' },
        position: { x: 9, y: 0, width: 3, height: 2 },
        refreshInterval: 1800
      },
      {
        id: uuidv4(),
        type: 'chart',
        title: 'Usage by File Type',
        dataSource: 'usage_patterns',
        configuration: { chartType: 'pie' },
        position: { x: 0, y: 4, width: 4, height: 4 },
        refreshInterval: 1800
      },
      {
        id: uuidv4(),
        type: 'table',
        title: 'Top Performance Metrics',
        dataSource: 'performance_benchmarks',
        configuration: { columns: ['metric', 'current', 'target', 'trend'] },
        position: { x: 4, y: 4, width: 8, height: 4 },
        refreshInterval: 3600
      }
    ];
  }

  private getOperationalWidgets(): DashboardWidget[] {
    return [
      {
        id: uuidv4(),
        type: 'chart',
        title: 'Processing Volume (24h)',
        dataSource: 'real_time_metrics',
        configuration: { chartType: 'area', timeRange: '24h' },
        position: { x: 0, y: 0, width: 8, height: 3 },
        refreshInterval: 300
      },
      {
        id: uuidv4(),
        type: 'metric',
        title: 'Success Rate',
        dataSource: 'quality_metrics',
        configuration: { format: 'percentage' },
        position: { x: 8, y: 0, width: 2, height: 2 },
        refreshInterval: 300
      },
      {
        id: uuidv4(),
        type: 'metric',
        title: 'Avg Processing Time',
        dataSource: 'performance_metrics',
        configuration: { format: 'time', unit: 'seconds' },
        position: { x: 10, y: 0, width: 2, height: 2 },
        refreshInterval: 300
      },
      {
        id: uuidv4(),
        type: 'heatmap',
        title: 'Usage Heatmap',
        dataSource: 'usage_patterns',
        configuration: { timeDimension: 'hour', valueDimension: 'actions' },
        position: { x: 0, y: 3, width: 6, height: 3 },
        refreshInterval: 1800
      },
      {
        id: uuidv4(),
        type: 'chart',
        title: 'Error Rate Trend',
        dataSource: 'error_metrics',
        configuration: { chartType: 'line', timeRange: '7d' },
        position: { x: 6, y: 3, width: 6, height: 3 },
        refreshInterval: 1800
      }
    ];
  }

  private getTechnicalWidgets(): DashboardWidget[] {
    return [
      {
        id: uuidv4(),
        type: 'chart',
        title: 'Algorithm Performance Comparison',
        dataSource: 'compression_analytics',
        configuration: { chartType: 'bar', groupBy: 'algorithm' },
        position: { x: 0, y: 0, width: 6, height: 4 },
        refreshInterval: 1800
      },
      {
        id: uuidv4(),
        type: 'chart',
        title: 'Quality Score Distribution',
        dataSource: 'quality_metrics',
        configuration: { chartType: 'histogram', bins: 10 },
        position: { x: 6, y: 0, width: 6, height: 4 },
        refreshInterval: 1800
      },
      {
        id: uuidv4(),
        type: 'gauge',
        title: 'Compression Efficiency',
        dataSource: 'compression_metrics',
        configuration: { metric: 'efficiency_score' },
        position: { x: 0, y: 4, width: 3, height: 3 },
        refreshInterval: 3600
      },
      {
        id: uuidv4(),
        type: 'table',
        title: 'System Health Indicators',
        dataSource: 'system_metrics',
        configuration: { columns: ['component', 'status', 'last_check'] },
        position: { x: 3, y: 4, width: 9, height: 3 },
        refreshInterval: 600
      }
    ];
  }

  private getCostManagementWidgets(): DashboardWidget[] {
    return [
      {
        id: uuidv4(),
        type: 'chart',
        title: 'Cost Breakdown',
        dataSource: 'cost_analysis',
        configuration: { chartType: 'donut', groupBy: 'category' },
        position: { x: 0, y: 0, width: 6, height: 4 },
        refreshInterval: 3600
      },
      {
        id: uuidv4(),
        type: 'chart',
        title: 'Cost Trend (6 months)',
        dataSource: 'cost_trends',
        configuration: { chartType: 'line', timeRange: '6m' },
        position: { x: 6, y: 0, width: 6, height: 4 },
        refreshInterval: 3600
      },
      {
        id: uuidv4(),
        type: 'metric',
        title: 'Monthly Budget Status',
        dataSource: 'budget_tracking',
        configuration: { format: 'percentage', showRemaining: true },
        position: { x: 0, y: 4, width: 3, height: 2 },
        refreshInterval: 1800
      },
      {
        id: uuidv4(),
        type: 'table',
        title: 'Cost Optimization Opportunities',
        dataSource: 'cost_optimizations',
        configuration: { columns: ['type', 'potential_savings', 'effort', 'impact'] },
        position: { x: 3, y: 4, width: 9, height: 4 },
        refreshInterval: 3600
      }
    ];
  }

  private generateSampleCostAnalyses(): void {
    const users = ['user_1', 'user_2', 'user_3', 'admin'];
    const periods = ['30d', '90d', '1y'];

    users.forEach(userId => {
      periods.forEach(period => {
        const analysis = this.generateCostAnalysis(userId, period);
        
        if (!this.costAnalyses.has(userId)) {
          this.costAnalyses.set(userId, []);
        }
        this.costAnalyses.get(userId)!.push(analysis);
      });
    });
  }

  private generateCostAnalysis(userId: string, period: string): CostAnalysis {
    const now = new Date();
    const startDate = this.getDateRangeStart(period, now);
    
    // Generate sample cost data
    const baseCost = Math.random() * 1000 + 200; // $200-$1200 base cost
    const storageCost = baseCost * (0.3 + Math.random() * 0.2);
    const bandwidthCost = baseCost * (0.25 + Math.random() * 0.25);
    const processingCost = baseCost * (0.2 + Math.random() * 0.2);
    const requestsCost = baseCost * (0.15 + Math.random() * 0.15);

    const totalCost = storageCost + bandwidthCost + processingCost + requestsCost;
    const savingsRate = Math.random() * 0.4 + 0.2; // 20-60% savings

    const optimizations: CostOptimization[] = [
      {
        id: uuidv4(),
        type: 'storage',
        description: 'Implement data deduplication',
        potentialSavings: storageCost * 0.15,
        effort: 'medium',
        impact: 'high',
        implemented: Math.random() > 0.7,
        confidence: 0.85
      },
      {
        id: uuidv4(),
        type: 'bandwidth',
        description: 'Optimize CDN configuration',
        potentialSavings: bandwidthCost * 0.2,
        effort: 'low',
        impact: 'medium',
        implemented: Math.random() > 0.5,
        confidence: 0.92
      },
      {
        id: uuidv4(),
        type: 'processing',
        description: 'Use spot instances for batch processing',
        potentialSavings: processingCost * 0.3,
        effort: 'high',
        impact: 'high',
        implemented: Math.random() > 0.8,
        confidence: 0.78
      }
    ];

    const trendData = this.generateCostTrends(startDate, now);

    return {
      id: uuidv4(),
      userId,
      period: { start: startDate, end: now },
      breakdown: {
        storage: {
          amount: storageCost,
          unit: 'GB',
          unitCost: 0.023,
          quantity: storageCost / 0.023,
          growth: (Math.random() - 0.5) * 20, // -10% to +10%
          category: 'storage'
        },
        bandwidth: {
          amount: bandwidthCost,
          unit: 'GB',
          unitCost: 0.09,
          quantity: bandwidthCost / 0.09,
          growth: (Math.random() - 0.5) * 30,
          category: 'bandwidth'
        },
        processing: {
          amount: processingCost,
          unit: 'hours',
          unitCost: 0.05,
          quantity: processingCost / 0.05,
          growth: (Math.random() - 0.5) * 25,
          category: 'processing'
        },
        requests: {
          amount: requestsCost,
          unit: 'requests',
          unitCost: 0.001,
          quantity: requestsCost / 0.001,
          growth: (Math.random() - 0.5) * 40,
          category: 'requests'
        }
      },
      trends: trendData,
      projections: {
        next30Days: totalCost * 1.1,
        next90Days: totalCost * 3.5,
        nextYear: totalCost * 12 * 1.05
      },
      optimizations,
      totalCost,
      budget: {
        allocated: totalCost * 1.2, // 20% buffer
        spent: totalCost * (0.8 + Math.random() * 0.3),
        remaining: 0,
        alerts: this.generateBudgetAlerts(totalCost * 1.2)
      }
    };
  }

  private generateCostTrends(start: Date, end: Date): CostTrend[] {
    const trends: CostTrend[] = [];
    const categories = ['storage', 'bandwidth', 'processing', 'requests'];
    const days = Math.floor((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
    
    for (let i = 0; i <= days; i += Math.max(1, Math.floor(days / 30))) {
      const date = new Date(start.getTime() + i * 24 * 60 * 60 * 1000);
      categories.forEach(category => {
        trends.push({
          date,
          amount: Math.random() * 50 + 10,
          category
        });
      });
    }
    
    return trends;
  }

  private generateBudgetAlerts(allocated: number): BudgetAlert[] {
    const alerts: BudgetAlert[] = [
      {
        threshold: 80,
        message: 'Budget usage exceeded 80%',
        severity: 'warning',
        triggered: false
      },
      {
        threshold: 95,
        message: 'Budget usage exceeded 95%',
        severity: 'critical',
        triggered: false
      },
      {
        threshold: 100,
        message: 'Budget exceeded',
        severity: 'critical',
        triggered: false
      }
    ];

    return alerts;
  }

  private generateSampleBenchmarks(): void {
    const benchmarks: PerformanceBenchmark[] = [
      {
        id: uuidv4(),
        category: 'compression',
        metric: 'Average Compression Ratio',
        currentValue: 0.75,
        targetValue: 0.8,
        industryAverage: 0.68,
        bestInClass: 0.85,
        percentile: 75,
        trend: 'improving',
        lastUpdated: new Date()
      },
      {
        id: uuidv4(),
        category: 'compression',
        metric: 'Processing Speed (files/sec)',
        currentValue: 45,
        targetValue: 50,
        industryAverage: 35,
        bestInClass: 65,
        percentile: 80,
        trend: 'stable',
        lastUpdated: new Date()
      },
      {
        id: uuidv4(),
        category: 'quality',
        metric: 'Quality Score',
        currentValue: 87,
        targetValue: 90,
        industryAverage: 82,
        bestInClass: 95,
        percentile: 70,
        trend: 'improving',
        lastUpdated: new Date()
      },
      {
        id: uuidv4(),
        category: 'batch_processing',
        metric: 'Batch Efficiency',
        currentValue: 92,
        targetValue: 95,
        industryAverage: 85,
        bestInClass: 98,
        percentile: 85,
        trend: 'stable',
        lastUpdated: new Date()
      }
    ];

    if (!this.benchmarks.has('global')) {
      this.benchmarks.set('global', []);
    }
    this.benchmarks.get('global')!.push(...benchmarks);
  }

  private generateSampleTrendAnalyses(): void {
    const analyses: TrendAnalysis[] = [
      this.generateUsageTrendAnalysis(),
      this.generatePerformanceTrendAnalysis(),
      this.generateCostTrendAnalysis(),
      this.generateQualityTrendAnalysis()
    ];

    if (!this.trendAnalyses.has('global')) {
      this.trendAnalyses.set('global', []);
    }
    this.trendAnalyses.get('global')!.push(...analyses);
  }

  private generateUsageTrendAnalysis(): TrendAnalysis {
    const dataPoints: TrendDataPoint[] = [];
    const now = new Date();
    
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      dataPoints.push({
        timestamp: date,
        value: Math.floor(Math.random() * 1000) + 500 + Math.sin(i / 7) * 100, // Weekly pattern
        metadata: { dayOfWeek: date.getDay() }
      });
    }

    return {
      id: uuidv4(),
      type: 'usage',
      timeframe: '30d',
      dataPoints,
      insights: [
        {
          type: 'seasonality',
          description: 'Usage peaks on weekdays, with highest activity on Tuesdays and Wednesdays',
          confidence: 0.85,
          impact: 'positive'
        },
        {
          type: 'pattern',
          description: 'Consistent 15% increase in usage over the past month',
          confidence: 0.92,
          impact: 'positive'
        }
      ],
      predictions: [
        {
          timeframe: 'next_7_days',
          predictedValue: 750,
          confidence: 0.78,
          factors: ['historical_patterns', 'seasonal_trends']
        }
      ],
      anomalies: []
    };
  }

  private generatePerformanceTrendAnalysis(): TrendAnalysis {
    const dataPoints: TrendDataPoint[] = [];
    const now = new Date();
    
    for (let i = 30; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      dataPoints.push({
        timestamp: date,
        value: 85 + Math.random() * 20 - 5, // 80-100 range with some variation
        metadata: { algorithm: 'mixed' }
      });
    }

    return {
      id: uuidv4(),
      type: 'performance',
      timeframe: '30d',
      dataPoints,
      insights: [
        {
          type: 'correlation',
          description: 'Performance improvements correlate with algorithm optimizations',
          confidence: 0.76,
          impact: 'positive'
        }
      ],
      predictions: [
        {
          timeframe: 'next_30_days',
          predictedValue: 92,
          confidence: 0.68,
          factors: ['optimization_trends', 'system_upgrades']
        }
      ],
      anomalies: []
    };
  }

  private generateCostTrendAnalysis(): TrendAnalysis {
    const dataPoints: TrendDataPoint[] = [];
    const now = new Date();
    
    for (let i = 90; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      dataPoints.push({
        timestamp: date,
        value: 100 + Math.random() * 50 + i * 0.5, // Gradual increase with noise
        metadata: { category: 'total' }
      });
    }

    return {
      id: uuidv4(),
      type: 'cost',
      timeframe: '90d',
      dataPoints,
      insights: [
        {
          type: 'pattern',
          description: 'Monthly cost increase of 3-5% due to growing usage',
          confidence: 0.88,
          impact: 'negative'
        }
      ],
      predictions: [
        {
          timeframe: 'next_90_days',
          predictedValue: 250,
          confidence: 0.72,
          factors: ['usage_growth', 'cost_optimization']
        }
      ],
      anomalies: []
    };
  }

  private generateQualityTrendAnalysis(): TrendAnalysis {
    const dataPoints: TrendDataPoint[] = [];
    const now = new Date();
    
    for (let i = 60; i >= 0; i--) {
      const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
      dataPoints.push({
        timestamp: date,
        value: 82 + Math.random() * 10 + Math.sin(i / 10) * 5, // Improving trend with oscillation
        metadata: { quality_metric: 'overall' }
      });
    }

    return {
      id: uuidv4(),
      type: 'quality',
      timeframe: '60d',
      dataPoints,
      insights: [
        {
          type: 'pattern',
          description: 'Quality scores show steady improvement with recent optimizations',
          confidence: 0.91,
          impact: 'positive'
        }
      ],
      predictions: [
        {
          timeframe: 'next_30_days',
          predictedValue: 89,
          confidence: 0.75,
          factors: ['optimization_impact', 'training_data_quality']
        }
      ],
      anomalies: []
    };
  }

  // Dashboard Management
  async getDashboards(req: Request, res: Response): Promise<void> {
    try {
      const { userId, type, isPublic } = req.query;
      
      let dashboards = Array.from(this.dashboards.values()).flat();
      
      if (userId) {
        dashboards = dashboards.filter(d => 
          d.permissions.owner === userId || 
          d.permissions.viewers.includes(userId as string) ||
          d.permissions.editors.includes(userId as string)
        );
      }
      
      if (type) {
        dashboards = dashboards.filter(d => d.type === type);
      }
      
      if (isPublic !== undefined) {
        dashboards = dashboards.filter(d => d.isPublic === (isPublic === 'true'));
      }

      res.json(dashboards);
    } catch (error) {
      console.error('Error getting dashboards:', error);
      res.status(500).json({ error: 'Failed to retrieve dashboards' });
    }
  }

  async createDashboard(req: Request, res: Response): Promise<void> {
    try {
      const { name, type, widgets, layout, isPublic, permissions } = req.body;
      
      const dashboard: Dashboard = {
        id: uuidv4(),
        name,
        type,
        widgets: widgets || [],
        layout: layout || { columns: 12, rows: 8, responsive: true },
        permissions: permissions || { owner: 'admin', viewers: [], editors: [] },
        createdAt: new Date(),
        updatedAt: new Date(),
        isPublic: isPublic || false
      };

      if (!this.dashboards.has(dashboard.permissions.owner)) {
        this.dashboards.set(dashboard.permissions.owner, []);
      }
      this.dashboards.get(dashboard.permissions.owner)!.push(dashboard);

      this.broadcastBusinessUpdate('dashboard_created', dashboard);
      res.status(201).json(dashboard);
    } catch (error) {
      console.error('Error creating dashboard:', error);
      res.status(500).json({ error: 'Failed to create dashboard' });
    }
  }

  async updateDashboard(req: Request, res: Response): Promise<void> {
    try {
      const { id } = req.params;
      const updates = req.body;
      
      let dashboard: Dashboard | undefined;
      
      // Find dashboard across all users
      for (const [userId, userDashboards] of this.dashboards.entries()) {
        const found = userDashboards.find(d => d.id === id);
        if (found) {
          dashboard = found;
          break;
        }
      }
      
      if (!dashboard) {
        return res.status(404).json({ error: 'Dashboard not found' });
      }
      
      Object.assign(dashboard, updates, { updatedAt: new Date() });
      
      this.broadcastBusinessUpdate('dashboard_updated', dashboard);
      res.json(dashboard);
    } catch (error) {
      console.error('Error updating dashboard:', error);
      res.status(500).json({ error: 'Failed to update dashboard' });
    }
  }

  // Cost Analysis
  async getCostAnalysis(req: Request, res: Response): Promise<void> {
    try {
      const { userId, period } = req.query;
      
      if (!userId) {
        return res.status(400).json({ error: 'userId is required' });
      }

      const userAnalyses = this.costAnalyses.get(userId as string) || [];
      let analysis = userAnalyses.find(a => {
        const analysisPeriod = this.getPeriodString(a.period.start, a.period.end);
        return analysisPeriod === (period as string || '30d');
      });

      if (!analysis) {
        analysis = this.generateCostAnalysis(userId as string, period as string || '30d');
        userAnalyses.push(analysis);
        this.costAnalyses.set(userId as string, userAnalyses);
      }

      // Update budget remaining
      if (analysis.budget) {
        analysis.budget.remaining = analysis.budget.allocated - analysis.budget.spent;
        
        // Check for budget alerts
        const usagePercentage = (analysis.budget.spent / analysis.budget.allocated) * 100;
        analysis.budget.alerts.forEach(alert => {
          alert.triggered = usagePercentage >= alert.threshold;
        });
      }

      this.broadcastBusinessUpdate('cost_analysis', analysis);
      res.json(analysis);
    } catch (error) {
      console.error('Error getting cost analysis:', error);
      res.status(500).json({ error: 'Failed to retrieve cost analysis' });
    }
  }

  private getPeriodString(start: Date, end: Date): string {
    const days = Math.floor((end.getTime() - start.getTime()) / (1000 * 60 * 60 * 24));
    if (days <= 1) return '24h';
    if (days <= 7) return '7d';
    if (days <= 30) return '30d';
    if (days <= 90) return '90d';
    return '1y';
  }

  private getDateRangeStart(range: string, end: Date): Date {
    const days = range === '24h' ? 1 : range === '7d' ? 7 : range === '30d' ? 30 : range === '90d' ? 90 : 365;
    return new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
  }

  // Performance Benchmarking
  async getPerformanceBenchmarks(req: Request, res: Response): Promise<void> {
    try {
      const { category, metric } = req.query;
      
      let benchmarks = Array.from(this.benchmarks.values()).flat();
      
      if (category) {
        benchmarks = benchmarks.filter(b => b.category === category);
      }
      
      if (metric) {
        benchmarks = benchmarks.filter(b => b.metric.toLowerCase().includes((metric as string).toLowerCase()));
      }

      const benchmarkSummary = {
        totalBenchmarks: benchmarks.length,
        categories: this.groupBenchmarksByCategory(benchmarks),
        topPerformers: this.getTopPerformers(benchmarks),
        improvementAreas: this.getImprovementAreas(benchmarks),
        benchmarks: benchmarks
      };

      this.broadcastBusinessUpdate('performance_benchmarks', benchmarkSummary);
      res.json(benchmarkSummary);
    } catch (error) {
      console.error('Error getting performance benchmarks:', error);
      res.status(500).json({ error: 'Failed to retrieve performance benchmarks' });
    }
  }

  private groupBenchmarksByCategory(benchmarks: PerformanceBenchmark[]): any {
    const grouped: { [key: string]: PerformanceBenchmark[] } = {};
    
    benchmarks.forEach(benchmark => {
      if (!grouped[benchmark.category]) {
        grouped[benchmark.category] = [];
      }
      grouped[benchmark.category].push(benchmark);
    });

    return Object.fromEntries(
      Object.entries(grouped).map(([category, items]) => [
        category,
        {
          count: items.length,
          averagePerformance: items.reduce((sum, b) => sum + b.percentile, 0) / items.length,
          trends: {
            improving: items.filter(b => b.trend === 'improving').length,
            stable: items.filter(b => b.trend === 'stable').length,
            declining: items.filter(b => b.trend === 'declining').length
          }
        }
      ])
    );
  }

  private getTopPerformers(benchmarks: PerformanceBenchmark[]): any[] {
    return benchmarks
      .filter(b => b.percentile >= 80)
      .sort((a, b) => b.percentile - a.percentile)
      .slice(0, 5)
      .map(b => ({
        metric: b.metric,
        value: b.currentValue,
        percentile: b.percentile,
        category: b.category
      }));
  }

  private getImprovementAreas(benchmarks: PerformanceBenchmark[]): any[] {
    return benchmarks
      .filter(b => b.percentile < 60)
      .sort((a, b) => a.percentile - b.percentile)
      .slice(0, 5)
      .map(b => ({
        metric: b.metric,
        currentValue: b.currentValue,
        targetValue: b.targetValue,
        gap: b.targetValue - b.currentValue,
        category: b.category
      }));
  }

  // Trend Analysis
  async getTrendAnalysis(req: Request, res: Response): Promise<void> {
    try {
      const { type, timeframe } = req.query;
      
      let analyses = Array.from(this.trendAnalyses.values()).flat();
      
      if (type) {
        analyses = analyses.filter(a => a.type === type);
      }
      
      if (timeframe) {
        analyses = analyses.filter(a => a.timeframe === timeframe);
      }

      const trendSummary = {
        totalAnalyses: analyses.length,
        types: this.groupAnalysesByType(analyses),
        keyInsights: this.extractKeyInsights(analyses),
        predictions: this.extractPredictions(analyses),
        anomalies: this.extractAnomalies(analyses),
        analyses
      };

      this.broadcastBusinessUpdate('trend_analysis', trendSummary);
      res.json(trendSummary);
    } catch (error) {
      console.error('Error getting trend analysis:', error);
      res.status(500).json({ error: 'Failed to retrieve trend analysis' });
    }
  }

  private groupAnalysesByType(analyses: TrendAnalysis[]): any {
    const grouped: { [key: string]: number } = {};
    
    analyses.forEach(analysis => {
      grouped[analysis.type] = (grouped[analysis.type] || 0) + 1;
    });

    return grouped;
  }

  private extractKeyInsights(analyses: TrendAnalysis[]): any[] {
    return analyses
      .flatMap(a => a.insights)
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 10);
  }

  private extractPredictions(analyses: TrendAnalysis[]): any[] {
    return analyses.flatMap(a => a.predictions);
  }

  private extractAnomalies(analyses: TrendAnalysis[]): any[] {
    return analyses.flatMap(a => a.anomalies);
  }

  // Real-time Dashboard Data
  async getRealtimeDashboardData(req: Request, res: Response): Promise<void> {
    try {
      const { dashboardId } = req.params;
      
      let dashboard: Dashboard | undefined;
      
      // Find dashboard
      for (const [userId, userDashboards] of this.dashboards.entries()) {
        const found = userDashboards.find(d => d.id === dashboardId);
        if (found) {
          dashboard = found;
          break;
        }
      }
      
      if (!dashboard) {
        return res.status(404).json({ error: 'Dashboard not found' });
      }

      const realtimeData = {
        dashboardId,
        lastUpdated: new Date(),
        widgets: dashboard.widgets.map(widget => ({
          ...widget,
          data: this.generateWidgetData(widget)
        }))
      };

      res.json(realtimeData);
    } catch (error) {
      console.error('Error getting realtime dashboard data:', error);
      res.status(500).json({ error: 'Failed to retrieve realtime dashboard data' });
    }
  }

  private generateWidgetData(widget: DashboardWidget): any {
    switch (widget.type) {
      case 'metric':
        return {
          value: Math.floor(Math.random() * 1000) + 100,
          unit: 'ops',
          change: (Math.random() - 0.5) * 20,
          trend: Math.random() > 0.5 ? 'up' : 'down'
        };
      case 'chart':
        return {
          data: Array.from({ length: 24 }, (_, i) => ({
            x: i,
            y: Math.floor(Math.random() * 100) + 20
          })),
          series: ['current', 'previous'],
          labels: ['Current Period', 'Previous Period']
        };
      case 'gauge':
        return {
          value: Math.floor(Math.random() * 40) + 60,
          min: 0,
          max: 100,
          thresholds: [
            { value: 70, color: '#ff6b6b' },
            { value: 85, color: '#feca57' },
            { value: 100, color: '#48dbfb' }
          ]
        };
      case 'table':
        return {
          rows: Array.from({ length: 10 }, (_, i) => ({
            id: i + 1,
            name: `Metric ${i + 1}`,
            value: Math.floor(Math.random() * 100),
            status: Math.random() > 0.7 ? 'warning' : 'normal'
          })),
          columns: ['name', 'value', 'status']
        };
      case 'heatmap':
        return {
          data: Array.from({ length: 7 }, (_, day) =>
            Array.from({ length: 24 }, (_, hour) => ({
              day,
              hour,
              value: Math.floor(Math.random() * 100)
            }))
          ),
          labels: ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        };
      default:
        return {};
    }
  }

  // WebSocket Broadcasting
  broadcastBusinessUpdate(type: string, data: any): void {
    const message = {
      type: 'business_update',
      dataType: type,
      data,
      timestamp: new Date().toISOString()
    };

    this.wsClients.forEach(client => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(JSON.stringify(message));
      }
    });
  }

  addWebSocketClient(ws: WebSocket): void {
    this.wsClients.add(ws);
    
    ws.on('close', () => {
      this.wsClients.delete(ws);
    });
  }
}

export default BusinessIntelligenceService;