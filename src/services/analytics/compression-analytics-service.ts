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

interface CompressionMetrics {
  id: string;
  fileId: string;
  originalSize: number;
  compressedSize: number;
  compressionRatio: number;
  qualityScore: number;
  processingTime: number;
  algorithm: string;
  metadata: {
    mimeType: string;
    dimensions?: { width: number; height: number };
    colorDepth?: number;
    complexity: number;
  };
  createdAt: Date;
}

interface UsagePattern {
  id: string;
  userId: string;
  action: string;
  fileType: string;
  compressionLevel: string;
  timestamp: Date;
  duration: number;
  success: boolean;
  errorMessage?: string;
}

interface PerformanceOptimization {
  id: string;
  userId: string;
  currentStrategy: string;
  recommendedStrategy: string;
  expectedImprovement: {
    compressionRatio: number;
    processingSpeed: number;
    qualityRetention: number;
  };
  reasoning: string;
  confidence: number;
  implemented: boolean;
  createdAt: Date;
}

interface ROICalculation {
  id: string;
  userId: string;
  period: {
    start: Date;
    end: Date;
  };
  costs: {
    storage: number;
    bandwidth: number;
    processing: number;
    storageSavings: number;
    bandwidthSavings: number;
    processingSavings: number;
  };
  roi: number;
  paybackPeriod: number;
  projectedAnnualSavings: number;
}

class CompressionAnalyticsService {
  private metrics: Map<string, CompressionMetrics[]> = new Map();
  private patterns: Map<string, UsagePattern[]> = new Map();
  private optimizations: Map<string, PerformanceOptimization[]> = new Map();
  private roiCalculations: Map<string, ROICalculation[]> = new Map();
  private wsClients: Set<WebSocket> = new Set();

  constructor() {
    this.initializeAnalytics();
  }

  private initializeAnalytics(): void {
    // Initialize with sample data for demonstration
    this.generateSampleMetrics();
    this.generateSamplePatterns();
    this.generateSampleOptimizations();
  }

  private generateSampleMetrics(): void {
    const algorithms = ['JPEG', 'PNG', 'WebP', 'AVIF', 'HEIC'];
    const mimeTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/heic'];
    
    for (let i = 0; i < 100; i++) {
      const metric: CompressionMetrics = {
        id: uuidv4(),
        fileId: `file_${i}`,
        originalSize: Math.floor(Math.random() * 10000000) + 1000000, // 1MB - 10MB
        compressedSize: Math.floor(Math.random() * 5000000) + 500000, // 0.5MB - 5MB
        compressionRatio: 0,
        qualityScore: Math.floor(Math.random() * 30) + 70, // 70-100
        processingTime: Math.floor(Math.random() * 5000) + 1000, // 1-6 seconds
        algorithm: algorithms[Math.floor(Math.random() * algorithms.length)],
        metadata: {
          mimeType: mimeTypes[Math.floor(Math.random() * mimeTypes.length)],
          dimensions: {
            width: Math.floor(Math.random() * 4000) + 1000,
            height: Math.floor(Math.random() * 3000) + 800
          },
          colorDepth: Math.floor(Math.random() * 24) + 8,
          complexity: Math.random()
        },
        createdAt: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000) // Last 30 days
      };
      
      metric.compressionRatio = 1 - (metric.compressedSize / metric.originalSize);
      
      if (!this.metrics.has(metric.algorithm)) {
        this.metrics.set(metric.algorithm, []);
      }
      this.metrics.get(metric.algorithm)!.push(metric);
    }
  }

  private generateSamplePatterns(): void {
    const actions = ['compress', 'decompress', 'batch_process', 'convert_format'];
    const fileTypes = ['JPEG', 'PNG', 'WebP', 'AVIF', 'HEIC'];
    const compressionLevels = ['lossless', 'high', 'medium', 'low'];
    
    for (let i = 0; i < 200; i++) {
      const pattern: UsagePattern = {
        id: uuidv4(),
        userId: `user_${Math.floor(Math.random() * 10) + 1}`,
        action: actions[Math.floor(Math.random() * actions.length)],
        fileType: fileTypes[Math.floor(Math.random() * fileTypes.length)],
        compressionLevel: compressionLevels[Math.floor(Math.random() * compressionLevels.length)],
        timestamp: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000),
        duration: Math.floor(Math.random() * 10000) + 1000,
        success: Math.random() > 0.1, // 90% success rate
        errorMessage: Math.random() > 0.9 ? 'Processing timeout' : undefined
      };
      
      if (!this.patterns.has(pattern.userId)) {
        this.patterns.set(pattern.userId, []);
      }
      this.patterns.get(pattern.userId)!.push(pattern);
    }
  }

  private generateSampleOptimizations(): void {
    const strategies = ['Aggressive', 'Balanced', 'Quality-focused', 'Speed-optimized'];
    
    for (let i = 0; i < 50; i++) {
      const optimization: PerformanceOptimization = {
        id: uuidv4(),
        userId: `user_${Math.floor(Math.random() * 10) + 1}`,
        currentStrategy: strategies[Math.floor(Math.random() * strategies.length)],
        recommendedStrategy: strategies[Math.floor(Math.random() * strategies.length)],
        expectedImprovement: {
          compressionRatio: Math.random() * 0.3 + 0.1, // 10-40% improvement
          processingSpeed: Math.random() * 0.5 + 0.2, // 20-70% faster
          qualityRetention: Math.random() * 0.1 + 0.05 // 5-15% better quality
        },
        reasoning: 'Based on your usage patterns and file types, this strategy would optimize performance.',
        confidence: Math.random() * 0.3 + 0.7, // 70-100% confidence
        implemented: Math.random() > 0.7, // 30% implemented
        createdAt: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000) // Last 7 days
      };
      
      if (!this.optimizations.has(optimization.userId)) {
        this.optimizations.set(optimization.userId, []);
      }
      this.optimizations.get(optimization.userId)!.push(optimization);
    }
  }

  // Quality Metrics and Scoring
  async getQualityMetrics(req: Request, res: Response): Promise<void> {
    try {
      const { algorithm, timeRange, userId } = req.query;
      const range = timeRange as string || '7d';
      const now = new Date();
      const startDate = this.getDateRangeStart(range, now);

      let metrics = Array.from(this.metrics.values()).flat();
      
      if (algorithm) {
        metrics = metrics.filter(m => m.algorithm === algorithm);
      }
      
      if (userId) {
        // Filter by user patterns if available
        const userPatterns = this.patterns.get(userId as string) || [];
        const userFileIds = userPatterns.map(p => `file_${parseInt(p.id.split('_')[1]) % 100}`);
        metrics = metrics.filter(m => userFileIds.includes(m.fileId));
      }
      
      metrics = metrics.filter(m => m.createdAt >= startDate);

      const qualityMetrics = {
        averageQualityScore: metrics.reduce((sum, m) => sum + m.qualityScore, 0) / metrics.length,
        averageCompressionRatio: metrics.reduce((sum, m) => sum + m.compressionRatio, 0) / metrics.length,
        averageProcessingTime: metrics.reduce((sum, m) => sum + m.processingTime, 0) / metrics.length,
        algorithmPerformance: this.calculateAlgorithmPerformance(metrics),
        qualityDistribution: this.calculateQualityDistribution(metrics),
        efficiencyScore: this.calculateEfficiencyScore(metrics),
        trends: this.calculateTrends(metrics, range)
      };

      this.broadcastAnalyticsUpdate('quality_metrics', qualityMetrics);
      res.json(qualityMetrics);
    } catch (error) {
      console.error('Error getting quality metrics:', error);
      res.status(500).json({ error: 'Failed to retrieve quality metrics' });
    }
  }

  private calculateAlgorithmPerformance(metrics: CompressionMetrics[]): any {
    const algorithmStats: { [key: string]: any } = {};
    
    metrics.forEach(metric => {
      if (!algorithmStats[metric.algorithm]) {
        algorithmStats[metric.algorithm] = {
          count: 0,
          totalQualityScore: 0,
          totalCompressionRatio: 0,
          totalProcessingTime: 0,
          averageFileSize: 0
        };
      }
      
      const stats = algorithmStats[metric.algorithm];
      stats.count++;
      stats.totalQualityScore += metric.qualityScore;
      stats.totalCompressionRatio += metric.compressionRatio;
      stats.totalProcessingTime += metric.processingTime;
      stats.averageFileSize += metric.originalSize;
    });

    Object.keys(algorithmStats).forEach(algo => {
      const stats = algorithmStats[algo];
      algorithmStats[algo] = {
        ...stats,
        averageQualityScore: stats.totalQualityScore / stats.count,
        averageCompressionRatio: stats.totalCompressionRatio / stats.count,
        averageProcessingTime: stats.totalProcessingTime / stats.count,
        averageFileSize: stats.averageFileSize / stats.count,
        efficiency: (stats.totalQualityScore / stats.count) / (stats.totalProcessingTime / stats.count)
      };
      delete algorithmStats[algo].totalQualityScore;
      delete algorithmStats[algo].totalCompressionRatio;
      delete algorithmStats[algo].totalProcessingTime;
      delete algorithmStats[algo].averageFileSize;
    });

    return algorithmStats;
  }

  private calculateQualityDistribution(metrics: CompressionMetrics[]): any {
    const ranges = [
      { min: 90, max: 100, label: 'Excellent' },
      { min: 80, max: 89, label: 'Good' },
      { min: 70, max: 79, label: 'Average' },
      { min: 60, max: 69, label: 'Below Average' },
      { min: 0, max: 59, label: 'Poor' }
    ];

    const distribution: { [key: string]: number } = {};
    ranges.forEach(range => {
      distribution[range.label] = metrics.filter(m => 
        m.qualityScore >= range.min && m.qualityScore <= range.max
      ).length;
    });

    return {
      ...distribution,
      total: metrics.length,
      percentage: Object.fromEntries(
        Object.entries(distribution).map(([key, value]) => [
          key,
          ((value / metrics.length) * 100).toFixed(2)
        ])
      )
    };
  }

  private calculateEfficiencyScore(metrics: CompressionMetrics[]): number {
    const weightedScore = metrics.reduce((sum, metric) => {
      const qualityWeight = 0.4;
      const speedWeight = 0.3;
      const compressionWeight = 0.3;
      
      const qualityScore = metric.qualityScore / 100;
      const speedScore = Math.max(0, 1 - (metric.processingTime / 10000)); // Normalize to 0-1
      const compressionScore = metric.compressionRatio;
      
      return sum + (qualityWeight * qualityScore + speedWeight * speedScore + compressionWeight * compressionScore);
    }, 0);
    
    return (weightedScore / metrics.length) * 100;
  }

  private calculateTrends(metrics: CompressionMetrics[], range: string): any {
    const days = range === '24h' ? 1 : range === '7d' ? 7 : range === '30d' ? 30 : 90;
    const interval = Math.max(1, Math.floor(days / 10)); // 10 data points
    
    const trends: { [key: string]: any[] } = {
      quality: [],
      compression: [],
      processingTime: []
    };
    
    for (let i = 0; i < 10; i++) {
      const startTime = new Date(Date.now() - (days - i * interval) * 24 * 60 * 60 * 1000);
      const endTime = new Date(Date.now() - (days - (i + 1) * interval) * 24 * 60 * 60 * 1000);
      
      const periodMetrics = metrics.filter(m => m.createdAt >= endTime && m.createdAt <= startTime);
      
      if (periodMetrics.length > 0) {
        trends.quality.push({
          date: startTime.toISOString(),
          value: periodMetrics.reduce((sum, m) => sum + m.qualityScore, 0) / periodMetrics.length
        });
        
        trends.compression.push({
          date: startTime.toISOString(),
          value: periodMetrics.reduce((sum, m) => sum + m.compressionRatio, 0) / periodMetrics.length
        });
        
        trends.processingTime.push({
          date: startTime.toISOString(),
          value: periodMetrics.reduce((sum, m) => sum + m.processingTime, 0) / periodMetrics.length
        });
      }
    }
    
    return trends;
  }

  // Usage Pattern Analysis
  async getUsagePatterns(req: Request, res: Response): Promise<void> {
    try {
      const { userId, timeRange, action } = req.query;
      const range = timeRange as string || '7d';
      const now = new Date();
      const startDate = this.getDateRangeStart(range, now);

      let patterns = Array.from(this.patterns.values()).flat();
      
      if (userId) {
        patterns = patterns.filter(p => p.userId === userId);
      }
      
      if (action) {
        patterns = patterns.filter(p => p.action === action);
      }
      
      patterns = patterns.filter(p => p.timestamp >= startDate);

      const usagePatterns = {
        totalActions: patterns.length,
        successfulActions: patterns.filter(p => p.success).length,
        successRate: (patterns.filter(p => p.success).length / patterns.length * 100).toFixed(2),
        actionBreakdown: this.calculateActionBreakdown(patterns),
        fileTypeUsage: this.calculateFileTypeUsage(patterns),
        compressionLevelUsage: this.calculateCompressionLevelUsage(patterns),
        peakUsageHours: this.calculatePeakUsageHours(patterns),
        averageSessionDuration: this.calculateAverageSessionDuration(patterns),
        userEngagement: this.calculateUserEngagement(patterns, userId as string),
        patterns: this.identifyUsagePatterns(patterns)
      };

      this.broadcastAnalyticsUpdate('usage_patterns', usagePatterns);
      res.json(usagePatterns);
    } catch (error) {
      console.error('Error getting usage patterns:', error);
      res.status(500).json({ error: 'Failed to retrieve usage patterns' });
    }
  }

  private calculateActionBreakdown(patterns: UsagePattern[]): any {
    const breakdown: { [key: string]: number } = {};
    patterns.forEach(pattern => {
      breakdown[pattern.action] = (breakdown[pattern.action] || 0) + 1;
    });
    
    const total = patterns.length;
    return Object.fromEntries(
      Object.entries(breakdown).map(([action, count]) => [
        action,
        {
          count,
          percentage: ((count / total) * 100).toFixed(2)
        }
      ])
    );
  }

  private calculateFileTypeUsage(patterns: UsagePattern[]): any {
    const usage: { [key: string]: { count: number; totalDuration: number } } = {};
    patterns.forEach(pattern => {
      if (!usage[pattern.fileType]) {
        usage[pattern.fileType] = { count: 0, totalDuration: 0 };
      }
      usage[pattern.fileType].count++;
      usage[pattern.fileType].totalDuration += pattern.duration;
    });

    return Object.fromEntries(
      Object.entries(usage).map(([type, data]) => [
        type,
        {
          ...data,
          averageDuration: (data.totalDuration / data.count).toFixed(0)
        }
      ])
    );
  }

  private calculateCompressionLevelUsage(patterns: UsagePattern[]): any {
    const usage: { [key: string]: number } = {};
    patterns.forEach(pattern => {
      usage[pattern.compressionLevel] = (usage[pattern.compressionLevel] || 0) + 1;
    });
    
    const total = patterns.length;
    return Object.fromEntries(
      Object.entries(usage).map(([level, count]) => [
        level,
        {
          count,
          percentage: ((count / total) * 100).toFixed(2)
        }
      ])
    );
  }

  private calculatePeakUsageHours(patterns: UsagePattern[]): any {
    const hourCounts = new Array(24).fill(0);
    patterns.forEach(pattern => {
      const hour = pattern.timestamp.getHours();
      hourCounts[hour]++;
    });

    const peakHours = hourCounts
      .map((count, hour) => ({ hour, count }))
      .sort((a, b) => b.count - a.count)
      .slice(0, 3);

    return {
      peakHours,
      distribution: hourCounts.map((count, hour) => ({ hour, count }))
    };
  }

  private calculateAverageSessionDuration(patterns: UsagePattern[]): number {
    const totalDuration = patterns.reduce((sum, p) => sum + p.duration, 0);
    return (totalDuration / patterns.length) / 1000; // Convert to seconds
  }

  private calculateUserEngagement(patterns: UsagePattern[], userId?: string): any {
    if (!userId) {
      return { message: 'User engagement analysis requires userId parameter' };
    }

    const userPatterns = patterns.filter(p => p.userId === userId);
    const dailyActivity = this.groupByDate(userPatterns.map(p => p.timestamp));
    
    return {
      totalSessions: userPatterns.length,
      activeDays: Object.keys(dailyActivity).length,
      averageSessionsPerDay: (userPatterns.length / Object.keys(dailyActivity).length).toFixed(2),
      engagementScore: this.calculateEngagementScore(userPatterns),
      activityPattern: dailyActivity
    };
  }

  private identifyUsagePatterns(patterns: UsagePattern[]): any {
    return {
      frequentUsers: this.identifyFrequentUsers(patterns),
      preferredFileTypes: this.identifyPreferredFileTypes(patterns),
      compressionPreferences: this.identifyCompressionPreferences(patterns),
      workflowPatterns: this.identifyWorkflowPatterns(patterns)
    };
  }

  private identifyFrequentUsers(patterns: UsagePattern[]): any {
    const userActivity: { [key: string]: number } = {};
    patterns.forEach(pattern => {
      userActivity[pattern.userId] = (userActivity[pattern.userId] || 0) + 1;
    });

    return Object.entries(userActivity)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([userId, activity]) => ({ userId, activity }));
  }

  private identifyPreferredFileTypes(patterns: UsagePattern[]): any {
    const fileTypeUsage: { [key: string]: Set<string> } = {};
    patterns.forEach(pattern => {
      if (!fileTypeUsage[pattern.userId]) {
        fileTypeUsage[pattern.userId] = new Set();
      }
      fileTypeUsage[pattern.userId].add(pattern.fileType);
    });

    return Object.entries(fileTypeUsage).map(([userId, types]) => ({
      userId,
      preferredTypes: Array.from(types)
    }));
  }

  private identifyCompressionPreferences(patterns: UsagePattern[]): any {
    const preferences: { [key: string]: { [key: string]: number } } = {};
    patterns.forEach(pattern => {
      if (!preferences[pattern.userId]) {
        preferences[pattern.userId] = {};
      }
      preferences[pattern.userId][pattern.compressionLevel] = 
        (preferences[pattern.userId][pattern.compressionLevel] || 0) + 1;
    });

    return Object.entries(preferences).map(([userId, prefs]) => ({
      userId,
      preferredLevel: Object.entries(prefs).sort(([, a], [, b]) => b - a)[0][0]
    }));
  }

  private identifyWorkflowPatterns(patterns: UsagePattern[]): any {
    // Simple pattern identification based on consecutive actions
    const workflows: { [key: string]: number } = {};
    patterns
      .sort((a, b) => a.timestamp.getTime() - b.timestamp.getTime())
      .forEach((pattern, index) => {
        if (index < patterns.length - 1) {
          const nextPattern = patterns[index + 1];
          const workflow = `${pattern.action} -> ${nextPattern.action}`;
          workflows[workflow] = (workflows[workflow] || 0) + 1;
        }
      });

    return Object.entries(workflows)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 10)
      .map(([workflow, count]) => ({ workflow, count }));
  }

  private calculateEngagementScore(patterns: UsagePattern[]): number {
    // Simple engagement score based on frequency, duration, and variety
    const frequency = patterns.length;
    const avgDuration = patterns.reduce((sum, p) => sum + p.duration, 0) / patterns.length;
    const variety = new Set(patterns.map(p => p.action)).size;
    
    return Math.min(100, (frequency * 0.4 + (avgDuration / 1000) * 0.3 + variety * 10) * 0.5);
  }

  private groupByDate(dates: Date[]): { [key: string]: number } {
    return dates.reduce((acc, date) => {
      const dateKey = date.toISOString().split('T')[0];
      acc[dateKey] = (acc[dateKey] || 0) + 1;
      return acc;
    }, {} as { [key: string]: number });
  }

  // Performance Optimization Recommendations
  async getOptimizationRecommendations(req: Request, res: Response): Promise<void> {
    try {
      const { userId } = req.query;
      
      if (!userId) {
        return res.status(400).json({ error: 'userId is required' });
      }

      const userOptimizations = this.optimizations.get(userId as string) || [];
      const userPatterns = this.patterns.get(userId as string) || [];
      const userMetrics = this.getUserMetrics(userId as string);

      const recommendations = {
        currentOptimizations: userOptimizations,
        implementationRate: (userOptimizations.filter(o => o.implemented).length / userOptimizations.length * 100).toFixed(2),
        potentialImprovements: this.generatePotentialImprovements(userPatterns, userMetrics),
        personalizedTips: this.generatePersonalizedTips(userPatterns),
        systemRecommendations: this.generateSystemRecommendations(userPatterns, userMetrics)
      };

      this.broadcastAnalyticsUpdate('optimization_recommendations', recommendations);
      res.json(recommendations);
    } catch (error) {
      console.error('Error getting optimization recommendations:', error);
      res.status(500).json({ error: 'Failed to retrieve optimization recommendations' });
    }
  }

  private getUserMetrics(userId: string): CompressionMetrics[] {
    const userPatterns = this.patterns.get(userId) || [];
    const userFileIds = userPatterns.map(p => p.fileId);
    
    return Array.from(this.metrics.values())
      .flat()
      .filter(m => userFileIds.includes(m.fileId));
  }

  private generatePotentialImprovements(patterns: UsagePattern[], metrics: CompressionMetrics[]): any[] {
    const improvements = [];
    
    // Algorithm optimization
    const algorithmUsage = patterns.reduce((acc, p) => {
      acc[p.fileType] = (acc[p.fileType] || 0) + 1;
      return acc;
    }, {} as { [key: string]: number });
    
    Object.entries(algorithmUsage).forEach(([fileType, count]) => {
      if (count > 10) { // Frequent usage
        improvements.push({
          type: 'algorithm_switch',
          priority: 'high',
          description: `Switch to optimized algorithm for ${fileType} files`,
          expectedGain: '15-25% performance improvement',
          reasoning: 'High frequency usage detected for this file type'
        });
      }
    });

    // Compression level optimization
    const compressionUsage = patterns.reduce((acc, p) => {
      acc[p.compressionLevel] = (acc[p.compressionLevel] || 0) + 1;
      return acc;
    }, {} as { [key: string]: number });

    const totalUsage = Object.values(compressionUsage).reduce((sum, count) => sum + count, 0);
    Object.entries(compressionUsage).forEach(([level, count]) => {
      const percentage = (count / totalUsage) * 100;
      if (percentage < 10 && level !== 'lossless') {
        improvements.push({
          type: 'compression_optimization',
          priority: 'medium',
          description: `Consider using ${level} compression more frequently`,
          expectedGain: '10-20% storage savings',
          reasoning: `Low usage detected (${percentage.toFixed(1)}%) despite potential benefits`
        });
      }
    });

    return improvements;
  }

  private generatePersonalizedTips(patterns: UsagePattern[]): string[] {
    const tips = [];
    
    // Success rate tip
    const successRate = patterns.filter(p => p.success).length / patterns.length;
    if (successRate < 0.9) {
      tips.push('Consider using lower compression levels for better success rates');
    }
    
    // Processing time tip
    const avgTime = patterns.reduce((sum, p) => sum + p.duration, 0) / patterns.length;
    if (avgTime > 5000) {
      tips.push('Try batch processing for multiple files to improve efficiency');
    }
    
    // File type diversity
    const uniqueTypes = new Set(patterns.map(p => p.fileType)).size;
    if (uniqueTypes < 3) {
      tips.push('Experiment with different file formats for optimal results');
    }
    
    return tips;
  }

  private generateSystemRecommendations(patterns: UsagePattern[], metrics: CompressionMetrics[]): any[] {
    return [
      {
        category: 'hardware',
        recommendation: 'Consider upgrading to SSD storage for faster I/O operations',
        impact: 'high',
        effort: 'medium',
        reasoning: 'High file processing volume detected'
      },
      {
        category: 'software',
        recommendation: 'Enable parallel processing for batch operations',
        impact: 'medium',
        effort: 'low',
        reasoning: 'Multiple sequential operations detected'
      },
      {
        category: 'workflow',
        recommendation: 'Implement automated quality checks',
        impact: 'high',
        effort: 'high',
        reasoning: 'Quality consistency can be improved'
      }
    ];
  }

  // ROI Calculations
  async getROICalculation(req: Request, res: Response): Promise<void> {
    try {
      const { userId, period } = req.query;
      
      if (!userId) {
        return res.status(400).json({ error: 'userId is required' });
      }

      const userCalculations = this.roiCalculations.get(userId as string) || [];
      const userPatterns = this.patterns.get(userId as string) || [];
      const userMetrics = this.getUserMetrics(userId as string);

      const calculation = this.calculateROI(userId as string, userPatterns, userMetrics, period as string);
      
      this.broadcastAnalyticsUpdate('roi_calculation', calculation);
      res.json(calculation);
    } catch (error) {
      console.error('Error calculating ROI:', error);
      res.status(500).json({ error: 'Failed to calculate ROI' });
    }
  }

  private calculateROI(userId: string, patterns: UsagePattern[], metrics: CompressionMetrics[], period: string): ROICalculation {
    const now = new Date();
    const startDate = this.getDateRangeStart(period || '30d', now);
    
    const periodPatterns = patterns.filter(p => p.timestamp >= startDate);
    const periodMetrics = metrics.filter(m => m.createdAt >= startDate);

    // Calculate costs
    const totalOriginalSize = periodMetrics.reduce((sum, m) => sum + m.originalSize, 0);
    const totalCompressedSize = periodMetrics.reduce((sum, m) => sum + m.compressedSize, 0);
    const storageSavings = totalOriginalSize - totalCompressedSize;
    
    const processingCosts = periodPatterns.length * 0.01; // $0.01 per operation
    const storageCosts = (totalCompressedSize / (1024 * 1024 * 1024)) * 0.023; // $0.023 per GB per month
    const bandwidthCosts = (totalCompressedSize / (1024 * 1024)) * 0.09; // $0.09 per GB transfer
    
    const storageSavingsValue = (storageSavings / (1024 * 1024 * 1024)) * 0.023;
    const bandwidthSavingsValue = (storageSavings / (1024 * 1024)) * 0.09;
    const processingSavingsValue = processingCosts * 0.3; // 30% efficiency improvement

    const totalCosts = processingCosts + storageCosts + bandwidthCosts;
    const totalSavings = storageSavingsValue + bandwidthSavingsValue + processingSavingsValue;
    const netBenefit = totalSavings - totalCosts;
    const roi = totalCosts > 0 ? (netBenefit / totalCosts) * 100 : 0;
    const paybackPeriod = netBenefit > 0 ? totalCosts / netBenefit : 0;
    const projectedAnnualSavings = netBenefit * (365 / ((now.getTime() - startDate.getTime()) / (1000 * 60 * 60 * 24)));

    const calculation: ROICalculation = {
      id: uuidv4(),
      userId,
      period: { start: startDate, end: now },
      costs: {
        storage: storageCosts,
        bandwidth: bandwidthCosts,
        processing: processingCosts,
        storageSavings: storageSavingsValue,
        bandwidthSavings: bandwidthSavingsValue,
        processingSavings: processingSavingsValue
      },
      roi: Math.round(roi * 100) / 100,
      paybackPeriod: Math.round(paybackPeriod * 100) / 100,
      projectedAnnualSavings: Math.round(projectedAnnualSavings * 100) / 100
    };

    // Store calculation
    if (!this.roiCalculations.has(userId)) {
      this.roiCalculations.set(userId, []);
    }
    this.roiCalculations.get(userId)!.push(calculation);

    return calculation;
  }

  private getDateRangeStart(range: string, end: Date): Date {
    const days = range === '24h' ? 1 : range === '7d' ? 7 : range === '30d' ? 30 : range === '90d' ? 90 : 30;
    return new Date(end.getTime() - days * 24 * 60 * 60 * 1000);
  }

  // WebSocket Broadcasting
  broadcastAnalyticsUpdate(type: string, data: any): void {
    const message = {
      type: 'analytics_update',
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

  // Record new metrics
  recordCompressionMetrics(metrics: Omit<CompressionMetrics, 'id' | 'createdAt'>): void {
    const fullMetrics: CompressionMetrics = {
      ...metrics,
      id: uuidv4(),
      createdAt: new Date()
    };

    if (!this.metrics.has(metrics.algorithm)) {
      this.metrics.set(metrics.algorithm, []);
    }
    this.metrics.get(metrics.algorithm)!.push(fullMetrics);

    this.broadcastAnalyticsUpdate('new_metrics', fullMetrics);
  }

  recordUsagePattern(pattern: Omit<UsagePattern, 'id' | 'timestamp'>): void {
    const fullPattern: UsagePattern = {
      ...pattern,
      id: uuidv4(),
      timestamp: new Date()
    };

    if (!this.patterns.has(pattern.userId)) {
      this.patterns.set(pattern.userId, []);
    }
    this.patterns.get(pattern.userId)!.push(fullPattern);

    this.broadcastAnalyticsUpdate('new_pattern', fullPattern);
  }
}

export default CompressionAnalyticsService;