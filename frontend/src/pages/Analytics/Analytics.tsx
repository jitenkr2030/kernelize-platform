import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Paper,
  Tabs,
  Tab,
  Button,
  IconButton,
} from '@mui/material'
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown as TrendingDownIcon,
  FileDownload as DownloadIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  RadialBarChart,
  RadialBar,
} from 'recharts'
import { useQuery } from 'react-query'
import axios from 'axios'

interface AnalyticsData {
  compressionTrends: Array<{
    date: string
    total_compressions: number
    avg_ratio: number
    total_size_mb: number
  }>
  domainDistribution: Array<{
    domain: string
    count: number
    avg_compression_ratio: number
    total_size_mb: number
  }>
  performanceMetrics: {
    total_queries: number
    avg_response_time: number
    success_rate: number
    peak_concurrent_users: number
  }
  algorithmPerformance: Array<{
    algorithm: string
    usage_count: number
    avg_compression_ratio: number
    avg_processing_time: number
  }>
  userEngagement: Array<{
    date: string
    active_users: number
    new_registrations: number
    queries_per_user: number
  }>
  systemHealth: {
    uptime_percentage: number
    avg_cpu_usage: number
    avg_memory_usage: number
    error_rate: number
  }
}

const COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444', '#84cc16', '#f97316']

const RADIAN = Math.PI / 180

export default function Analytics() {
  const [activeTab, setActiveTab] = useState(0)
  const [timeRange, setTimeRange] = useState('7d')
  const [selectedDomain, setSelectedDomain] = useState('all')

  const { data: analyticsData, isLoading, refetch } = useQuery<AnalyticsData>(
    ['analytics', timeRange, selectedDomain],
    async () => {
      const response = await axios.get('/analytics/overview', {
        params: { time_range: timeRange, domain: selectedDomain },
      })
      return response.data
    },
    {
      refetchInterval: 60000, // Refresh every minute
    }
  )

  const handleDownloadReport = () => {
    // Generate and download analytics report
    window.open('/analytics/export', '_blank')
  }

  if (isLoading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '50vh' }}>
        <Typography>Loading analytics data...</Typography>
      </Box>
    )
  }

  const renderCustomizedLabel = ({ cx, cy, midAngle, innerRadius, outerRadius, percent }: any) => {
    const radius = innerRadius + (outerRadius - innerRadius) * 0.5
    const x = cx + radius * Math.cos(-midAngle * RADIAN)
    const y = cy + radius * Math.sin(-midAngle * RADIAN)

    return (
      <text
        x={x}
        y={y}
        fill="white"
        textAnchor={x > cx ? 'start' : 'end'}
        dominantBaseline="central"
      >
        {`${(percent * 100).toFixed(0)}%`}
      </text>
    )
  }

  const pieData = analyticsData?.domainDistribution.map((item, index) => ({
    name: item.domain,
    value: item.count,
    fill: COLORS[index % COLORS.length],
  })) || []

  const radialData = [
    {
      name: 'Uptime',
      value: analyticsData?.systemHealth.uptime_percentage || 0,
      fill: '#10b981',
    },
    {
      name: 'Success Rate',
      value: analyticsData?.performanceMetrics.success_rate || 0,
      fill: '#6366f1',
    },
    {
      name: 'CPU Usage',
      value: analyticsData?.systemHealth.avg_cpu_usage || 0,
      fill: '#f59e0b',
    },
  ]

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Analytics & Performance
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Comprehensive insights into compression performance and system metrics
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <IconButton onClick={() => refetch()}>
            <RefreshIcon />
          </IconButton>
          <Button
            variant="outlined"
            startIcon={<DownloadIcon />}
            onClick={handleDownloadReport}
          >
            Export Report
          </Button>
        </Box>
      </Box>

      {/* Filters */}
      <Card sx={{ mb: 3 }}>
        <CardContent>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel>Time Range</InputLabel>
                <Select
                  value={timeRange}
                  label="Time Range"
                  onChange={(e) => setTimeRange(e.target.value)}
                >
                  <MenuItem value="24h">Last 24 Hours</MenuItem>
                  <MenuItem value="7d">Last 7 Days</MenuItem>
                  <MenuItem value="30d">Last 30 Days</MenuItem>
                  <MenuItem value="90d">Last 90 Days</MenuItem>
                  <MenuItem value="1y">Last Year</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <FormControl fullWidth>
                <InputLabel>Domain Filter</InputLabel>
                <Select
                  value={selectedDomain}
                  label="Domain Filter"
                  onChange={(e) => setSelectedDomain(e.target.value)}
                >
                  <MenuItem value="all">All Domains</MenuItem>
                  <MenuItem value="general">General</MenuItem>
                  <MenuItem value="genomics">Genomics</MenuItem>
                  <MenuItem value="finance">Finance</MenuItem>
                  <MenuItem value="cybersecurity">Cybersecurity</MenuItem>
                  <MenuItem value="legal">Legal</MenuItem>
                  <MenuItem value="healthcare">Healthcare</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon sx={{ color: 'success.main', mr: 2, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {analyticsData?.performanceMetrics.total_queries.toLocaleString() || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Queries
                  </Typography>
                </Box>
              </Box>
              <Chip
                label="+12.5% vs last period"
                color="success"
                size="small"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon sx={{ color: 'primary.main', mr: 2, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {analyticsData?.performanceMetrics.avg_response_time.toFixed(0) || 0}ms
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Response Time
                  </Typography>
                </Box>
              </Box>
              <Chip
                label="-8.2% vs last period"
                color="success"
                size="small"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon sx={{ color: 'secondary.main', mr: 2, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {analyticsData?.performanceMetrics.success_rate.toFixed(1) || 0}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Success Rate
                  </Typography>
                </Box>
              </Box>
              <Chip
                label="99.8% uptime"
                color="success"
                size="small"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TrendingUpIcon sx={{ color: 'warning.main', mr: 2, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {analyticsData?.performanceMetrics.peak_concurrent_users || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Peak Concurrent Users
                  </Typography>
                </Box>
              </Box>
              <Chip
                label="Current load optimal"
                color="warning"
                size="small"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Main Analytics Tabs */}
      <Card>
        <CardContent>
          <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
            <Tab label="Compression Trends" />
            <Tab label="Domain Analysis" />
            <Tab label="Performance Metrics" />
            <Tab label="System Health" />
          </Tabs>

          <Box sx={{ mt: 3 }}>
            {activeTab === 0 && (
              <Grid container spacing={3}>
                <Grid item xs={12} lg={8}>
                  <Typography variant="h6" gutterBottom>
                    Compression Trends Over Time
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <AreaChart data={analyticsData?.compressionTrends || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="date" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                      />
                      <Legend />
                      <Area
                        type="monotone"
                        dataKey="total_compressions"
                        stackId="1"
                        stroke="#6366f1"
                        fill="#6366f1"
                        fillOpacity={0.6}
                        name="Total Compressions"
                      />
                      <Area
                        type="monotone"
                        dataKey="avg_ratio"
                        stackId="2"
                        stroke="#10b981"
                        fill="#10b981"
                        fillOpacity={0.6}
                        name="Avg Compression Ratio"
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                </Grid>
                <Grid item xs={12} lg={4}>
                  <Typography variant="h6" gutterBottom>
                    Domain Distribution
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={renderCustomizedLabel}
                        outerRadius={120}
                        fill="#8884d8"
                        dataKey="value"
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </Grid>
              </Grid>
            )}

            {activeTab === 1 && (
              <Grid container spacing={3}>
                <Grid item xs={12} lg={8}>
                  <Typography variant="h6" gutterBottom>
                    Compression Performance by Domain
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <BarChart data={analyticsData?.domainDistribution || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="domain" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                      />
                      <Legend />
                      <Bar dataKey="avg_compression_ratio" fill="#6366f1" name="Avg Compression Ratio" />
                      <Bar dataKey="count" fill="#10b981" name="Total Compressions" />
                    </BarChart>
                  </ResponsiveContainer>
                </Grid>
                <Grid item xs={12} lg={4}>
                  <Typography variant="h6" gutterBottom>
                    Algorithm Performance
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={analyticsData?.algorithmPerformance || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="algorithm" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="avg_compression_ratio"
                        stroke="#6366f1"
                        strokeWidth={2}
                        name="Compression Ratio"
                      />
                      <Line
                        type="monotone"
                        dataKey="avg_processing_time"
                        stroke="#f59e0b"
                        strokeWidth={2}
                        name="Processing Time (ms)"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Grid>
              </Grid>
            )}

            {activeTab === 2 && (
              <Grid container spacing={3}>
                <Grid item xs={12} lg={8}>
                  <Typography variant="h6" gutterBottom>
                    User Engagement Metrics
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={analyticsData?.userEngagement || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="date" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="active_users"
                        stroke="#6366f1"
                        strokeWidth={2}
                        name="Active Users"
                      />
                      <Line
                        type="monotone"
                        dataKey="queries_per_user"
                        stroke="#10b981"
                        strokeWidth={2}
                        name="Queries per User"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Grid>
                <Grid item xs={12} lg={4}>
                  <Typography variant="h6" gutterBottom>
                    System Performance
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <RadialBarChart cx="50%" cy="50%" innerRadius="10%" outerRadius="80%" data={radialData}>
                      <RadialBar
                        minAngle={15}
                        label={{ position: 'insideStart', fill: '#fff' }}
                        background
                        clockWise
                        dataKey="value"
                      />
                      <Tooltip
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                        }}
                      />
                    </RadialBarChart>
                  </ResponsiveContainer>
                </Grid>
              </Grid>
            )}

            {activeTab === 3 && (
              <Grid container spacing={3}>
                <Grid item xs={12} lg={6}>
                  <Typography variant="h6" gutterBottom>
                    System Health Metrics
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Uptime: {analyticsData?.systemHealth.uptime_percentage.toFixed(2)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          System has been running continuously without issues
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          CPU Usage: {analyticsData?.systemHealth.avg_cpu_usage.toFixed(1)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Average CPU utilization across all nodes
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Memory Usage: {analyticsData?.systemHealth.avg_memory_usage.toFixed(1)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Average memory utilization across all nodes
                        </Typography>
                      </Paper>
                    </Grid>
                    <Grid item xs={12}>
                      <Paper sx={{ p: 2 }}>
                        <Typography variant="subtitle2" gutterBottom>
                          Error Rate: {analyticsData?.systemHealth.error_rate.toFixed(3)}%
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          Percentage of failed operations
                        </Typography>
                      </Paper>
                    </Grid>
                  </Grid>
                </Grid>
                <Grid item xs={12} lg={6}>
                  <Typography variant="h6" gutterBottom>
                    Performance Insights
                  </Typography>
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <Paper sx={{ p: 2, bgcolor: 'success.light', color: 'success.contrastText' }}>
                      <Typography variant="subtitle2" gutterBottom>
                        ✓ Performance Optimal
                      </Typography>
                      <Typography variant="body2">
                        All systems are performing within expected parameters
                      </Typography>
                    </Paper>
                    <Paper sx={{ p: 2, bgcolor: 'primary.light', color: 'primary.contrastText' }}>
                      <Typography variant="subtitle2" gutterBottom>
                        ↗ Trend Analysis
                      </Typography>
                      <Typography variant="body2">
                        Compression efficiency has improved by 15% this month
                      </Typography>
                    </Paper>
                    <Paper sx={{ p: 2, bgcolor: 'warning.light', color: 'warning.contrastText' }}>
                      <Typography variant="subtitle2" gutterBottom>
                        ⚠ Recommendation
                      </Typography>
                      <Typography variant="body2">
                        Consider scaling up during peak hours (2-4 PM)
                      </Typography>
                    </Paper>
                  </Box>
                </Grid>
              </Grid>
            )}
          </Box>
        </CardContent>
      </Card>
    </Box>
  )
}