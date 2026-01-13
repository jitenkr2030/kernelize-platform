import React, { useEffect, useState } from 'react'
import {
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  LinearProgress,
  Chip,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Avatar,
} from '@mui/material'
import {
  CloudUpload as CloudUploadIcon,
  Search as SearchIcon,
  People as PeopleIcon,
  Speed as SpeedIcon,
  TrendingUp as TrendingUpIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Schedule as ScheduleIcon,
} from '@mui/icons-material'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import { useQuery } from 'react-query'
import { useAuthStore } from '../../store/authStore'
import axios from 'axios'

interface DashboardStats {
  totalKernels: number
  totalQueries: number
  activeUsers: number
  avgCompressionRatio: number
  recentJobs: Array<{
    id: string
    status: string
    domain: string
    created_at: string
    compression_ratio?: number
  }>
  performanceMetrics: {
    cpu_usage: number
    memory_usage: number
    disk_usage: number
    network_io: number
  }
  compressionStats: Array<{
    domain: string
    count: number
    avgRatio: number
  }>
  queryHistory: Array<{
    timestamp: string
    queries: number
    avgResponseTime: number
  }>
}

const COLORS = ['#6366f1', '#8b5cf6', '#06b6d4', '#10b981', '#f59e0b', '#ef4444']

export default function Dashboard() {
  const { user } = useAuthStore()
  const [realTimeData, setRealTimeData] = useState({
    activeJobs: 0,
    systemLoad: 0,
    recentAlerts: 0,
  })

  const { data: dashboardData, isLoading } = useQuery<DashboardStats>(
    'dashboard-stats',
    async () => {
      const response = await axios.get('/dashboard/stats')
      return response.data
    },
    {
      refetchInterval: 30000, // Refresh every 30 seconds
    }
  )

  // Simulate real-time updates
  useEffect(() => {
    const interval = setInterval(() => {
      setRealTimeData(prev => ({
        activeJobs: Math.floor(Math.random() * 10) + 1,
        systemLoad: Math.random() * 100,
        recentAlerts: Math.floor(Math.random() * 3),
      }))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  if (isLoading) {
    return (
      <Box sx={{ width: '100%' }}>
        <LinearProgress />
      </Box>
    )
  }

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'success'
      case 'processing':
        return 'primary'
      case 'failed':
        return 'error'
      case 'queued':
        return 'warning'
      default:
        return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return <CheckCircleIcon color="success" />
      case 'processing':
        return <ScheduleIcon color="primary" />
      case 'failed':
        return <WarningIcon color="error" />
      default:
        return <ScheduleIcon color="warning" />
    }
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      {/* Welcome Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Welcome back, {user?.full_name || user?.username}!
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Here's what's happening with your KERNELIZE platform today.
        </Typography>
      </Box>

      {/* Key Metrics Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'primary.main', mr: 2 }}>
                  <CloudUploadIcon />
                </Avatar>
                <Box>
                  <Typography variant="h4" component="div">
                    {dashboardData?.totalKernels || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Kernels
                  </Typography>
                </Box>
              </Box>
              <Chip
                label={`${realTimeData.activeJobs} active`}
                size="small"
                color="primary"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'secondary.main', mr: 2 }}>
                  <SearchIcon />
                </Avatar>
                <Box>
                  <Typography variant="h4" component="div">
                    {dashboardData?.totalQueries || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Total Queries
                  </Typography>
                </Box>
              </Box>
              <Chip
                label="+12% this week"
                size="small"
                color="success"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'success.main', mr: 2 }}>
                  <PeopleIcon />
                </Avatar>
                <Box>
                  <Typography variant="h4" component="div">
                    {dashboardData?.activeUsers || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Users
                  </Typography>
                </Box>
              </Box>
              <Chip
                label="2 online now"
                size="small"
                color="success"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Avatar sx={{ bgcolor: 'warning.main', mr: 2 }}>
                  <SpeedIcon />
                </Avatar>
                <Box>
                  <Typography variant="h4" component="div">
                    {dashboardData?.avgCompressionRatio ? `${dashboardData.avgCompressionRatio.toFixed(1)}x` : '0x'}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Avg Compression
                  </Typography>
                </Box>
              </Box>
              <Chip
                label="Optimal"
                size="small"
                color="success"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        {/* Query Performance Chart */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Query Performance (Last 24 Hours)
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={dashboardData?.queryHistory || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="timestamp" stroke="#9CA3AF" />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '8px',
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="queries"
                    stroke="#6366f1"
                    strokeWidth={2}
                    dot={{ fill: '#6366f1' }}
                  />
                  <Line
                    type="monotone"
                    dataKey="avgResponseTime"
                    stroke="#10b981"
                    strokeWidth={2}
                    dot={{ fill: '#10b981' }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        {/* Compression by Domain */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Compression by Domain
              </Typography>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={dashboardData?.compressionStats || []}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={100}
                    paddingAngle={5}
                    dataKey="count"
                  >
                    {(dashboardData?.compressionStats || []).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
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
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Activity and System Status */}
      <Grid container spacing={3}>
        {/* Recent Compression Jobs */}
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Recent Compression Jobs
              </Typography>
              <List>
                {dashboardData?.recentJobs?.slice(0, 5).map((job) => (
                  <ListItem key={job.id} divider>
                    <ListItemIcon>
                      {getStatusIcon(job.status)}
                    </ListItemIcon>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body1">
                            {job.domain} Domain
                          </Typography>
                          <Chip
                            label={job.status}
                            size="small"
                            color={getStatusColor(job.status) as any}
                            variant="outlined"
                          />
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary">
                            Created: {new Date(job.created_at).toLocaleString()}
                          </Typography>
                          {job.compression_ratio && (
                            <Typography variant="body2" color="text.secondary">
                              Compression: {job.compression_ratio.toFixed(2)}x
                            </Typography>
                          )}
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        {/* System Performance */}
        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Performance
              </Typography>
              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">CPU Usage</Typography>
                  <Typography variant="body2">
                    {dashboardData?.performanceMetrics.cpu_usage.toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={dashboardData?.performanceMetrics.cpu_usage || 0}
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Memory Usage</Typography>
                  <Typography variant="body2">
                    {dashboardData?.performanceMetrics.memory_usage.toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={dashboardData?.performanceMetrics.memory_usage || 0}
                  color="secondary"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Box sx={{ mb: 3 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography variant="body2">Disk Usage</Typography>
                  <Typography variant="body2">
                    {dashboardData?.performanceMetrics.disk_usage.toFixed(1)}%
                  </Typography>
                </Box>
                <LinearProgress
                  variant="determinate"
                  value={dashboardData?.performanceMetrics.disk_usage || 0}
                  color="warning"
                  sx={{ height: 8, borderRadius: 4 }}
                />
              </Box>

              <Chip
                icon={<TrendingUpIcon />}
                label={`Load: ${realTimeData.systemLoad.toFixed(1)}%`}
                color="success"
                variant="outlined"
              />
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}