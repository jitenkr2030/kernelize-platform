import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  LinearProgress,
  Chip,
  Alert,
  Button,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material'
import {
  ExpandMore as ExpandMoreIcon,
  Warning as WarningIcon,
  CheckCircle as CheckCircleIcon,
  Error as ErrorIcon,
  Refresh as RefreshIcon,
  Speed as SpeedIcon,
  Memory as MemoryIcon,
  Storage as StorageIcon,
  NetworkCheck as NetworkIcon,
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
} from 'recharts'
import { useQuery } from 'react-query'
import axios from 'axios'

interface SystemMetrics {
  cpu_usage: number
  memory_usage: number
  disk_usage: number
  network_io: number
  active_connections: number
  uptime: number
}

interface ServiceStatus {
  name: string
  status: 'healthy' | 'degraded' | 'down'
  response_time: number
  last_check: string
}

interface AlertItem {
  id: string
  level: 'info' | 'warning' | 'error'
  message: string
  timestamp: string
  service?: string
}

interface PerformanceData {
  timestamp: string
  cpu: number
  memory: number
  disk: number
  network: number
}

export default function SystemMonitor() {
  const [timeRange, setTimeRange] = useState('1h')
  const [autoRefresh, setAutoRefresh] = useState(true)

  const { data: metricsData, isLoading, refetch } = useQuery<SystemMetrics>(
    'system-metrics',
    async () => {
      const response = await axios.get('/system/metrics')
      return response.data
    },
    {
      refetchInterval: autoRefresh ? 5000 : false,
    }
  )

  const { data: servicesData } = useQuery<ServiceStatus[]>(
    'service-status',
    async () => {
      const response = await axios.get('/system/services')
      return response.data
    },
    {
      refetchInterval: autoRefresh ? 10000 : false,
    }
  )

  const { data: alertsData } = useQuery<AlertItem[]>(
    'system-alerts',
    async () => {
      const response = await axios.get('/system/alerts')
      return response.data
    },
    {
      refetchInterval: autoRefresh ? 30000 : false,
    }
  )

  const { data: performanceHistory } = useQuery<PerformanceData[]>(
    'performance-history',
    async () => {
      const response = await axios.get('/system/performance-history', {
        params: { range: timeRange }
      })
      return response.data
    },
    {
      refetchInterval: autoRefresh ? 60000 : false,
    }
  )

  // Simulate real-time data updates
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        refetch()
      }, 5000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh, refetch])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success'
      case 'degraded':
        return 'warning'
      case 'down':
        return 'error'
      default:
        return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleIcon color="success" />
      case 'degraded':
        return <WarningIcon color="warning" />
      case 'down':
        return <ErrorIcon color="error" />
      default:
        return <CheckCircleIcon />
    }
  }

  const getAlertColor = (level: string) => {
    switch (level) {
      case 'error':
        return 'error'
      case 'warning':
        return 'warning'
      case 'info':
        return 'info'
      default:
        return 'default'
    }
  }

  const getUsageColor = (usage: number) => {
    if (usage >= 90) return 'error'
    if (usage >= 70) return 'warning'
    return 'success'
  }

  const formatUptime = (uptime: number) => {
    const days = Math.floor(uptime / 86400)
    const hours = Math.floor((uptime % 86400) / 3600)
    const minutes = Math.floor((uptime % 3600) / 60)
    return `${days}d ${hours}h ${minutes}m`
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            System Monitor
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Real-time system performance and health monitoring
          </Typography>
        </Box>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Time Range</InputLabel>
            <Select
              value={timeRange}
              label="Time Range"
              onChange={(e) => setTimeRange(e.target.value)}
            >
              <MenuItem value="15m">15 Minutes</MenuItem>
              <MenuItem value="1h">1 Hour</MenuItem>
              <MenuItem value="6h">6 Hours</MenuItem>
              <MenuItem value="24h">24 Hours</MenuItem>
              <MenuItem value="7d">7 Days</MenuItem>
            </Select>
          </FormControl>
          <Button
            variant="outlined"
            startIcon={<RefreshIcon />}
            onClick={() => refetch()}
          >
            Refresh
          </Button>
          <Chip
            label={autoRefresh ? 'Auto Refresh ON' : 'Auto Refresh OFF'}
            color={autoRefresh ? 'success' : 'default'}
            onClick={() => setAutoRefresh(!autoRefresh)}
          />
        </Box>
      </Box>

      {/* System Overview Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <SpeedIcon sx={{ color: 'primary.main', mr: 2, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {metricsData?.cpu_usage.toFixed(1) || 0}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    CPU Usage
                  </Typography>
                </Box>
              </Box>
              <LinearProgress
                variant="determinate"
                value={metricsData?.cpu_usage || 0}
                color={getUsageColor(metricsData?.cpu_usage || 0) as any}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <MemoryIcon sx={{ color: 'secondary.main', mr: 2, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {metricsData?.memory_usage.toFixed(1) || 0}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Memory Usage
                  </Typography>
                </Box>
              </Box>
              <LinearProgress
                variant="determinate"
                value={metricsData?.memory_usage || 0}
                color={getUsageColor(metricsData?.memory_usage || 0) as any}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <StorageIcon sx={{ color: 'warning.main', mr: 2, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {metricsData?.disk_usage.toFixed(1) || 0}%
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Disk Usage
                  </Typography>
                </Box>
              </Box>
              <LinearProgress
                variant="determinate"
                value={metricsData?.disk_usage || 0}
                color={getUsageColor(metricsData?.disk_usage || 0) as any}
                sx={{ height: 8, borderRadius: 4 }}
              />
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <NetworkIcon sx={{ color: 'success.main', mr: 2, fontSize: 32 }} />
                <Box>
                  <Typography variant="h4">
                    {metricsData?.active_connections || 0}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Active Connections
                  </Typography>
                </Box>
              </Box>
              <Typography variant="caption" color="text.secondary">
                Uptime: {metricsData ? formatUptime(metricsData.uptime) : '0s'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Performance Charts */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} lg={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Performance Metrics (Last {timeRange})
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <LineChart data={performanceHistory || []}>
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
                    dataKey="cpu"
                    stroke="#6366f1"
                    strokeWidth={2}
                    name="CPU %"
                  />
                  <Line
                    type="monotone"
                    dataKey="memory"
                    stroke="#8b5cf6"
                    strokeWidth={2}
                    name="Memory %"
                  />
                  <Line
                    type="monotone"
                    dataKey="disk"
                    stroke="#06b6d4"
                    strokeWidth={2}
                    name="Disk %"
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Network Activity
              </Typography>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={performanceHistory || []}>
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
                  <Area
                    type="monotone"
                    dataKey="network"
                    stroke="#10b981"
                    fill="#10b981"
                    fillOpacity={0.6}
                    name="Network I/O"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Service Status and Alerts */}
      <Grid container spacing={3}>
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Service Status
              </Typography>
              <List>
                {servicesData?.map((service) => (
                  <ListItem key={service.name} divider>
                    <ListItemIcon>
                      {getStatusIcon(service.status)}
                    </ListItemIcon>
                    <ListItemText
                      primary={service.name}
                      secondary={
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Response Time: {service.response_time}ms
                          </Typography>
                          <br />
                          <Typography variant="caption" color="text.secondary">
                            Last Check: {new Date(service.last_check).toLocaleString()}
                          </Typography>
                        </Box>
                      }
                    />
                    <Chip
                      label={service.status}
                      color={getStatusColor(service.status) as any}
                      size="small"
                    />
                  </ListItem>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Alerts
              </Typography>
              {alertsData?.length === 0 ? (
                <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                  No active alerts
                </Typography>
              ) : (
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                  {alertsData?.map((alert) => (
                    <Alert
                      key={alert.id}
                      severity={getAlertColor(alert.level) as any}
                      action={
                        <Button size="small" color="inherit">
                          Dismiss
                        </Button>
                      }
                    >
                      <Box>
                        <Typography variant="body2" fontWeight={500}>
                          {alert.message}
                        </Typography>
                        <Typography variant="caption">
                          {alert.service} • {new Date(alert.timestamp).toLocaleString()}
                        </Typography>
                      </Box>
                    </Alert>
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}