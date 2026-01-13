import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Grid,
  Chip,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  IconButton,
  Menu,
  MenuItem,
} from '@mui/material'
import {
  CloudUpload as UploadIcon,
  PlayArrow as PlayIcon,
  Pause as PauseIcon,
  MoreVert as MoreIcon,
  CheckCircle as SuccessIcon,
  Error as ErrorIcon,
  Schedule as PendingIcon,
} from '@mui/icons-material'
import { useQuery, useMutation } from 'react-query'
import axios from 'axios'

interface CompressionJob {
  id: string
  filename: string
  domain: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  compression_ratio: number
  original_size: number
  compressed_size: number
  created_at: string
  completed_at?: string
  error_message?: string
}

export default function CompressionJobs() {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const [selectedJob, setSelectedJob] = useState<CompressionJob | null>(null)

  const { data: jobs, isLoading, refetch } = useQuery<CompressionJob[]>(
    'compression-jobs',
    async () => {
      const response = await axios.get('/compression/jobs')
      return response.data
    },
    {
      refetchInterval: 5000, // Refresh every 5 seconds
    }
  )

  const pauseJobMutation = useMutation(
    async (jobId: string) => {
      await axios.post(`/compression/jobs/${jobId}/pause`)
    },
    {
      onSuccess: () => refetch(),
    }
  )

  const resumeJobMutation = useMutation(
    async (jobId: string) => {
      await axios.post(`/compression/jobs/${jobId}/resume`)
    },
    {
      onSuccess: () => refetch(),
    }
  )

  const handleMenuOpen = (event: React.MouseEvent<HTMLElement>, job: CompressionJob) => {
    setAnchorEl(event.currentTarget)
    setSelectedJob(job)
  }

  const handleMenuClose = () => {
    setAnchorEl(null)
    setSelectedJob(null)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed':
        return 'success'
      case 'processing':
        return 'primary'
      case 'failed':
        return 'error'
      case 'pending':
        return 'warning'
      default:
        return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <SuccessIcon />
      case 'processing':
        return <PlayIcon />
      case 'failed':
        return <ErrorIcon />
      case 'pending':
        return <PendingIcon />
      default:
        return <PendingIcon />
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
        <Box>
          <Typography variant="h4" gutterBottom>
            Compression Jobs
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Monitor and manage all compression operations
          </Typography>
        </Box>
        <Button
          variant="contained"
          startIcon={<UploadIcon />}
          onClick={() => {
            // Handle file upload
          }}
        >
          New Compression Job
        </Button>
      </Box>

      {/* Summary Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Jobs
              </Typography>
              <Typography variant="h4" color="primary">
                {jobs?.length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Processing
              </Typography>
              <Typography variant="h4" color="primary">
                {jobs?.filter(job => job.status === 'processing').length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Completed
              </Typography>
              <Typography variant="h4" color="success.main">
                {jobs?.filter(job => job.status === 'completed').length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Failed
              </Typography>
              <Typography variant="h4" color="error.main">
                {jobs?.filter(job => job.status === 'failed').length || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Jobs Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            All Compression Jobs
          </Typography>
          
          {isLoading ? (
            <LinearProgress />
          ) : (
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow>
                    <TableCell>File</TableCell>
                    <TableCell>Domain</TableCell>
                    <TableCell>Status</TableCell>
                    <TableCell>Progress</TableCell>
                    <TableCell>Compression Ratio</TableCell>
                    <TableCell>Size</TableCell>
                    <TableCell>Created</TableCell>
                    <TableCell>Actions</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {jobs?.map((job) => (
                    <TableRow key={job.id}>
                      <TableCell>
                        <Typography variant="body2" fontWeight={500}>
                          {job.filename}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Chip
                          label={job.domain}
                          size="small"
                          variant="outlined"
                        />
                      </TableCell>
                      <TableCell>
                        <Chip
                          icon={getStatusIcon(job.status)}
                          label={job.status}
                          color={getStatusColor(job.status) as any}
                          size="small"
                        />
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <LinearProgress
                            variant="determinate"
                            value={job.progress}
                            sx={{ width: 100 }}
                          />
                          <Typography variant="caption">
                            {job.progress}%
                          </Typography>
                        </Box>
                      </TableCell>
                      <TableCell>
                        {job.compression_ratio ? (
                          <Typography variant="body2" color="primary">
                            {job.compression_ratio.toFixed(2)}x
                          </Typography>
                        ) : (
                          <Typography variant="body2" color="text.secondary">
                            -
                          </Typography>
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {formatFileSize(job.compressed_size)} / {formatFileSize(job.original_size)}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          {new Date(job.created_at).toLocaleString()}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <IconButton onClick={(e) => handleMenuOpen(e, job)}>
                          <MoreIcon />
                        </IconButton>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </CardContent>
      </Card>

      {/* Context Menu */}
      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleMenuClose}>
          View Details
        </MenuItem>
        <MenuItem onClick={handleMenuClose}>
          Download Result
        </MenuItem>
        {selectedJob?.status === 'processing' && (
          <MenuItem
            onClick={() => {
              pauseJobMutation.mutate(selectedJob.id)
              handleMenuClose()
            }}
          >
            Pause Job
          </MenuItem>
        )}
        {selectedJob?.status === 'pending' && (
          <MenuItem
            onClick={() => {
              resumeJobMutation.mutate(selectedJob.id)
              handleMenuClose()
            }}
          >
            Resume Job
          </MenuItem>
        )}
        <MenuItem onClick={handleMenuClose}>
          Delete Job
        </MenuItem>
      </Menu>
    </Box>
  )
}