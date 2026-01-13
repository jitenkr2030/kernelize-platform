import React, { useState, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  Grid,
  Chip,
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
  Switch,
  FormControlLabel,
  Alert,
  CircularProgress,
  Divider,
  Tabs,
  Tab,
} from '@mui/material'
import {
  Search as SearchIcon,
  ExpandMore as ExpandMoreIcon,
  ContentCopy as CopyIcon,
  Download as DownloadIcon,
  History as HistoryIcon,
  FilterList as FilterIcon,
  PlayArrow as PlayIcon,
  Clear as ClearIcon,
} from '@mui/icons-material'
import { useForm, Controller } from 'react-hook-form'
import { useMutation, useQuery } from 'react-query'
import axios from 'axios'
import { useAuthStore } from '../../store/authStore'

interface QueryRequest {
  query: string
  domain: string
  compression_level: number
  max_results: number
  include_metadata: boolean
  streaming_mode: boolean
}

interface QueryResult {
  id: string
  query: string
  compressed_content: string
  original_content: string
  compression_ratio: number
  domain: string
  metadata: {
    processing_time: number
    tokens_processed: number
    compression_algorithm: string
    quality_score: number
  }
  created_at: string
}

const DOMAINS = [
  'general',
  'genomics',
  'finance',
  'cybersecurity',
  'legal',
  'manufacturing',
  'startup',
  'healthcare',
  'education',
  'research',
]

const COMPRESSION_LEVELS = [
  { value: 1, label: 'Minimal (1)' },
  { value: 3, label: 'Light (3)' },
  { value: 5, label: 'Medium (5)' },
  { value: 7, label: 'High (7)' },
  { value: 10, label: 'Maximum (10)' },
]

export default function QueryInterface() {
  const [queryResults, setQueryResults] = useState<QueryResult[]>([])
  const [queryHistory, setQueryHistory] = useState<QueryResult[]>([])
  const [activeTab, setActiveTab] = useState(0)
  const [filters, setFilters] = useState({
    domain: '',
    dateRange: '',
    compressionRatio: '',
  })

  const { user } = useAuthStore()

  const {
    control,
    handleSubmit,
    reset,
    watch,
    formState: { errors },
  } = useForm<QueryRequest>({
    defaultValues: {
      query: '',
      domain: 'general',
      compression_level: 5,
      max_results: 10,
      include_metadata: true,
      streaming_mode: false,
    },
  })

  const watchedCompressionLevel = watch('compression_level')
  const watchedDomain = watch('domain')

  // Query history
  const { data: historyData } = useQuery('query-history', async () => {
    const response = await axios.get('/queries/history')
    return response.data
  })

  useEffect(() => {
    if (historyData) {
      setQueryHistory(historyData)
    }
  }, [historyData])

  // Execute query mutation
  const executeQueryMutation = useMutation(
    async (data: QueryRequest) => {
      const response = await axios.post('/queries/execute', data)
      return response.data
    },
    {
      onSuccess: (result) => {
        setQueryResults([result, ...queryResults])
        reset()
      },
    }
  )

  // Streaming query mutation
  const executeStreamingQueryMutation = useMutation(
    async (data: QueryRequest) => {
      const response = await axios.post('/queries/stream', data, {
        responseType: 'stream',
      })
      return response.data
    },
    {
      onSuccess: (result) => {
        // Handle streaming results
        console.log('Streaming results:', result)
      },
    }
  )

  const onSubmit = (data: QueryRequest) => {
    if (data.streaming_mode) {
      executeStreamingQueryMutation.mutate(data)
    } else {
      executeQueryMutation.mutate(data)
    }
  }

  const handleCopyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text)
  }

  const handleDownloadResult = (result: QueryResult) => {
    const blob = new Blob([result.compressed_content], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `query_result_${result.id}.txt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const getCompressionLevelColor = (level: number) => {
    if (level <= 3) return 'success'
    if (level <= 7) return 'warning'
    return 'error'
  }

  const filteredHistory = queryHistory.filter((item) => {
    if (filters.domain && item.domain !== filters.domain) return false
    return true
  })

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Interactive Query Interface
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Execute compression queries with advanced options and real-time processing
      </Typography>

      <Grid container spacing={3}>
        {/* Query Form */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <SearchIcon />
                Query Configuration
              </Typography>

              <Box component="form" onSubmit={handleSubmit(onSubmit)}>
                <Controller
                  name="query"
                  control={control}
                  rules={{ required: 'Query is required' }}
                  render={({ field }) => (
                    <TextField
                      {...field}
                      fullWidth
                      multiline
                      rows={4}
                      label="Query Content"
                      placeholder="Enter the content you want to compress..."
                      error={!!errors.query}
                      helperText={errors.query?.message}
                      margin="normal"
                    />
                  )}
                />

                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={12} sm={6}>
                    <Controller
                      name="domain"
                      control={control}
                      render={({ field }) => (
                        <FormControl fullWidth>
                          <InputLabel>Domain</InputLabel>
                          <Select {...field} label="Domain">
                            {DOMAINS.map((domain) => (
                              <MenuItem key={domain} value={domain}>
                                {domain.charAt(0).toUpperCase() + domain.slice(1)}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      )}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <Controller
                      name="compression_level"
                      control={control}
                      render={({ field }) => (
                        <FormControl fullWidth>
                          <InputLabel>Compression Level</InputLabel>
                          <Select {...field} label="Compression Level">
                            {COMPRESSION_LEVELS.map((level) => (
                              <MenuItem key={level.value} value={level.value}>
                                {level.label}
                              </MenuItem>
                            ))}
                          </Select>
                        </FormControl>
                      )}
                    />
                  </Grid>
                </Grid>

                <Grid container spacing={2} sx={{ mt: 1 }}>
                  <Grid item xs={12} sm={6}>
                    <Controller
                      name="max_results"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          fullWidth
                          type="number"
                          label="Max Results"
                          inputProps={{ min: 1, max: 100 }}
                        />
                      )}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 1 }}>
                      <Controller
                        name="include_metadata"
                        control={control}
                        render={({ field }) => (
                          <FormControlLabel
                            control={<Switch {...field} checked={field.value} />}
                            label="Include Metadata"
                          />
                        )}
                      />

                      <Controller
                        name="streaming_mode"
                        control={control}
                        render={({ field }) => (
                          <FormControlLabel
                            control={<Switch {...field} checked={field.value} />}
                            label="Streaming Mode"
                          />
                        )}
                      />
                    </Box>
                  </Grid>
                </Grid>

                <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                  <Button
                    type="submit"
                    variant="contained"
                    startIcon={<PlayIcon />}
                    disabled={executeQueryMutation.isLoading || executeStreamingQueryMutation.isLoading}
                    sx={{ flex: 1 }}
                  >
                    {executeStreamingQueryMutation.isLoading ? (
                      <CircularProgress size={24} />
                    ) : watchedStreamingMode ? (
                      'Stream Query'
                    ) : (
                      'Execute Query'
                    )}
                  </Button>

                  <Button
                    variant="outlined"
                    startIcon={<ClearIcon />}
                    onClick={() => reset()}
                  >
                    Clear
                  </Button>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Results */}
        <Grid item xs={12} lg={6}>
          <Card>
            <CardContent>
              <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
                <Tab label="Current Results" />
                <Tab label="Query History" />
              </Tabs>

              <Box sx={{ mt: 2 }}>
                {activeTab === 0 && (
                  <Box>
                    {queryResults.length === 0 ? (
                      <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4 }}>
                        No results yet. Execute a query to see results here.
                      </Typography>
                    ) : (
                      queryResults.map((result) => (
                        <Accordion key={result.id}>
                          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, width: '100%' }}>
                              <Chip
                                label={`${result.compression_ratio.toFixed(2)}x`}
                                color="primary"
                                size="small"
                              />
                              <Typography variant="body2" sx={{ flex: 1 }}>
                                {result.domain} domain
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                {new Date(result.created_at).toLocaleTimeString()}
                              </Typography>
                            </Box>
                          </AccordionSummary>
                          <AccordionDetails>
                            <Typography variant="subtitle2" gutterBottom>
                              Original Content:
                            </Typography>
                            <Paper sx={{ p: 2, mb: 2, bgcolor: 'grey.100', maxHeight: 200, overflow: 'auto' }}>
                              <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap' }}>
                                {result.original_content}
                              </Typography>
                            </Paper>

                            <Typography variant="subtitle2" gutterBottom>
                              Compressed Content:
                            </Typography>
                            <Paper sx={{ p: 2, mb: 2, bgcolor: 'primary.light', maxHeight: 200, overflow: 'auto' }}>
                              <Typography variant="body2" sx={{ whiteSpace: 'pre-wrap', color: 'primary.contrastText' }}>
                                {result.compressed_content}
                              </Typography>
                            </Paper>

                            <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                              <Button
                                size="small"
                                startIcon={<CopyIcon />}
                                onClick={() => handleCopyToClipboard(result.compressed_content)}
                              >
                                Copy
                              </Button>
                              <Button
                                size="small"
                                startIcon={<DownloadIcon />}
                                onClick={() => handleDownloadResult(result)}
                              >
                                Download
                              </Button>
                            </Box>

                            {result.metadata && (
                              <Box>
                                <Typography variant="subtitle2" gutterBottom>
                                  Metadata:
                                </Typography>
                                <Grid container spacing={1}>
                                  <Grid item xs={6}>
                                    <Typography variant="caption">
                                      Processing Time: {result.metadata.processing_time}ms
                                    </Typography>
                                  </Grid>
                                  <Grid item xs={6}>
                                    <Typography variant="caption">
                                      Tokens: {result.metadata.tokens_processed}
                                    </Typography>
                                  </Grid>
                                  <Grid item xs={6}>
                                    <Typography variant="caption">
                                      Algorithm: {result.metadata.compression_algorithm}
                                    </Typography>
                                  </Grid>
                                  <Grid item xs={6}>
                                    <Typography variant="caption">
                                      Quality Score: {result.metadata.quality_score.toFixed(2)}
                                    </Typography>
                                  </Grid>
                                </Grid>
                              </Box>
                            )}
                          </AccordionDetails>
                        </Accordion>
                      ))
                    )}
                  </Box>
                )}

                {activeTab === 1 && (
                  <Box>
                    <Box sx={{ mb: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        Filters:
                      </Typography>
                      <Grid container spacing={2}>
                        <Grid item xs={4}>
                          <TextField
                            size="small"
                            label="Domain"
                            value={filters.domain}
                            onChange={(e) => setFilters({ ...filters, domain: e.target.value })}
                          />
                        </Grid>
                      </Grid>
                    </Box>

                    <List>
                      {filteredHistory.map((result) => (
                        <ListItem key={result.id} divider>
                          <ListItemIcon>
                            <SearchIcon />
                          </ListItemIcon>
                          <ListItemText
                            primary={result.query.substring(0, 100) + '...'}
                            secondary={
                              <Box>
                                <Typography variant="caption" color="text.secondary">
                                  {result.domain} • {new Date(result.created_at).toLocaleString()}
                                </Typography>
                                <Box sx={{ mt: 1 }}>
                                  <Chip
                                    label={`${result.compression_ratio.toFixed(2)}x`}
                                    size="small"
                                    color="primary"
                                  />
                                </Box>
                              </Box>
                            }
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}