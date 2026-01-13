import React, { useState, useRef, useEffect } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Button,
  Chip,
  Paper,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  List,
  ListItem,
  ListItemText,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material'
import {
  Search as SearchIcon,
  ZoomIn as ZoomInIcon,
  ZoomOut as ZoomOutIcon,
  CenterFocusStrong as CenterIcon,
  ExpandMore as ExpandMoreIcon,
  AccountTree as GraphIcon,
} from '@mui/icons-material'
import * as d3 from 'd3'
import { useQuery } from 'react-query'
import axios from 'axios'

interface GraphNode {
  id: string
  label: string
  type: 'concept' | 'entity' | 'relation'
  domain: string
  importance: number
  x?: number
  y?: number
}

interface GraphLink {
  source: string
  target: string
  type: string
  strength: number
}

interface GraphData {
  nodes: GraphNode[]
  links: GraphLink[]
}

const DOMAINS = ['general', 'genomics', 'finance', 'cybersecurity', 'legal', 'healthcare', 'education']

export default function KnowledgeGraph() {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] })
  const [selectedDomain, setSelectedDomain] = useState('all')
  const [searchTerm, setSearchTerm] = useState('')
  const [minImportance, setMinImportance] = useState(0)
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null)
  
  const svgRef = useRef<SVGSVGElement>(null)
  const simulationRef = useRef<d3.Simulation<GraphNode, undefined> | null>(null)

  const { data, isLoading } = useQuery<GraphData>(
    ['knowledge-graph', selectedDomain, minImportance],
    async () => {
      const response = await axios.get('/knowledge-graph', {
        params: {
          domain: selectedDomain,
          min_importance: minImportance,
        },
      })
      return response.data
    },
    {
      refetchInterval: 30000,
    }
  )

  useEffect(() => {
    if (data) {
      setGraphData(data)
      if (svgRef.current) {
        renderGraph(data)
      }
    }
  }, [data])

  const renderGraph = (data: GraphData) => {
    if (!svgRef.current) return

    // Clear previous graph
    d3.select(svgRef.current).selectAll("*").remove()

    const width = 800
    const height = 600
    const margin = { top: 20, right: 20, bottom: 20, left: 20 }

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Color scale for domains
    const colorScale = d3.scaleOrdinal(d3.schemeCategory10)

    // Create simulation
    const simulation = d3.forceSimulation<GraphNode>(data.nodes)
      .force('link', d3.forceLink<GraphNode, GraphLink>(data.links)
        .id(d => d.id)
        .distance(100)
        .strength(0.5))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30))

    simulationRef.current = simulation

    // Create links
    const links = g.append('g')
      .selectAll('line')
      .data(data.links)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', d => Math.sqrt(d.strength * 10))

    // Create nodes
    const nodes = g.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .enter()
      .append('circle')
      .attr('r', d => Math.sqrt(d.importance) * 10 + 5)
      .attr('fill', d => colorScale(d.domain))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .call(d3.drag<SVGCircleElement, GraphNode>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended))
      .on('click', (event, d) => {
        setSelectedNode(d)
      })

    // Add labels
    const labels = g.append('g')
      .selectAll('text')
      .data(data.nodes)
      .enter()
      .append('text')
      .text(d => d.label)
      .attr('font-size', '12px')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('fill', '#fff')
      .style('pointer-events', 'none')

    // Update positions on tick
    simulation.on('tick', () => {
      links
        .attr('x1', d => (d.source as any).x)
        .attr('y1', d => (d.source as any).y)
        .attr('x2', d => (d.target as any).x)
        .attr('y2', d => (d.target as any).y)

      nodes
        .attr('cx', d => (d as any).x)
        .attr('cy', d => (d as any).y)

      labels
        .attr('x', d => (d as any).x)
        .attr('y', d => (d as any).y + 25)
    })

    function dragstarted(event: any, d: GraphNode) {
      if (!event.active) simulation.alphaTarget(0.3).restart()
      d.fx = d.x
      d.fy = d.y
    }

    function dragged(event: any, d: GraphNode) {
      d.fx = event.x
      d.fy = event.y
    }

    function dragended(event: any, d: GraphNode) {
      if (!event.active) simulation.alphaTarget(0)
      d.fx = null
      d.fy = null
    }

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 10])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)
  }

  const handleZoomIn = () => {
    if (svgRef.current) {
      d3.select(svgRef.current).transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 1.5
      )
    }
  }

  const handleZoomOut = () => {
    if (svgRef.current) {
      d3.select(svgRef.current).transition().call(
        d3.zoom<SVGSVGElement, unknown>().scaleBy as any, 0.75
      )
    }
  }

  const handleCenter = () => {
    if (svgRef.current) {
      d3.select(svgRef.current).transition().call(
        d3.zoom<SVGSVGElement, unknown>().transform as any,
        d3.zoomIdentity
      )
    }
  }

  const handleSearch = () => {
    // Implement search functionality
    console.log('Searching for:', searchTerm)
  }

  const getNodeTypeColor = (type: string) => {
    switch (type) {
      case 'concept':
        return 'primary'
      case 'entity':
        return 'secondary'
      case 'relation':
        return 'success'
      default:
        return 'default'
    }
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Knowledge Graph Visualization
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Explore relationships between compressed knowledge concepts and entities
      </Typography>

      <Grid container spacing={3}>
        {/* Controls */}
        <Grid item xs={12} lg={3}>
          <Card sx={{ mb: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Graph Controls
              </Typography>
              
              <TextField
                fullWidth
                label="Search Nodes"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                InputProps={{
                  endAdornment: (
                    <Button onClick={handleSearch} size="small">
                      <SearchIcon />
                    </Button>
                  ),
                }}
                sx={{ mb: 2 }}
              />

              <FormControl fullWidth sx={{ mb: 2 }}>
                <InputLabel>Domain Filter</InputLabel>
                <Select
                  value={selectedDomain}
                  label="Domain Filter"
                  onChange={(e) => setSelectedDomain(e.target.value)}
                >
                  <MenuItem value="all">All Domains</MenuItem>
                  {DOMAINS.map(domain => (
                    <MenuItem key={domain} value={domain}>
                      {domain.charAt(0).toUpperCase() + domain.slice(1)}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Typography variant="body2" gutterBottom>
                Minimum Importance: {minImportance}
              </Typography>
              <Slider
                value={minImportance}
                onChange={(_, value) => setMinImportance(value as number)}
                min={0}
                max={1}
                step={0.1}
                valueLabelDisplay="auto"
                sx={{ mb: 2 }}
              />

              <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleZoomIn}
                  startIcon={<ZoomInIcon />}
                >
                  Zoom In
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={handleZoomOut}
                  startIcon={<ZoomOutIcon />}
                >
                  Zoom Out
                </Button>
              </Box>

              <Button
                fullWidth
                variant="outlined"
                onClick={handleCenter}
                startIcon={<CenterIcon />}
              >
                Center View
              </Button>
            </CardContent>
          </Card>

          {/* Selected Node Details */}
          {selectedNode && (
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Node Details
                </Typography>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Label:</Typography>
                  <Typography variant="body1">{selectedNode.label}</Typography>
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Type:</Typography>
                  <Chip
                    label={selectedNode.type}
                    color={getNodeTypeColor(selectedNode.type) as any}
                    size="small"
                  />
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Domain:</Typography>
                  <Chip
                    label={selectedNode.domain}
                    variant="outlined"
                    size="small"
                  />
                </Box>
                <Box sx={{ mb: 2 }}>
                  <Typography variant="subtitle2">Importance:</Typography>
                  <Typography variant="body1">
                    {(selectedNode.importance * 100).toFixed(1)}%
                  </Typography>
                </Box>
              </CardContent>
            </Card>
          )}
        </Grid>

        {/* Graph Visualization */}
        <Grid item xs={12} lg={9}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">
                  Knowledge Graph
                </Typography>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <Chip label="Concepts" color="primary" size="small" />
                  <Chip label="Entities" color="secondary" size="small" />
                  <Chip label="Relations" color="success" size="small" />
                </Box>
              </Box>

              <Box sx={{ 
                border: '1px solid rgba(99, 102, 241, 0.2)', 
                borderRadius: 1,
                backgroundColor: '#0f172a',
                display: 'flex',
                justifyContent: 'center',
                alignItems: 'center',
                minHeight: '600px'
              }}>
                {isLoading ? (
                  <Typography>Loading graph...</Typography>
                ) : (
                  <svg ref={svgRef}></svg>
                )}
              </Box>
            </CardContent>
          </Card>

          {/* Graph Statistics */}
          <Card sx={{ mt: 3 }}>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Graph Statistics
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="primary">
                      {graphData.nodes.length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Total Nodes
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="secondary">
                      {graphData.links.length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Total Links
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="success.main">
                      {graphData.nodes.filter(n => n.type === 'concept').length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Concepts
                    </Typography>
                  </Box>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography variant="h4" color="warning.main">
                      {graphData.nodes.filter(n => n.type === 'entity').length}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      Entities
                    </Typography>
                  </Box>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}