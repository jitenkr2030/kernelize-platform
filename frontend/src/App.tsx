import React from 'react'
import { Routes, Route } from 'react-router-dom'
import { Box } from '@mui/material'

import Layout from './components/Layout/Layout'
import Dashboard from './pages/Dashboard/Dashboard'
import CompressionJobs from './pages/CompressionJobs/CompressionJobs'
import QueryInterface from './pages/QueryInterface/QueryInterface'
import UserManagement from './pages/UserManagement/UserManagement'
import Analytics from './pages/Analytics/Analytics'
import KnowledgeGraph from './pages/KnowledgeGraph/KnowledgeGraph'
import SystemMonitor from './pages/SystemMonitor/SystemMonitor'
import Settings from './pages/Settings/Settings'
import Login from './pages/Auth/Login'
import { useAuthStore } from './store/authStore'
import { ProtectedRoute } from './components/Auth/ProtectedRoute'

function App() {
  const { isAuthenticated } = useAuthStore()

  if (!isAuthenticated) {
    return (
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="*" element={<Login />} />
      </Routes>
    )
  }

  return (
    <Layout>
      <Box sx={{ flex: 1, overflow: 'auto' }}>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/compression-jobs" element={<CompressionJobs />} />
          <Route path="/query-interface" element={<QueryInterface />} />
          <Route path="/knowledge-graph" element={<KnowledgeGraph />} />
          <Route path="/analytics" element={<Analytics />} />
          <Route path="/system-monitor" element={<SystemMonitor />} />
          <Route path="/user-management" element={
            <ProtectedRoute requiredRole="admin">
              <UserManagement />
            </ProtectedRoute>
          } />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Box>
    </Layout>
  )
}

export default App