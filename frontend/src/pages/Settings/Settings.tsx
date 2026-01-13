import React, { useState } from 'react'
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Chip,
  Tabs,
  Tab,
  Paper,
  Avatar,
} from '@mui/material'
import {
  Save as SaveIcon,
  Add as AddIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  VpnKey as ApiKeyIcon,
  Notifications as NotificationIcon,
  Security as SecurityIcon,
  Palette as ThemeIcon,
} from '@mui/icons-material'
import { useForm, Controller } from 'react-hook-form'
import { useMutation, useQuery, useQueryClient } from 'react-query'
import { useAuthStore } from '../../store/authStore'
import axios from 'axios'

interface SettingsForm {
  full_name: string
  organization: string
  email_notifications: boolean
  system_alerts: boolean
  marketing_emails: boolean
  theme: 'light' | 'dark' | 'auto'
  language: string
  timezone: string
}

interface APIKey {
  id: string
  name: string
  key: string
  created_at: string
  last_used?: string
  expires_at?: string
  is_active: boolean
}

export default function Settings() {
  const [activeTab, setActiveTab] = useState(0)
  const [apiKeyDialog, setApiKeyDialog] = useState(false)
  const [newApiKeyName, setNewApiKeyName] = useState('')
  const [editingApiKey, setEditingApiKey] = useState<APIKey | null>(null)
  
  const { user, updateUser } = useAuthStore()
  const queryClient = useQueryClient()

  const { control, handleSubmit, reset, watch, formState: { errors } } = useForm<SettingsForm>({
    defaultValues: {
      full_name: user?.full_name || '',
      organization: user?.organization || '',
      email_notifications: true,
      system_alerts: true,
      marketing_emails: false,
      theme: 'dark',
      language: 'en',
      timezone: 'UTC',
    }
  })

  const { data: apiKeys } = useQuery<APIKey[]>(
    'api-keys',
    async () => {
      const response = await axios.get('/settings/api-keys')
      return response.data
    }
  )

  const updateProfileMutation = useMutation(
    async (data: Partial<SettingsForm>) => {
      const response = await axios.put('/settings/profile', data)
      return response.data
    },
    {
      onSuccess: (result) => {
        updateUser(result)
        queryClient.invalidateQueries('user')
      },
    }
  )

  const updateNotificationsMutation = useMutation(
    async (data: Partial<SettingsForm>) => {
      const response = await axios.put('/settings/notifications', data)
      return response.data
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('user')
      },
    }
  )

  const createApiKeyMutation = useMutation(
    async (name: string) => {
      const response = await axios.post('/settings/api-keys', { name })
      return response.data
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('api-keys')
        setApiKeyDialog(false)
        setNewApiKeyName('')
      },
    }
  )

  const deleteApiKeyMutation = useMutation(
    async (keyId: string) => {
      await axios.delete(`/settings/api-keys/${keyId}`)
    },
    {
      onSuccess: () => {
        queryClient.invalidateQueries('api-keys')
      },
    }
  )

  const onSubmit = (data: SettingsForm) => {
    if (activeTab === 0) {
      updateProfileMutation.mutate(data)
    } else if (activeTab === 1) {
      updateNotificationsMutation.mutate(data)
    }
  }

  const handleCreateApiKey = () => {
    if (newApiKeyName.trim()) {
      createApiKeyMutation.mutate(newApiKeyName.trim())
    }
  }

  const handleDeleteApiKey = (keyId: string) => {
    if (window.confirm('Are you sure you want to delete this API key?')) {
      deleteApiKeyMutation.mutate(keyId)
    }
  }

  const maskApiKey = (key: string) => {
    if (key.length <= 8) return key
    return key.substring(0, 4) + '...' + key.substring(key.length - 4)
  }

  return (
    <Box sx={{ flexGrow: 1 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      <Typography variant="body1" color="text.secondary" sx={{ mb: 4 }}>
        Manage your account settings, preferences, and API keys
      </Typography>

      <Card>
        <CardContent>
          <Tabs value={activeTab} onChange={(_, newValue) => setActiveTab(newValue)}>
            <Tab label="Profile" icon={<SecurityIcon />} />
            <Tab label="Notifications" icon={<NotificationIcon />} />
            <Tab label="API Keys" icon={<ApiKeyIcon />} />
            <Tab label="Appearance" icon={<ThemeIcon />} />
          </Tabs>

          <Box sx={{ mt: 3 }}>
            {activeTab === 0 && (
              <Box component="form" onSubmit={handleSubmit(onSubmit)}>
                <Grid container spacing={3}>
                  <Grid item xs={12}>
                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                      <Avatar sx={{ width: 80, height: 80, mr: 3 }}>
                        {user?.full_name?.[0] || user?.username?.[0] || 'U'}
                      </Avatar>
                      <Box>
                        <Typography variant="h6">
                          {user?.full_name || user?.username}
                        </Typography>
                        <Typography variant="body2" color="text.secondary">
                          {user?.email}
                        </Typography>
                        <Chip
                          label={user?.role}
                          size="small"
                          color="primary"
                          variant="outlined"
                          sx={{ mt: 1 }}
                        />
                      </Box>
                    </Box>
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <Controller
                      name="full_name"
                      control={control}
                      rules={{ required: 'Full name is required' }}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          fullWidth
                          label="Full Name"
                          error={!!errors.full_name}
                          helperText={errors.full_name?.message}
                        />
                      )}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <Controller
                      name="organization"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          fullWidth
                          label="Organization"
                        />
                      )}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Email"
                      value={user?.email || ''}
                      disabled
                      helperText="Email cannot be changed. Contact admin if needed."
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <TextField
                      fullWidth
                      label="Username"
                      value={user?.username || ''}
                      disabled
                      helperText="Username cannot be changed."
                    />
                  </Grid>

                  <Grid item xs={12}>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="h6" gutterBottom>
                      Statistics
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} sm={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h4" color="primary">
                            {user?.total_kernels_created || 0}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Kernels Created
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h4" color="secondary">
                            {user?.total_queries_executed || 0}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Queries Executed
                          </Typography>
                        </Paper>
                      </Grid>
                      <Grid item xs={12} sm={4}>
                        <Paper sx={{ p: 2, textAlign: 'center' }}>
                          <Typography variant="h4" color="success.main">
                            {user?.total_compressed_mb?.toFixed(1) || 0} MB
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Data Compressed
                          </Typography>
                        </Paper>
                      </Grid>
                    </Grid>
                  </Grid>

                  <Grid item xs={12}>
                    <Button
                      type="submit"
                      variant="contained"
                      startIcon={<SaveIcon />}
                      disabled={updateProfileMutation.isLoading}
                    >
                      Save Changes
                    </Button>
                  </Grid>
                </Grid>
              </Box>
            )}

            {activeTab === 1 && (
              <Box component="form" onSubmit={handleSubmit(onSubmit)}>
                <Typography variant="h6" gutterBottom>
                  Notification Preferences
                </Typography>
                
                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <Controller
                    name="email_notifications"
                    control={control}
                    render={({ field }) => (
                      <FormControlLabel
                        control={<Switch {...field} checked={field.value} />}
                        label={
                          <Box>
                            <Typography variant="body1">Email Notifications</Typography>
                            <Typography variant="body2" color="text.secondary">
                              Receive notifications via email about important updates
                            </Typography>
                          </Box>
                        }
                      />
                    )}
                  />

                  <Controller
                    name="system_alerts"
                    control={control}
                    render={({ field }) => (
                      <FormControlLabel
                        control={<Switch {...field} checked={field.value} />}
                        label={
                          <Box>
                            <Typography variant="body1">System Alerts</Typography>
                            <Typography variant="body2" color="text.secondary">
                              Get notified about system status and maintenance
                            </Typography>
                          </Box>
                        }
                      />
                    )}
                  />

                  <Controller
                    name="marketing_emails"
                    control={control}
                    render={({ field }) => (
                      <FormControlLabel
                        control={<Switch {...field} checked={field.value} />}
                        label={
                          <Box>
                            <Typography variant="body1">Marketing Communications</Typography>
                            <Typography variant="body2" color="text.secondary">
                              Receive updates about new features and product announcements
                            </Typography>
                          </Box>
                        }
                      />
                    )}
                  />
                </Box>

                <Box sx={{ mt: 3 }}>
                  <Button
                    type="submit"
                    variant="contained"
                    startIcon={<SaveIcon />}
                    disabled={updateNotificationsMutation.isLoading}
                  >
                    Save Preferences
                  </Button>
                </Box>
              </Box>
            )}

            {activeTab === 2 && (
              <Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
                  <Typography variant="h6">
                    API Keys
                  </Typography>
                  <Button
                    variant="contained"
                    startIcon={<AddIcon />}
                    onClick={() => setApiKeyDialog(true)}
                  >
                    Create New Key
                  </Button>
                </Box>

                {apiKeys?.length === 0 ? (
                  <Alert severity="info">
                    No API keys created yet. Create your first API key to start using the KERNELIZE API.
                  </Alert>
                ) : (
                  <List>
                    {apiKeys?.map((apiKey) => (
                      <ListItem key={apiKey.id} divider>
                        <ListItemText
                          primary={
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                              <Typography variant="body1">
                                {apiKey.name}
                              </Typography>
                              <Chip
                                label={apiKey.is_active ? 'Active' : 'Inactive'}
                                color={apiKey.is_active ? 'success' : 'default'}
                                size="small"
                              />
                            </Box>
                          }
                          secondary={
                            <Box>
                              <Typography variant="body2" fontFamily="monospace">
                                {maskApiKey(apiKey.key)}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                Created: {new Date(apiKey.created_at).toLocaleString()}
                                {apiKey.last_used && ` • Last used: ${new Date(apiKey.last_used).toLocaleString()}`}
                                {apiKey.expires_at && ` • Expires: ${new Date(apiKey.expires_at).toLocaleString()}`}
                              </Typography>
                            </Box>
                          }
                        />
                        <ListItemSecondaryAction>
                          <IconButton
                            edge="end"
                            onClick={() => handleDeleteApiKey(apiKey.id)}
                            color="error"
                          >
                            <DeleteIcon />
                          </IconButton>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                )}
              </Box>
            )}

            {activeTab === 3 && (
              <Box>
                <Typography variant="h6" gutterBottom>
                  Appearance Settings
                </Typography>
                
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={6}>
                    <Controller
                      name="theme"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          select
                          fullWidth
                          label="Theme"
                          SelectProps={{ native: true }}
                        >
                          <option value="light">Light</option>
                          <option value="dark">Dark</option>
                          <option value="auto">Auto (System)</option>
                        </TextField>
                      )}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <Controller
                      name="language"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          select
                          fullWidth
                          label="Language"
                          SelectProps={{ native: true }}
                        >
                          <option value="en">English</option>
                          <option value="es">Español</option>
                          <option value="fr">Français</option>
                          <option value="de">Deutsch</option>
                          <option value="zh">中文</option>
                          <option value="ja">日本語</option>
                        </TextField>
                      )}
                    />
                  </Grid>

                  <Grid item xs={12} sm={6}>
                    <Controller
                      name="timezone"
                      control={control}
                      render={({ field }) => (
                        <TextField
                          {...field}
                          select
                          fullWidth
                          label="Timezone"
                          SelectProps={{ native: true }}
                        >
                          <option value="UTC">UTC</option>
                          <option value="America/New_York">Eastern Time</option>
                          <option value="America/Chicago">Central Time</option>
                          <option value="America/Denver">Mountain Time</option>
                          <option value="America/Los_Angeles">Pacific Time</option>
                          <option value="Europe/London">London</option>
                          <option value="Europe/Paris">Paris</option>
                          <option value="Asia/Tokyo">Tokyo</option>
                        </TextField>
                      )}
                    />
                  </Grid>
                </Grid>

                <Box sx={{ mt: 3 }}>
                  <Alert severity="info">
                    Theme and language changes will be applied immediately. Timezone changes will affect all timestamps in the application.
                  </Alert>
                </Box>
              </Box>
            )}
          </Box>
        </CardContent>
      </Card>

      {/* Create API Key Dialog */}
      <Dialog open={apiKeyDialog} onClose={() => setApiKeyDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Create New API Key</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            fullWidth
            label="Key Name"
            value={newApiKeyName}
            onChange={(e) => setNewApiKeyName(e.target.value)}
            placeholder="e.g., Production API Key"
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setApiKeyDialog(false)}>Cancel</Button>
          <Button
            onClick={handleCreateApiKey}
            variant="contained"
            disabled={!newApiKeyName.trim() || createApiKeyMutation.isLoading}
          >
            Create Key
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}