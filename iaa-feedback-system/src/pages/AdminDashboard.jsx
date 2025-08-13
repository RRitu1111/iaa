import { useState, useEffect, useMemo, useCallback } from 'react'
import { motion, useAnimation } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import {
  Plane,
  Settings,
  FileText,
  BarChart3,
  Users,
  Calendar,
  Plus,
  Search,
  Trash2,
  Clock,
  CheckCircle,
  XCircle,
  BarChart,
  TrendingUp,
  Eye,
  EyeOff,
  LogOut
} from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import { useNotifications, NotificationBell } from '../components/NotificationSystem'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header'
import Footer from '../components/Footer'
import adminService from '../api/adminService'
import { API_BASE_URL } from '../api/config'
import { formatDate, formatDateTime, getRelativeTime } from '../utils/dateUtils'
import './AdminDashboard.css'

const AdminDashboard = ({ theme, toggleTheme }) => {
  const [activeTab, setActiveTab] = useState('overview')
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [pendingRequests, setPendingRequests] = useState([])
  const [deletionRequests, setDeletionRequests] = useState([])
  const [departments, setDepartments] = useState([])
  const [trainers, setTrainers] = useState([])
  const [forms, setForms] = useState([])
  const [dashboardStats, setDashboardStats] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [reportSettings, setReportSettings] = useState({ auto_generation_delay_days: { value: '1' } })
  const [settingsLoading, setSettingsLoading] = useState(false)

  const { t } = useLanguage()
  const navigate = useNavigate()
  const planeControls = useAnimation()
  const { ref, inView } = useInView({
    threshold: 0.1,
    triggerOnce: true
  })

  // Helper function to get due date status and styling
  const getDueDateStatus = (dueDate) => {
    if (!dueDate) return { status: 'none', className: '', text: 'No due date' }

    const now = new Date()
    const due = new Date(dueDate)
    const timeDiff = due.getTime() - now.getTime()
    const daysDiff = Math.ceil(timeDiff / (1000 * 3600 * 24))

    if (timeDiff < 0) {
      return {
        status: 'overdue',
        className: 'due-overdue',
        text: `Overdue by ${Math.abs(daysDiff)} day${Math.abs(daysDiff) !== 1 ? 's' : ''}`
      }
    } else if (daysDiff <= 2) {
      return {
        status: 'soon',
        className: 'due-soon',
        text: `Due in ${daysDiff} day${daysDiff !== 1 ? 's' : ''}`
      }
    } else {
      return {
        status: 'normal',
        className: 'due-normal',
        text: `Due ${due.toLocaleDateString()}`
      }
    }
  }

  // Scroll to top when page loads
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  // Load cached data from localStorage
  const loadCachedData = () => {
    try {
      const cached = localStorage.getItem('admin_dashboard_cache')
      if (cached) {
        const data = JSON.parse(cached)
        const cacheAge = Date.now() - data.timestamp

        // Use cache if less than 5 minutes old
        if (cacheAge < 5 * 60 * 1000) {
          setDepartments(data.departments || [])
          setTrainers(data.trainers || [])
          setForms(data.forms || [])
          setPendingRequests(data.requests || [])
          setDeletionRequests(data.deletionRequests || [])
          setDashboardStats(data.stats || {})
          return true
        }
      }
    } catch (error) {
      // Error loading cached data
    }
    return false
  }

  // Save data to localStorage cache
  const saveCachedData = (departments, trainers, forms, requests, deletionRequests, stats) => {
    try {
      const cacheData = {
        departments,
        trainers,
        forms,
        requests,
        deletionRequests,
        stats,
        timestamp: Date.now()
      }
      localStorage.setItem('admin_dashboard_cache', JSON.stringify(cacheData))
    } catch (error) {
      // Error saving cached data
    }
  }

  // Centralized data fetching function
  const fetchDashboardData = useCallback(async (useCache = true) => {
    try {
      setLoading(true)
      setError(null)

      // Try to load from cache first
      if (useCache && loadCachedData()) {
        setLoading(false)
        return
      }



      // Fetch all required data including real-time stats
      const [departmentsData, trainersData, formsData, requestsData, deletionRequestsData, statsData] = await Promise.all([
        adminService.getDepartments(),
        adminService.getTrainers(),
        adminService.getForms(),
        adminService.getFormRequests(),
        adminService.getFormDeletionRequests(),
        adminService.getDashboardStats()
      ])

      setDepartments(departmentsData)
      setTrainers(trainersData)
      setForms(formsData)
      setPendingRequests(requestsData)
      setDeletionRequests(deletionRequestsData)

      // Store real-time stats
      setDashboardStats(statsData)

      // Cache the data
      saveCachedData(departmentsData, trainersData, formsData, requestsData, deletionRequestsData, statsData)



    } catch (error) {
      setError(error.message)

      // Try to load cached data as fallback
      if (!loadCachedData()) {
        // Set fallback empty states to prevent undefined errors
        setDepartments([])
        setTrainers([])
        setForms([])
        setPendingRequests([])
        setDeletionRequests([])
        setDashboardStats({
          totalForms: 0,
          totalResponses: 0,
          totalUsers: 0,
          avgRating: 0,
          pendingRequests: 0,
          activeForms: 0
        })
      }
    } finally {
      setLoading(false)
    }
  }, [])

  // Memoized dashboard stats for performance
  const finalDashboardStats = useMemo(() => {
    return dashboardStats || {
      totalForms: forms.length,
      activeForms: forms.filter(form => form.is_published).length,
      draftForms: forms.filter(form => !form.is_published).length,
      totalResponses: forms.reduce((sum, form) => sum + (form.response_count || 0), 0),
      pendingRequests: pendingRequests.filter(req => req.status === 'pending').length,
      totalDepartments: departments.length,
      totalTrainers: trainers.length
    }
  }, [dashboardStats, forms, trainers, pendingRequests, deletionRequests, departments])

  // Fetch data from backend
  useEffect(() => {
    // Check if we're returning from form builder or if cache should be invalidated
    const shouldInvalidateCache = sessionStorage.getItem('invalidate_admin_cache')
    if (shouldInvalidateCache) {
      localStorage.removeItem('admin_dashboard_cache')
      sessionStorage.removeItem('invalidate_admin_cache')
    }

    fetchDashboardData()
  }, [fetchDashboardData])

  // Airplane animation when page loads
  useEffect(() => {
    if (inView) {
      planeControls.start({
        x: ['-100%', '20%'],
        y: ['0%', '0%'],
        rotate: [0, 0],
        scale: [0.8, 1],
        transition: {
          duration: 2,
          ease: "easeOut"
        }
      })
    }
  }, [inView, planeControls])

  // Dashboard stats now handled by memoized version above

  // Get recent forms from real data (last 5 forms)
  const recentForms = forms
    .sort((a, b) => new Date(b.created_at) - new Date(a.created_at))
    .slice(0, 5)
    .map(form => ({
      id: form.id,
      title: form.title,
      trainer: form.created_by_name || 'Unknown',
      session: form.description || 'No description',
      status: form.is_published ? 'active' : 'draft',
      responses: form.response_count || 0,
      created: new Date(form.created_at).toLocaleDateString()
    }))

  // All data now comes from real backend endpoints

  const handleTabChange = (tab) => {
    setActiveTab(tab)

    // Load settings when settings tab is opened
    if (tab === 'settings') {
      fetchReportSettings()
    }
  }

  // Fetch report settings
  const fetchReportSettings = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/reports/settings`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('iaa_token')}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setReportSettings(data.settings)
      }
    } catch (error) {
      // Error fetching report settings
    }
  }

  // Save report settings
  const handleSaveSettings = async () => {
    setSettingsLoading(true)
    try {
      const response = await fetch(`${API_BASE_URL}/admin/reports/settings`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('iaa_token')}`
        },
        body: JSON.stringify({
          auto_generation_delay_days: reportSettings.auto_generation_delay_days?.value || '1'
        })
      })

      if (response.ok) {
        alert('Settings saved successfully!')
      } else {
        const errorData = await response.json()
        alert(`Error saving settings: ${errorData.detail}`)
      }
    } catch (error) {
      alert('Failed to save settings')
    } finally {
      setSettingsLoading(false)
    }
  }

  // Removed handleApproveRequest - admins no longer approve/edit forms

  // Removed handleReviewForm and handleApproveCompleteForm - admins can only publish or reject complete forms

  const handlePublishForm = async (requestId) => {
    // New workflow: Directly publish trainer-created form without editing
    try {
      const request = pendingRequests.find(req => req.id === requestId)
      if (!request || !request.form_data) {
        alert('Invalid request data - form must have complete form data')
        return
      }



      // Create form directly from request data
      const formData = {
        title: request.title,
        description: request.description,
        trainerId: request.trainer_id,  // ✅ Convert to camelCase for adminService
        departmentId: request.department_id,  // ✅ Convert to camelCase for adminService
        dueDate: request.due_date,  // ✅ Keep as string - adminService will handle ISO conversion
        questions: request.form_data?.questions || [],  // ✅ Extract questions directly
        settings: request.form_data?.settings || {},  // ✅ Extract settings directly
        session: request.form_data?.session || {},  // ✅ Extract session directly
        status: 'published',
        is_published: true
      }



      // Create and publish the form
      const createdForm = await adminService.createForm(formData)


      if (createdForm && createdForm.id) {
        // Mark the request as processed
        await adminService.markRequestAsProcessed(requestId)

        // Update local state
        setPendingRequests(prev => prev.filter(req => req.id !== requestId))

        alert('Form published successfully!')

        // Refresh data
        fetchDashboardData(false)
      }
    } catch (error) {
      alert(`Failed to publish form: ${error.message}`)
    }
  }

  const handleCreateFormFromRequest = async (requestId) => {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/form-requests/${requestId}/create-form`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        const result = await response.json()
        alert(`✅ Form created successfully and assigned to ${result.trainer.name}!`)

        // Refresh the data
        fetchDashboardData(false)

        // Navigate to the form editor
        navigate(`/admin/form-builder/${result.form.id}`)
      } else {
        const errorData = await response.json()
        alert(`❌ Error: ${errorData.detail}`)
      }
    } catch (error) {
      alert('Failed to create form from request')
    }
  }

  const handleRejectRequest = async (requestId) => {
    const reason = prompt('Please provide a reason for rejecting this request:')
    if (!reason) {
      return // User cancelled
    }

    try {
      const updatedRequest = await adminService.rejectFormRequest(requestId, reason)
      setPendingRequests(prev => prev.map(req =>
        req.id === requestId ? updatedRequest : req
      ))
      alert('Form request rejected successfully.')

      // Refresh the data to remove rejected requests from the list
      fetchDashboardData(false)
    } catch (error) {
      alert('Failed to reject request: ' + error.message)
    }
  }

  const handleCreateForm = () => {
    navigate('/admin/form-builder')
  }

  // Removed handleEditForm - admins can no longer edit forms

  const handlePreviewForm = (formId) => {
    // Open form preview in a new tab with preview mode
    window.open(`/form/${formId}?preview=true`, '_blank')
  }

  const handlePublishExistingForm = async (formId) => {
    try {
      await adminService.publishForm(formId)
      // Refresh forms data
      const updatedForms = await adminService.getForms()
      setForms(updatedForms)
    } catch (error) {
      setError('Failed to publish form')
    }
  }

  const handleDeleteForm = async (formId) => {
    if (window.confirm('Are you sure you want to delete this form? This action cannot be undone.')) {
      try {
        await adminService.deleteForm(formId)

        // Clear cache to ensure fresh data
        localStorage.removeItem('admin_dashboard_cache')
        sessionStorage.setItem('invalidate_admin_cache', 'true')

        // Refresh all dashboard data to ensure consistency
        await fetchDashboardData(false) // Force refresh without cache

      } catch (error) {
        setError('Failed to delete form')
      }
    }
  }

  const handleViewResponses = (formId) => {
    navigate(`/admin/form-responses/${formId}`)
  }

  const handleViewReports = () => {
    navigate('/admin/reports')
  }

  // Form deletion request handlers
  const handleApproveDeletionRequest = async (requestId, formTitle) => {
    if (window.confirm(`Are you sure you want to approve the deletion of "${formTitle}"? This action cannot be undone.`)) {
      try {
        await adminService.approveFormDeletionRequest(requestId)
        alert('Form deletion approved and form deleted successfully!')

        // Refresh data
        fetchDashboardData(false)
      } catch (error) {
        alert(`Failed to approve deletion request: ${error.message}`)
      }
    }
  }

  const handleRejectDeletionRequest = async (requestId, formTitle) => {
    const reason = prompt(`Please provide a reason for rejecting the deletion request for "${formTitle}":\n\nExamples:\n- Form is still needed for ongoing training\n- Form has valuable response data\n- Request lacks sufficient justification\n- Form can be modified instead of deleted`)
    if (reason === null) {
      return // User cancelled
    }

    // Make reason optional - if empty, provide a default
    const finalReason = reason.trim() || 'Request rejected by admin'

    try {
      await adminService.rejectFormDeletionRequest(requestId, finalReason)
      alert('Deletion request rejected successfully!')

      // Refresh data
      fetchDashboardData(false)
    } catch (error) {
      alert(`Failed to reject deletion request: ${error.message}`)
    }
  }

  const handleLogout = () => {
    if (confirm('Are you sure you want to logout?')) {
      localStorage.removeItem('iaa_token')
      localStorage.removeItem('iaa_user')
      navigate('/login')
    }
  }

  // Reset system function removed for production security



  return (
    <div className="admin-dashboard">
      <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
      
      <main className="admin-main" ref={ref}>
        {/* Loading State */}
        {loading && (
          <div className="loading-state">
            <div className="loading-spinner"></div>
            <p>Loading admin dashboard...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="error-state">
            <p>Error loading dashboard: {error}</p>
            <button onClick={() => window.location.reload()}>Retry</button>
          </div>
        )}

        {/* Main Content */}
        {!loading && !error && (
          <>
        {/* Header Animation Area */}
        <div className="dashboard-header">
          <div className="header-animation">
            <motion.div 
              className="dashboard-airplane"
              animate={planeControls}
              initial={{ 
                x: '-100%', 
                y: '0%', 
                rotate: 0, 
                scale: 0.8 
              }}
            >
              <Plane size={60} className="dashboard-plane-icon" />
            </motion.div>
            <div className="header-clouds">
              <div className="cloud cloud-1"></div>
              <div className="cloud cloud-2"></div>
            </div>
          </div>
          
          <div className="header-content">
            <motion.div
              className="welcome-section"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: inView ? 1 : 0, y: inView ? 0 : 30 }}
              transition={{ delay: 0.5, duration: 0.8 }}
            >
              <div className="welcome-content">
                <div className="welcome-text">
                  <h1>Admin Dashboard</h1>
                  <p>Manage feedback forms and generate comprehensive reports</p>
                </div>
                <div className="header-actions">
                  <button
                    className="refresh-btn"
                    onClick={() => fetchDashboardData(false)}
                    disabled={loading}
                    title="Refresh Dashboard Data"
                  >
                    <Settings size={20} className={loading ? 'spinning' : ''} />
                    {loading ? 'Refreshing...' : 'Refresh'}
                  </button>
                  <button
                    className="logout-btn"
                    onClick={handleLogout}
                    title="Logout"
                    style={{
                      background: 'linear-gradient(135deg, #6c757d 0%, #5a6268 100%)',
                      color: 'white',
                      border: 'none',
                      padding: '0.5rem 1rem',
                      borderRadius: '6px',
                      fontSize: '0.85rem',
                      fontWeight: '600',
                      cursor: 'pointer',
                      marginLeft: '0.5rem',
                      display: 'flex',
                      alignItems: 'center',
                      gap: '0.5rem'
                    }}
                  >
                    <LogOut size={16} />
                    Logout
                  </button>

                  <NotificationBell className="admin-notification-bell" />
                </div>
              </div>
            </motion.div>

            <motion.div
              className="stats-grid"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: inView ? 1 : 0, y: inView ? 0 : 30 }}
              transition={{ delay: 0.7, duration: 0.8 }}
            >
              <div className="stat-card">
                <div className="stat-info">
                  <h3>{finalDashboardStats.totalForms}</h3>
                  <p>Total Forms</p>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-info">
                  <h3>{finalDashboardStats.activeForms}</h3>
                  <p>Active Forms</p>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-info">
                  <h3>{finalDashboardStats.totalResponses}</h3>
                  <p>Total Responses</p>
                </div>
              </div>
              <div className="stat-card">
                <div className="stat-info">
                  <h3>{finalDashboardStats.pendingRequests}</h3>
                  <p>Pending Requests</p>
                </div>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <motion.div
          className="dashboard-nav"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.9, duration: 0.6 }}
        >
          <div className="nav-tabs">
            <button
              className={`nav-tab ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => handleTabChange('overview')}
            >
              <Settings size={20} />
              Overview
            </button>
            <button
              className={`nav-tab ${activeTab === 'forms' ? 'active' : ''}`}
              onClick={() => handleTabChange('forms')}
            >
              <FileText size={20} />
              Form Management
            </button>
            <button
              className={`nav-tab ${activeTab === 'reports' ? 'active' : ''}`}
              onClick={() => handleTabChange('reports')}
            >
              <BarChart3 size={20} />
              Reports & Analytics
            </button>
            <button
              className={`nav-tab ${activeTab === 'requests' ? 'active' : ''}`}
              onClick={() => handleTabChange('requests')}
            >
              <Clock size={20} />
              Form Requests
              {pendingRequests.filter(req => req.status === 'pending').length > 0 && (
                <span className="request-badge">
                  {pendingRequests.filter(req => req.status === 'pending').length}
                </span>
              )}
            </button>
            <button
              className={`nav-tab ${activeTab === 'deletion-requests' ? 'active' : ''}`}
              onClick={() => handleTabChange('deletion-requests')}
            >
              <Trash2 size={20} />
              Deletion Requests
              {deletionRequests.filter(req => req.status === 'pending').length > 0 && (
                <span className="request-badge">
                  {deletionRequests.filter(req => req.status === 'pending').length}
                </span>
              )}
            </button>
            <button
              className={`nav-tab ${activeTab === 'settings' ? 'active' : ''}`}
              onClick={() => handleTabChange('settings')}
            >
              <Settings size={20} />
              Settings
            </button>
          </div>
        </motion.div>

        {/* Content Area */}
        <motion.div
          className="dashboard-content"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1.1, duration: 0.6 }}
        >
          {activeTab === 'overview' && (
            <div className="overview-content">
              <div className="quick-actions">
                <h2>Quick Actions</h2>
                <div className="action-buttons">
                  <motion.button
                    className="action-btn primary"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleCreateForm}
                  >
                    <Plus size={20} />
                    Create New Form
                  </motion.button>
                  <motion.button
                    className="action-btn secondary"
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleViewReports}
                  >
                    <BarChart3 size={20} />
                    View Reports
                  </motion.button>

                </div>
              </div>

              <div className="recent-activity">
                <h2>Recent Forms</h2>
                <div className="forms-grid">
                  {recentForms.map((form) => {
                    // Find the original form data to get all properties
                    const originalForm = forms.find(f => f.id === form.id)
                    if (!originalForm) return null

                    return (
                      <div key={form.id} className="form-card">
                        <div className="form-card-header">
                          <h3>{originalForm.title}</h3>
                          <div className="form-status">
                            <span className={`status-badge ${originalForm.is_published ? 'active' : 'draft'}`}>
                              {originalForm.is_published ? 'Published' : 'Draft'}
                            </span>
                          </div>
                        </div>

                        <div className="form-card-body">
                          <p className="form-description">
                            {originalForm.description || 'No description provided'}
                          </p>

                          <div className="form-meta">
                            <div className="meta-item">
                              <Calendar size={16} />
                              <span>Created: {new Date(originalForm.created_at).toLocaleDateString()}</span>
                            </div>
                            {originalForm.published_at && (
                              <div className="meta-item">
                                <Clock size={16} />
                                <span>Published: {new Date(originalForm.published_at).toLocaleDateString()}</span>
                              </div>
                            )}
                            <div className="meta-item">
                              <Users size={16} />
                              <span>{originalForm.response_count || 0} responses</span>
                            </div>
                            {originalForm.due_date && (
                              <div className={`meta-item due-date-item ${getDueDateStatus(originalForm.due_date).className}`}>
                                <Calendar size={16} />
                                <span>{getDueDateStatus(originalForm.due_date).text}</span>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="form-card-actions">
                          <button
                            className="btn-outline"
                            onClick={() => handlePreviewForm(originalForm.id)}
                            title="Preview Form"
                          >
                            <Eye size={16} />
                            Preview
                          </button>

                          {/* Removed edit button - admins can no longer edit forms */}

                          {!originalForm.is_published ? (
                            <button
                              className="btn-primary"
                              onClick={() => handlePublishExistingForm(originalForm.id)}
                              title="Publish Form"
                            >
                              <Send size={16} />
                              Publish
                            </button>
                          ) : (
                            <button
                              className="btn-outline"
                              onClick={() => handleViewResponses(originalForm.id)}
                              title="View Responses"
                            >
                              <BarChart size={16} />
                              Responses
                            </button>
                          )}

                          <button
                            className="btn-danger"
                            onClick={() => handleDeleteForm(originalForm.id)}
                            title="Delete Form"
                          >
                            <Trash2 size={16} />
                            Delete
                          </button>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'forms' && (
            <div className="forms-content">
              <div className="forms-header">
                <h2>Form Management</h2>
                <button className="create-form-btn" onClick={handleCreateForm}>
                  <Plus size={20} />
                  Create New Form
                </button>
              </div>
              <div className="forms-filters">
                <div className="search-bar">
                  <Search size={20} />
                  <input
                    type="text"
                    placeholder="Search forms..."
                    value={searchTerm}
                    onChange={(e) => setSearchTerm(e.target.value)}
                  />
                </div>
                <select
                  value={statusFilter}
                  onChange={(e) => setStatusFilter(e.target.value)}
                  className="filter-select"
                >
                  <option value="all">All Forms</option>
                  <option value="active">Active</option>
                  <option value="draft">Draft</option>
                  <option value="archived">Archived</option>
                </select>
              </div>

              {/* Real Forms List from Cloud Database */}
              <div className="forms-grid">
                {loading ? (
                  <div className="loading-state">
                    <div className="loading-spinner"></div>
                    <p>Loading forms from cloud database...</p>
                  </div>
                ) : forms.length === 0 ? (
                  <div className="empty-state">
                    <FileText size={48} />
                    <h3>No Forms Found</h3>
                    <p>Create your first form to get started</p>
                    <button className="create-form-btn" onClick={handleCreateForm}>
                      <Plus size={20} />
                      Create New Form
                    </button>
                  </div>
                ) : (
                  forms
                    .filter(form => {
                      const matchesSearch = form.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                                          form.description?.toLowerCase().includes(searchTerm.toLowerCase())
                      const matchesStatus = statusFilter === 'all' ||
                                          (statusFilter === 'active' && form.is_published) ||
                                          (statusFilter === 'draft' && !form.is_published)
                      return matchesSearch && matchesStatus
                    })
                    .map((form) => (
                      <div key={form.id} className="form-card">
                        <div className="form-card-header">
                          <h3>{form.title}</h3>
                          <div className="form-status">
                            <span className={`status-badge ${form.is_published ? 'active' : 'draft'}`}>
                              {form.is_published ? 'Published' : 'Draft'}
                            </span>
                          </div>
                        </div>

                        <div className="form-card-body">
                          <p className="form-description">
                            {form.description || 'No description provided'}
                          </p>

                          <div className="form-meta">
                            <div className="meta-item">
                              <Calendar size={16} />
                              <span>Created: {new Date(form.created_at).toLocaleDateString()}</span>
                            </div>
                            {form.published_at && (
                              <div className="meta-item">
                                <Clock size={16} />
                                <span>Published: {new Date(form.published_at).toLocaleDateString()}</span>
                              </div>
                            )}
                            <div className="meta-item">
                              <Users size={16} />
                              <span>{form.response_count || 0} responses</span>
                            </div>
                            {form.due_date && (
                              <div className={`meta-item due-date-item ${getDueDateStatus(form.due_date).className}`}>
                                <Calendar size={16} />
                                <span>{getDueDateStatus(form.due_date).text}</span>
                              </div>
                            )}
                          </div>
                        </div>

                        <div className="form-card-actions">
                          <button
                            className="btn-outline"
                            onClick={() => handlePreviewForm(form.id)}
                            title="Preview Form"
                          >
                            <Eye size={16} />
                            Preview
                          </button>

                          {/* Removed edit button - admins can no longer edit forms */}

                          {!form.is_published ? (
                            <button
                              className="btn-primary"
                              onClick={() => handlePublishExistingForm(form.id)}
                              title="Publish Form"
                            >
                              <Send size={16} />
                              Publish
                            </button>
                          ) : (
                            <button
                              className="btn-outline"
                              onClick={() => handleViewResponses(form.id)}
                              title="View Responses"
                            >
                              <BarChart size={16} />
                              Responses
                            </button>
                          )}

                          <button
                            className="btn-danger"
                            onClick={() => handleDeleteForm(form.id)}
                            title="Delete Form"
                          >
                            <Trash2 size={16} />
                            Delete
                          </button>
                        </div>
                      </div>
                    ))
                )}</div>
            </div>
          )}

          {activeTab === 'reports' && (
            <div className="reports-content">
              <div className="reports-header">
                <h2>Reports & Analytics</h2>
                <p>Generate comprehensive analytics and insights for individual forms</p>
              </div>

              <div className="reports-interface">
                <div className="reports-grid">
                  {/* Form-Specific Reports Section */}
                  <motion.div
                    className="report-section form-reports"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.5 }}
                  >
                    <div className="report-icon">
                      <FileText size={32} />
                    </div>
                    <h3>Form-Specific Reports</h3>
                    <p>Generate detailed analytics and insights for individual forms with enhanced visualizations.</p>
                    <div className="report-features">
                      <span>• Response analytics</span>
                      <span>• Sentiment analysis</span>
                      <span>• Performance metrics</span>
                      <span>• Visual charts</span>
                    </div>
                    <button
                      className="btn btn-primary report-btn"
                      onClick={() => navigate('/admin/reports?type=form')}
                    >
                      <BarChart3 size={16} />
                      Generate Form Reports
                    </button>
                  </motion.div>
                </div>

                {/* Quick Stats Overview */}
                <motion.div
                  className="reports-overview"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <h3>Quick Overview</h3>
                  <div className="overview-stats">
                    <div className="stat-item">
                      <FileText size={20} />
                      <span className="stat-number">{dashboardStats?.totalForms || 0}</span>
                      <span className="stat-label">Total Forms</span>
                    </div>
                    <div className="stat-item">
                      <Users size={20} />
                      <span className="stat-number">{departments?.length || 0}</span>
                      <span className="stat-label">Departments</span>
                    </div>
                    <div className="stat-item">
                      <BarChart3 size={20} />
                      <span className="stat-number">{dashboardStats?.totalResponses || 0}</span>
                      <span className="stat-label">Total Responses</span>
                    </div>
                    <div className="stat-item">
                      <TrendingUp size={20} />
                      <span className="stat-number">{dashboardStats?.averageRating || 0}</span>
                      <span className="stat-label">Avg Rating</span>
                    </div>
                  </div>
                </motion.div>
              </div>
            </div>
          )}

          {activeTab === 'requests' && (
            <div className="requests-content">
              <div className="requests-header">
                <h2>Form Requests</h2>
                <p>Review trainer-submitted forms and publish or reject them</p>
              </div>

              {pendingRequests.filter(req => req.status === 'pending').length === 0 ? (
                <div className="empty-state">
                  <Clock size={48} />
                  <h3>No pending requests</h3>
                  <p>All form requests have been processed.</p>
                </div>
              ) : (
                <div className="requests-list">
                  {pendingRequests.filter(req => req.status === 'pending').map((request) => (
                    <motion.div
                      key={request.id}
                      className="request-card"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="request-header">
                        <div className="request-info">
                          <h3>{request.title}</h3>
                          <p className="trainer-info">
                            Requested by: <strong>{request.trainer_name || 'Unknown Trainer'}</strong>
                          </p>
                          <p className="form-type-info">
                            Form Type: <strong>{request.form_type || 'feedback'}</strong>
                          </p>
                          <p className="request-date">
                            Submitted: {formatDate(request.created_at)}
                          </p>
                        </div>
                        <div className="request-status">
                          <span className={`status-badge ${request.status}`}>
                            {request.status}
                          </span>
                        </div>
                      </div>

                      <div className="request-details">
                        <div className="detail-row">
                          <strong>Department:</strong>
                          <span>{request.department_name || 'Unknown Department'}</span>
                        </div>
                        {request.session_name && (
                          <div className="detail-row">
                            <strong>Session:</strong>
                            <span>{request.session_name}</span>
                          </div>
                        )}
                        {request.session_date && (
                          <div className="detail-row">
                            <strong>Session Date:</strong>
                            <span>{formatDate(request.session_date)}</span>
                          </div>
                        )}
                        <div className="detail-row">
                          <strong>Session Duration:</strong>
                          <span>{request.session_duration || 60} minutes</span>
                        </div>
                        <div className="detail-row">
                          <strong>Form Validity:</strong>
                          <span>{request.form_validity_duration || 7} days</span>
                        </div>
                        <div className="detail-row">
                          <strong>Priority:</strong>
                          <span className={`priority-badge ${request.priority || 'normal'}`}>
                            {(request.priority || 'normal').toUpperCase()}
                          </span>
                        </div>
                        {request.additional_notes && (
                          <div className="detail-row">
                            <strong>Additional Notes:</strong>
                            <span>{request.additional_notes}</span>
                          </div>
                        )}

                        {/* Display form questions if available */}
                        {request.form_data && (
                          <div className="form-questions-preview">
                            <div className="detail-row">
                              <strong>Form Questions:</strong>
                            </div>
                            <div className="questions-list">
                              {(() => {
                                try {
                                  const formData = typeof request.form_data === 'string'
                                    ? JSON.parse(request.form_data)
                                    : request.form_data;
                                  const questions = formData?.questions || [];



                                  if (questions.length === 0) {
                                    return <span className="no-questions">No questions found in form data</span>;
                                  }

                                  return questions.map((question, index) => (
                                    <div key={index} className="question-preview">
                                      <span className="question-number">{index + 1}.</span>
                                      <span className="question-title">{question.title || 'Untitled Question'}</span>
                                      <span className="question-type">({question.type || 'unknown'})</span>
                                    </div>
                                  ));
                                } catch (error) {
                                  return <span className="error-message">Error parsing form questions</span>;
                                }
                              })()}
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="request-actions">
                        <button
                          className="reject-btn"
                          onClick={() => handleRejectRequest(request.id)}
                        >
                          <XCircle size={16} />
                          Reject
                        </button>
                        {request.form_data ? (
                          // New workflow: Trainer created complete form - admin can only publish or reject
                          <button
                            className="publish-btn"
                            onClick={() => handlePublishForm(request.id)}
                          >
                            <Send size={16} />
                            Publish Form
                          </button>
                        ) : (
                          // Incomplete request - show message
                          <span className="incomplete-request-message">
                            Incomplete form request - trainer needs to submit complete form
                          </span>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}

              {/* Removed approved requests section - new workflow doesn't use approvals */}
            </div>
          )}

          {activeTab === 'deletion-requests' && (
            <div className="deletion-requests-content">
              <div className="deletion-requests-header">
                <h2>Form Deletion Requests</h2>
                <p>Review trainer requests to delete forms</p>
              </div>

              {deletionRequests.length === 0 ? (
                <div className="empty-state">
                  <Trash2 size={48} />
                  <h3>No deletion requests</h3>
                  <p>No form deletion requests have been submitted.</p>
                </div>
              ) : (
                <div className="deletion-requests-grid">
                  {deletionRequests.map((request) => (
                    <motion.div
                      key={request.id}
                      className={`deletion-request-card ${request.status}`}
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="deletion-request-header">
                        <h3>{request.form_title}</h3>
                        <div className={`status-badge ${request.status}`}>
                          {request.status.toUpperCase()}
                        </div>
                      </div>

                      <div className="deletion-request-details">
                        <p><strong>Trainer:</strong> {request.trainer_name}</p>
                        <p><strong>Email:</strong> {request.trainer_email}</p>
                        <p><strong>Requested:</strong> {formatDateTime(request.created_at)}</p>
                        {request.reason && (
                          <div className="deletion-reason">
                            <strong>Reason:</strong>
                            <p>{request.reason}</p>
                          </div>
                        )}
                        {request.admin_response && (
                          <div className="admin-response">
                            <strong>Admin Response:</strong>
                            <p>{request.admin_response}</p>
                            <small>Reviewed: {formatDateTime(request.reviewed_at)}</small>
                          </div>
                        )}
                      </div>

                      {request.status === 'pending' && (
                        <div className="deletion-request-actions">
                          <button
                            className="approve-btn"
                            onClick={() => handleApproveDeletionRequest(request.id, request.form_title)}
                          >
                            <CheckCircle size={16} />
                            Approve Deletion
                          </button>
                          <button
                            className="reject-btn"
                            onClick={() => handleRejectDeletionRequest(request.id, request.form_title)}
                          >
                            <XCircle size={16} />
                            Reject Request
                          </button>
                        </div>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'settings' && (
            <div className="settings-content">
              <div className="settings-header">
                <h2>System Settings</h2>
                <p>Configure system-wide settings and preferences</p>
              </div>

              <div className="settings-section">
                <h3>Report Generation Settings</h3>
                <div className="setting-item">
                  <label htmlFor="delay-days">
                    Auto-generation delay (days after due date):
                  </label>
                  <input
                    id="delay-days"
                    type="number"
                    min="0"
                    max="30"
                    value={reportSettings.auto_generation_delay_days?.value || '1'}
                    onChange={(e) => setReportSettings(prev => ({
                      ...prev,
                      auto_generation_delay_days: {
                        ...prev.auto_generation_delay_days,
                        value: e.target.value
                      }
                    }))}
                    className="setting-input"
                  />
                  <small className="setting-description">
                    Number of days after form due date to automatically generate reports
                  </small>
                </div>

                <button
                  className="save-settings-btn"
                  onClick={handleSaveSettings}
                  disabled={settingsLoading}
                >
                  {settingsLoading ? 'Saving...' : 'Save Settings'}
                </button>
              </div>
            </div>
          )}
        </motion.div>
        </>
        )}
      </main>

      <Footer />
    </div>
  )
}

export default AdminDashboard
