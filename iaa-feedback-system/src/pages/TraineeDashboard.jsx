import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  FileText,
  Clock,
  CheckCircle,
  AlertCircle,
  Calendar,
  User,
  Building,
  Star,
  Play,
  RotateCcw,
  LogOut
} from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate } from 'react-router-dom'
import adminService from '../api/adminService'
import { formatDate, formatDateTime, getRelativeTime } from '../utils/dateUtils'
import Header from '../components/Header'
import Footer from '../components/Footer'
import './TraineeDashboard.css'

const TraineeDashboard = ({ theme, toggleTheme }) => {
  const { user, logout, markFormCompleted, hasCompletedForm } = useAuth()
  const navigate = useNavigate()
  const [availableForms, setAvailableForms] = useState([])
  const [completedForms, setCompletedForms] = useState([])
  const [isLoading, setIsLoading] = useState(true)
  const [departmentName, setDepartmentName] = useState('')
  const [loadingDepartment, setLoadingDepartment] = useState(true)

  // Real forms data will be fetched from backend

  // Fetch department name
  const fetchDepartmentName = async () => {
    if (!user?.department_id) {
      setLoadingDepartment(false)
      setDepartmentName('Unknown Department')
      return
    }

    try {
      setLoadingDepartment(true)
      const departments = await adminService.getDepartments()
      const department = departments.find(dept => dept.id === parseInt(user.department_id))
      setDepartmentName(department ? department.name : `Department ${user.department_id}`)
    } catch (error) {
      console.error('Error fetching department:', error)
      setDepartmentName(`Department ${user.department_id}`)
    } finally {
      setLoadingDepartment(false)
    }
  }

  useEffect(() => {
    console.log('🔄 TraineeDashboard useEffect triggered with user:', user)

    if (!user || user.role !== 'trainee') {
      console.warn('⚠️ User not found or not a trainee, redirecting to login')
      navigate('/login')
      return
    }

    console.log('✅ Valid trainee user found, fetching data')
    // Fetch department name and forms
    fetchDepartmentName()
    fetchTraineeForms()
  }, [user, navigate, hasCompletedForm])

  // Fetch real forms from backend
  const fetchTraineeForms = async () => {
    try {
      setIsLoading(true)
      console.log('🔄 Fetching forms for trainee:', user)

      // Check if user has department_id
      if (!user?.department_id) {
        console.warn('⚠️ User has no department_id, cannot fetch forms')
        setAvailableForms([])
        setCompletedForms([])
        return
      }

      // Fetch all published forms
      const formsData = await adminService.getForms()
      console.log('📋 All forms data:', formsData)

      // Backend already filters for trainee's department and published forms
      // So we can use all returned forms directly
      const departmentForms = formsData || []

      console.log('🏢 Forms for trainee dashboard:', departmentForms)

      const available = []
      const completed = []

      departmentForms.forEach(form => {
        const isCompleted = form.is_completed || false

        // Transform backend form data to match frontend expectations
        const formQuestions = form.form_data?.questions || []
        const transformedForm = {
          id: form.id,
          title: form.title,
          description: form.description,
          trainerId: form.trainer_id,
          departmentId: form.department_id,
          type: form.type || 'single-use',
          status: form.is_published ? 'active' : 'draft',
          dueDate: form.due_date || 'No due date',
          estimatedTime: `${Math.max(1, Math.ceil(formQuestions.length * 0.5))} minutes`, // Estimate based on questions
          questions: formQuestions.length,
          questionsData: formQuestions, // Store actual questions data
          priority: 'medium', // Default - could be stored in form data
          category: form.form_data?.session?.course || 'General',
          isCompleted: isCompleted,
          // Store complete form data for form filling
          formData: form.form_data,
          session: form.form_data?.session || {}
        }

        if (isCompleted) {
          completed.push(transformedForm)
          // For multi-use forms, also keep them in available
          if (transformedForm.type === 'multi-use') {
            available.push(transformedForm)
          }
        } else {
          available.push(transformedForm)
        }
      })

      setAvailableForms(available)
      setCompletedForms(completed)

    } catch (error) {
      console.error('Error fetching trainee forms:', error)
      // Fallback to empty arrays
      setAvailableForms([])
      setCompletedForms([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFormSubmit = (formId, formType) => {
    // Mark form as completed
    markFormCompleted(formId)
    
    // Update local state
    const form = availableForms.find(f => f.id === formId)
    if (form) {
      setCompletedForms(prev => [...prev, form])
      
      // Remove from available if single-use
      if (formType === 'single-use') {
        setAvailableForms(prev => prev.filter(f => f.id !== formId))
      }
    }
  }

  const getPriorityColor = (priority, isDark = false) => {
    if (isDark) {
      // Enhanced colors for dark theme with better contrast
      switch (priority) {
        case 'high': return '#ff5252'     // Brighter red
        case 'medium': return '#ffb74d'   // Brighter orange
        case 'low': return '#66bb6a'      // Brighter green
        default: return '#42a5f5'         // Brighter blue
      }
    } else {
      // Original colors for light theme
      switch (priority) {
        case 'high': return '#f44336'
        case 'medium': return '#ff9800'
        case 'low': return '#4caf50'
        default: return '#2196f3'
      }
    }
  }

  const getDaysUntilDue = (dueDate) => {
    if (!dueDate || dueDate === 'No due date') return Infinity
    const today = new Date()
    const due = new Date(dueDate)
    if (isNaN(due.getTime())) return Infinity
    const diffTime = due - today
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24))
    return diffDays
  }

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  if (isLoading) {
    return (
      <div className="trainee-dashboard">
        <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading your dashboard...</p>
        </div>
        <Footer />
      </div>
    )
  }

  // Show loading state if user is not loaded yet
  if (!user) {
    return (
      <div className="trainee-dashboard">
        <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
        <main className="dashboard-main">
          <div className="loading-state">
            <div className="loading-spinner"></div>
            <p>Loading dashboard...</p>
          </div>
        </main>
        <Footer />
      </div>
    )
  }

  return (
    <div className="trainee-dashboard">
      <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />

      <main className="dashboard-main">
        {/* Header Section */}
        <div className="dashboard-header">
          <div className="header-content">
            <div className="user-info">
              <img
                src={user.avatar || '/default-avatar.svg'}
                alt="Avatar"
                className="user-avatar"
                onError={(e) => {
                  e.target.src = '/default-avatar.svg'
                }}
              />
              <div className="user-details">
                <h1>Welcome, {user.firstName || user.first_name || user.name || 'User'}</h1>
                <p className="user-role">
                  <Building size={16} />
                  {loadingDepartment ? (
                    'Loading department...'
                  ) : (
                    `Trainee - ${departmentName}`
                  )}
                </p>
              </div>
            </div>
            <button className="logout-btn" onClick={handleLogout}>
              <LogOut size={16} />
              Logout
            </button>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="stats-section">
          <div className="stats-grid">
            <motion.div 
              className="stat-card"
              whileHover={{ scale: 1.02 }}
            >
              <div className="stat-icon available">
                <FileText size={24} />
              </div>
              <div className="stat-content">
                <h3>{availableForms.length}</h3>
                <p>Available Forms</p>
              </div>
            </motion.div>

            <motion.div 
              className="stat-card"
              whileHover={{ scale: 1.02 }}
            >
              <div className="stat-icon completed">
                <CheckCircle size={24} />
              </div>
              <div className="stat-content">
                <h3>{completedForms.length}</h3>
                <p>Completed Forms</p>
              </div>
            </motion.div>

            <motion.div 
              className="stat-card"
              whileHover={{ scale: 1.02 }}
            >
              <div className="stat-icon pending">
                <Clock size={24} />
              </div>
              <div className="stat-content">
                <h3>{availableForms.filter(f => getDaysUntilDue(f.dueDate) <= 3).length}</h3>
                <p>Due Soon</p>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Available Forms Section */}
        <div className="forms-section">
          <div className="section-header">
            <h2>Available Forms</h2>
            <p>Complete feedback forms assigned to your department</p>
          </div>

          {isLoading ? (
            <div className="loading-state">
              <div className="loading-spinner"></div>
              <p>Loading forms...</p>
            </div>
          ) : availableForms.length === 0 ? (
            <div className="empty-state">
              <FileText size={48} />
              <h3>No forms available</h3>
              <p>All assigned forms have been completed or no new forms are available for your department.</p>
              <button
                className="refresh-btn"
                onClick={fetchTraineeForms}
              >
                <RotateCcw size={16} />
                Refresh
              </button>
            </div>
          ) : (
            <div className="forms-grid">
              {availableForms.map((form) => {
                const daysUntilDue = getDaysUntilDue(form.dueDate)
                const isUrgent = daysUntilDue <= 3
                const isCompleted = form.isCompleted || false
                
                return (
                  <motion.div
                    key={form.id}
                    className={`form-card ${isUrgent ? 'urgent' : ''} ${isCompleted && form.type === 'multi-use' ? 'completed-multi' : ''}`}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div className="form-header">
                      <div className="form-priority" style={{ backgroundColor: getPriorityColor(form.priority, theme === 'dark') }}>
                        {form.priority}
                      </div>
                      <div className="form-type">
                        {form.type === 'single-use' ? (
                          <span className="single-use">Single Use</span>
                        ) : (
                          <span className="multi-use">
                            <RotateCcw size={12} />
                            Repeatable
                          </span>
                        )}
                      </div>
                    </div>

                    <div className="form-content">
                      <h3>{form.title}</h3>
                      <p className="form-description">{form.description}</p>
                      
                      <div className="form-meta">
                        <div className="meta-item">
                          <Clock size={14} />
                          <span>{form.estimatedTime}</span>
                        </div>
                        <div className="meta-item">
                          <FileText size={14} />
                          <span>{form.questions} questions</span>
                        </div>
                        <div className="meta-item">
                          <Calendar size={14} />
                          <span>Due: {form.dueDate && form.dueDate !== 'No due date' ? new Date(form.dueDate).toLocaleDateString() : 'No due date'}</span>
                        </div>
                      </div>

                      {isUrgent && (
                        <div className="urgency-warning">
                          <AlertCircle size={16} />
                          <span>Due in {daysUntilDue} day{daysUntilDue !== 1 ? 's' : ''}</span>
                        </div>
                      )}
                    </div>

                    <div className="form-actions">
                      <button
                        className="start-form-btn"
                        onClick={() => navigate(`/form/${form.id}`)}
                      >
                        <Play size={16} />
                        {isCompleted && form.type === 'multi-use' ? 'Complete Again' : 'Start Form'}
                      </button>
                    </div>
                  </motion.div>
                )
              })}
            </div>
          )}
        </div>

        {/* Completed Forms Section */}
        {completedForms.length > 0 && (
          <div className="completed-section">
            <div className="section-header">
              <h2>Completed Forms</h2>
              <p>Forms you have successfully submitted</p>
            </div>
            
            <div className="completed-list">
              {completedForms.map((form) => (
                <div key={`completed-${form.id}`} className="completed-item">
                  <div className="completed-icon">
                    <CheckCircle size={20} />
                  </div>
                  <div className="completed-content">
                    <h4>{form.title}</h4>
                    <p>{form.category}</p>
                  </div>
                  <div className="completed-meta">
                    <Star size={16} />
                    <span>Completed</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}

export default TraineeDashboard
