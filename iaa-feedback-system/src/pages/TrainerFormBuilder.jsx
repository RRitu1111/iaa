import { useState, useEffect } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Send, Eye, Edit, Bell } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import { useAuth } from '../contexts/AuthContext'
import { formatDate, formatDateTime } from '../utils/dateUtils'
import Header from '../components/Header'
import Footer from '../components/Footer'
import FormBuilder from '../components/FormBuilder'
import adminService from '../api/adminService'
import './FormBuilderPage.css'

const TrainerFormBuilder = ({ theme, toggleTheme }) => {
  const { t } = useLanguage()
  const { user } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    trainerId: user?.id || '',
    departmentId: user?.department_id || '',
    dueDate: null,
    session: {
      name: '',
      date: '',
      course: '',
      duration: 60
    },
    questions: [],
    settings: {
      allowAnonymous: true,
      requireAll: false,
      showProgress: true,
      randomizeQuestions: false
    },
    status: 'draft'
  })
  
  const [saveStatus, setSaveStatus] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [currentView, setCurrentView] = useState('builder')
  const [submitting, setSubmitting] = useState(false)
  const [departments, setDepartments] = useState([])
  const [loading, setLoading] = useState(true)
  const [isReEdit, setIsReEdit] = useState(false)
  const [rejectedRequest, setRejectedRequest] = useState(null)

  // Load default template and departments on component mount
  useEffect(() => {
    // Check if we're re-editing a rejected request
    if (location.state?.rejectedRequest && location.state?.isReEdit) {
      setIsReEdit(true)
      setRejectedRequest(location.state.rejectedRequest)
      loadRejectedRequestData(location.state.rejectedRequest)
    } else {
      // Initialize with empty form instead of loading default template
      setFormData({
        id: null,
        title: '',
        description: '',
        trainerId: user?.id || '',
        departmentId: user?.department_id || '',
        dueDate: null,
        session: {
          name: '',
          date: '',
          course: '',
          duration: ''
        },
        questions: [],
        settings: {
          allowAnonymous: true,
          requireAll: false,
          showProgress: true,
          randomizeQuestions: false
        },
        status: 'draft'
      })
    }
    fetchDepartments()
  }, [])

  const loadRejectedRequestData = (request) => {
    try {
      // Parse form data if it's a string
      const formData = typeof request.form_data === 'string'
        ? JSON.parse(request.form_data)
        : request.form_data

      setFormData({
        title: request.title || '',
        description: request.description || '',
        trainerId: user?.id || '',
        departmentId: request.department_id || user?.department_id || '',
        dueDate: request.due_date || null,
        session: formData?.session || {
          name: request.session_name || '',
          date: request.session_date || '',
          course: '',
          duration: request.session_duration || 60
        },
        questions: formData?.questions || [],
        settings: formData?.settings || {
          allowAnonymous: true,
          requireAll: false,
          showProgress: true,
          randomizeQuestions: false
        },
        status: 'draft'
      })

      console.log('Loaded rejected request data:', formData)
    } catch (error) {
      console.error('Error loading rejected request data:', error)
      // Fallback to default template
      loadDefaultTemplate()
    }
  }

  const fetchDepartments = async () => {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch('http://localhost:8001/departments', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setDepartments(data.departments || [])
      } else {
        console.error('Failed to fetch departments')
      }
    } catch (error) {
      console.error('Error fetching departments:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadDefaultTemplate = async () => {
    setIsLoading(true)
    try {
      const template = await adminService.getTrainerFormTemplate()

      setFormData(prev => ({
        ...prev,
        ...template,
        trainerId: user?.id || '',
        departmentId: user?.department_id || '',
        questions: template.form_data?.questions || [],
        settings: template.form_data?.settings || {
          allowAnonymous: true,
          requireAll: false,
          showProgress: true,
          randomizeQuestions: false
        },
        status: 'draft'
      }))
    } catch (error) {
      console.error('Error loading template:', error)
      alert('Failed to load form template. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  const handleSendRequest = async (form) => {
    // Validate form before sending
    if (!form.title.trim()) {
      alert('Please enter a form title')
      return
    }

    if (form.questions.length === 0) {
      alert('Please add at least one question to your form')
      return
    }

    setSubmitting(true)
    setSaveStatus('saving')

    try {
      // Prepare request data with complete form structure
      const requestData = {
        title: form.title,
        description: form.description || `Feedback form for ${form.session.name || 'training session'}`,
        trainer_id: user.id,
        department_id: user.department_id,
        session_name: form.session.name,
        session_date: form.session.date,
        session_duration: form.session.duration || 60,
        form_validity_duration: 7, // Default 7 days
        form_type: 'feedback',
        priority: 'normal',
        additional_notes: `Complete form created by trainer with ${form.questions.length} questions`,
        // Include the complete form structure
        form_data: {
          questions: form.questions,
          settings: form.settings,
          session: form.session
        },
        due_date: form.dueDate ? form.dueDate.toISOString() : null
      }

      console.log('🚀 Sending form request:', requestData)

      const response = await adminService.createFormRequest(requestData)
      
      setSaveStatus('saved')
      
      // Show success message
      alert('Form request sent successfully! The admin will review your form and approve it for publication.')

      // Update pending requests count in localStorage for real-time update
      const currentCount = parseInt(localStorage.getItem('pending_requests_count') || '0')
      localStorage.setItem('pending_requests_count', (currentCount + 1).toString())

      // Navigate back to trainer dashboard
      setTimeout(() => {
        navigate('/trainer-dashboard')
      }, 1500)

    } catch (error) {
      console.error('Error sending form request:', error)
      setSaveStatus('error')
      alert(`Failed to send form request: ${error.message}`)
    } finally {
      setSubmitting(false)
      setTimeout(() => setSaveStatus(null), 3000)
    }
  }

  const getSaveStatusMessage = () => {
    switch (saveStatus) {
      case 'saving':
        return 'Sending request...'
      case 'saved':
        return 'Request sent successfully!'
      case 'error':
        return 'Error sending request'
      default:
        return null
    }
  }

  const getSaveStatusClass = () => {
    switch (saveStatus) {
      case 'saving':
        return 'status-saving'
      case 'saved':
        return 'status-saved'
      case 'error':
        return 'status-error'
      default:
        return ''
    }
  }

  if (isLoading || loading) {
    return (
      <div className="app">
        <Header theme={theme} toggleTheme={toggleTheme} />
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading form builder...</p>
        </div>
        <Footer />
      </div>
    )
  }

  return (
    <div className={`app ${theme}`}>
      <Header theme={theme} toggleTheme={toggleTheme} />

      {/* Status Message */}
      {saveStatus && (
        <motion.div
          className={`save-status ${getSaveStatusClass()}`}
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -50 }}
        >
          {getSaveStatusMessage()}
        </motion.div>
      )}

      <div className="form-builder-nav">
        <div className="nav-content">
          <button 
            className="back-button"
            onClick={() => navigate('/trainer-dashboard')}
          >
            <ArrowLeft size={20} />
            Back to Dashboard
          </button>

          <div className="trainer-indicator">
            <Bell size={16} />
            <span>{isReEdit ? 'Re-editing rejected form' : 'Creating form request'}</span>
          </div>

          <div className="nav-center">
            <div className="view-toggle">
              <button
                className={`toggle-btn ${currentView === 'builder' ? 'active' : ''}`}
                onClick={() => setCurrentView('builder')}
              >
                <Edit size={16} />
                Builder
              </button>
              <button
                className={`toggle-btn ${currentView === 'preview' ? 'active' : ''}`}
                onClick={() => setCurrentView('preview')}
              >
                <Eye size={16} />
                Preview
              </button>
            </div>
          </div>

          <div className="nav-actions">
            <button
              className="load-template-btn"
              onClick={loadDefaultTemplate}
              disabled={isLoading}
              style={{
                marginRight: '1rem',
                padding: '0.5rem 1rem',
                backgroundColor: '#6c757d',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '0.5rem'
              }}
            >
              <Edit size={16} />
              {isLoading ? 'Loading...' : 'Load Template'}
            </button>
            <button
              className="send-request-btn"
              onClick={() => handleSendRequest(formData)}
              disabled={submitting}
            >
              <Send size={16} />
              {submitting
                ? 'Sending...'
                : isReEdit
                  ? 'Resubmit Form'
                  : 'Send Request'
              }
            </button>
          </div>
        </div>
      </div>

      <main className="form-builder-main">
        <FormBuilder
          initialForm={formData}
          onFormChange={setFormData}
          currentView={currentView}
          isTrainerMode={true}
          currentUser={user}
          departments={departments}
        />
      </main>

      <Footer />
    </div>
  )
}

export default TrainerFormBuilder
