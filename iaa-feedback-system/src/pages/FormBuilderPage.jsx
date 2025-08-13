import { useState, useEffect } from 'react'
import { useNavigate, useParams, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Save, Eye, Send, Edit, Bell, Calendar } from 'lucide-react'
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'
import { useLanguage } from '../contexts/LanguageContext'
import { formatDate, formatDateTime } from '../utils/dateUtils'
import Header from '../components/Header'
import Footer from '../components/Footer'
import FormBuilder from '../components/FormBuilder'
// import FormPreview from '../components/FormPreview'
import adminService from '../api/adminService'
import './FormBuilderPage.css'

const FormBuilderPage = ({ theme, toggleTheme }) => {
  const [currentView, setCurrentView] = useState('builder') // 'builder' or 'preview'
  const [formData, setFormData] = useState(null)
  const [isLoading, setIsLoading] = useState(true) // Start with loading true
  const [saveStatus, setSaveStatus] = useState(null) // 'saving', 'saved', 'error'
  const [departments, setDepartments] = useState([])
  const [trainers, setTrainers] = useState([])
  const [fromRequest, setFromRequest] = useState(false)
  const [reviewMode, setReviewMode] = useState(false)
  const [dueDateError, setDueDateError] = useState('')

  const { t } = useLanguage()
  const navigate = useNavigate()
  const { formId } = useParams()
  const location = useLocation()

  // Due date validation function
  const validateDueDate = (date) => {
    if (!date) {
      setDueDateError('')
      return true
    }

    const now = new Date()
    const twentyFourHoursFromNow = new Date(now.getTime() + 24 * 60 * 60 * 1000)

    if (date < twentyFourHoursFromNow) {
      setDueDateError('Due date must be at least 24 hours in the future')
      return false
    }

    setDueDateError('')
    return true
  }



  // Scroll to top when page loads
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  // Load departments and trainers
  useEffect(() => {
    const fetchData = async () => {
      try {
        const [departmentsData, trainersData] = await Promise.all([
          adminService.getDepartments(),
          adminService.getTrainers()
        ])
        setDepartments(departmentsData)
        setTrainers(trainersData)
      } catch (error) {
        console.error('Error fetching form builder data:', error)
      }
    }
    fetchData()
  }, [])

  // Handle pre-filled data from form requests or load existing form
  useEffect(() => {

    if (location.state?.fromRequest && location.state?.requestData) {
      // Pre-fill form from request

      setFromRequest(true)
      setReviewMode(location.state?.reviewMode || false)
      loadDefaultTemplateWithRequestData(location.state.requestData)
    } else if (formId && formId !== 'new') {
      // Load existing form

      loadForm(formId)
    } else {
      // Initialize with empty form for new form instead of loading default template
      setFormData({
        id: null,
        title: '',
        description: '',
        trainerId: '',
        departmentId: '',
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
    }
  }, [formId, location.state])

  const loadDefaultTemplate = async () => {
    setIsLoading(true)
    try {
      const template = await adminService.getDefaultFormTemplate()
      setFormData({
        ...template,
        trainerId: '',
        departmentId: '',
        dueDate: null,
        session: {
          name: '',
          date: '',
          course: '',
          duration: 60
        },
        questions: template.form_data?.sections?.flatMap(section =>
          section.questions || []
        ) || [],
        status: 'draft'
      })
    } catch (error) {
      console.error('Error loading default template:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadDefaultTemplateWithRequestData = async (requestData) => {
    setIsLoading(true)
    try {
      const template = await adminService.getDefaultFormTemplate()

      // Calculate due date from session date and validity duration
      let calculatedDueDate = null
      if (requestData.session_date && requestData.form_validity_duration) {
        const sessionDate = new Date(requestData.session_date)
        calculatedDueDate = new Date(sessionDate.getTime() + (requestData.form_validity_duration * 24 * 60 * 60 * 1000))
      }

      // Pre-fill with comprehensive request data
      setFormData({
        ...template,
        title: requestData.title || `${requestData.session_name || 'Training'} Feedback Form`,
        description: requestData.description || `Please provide your feedback for the ${requestData.session_name || 'training session'}.`,
        trainerId: requestData.trainer_id || '',
        departmentId: requestData.department_id || '',
        dueDate: calculatedDueDate,
        session: {
          name: requestData.session_name || '',
          date: requestData.session_date || '',
          course: requestData.course || '',
          duration: requestData.session_duration || 60
        },
        // Start with template questions but allow admin to modify
        questions: template.form_data?.sections?.flatMap(section =>
          section.questions || []
        ) || [],
        form_type: requestData.form_type || 'feedback',
        priority: requestData.priority || 'normal',
        status: 'draft',
        // Store original request data for approval later
        originalRequestId: requestData.id,
        requestNotes: requestData.additional_notes || ''
      })

      console.log('📋 Form pre-populated from request:', requestData)
    } catch (error) {
      console.error('Error loading template with request data:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadForm = async (id) => {
    setIsLoading(true)
    try {
      console.log(`🔄 Loading form data for form ID: ${id}`)

      // Fetch form data from API
      const formData = await adminService.getForm(id)
      console.log('📋 Loaded form data:', formData)

      // Transform backend data to frontend format
      const transformedForm = {
        id: formData.id,
        title: formData.title || '',
        description: formData.description || '',
        trainerId: formData.trainer_id || '',
        departmentId: formData.department_id || '',
        dueDate: formData.due_date ? new Date(formData.due_date) : null,
        session: formData.form_data?.session || {
          name: '',
          date: '',
          course: '',
          duration: 60
        },
        questions: formData.form_data?.questions || [],
        settings: formData.form_data?.settings || {
          allowAnonymous: true,
          requireAll: false,
          showProgress: true,
          randomizeQuestions: false
        },
        status: formData.status || 'draft',
        is_published: formData.is_published || false,
        created_at: formData.created_at,
        updated_at: formData.updated_at
      }

      console.log('🔄 Transformed form data:', transformedForm)
      setFormData(transformedForm)
      setIsLoading(false)
    } catch (error) {
      console.error('Error loading form:', error)
      setIsLoading(false)

      // Handle specific error cases
      if (error.message.includes('404') || error.message.includes('not found')) {
        alert('Form not found. It may have been deleted or you may not have permission to access it.')
        navigate('/admin-dashboard')
      } else if (error.message.includes('403') || error.message.includes('Access denied')) {
        alert('You do not have permission to edit this form.')
        navigate('/admin-dashboard')
      } else {
        alert(`Error loading form: ${error.message}`)
      }
    }
  }

  const handleSave = async (form) => {
    // Validate due date before saving
    if (form.dueDate && !validateDueDate(form.dueDate)) {
      setSaveStatus('error')
      setTimeout(() => setSaveStatus(null), 3000)
      return
    }

    setSaveStatus('saving')
    try {
      let savedForm

      if (formData.id) {
        // Update existing form
        savedForm = await adminService.updateForm(formData.id, form)
      } else {
        // Create new form
        savedForm = await adminService.createForm(form)
      }

      setSaveStatus('saved')
      setFormData(savedForm)

      // If this form was created from a request, approve the original request
      if (form.originalRequestId) {
        try {
          await adminService.approveFormRequest(form.originalRequestId)
          console.log('✅ Original request approved:', form.originalRequestId)
        } catch (error) {
          console.error('❌ Error approving original request:', error)
          // Don't fail the form save if request approval fails
        }
      }

      // Clear admin dashboard cache to ensure new form appears
      localStorage.removeItem('admin_dashboard_cache')
      sessionStorage.setItem('invalidate_admin_cache', 'true')

      // Clear save status after 3 seconds
      setTimeout(() => {
        setSaveStatus(null)
      }, 3000)
    } catch (error) {
      console.error('Error saving form:', error)
      console.error('Error message:', error.message)
      setSaveStatus('error')

      // Show detailed error message to user
      alert(`Error saving form: ${error.message}`)

      setTimeout(() => {
        setSaveStatus(null)
      }, 3000)
    }
  }

  const handlePreview = (form) => {
    setFormData(form)
    setCurrentView('preview')
  }

  const handlePublish = async () => {
    if (!formData) {
      alert('No form data to publish')
      return
    }

    console.log('📢 Starting form publish process')
    console.log('Form data:', formData)

    setSaveStatus('saving')
    try {
      // First save the form if it hasn't been saved yet
      let formToPublish = formData
      if (!formData.id) {
        console.log('📝 Form not saved yet, creating first...')
        formToPublish = await adminService.createForm(formData)
        console.log('✅ Form created:', formToPublish)
        setFormData(formToPublish)
      }

      // Then publish it
      console.log('📢 Publishing form with ID:', formToPublish.id)
      const publishedForm = await adminService.publishForm(formToPublish.id)
      console.log('✅ Form published:', publishedForm)

      setSaveStatus('saved')
      setFormData(publishedForm)

      // Clear admin dashboard cache to ensure published form appears
      localStorage.removeItem('admin_dashboard_cache')
      sessionStorage.setItem('invalidate_admin_cache', 'true')

      // Show success message and redirect
      setTimeout(() => {
        alert('Form published successfully! It will now appear on trainees\' dashboards.')
        navigate('/admin-dashboard')
      }, 1000)
    } catch (error) {
      console.error('❌ Error publishing form:', error)
      setSaveStatus('error')
      alert(`Failed to publish form: ${error.message}`)
      setTimeout(() => {
        setSaveStatus(null)
      }, 3000)
    }
  }

  const getSaveStatusMessage = () => {
    switch (saveStatus) {
      case 'saving':
        return 'Saving...'
      case 'saved':
        return 'Saved successfully!'
      case 'error':
        return 'Error saving form'
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

  if (isLoading) {
    return (
      <div className={`form-builder-page ${theme}`}>
        <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading form...</p>
        </div>
        <Footer />
      </div>
    )
  }

  return (
    <div className={`form-builder-page ${theme}`}>
      <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />

      {/* Notification for pre-filled form */}
      {fromRequest && (
        <motion.div
          className="request-notification"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Bell size={20} />
          <span>This form has been pre-filled from a trainer's request</span>
        </motion.div>
      )}

      <div className="form-builder-nav">
        <div className="nav-content">
          <button
            className="back-button"
            onClick={() => navigate('/admin-dashboard')}
          >
            <ArrowLeft size={20} />
            Back to Dashboard
          </button>

          {fromRequest && (
            <div className="request-indicator">
              <Bell size={16} />
              <span>
                {reviewMode ? 'Reviewing trainer-created form' : 'Creating form from trainer request'}
              </span>
            </div>
          )}

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
                disabled={!formData}
              >
                <Eye size={16} />
                Preview
              </button>
            </div>

            {saveStatus && (
              <motion.div
                className={`save-status ${getSaveStatusClass()}`}
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.8 }}
              >
                {getSaveStatusMessage()}
              </motion.div>
            )}
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

            {formData && formData.status === 'draft' && (
              <button
                className="publish-button"
                onClick={handlePublish}
                disabled={saveStatus === 'saving'}
              >
                <Send size={16} />
                Publish Form
              </button>
            )}

            {formData && formData.status === 'published' && (
              <div className="published-badge">
                <Send size={16} />
                Published
              </div>
            )}
          </div>
        </div>
      </div>

      <main className="form-builder-main">
        {isLoading ? (
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading form builder...</p>
          </div>
        ) : formData ? (
          currentView === 'builder' ? (
            <FormBuilder
              initialForm={formData}
              onSave={handleSave}
              onPreview={handlePreview}
              trainers={trainers}
              departments={departments}
            />
          ) : (
            <div className="preview-placeholder">
              <h3>Form Preview Temporarily Disabled</h3>
              <p>Form preview functionality is being rebuilt.</p>
              <button onClick={() => setCurrentView('builder')} className="btn-primary">
                Back to Builder
              </button>
            </div>
          )
        ) : (
          <div className="error-state">
            <p>Failed to load form template. Please try again.</p>
            <button onClick={() => window.location.reload()}>Retry</button>
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}

export default FormBuilderPage
