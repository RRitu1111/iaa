import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { ArrowLeft, Send, CheckCircle, AlertCircle, Star, Eye } from 'lucide-react'
import Header from '../components/Header'
import Footer from '../components/Footer'
import { useAuth } from '../contexts/AuthContext'
import adminService from '../api/adminService'
import { API_BASE_URL } from '../api/config'
import { formatDate, formatDateTime } from '../utils/dateUtils'
import './FormFilling.css'

const FormFilling = ({ theme, toggleTheme }) => {
  const { formId } = useParams()
  const navigate = useNavigate()
  const { user } = useAuth()
  const [form, setForm] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [responses, setResponses] = useState({})
  const [submitting, setSubmitting] = useState(false)
  const [submitted, setSubmitted] = useState(false)
  const [progress, setProgress] = useState(0)

  // Check if this is preview mode
  const searchParams = new URLSearchParams(window.location.search)
  const isPreviewMode = searchParams.get('preview') === 'true'

  useEffect(() => {
    const fetchForm = async () => {
      try {
        setLoading(true)
        const formData = await adminService.getForm(formId)
        console.log('📋 Form data loaded:', formData)
        console.log('📋 Form questions:', formData?.form_data?.questions)
        setForm(formData)
      } catch (error) {
        console.error('Error fetching form:', error)
        setError('Failed to load form')
      } finally {
        setLoading(false)
      }
    }

    if (formId) {
      fetchForm()
    }
  }, [formId])

  const handleInputChange = (questionId, value) => {
    setResponses(prev => {
      const newResponses = {
        ...prev,
        [questionId]: value
      }

      // Calculate progress
      const questions = form?.form_data?.questions || []
      const answeredQuestions = questions.filter(q =>
        newResponses[q.id] && newResponses[q.id] !== ''
      ).length
      const progressPercent = questions.length > 0 ? (answeredQuestions / questions.length) * 100 : 0
      setProgress(progressPercent)

      return newResponses
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)

    try {
      // Validate required fields
      const questions = form?.form_data?.questions || []
      const missingRequired = []

      questions.forEach(question => {
        if (question.required && (!responses[question.id] || responses[question.id] === '')) {
          missingRequired.push(question.title || `Question ${question.id}`)
        }
      })

      if (missingRequired.length > 0) {
        throw new Error(`Please fill in all required fields: ${missingRequired.join(', ')}`)
      }

      // Check if form is still accepting responses (due date check)
      if (form.due_date) {
        const dueDate = new Date(form.due_date)
        const now = new Date()
        if (now > dueDate) {
          throw new Error('This form is no longer accepting responses as the due date has passed.')
        }
      }
      const token = localStorage.getItem('iaa_token')
      if (!token) {
        throw new Error('No authentication token found')
      }

      const result = await adminService.submitFormResponse(form.id, responses)
      setSubmitted(true)

      // Refresh the page after a short delay to show success message
      setTimeout(() => {
        navigate('/trainee-dashboard', { replace: true })
      }, 2000)
    } catch (error) {
      console.error('Error submitting form:', error)
      setError(error.message || 'Failed to submit form')
    } finally {
      setSubmitting(false)
    }
  }

  const renderQuestion = (question) => {
    const value = responses[question.id] || ''

    switch (question.type) {
      case 'rating':
        return (
          <div className="rating-input">
            {[1, 2, 3, 4, 5].map((rating) => (
              <button
                key={rating}
                type="button"
                className={`rating-star ${value >= rating ? 'active' : ''}`}
                onClick={() => !isPreviewMode && handleInputChange(question.id, rating)}
                disabled={isPreviewMode}
              >
                <Star size={24} fill={value >= rating ? '#ffd700' : 'none'} />
              </button>
            ))}
            <span className="rating-text">
              {value ? `${value}/5` : isPreviewMode ? 'Preview Mode' : 'Click to rate'}
            </span>
          </div>
        )

      case 'textarea':
        return (
          <textarea
            className="question-textarea"
            placeholder={isPreviewMode ? "Preview Mode" : "Enter your response..."}
            value={value}
            onChange={(e) => !isPreviewMode && handleInputChange(question.id, e.target.value)}
            rows={4}
            required={question.required}
            disabled={isPreviewMode}
          />
        )

      case 'text':
        return (
          <input
            type="text"
            className="question-input"
            placeholder={isPreviewMode ? "Preview Mode" : "Enter your response..."}
            value={value}
            onChange={(e) => !isPreviewMode && handleInputChange(question.id, e.target.value)}
            required={question.required}
            disabled={isPreviewMode}
          />
        )

      case 'select':
        return (
          <select
            className="question-select"
            value={value}
            onChange={(e) => !isPreviewMode && handleInputChange(question.id, e.target.value)}
            required={question.required}
            disabled={isPreviewMode}
          >
            <option value="">{isPreviewMode ? "Preview Mode" : "Select an option..."}</option>
            {question.options?.map((option, index) => (
              <option key={index} value={option}>
                {option}
              </option>
            ))}
          </select>
        )

      case 'radio':
        return (
          <div className="radio-group">
            {question.options?.map((option, index) => (
              <label key={index} className="radio-option">
                <input
                  type="radio"
                  name={question.id}
                  value={option}
                  checked={value === option}
                  onChange={(e) => !isPreviewMode && handleInputChange(question.id, e.target.value)}
                  required={question.required}
                  disabled={isPreviewMode}
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
        )

      case 'checkbox':
        return (
          <div className="checkbox-group">
            {question.options?.map((option, index) => (
              <label key={index} className="checkbox-option">
                <input
                  type="checkbox"
                  value={option}
                  checked={Array.isArray(value) ? value.includes(option) : false}
                  onChange={(e) => {
                    if (!isPreviewMode) {
                      const currentValues = Array.isArray(value) ? value : []
                      if (e.target.checked) {
                        handleInputChange(question.id, [...currentValues, option])
                      } else {
                        handleInputChange(question.id, currentValues.filter(v => v !== option))
                      }
                    }
                  }}
                  disabled={isPreviewMode}
                />
                <span>{option}</span>
              </label>
            ))}
          </div>
        )

      default:
        return (
          <input
            type="text"
            className="question-input"
            placeholder={isPreviewMode ? "Preview Mode" : "Enter your response..."}
            value={value}
            onChange={(e) => !isPreviewMode && handleInputChange(question.id, e.target.value)}
            required={question.required}
            disabled={isPreviewMode}
          />
        )
    }
  }

  if (loading) {
    return (
      <div className={`form-filling ${theme}`}>
        <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading form...</p>
        </div>
        <Footer />
      </div>
    )
  }

  if (error && !form) {
    return (
      <div className={`form-filling ${theme}`}>
        <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
        <div className="error-container">
          <AlertCircle size={48} />
          <h2>Form Not Found</h2>
          <p>{error}</p>
          <button onClick={() => navigate('/trainee-dashboard')} className="btn-primary">
            Return to Dashboard
          </button>
        </div>
        <Footer />
      </div>
    )
  }

  if (submitted) {
    return (
      <div className={`form-filling ${theme}`}>
        <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
        <div className="success-container">
          <CheckCircle size={64} className="success-icon" />
          <h2>Form Submitted Successfully!</h2>
          <p>Thank you for your feedback. Your response has been recorded.</p>
          <button 
            onClick={() => navigate('/trainee-dashboard')} 
            className="btn-primary"
          >
            Return to Dashboard
          </button>
        </div>
        <Footer />
      </div>
    )
  }

  const questions = form?.form_data?.questions || []

  return (
    <div className={`form-filling ${theme}`}>
      <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
      
      <div className="form-filling-container">
        <div className="form-filling-header">
          <button onClick={() => navigate('/trainee-dashboard')} className="back-button">
            <ArrowLeft size={20} />
            Back to Dashboard
          </button>
        </div>

        <motion.div 
          className="form-content"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="form-header">
            <h1>{form?.title}</h1>
            {form?.description && <p className="form-description">{form.description}</p>}

            {/* Due Date Warning */}
            {form?.due_date && (
              <div className="due-date-warning">
                <AlertCircle size={16} />
                <span>
                  Due: {formatDateTime(form.due_date)}
                </span>
              </div>
            )}

            {/* Progress Bar */}
            {!submitted && (
              <div className="progress-container">
                <div className="progress-info">
                  <span>Progress: {Math.round(progress)}% complete</span>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>

          <form onSubmit={handleSubmit} className="form-questions">
            {questions.map((question, index) => (
              <motion.div
                key={question.id}
                className="question-card"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.1 }}
              >
                <div className="question-header">
                  <h3>
                    {index + 1}. {question.question || question.title}
                    {question.required && <span className="required">*</span>}
                  </h3>
                  {question.description && (
                    <p className="question-description">{question.description}</p>
                  )}
                </div>
                
                <div className="question-input">
                  {renderQuestion(question)}
                </div>
              </motion.div>
            ))}

            {error && (
              <div className="error-message">
                <AlertCircle size={16} />
                {error}
              </div>
            )}

            {!isPreviewMode && (
              <div className="form-actions">
                <button
                  type="submit"
                  className="submit-button"
                  disabled={submitting}
                >
                  <Send size={16} />
                  {submitting ? 'Submitting...' : 'Submit Response'}
                </button>
              </div>
            )}

            {isPreviewMode && (
              <div className="preview-actions">
                <div className="preview-banner">
                  <Eye size={16} />
                  <span>Preview Mode - Form submission is disabled</span>
                </div>
                <button
                  type="button"
                  className="back-button"
                  onClick={() => navigate(-1)}
                >
                  <ArrowLeft size={16} />
                  Back to Dashboard
                </button>
              </div>
            )}
          </form>
        </motion.div>
      </div>

      <Footer />
    </div>
  )
}

export default FormFilling
