import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import adminService from '../api/adminService'
import { API_BASE_URL } from '../api/config'
import './FormResponses.css'

const FormResponses = ({ theme, toggleTheme }) => {
  const { formId } = useParams()
  const navigate = useNavigate()
  const { user } = useAuth()
  const [responses, setResponses] = useState([])
  const [form, setForm] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  // Get the appropriate dashboard URL based on user role
  const getDashboardUrl = () => {
    if (!user) return '/admin-dashboard'
    switch (user.role) {
      case 'trainer':
        return '/trainer-dashboard'
      case 'admin':
        return '/admin-dashboard'
      case 'trainee':
        return '/trainee-dashboard'
      default:
        return '/admin-dashboard'
    }
  }

  useEffect(() => {
    fetchFormResponses()
  }, [formId])

  const fetchFormResponses = async () => {
    try {
      setLoading(true)
      setError('')
      const data = await adminService.getFormResponses(formId)
      setForm(data.form)
      setResponses(data.responses)
    } catch (error) {
      console.error('Error fetching form responses:', error)
      setError(error.message || 'Failed to fetch form responses')
    } finally {
      setLoading(false)
    }
  }

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString()
  }

  const exportToCSV = () => {
    if (responses.length === 0) {
      alert('No responses to export')
      return
    }

    try {
      // Get all unique question IDs and their texts
      const questions = form?.form_data?.questions || []
      const questionMap = {}
      questions.forEach(q => {
        questionMap[q.id] = q.title || q.question || `Question ${q.id}`
      })

      // Create CSV headers
      const headers = [
        'Response ID',
        'Respondent',
        'Department',
        'Submission Date',
        ...questions.map(q => questionMap[q.id])
      ]

      // Create CSV rows
      const csvRows = [headers.join(',')]

      responses.forEach((response, index) => {
        const row = [
          `Response ${index + 1}`,
          'Anonymous Trainee',
          response.department_name || 'Unknown',
          new Date(response.submitted_at).toLocaleString(),
          ...questions.map(q => {
            const value = response.response_data?.[q.id] || ''
            // Handle arrays and escape commas/quotes
            const stringValue = Array.isArray(value) ? value.join('; ') : String(value)
            return `"${stringValue.replace(/"/g, '""')}"`
          })
        ]
        csvRows.push(row.join(','))
      })

      // Create and download CSV file
      const csvContent = csvRows.join('\n')
      const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.style.display = 'none'
      a.href = url
      a.download = `${form.title}_responses_data.csv`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)

      alert('CSV export completed successfully!')
    } catch (error) {
      console.error('Error exporting CSV:', error)
      alert('Failed to export CSV: ' + error.message)
    }
  }

  if (loading) {
    return (
      <div className={`form-responses-page ${theme}`}>
        <div className="loading">Loading form responses...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className={`form-responses-page ${theme}`}>
        <div className="error-message">
          <h2>Error</h2>
          <p>{error}</p>
          <button onClick={() => navigate(getDashboardUrl())} className="btn btn-primary">
            Back to Dashboard
          </button>
        </div>
      </div>
    )
  }

  return (
    <div className={`form-responses-page ${theme}`}>
      <header className="page-header">
        <div className="header-content">
          <button onClick={() => navigate(getDashboardUrl())} className="back-btn">
            ← Back to Dashboard
          </button>
          <h1>Form Responses</h1>
          <button onClick={toggleTheme} className="theme-toggle">
            {theme === 'light' ? '🌙' : '☀️'}
          </button>
        </div>
      </header>

      <main className="responses-content">
        {form && (
          <div className="form-info">
            <h2>{form.title}</h2>
            <p>{form.description}</p>
            <div className="response-stats">
              <span className="response-count">
                Total Responses: {responses.length}
              </span>
              {responses.length > 0 && (
                <div className="export-buttons">
                  <button onClick={exportToCSV} className="btn btn-secondary">
                    Export CSV Data
                  </button>
                </div>
              )}
            </div>
          </div>
        )}

        {responses.length === 0 ? (
          <div className="no-responses">
            <h3>No responses yet</h3>
            <p>This form hasn't received any responses yet.</p>
          </div>
        ) : (
          <div className="responses-list">
            {responses.map((response, index) => (
              <div key={response.id} className="response-card">
                <div className="response-header">
                  <h3>Response #{index + 1}</h3>
                  <div className="response-meta">
                    <span className="user-info">
                      Anonymous Trainee
                    </span>
                    <span className="department">
                      {response.department_name}
                    </span>
                    <span className="submitted-date">
                      {formatDate(response.submitted_at)}
                    </span>
                  </div>
                </div>
                <div className="response-data">
                  {response.response_data && Object.entries(response.response_data).map(([questionId, value]) => {
                    // Find the question text from form data
                    const question = form?.form_data?.questions?.find(q => q.id === questionId)
                    const questionText = question?.title || question?.question || `Question ${questionId}`

                    return (
                      <div key={questionId} className="response-field">
                        <label>{questionText}:</label>
                        <div className="field-value">
                          {Array.isArray(value) ? value.join(', ') : String(value)}
                        </div>
                      </div>
                    )
                  })}
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

export default FormResponses
