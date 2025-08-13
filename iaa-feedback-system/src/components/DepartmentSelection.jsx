import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Building, Check, X, Users, BookOpen } from 'lucide-react'
import adminService from '../api/adminService'
import './DepartmentSelection.css'

const DepartmentSelection = ({ onComplete, onSkip, userRole = 'trainer' }) => {
  const [departments, setDepartments] = useState([])
  const [selectedDepartments, setSelectedDepartments] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    fetchDepartments()
  }, [])

  const fetchDepartments = async () => {
    try {
      setLoading(true)
      const departmentData = await adminService.getDepartments()
      setDepartments(departmentData || [])
    } catch (error) {
      console.error('Error fetching departments:', error)
      setError('Failed to load departments. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const handleDepartmentToggle = (departmentId) => {
    setSelectedDepartments(prev => {
      if (prev.includes(departmentId)) {
        return prev.filter(id => id !== departmentId)
      } else {
        // For trainers, allow multiple selections
        // For trainees, allow only one selection
        if (userRole === 'trainee') {
          return [departmentId]
        } else {
          return [...prev, departmentId]
        }
      }
    })
  }

  const handleSubmit = async () => {
    if (selectedDepartments.length === 0) {
      setError('Please select at least one department.')
      return
    }

    setSubmitting(true)
    try {
      // Call the API to update user departments
      await adminService.updateUserDepartments(selectedDepartments)
      onComplete(selectedDepartments)
    } catch (error) {
      console.error('Error updating departments:', error)
      // For now, allow users to skip if there's an API error
      // This ensures the system remains functional while we debug
      setError('Department update failed, but you can continue. This will be fixed soon.')

      // Auto-complete after 3 seconds to not block the user
      setTimeout(() => {
        onComplete(selectedDepartments)
      }, 3000)
    } finally {
      setSubmitting(false)
    }
  }

  const getDepartmentIcon = (departmentName) => {
    const name = departmentName.toLowerCase()
    if (name.includes('pilot') || name.includes('flight')) return '✈️'
    if (name.includes('maintenance') || name.includes('engineering')) return '🔧'
    if (name.includes('cabin') || name.includes('crew')) return '👥'
    if (name.includes('ground') || name.includes('operations')) return '🏢'
    if (name.includes('safety') || name.includes('security')) return '🛡️'
    if (name.includes('training') || name.includes('education')) return '📚'
    return '🏛️'
  }

  if (loading) {
    return (
      <div className="department-selection-overlay">
        <div className="department-selection-modal">
          <div className="loading-container">
            <div className="loading-spinner"></div>
            <p>Loading departments...</p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="department-selection-overlay">
      <motion.div 
        className="department-selection-modal"
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.3 }}
      >
        <div className="modal-header">
          <div className="header-icon">
            <Building size={32} />
          </div>
          <h2>Select Your Department{userRole === 'trainer' ? 's' : ''}</h2>
          <p>
            {userRole === 'trainer' 
              ? 'Choose the departments you will be training in. You can select multiple departments.'
              : 'Choose your department to complete your profile setup.'
            }
          </p>
        </div>

        {error && (
          <div className="error-banner">
            <X size={16} />
            {error}
          </div>
        )}

        <div className="departments-grid">
          {departments.map((department) => (
            <motion.div
              key={department.id}
              className={`department-card ${selectedDepartments.includes(department.id) ? 'selected' : ''}`}
              onClick={() => handleDepartmentToggle(department.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
            >
              <div className="department-icon">
                {getDepartmentIcon(department.name)}
              </div>
              <div className="department-info">
                <h3>{department.name}</h3>
                {department.description && (
                  <p className="department-description">{department.description}</p>
                )}
              </div>
              <div className="selection-indicator">
                {selectedDepartments.includes(department.id) && (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="check-icon"
                  >
                    <Check size={16} />
                  </motion.div>
                )}
              </div>
            </motion.div>
          ))}
        </div>

        <div className="modal-actions">
          <div className="selection-summary">
            {selectedDepartments.length > 0 && (
              <span>
                {selectedDepartments.length} department{selectedDepartments.length !== 1 ? 's' : ''} selected
              </span>
            )}
          </div>
          
          <div className="action-buttons">
            {onSkip && (
              <button 
                className="btn-secondary" 
                onClick={onSkip}
                disabled={submitting}
              >
                Skip for now
              </button>
            )}
            <button 
              className="btn-primary" 
              onClick={handleSubmit}
              disabled={selectedDepartments.length === 0 || submitting}
            >
              {submitting ? (
                <>
                  <div className="button-spinner"></div>
                  Saving...
                </>
              ) : (
                <>
                  <Check size={16} />
                  Confirm Selection
                </>
              )}
            </button>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default DepartmentSelection
