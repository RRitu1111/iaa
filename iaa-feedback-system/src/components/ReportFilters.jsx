import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  X, 
  Calendar, 
  Users, 
  BookOpen, 
  FileText,
  Filter,
  RotateCcw
} from 'lucide-react'
import './ReportFilters.css'

const ReportFilters = ({ filters, onChange, onClose }) => {
  const [localFilters, setLocalFilters] = useState(filters)

  // Mock data for dropdowns
  const trainers = [
    { id: 'all', name: 'All Trainers' },
    { id: '1', name: 'Capt. John Smith' },
    { id: '2', name: 'Eng. Sarah Johnson' },
    { id: '3', name: 'Capt. Mike Wilson' },
    { id: '4', name: 'Inst. Lisa Brown' }
  ]

  const sessions = [
    { id: 'all', name: 'All Sessions' },
    { id: '1', name: 'Basic Navigation' },
    { id: '2', name: 'Engine Inspection' },
    { id: '3', name: 'Emergency Landing' },
    { id: '4', name: 'Instrument Flying' },
    { id: '5', name: 'Radio Communication' }
  ]

  const formTypes = [
    { id: 'all', name: 'All Form Types' },
    { id: 'training', name: 'Training Feedback' },
    { id: 'evaluation', name: 'Performance Evaluation' },
    { id: 'course', name: 'Course Feedback' },
    { id: 'instructor', name: 'Instructor Feedback' }
  ]

  const statusOptions = [
    { id: 'all', name: 'All Statuses' },
    { id: 'active', name: 'Active' },
    { id: 'completed', name: 'Completed' },
    { id: 'draft', name: 'Draft' },
    { id: 'archived', name: 'Archived' }
  ]

  const dateRangeOptions = [
    { id: 'today', name: 'Today' },
    { id: 'yesterday', name: 'Yesterday' },
    { id: 'last-7-days', name: 'Last 7 days' },
    { id: 'last-30-days', name: 'Last 30 days' },
    { id: 'last-90-days', name: 'Last 90 days' },
    { id: 'this-month', name: 'This month' },
    { id: 'last-month', name: 'Last month' },
    { id: 'this-year', name: 'This year' },
    { id: 'custom', name: 'Custom range' }
  ]

  const handleFilterChange = (key, value) => {
    const newFilters = { ...localFilters, [key]: value }
    setLocalFilters(newFilters)
  }

  const handleApplyFilters = () => {
    onChange(localFilters)
    onClose()
  }

  const handleResetFilters = () => {
    const resetFilters = {
      dateRange: 'last-30-days',
      startDate: '',
      endDate: '',
      trainer: 'all',
      session: 'all',
      formType: 'all',
      status: 'all'
    }
    setLocalFilters(resetFilters)
    onChange(resetFilters)
  }

  const isCustomDateRange = localFilters.dateRange === 'custom'

  return (
    <motion.div
      className="report-filters"
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
    >
      <div className="filters-header">
        <div className="header-left">
          <Filter size={20} />
          <h3>Filter Reports</h3>
        </div>
        <button className="close-btn" onClick={onClose}>
          <X size={20} />
        </button>
      </div>

      <div className="filters-content">
        <div className="filters-grid">
          {/* Date Range Filter */}
          <div className="filter-group">
            <label>
              <Calendar size={16} />
              Date Range
            </label>
            <select
              value={localFilters.dateRange}
              onChange={(e) => handleFilterChange('dateRange', e.target.value)}
            >
              {dateRangeOptions.map(option => (
                <option key={option.id} value={option.id}>
                  {option.name}
                </option>
              ))}
            </select>
          </div>

          {/* Custom Date Range */}
          {isCustomDateRange && (
            <>
              <div className="filter-group">
                <label>Start Date</label>
                <input
                  type="date"
                  value={localFilters.startDate}
                  onChange={(e) => handleFilterChange('startDate', e.target.value)}
                />
              </div>
              <div className="filter-group">
                <label>End Date</label>
                <input
                  type="date"
                  value={localFilters.endDate}
                  onChange={(e) => handleFilterChange('endDate', e.target.value)}
                />
              </div>
            </>
          )}

          {/* Trainer Filter */}
          <div className="filter-group">
            <label>
              <Users size={16} />
              Trainer
            </label>
            <select
              value={localFilters.trainer}
              onChange={(e) => handleFilterChange('trainer', e.target.value)}
            >
              {trainers.map(trainer => (
                <option key={trainer.id} value={trainer.id}>
                  {trainer.name}
                </option>
              ))}
            </select>
          </div>

          {/* Session Filter */}
          <div className="filter-group">
            <label>
              <BookOpen size={16} />
              Training Session
            </label>
            <select
              value={localFilters.session}
              onChange={(e) => handleFilterChange('session', e.target.value)}
            >
              {sessions.map(session => (
                <option key={session.id} value={session.id}>
                  {session.name}
                </option>
              ))}
            </select>
          </div>

          {/* Form Type Filter */}
          <div className="filter-group">
            <label>
              <FileText size={16} />
              Form Type
            </label>
            <select
              value={localFilters.formType}
              onChange={(e) => handleFilterChange('formType', e.target.value)}
            >
              {formTypes.map(type => (
                <option key={type.id} value={type.id}>
                  {type.name}
                </option>
              ))}
            </select>
          </div>

          {/* Status Filter */}
          <div className="filter-group">
            <label>Status</label>
            <select
              value={localFilters.status}
              onChange={(e) => handleFilterChange('status', e.target.value)}
            >
              {statusOptions.map(status => (
                <option key={status.id} value={status.id}>
                  {status.name}
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="filters-actions">
          <button 
            className="reset-btn"
            onClick={handleResetFilters}
          >
            <RotateCcw size={16} />
            Reset Filters
          </button>
          <div className="action-buttons">
            <button 
              className="cancel-btn"
              onClick={onClose}
            >
              Cancel
            </button>
            <button 
              className="apply-btn"
              onClick={handleApplyFilters}
            >
              Apply Filters
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default ReportFilters
