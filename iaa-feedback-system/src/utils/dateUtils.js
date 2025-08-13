/**
 * Date utility functions for consistent date formatting across the application
 */

/**
 * Format date to dd/mm/yyyy format
 * @param {Date|string} date - Date object or date string
 * @returns {string} Formatted date string in dd/mm/yyyy format
 */
export const formatDate = (date) => {
  if (!date) return 'N/A'
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date
    if (isNaN(dateObj.getTime())) return 'Invalid Date'
    
    return dateObj.toLocaleDateString('en-GB')
  } catch (error) {
    console.error('Error formatting date:', error)
    return 'Invalid Date'
  }
}

/**
 * Format date and time to dd/mm/yyyy hh:mm format
 * @param {Date|string} date - Date object or date string
 * @returns {string} Formatted date and time string
 */
export const formatDateTime = (date) => {
  if (!date) return 'N/A'
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date
    if (isNaN(dateObj.getTime())) return 'Invalid Date'
    
    const dateStr = dateObj.toLocaleDateString('en-GB')
    const timeStr = dateObj.toLocaleTimeString('en-GB', { 
      hour: '2-digit', 
      minute: '2-digit' 
    })
    
    return `${dateStr} ${timeStr}`
  } catch (error) {
    console.error('Error formatting date time:', error)
    return 'Invalid Date'
  }
}

/**
 * Format date for input fields (yyyy-mm-dd)
 * @param {Date|string} date - Date object or date string
 * @returns {string} Formatted date string for input fields
 */
export const formatDateForInput = (date) => {
  if (!date) return ''
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date
    if (isNaN(dateObj.getTime())) return ''
    
    return dateObj.toISOString().split('T')[0]
  } catch (error) {
    console.error('Error formatting date for input:', error)
    return ''
  }
}

/**
 * Get relative time string (e.g., "2 days ago", "in 3 hours")
 * @param {Date|string} date - Date object or date string
 * @returns {string} Relative time string
 */
export const getRelativeTime = (date) => {
  if (!date) return 'N/A'
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date
    if (isNaN(dateObj.getTime())) return 'Invalid Date'
    
    const now = new Date()
    const diffMs = dateObj.getTime() - now.getTime()
    const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24))
    const diffHours = Math.ceil(diffMs / (1000 * 60 * 60))
    const diffMinutes = Math.ceil(diffMs / (1000 * 60))
    
    if (Math.abs(diffDays) >= 1) {
      return diffDays > 0 ? `in ${diffDays} day${diffDays !== 1 ? 's' : ''}` 
                          : `${Math.abs(diffDays)} day${Math.abs(diffDays) !== 1 ? 's' : ''} ago`
    } else if (Math.abs(diffHours) >= 1) {
      return diffHours > 0 ? `in ${diffHours} hour${diffHours !== 1 ? 's' : ''}` 
                           : `${Math.abs(diffHours)} hour${Math.abs(diffHours) !== 1 ? 's' : ''} ago`
    } else if (Math.abs(diffMinutes) >= 1) {
      return diffMinutes > 0 ? `in ${diffMinutes} minute${diffMinutes !== 1 ? 's' : ''}` 
                             : `${Math.abs(diffMinutes)} minute${Math.abs(diffMinutes) !== 1 ? 's' : ''} ago`
    } else {
      return 'just now'
    }
  } catch (error) {
    console.error('Error getting relative time:', error)
    return 'Invalid Date'
  }
}

/**
 * Check if a date is overdue
 * @param {Date|string} date - Date object or date string
 * @returns {boolean} True if date is in the past
 */
export const isOverdue = (date) => {
  if (!date) return false
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date
    if (isNaN(dateObj.getTime())) return false
    
    return dateObj.getTime() < new Date().getTime()
  } catch (error) {
    console.error('Error checking if overdue:', error)
    return false
  }
}

/**
 * Check if a date is due soon (within specified days)
 * @param {Date|string} date - Date object or date string
 * @param {number} days - Number of days to consider as "soon" (default: 2)
 * @returns {boolean} True if date is due within specified days
 */
export const isDueSoon = (date, days = 2) => {
  if (!date) return false
  
  try {
    const dateObj = typeof date === 'string' ? new Date(date) : date
    if (isNaN(dateObj.getTime())) return false
    
    const now = new Date()
    const diffMs = dateObj.getTime() - now.getTime()
    const diffDays = diffMs / (1000 * 60 * 60 * 24)
    
    return diffDays > 0 && diffDays <= days
  } catch (error) {
    console.error('Error checking if due soon:', error)
    return false
  }
}
