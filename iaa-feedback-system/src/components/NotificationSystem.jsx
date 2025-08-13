import { useState, useEffect, createContext, useContext } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Bell, 
  AlertTriangle, 
  CheckCircle, 
  Info, 
  X,
  Clock,
  TrendingDown,
  Mail
} from 'lucide-react'
import './NotificationSystem.css'

const NotificationContext = createContext()

export const useNotifications = () => {
  const context = useContext(NotificationContext)
  if (!context) {
    throw new Error('useNotifications must be used within a NotificationProvider')
  }
  return context
}

export const NotificationProvider = ({ children }) => {
  const [notifications, setNotifications] = useState([])

  const addNotification = (notification) => {
    const id = Date.now() + Math.random()
    const newNotification = {
      id,
      timestamp: new Date().toISOString(),
      read: false,
      ...notification
    }
    
    setNotifications(prev => [newNotification, ...prev])
    
    // Auto-remove after delay for non-persistent notifications
    if (!notification.persistent) {
      setTimeout(() => {
        removeNotification(id)
      }, notification.duration || 5000)
    }
    
    return id
  }

  const removeNotification = (id) => {
    setNotifications(prev => prev.filter(n => n.id !== id))
  }

  const markAsRead = (id) => {
    setNotifications(prev => prev.map(n => 
      n.id === id ? { ...n, read: true } : n
    ))
  }

  const markAllAsRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })))
  }

  const clearAll = () => {
    setNotifications([])
  }

  // Notification types with predefined configurations
  const showSuccess = (message, options = {}) => {
    return addNotification({
      type: 'success',
      title: 'Success',
      message,
      icon: CheckCircle,
      ...options
    })
  }

  const showError = (message, options = {}) => {
    return addNotification({
      type: 'error',
      title: 'Error',
      message,
      icon: AlertTriangle,
      persistent: true,
      ...options
    })
  }

  const showWarning = (message, options = {}) => {
    return addNotification({
      type: 'warning',
      title: 'Warning',
      message,
      icon: AlertTriangle,
      ...options
    })
  }

  const showInfo = (message, options = {}) => {
    return addNotification({
      type: 'info',
      title: 'Information',
      message,
      icon: Info,
      ...options
    })
  }

  const showThresholdAlert = (formTitle, currentRating, threshold, options = {}) => {
    return addNotification({
      type: 'threshold',
      title: 'Rating Threshold Alert',
      message: `${formTitle} ratings dropped to ${currentRating} (below ${threshold} threshold)`,
      icon: TrendingDown,
      persistent: true,
      priority: 'high',
      formId: options.formId,
      ...options
    })
  }

  const showDeadlineAlert = (formTitle, daysLeft, options = {}) => {
    return addNotification({
      type: 'deadline',
      title: 'Form Deadline Alert',
      message: `${formTitle} expires in ${daysLeft} day${daysLeft !== 1 ? 's' : ''}`,
      icon: Clock,
      persistent: true,
      priority: daysLeft <= 1 ? 'high' : 'medium',
      formId: options.formId,
      ...options
    })
  }

  const showReportGenerated = (formTitle, reportType, options = {}) => {
    return addNotification({
      type: 'report',
      title: 'Report Generated',
      message: `${reportType} report for "${formTitle}" is ready for download`,
      icon: Mail,
      persistent: true,
      priority: 'medium',
      ...options
    })
  }

  const showFormRejected = (formTitle, reason, options = {}) => {
    return addNotification({
      type: 'rejection',
      title: 'Form Request Rejected',
      message: `Your form request "${formTitle}" was rejected. Reason: ${reason}`,
      icon: X,
      persistent: true,
      priority: 'high',
      actionable: true,
      ...options
    })
  }

  const showFormApproved = (formTitle, options = {}) => {
    return addNotification({
      type: 'approval',
      title: 'Form Request Approved',
      message: `Your form request "${formTitle}" has been approved and published!`,
      icon: CheckCircle,
      persistent: true,
      priority: 'medium',
      ...options
    })
  }

  const value = {
    notifications,
    addNotification,
    removeNotification,
    markAsRead,
    markAllAsRead,
    clearAll,
    showSuccess,
    showError,
    showWarning,
    showInfo,
    showThresholdAlert,
    showDeadlineAlert,
    showReportGenerated,
    showFormRejected,
    showFormApproved,
    unreadCount: notifications.filter(n => !n.read).length
  }

  return (
    <NotificationContext.Provider value={value}>
      {children}
    </NotificationContext.Provider>
  )
}

// Notification Toast Component
export const NotificationToast = () => {
  const { notifications, removeNotification, markAsRead } = useNotifications()
  
  const visibleNotifications = notifications
    .filter(n => !n.persistent)
    .slice(0, 3) // Show max 3 toasts

  return (
    <div className="notification-toast-container">
      <AnimatePresence>
        {visibleNotifications.map((notification) => (
          <motion.div
            key={notification.id}
            className={`notification-toast ${notification.type} ${notification.priority || 'medium'}`}
            initial={{ opacity: 0, x: 300, scale: 0.8 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: 300, scale: 0.8 }}
            transition={{ duration: 0.3 }}
          >
            <div className="toast-icon">
              <notification.icon size={20} />
            </div>
            
            <div className="toast-content">
              <h4>{notification.title}</h4>
              <p>{notification.message}</p>
            </div>
            
            <button 
              className="toast-close"
              onClick={() => removeNotification(notification.id)}
            >
              <X size={16} />
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  )
}

// Notification Bell Component
export const NotificationBell = ({ className = '' }) => {
  const { 
    notifications, 
    unreadCount, 
    markAsRead, 
    markAllAsRead, 
    clearAll 
  } = useNotifications()
  
  const [isOpen, setIsOpen] = useState(false)
  
  const persistentNotifications = notifications
    .filter(n => n.persistent)
    .sort((a, b) => {
      // Sort by priority and timestamp
      const priorityOrder = { high: 3, medium: 2, low: 1 }
      const aPriority = priorityOrder[a.priority] || 2
      const bPriority = priorityOrder[b.priority] || 2
      
      if (aPriority !== bPriority) {
        return bPriority - aPriority
      }
      
      return new Date(b.timestamp) - new Date(a.timestamp)
    })

  const handleNotificationClick = (notification) => {
    markAsRead(notification.id)
    
    // Handle specific notification actions
    if (notification.formId) {
      // Could navigate to form details
      console.log('Navigate to form:', notification.formId)
    }
  }

  return (
    <div className={`notification-bell ${className}`}>
      <button 
        className="bell-button"
        onClick={() => setIsOpen(!isOpen)}
      >
        <Bell size={20} />
        {unreadCount > 0 && (
          <span className="notification-badge">
            {unreadCount > 99 ? '99+' : unreadCount}
          </span>
        )}
      </button>

      <AnimatePresence>
        {isOpen && (
          <motion.div
            className="notification-dropdown"
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
          >
            <div className="dropdown-header">
              <h3>Notifications</h3>
              <div className="header-actions">
                {unreadCount > 0 && (
                  <button onClick={markAllAsRead} className="mark-all-read">
                    Mark all read
                  </button>
                )}
                <button onClick={clearAll} className="clear-all">
                  Clear all
                </button>
              </div>
            </div>

            <div className="notifications-list">
              {persistentNotifications.length === 0 ? (
                <div className="empty-notifications">
                  <Bell size={32} />
                  <p>No notifications</p>
                </div>
              ) : (
                persistentNotifications.map((notification) => (
                  <motion.div
                    key={notification.id}
                    className={`notification-item ${notification.type} ${!notification.read ? 'unread' : ''}`}
                    onClick={() => handleNotificationClick(notification)}
                    whileHover={{ backgroundColor: 'rgba(0,0,0,0.05)' }}
                  >
                    <div className="notification-icon">
                      <notification.icon size={16} />
                    </div>
                    
                    <div className="notification-content">
                      <h4>{notification.title}</h4>
                      <p>{notification.message}</p>
                      <span className="notification-time">
                        {new Date(notification.timestamp).toLocaleString()}
                      </span>
                    </div>
                    
                    {!notification.read && (
                      <div className="unread-indicator" />
                    )}
                  </motion.div>
                ))
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// All components are already exported above with individual export statements
