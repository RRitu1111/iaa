import { createContext, useContext, useState, useEffect } from 'react'
import authService from '../api/authService'

const AuthContext = createContext()

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)
  const [needsDepartmentSelection, setNeedsDepartmentSelection] = useState(false)

  // Initialize authentication state on component mount
  useEffect(() => {
    initializeAuth()
  }, [])

  /**
   * Initialize authentication state from stored data
   */
  const initializeAuth = () => {
    try {
      const currentUser = authService.getCurrentUser()
      const isAuth = authService.isAuthenticated()

      setUser(currentUser)
      setIsAuthenticated(isAuth)
    } catch (error) {
      console.error('Error initializing auth:', error)
      setError('Failed to initialize authentication')
    } finally {
      setIsLoading(false)
    }
  }

  /**
   * Login user
   * @param {Object} credentials - User credentials
   * @returns {Promise<Object>} Login result
   */
  const login = async (credentials) => {
    setIsLoading(true)
    setError(null)

    try {
      let result

      // Handle ID-based login (preferred method)
      if (credentials.role && credentials.userId) {
        console.log('Using ID-based login:', credentials.role, credentials.userId)
        result = await authService.loginWithId(credentials.role, credentials.userId, credentials.password)
      }
      // Handle email-based login (backward compatibility)
      else if (credentials.email) {
        console.log('Using email-based login:', credentials.email)
        result = await authService.login(credentials.email, credentials.password)
      }
      else {
        throw new Error('Invalid login credentials format')
      }

      if (result.success) {
        setUser(result.user)
        setIsAuthenticated(true)

        // Check if trainer needs department selection
        if (result.user.role === 'trainer' && !result.user.has_selected_departments) {
          setNeedsDepartmentSelection(true)
        }

        return {
          success: true,
          message: result.message,
          user: result.user
        }
      } else {
        setError(result.message)
        return {
          success: false,
          message: result.message
        }
      }
    } catch (error) {
      const errorMessage = 'Login failed. Please try again.'
      setError(errorMessage)
      return {
        success: false,
        message: errorMessage
      }
    } finally {
      setIsLoading(false)
    }
  }

  /**
   * Complete department selection for trainers
   */
  const completeDepartmentSelection = (selectedDepartments) => {
    setNeedsDepartmentSelection(false)
    // Update user object with department info if needed
    if (user) {
      setUser(prev => ({
        ...prev,
        departments: selectedDepartments,
        has_selected_departments: true
      }))
    }
  }

  /**
   * Skip department selection (optional)
   */
  const skipDepartmentSelection = () => {
    setNeedsDepartmentSelection(false)
  }

  /**
   * Logout user
   */
  const logout = () => {
    authService.logout()
    setUser(null)
    setIsAuthenticated(false)
    setError(null)
    setNeedsDepartmentSelection(false)
  }

  /**
   * Refresh authentication token
   * @returns {Promise<boolean>} Refresh success status
   */
  const refreshToken = async () => {
    try {
      const result = await authService.refreshAuthToken()
      return result.success
    } catch (error) {
      console.error('Token refresh failed:', error)
      logout() // Clear invalid session
      return false
    }
  }

  const updateUser = (updates) => {
    const updatedUser = { ...user, ...updates }
    setUser(updatedUser)
    localStorage.setItem('iaa_user', JSON.stringify(updatedUser))
  }

  const markFormCompleted = (formId) => {
    if (user && user.role === 'trainee') {
      const completedForms = [...(user.completedForms || []), formId]
      updateUser({ completedForms })
    }
  }

  const hasCompletedForm = (formId) => {
    if (user && user.role === 'trainee') {
      return user.completedForms?.includes(formId) || false
    }
    return false
  }

  const getDashboardRoute = () => {
    if (!user) return '/login'
    
    switch (user.role) {
      case 'trainee':
        return '/trainee-dashboard'
      case 'trainer':
        return '/trainer-dashboard'
      case 'admin':
        return '/admin-dashboard'
      default:
        return '/login'
    }
  }

  /**
   * Check if user has specific role
   * @param {string} role - Role to check
   * @returns {boolean} Role check result
   */
  const hasRole = (role) => {
    return authService.hasRole(role)
  }

  /**
   * Clear error state
   */
  const clearError = () => {
    setError(null)
  }

  // Context value object
  const value = {
    // State
    user,
    isAuthenticated,
    isLoading,
    error,
    needsDepartmentSelection,

    // Actions
    login,
    logout,
    refreshToken,
    clearError,
    updateUser,
    markFormCompleted,
    hasCompletedForm,
    getDashboardRoute,
    completeDepartmentSelection,
    skipDepartmentSelection,

    // Role checks
    hasRole,
    isAdmin: () => authService.isAdmin(),
    isTrainer: () => authService.isTrainer(),
    isTrainee: () => authService.isTrainee()
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export default AuthContext
