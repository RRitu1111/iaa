import { useState, useEffect } from 'react'
import { motion, useAnimation } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Plane, User, Lock, Eye, EyeOff, Building } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import { useNavigate } from 'react-router-dom'
import { useAuth } from '../contexts/AuthContext'
import authService from '../api/authService'
import adminService from '../api/adminService'
import Header from '../components/Header'
import Footer from '../components/Footer'
import RoleAvatar from '../components/RoleAvatar'
import './RegisterPage.css'

const RegisterPage = ({ theme, toggleTheme }) => {
  const [selectedRole, setSelectedRole] = useState(null)
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [formData, setFormData] = useState({
    fullName: '',
    roleId: '',
    departmentId: '',
    password: '',
    confirmPassword: '',
    supervisorEmail: ''
  })
  const [errors, setErrors] = useState({})
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [departments, setDepartments] = useState([])
  const [loadingDepartments, setLoadingDepartments] = useState(false)

  const { t } = useLanguage()
  const navigate = useNavigate()
  const planeControls = useAnimation()
  const { ref, inView } = useInView({
    threshold: 0.3,
    triggerOnce: false
  })

  // Scroll to top when page loads
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  // Airplane landing animation when page loads
  useEffect(() => {
    if (inView) {
      planeControls.start({
        x: ['-100vw', '0%'],
        y: ['-20vh', '0%'],
        rotate: [10, 0],
        scale: [0.5, 1],
        transition: {
          duration: 3,
          ease: "easeOut"
        }
      })
    }
  }, [inView, planeControls])

  // Fetch departments when needed
  const fetchDepartments = async () => {
    if (departments.length > 0) return // Already loaded

    setLoadingDepartments(true)
    try {
      const departmentData = await adminService.getDepartments()
      setDepartments(departmentData || [])
    } catch (error) {
      setErrors(prev => ({ ...prev, departments: 'Failed to load departments' }))
    } finally {
      setLoadingDepartments(false)
    }
  }

  const handleRoleSelect = (role) => {
    setSelectedRole(role)
    setFormData(prev => ({ ...prev, roleId: '', departmentId: '' }))
    setErrors({})

    // Fetch departments when trainee role is selected
    if (role === 'trainee') {
      fetchDepartments()
    }
  }

  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prev => ({ ...prev, [name]: value }))
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: '' }))
    }
  }

  const validateForm = () => {
    const newErrors = {}

    // Full name validation
    if (!formData.fullName.trim()) {
      newErrors.fullName = 'Full name is required'
    } else if (formData.fullName.trim().length < 2) {
      newErrors.fullName = 'Full name must be at least 2 characters'
    } else if (!/^[a-zA-Z\s]+$/.test(formData.fullName.trim())) {
      newErrors.fullName = 'Full name can only contain letters and spaces'
    }

    // Role ID validation
    if (!formData.roleId.trim()) {
      newErrors.roleId = `${selectedRole} ID is required`
    } else if (formData.roleId.trim().length < 3) {
      newErrors.roleId = `${selectedRole} ID must be at least 3 characters`
    }

    // Department validation (required for trainees)
    if (selectedRole === 'trainee') {
      if (!formData.departmentId) {
        newErrors.departmentId = 'Please select a department'
      }
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required'
    } else if (formData.password.length < 6) {
      newErrors.password = 'Password must be at least 6 characters'
    } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(formData.password)) {
      newErrors.password = 'Password must contain at least one uppercase letter, one lowercase letter, and one number'
    }

    // Confirm password validation
    if (!formData.confirmPassword) {
      newErrors.confirmPassword = 'Please confirm your password'
    } else if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = 'Passwords do not match'
    }

    // Supervisor email validation (required for trainers)
    if (selectedRole === 'trainer') {
      if (!formData.supervisorEmail.trim()) {
        newErrors.supervisorEmail = 'Supervisor email is required'
      } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.supervisorEmail.trim())) {
        newErrors.supervisorEmail = 'Please enter a valid email address'
      } else if (!formData.supervisorEmail.toLowerCase().includes('iaa.edu.in')) {
        newErrors.supervisorEmail = 'Supervisor email should be from IAA domain (@iaa.edu.in)'
      }
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!validateForm()) return

    setIsSubmitting(true)

    // Actually register the user
    try {

      const registrationData = {
        email: `${formData.roleId}@iaa.edu.in`, // Convert roleId to email format
        password: formData.password,
        firstName: formData.fullName.split(' ')[0] || formData.fullName,
        lastName: formData.fullName.split(' ').slice(1).join(' ') || 'User',
        role: selectedRole,
        departmentId: selectedRole === 'trainee' ? parseInt(formData.departmentId) : 1, // Use selected department for trainees, default for others
        supervisorEmail: selectedRole === 'trainer' ? formData.supervisorEmail : null
      }

      const result = await authService.register(registrationData)

      if (result.success) {
        const generatedEmail = registrationData.email
        alert(`Registration successful! 🎉\n\nYour login email is: ${generatedEmail}\n\nPlease save this email for future logins.\n\nRedirecting to login page...`)
        setTimeout(() => navigate('/login'), 2000)
      } else {
        setErrors({ general: result.message || 'Registration failed. Please try again.' })
        setIsSubmitting(false)
      }
    } catch (error) {
      setErrors({ general: error.message || 'Registration failed. Please try again.' })
      setIsSubmitting(false)
    }
  }

  const getIdLabel = () => {
    if (!selectedRole) return ''
    return t(`register.form.${selectedRole}Id`)
  }

  return (
    <div className="register-page">
      <Header theme={theme} toggleTheme={toggleTheme} />
      
      <main className="register-main" ref={ref}>
        {/* Animation Area */}
        <div className="animation-area">
          <div className="sky-background">
            <div className="cloud cloud-1"></div>
            <div className="cloud cloud-2"></div>
            <div className="cloud cloud-3"></div>
          </div>

          <motion.div 
            className="airplane"
            animate={planeControls}
            initial={{ 
              x: '-100vw', 
              y: '-20vh', 
              rotate: 10, 
              scale: 0.5 
            }}
          >
            <Plane size={120} className="plane-icon" />
            <div className="plane-trail"></div>
          </motion.div>

          <div className="ground"></div>
        </div>

        {/* Content Area */}
        <div className="register-content">
          <motion.div
            className="register-container"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: inView ? 1 : 0, y: inView ? 0 : 50 }}
            transition={{ delay: 1, duration: 0.8 }}
          >
            <div className="register-header">
              <h1>{t('register.title')}</h1>
              <p>{t('register.subtitle')}</p>
            </div>

            <div className="role-selection">
              <h2>{t('register.question')}</h2>
              <div className="avatars-container">
                {['trainee', 'trainer'].map((role) => (
                  <RoleAvatar
                    key={role}
                    role={role}
                    isSelected={selectedRole === role}
                    onClick={handleRoleSelect}
                    t={t}
                  />
                ))}
              </div>

            </div>

            {selectedRole && (
              <motion.div
                className="registration-form-container"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              >

                <form onSubmit={handleSubmit} className="registration-form">
                  <div className="form-group">
                    <div className="input-container">
                      <User className="input-icon" size={20} />
                      <div className="input-wrapper">
                        <input
                          type="text"
                          name="fullName"
                          placeholder={t('register.form.fullName')}
                          value={formData.fullName}
                          onChange={handleInputChange}
                          className={errors.fullName ? 'error' : ''}
                        />
                      </div>
                    </div>
                    {errors.fullName && <span className="error-message">{errors.fullName}</span>}
                  </div>

                  <div className="form-group">
                    <div className="input-container">
                      <User className="input-icon" size={20} />
                      <div className="input-wrapper">
                        <input
                          type="text"
                          name="roleId"
                          placeholder={getIdLabel()}
                          value={formData.roleId}
                          onChange={handleInputChange}
                          className={errors.roleId ? 'error' : ''}
                        />
                      </div>
                    </div>
                    {errors.roleId && <span className="error-message">{errors.roleId}</span>}
                  </div>

                  {selectedRole === 'trainee' && (
                    <div className="form-group">
                      <div className="input-container">
                        <Building className="input-icon" size={20} />
                        <div className="input-wrapper">
                          <select
                            name="departmentId"
                            value={formData.departmentId}
                            onChange={handleInputChange}
                            className={errors.departmentId ? 'error' : ''}
                            disabled={loadingDepartments}
                          >
                            <option value="">
                              {loadingDepartments ? 'Loading departments...' : 'Select your department'}
                            </option>
                            {departments.map((dept) => (
                              <option key={dept.id} value={dept.id}>
                                {dept.name}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                      {errors.departmentId && <span className="error-message">{errors.departmentId}</span>}
                      {errors.departments && <span className="error-message">{errors.departments}</span>}
                    </div>
                  )}

                  <div className="form-group">
                    <div className="input-container">
                      <Lock className="input-icon" size={20} />
                      <div className="input-wrapper">
                        <input
                          type={showPassword ? 'text' : 'password'}
                          name="password"
                          placeholder={t('register.form.password')}
                          value={formData.password}
                          onChange={handleInputChange}
                          className={errors.password ? 'error' : ''}
                        />
                        <button
                          type="button"
                          className="password-toggle"
                          onClick={() => setShowPassword(!showPassword)}
                        >
                          {showPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                        </button>
                      </div>
                    </div>
                    {errors.password && <span className="error-message">{errors.password}</span>}
                  </div>

                  <div className="form-group">
                    <div className="input-container">
                      <Lock className="input-icon" size={20} />
                      <div className="input-wrapper">
                        <input
                          type={showConfirmPassword ? 'text' : 'password'}
                          name="confirmPassword"
                          placeholder={t('register.form.confirmPassword')}
                          value={formData.confirmPassword}
                          onChange={handleInputChange}
                          className={errors.confirmPassword ? 'error' : ''}
                        />
                        <button
                          type="button"
                          className="password-toggle"
                          onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        >
                          {showConfirmPassword ? <EyeOff size={20} /> : <Eye size={20} />}
                        </button>
                      </div>
                    </div>
                    {errors.confirmPassword && <span className="error-message">{errors.confirmPassword}</span>}
                  </div>

                  {/* Supervisor Email Field - Only for Trainers */}
                  {selectedRole === 'trainer' && (
                    <div className="form-group">
                      <label>Supervisor Email *</label>
                      <input
                        type="email"
                        name="supervisorEmail"
                        placeholder="supervisor@iaa.edu.in"
                        value={formData.supervisorEmail}
                        onChange={handleInputChange}
                        className={errors.supervisorEmail ? 'error' : ''}
                        required
                      />
                      {errors.supervisorEmail && <span className="error-message">{errors.supervisorEmail}</span>}
                      <small className="field-help">
                        Enter your supervisor's email address. They will receive notifications about your performance ratings.
                      </small>
                    </div>
                  )}

                  {/* General error display */}
                  {errors.general && (
                    <div className="error-message general-error">
                      {errors.general}
                    </div>
                  )}

                  <motion.button
                    type="submit"
                    className="register-button"
                    disabled={isSubmitting}
                    whileHover={{ scale: isSubmitting ? 1 : 1.02 }}
                    whileTap={{ scale: isSubmitting ? 1 : 0.98 }}
                  >
                    {isSubmitting ? 'Registering...' : t('register.form.register')}
                  </motion.button>
                </form>

                <div className="form-footer">
                  <p>
                    Already have an account?{' '}
                    <button
                      type="button"
                      className="link-button"
                      onClick={() => {
                        navigate('/login')
                        setTimeout(() => window.scrollTo(0, 0), 100)
                      }}
                    >
                      Sign In
                    </button>
                  </p>
                </div>
              </motion.div>
            )}

            {/* Bottom Sign In Section */}
            <motion.div
              className="bottom-signin-section"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1.5, duration: 0.6 }}
            >
              <div className="signin-divider">
                <span>or</span>
              </div>
              <div className="signin-content">
                <p>Already have an account?</p>
                <motion.button
                  type="button"
                  className="signin-bottom-button"
                  onClick={() => {
                    navigate('/login')
                    setTimeout(() => window.scrollTo(0, 0), 100)
                  }}
                  whileHover={{ scale: 1.05, boxShadow: "0 8px 25px rgba(0, 123, 255, 0.3)" }}
                  whileTap={{ scale: 0.95 }}
                >
                  Sign In Here
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  )
}

export default RegisterPage
