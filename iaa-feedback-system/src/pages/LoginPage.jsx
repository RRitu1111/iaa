import { useState, useEffect } from 'react'
import { motion, useAnimation } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Plane, Lock, Eye, EyeOff } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header'
import Footer from '../components/Footer'

import './LoginPage.css'

const LoginPage = ({ theme, toggleTheme }) => {

  const [showPassword, setShowPassword] = useState(false)
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  })
  const [errors, setErrors] = useState({})
  const [isSubmitting, setIsSubmitting] = useState(false)

  const { t } = useLanguage()
  const { login } = useAuth()
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
    if (inView && planeControls) {
      planeControls.start({
        x: ['-100vw', '0%'],
        y: ['-20vh', '0%'],
        rotate: [10, 0],
        scale: [0.5, 1],
        transition: {
          duration: 3,
          ease: "easeOut"
        }
      }).catch(() => {})
    }
  }, [inView, planeControls])



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

    // Email validation
    if (!formData.email.trim()) {
      newErrors.email = 'Email is required'
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address'
    }

    // Password validation
    if (!formData.password) {
      newErrors.password = 'Password is required'
    } else if (formData.password.length < 3) {
      newErrors.password = 'Password must be at least 3 characters'
    }

    setErrors(newErrors)
    return Object.keys(newErrors).length === 0
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!validateForm()) return

    setIsSubmitting(true)

    // Airplane takeoff animation
    await planeControls.start({
      x: '100vw',
      y: '-30vh',
      rotate: -15,
      scale: 0.8,
      transition: {
        duration: 2.5,
        ease: "easeIn"
      }
    })

    // Authenticate user
    try {
      const userData = await login({
        email: formData.email,
        password: formData.password
      })

      if (userData.success) {
        alert(t('loginPage.form.success'))

        // Navigate to role-specific dashboard based on user's actual role
        const dashboardRoute = getDashboardRoute(userData.user.role)
        navigate(dashboardRoute)
      } else {
        setErrors({ general: userData.message || 'Login failed. Please try again.' })
        setIsSubmitting(false)
      }
    } catch (error) {
      setErrors({ general: error.message || 'Login failed. Please try again.' })
      setIsSubmitting(false)
    } finally {
      // Reset submitting state if not successful
      if (isSubmitting) {
        setIsSubmitting(false)
      }
    }
  }

  const getDashboardRoute = (role) => {
    switch (role) {
      case 'trainee':
        return '/trainee-dashboard'
      case 'trainer':
        return '/trainer-dashboard'
      case 'admin':
        return '/admin-dashboard'
      default:
        return '/'
    }
  }



  return (
    <div className="login-page">
      <Header theme={theme} toggleTheme={toggleTheme} />
      
      <main className="login-main" ref={ref}>
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
        <div className="login-content">
          <motion.div
            className="login-container"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: inView ? 1 : 0, y: inView ? 0 : 50 }}
            transition={{ delay: 1, duration: 0.8 }}
          >
            <div className="login-header">
              <h1>{t('loginPage.title')}</h1>
              <p>{t('loginPage.subtitle')}</p>
            </div>

            <motion.div
                className="login-form-container"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                transition={{ duration: 0.5, ease: "easeOut" }}
              >
                {/* Demo Instructions */}
                <div className="demo-info">
                  <p><strong>Demo Login:</strong> Register first, then use your ID@iaa.edu.in</p>
                  <p><em>Example: If you register with ID "admin123", login with "admin123@iaa.edu.in"</em></p>
                </div>

                <form onSubmit={handleSubmit} className="login-form">
                  <div className="form-group">
                    <div className="input-container">
                      <Lock className="input-icon" size={20} />
                      <div className="input-wrapper">
                        <input
                          type="email"
                          name="email"
                          placeholder="Email Address"
                          value={formData.email}
                          onChange={handleInputChange}
                          className={errors.email ? 'error' : ''}
                        />
                      </div>
                    </div>
                    {errors.email && <span className="error-message">{errors.email}</span>}
                  </div>

                  <div className="form-group">
                    <div className="input-container">
                      <Lock className="input-icon" size={20} />
                      <div className="input-wrapper">
                        <input
                          type={showPassword ? 'text' : 'password'}
                          name="password"
                          placeholder={t('loginPage.form.password')}
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

                  {/* General error display */}
                  {errors.general && (
                    <div className="error-message general-error">
                      {errors.general}
                    </div>
                  )}

                  <div className="forgot-password">
                    <button
                      type="button"
                      className="link-button"
                      onClick={() => navigate('/forgot-password')}
                    >
                      {t('loginPage.form.forgotPassword')}
                    </button>
                  </div>

                  <motion.button
                    type="submit"
                    className="login-button"
                    disabled={isSubmitting}
                    whileHover={{ scale: isSubmitting ? 1 : 1.02 }}
                    whileTap={{ scale: isSubmitting ? 1 : 0.98 }}
                  >
                    {isSubmitting ? 'Signing In...' : t('loginPage.form.login')}
                  </motion.button>
                </form>

                <div className="form-footer">
                  <p>
                    Don't have an account?{' '}
                    <button
                      type="button"
                      className="link-button"
                      onClick={() => {
                        navigate('/register')
                        setTimeout(() => window.scrollTo(0, 0), 100)
                      }}
                    >
                      Sign Up
                    </button>
                  </p>
                </div>
              </motion.div>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  )
}

export default LoginPage
