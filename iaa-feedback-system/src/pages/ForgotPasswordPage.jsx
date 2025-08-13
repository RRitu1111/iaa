import { useState, useEffect } from 'react'
import { motion, useAnimation } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { Plane, Lock, Mail, ArrowLeft } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header'
import Footer from '../components/Footer'
import './ForgotPasswordPage.css'

const ForgotPasswordPage = ({ theme, toggleTheme }) => {
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

  // Plane animation
  useEffect(() => {
    if (inView) {
      planeControls.start({
        x: '50vw',
        y: '10vh',
        rotate: 15,
        scale: 1,
        transition: {
          duration: 3,
          ease: "easeInOut"
        }
      })
    }
  }, [inView, planeControls])

  return (
    <div className="forgot-password-page">
      <Header theme={theme} toggleTheme={toggleTheme} />
      
      <main className="forgot-password-main" ref={ref}>
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
        <div className="forgot-password-content">
          <motion.div
            className="forgot-password-container"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: inView ? 1 : 0, y: inView ? 0 : 50 }}
            transition={{ delay: 1, duration: 0.8 }}
          >
            <div className="forgot-password-header">
              <Lock size={48} className="lock-icon" />
              <h1>Password Recovery</h1>
              <p>Contact admin to retrieve your password</p>
            </div>

            <div className="contact-info">
              <div className="info-card">
                <Mail size={32} className="info-icon" />
                <h3>Contact Administrator</h3>
                <p>
                  To reset your password, please contact your system administrator. 
                  They will be able to help you regain access to your account.
                </p>
                <div className="contact-details">
                  <p><strong>Email:</strong> admin@iaa.edu.in</p>
                  <p><strong>Phone:</strong> +91-80-2522-3001</p>
                  <p><strong>Office Hours:</strong> Monday - Friday, 9:00 AM - 5:00 PM</p>
                </div>
              </div>

              <div className="security-note">
                <h4>🔒 Security Notice</h4>
                <p>
                  For security reasons, password resets must be handled by the administrator. 
                  Please provide your role ID and department information when contacting support.
                </p>
              </div>
            </div>

            <div className="action-buttons">
              <motion.button
                type="button"
                className="back-button"
                onClick={() => navigate('/login')}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <ArrowLeft size={16} />
                Back to Login
              </motion.button>

              <motion.button
                type="button"
                className="contact-button"
                onClick={() => window.location.href = 'mailto:admin@iaa.edu.in?subject=Password Reset Request'}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <Mail size={16} />
                Send Email
              </motion.button>
            </div>
          </motion.div>
        </div>
      </main>

      <Footer />
    </div>
  )
}

export default ForgotPasswordPage
