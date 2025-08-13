import { useState, useEffect } from 'react'
import { motion, useAnimation } from 'framer-motion'
import { 
  Plane, 
  AlertTriangle, 
  Home, 
  RefreshCw, 
  ArrowLeft, 
  Wrench, 
  Settings,
  Phone,
  Mail
} from 'lucide-react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useLanguage } from '../contexts/LanguageContext'
import './ErrorPage.css'

const ErrorPage = ({ 
  errorCode = "404",
  errorTitle = "Page Not Found",
  errorMessage = "The page you're looking for seems to have taken an unexpected flight path.",
  showRetry = true,
  showGoHome = true,
  showGoBack = true,
  customActions = []
}) => {
  const [repairProgress, setRepairProgress] = useState(0)
  const [isRepairing, setIsRepairing] = useState(false)
  const [sparkles, setSparkles] = useState([])
  
  const navigate = useNavigate()
  const location = useLocation()
  const { t } = useLanguage()
  
  const planeControls = useAnimation()
  const wrenchControls = useAnimation()
  const sparkleControls = useAnimation()

  // Error type configurations
  const errorConfigs = {
    "404": {
      title: "Flight Path Not Found",
      message: "The page you're looking for seems to have taken an unexpected flight path.",
      icon: Plane,
      color: "#ff9800"
    },
    "500": {
      title: "Engine Malfunction",
      message: "Our servers are experiencing technical difficulties. Our maintenance crew is working on it.",
      icon: Settings,
      color: "#f44336"
    },
    "403": {
      title: "Restricted Airspace",
      message: "You don't have permission to access this area. Please check your credentials.",
      icon: AlertTriangle,
      color: "#ff5722"
    },
    "network": {
      title: "Communication Lost",
      message: "Unable to establish connection with ground control. Please check your internet connection.",
      icon: Phone,
      color: "#9c27b0"
    }
  }

  const currentConfig = errorConfigs[errorCode] || errorConfigs["404"]
  const ErrorIcon = currentConfig.icon

  // Plane repair animation
  useEffect(() => {
    planeControls.start({
      y: [0, -10, 0],
      rotate: [0, -2, 2, 0],
      transition: {
        duration: 3,
        repeat: Infinity,
        ease: "easeInOut"
      }
    })
  }, [planeControls])

  // Wrench animation
  useEffect(() => {
    wrenchControls.start({
      rotate: [0, 15, -15, 0],
      x: [0, 5, -5, 0],
      transition: {
        duration: 2,
        repeat: Infinity,
        ease: "easeInOut"
      }
    })
  }, [wrenchControls])

  // Generate sparkles for repair effect
  const generateSparkles = () => {
    const newSparkles = Array.from({ length: 8 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      delay: Math.random() * 0.5
    }))
    setSparkles(newSparkles)
  }

  // Repair simulation
  const handleRepair = () => {
    setIsRepairing(true)
    setRepairProgress(0)
    generateSparkles()
    
    const interval = setInterval(() => {
      setRepairProgress(prev => {
        const newProgress = prev + Math.random() * 20
        if (newProgress >= 100) {
          clearInterval(interval)
          setTimeout(() => {
            setIsRepairing(false)
            setRepairProgress(0)
            setSparkles([])
            // Simulate successful repair
            window.location.reload()
          }, 1000)
        }
        return Math.min(newProgress, 100)
      })
    }, 300)
  }

  // Navigation handlers
  const handleGoHome = () => {
    navigate('/')
  }

  const handleGoBack = () => {
    if (window.history.length > 1) {
      navigate(-1)
    } else {
      navigate('/')
    }
  }

  const handleRetry = () => {
    window.location.reload()
  }

  return (
    <div className="error-page">
      {/* Background */}
      <div className="error-background">
        <div className="sky-gradient"></div>
        
        {/* Animated Clouds */}
        <div className="cloud-layer">
          <div className="error-cloud cloud-1"></div>
          <div className="error-cloud cloud-2"></div>
          <div className="error-cloud cloud-3"></div>
        </div>
      </div>

      {/* Main Content */}
      <div className="error-content">
        {/* Error Header */}
        <motion.div 
          className="error-header"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <div className="error-code" style={{ color: currentConfig.color }}>
            {errorCode}
          </div>
          <h1>{errorTitle || currentConfig.title}</h1>
          <p>{errorMessage || currentConfig.message}</p>
        </motion.div>

        {/* Airplane with Repair Animation */}
        <motion.div 
          className="airplane-section"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5, duration: 1 }}
        >
          <div className="repair-scene">
            {/* Main Airplane */}
            <motion.div
              className="error-airplane"
              animate={planeControls}
            >
              <ErrorIcon size={80} style={{ color: currentConfig.color }} />
            </motion.div>

            {/* Repair Tools */}
            <motion.div
              className="repair-tools"
              animate={wrenchControls}
            >
              <Wrench size={24} className="repair-wrench" />
            </motion.div>

            {/* Sparkle Effects */}
            {sparkles.map((sparkle) => (
              <motion.div
                key={sparkle.id}
                className="sparkle"
                style={{
                  left: `${sparkle.x}%`,
                  top: `${sparkle.y}%`
                }}
                initial={{ opacity: 0, scale: 0 }}
                animate={{ 
                  opacity: [0, 1, 0], 
                  scale: [0, 1, 0],
                  rotate: [0, 180, 360]
                }}
                transition={{
                  duration: 1,
                  delay: sparkle.delay,
                  repeat: isRepairing ? Infinity : 0
                }}
              >
                ✨
              </motion.div>
            ))}

            {/* Repair Progress */}
            {isRepairing && (
              <motion.div 
                className="repair-progress"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <div className="progress-label">Repairing... {Math.round(repairProgress)}%</div>
                <div className="progress-bar">
                  <motion.div 
                    className="progress-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${repairProgress}%` }}
                    transition={{ duration: 0.3 }}
                  />
                </div>
              </motion.div>
            )}
          </div>
        </motion.div>

        {/* Action Buttons */}
        <motion.div 
          className="error-actions"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1, duration: 1 }}
        >
          <div className="action-buttons">
            {showGoHome && (
              <motion.button
                className="action-btn primary"
                onClick={handleGoHome}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <Home size={20} />
                Return to Base
              </motion.button>
            )}

            {showRetry && (
              <motion.button
                className="action-btn secondary"
                onClick={handleRetry}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <RefreshCw size={20} />
                Retry Flight
              </motion.button>
            )}

            {showGoBack && (
              <motion.button
                className="action-btn outline"
                onClick={handleGoBack}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <ArrowLeft size={20} />
                Go Back
              </motion.button>
            )}

            {/* Repair Button */}
            <motion.button
              className="action-btn repair"
              onClick={handleRepair}
              disabled={isRepairing}
              whileHover={{ scale: isRepairing ? 1 : 1.05 }}
              whileTap={{ scale: isRepairing ? 1 : 0.95 }}
            >
              <Wrench size={20} />
              {isRepairing ? 'Repairing...' : 'Emergency Repair'}
            </motion.button>

            {/* Custom Actions */}
            {customActions.map((action, index) => (
              <motion.button
                key={index}
                className={`action-btn ${action.variant || 'outline'}`}
                onClick={action.onClick}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {action.icon && <action.icon size={20} />}
                {action.label}
              </motion.button>
            ))}
          </div>
        </motion.div>

        {/* Support Information */}
        <motion.div 
          className="support-info"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
        >
          <h3>Need Technical Support?</h3>
          <div className="support-contacts">
            <div className="contact-item">
              <Mail size={16} />
              <span>support@iaa.edu.in</span>
            </div>
            <div className="contact-item">
              <Phone size={16} />
              <span>+91-XXX-XXX-XXXX</span>
            </div>
          </div>
          <p className="error-details">
            Error occurred at: {new Date().toLocaleString()}<br/>
            Path: {location.pathname}
          </p>
        </motion.div>
      </div>
    </div>
  )
}

export default ErrorPage
