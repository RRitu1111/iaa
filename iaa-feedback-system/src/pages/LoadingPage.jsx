import { useEffect, useState } from 'react'
import { motion, useAnimation } from 'framer-motion'
import { Plane, Loader, Wifi, WifiOff } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import './LoadingPage.css'

const LoadingPage = ({ 
  message = "Loading...", 
  subMessage = "Please wait while we prepare your experience",
  onComplete,
  duration = 3000,
  showProgress = true 
}) => {
  const [progress, setProgress] = useState(0)
  const [loadingStage, setLoadingStage] = useState('initializing')
  const [isOnline, setIsOnline] = useState(navigator.onLine)
  
  const { t } = useLanguage()
  const planeControls = useAnimation()
  const cloudControls = useAnimation()
  const progressControls = useAnimation()

  // Network status monitoring
  useEffect(() => {
    const handleOnline = () => setIsOnline(true)
    const handleOffline = () => setIsOnline(false)
    
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  // Loading stages
  const loadingStages = [
    { stage: 'initializing', message: 'Initializing system...', progress: 20 },
    { stage: 'connecting', message: 'Connecting to servers...', progress: 40 },
    { stage: 'loading', message: 'Loading resources...', progress: 70 },
    { stage: 'finalizing', message: 'Finalizing setup...', progress: 90 },
    { stage: 'complete', message: 'Ready for takeoff!', progress: 100 }
  ]

  // Progress simulation
  useEffect(() => {
    const interval = setInterval(() => {
      setProgress(prev => {
        const newProgress = Math.min(prev + Math.random() * 15, 100)
        
        // Update loading stage based on progress
        const currentStage = loadingStages.find(stage => 
          newProgress <= stage.progress
        ) || loadingStages[loadingStages.length - 1]
        
        setLoadingStage(currentStage.stage)
        
        if (newProgress >= 100) {
          clearInterval(interval)
          setTimeout(() => {
            onComplete && onComplete()
          }, 500)
        }
        
        return newProgress
      })
    }, duration / 20)

    return () => clearInterval(interval)
  }, [duration, onComplete])

  // Airplane animation
  useEffect(() => {
    planeControls.start({
      x: ['-100%', '100%'],
      y: [0, -20, 0, -10, 0],
      rotate: [0, 5, -5, 2, 0],
      transition: {
        duration: 4,
        repeat: Infinity,
        ease: "easeInOut"
      }
    })
  }, [planeControls])

  // Cloud animation
  useEffect(() => {
    cloudControls.start({
      x: ['100%', '-100%'],
      transition: {
        duration: 8,
        repeat: Infinity,
        ease: "linear"
      }
    })
  }, [cloudControls])

  // Progress bar animation
  useEffect(() => {
    progressControls.start({
      width: `${progress}%`,
      transition: {
        duration: 0.5,
        ease: "easeOut"
      }
    })
  }, [progress, progressControls])

  const getCurrentStageInfo = () => {
    return loadingStages.find(stage => stage.stage === loadingStage) || loadingStages[0]
  }

  return (
    <div className="loading-page">
      {/* Background Animation */}
      <div className="loading-background">
        <div className="sky-gradient"></div>
        
        {/* Animated Clouds */}
        <motion.div 
          className="cloud-layer"
          animate={cloudControls}
        >
          <div className="cloud cloud-1"></div>
          <div className="cloud cloud-2"></div>
          <div className="cloud cloud-3"></div>
          <div className="cloud cloud-4"></div>
        </motion.div>
      </div>

      {/* Main Content */}
      <div className="loading-content">
        {/* IAA Header */}
        <motion.div 
          className="loading-header"
          initial={{ opacity: 0, y: -50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
        >
          <div className="logo-section">
            <img
              src="/api/placeholder/80/80"
              alt="IAA Logo"
              className="loading-logo"
            />
            <div className="header-text">
              <h1>Indian Aviation Academy</h1>
              <p>Excellence in Aviation Training</p>
            </div>
          </div>
        </motion.div>

        {/* Airplane Animation */}
        <motion.div 
          className="airplane-container"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5, duration: 1 }}
        >
          <motion.div
            className="loading-airplane"
            animate={planeControls}
          >
            <Plane size={60} />
          </motion.div>
          
          {/* Flight Path */}
          <div className="flight-path">
            <div className="path-line"></div>
            <div className="path-dots">
              <span className="dot"></span>
              <span className="dot"></span>
              <span className="dot"></span>
            </div>
          </div>
        </motion.div>

        {/* Loading Information */}
        <motion.div 
          className="loading-info"
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 1, duration: 1 }}
        >
          <h2>{message}</h2>
          <p className="sub-message">{subMessage}</p>
          
          {/* Current Stage */}
          <div className="loading-stage">
            <motion.div 
              className="stage-indicator"
              key={loadingStage}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Loader className="stage-icon spinning" size={16} />
              <span>{getCurrentStageInfo().message}</span>
            </motion.div>
          </div>

          {/* Progress Bar */}
          {showProgress && (
            <div className="progress-container">
              <div className="progress-bar">
                <motion.div 
                  className="progress-fill"
                  animate={progressControls}
                />
              </div>
              <span className="progress-text">{Math.round(progress)}%</span>
            </div>
          )}

          {/* Network Status */}
          <div className="network-status">
            {isOnline ? (
              <div className="status online">
                <Wifi size={16} />
                <span>Connected</span>
              </div>
            ) : (
              <div className="status offline">
                <WifiOff size={16} />
                <span>Offline Mode</span>
              </div>
            )}
          </div>
        </motion.div>

        {/* Loading Dots Animation */}
        <motion.div 
          className="loading-dots"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5, duration: 1 }}
        >
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0 }}
          >
            •
          </motion.span>
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 0.5 }}
          >
            •
          </motion.span>
          <motion.span
            animate={{ opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: 1 }}
          >
            •
          </motion.span>
        </motion.div>
      </div>
    </div>
  )
}

// Example usage component
export const LoadingWrapper = ({ children, isLoading, ...loadingProps }) => {
  if (isLoading) {
    return <LoadingPage {...loadingProps} />
  }
  return children
}

export default LoadingPage
