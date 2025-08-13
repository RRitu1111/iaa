import { useEffect, useRef } from 'react'
import { motion, useAnimation } from 'framer-motion'
import { useInView } from 'react-intersection-observer'
import { useNavigate } from 'react-router-dom'
import { Plane } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import './LoginSection.css'

const LoginSection = () => {
  const { t } = useLanguage()
  const navigate = useNavigate()

  const planeControls = useAnimation()
  const runwayControls = useAnimation()
  const timeoutRef = useRef(null)
  const { ref, inView } = useInView({
    threshold: 0.3,
    triggerOnce: false
  })

  // Airplane landing animation when section comes into view
  useEffect(() => {
    if (inView) {
      // Airplane landing sequence
      planeControls.start({
        x: ['-50%', '30%'],
        y: ['-20vh', '0%'],
        rotate: [10, 0],
        scale: [0.5, 1],
        transition: {
          duration: 3,
          ease: "easeOut",
          times: [0, 1]
        }
      })

      // Runway extension animation
      runwayControls.start({
        width: ['0%', '100%'],
        transition: {
          duration: 2,
          ease: "easeInOut"
        }
      })
    }
  }, [inView, planeControls, runwayControls])

  // Cleanup timeout on component unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current)
      }
    }
  }, [])

  // Airplane takeoff animation on button click
  const handleButtonClick = async () => {
    // Takeoff animation
    await planeControls.start({
      x: '120%',
      y: '-30vh',
      rotate: -15,
      scale: 0.8,
      transition: {
        duration: 2.5,
        ease: "easeIn"
      }
    })

    // Clear any existing timeout
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
    }

    // Reset airplane position after takeoff using useEffect pattern
    // This ensures the component is still mounted when the animation runs
    timeoutRef.current = setTimeout(() => {
      // Check if component is still mounted before starting animation
      if (planeControls && typeof planeControls.start === 'function') {
        planeControls.start({
          x: '-50%',
          y: '-20vh',
          rotate: 10,
          scale: 0.5,
          transition: { duration: 0.5 }
        })
      }
    }, 3000)
  }

  return (
    <section id="login-section" className="login-section" ref={ref}>
      {/* Full Width Animation Area */}
      <div className="animation-area">
        {/* Sky Background */}
        <div className="sky-background">
          <div className="cloud cloud-1"></div>
          <div className="cloud cloud-2"></div>
          <div className="cloud cloud-3"></div>
          <div className="cloud cloud-4"></div>
        </div>

        {/* Airplane */}
        <motion.div
          className="airplane"
          animate={planeControls}
          initial={{
            x: '-50%',
            y: '-20vh',
            rotate: 10,
            scale: 0.5
          }}
        >
          <Plane size={120} className="plane-icon" />
          <div className="plane-trail"></div>
        </motion.div>

        {/* Runway */}
        <div className="runway-container">
          <motion.div
            className="runway"
            animate={runwayControls}
            initial={{ width: '0%' }}
          >
            <div className="runway-lines">
              {[...Array(10)].map((_, i) => (
                <div key={i} className="runway-line"></div>
              ))}
            </div>
          </motion.div>
          <div className="runway-lights">
            {[...Array(20)].map((_, i) => (
              <div key={i} className="runway-light"></div>
            ))}
          </div>
        </div>

        {/* Ground */}
        <div className="ground"></div>
      </div>

      {/* Sign In/Sign Up Ribbon on Right */}
      <div className="form-area">
        <motion.div
          className="auth-container"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: inView ? 1 : 0, x: inView ? 0 : 50 }}
          transition={{ delay: 1, duration: 0.8 }}
        >
          <div className="auth-header">
            <h2>{t('login.welcomeBack')}</h2>
            <p>Choose an option to continue</p>
          </div>

          <div className="auth-buttons">
            <motion.button
              className="auth-button register-button"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={async () => {
                await handleButtonClick()
                navigate('/register')
              }}
            >
              {t('login.signUp')}
            </motion.button>

            <motion.button
              className="auth-button login-button"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={async () => {
                await handleButtonClick()
                navigate('/login')
              }}
            >
              {t('login.signIn')}
            </motion.button>
          </div>
        </motion.div>
      </div>
    </section>
  )
}

export default LoginSection
