import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import './ScrollToTop.css'

const ScrollToTop = () => {
  const [isVisible, setIsVisible] = useState(false)

  // Show button when page is scrolled down
  useEffect(() => {
    const toggleVisibility = () => {
      if (window.pageYOffset > 300) {
        setIsVisible(true)
      } else {
        setIsVisible(false)
      }
    }

    window.addEventListener('scroll', toggleVisibility)
    return () => window.removeEventListener('scroll', toggleVisibility)
  }, [])

  const scrollToTop = () => {
    window.scrollTo({
      top: 0,
      behavior: 'smooth'
    })
  }

  return (
    <AnimatePresence>
      {isVisible && (
        <motion.button
          className="scroll-to-top"
          onClick={scrollToTop}
          initial={{ opacity: 0, scale: 0, rotate: 180 }}
          animate={{ opacity: 1, scale: 1, rotate: 0 }}
          exit={{ opacity: 0, scale: 0, rotate: 180 }}
          whileHover={{ 
            scale: 1.1, 
            y: -5,
            boxShadow: "0 10px 30px rgba(33, 150, 243, 0.4)"
          }}
          whileTap={{ scale: 0.9 }}
          transition={{ duration: 0.3 }}
          aria-label="Scroll to top"
        >
          {/* Fighter Jet SVG */}
          <svg
            className="fighter-jet"
            viewBox="0 0 100 100"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            {/* Main body */}
            <path
              d="M50 10 L45 25 L40 40 L35 55 L40 70 L45 75 L50 85 L55 75 L60 70 L65 55 L60 40 L55 25 Z"
              fill="currentColor"
              className="jet-body"
            />
            
            {/* Wings */}
            <path
              d="M35 45 L25 50 L30 55 L40 50 Z"
              fill="currentColor"
              className="jet-wing-left"
            />
            <path
              d="M65 45 L75 50 L70 55 L60 50 Z"
              fill="currentColor"
              className="jet-wing-right"
            />
            
            {/* Cockpit */}
            <circle
              cx="50"
              cy="30"
              r="4"
              fill="rgba(255, 255, 255, 0.8)"
              className="jet-cockpit"
            />
            
            {/* Exhaust flames */}
            <motion.path
              d="M45 85 L47 90 L50 95 L53 90 L55 85"
              fill="#ff6b35"
              className="jet-flame"
              animate={{
                opacity: [0.6, 1, 0.6],
                scaleY: [0.8, 1.2, 0.8]
              }}
              transition={{
                duration: 0.5,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
            
            {/* Navigation lights */}
            <motion.circle
              cx="35"
              cy="50"
              r="2"
              fill="#ff0000"
              className="nav-light-left"
              animate={{
                opacity: [0.3, 1, 0.3]
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
                ease: "easeInOut"
              }}
            />
            <motion.circle
              cx="65"
              cy="50"
              r="2"
              fill="#00ff00"
              className="nav-light-right"
              animate={{
                opacity: [0.3, 1, 0.3]
              }}
              transition={{
                duration: 1,
                repeat: Infinity,
                ease: "easeInOut",
                delay: 0.5
              }}
            />
          </svg>
          
          {/* Boost effect */}
          <motion.div
            className="boost-effect"
            animate={{
              opacity: [0, 0.8, 0],
              scale: [0.8, 1.2, 0.8]
            }}
            transition={{
              duration: 1,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          />
        </motion.button>
      )}
    </AnimatePresence>
  )
}

export default ScrollToTop
