import { motion } from 'framer-motion'
import { ChevronLeft, ChevronRight } from 'lucide-react'
import { useState, useEffect } from 'react'
import { useLanguage } from '../contexts/LanguageContext'
import DataFlowVisualization from './DataFlowVisualization'
import AnonymousVisualization from './AnonymousVisualization'
import './HeroSection.css'

const HeroSection = () => {
  const [currentSlide, setCurrentSlide] = useState(0)
  const { t } = useLanguage()

  const heroImages = [
    {
      id: 1,
      titleKey: "hero.welcome.title",
      subtitleKey: "hero.welcome.subtitle",
      descriptionKey: "hero.welcome.description",
      image: "/images/iaa-building.jpg",
      alt: "Welcome to IAA - Aviation Training Excellence"
    },
    {
      id: 2,
      titleKey: "hero.dataFlow.title",
      subtitleKey: "hero.dataFlow.subtitle",
      descriptionKey: "hero.dataFlow.description",
      image: "/api/placeholder/800/600",
      alt: "Data Flow - Training Analytics System"
    },
    {
      id: 3,
      titleKey: "hero.anonymity.title",
      subtitleKey: "hero.anonymity.subtitle",
      descriptionKey: "hero.anonymity.description",
      image: "/api/placeholder/800/600",
      alt: "Anonymous Feedback - Secure and Private"
    }
  ]

  const nextSlide = () => {
    setCurrentSlide((prev) => (prev + 1) % heroImages.length)
  }

  const prevSlide = () => {
    setCurrentSlide((prev) => (prev - 1 + heroImages.length) % heroImages.length)
  }

  // Auto-slide functionality
  useEffect(() => {
    const interval = setInterval(nextSlide, 8000) // Increased from 5000 to 8000ms
    return () => clearInterval(interval)
  }, [])

  return (
    <section className="hero-section">
      <div className="hero-container">
        {/* Left Arrow Area (10%) */}
        <div className="hero-left-arrow">
          <button
            className="nav-arrow nav-arrow-left"
            onClick={prevSlide}
            aria-label="Previous slide"
          >
            <ChevronLeft size={24} />
          </button>
        </div>

        {/* Content Area (80%) */}
        <div className="hero-content-area">
          {/* Main Hero Content */}
          <div className="hero-content">
            <motion.div
              className={`hero-slide ${currentSlide === 1 ? 'data-flow-slide' : ''} ${currentSlide === 2 ? 'anonymous-slide' : ''}`}
              key={currentSlide}
              initial={{ opacity: 0, x: 100 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -100 }}
              transition={{ duration: 0.8, ease: "easeOut" }}
            >
              <div className={`hero-text ${(currentSlide === 0 || currentSlide === 1 || currentSlide === 2) ? 'text-35' : ''}`}>
                <motion.h1
                  initial={{ y: 50, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.2, duration: 0.8 }}
                >
                  {t(heroImages[currentSlide].titleKey)}
                </motion.h1>
                <motion.h2
                  initial={{ y: 50, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.4, duration: 0.8 }}
                >
                  {t(heroImages[currentSlide].subtitleKey)}
                </motion.h2>
                <motion.p
                  initial={{ y: 50, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  transition={{ delay: 0.6, duration: 0.8 }}
                >
                  {t(heroImages[currentSlide].descriptionKey)}
                </motion.p>
                {(currentSlide !== 1 && currentSlide !== 2) && (
                  <motion.div
                    className="hero-buttons"
                    initial={{ y: 50, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.8, duration: 0.8 }}
                  >
                    <button
                      className="cta-button primary"
                      onClick={() => {
                        const loginSection = document.getElementById('login-section')
                        if (loginSection) {
                          loginSection.scrollIntoView({ behavior: 'smooth' })
                        }
                      }}
                    >
                      {t('hero.getStarted')}
                    </button>
                    <button className="cta-button secondary">
                      {t('hero.learnMore')}
                    </button>
                  </motion.div>
                )}
              </div>
              <div className={`hero-image ${(currentSlide === 0 || currentSlide === 1 || currentSlide === 2) ? 'graphic-65' : ''}`}>
                {currentSlide === 1 ? (
                  <motion.div
                    className="data-flow-container"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3, duration: 0.8 }}
                  >
                    <DataFlowVisualization />
                  </motion.div>
                ) : currentSlide === 2 ? (
                  <motion.div
                    className="anonymous-container"
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3, duration: 0.8 }}
                  >
                    <AnonymousVisualization />
                  </motion.div>
                ) : (
                  <motion.img
                    src={heroImages[currentSlide].image}
                    alt={heroImages[currentSlide].alt}
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ delay: 0.3, duration: 0.8 }}
                  />
                )}
              </div>
            </motion.div>
          </div>
        </div>

        {/* Right Arrow Area (10%) */}
        <div className="hero-right-arrow">
          <button
            className="nav-arrow nav-arrow-right"
            onClick={nextSlide}
            aria-label="Next slide"
          >
            <ChevronRight size={24} />
          </button>
        </div>

        {/* Slide Indicators */}
        <div className="slide-indicators">
          {heroImages.map((_, index) => (
            <button
              key={index}
              className={`indicator ${index === currentSlide ? 'active' : ''}`}
              onClick={() => setCurrentSlide(index)}
              aria-label={`Go to slide ${index + 1}`}
            />
          ))}
        </div>

        {/* Scroll Indicator */}
        <motion.div
          className="scroll-indicator"
          animate={{ y: [0, 10, 0] }}
          transition={{ duration: 2, repeat: Infinity }}
        >
          <div className="scroll-arrow"></div>
          <span>{t('hero.scrollDown')}</span>
        </motion.div>
      </div>
    </section>
  )
}

export default HeroSection
