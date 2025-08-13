import { useState } from 'react'
import { motion } from 'framer-motion'
import { Sun, Moon, Globe, Menu, X, Plane } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import { useNavigate } from 'react-router-dom'
import './Header.css'

const Header = ({ theme, toggleTheme, hideAuth = false }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isLanguageOpen, setIsLanguageOpen] = useState(false)
  const navigate = useNavigate()

  const { currentLanguage, changeLanguage, t, getAvailableLanguages, isRTL } = useLanguage()
  const availableLanguages = getAvailableLanguages()

  const scrollToLogin = () => {
    const loginSection = document.getElementById('login-section')
    if (loginSection) {
      loginSection.scrollIntoView({ behavior: 'smooth' })
    }
  }

  const handleLogoClick = () => {
    navigate('/')
  }

  return (
    <motion.header 
      className="header"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.8, ease: "easeOut" }}
    >
      <div className="header-container">
        {/* Left Section - Logo and IAA */}
        <div className="header-left">
          <motion.div
            className="logo-container"
            onClick={handleLogoClick}
            style={{ cursor: 'pointer' }}
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            transition={{ duration: 0.2 }}
          >
            <img
              src="/images/iaa-logo.png"
              alt="IAA Logo"
              className="iaa-logo"
            />
            <div className="iaa-text">
              <h1>Indian Aviation Academy</h1>
              <p className="motto">{t('header.motto')}</p>
            </div>
          </motion.div>
        </div>

        {/* Center Section - Aviation Animation */}
        <div className="header-center">
          <motion.div 
            className="aviation-animation"
            animate={{ 
              x: [0, 50, 0],
              rotate: [0, 5, 0]
            }}
            transition={{ 
              duration: 4,
              repeat: Infinity,
              ease: "easeInOut"
            }}
          >
            <Plane size={40} className="plane-icon" />
            <div className="cloud cloud-1"></div>
            <div className="cloud cloud-2"></div>
            <div className="cloud cloud-3"></div>
          </motion.div>
        </div>

        {/* Right Section - Navigation */}
        <div className="header-right">
          <nav className="nav-links">
            {/* Theme Toggle */}
            <button
              className="theme-toggle"
              onClick={toggleTheme}
              aria-label={t('header.toggleTheme')}
            >
              {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
            </button>

            {/* Language Selector */}
            <div className="language-selector">
              <button
                className="language-toggle"
                onClick={() => setIsLanguageOpen(!isLanguageOpen)}
              >
                <Globe size={20} />
                <span>{availableLanguages.find(lang => lang.code === currentLanguage)?.name}</span>
              </button>
              {isLanguageOpen && (
                <motion.div
                  className="language-dropdown"
                  initial={{ opacity: 0, y: -10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  {availableLanguages.map((lang) => (
                    <button
                      key={lang.code}
                      onClick={() => {
                        changeLanguage(lang.code)
                        setIsLanguageOpen(false)
                      }}
                      className={currentLanguage === lang.code ? 'active' : ''}
                    >
                      {lang.name}
                    </button>
                  ))}
                </motion.div>
              )}
            </div>

            {/* Navigation Links */}
            <a
              href="https://www.iaa.edu.in/about-iaa"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link"
            >
              {t('header.aboutIAA')}
            </a>
            <a
              href="https://www.iaa.edu.in/contact-us"
              target="_blank"
              rel="noopener noreferrer"
              className="nav-link"
            >
              {t('header.contacts')}
            </a>

            {/* Login Button */}
            {!hideAuth && (
              <motion.button
                className="login-button"
                onClick={scrollToLogin}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {t('header.loginRegister')}
              </motion.button>
            )}
          </nav>

          {/* Mobile Menu Toggle */}
          <button 
            className="mobile-menu-toggle"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {isMenuOpen && (
        <motion.div 
          className="mobile-menu"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
        >
          <div className="mobile-nav-links">
            <button onClick={toggleTheme} className="mobile-nav-item">
              {theme === 'light' ? <Moon size={20} /> : <Sun size={20} />}
              {t('header.toggleTheme')}
            </button>
            <div className="mobile-language-selector">
              <Globe size={20} />
              <select
                value={currentLanguage}
                onChange={(e) => changeLanguage(e.target.value)}
              >
                {availableLanguages.map((lang) => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name}
                  </option>
                ))}
              </select>
            </div>
            <a href="https://www.iaa.edu.in/about-iaa" className="mobile-nav-item">
              {t('header.aboutIAA')}
            </a>
            <a href="https://www.iaa.edu.in/contact-us" className="mobile-nav-item">
              {t('header.contacts')}
            </a>
            {!hideAuth && (
              <button onClick={scrollToLogin} className="mobile-login-button">
                {t('header.loginRegister')}
              </button>
            )}
          </div>
        </motion.div>
      )}
    </motion.header>
  )
}

export default Header
