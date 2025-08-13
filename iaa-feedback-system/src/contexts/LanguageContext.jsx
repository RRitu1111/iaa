import { createContext, useContext, useState, useEffect } from 'react'
import { translations } from '../utils/translations'

const LanguageContext = createContext()

export const useLanguage = () => {
  const context = useContext(LanguageContext)
  if (!context) {
    throw new Error('useLanguage must be used within a LanguageProvider')
  }
  return context
}

export const LanguageProvider = ({ children }) => {
  const [currentLanguage, setCurrentLanguage] = useState('en')

  // Load saved language from localStorage
  useEffect(() => {
    const savedLanguage = localStorage.getItem('iaa-language')
    if (savedLanguage && translations[savedLanguage]) {
      setCurrentLanguage(savedLanguage)
    }
  }, [])

  // Save language to localStorage when changed
  useEffect(() => {
    localStorage.setItem('iaa-language', currentLanguage)
    // Update document language attribute
    document.documentElement.lang = currentLanguage
  }, [currentLanguage])

  const changeLanguage = (languageCode) => {
    if (translations[languageCode]) {
      setCurrentLanguage(languageCode)
    }
  }

  const t = (key) => {
    const keys = key.split('.')
    let value = translations[currentLanguage]
    
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k]
      } else {
        // Fallback to English if translation not found
        value = translations.en
        for (const fallbackKey of keys) {
          if (value && typeof value === 'object' && fallbackKey in value) {
            value = value[fallbackKey]
          } else {
            return key // Return key if no translation found
          }
        }
        break
      }
    }
    
    return value || key
  }

  const getCurrentLanguageInfo = () => {
    const languageInfo = {
      en: { name: 'English', nativeName: 'English', dir: 'ltr' },
      hi: { name: 'Hindi', nativeName: 'हिंदी', dir: 'ltr' },
      bn: { name: 'Bengali', nativeName: 'বাংলা', dir: 'ltr' },
      te: { name: 'Telugu', nativeName: 'తెలుగు', dir: 'ltr' },
      mr: { name: 'Marathi', nativeName: 'मराठी', dir: 'ltr' },
      ta: { name: 'Tamil', nativeName: 'தமிழ்', dir: 'ltr' },
      gu: { name: 'Gujarati', nativeName: 'ગુજરાતી', dir: 'ltr' },
      kn: { name: 'Kannada', nativeName: 'ಕನ್ನಡ', dir: 'ltr' },
      ml: { name: 'Malayalam', nativeName: 'മലയാളം', dir: 'ltr' },
      or: { name: 'Odia', nativeName: 'ଓଡ଼ିଆ', dir: 'ltr' },
      pa: { name: 'Punjabi', nativeName: 'ਪੰਜਾਬੀ', dir: 'ltr' },
      as: { name: 'Assamese', nativeName: 'অসমীয়া', dir: 'ltr' },
      ur: { name: 'Urdu', nativeName: 'اردو', dir: 'rtl' }
    }
    return languageInfo[currentLanguage] || languageInfo.en
  }

  const getAvailableLanguages = () => {
    return Object.keys(translations).map(code => ({
      code,
      name: {
        en: 'English', hi: 'हिंदी', bn: 'বাংলা', te: 'తెలుగు',
        mr: 'मराठी', ta: 'தமிழ்', gu: 'ગુજરાતી', kn: 'ಕನ್ನಡ',
        ml: 'മലയാളം', or: 'ଓଡ଼ିଆ', pa: 'ਪੰਜਾਬੀ', as: 'অসমীয়া', ur: 'اردو'
      }[code] || code
    }))
  }

  const value = {
    currentLanguage,
    changeLanguage,
    t,
    getCurrentLanguageInfo,
    getAvailableLanguages,
    isRTL: getCurrentLanguageInfo().dir === 'rtl'
  }

  return (
    <LanguageContext.Provider value={value}>
      <div dir={getCurrentLanguageInfo().dir}>
        {children}
      </div>
    </LanguageContext.Provider>
  )
}
