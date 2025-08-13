import { motion } from 'framer-motion'
import { Phone, Mail, MapPin, Facebook, Twitter, Instagram, Youtube } from 'lucide-react'
import { useLanguage } from '../contexts/LanguageContext'
import './Footer.css'

const Footer = () => {
  const { t } = useLanguage()
  const partners = [
    {
      name: 'DGCA',
      logo: '/images/dgca-logo.png',
      alt: 'DGCA Logo',
      url: 'https://www.dgca.gov.in',
      fullName: 'Directorate General of Civil Aviation'
    },
    {
      name: 'AAI',
      logo: '/images/aai-logo.png',
      alt: 'AAI Logo',
      url: 'https://www.aai.aero',
      fullName: 'Airports Authority of India'
    },
    {
      name: 'BCAS',
      logo: '/images/bcas-logo.png',
      alt: 'BCAS Logo',
      url: 'https://bcasindia.gov.in',
      fullName: 'Bureau of Civil Aviation Security'
    }
  ]

  return (
    <footer className="footer">
      {/* Joint Venture Partners Section */}
      <section className="partners-section">
        <div className="container">
          <motion.h2
            className="partners-title"
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
          >
            {t('footer.partners')}
          </motion.h2>
          <motion.div 
            className="partners-grid"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ duration: 1, delay: 0.2 }}
            viewport={{ once: true }}
          >
            {partners.map((partner, index) => (
              <motion.a
                key={partner.name}
                href={partner.url}
                target="_blank"
                rel="noopener noreferrer"
                className="partner-item"
                initial={{ opacity: 0, scale: 0.8 }}
                whileInView={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}
                whileHover={{ scale: 1.05, y: -5 }}
                whileTap={{ scale: 0.95 }}
              >
                <div className="partner-logo">
                  <img src={partner.logo} alt={partner.alt} />
                </div>
                <div className="partner-info">
                  <h4>{partner.name}</h4>
                  <p>{partner.fullName}</p>
                </div>
              </motion.a>
            ))}
          </motion.div>
        </div>
      </section>

      {/* Mobile Application Section */}
      <section className="mobile-app-section">
        <div className="container">
          <div className="mobile-app-content">
            <div className="mobile-app-text">
              <motion.h3
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
              >
                {t('footer.mobileApp.title')}
              </motion.h3>
              <motion.p
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8, delay: 0.2 }}
                viewport={{ once: true }}
              >
                {t('footer.mobileApp.description')}
              </motion.p>
              <motion.div
                className="app-buttons"
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8, delay: 0.4 }}
                viewport={{ once: true }}
              >
                <a
                  href="https://apps.apple.com/in/app/iaa-training/id123456789"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="app-button"
                >
                  <img src="/images/app-store-badge.png" alt="Download on App Store" />
                </a>
                <a
                  href="https://play.google.com/store/apps/details?id=in.edu.iaa.training"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="app-button"
                >
                  <img src="/images/google-play-badge.png" alt="Get it on Google Play" />
                </a>
              </motion.div>
            </div>
            <motion.div
              className="mobile-app-image"
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.8, delay: 0.3 }}
              viewport={{ once: true }}
            >
              <img
                src="/images/mobile.png"
                alt="Mobile App Preview"
                className="mobile-preview-image"
              />
            </motion.div>
          </div>
        </div>
      </section>

      {/* Main Footer */}
      <section className="main-footer">
        <div className="container">
          <div className="footer-content">
            {/* IAA Logo and Map */}
            <div className="footer-column logo-map-column">
              <div className="iaa-logo">
                <img src="/images/iaa-logo.png" alt="IAA Logo" />
              </div>
              <div className="location-map">
                <a
                  href="https://maps.app.goo.gl/hHmnfYewSeWdUT5ZA"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="map-link"
                >
                  <iframe
                    src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3505.3269333056523!2d77.13739807533167!3d28.5298913757209!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x390d1c72fb860727%3A0xa793309dbc24f52!2sIndian%20Aviation%20Academy!5e0!3m2!1sen!2sin!4v1754987702670!5m2!1sen!2sin"
                    width="250"
                    height="150"
                    style={{ border: 0 }}
                    allowFullScreen=""
                    loading="lazy"
                    referrerPolicy="no-referrer-when-downgrade"
                    title="IAA Location"
                  ></iframe>
                </a>
              </div>
            </div>

            {/* Quick Links */}
            <div className="footer-column">
              <h4>{t('footer.quickLinks.title')}</h4>
              <ul>
                <li><a href="https://iaa.edu.in/faq" target="_blank" rel="noopener noreferrer">{t('footer.quickLinks.faq')}</a></li>
                <li><a href="https://iaa.edu.in/contact-us" target="_blank" rel="noopener noreferrer">{t('footer.quickLinks.contactUs')}</a></li>
                <li><a href="https://iaa.edu.in/hotels" target="_blank" rel="noopener noreferrer">{t('footer.quickLinks.nearbyHotels')}</a></li>
                <li><a href="https://iaa.edu.in/RTI" target="_blank" rel="noopener noreferrer">{t('footer.quickLinks.rti')}</a></li>
                <li><a href="/register">{t('footer.quickLinks.register')}</a></li>
              </ul>
            </div>

            {/* Policies */}
            <div className="footer-column">
              <h4>{t('footer.policies.title')}</h4>
              <ul>
                <li><a href="https://iaa.edu.in/privacy_policy" target="_blank" rel="noopener noreferrer">{t('footer.policies.privacy')}</a></li>
                <li><a href="https://iaa.edu.in/cookie-policy" target="_blank" rel="noopener noreferrer">{t('footer.policies.cookie')}</a></li>
                <li><a href="https://iaa.edu.in/website-policies" target="_blank" rel="noopener noreferrer">{t('footer.policies.website')}</a></li>
              </ul>
            </div>
          </div>
        </div>
      </section>

      {/* Copyright */}
      <section className="copyright">
        <div className="container">
          <div className="copyright-content">
            <p>{t('footer.copyright')}</p>
          </div>
        </div>
      </section>
    </footer>
  )
}

export default Footer
