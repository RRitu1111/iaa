import Header from '../components/Header'
import HeroSection from '../components/HeroSection'
import LoginSection from '../components/LoginSection'
import Footer from '../components/Footer'

const HomePage = ({ theme, toggleTheme }) => {
  return (
    <>
      <Header
        theme={theme}
        toggleTheme={toggleTheme}
      />
      <HeroSection />
      <LoginSection />
      <Footer />
    </>
  )
}

export default HomePage
