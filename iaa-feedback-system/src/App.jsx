import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { LanguageProvider } from './contexts/LanguageContext'
import { AuthProvider, useAuth } from './contexts/AuthContext'
import { NotificationProvider, NotificationToast } from './components/NotificationSystem'
import ErrorBoundary from './components/ErrorBoundary'
import HomePage from './pages/HomePage'
import RegisterPage from './pages/RegisterPage'
import LoginPage from './pages/LoginPage'
import ForgotPasswordPage from './pages/ForgotPasswordPage'
import AdminDashboard from './pages/AdminDashboard'
import TraineeDashboard from './pages/TraineeDashboard'
import TrainerDashboard from './pages/TrainerDashboard'
import FormBuilderPage from './pages/FormBuilderPage'
import TrainerFormBuilder from './pages/TrainerFormBuilder'
import FormFilling from './pages/FormFilling'
import ReportsPage from './pages/ReportsPage'
import FormResponses from './pages/FormResponses'
import LoadingPage from './pages/LoadingPage'
import ErrorPage from './pages/ErrorPage'
import BackendTest from './components/BackendTest'
import DepartmentSelection from './components/DepartmentSelection'
import ScrollToTop from './components/ScrollToTop'
import './App.css'

// AppContent component that handles department selection
const AppContent = ({ theme, toggleTheme }) => {
  const { needsDepartmentSelection, completeDepartmentSelection, skipDepartmentSelection } = useAuth()

  return (
    <>
      <Router>
        <Routes>
          <Route
            path="/"
            element={<HomePage theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/register"
            element={<RegisterPage theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/login"
            element={<LoginPage theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/forgot-password"
            element={<ForgotPasswordPage theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/admin-dashboard"
            element={<AdminDashboard theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/admin/form-builder/:formId?"
            element={<FormBuilderPage theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/form/:formId"
            element={<FormFilling theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/admin/reports"
            element={<ReportsPage theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/admin"
            element={<AdminDashboard theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/trainer"
            element={<TrainerDashboard theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/trainee-dashboard"
            element={<TraineeDashboard theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/trainer-dashboard"
            element={<TrainerDashboard theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/trainer/form-builder"
            element={<TrainerFormBuilder theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/admin/form-responses/:formId"
            element={<FormResponses theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/form-responses/:formId"
            element={<FormResponses theme={theme} toggleTheme={toggleTheme} />}
          />
          <Route
            path="/loading"
            element={<LoadingPage />}
          />
          <Route
            path="/error"
            element={<ErrorPage />}
          />
          <Route
            path="/backend-test"
            element={<BackendTest />}
          />
          {/* Catch-all route for 404 errors */}
          <Route
            path="*"
            element={<ErrorPage errorCode="404" />}
          />
        </Routes>
      </Router>

      {/* Department Selection Modal */}
      {needsDepartmentSelection && (
        <DepartmentSelection
          onComplete={completeDepartmentSelection}
          onSkip={skipDepartmentSelection}
          userRole="trainer"
        />
      )}

      {/* Scroll to Top Button */}
      <ScrollToTop />
    </>
  )
}

function App() {
  const [theme, setTheme] = useState('light')

  // Load saved theme from localStorage
  useEffect(() => {
    const savedTheme = localStorage.getItem('iaa-theme')
    if (savedTheme) {
      setTheme(savedTheme)
    }
  }, [])

  // Save theme to localStorage when changed
  useEffect(() => {
    localStorage.setItem('iaa-theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(theme === 'light' ? 'dark' : 'light')
  }

  return (
    <LanguageProvider>
      <AuthProvider>
        <NotificationProvider>
          <div className={`app ${theme}`}>
            <NotificationToast />
            <AppContent theme={theme} toggleTheme={toggleTheme} />
          </div>
        </NotificationProvider>
      </AuthProvider>
    </LanguageProvider>
  )
}

export default App
