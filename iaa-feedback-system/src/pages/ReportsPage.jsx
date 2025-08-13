import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  ArrowLeft,
  Filter,
  Download,
  Calendar,
  Users,
  FileText,
  BarChart3,
  TrendingUp,
  Eye,
  Search,
  RefreshCw,
  Settings
} from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { useLanguage } from '../contexts/LanguageContext'
import Header from '../components/Header'
import Footer from '../components/Footer'
import ReportFilters from '../components/ReportFilters'
import ReportCharts from '../components/ReportCharts'
import ReportTable from '../components/ReportTable'
import FormSpecificReports from '../components/FormSpecificReports'
import { API_BASE_URL } from '../api/config'
import './ReportsPage.css'

const ReportsPage = ({ theme, toggleTheme }) => {

  const [activeTab, setActiveTab] = useState('overview')
  const [filters, setFilters] = useState({
    dateRange: 'last-30-days',
    startDate: '',
    endDate: '',
    trainer: 'all',
    session: 'all',
    formType: 'all',
    status: 'all'
  })
  const [reportData, setReportData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [showFilters, setShowFilters] = useState(false)
  const [availableForms, setAvailableForms] = useState([])
  const [selectedForm, setSelectedForm] = useState(null)
  const [generatedReport, setGeneratedReport] = useState(null)
  const [reportSettings, setReportSettings] = useState({
    threshold_low: 3.0,
    threshold_medium: 3.5,
    auto_generate: true,
    email_alerts: false
  })

  const { t } = useLanguage()
  const navigate = useNavigate()

  // Helper functions for data transformation
  const calculateResponseRate = (responses, forms) => {
    if (forms === 0) return 0
    return Math.round((responses / forms) * 100)
  }

  const generateResponseTrends = (stats) => {
    // Generate mock trend data based on total responses
    const totalResponses = stats?.totalResponses || 0
    const days = 7
    const data = []

    for (let i = days - 1; i >= 0; i--) {
      const date = new Date()
      date.setDate(date.getDate() - i)
      const responses = Math.floor(totalResponses / days) + Math.floor(Math.random() * 5)
      data.push({
        date: date.toISOString().split('T')[0],
        responses: responses
      })
    }
    return data
  }

  const generateRatingDistribution = (stats) => {
    // Generate mock rating distribution
    const totalResponses = stats?.totalResponses || 0
    if (totalResponses === 0) return []

    return [
      { rating: 1, count: Math.floor(totalResponses * 0.05) },
      { rating: 2, count: Math.floor(totalResponses * 0.10) },
      { rating: 3, count: Math.floor(totalResponses * 0.20) },
      { rating: 4, count: Math.floor(totalResponses * 0.35) },
      { rating: 5, count: Math.floor(totalResponses * 0.30) }
    ]
  }

  const generateTrainerPerformance = (stats) => {
    // Generate mock trainer performance data
    const totalTrainers = stats?.totalTrainers || 0
    if (totalTrainers === 0) return []

    const data = []
    for (let i = 1; i <= Math.min(totalTrainers, 5); i++) {
      data.push({
        name: `Trainer ${i}`,
        forms: Math.floor(Math.random() * 10) + 1,
        responses: Math.floor(Math.random() * 50) + 10,
        rating: (Math.random() * 2 + 3).toFixed(1)
      })
    }
    return data
  }

  const generateTableData = (stats) => {
    // Generate mock table data
    const totalForms = stats?.totalForms || 0
    if (totalForms === 0) return []

    const data = []
    for (let i = 1; i <= Math.min(totalForms, 10); i++) {
      data.push({
        id: i,
        title: `Training Form ${i}`,
        trainer: `Trainer ${Math.floor(Math.random() * 5) + 1}`,
        session: `Session ${Math.floor(Math.random() * 3) + 1}`,
        responses: Math.floor(Math.random() * 20) + 5,
        rating: (Math.random() * 2 + 3).toFixed(1),
        status: Math.random() > 0.5 ? 'Active' : 'Completed',
        date: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toLocaleDateString()
      })
    }
    return data
  }

  // Scroll to top when page loads
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [])

  // Load initial report data
  useEffect(() => {
    loadReportData()
    loadAvailableForms()
    loadReportSettings()
  }, [filters])

  const loadAvailableForms = async () => {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/reports/forms`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setAvailableForms(data.forms || [])
      }
    } catch (error) {
      console.error('Error loading available forms:', error)
    }
  }

  const loadReportSettings = async () => {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/reports/settings`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setReportSettings(data.settings || reportSettings)
      }
    } catch (error) {
      console.error('Error loading report settings:', error)
    }
  }

  const generateFormReport = async (formId) => {
    setIsLoading(true)
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/reports/generate/${formId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setGeneratedReport(data.report)
        setActiveTab('detailed-report')
      } else {
        const errorData = await response.json()
        alert(`Error generating report: ${errorData.detail}`)
      }
    } catch (error) {
      console.error('Error generating report:', error)
      alert('Failed to generate report')
    } finally {
      setIsLoading(false)
    }
  }

  const updateReportSettings = async (newSettings) => {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/reports/settings`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(newSettings)
      })

      if (response.ok) {
        const data = await response.json()
        setReportSettings(prev => ({ ...prev, ...data.settings }))
        alert('Report settings updated successfully')
      } else {
        const errorData = await response.json()
        alert(`Error updating settings: ${errorData.detail}`)
      }
    } catch (error) {
      console.error('Error updating report settings:', error)
      alert('Failed to update settings')
    }
  }

  const loadReportData = async () => {
    setIsLoading(true)
    try {
      // Fetch real data from backend reports endpoint
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/reports/dashboard`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        console.log('📊 Reports dashboard data received:', data)

        // Transform backend data to frontend format
        const realData = {
          summary: {
            totalForms: data.stats?.totalForms || 0,
            totalResponses: data.stats?.totalResponses || 0,
            averageRating: data.stats?.averageRating || 0,
            responseRate: data.stats?.responseRate || 0
          },
          chartData: {
            responsesByDate: data.stats?.responseTrends || [],
            ratingDistribution: generateRatingDistribution(data.stats),
            trainerPerformance: generateTrainerPerformance(data.stats)
          },
          tableData: data.stats?.formsData || []
        }

        console.log('📈 Transformed report data:', realData)
        setReportData(realData)
      } else {
        console.error('Failed to fetch dashboard stats')
        // Fallback to empty data
        setReportData({
          summary: { totalForms: 0, totalResponses: 0, averageRating: 0, responseRate: 0 },
          chartData: { responsesByDate: [], ratingDistribution: [], trainerPerformance: [] },
          tableData: []
        })
      }
    } catch (error) {
      console.error('Error loading report data:', error)
      // Fallback to empty data on error
      setReportData({
        summary: { totalForms: 0, totalResponses: 0, averageRating: 0, responseRate: 0 },
        chartData: { responsesByDate: [], ratingDistribution: [], trainerPerformance: [] },
        tableData: []
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleFilterChange = (newFilters) => {
    setFilters(prev => ({ ...prev, ...newFilters }))
  }

  const handleExportPDF = async () => {
    if (!generatedReport) {
      alert('Please generate a report first')
      return
    }

    try {
      const token = localStorage.getItem('iaa_token')
      const formId = generatedReport.form.id

      const response = await fetch(`${API_BASE_URL}/reports/export/${formId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.style.display = 'none'
        a.href = url
        a.download = `Report_${generatedReport.form.title.replace(/\s+/g, '_')}_${formId}.pdf`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)

        // Show success message
        alert('PDF report exported successfully!')
      } else {
        const errorData = await response.json()
        alert(`Error exporting PDF: ${errorData.detail}`)
      }
    } catch (error) {
      console.error('Error exporting PDF:', error)
      alert('Failed to export PDF report')
    }
  }

  const handleRefresh = () => {
    loadReportData()
  }

  const handleViewDetails = (formId) => {
    navigate(`/admin/form-details/${formId}`)
  }

  return (
    <div className={`reports-page ${theme}`}>
      <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
      
      <div className="reports-nav">
        <div className="nav-content">
          <button 
            className="back-button"
            onClick={() => navigate('/admin-dashboard')}
          >
            <ArrowLeft size={20} />
            Back to Dashboard
          </button>

          <div className="nav-center">
            <h1>Reports & Analytics</h1>
          </div>

          <div className="nav-actions">
            <button
              className="filter-toggle"
              onClick={() => setShowFilters(!showFilters)}
            >
              <Filter size={16} />
              Filters
            </button>
            <button
              className="refresh-btn"
              onClick={handleRefresh}
              disabled={isLoading}
            >
              <RefreshCw size={16} className={isLoading ? 'spinning' : ''} />
              Refresh
            </button>
            <button
              className="export-btn"
              onClick={() => handleExportPDF()}
              disabled={!generatedReport}
            >
              <Download size={16} />
              Export PDF
            </button>
          </div>
        </div>
      </div>

      {showFilters && (
        <motion.div
          className="filters-container"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
        >
          <ReportFilters
            filters={filters}
            onChange={handleFilterChange}
            onClose={() => setShowFilters(false)}
          />
        </motion.div>
      )}

      <main className="reports-main">
        <div className="reports-content">
          <FormSpecificReports theme={theme} />
        </div>
      </main>

      <Footer />
    </div>
  )
}

export default ReportsPage
