import { useState, useEffect, useMemo, useCallback } from 'react'
import { motion } from 'framer-motion'
import {
  Plus,
  BarChart3,
  AlertTriangle,
  Clock,
  Users,
  FileText,
  TrendingUp,
  TrendingDown,
  Calendar,
  Download,
  Bell,
  LogOut,
  Send,
  Eye,
  Trash2,
  XCircle,
  Edit,
  CheckCircle
} from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { useNotifications, NotificationBell } from '../components/NotificationSystem'
import { useNavigate } from 'react-router-dom'
import Header from '../components/Header'
import Footer from '../components/Footer'
import adminService from '../api/adminService'
import analysisService from '../services/analysisService'
import TrainerAnalyticsDashboard from '../components/TrainerAnalyticsDashboard'
import { API_BASE_URL } from '../api/config'
import { formatDate, formatDateTime, getRelativeTime } from '../utils/dateUtils'
import './TrainerDashboard.css'

// Comprehensive rating calculation function
const calculateComprehensiveRating = (responses, questions) => {
  if (!responses || responses.length === 0) return 'N/A'

  let totalScore = 0
  let totalWeight = 0
  const ratings = []

  responses.forEach(response => {
    const responseData = response.response_data || {}

    questions.forEach(question => {
      const answer = responseData[question.id]
      if (!answer) return

      let score = 0
      let weight = 1

      switch (question.type) {
        case 'rating':
        case 'scale':
          // Direct numerical rating (1-5 or 1-10)
          const numValue = parseFloat(answer)
          if (!isNaN(numValue)) {
            // Normalize to 5-point scale
            const maxScale = question.max || 5
            score = (numValue / maxScale) * 5
            weight = 2 // Higher weight for direct ratings
          }
          break

        case 'multiple-choice':
          // Convert multiple choice to rating based on position
          if (question.options && Array.isArray(question.options)) {
            const optionIndex = question.options.findIndex(opt => opt === answer)
            if (optionIndex !== -1) {
              // Assume options are ordered from worst to best
              score = ((optionIndex + 1) / question.options.length) * 5
              weight = 1.5
            }
          }
          break

        case 'text':
        case 'textarea':
          // Use NLP sentiment analysis for text responses
          const sentiment = analyzeSentiment(answer)
          score = sentiment.score
          weight = 1.2 // Moderate weight for sentiment
          break

        case 'checkbox':
          // For checkbox, count positive selections
          if (Array.isArray(answer)) {
            const positiveKeywords = ['excellent', 'good', 'satisfied', 'helpful', 'clear']
            const negativeKeywords = ['poor', 'bad', 'unclear', 'confusing', 'unhelpful']

            let positiveCount = 0
            let negativeCount = 0

            answer.forEach(item => {
              const itemLower = item.toLowerCase()
              if (positiveKeywords.some(keyword => itemLower.includes(keyword))) {
                positiveCount++
              }
              if (negativeKeywords.some(keyword => itemLower.includes(keyword))) {
                negativeCount++
              }
            })

            if (positiveCount + negativeCount > 0) {
              score = (positiveCount / (positiveCount + negativeCount)) * 5
              weight = 1
            }
          }
          break

        default:
          // For other types, try to extract numerical value or sentiment
          if (typeof answer === 'string') {
            const numMatch = answer.match(/\d+/)
            if (numMatch) {
              const num = parseInt(numMatch[0])
              if (num >= 1 && num <= 10) {
                score = (num / 10) * 5
                weight = 1
              }
            } else {
              // Fallback to sentiment analysis
              const sentiment = analyzeSentiment(answer)
              score = sentiment.score
              weight = 0.8
            }
          }
          break
      }

      if (score > 0 && weight > 0) {
        totalScore += score * weight
        totalWeight += weight
        ratings.push({ score, weight, type: question.type })
      }
    })
  })

  if (totalWeight === 0) return 'N/A'

  const weightedAverage = totalScore / totalWeight
  return Math.max(0, Math.min(5, weightedAverage)).toFixed(1)
}

// Simple sentiment analysis function
const analyzeSentiment = (text) => {
  if (!text || typeof text !== 'string') {
    return { score: 2.5, sentiment: 'neutral' }
  }

  const textLower = text.toLowerCase()

  // Positive keywords with weights
  const positiveWords = {
    'excellent': 3, 'outstanding': 3, 'amazing': 3, 'fantastic': 3,
    'great': 2, 'good': 2, 'wonderful': 2, 'helpful': 2, 'clear': 2,
    'useful': 1.5, 'effective': 1.5, 'informative': 1.5, 'engaging': 1.5,
    'professional': 1, 'thorough': 1, 'comprehensive': 1, 'organized': 1
  }

  // Negative keywords with weights
  const negativeWords = {
    'terrible': -3, 'awful': -3, 'horrible': -3, 'worst': -3,
    'bad': -2, 'poor': -2, 'disappointing': -2, 'useless': -2,
    'confusing': -1.5, 'unclear': -1.5, 'boring': -1.5, 'difficult': -1.5,
    'inadequate': -1, 'incomplete': -1, 'rushed': -1, 'disorganized': -1
  }

  let sentimentScore = 0
  let wordCount = 0

  // Calculate sentiment based on keyword presence
  Object.entries(positiveWords).forEach(([word, weight]) => {
    const matches = (textLower.match(new RegExp(word, 'g')) || []).length
    sentimentScore += matches * weight
    wordCount += matches
  })

  Object.entries(negativeWords).forEach(([word, weight]) => {
    const matches = (textLower.match(new RegExp(word, 'g')) || []).length
    sentimentScore += matches * weight
    wordCount += matches
  })

  // Normalize to 1-5 scale
  if (wordCount === 0) {
    return { score: 2.5, sentiment: 'neutral' }
  }

  // Convert sentiment score to 1-5 rating scale
  const normalizedScore = 2.5 + (sentimentScore / wordCount) * 1.5
  const finalScore = Math.max(1, Math.min(5, normalizedScore))

  return {
    score: finalScore,
    sentiment: finalScore > 3.5 ? 'positive' : finalScore < 2.5 ? 'negative' : 'neutral'
  }
}

// Calculate rating trends and breakdowns
const calculateRatingBreakdown = (responses, questions) => {
  if (!responses || responses.length === 0) {
    return {
      byQuestionType: {},
      trend: 'stable',
      recentAverage: 'N/A',
      overallAverage: 'N/A',
      improvementSuggestions: []
    }
  }

  const ratingsByType = {}
  const ratingsByDate = []
  let totalRatings = []

  responses.forEach(response => {
    const responseData = response.response_data || {}
    const responseDate = new Date(response.submitted_at || response.created_at)

    questions.forEach(question => {
      const answer = responseData[question.id]
      if (!answer) return

      const rating = extractRatingFromAnswer(answer, question)
      if (rating > 0) {
        // Group by question type
        if (!ratingsByType[question.type]) {
          ratingsByType[question.type] = []
        }
        ratingsByType[question.type].push(rating)

        // Track by date for trend analysis
        ratingsByDate.push({ date: responseDate, rating })
        totalRatings.push(rating)
      }
    })
  })

  // Calculate averages by question type
  const averagesByType = {}
  Object.entries(ratingsByType).forEach(([type, ratings]) => {
    averagesByType[type] = {
      average: (ratings.reduce((sum, r) => sum + r, 0) / ratings.length).toFixed(1),
      count: ratings.length,
      distribution: calculateDistribution(ratings)
    }
  })

  // Calculate trend (last 5 vs previous 5 responses)
  const sortedByDate = ratingsByDate.sort((a, b) => a.date - b.date)
  let trend = 'stable'

  if (sortedByDate.length >= 10) {
    const recent5 = sortedByDate.slice(-5).map(r => r.rating)
    const previous5 = sortedByDate.slice(-10, -5).map(r => r.rating)

    const recentAvg = recent5.reduce((sum, r) => sum + r, 0) / recent5.length
    const previousAvg = previous5.reduce((sum, r) => sum + r, 0) / previous5.length

    if (recentAvg > previousAvg + 0.3) trend = 'improving'
    else if (recentAvg < previousAvg - 0.3) trend = 'declining'
  }

  // Generate improvement suggestions
  const suggestions = generateImprovementSuggestions(averagesByType, totalRatings)

  return {
    byQuestionType: averagesByType,
    trend,
    recentAverage: sortedByDate.length >= 5
      ? (sortedByDate.slice(-5).reduce((sum, r) => sum + r.rating, 0) / 5).toFixed(1)
      : 'N/A',
    overallAverage: totalRatings.length > 0
      ? (totalRatings.reduce((sum, r) => sum + r, 0) / totalRatings.length).toFixed(1)
      : 'N/A',
    improvementSuggestions: suggestions
  }
}

// Helper function to extract rating from any answer type
const extractRatingFromAnswer = (answer, question) => {
  switch (question.type) {
    case 'rating':
    case 'scale':
      const numValue = parseFloat(answer)
      if (!isNaN(numValue)) {
        const maxScale = question.max || 5
        return (numValue / maxScale) * 5
      }
      break
    case 'multiple-choice':
      if (question.options && Array.isArray(question.options)) {
        const optionIndex = question.options.findIndex(opt => opt === answer)
        if (optionIndex !== -1) {
          return ((optionIndex + 1) / question.options.length) * 5
        }
      }
      break
    case 'text':
    case 'textarea':
      return analyzeSentiment(answer).score
    default:
      if (typeof answer === 'string') {
        const numMatch = answer.match(/\d+/)
        if (numMatch) {
          const num = parseInt(numMatch[0])
          if (num >= 1 && num <= 10) {
            return (num / 10) * 5
          }
        }
        return analyzeSentiment(answer).score
      }
  }
  return 0
}

// Calculate rating distribution
const calculateDistribution = (ratings) => {
  const distribution = { 1: 0, 2: 0, 3: 0, 4: 0, 5: 0 }
  ratings.forEach(rating => {
    const rounded = Math.round(rating)
    if (rounded >= 1 && rounded <= 5) {
      distribution[rounded]++
    }
  })
  return distribution
}

// Generate improvement suggestions based on ratings
const generateImprovementSuggestions = (averagesByType, totalRatings) => {
  const suggestions = []
  const overallAvg = totalRatings.length > 0
    ? totalRatings.reduce((sum, r) => sum + r, 0) / totalRatings.length
    : 0

  if (overallAvg < 3.0) {
    suggestions.push("Overall ratings are below average. Consider reviewing training content and delivery methods.")
  }

  Object.entries(averagesByType).forEach(([type, data]) => {
    const avg = parseFloat(data.average)
    if (avg < 2.5) {
      suggestions.push(`${type} questions show low satisfaction. Focus on improving this aspect of training.`)
    } else if (avg > 4.5) {
      suggestions.push(`${type} questions show excellent results. Consider using this approach in other areas.`)
    }
  })

  if (suggestions.length === 0) {
    suggestions.push("Ratings are performing well. Continue current training methods.")
  }

  return suggestions
}

// Calculate overall average rating from forms
const calculateOverallAverageRating = (forms) => {
  if (!forms || forms.length === 0) return '0.0'

  const validRatings = forms
    .map(form => {
      if (typeof form.avgRating === 'number') return form.avgRating
      if (typeof form.avgRating === 'string' && !isNaN(parseFloat(form.avgRating))) {
        return parseFloat(form.avgRating)
      }
      return null
    })
    .filter(rating => rating !== null && rating > 0)

  if (validRatings.length === 0) return '0.0'

  const average = validRatings.reduce((sum, rating) => sum + rating, 0) / validRatings.length
  return average.toFixed(1)
}

const TrainerDashboard = ({ theme, toggleTheme }) => {
  const { user, logout } = useAuth()
  const { showThresholdAlert, showSuccess, showFormRejected } = useNotifications()
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('overview')

  const [formRequests, setFormRequests] = useState([])
  const [pendingRequestsCount, setPendingRequestsCount] = useState(0)
  const [liveResponses, setLiveResponses] = useState([])
  const [alerts, setAlerts] = useState([])
  const [reports, setReports] = useState([])
  const [isLoading, setIsLoading] = useState(true)

  const [departmentName, setDepartmentName] = useState('')
  const [loadingDepartment, setLoadingDepartment] = useState(true)
  const [trainees, setTrainees] = useState([])
  const [forms, setForms] = useState([])
  const [reportNotifications, setReportNotifications] = useState([])
  const [dashboardStats, setDashboardStats] = useState(null)
  const [analyticsData, setAnalyticsData] = useState(null)
  const [reportSettings, setReportSettings] = useState({ auto_generation_delay_days: { value: '1' } })
  const [rejectedRequests, setRejectedRequests] = useState([])

  // Real data will be fetched from backend

  // Calculate report generation date using configurable delay
  const calculateReportDate = (dueDate) => {
    if (!dueDate) return null
    const due = new Date(dueDate)
    const delayDays = parseInt(reportSettings.auto_generation_delay_days?.value || '1')
    const reportDate = new Date(due.getTime() + (delayDays * 24 * 60 * 60 * 1000))
    return reportDate
  }

  // Generate report notifications for forms
  const generateReportNotifications = (formsData) => {
    const notifications = []
    const now = new Date()

    formsData.forEach(form => {
      if ((form.status === 'published' || form.is_published) && form.due_date) {
        const dueDate = new Date(form.due_date)
        const reportDate = calculateReportDate(form.due_date)

        if (reportDate) {
          let status = 'collecting'
          let message = `Report will automatically generate on ${reportDate.toLocaleDateString()} at ${reportDate.toLocaleTimeString()}`

          if (now > dueDate) {
            if (now > reportDate) {
              status = 'ready'
              message = 'Report is ready for download'
            } else {
              status = 'generating'
              message = `Report generating... Will be ready on ${reportDate.toLocaleDateString()}`
            }
          }

          notifications.push({
            formId: form.id,
            formTitle: form.title,
            dueDate: form.due_date,
            reportDate: reportDate.toISOString(),
            responseCount: form.response_count || 0,
            status: status,
            message: message
          })
        }
      }
    })
    setReportNotifications(notifications)
  }

  // Fetch department name
  const fetchDepartmentName = async () => {
    if (!user?.department_id) {
      setLoadingDepartment(false)
      return
    }

    try {
      setLoadingDepartment(true)
      const departments = await adminService.getDepartments()
      const department = departments.find(dept => dept.id === parseInt(user.department_id))
      setDepartmentName(department ? department.name : `Department ${user.department_id}`)
    } catch (error) {
      setDepartmentName(`Department ${user.department_id}`)
    } finally {
      setLoadingDepartment(false)
    }
  }

  const fetchPendingRequestsCount = async () => {
    try {
      const response = await adminService.getFormRequests()
      if (response && response.success && Array.isArray(response.requests)) {
        const trainerRequests = response.requests.filter(req => req && req.trainer_id === user.id)

        // Count pending requests for this trainer
        const pendingCount = trainerRequests.filter(
          req => req.status === 'pending'
        ).length
        setPendingRequestsCount(pendingCount)
        // Store in localStorage for real-time updates
        localStorage.setItem('pending_requests_count', pendingCount.toString())

        // Get all rejected requests for display
        const allRejectedRequests = trainerRequests.filter(req => req.status === 'rejected')
        setRejectedRequests(allRejectedRequests)

        // Check for newly rejected requests and show notifications
        const newlyRejectedRequests = allRejectedRequests.filter(req =>
          !localStorage.getItem(`rejection_notified_${req.id}`)
        )

        // Show rejection notifications
        newlyRejectedRequests.forEach(request => {
          showFormRejected(
            request.title || 'Form Request',
            request.rejection_reason || 'No reason provided',
            {
              formId: request.id,
              persistent: true,
              priority: 'high'
            }
          )

          // Mark as notified to prevent duplicate notifications
          localStorage.setItem(`rejection_notified_${request.id}`, 'true')
        })
      } else {
        setPendingRequestsCount(0)
      }
    } catch (error) {
      // Don't show error to user for this non-critical feature
      setPendingRequestsCount(0)
    }
  }

  // Listen for localStorage changes to update counter in real-time
  useEffect(() => {
    const handleStorageChange = () => {
      const count = parseInt(localStorage.getItem('pending_requests_count') || '0')
      setPendingRequestsCount(count)
    }

    window.addEventListener('storage', handleStorageChange)

    // Also check on focus (when returning to the tab)
    window.addEventListener('focus', fetchPendingRequestsCount)

    return () => {
      window.removeEventListener('storage', handleStorageChange)
      window.removeEventListener('focus', fetchPendingRequestsCount)
    }
  }, [user])

  // Use default report settings for trainers (no admin access needed)
  const getDefaultReportSettings = () => {
    return { auto_generation_delay_days: { value: '1' } }
  }

  // Fetch real data
  // Function to load detailed ratings for a specific form (lazy loading)
  const loadFormRatings = useCallback(async (formId) => {
    try {
      const responsesData = await adminService.getFormResponses(formId)
      const responses = responsesData.responses || []
      const form = forms.find(f => f.id === formId)

      if (form) {
        const avgRating = calculateComprehensiveRating(responses, form.formData?.questions || [])
        const ratingBreakdown = calculateRatingBreakdown(responses, form.formData?.questions || [])

        // Update the specific form in state
        setForms(prevForms =>
          prevForms.map(f =>
            f.id === formId
              ? { ...f, avgRating, ratingBreakdown, responses, ratingsLoaded: true }
              : f
          )
        )
      }
    } catch (error) {
      // Error loading ratings - continue with default values
    }
  }, [])

  const fetchTrainerData = useCallback(async () => {
    try {
      setIsLoading(true)

      // Check cache first for faster loading
      const cacheKey = `trainer_data_${user.id}`
      const cached = localStorage.getItem(cacheKey)
      if (cached) {
        const { data, timestamp } = JSON.parse(cached)
        const cacheAge = Date.now() - timestamp
        // Use cache if less than 2 minutes old
        if (cacheAge < 2 * 60 * 1000) {
          setLiveResponses(data.liveResponses || [])
          setAlerts(data.alerts || [])
          setReports(data.reports || [])
          setDashboardStats(data.stats || null)
          setIsLoading(false)
          return
        }
      }

      // Fetch only essential data initially
      const formsData = await adminService.getForms().catch(() => [])

      const trainerForms = formsData ? formsData.filter(form => {
        if (!form || !form.id) {
          return false
        }
        return form.trainer_id === user.id || form.creator_id === user.id
      }) : []
      setForms(trainerForms)

      // Set basic dashboard stats (load detailed stats later)
      setDashboardStats({
        totalForms: trainerForms.length,
        activeForms: trainerForms.filter(f => f.is_published).length,
        totalResponses: trainerForms.reduce((sum, f) => sum + (f.response_count || 0), 0)
      })

      // Generate report notifications
      generateReportNotifications(trainerForms)

      // Generate enhanced alerts based on form performance
      const enhancedAlerts = generateEnhancedAlerts(trainerForms)
      setAlerts(enhancedAlerts)

      // Transform forms data to live responses format with basic data (optimize for speed)
      const liveResponsesData = trainerForms.map((form) => {
        try {
          if (!form || !form.id) {
            return null
          }

          // Use basic rating calculation for initial load (no API calls)
          const avgRating = form.response_count > 0 ? 'Loading...' : 'N/A'

          // Basic rating breakdown for initial display
          const ratingBreakdown = {
            trend: 'stable',
            recentAverage: 'N/A',
            overallAverage: 'N/A'
          }

          return {
            id: form.id,
            title: form.title || 'Untitled Form',
            totalResponses: form.response_count || 0,
            targetResponses: null, // No limit - unlimited responses
            completionRate: 100, // Always 100% since there's no limit
            avgRating: avgRating,
            lastResponse: form.updated_at ? new Date(form.updated_at).toLocaleString() : 'No responses yet',
            status: form.is_published ? 'active' : 'draft',
            dueDate: form.due_date || 'No due date',
            responses: [], // Will be loaded on demand
            formData: form.form_data || {},
            createdAt: form.created_at,
            departmentName: form.department_name || 'Unknown Department',
            ratingBreakdown: ratingBreakdown
          }
        } catch (error) {
          return {
            id: form.id,
            title: form.title || 'Untitled Form',
            totalResponses: form.response_count || 0,
            targetResponses: null, // No limit - unlimited responses
            completionRate: 100, // Always 100% since there's no limit
            avgRating: 'N/A',
            lastResponse: 'Error loading data',
            status: form.is_published ? 'active' : 'draft',
            dueDate: form.due_date || 'No due date',
            responses: [],
            formData: {},
            createdAt: form.created_at,
            departmentName: 'Unknown Department',
            ratingBreakdown: { trend: 'stable', recentAverage: 'N/A', overallAverage: 'N/A' }
          }
        }
      })

      // Filter out null values from invalid forms
      const validLiveResponsesData = liveResponsesData.filter(data => data !== null)

      setLiveResponses(validLiveResponsesData)

      // Generate alerts based on real data
      const alertsData = trainerForms.filter(form => {
        const responseCount = form.response_count || 0
        return responseCount < 5 // Alert if less than 5 responses
      }).map(form => ({
        id: `alert_${form.id}`,
        type: 'low_response',
        severity: 'medium',
        message: `Form "${form.title}" has low response count (${form.response_count || 0} responses)`,
        formId: form.id,
        timestamp: new Date().toISOString(),
        acknowledged: false
      }))

      setAlerts(alertsData)

      // Fetch real reports from backend
      try {
        const token = localStorage.getItem('iaa_token')
        const reportsResponse = await fetch(`${API_BASE_URL}/reports/forms`, {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (reportsResponse.ok) {
          const reportsData = await reportsResponse.json()
          const formattedReports = reportsData.forms.map(form => {
            const now = new Date()
            const dueDate = form.due_date ? new Date(form.due_date) : null
            const reportDate = calculateReportDate(form.due_date)

            let status = 'collecting' // Default status
            let statusText = 'Collecting Responses'

            if (dueDate && now > dueDate) {
              if (reportDate && now > reportDate) {
                // Past report generation date
                status = (form.response_count || 0) > 0 ? 'ready' : 'pending'
                statusText = status === 'ready' ? 'Report Ready' : 'Report Generating'
              } else {
                // Between due date and report generation date
                status = 'generating'
                statusText = 'Report Generating'
              }
            }

            return {
              id: `report_${form.id}`,
              formId: form.id,
              title: form.title,
              period: `${form.created_at ? new Date(form.created_at).toLocaleDateString() : 'Unknown'} to Present`,
              status: status,
              statusText: statusText,
              reportDate: reportDate,
              generatedDate: form.updated_at ? new Date(form.updated_at).toLocaleDateString() : 'Not generated',
              autoGenerated: true,
              downloadUrl: `/reports/${form.id}`
            }
          })
          setReports(formattedReports)
        } else {
          setReports([])
        }
      } catch (error) {
        setReports([])
      }

      // Cache the data for faster subsequent loads
      const cacheData = {
        liveResponses: validLiveResponsesData,
        alerts: alertsData,
        reports: [],
        stats: dashboardStats,
        timestamp: Date.now()
      }
      localStorage.setItem(`trainer_data_${user.id}`, JSON.stringify({ data: cacheData, timestamp: Date.now() }))

    } catch (error) {
      // Fallback to empty arrays
      setLiveResponses([])
      setAlerts([])
      setReports([])
    } finally {
      setIsLoading(false)
    }
  }, [user.id])

  const fetchRejectedRequests = async () => {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/form-requests/rejected/${user.id}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        const data = await response.json()
        setRejectedRequests(data.requests || [])

        // Show notifications for newly rejected requests
        data.requests?.forEach(request => {
          const notificationKey = `rejection_notified_${request.id}`
          if (!localStorage.getItem(notificationKey)) {
            showFormRejected(request.title, request.admin_response || 'No specific reason provided')
            localStorage.setItem(notificationKey, 'true')
          }
        })
      } else {
        // Failed to fetch rejected requests
      }
    } catch (error) {
      // Error fetching rejected requests
    }
  }

  const fetchAllFormRequests = async () => {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/form-requests/trainer/${user.id}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        const data = await response.json()
        setFormRequests(data.requests || [])
      } else {
        setFormRequests([])
      }
    } catch (error) {
      // Fallback to empty array
      setFormRequests([])
    }
  }

  useEffect(() => {
    if (!user || user.role !== 'trainer') {
      navigate('/login')
      return
    }

    // Fetch essential data first for faster initial load
    fetchDepartmentName()
    fetchTrainerData()
    fetchPendingRequestsCount()

    // Load secondary data after a short delay to improve perceived performance
    setTimeout(() => {
      fetchRejectedRequests()
      fetchAllFormRequests()
    }, 100)

    // Cleanup function
    return () => {
      // Cleanup any pending operations
    }
  }, [user, navigate, fetchTrainerData])

  // Memoized forms with responses for performance
  const formsWithResponses = useMemo(() => {
    return forms.filter(f => f.totalResponses > 0 && !f.ratingsLoaded)
  }, [forms])

  // Load detailed ratings in background after initial load
  useEffect(() => {
    if (formsWithResponses.length > 0 && !isLoading) {
      // Load ratings for forms with responses, but do it gradually to avoid blocking
      formsWithResponses.forEach((form, index) => {
        // Stagger the requests to avoid overwhelming the server
        setTimeout(() => {
          loadFormRatings(form.id)
        }, index * 500) // 500ms delay between each request
      })
    }
  }, [formsWithResponses, isLoading, loadFormRatings])

  const acknowledgeAlert = (alertId) => {
    setAlerts(prev => prev.map(alert =>
      alert.id === alertId ? { ...alert, acknowledged: true } : alert
    ))
  }

  const handleReEditForm = (rejectedRequest) => {
    // Navigate to form builder with the rejected request data for re-editing
    navigate('/trainer/form-builder', {
      state: {
        rejectedRequest: rejectedRequest,
        isReEdit: true
      }
    })
  }

  const handleDeleteRejectedRequest = async (requestId) => {
    if (window.confirm('Are you sure you want to delete this rejected form request? This action cannot be undone.')) {
      try {
        // Delete from backend
        await adminService.deleteFormRequest(requestId)

        // Remove from local state
        setRejectedRequests(prev => prev.filter(req => req.id !== requestId))

        // Remove notification flag
        localStorage.removeItem(`rejection_notified_${requestId}`)

        showSuccess('Rejected form request deleted successfully')
      } catch (error) {
        alert(`Failed to delete rejected request: ${error.message}`)
      }
    }
  }



  // Generate enhanced alerts based on form performance
  const generateEnhancedAlerts = (formsData) => {
    const alerts = []
    const now = new Date()

    formsData.forEach(form => {
      const responseCount = form.response_count || 0
      const avgRating = form.average_rating || 0
      const dueDate = form.due_date ? new Date(form.due_date) : null

      // Low response count alert
      if (responseCount < 5 && dueDate && now > dueDate) {
        alerts.push({
          id: `low_response_${form.id}`,
          type: 'warning',
          title: 'Low Response Count',
          message: `"${form.title}" has only ${responseCount} response${responseCount !== 1 ? 's' : ''}`,
          formId: form.id,
          severity: 'medium',
          actionRequired: true
        })
      }

      // Falling rating alert
      if (avgRating > 0 && avgRating < 3.5) {
        alerts.push({
          id: `low_rating_${form.id}`,
          type: 'error',
          title: 'Low Average Rating',
          message: `"${form.title}" has an average rating of ${avgRating.toFixed(1)}/5.0`,
          formId: form.id,
          severity: 'high',
          actionRequired: true
        })
      }

      // Due date approaching alert
      if (dueDate) {
        const daysUntilDue = Math.ceil((dueDate.getTime() - now.getTime()) / (1000 * 60 * 60 * 24))
        if (daysUntilDue <= 2 && daysUntilDue > 0 && responseCount < 10) {
          alerts.push({
            id: `due_soon_${form.id}`,
            type: 'warning',
            title: 'Form Due Soon',
            message: `"${form.title}" is due in ${daysUntilDue} day${daysUntilDue !== 1 ? 's' : ''} with only ${responseCount} response${responseCount !== 1 ? 's' : ''}`,
            formId: form.id,
            severity: 'medium',
            actionRequired: false
          })
        }
      }

      // High engagement alert (positive)
      if (responseCount >= 20 && avgRating >= 4.5) {
        alerts.push({
          id: `high_engagement_${form.id}`,
          type: 'success',
          title: 'Excellent Performance',
          message: `"${form.title}" has ${responseCount} responses with ${avgRating.toFixed(1)}/5.0 rating`,
          formId: form.id,
          severity: 'low',
          actionRequired: false
        })
      }

      // No responses alert for published forms
      if (responseCount === 0 && form.is_published && dueDate) {
        const daysSincePublished = Math.ceil((now.getTime() - new Date(form.created_at).getTime()) / (1000 * 60 * 60 * 24))
        if (daysSincePublished >= 3) {
          alerts.push({
            id: `no_responses_${form.id}`,
            type: 'warning',
            title: 'No Responses Yet',
            message: `"${form.title}" has been published for ${daysSincePublished} days with no responses`,
            formId: form.id,
            severity: 'medium',
            actionRequired: true
          })
        }
      }
    })

    return alerts
  }

  const generateReport = (reportId) => {
    setReports(prev => prev.map(report =>
      report.id === reportId ? {
        ...report,
        status: 'ready',
        generatedDate: new Date().toISOString().split('T')[0],
        downloadUrl: '#'
      } : report
    ))

    alert('Report generated successfully!')
  }

  const handleRequestFormDeletion = async (formId, formTitle) => {
    const reason = prompt(`Please provide a reason for requesting deletion of "${formTitle}":\n\nExamples:\n- Form is no longer needed\n- Duplicate form created\n- Content needs major revision\n- Training session cancelled`)
    if (reason === null) {
      return // User cancelled
    }

    // Make reason optional - if empty, provide a default
    const finalReason = reason.trim() || 'No specific reason provided'

    try {
      await adminService.createFormDeletionRequest(formId, finalReason)
      showSuccess('Form deletion request submitted successfully! An admin will review your request.')
      // Refresh the forms data
      fetchTrainerData()
    } catch (error) {
      showThresholdAlert('Failed to submit deletion request. Please try again.')
    }
  }

  const handleDownloadReport = async (formId) => {
    try {
      const token = localStorage.getItem('iaa_token')

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
        a.download = `Report_Form_${formId}.pdf`
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)

        showSuccess('Report downloaded successfully!')
      } else {
        const errorData = await response.json()
        alert(`Error downloading report: ${errorData.detail}`)
      }
    } catch (error) {
      alert('Failed to download report')
    }
  }

  const handleGenerateReport = async (formId) => {
    try {
      // Update report status to generating
      setReports(prev => prev.map(report =>
        report.formId === formId ? { ...report, status: 'generating' } : report
      ))

      const token = localStorage.getItem('iaa_token')

      const response = await fetch(`${API_BASE_URL}/reports/generate/${formId}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const reportData = await response.json()

        // Update report status to ready
        setReports(prev => prev.map(report =>
          report.formId === formId ? {
            ...report,
            status: 'ready',
            generatedDate: new Date().toLocaleDateString()
          } : report
        ))

        showSuccess('Report generated successfully!')
      } else {
        const errorData = await response.json()

        // Reset status on error
        setReports(prev => prev.map(report =>
          report.formId === formId ? { ...report, status: 'pending' } : report
        ))

        alert(`Error generating report: ${errorData.detail}`)
      }
    } catch (error) {
      // Reset status on error
      setReports(prev => prev.map(report =>
        report.formId === formId ? { ...report, status: 'pending' } : report
      ))

      alert('Failed to generate report')
    }
  }

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  if (isLoading) {
    return (
      <div className={`trainer-dashboard ${theme}`}>
        <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading your dashboard...</p>
        </div>
        <Footer />
      </div>
    )
  }

  // Safety check for user object
  if (!user) {
    return (
      <div className={`trainer-dashboard ${theme}`}>
        <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>Loading user data...</p>
        </div>
      </div>
    )
  }

  return (
    <div className={`trainer-dashboard ${theme}`}>
      <Header theme={theme} toggleTheme={toggleTheme} hideAuth={true} />
      
      <main className="dashboard-main">
        {/* Header Section */}
        <div className="dashboard-header">
          <div className="header-content">
            <div className="user-info">
              <img
                src={user.avatar || '/default-avatar.svg'}
                alt="Avatar"
                className="user-avatar"
                onError={(e) => {
                  e.target.src = '/default-avatar.svg'
                }}
              />
              <div className="user-details">
                <h1>Welcome, {user.firstName || user.first_name || user.name}</h1>
                <p className="user-role">
                  <Users size={16} />
                  {loadingDepartment ? (
                    'Loading department...'
                  ) : (
                    `Trainer - ${departmentName}`
                  )}
                </p>
              </div>
            </div>
            <div className="header-actions">
              <NotificationBell className="notification-bell-header" />

              {/* Pending Requests Counter */}
              {pendingRequestsCount > 0 && (
                <div className="pending-requests-counter">
                  <Clock size={16} />
                  <span>{pendingRequestsCount} pending request{pendingRequestsCount !== 1 ? 's' : ''}</span>
                </div>
              )}

              <button
                className="request-form-btn"
                onClick={() => navigate('/trainer/form-builder')}
              >
                <Plus size={16} />
                Create Form Request
              </button>
              <button className="logout-btn" onClick={handleLogout}>
                <LogOut size={16} />
                Logout
              </button>
            </div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <div className="dashboard-nav">
          <div className="nav-content">
            <div className="nav-tabs">
              <button 
                className={`tab-btn ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                <BarChart3 size={16} />
                Overview
              </button>
              <button 
                className={`tab-btn ${activeTab === 'responses' ? 'active' : ''}`}
                onClick={() => setActiveTab('responses')}
              >
                <Users size={16} />
                Live Responses
              </button>
              <button 
                className={`tab-btn ${activeTab === 'alerts' ? 'active' : ''}`}
                onClick={() => setActiveTab('alerts')}
              >
                <Bell size={16} />
                Alerts
                {alerts.filter(a => !a.acknowledged).length > 0 && (
                  <span className="alert-badge">
                    {alerts.filter(a => !a.acknowledged).length}
                  </span>
                )}
              </button>
              <button
                className={`tab-btn ${activeTab === 'forms' ? 'active' : ''}`}
                onClick={() => setActiveTab('forms')}
              >
                <FileText size={16} />
                My Forms
              </button>
              <button
                className={`tab-btn ${activeTab === 'requests' ? 'active' : ''}`}
                onClick={() => setActiveTab('requests')}
              >
                <Clock size={16} />
                My Requests
                {formRequests.filter(req => req.status === 'pending').length > 0 && (
                  <span className="request-badge">
                    {formRequests.filter(req => req.status === 'pending').length}
                  </span>
                )}
              </button>

              <button
                className={`tab-btn ${activeTab === 'reports' ? 'active' : ''}`}
                onClick={() => setActiveTab('reports')}
              >
                <FileText size={16} />
                Reports
              </button>
              <button
                className={`tab-btn ${activeTab === 'rejected' ? 'active' : ''}`}
                onClick={() => setActiveTab('rejected')}
              >
                <XCircle size={16} />
                Rejected Forms
                {rejectedRequests.length > 0 && (
                  <span className="rejection-badge">
                    {rejectedRequests.length}
                  </span>
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="dashboard-content">
          {activeTab === 'overview' && (
            <div className="overview-content">
              {/* Stats Cards */}
              <div className="stats-grid">
                <motion.div
                  className="stat-card clickable"
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setActiveTab('forms')}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="stat-icon my-forms">
                    <FileText size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{dashboardStats?.myForms || forms.length}</h3>
                    <p>My Forms</p>
                  </div>
                </motion.div>

                <motion.div
                  className="stat-card clickable"
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setActiveTab('forms')}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="stat-icon active-forms">
                    <FileText size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{dashboardStats?.activeForms !== undefined ? dashboardStats.activeForms : liveResponses.filter(f => f.status === 'active' || f.status === 'published').length}</h3>
                    <p>Active Forms</p>
                  </div>
                </motion.div>

                <motion.div
                  className="stat-card clickable"
                  whileHover={{ scale: 1.02 }}
                  onClick={() => navigate('/trainer/responses')}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="stat-icon total-responses">
                    <Users size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{dashboardStats?.totalResponses !== undefined ? dashboardStats.totalResponses : liveResponses.reduce((sum, form) => sum + (form.totalResponses || 0), 0)}</h3>
                    <p>Total Responses</p>
                  </div>
                </motion.div>

                <motion.div
                  className="stat-card clickable"
                  whileHover={{ scale: 1.02 }}
                  onClick={() => navigate('/trainer/analysis')}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="stat-icon avg-rating">
                    <TrendingUp size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{dashboardStats?.averageRating !== undefined ? dashboardStats.averageRating : calculateOverallAverageRating(liveResponses)}</h3>
                    <p>Avg Rating</p>
                  </div>
                </motion.div>

                <motion.div
                  className="stat-card clickable"
                  whileHover={{ scale: 1.02 }}
                  onClick={() => setActiveTab('alerts')}
                  style={{ cursor: 'pointer' }}
                >
                  <div className="stat-icon pending-alerts">
                    <AlertTriangle size={24} />
                  </div>
                  <div className="stat-content">
                    <h3>{alerts.filter(a => !a.acknowledged).length}</h3>
                    <p>Pending Alerts</p>
                  </div>
                </motion.div>
              </div>

              {/* Quick Actions */}
              <div className="quick-actions">
                <h2>Quick Actions</h2>
                <div className="actions-grid">
                  <motion.button
                    className="action-card"
                    whileHover={{ scale: 1.02 }}
                    onClick={() => navigate('/trainer/form-builder')}
                  >
                    <Plus size={32} />
                    <h3>Create Form Request</h3>
                    <p>Design your feedback form and send it for admin approval</p>
                  </motion.button>

                  <motion.button 
                    className="action-card"
                    whileHover={{ scale: 1.02 }}
                    onClick={() => setActiveTab('responses')}
                  >
                    <Eye size={32} />
                    <h3>View Live Responses</h3>
                    <p>Monitor real-time form submissions</p>
                  </motion.button>

                  <motion.button 
                    className="action-card"
                    whileHover={{ scale: 1.02 }}
                    onClick={() => setActiveTab('reports')}
                  >
                    <Download size={32} />
                    <h3>Generate Reports</h3>
                    <p>Create detailed analytics reports</p>
                  </motion.button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'responses' && (
            <div className="responses-content">
              <div className="section-header">
                <h2>Live Response Monitoring</h2>
                <p>Real-time tracking of form submissions and completion rates</p>
              </div>

              {liveResponses.length === 0 ? (
                <div className="empty-state">
                  <FileText size={48} />
                  <h3>No active forms</h3>
                  <p>You don't have any active forms at the moment.</p>
                </div>
              ) : (
                <div className="responses-grid">
                  {liveResponses.map((form) => (
                    <motion.div
                      key={form.id}
                      className="response-card"
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="response-header">
                        <h3>{form.title}</h3>
                        <div className={`status-badge ${form.status}`}>
                          {form.status}
                        </div>
                      </div>

                      <div className="response-stats">
                        <div className="stat-row">
                          <span>Responses:</span>
                          <span className="stat-value">
                            {form.totalResponses}
                          </span>
                        </div>
                        <div className="stat-row">
                          <span>Status:</span>
                          <span className="stat-value">{form.status === 'active' ? 'Active' : 'Draft'}</span>
                        </div>
                        <div className="stat-row">
                          <span>Average Rating:</span>
                          <span className="stat-value rating">
                            {form.avgRating}/5.0
                            {form.ratingBreakdown && (
                              <span className={`trend-indicator ${form.ratingBreakdown.trend}`}>
                                {form.ratingBreakdown.trend === 'improving' && '↗️'}
                                {form.ratingBreakdown.trend === 'declining' && '↘️'}
                                {form.ratingBreakdown.trend === 'stable' && '→'}
                              </span>
                            )}
                          </span>
                        </div>
                        {form.ratingBreakdown && form.ratingBreakdown.recentAverage !== 'N/A' && (
                          <div className="stat-row">
                            <span>Recent Trend:</span>
                            <span className="stat-value">
                              {form.ratingBreakdown.recentAverage}/5.0 (last 5)
                            </span>
                          </div>
                        )}
                        <div className="stat-row">
                          <span>Last Response:</span>
                          <span className="stat-value">{form.lastResponse}</span>
                        </div>
                      </div>

                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{
                            width: form.totalResponses > 0 ? '100%' : '0%',
                            backgroundColor: form.status === 'active' ? '#4CAF50' : '#9E9E9E'
                          }}
                        ></div>
                      </div>

                      <div className="response-actions">
                        <button
                          className="view-details-btn"
                          onClick={() => navigate(`/form-responses/${form.id}`)}
                        >
                          <Eye size={16} />
                          View Responses
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'forms' && (
            <div className="forms-content">
              <div className="section-header">
                <h2>My Forms</h2>
                <p>Manage your created forms and view their status</p>
              </div>

              {forms.length === 0 ? (
                <div className="empty-state">
                  <FileText size={48} />
                  <h3>No forms created</h3>
                  <p>You haven't requested any forms yet. Click "Request New Form" to get started.</p>
                </div>
              ) : (
                <div className="forms-grid">
                  {forms.map((form) => (
                    <motion.div
                      key={form.id}
                      className="form-card"
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="form-header">
                        <h3>{form.title}</h3>
                        <div className={`status-badge ${form.is_published ? 'published' : 'draft'}`}>
                          {form.is_published ? 'PUBLISHED' : 'DRAFT'}
                        </div>
                      </div>

                      <p className="form-description">{form.description}</p>

                      <div className="form-meta">
                        <div className="meta-item">
                          <Calendar size={16} />
                          <span>Created: {form.created_at ? new Date(form.created_at).toLocaleDateString() : 'Unknown'}</span>
                        </div>
                        {form.is_published && (
                          <div className="meta-item">
                            <Clock size={16} />
                            <span>Published: {form.updated_at ? new Date(form.updated_at).toLocaleDateString() : 'Unknown'}</span>
                          </div>
                        )}
                        <div className="meta-item">
                          <Users size={16} />
                          <span>{form.response_count || 0} responses</span>
                        </div>
                      </div>

                      <div className="form-actions">
                        <button
                          className="preview-btn"
                          onClick={() => navigate(`/form/${form.id}?preview=true`)}
                        >
                          <Eye size={16} />
                          Preview
                        </button>
                        <button
                          className="responses-btn"
                          onClick={() => navigate(`/form-responses/${form.id}`)}
                        >
                          <BarChart3 size={16} />
                          View Responses
                        </button>
                        <button
                          className="delete-btn"
                          onClick={() => handleRequestFormDeletion(form.id, form.title)}
                        >
                          <Trash2 size={16} />
                          Request Deletion
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'alerts' && (
            <div className="alerts-content">
              <div className="section-header">
                <h2>Alerts & Notifications</h2>
                <p>Threshold alerts and important notifications</p>
              </div>

              {alerts.length === 0 ? (
                <div className="empty-state">
                  <Bell size={48} />
                  <h3>No alerts</h3>
                  <p>All systems are running smoothly.</p>
                </div>
              ) : (
                <div className="alerts-list">
                  {alerts.map((alert) => (
                    <motion.div
                      key={alert.id}
                      className={`alert-card ${alert.severity} ${alert.acknowledged ? 'acknowledged' : ''}`}
                      whileHover={{ scale: 1.01 }}
                    >
                      <div className="alert-icon">
                        {alert.type === 'error' && <AlertTriangle size={24} />}
                        {alert.type === 'warning' && <AlertCircle size={24} />}
                        {alert.type === 'success' && <CheckCircle size={24} />}
                        {alert.type === 'threshold' && <TrendingDown size={24} />}
                        {!['error', 'warning', 'success', 'threshold'].includes(alert.type) && <Clock size={24} />}
                      </div>

                      <div className="alert-content">
                        <h4>{alert.title}</h4>
                        <p>{alert.message}</p>
                        {alert.formId && (
                          <small className="form-link">
                            Form ID: {alert.formId}
                          </small>
                        )}
                      </div>

                      {!alert.acknowledged && (
                        <button 
                          className="acknowledge-btn"
                          onClick={() => acknowledgeAlert(alert.id)}
                        >
                          Acknowledge
                        </button>
                      )}
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          )}



          {activeTab === 'reports' && (
            <div className="reports-content">
              <div className="section-header">
                <h2>Automated Reports</h2>
                <p>Advanced analytics and reporting features</p>
              </div>

              {/* Report Generation Notifications */}
              {reportNotifications.length > 0 && (
                <div className="report-notifications">
                  <h3>📅 Scheduled Report Generation</h3>
                  {reportNotifications.map(notification => (
                    <div key={notification.formId} className="notification-banner">
                      <div className="notification-content">
                        <div className="notification-header">
                          <strong>{notification.formTitle}</strong>
                          <span className="response-count">{notification.responseCount} responses</span>
                        </div>
                        <div className="notification-details">
                          <p>
                            📊 Report for this form will be automatically generated on{' '}
                            <strong>
                              {new Date(notification.reportDate).toLocaleDateString('en-US', {
                                weekday: 'long',
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric'
                              })}
                            </strong>
                            {notification.dueDate && (
                              <span className="due-info">
                                {' '}(1 day after due date: {new Date(notification.dueDate).toLocaleDateString()})
                              </span>
                            )}
                          </p>
                        </div>
                      </div>
                      <div className="notification-actions">
                        <button
                          className="btn-generate-now"
                          onClick={() => window.open(`/reports?form=${notification.formId}`, '_blank')}
                          disabled={new Date() < new Date(notification.reportDate)}
                          style={{
                            opacity: new Date() < new Date(notification.reportDate) ? 0.5 : 1,
                            cursor: new Date() < new Date(notification.reportDate) ? 'not-allowed' : 'pointer'
                          }}
                        >
                          Generate Now
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Available Reports */}
              <div className="reports-list">
                {reports.length === 0 ? (
                  <div className="empty-state">
                    <FileText size={48} />
                    <h3>No Reports Available</h3>
                    <p>Reports will be generated automatically when forms receive responses and reach their due dates.</p>
                  </div>
                ) : (
                  reports.map((report) => (
                    <motion.div
                      key={report.id}
                      className="report-card"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="report-content">
                        <h4>{report.title}</h4>
                        <p className="report-period">{report.period}</p>
                        <div className="report-meta">
                          {/* Removed status badge and auto-generated label */}
                        </div>
                        <p className="report-date">Generated: {report.generatedDate}</p>
                      </div>
                      <div className="report-actions">
                        {report.status === 'ready' ? (
                          <button
                            className="download-btn"
                            onClick={() => handleDownloadReport(report.formId)}
                          >
                            <Download size={16} />
                            Download PDF
                          </button>
                        ) : (
                          <button
                            className="generate-btn"
                            onClick={() => handleGenerateReport(report.formId)}
                            disabled={report.status === 'generating' || (report.reportDate && new Date() < new Date(report.reportDate))}
                          >
                            <BarChart3 size={16} />
                            {report.status === 'generating' ? 'Generating...' :
                             (report.reportDate && new Date() < new Date(report.reportDate)) ? 'Generate Now' : 'Generate Now'}
                          </button>
                        )}
                      </div>
                    </motion.div>
                  ))
                )}
              </div>
            </div>
          )}

          {activeTab === 'requests' && (
            <div className="requests-content">
              <div className="section-header">
                <h2>My Form Requests</h2>
                <p>Track the status of all your form requests</p>
              </div>

              {formRequests.length === 0 ? (
                <div className="empty-state">
                  <Clock size={48} />
                  <h3>No form requests</h3>
                  <p>You haven't submitted any form requests yet. Click "Create Form Request" to get started.</p>
                </div>
              ) : (
                <div className="requests-grid">
                  {formRequests.map((request) => (
                    <motion.div
                      key={request.id}
                      className={`request-card ${request.status}`}
                      whileHover={{ scale: 1.02 }}
                    >
                      <div className="request-header">
                        <h3>{request.title}</h3>
                        <div className={`status-badge ${request.status}`}>
                          {request.status === 'pending' && 'PENDING REVIEW'}
                          {request.status === 'approved' && 'APPROVED'}
                          {request.status === 'rejected' && 'REJECTED'}
                          {request.status === 'published' && 'PUBLISHED'}
                        </div>
                      </div>

                      <div className="request-details">
                        <p><strong>Description:</strong> {request.description || 'No description provided'}</p>
                        <p><strong>Session:</strong> {request.session_name || 'Not specified'}</p>
                        <p><strong>Submitted:</strong> {formatDateTime(request.created_at)}</p>

                        {request.session_date && (
                          <p><strong>Session Date:</strong> {formatDate(request.session_date)}</p>
                        )}

                        {request.due_date && (
                          <p><strong>Due Date:</strong> {formatDateTime(request.due_date)}</p>
                        )}

                        {request.form_data && (
                          <p><strong>Questions:</strong> {request.form_data.questions?.length || 0} questions included</p>
                        )}

                        {request.reviewed_at && (
                          <p><strong>Reviewed:</strong> {formatDateTime(request.reviewed_at)}</p>
                        )}

                        {request.admin_response && (
                          <div className="admin-response">
                            <strong>Admin Response:</strong>
                            <p>{request.admin_response}</p>
                          </div>
                        )}
                      </div>

                      <div className="request-actions">
                        {request.status === 'pending' && (
                          <span className="pending-message">
                            <Clock size={16} />
                            Awaiting admin review
                          </span>
                        )}

                        {request.status === 'approved' && (
                          <span className="approved-message">
                            <CheckCircle size={16} />
                            Form has been approved and published
                          </span>
                        )}

                        {request.status === 'rejected' && (
                          <button
                            className="re-edit-btn"
                            onClick={() => handleReEditForm(request)}
                          >
                            <Edit size={16} />
                            Re-edit & Resubmit
                          </button>
                        )}
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'rejected' && (
            <div className="rejected-content">
              <div className="section-header">
                <h2>Rejected Form Requests</h2>
                <p>Forms that were rejected by admins - you can re-edit and resubmit them</p>
              </div>

              {rejectedRequests.length === 0 ? (
                <div className="empty-state">
                  <CheckCircle size={48} />
                  <h3>No Rejected Requests</h3>
                  <p>All your form requests are either pending or approved!</p>
                </div>
              ) : (
                <div className="rejected-requests-grid">
                  {rejectedRequests.map((request) => (
                    <motion.div
                      key={request.id}
                      className="rejected-request-card"
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ duration: 0.3 }}
                    >
                      <div className="request-header">
                        <div className="request-info">
                          <h3>{request.title}</h3>
                          <p className="rejection-reason">
                            <strong>Rejection Reason:</strong> {request.rejection_reason || 'No reason provided'}
                          </p>
                          <p className="request-date">
                            Rejected: {request.reviewed_at ? new Date(request.reviewed_at).toLocaleDateString() : 'Recently'}
                          </p>
                        </div>
                        <div className="request-status">
                          <span className="status-badge rejected">
                            Rejected
                          </span>
                        </div>
                      </div>

                      <div className="request-details">
                        {request.description && (
                          <div className="request-description">
                            <strong>Description:</strong>
                            <span>{request.description}</span>
                          </div>
                        )}

                        {request.form_data && (
                          <div className="form-questions-preview">
                            <strong>Questions in this form:</strong>
                            <div className="questions-list">
                              {(() => {
                                try {
                                  const formData = typeof request.form_data === 'string'
                                    ? JSON.parse(request.form_data)
                                    : request.form_data;
                                  const questions = formData?.questions || [];

                                  if (questions.length === 0) {
                                    return <span className="no-questions">No questions found</span>;
                                  }

                                  return questions.slice(0, 3).map((question, index) => (
                                    <div key={index} className="question-preview">
                                      <span className="question-number">{index + 1}.</span>
                                      <span className="question-title">{question.title || 'Untitled Question'}</span>
                                    </div>
                                  ));
                                } catch (error) {
                                  return <span className="error-message">Error parsing questions</span>;
                                }
                              })()}
                            </div>
                          </div>
                        )}
                      </div>

                      <div className="request-actions">
                        <button
                          className="re-edit-btn"
                          onClick={() => handleReEditForm(request)}
                        >
                          <Edit size={16} />
                          Re-edit & Resubmit
                        </button>
                        <button
                          className="delete-btn"
                          onClick={() => handleDeleteRejectedRequest(request.id)}
                        >
                          <Trash2 size={16} />
                          Delete
                        </button>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </main>



      <Footer />
    </div>
  )
}




export default TrainerDashboard
