import React, { useState, useEffect, useMemo } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LineChart, Line, AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  RadialBarChart, RadialBar, ScatterChart, Scatter
} from 'recharts'
import analysisService from '../services/analysisService'
import realTimeService, { EVENTS } from '../services/realTimeService'
import { useAuth } from '../contexts/AuthContext'
import './TrainerAnalyticsDashboard.css'

const TrainerAnalyticsDashboard = ({ forms, responses, isRealTime = true }) => {
  const { user } = useAuth()
  const [selectedForm, setSelectedForm] = useState(null)
  const [analysisData, setAnalysisData] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [refreshInterval, setRefreshInterval] = useState(30000) // 30 seconds
  const [lastUpdated, setLastUpdated] = useState(new Date())
  const [realTimeData, setRealTimeData] = useState(null)
  const [connectionStatus, setConnectionStatus] = useState('disconnected')

  // Color schemes for different chart types
  const colors = {
    primary: '#2196F3',
    success: '#4CAF50',
    warning: '#FF9800',
    error: '#F44336',
    info: '#00BCD4',
    purple: '#9C27B0',
    gradient: ['#2196F3', '#21CBF3', '#4CAF50', '#8BC34A', '#CDDC39']
  }

  // Analyze all forms and responses
  const comprehensiveAnalysis = useMemo(() => {
    if (!forms || forms.length === 0) return null

    const allResponses = forms.flatMap(form => form.responses || [])
    const overallScore = analysisService.calculateTrainerScore(allResponses)

    // Form-specific analysis
    const formAnalyses = forms.map(form => {
      const formResponses = form.responses || []
      const formScore = analysisService.calculateTrainerScore(formResponses)
      
      // Text analysis for this form
      const textResponses = formResponses.flatMap(r => 
        (r.responses || []).filter(item => item.type === 'text' || item.type === 'textarea')
      )
      
      const textAnalyses = textResponses.map(item => 
        analysisService.analyzeTextResponse(item.value)
      )

      // Sentiment distribution
      const sentimentCounts = textAnalyses.reduce((acc, analysis) => {
        acc[analysis.sentiment.label] = (acc[analysis.sentiment.label] || 0) + 1
        return acc
      }, { positive: 0, negative: 0, neutral: 0 })

      // Topic analysis
      const allTopics = textAnalyses.flatMap(analysis => analysis.topics)
      const topicCounts = allTopics.reduce((acc, topic) => {
        acc[topic.topic] = (acc[topic.topic] || 0) + topic.relevance
        return acc
      }, {})

      return {
        formId: form.id,
        title: form.title,
        score: formScore,
        sentimentDistribution: sentimentCounts,
        topTopics: Object.entries(topicCounts)
          .sort((a, b) => b[1] - a[1])
          .slice(0, 5)
          .map(([topic, score]) => ({ topic, score: Math.round(score * 100) })),
        responseCount: formResponses.length,
        avgResponseLength: textResponses.length > 0 ? 
          textResponses.reduce((sum, r) => sum + (r.value?.length || 0), 0) / textResponses.length : 0,
        emojiAnalysis: textAnalyses.map(a => a.emojiAnalysis).filter(e => e.count > 0)
      }
    })

    // Time-based performance trends (simulated for demo)
    const performanceTrends = forms.map((form, index) => ({
      date: new Date(Date.now() - (forms.length - index) * 7 * 24 * 60 * 60 * 1000).toLocaleDateString(),
      score: Math.max(60, overallScore.overallScore + (Math.random() - 0.5) * 20),
      responses: form.totalResponses || 0,
      sentiment: Math.random() * 100
    }))

    return {
      overall: overallScore,
      byForm: formAnalyses,
      trends: performanceTrends,
      totalForms: forms.length,
      totalResponses: allResponses.length,
      lastAnalyzed: new Date()
    }
  }, [forms])

  // Initialize real-time connection
  useEffect(() => {
    if (!isRealTime || !user) return

    const initializeRealTime = async () => {
      try {
        const token = localStorage.getItem('access_token')
        if (token) {
          await realTimeService.initialize(token)
          setConnectionStatus(realTimeService.getStatus().connectionType)
        }
      } catch (error) {
        console.error('Failed to initialize real-time service:', error)
        setConnectionStatus('error')
      }
    }

    initializeRealTime()

    // Subscribe to relevant events
    const unsubscribers = [
      realTimeService.subscribe(EVENTS.FORM_RESPONSE, (data) => {
        console.log('📊 New form response received:', data)
        setRealTimeData(prev => ({ ...prev, newResponse: data }))
        setLastUpdated(new Date())
      }),

      realTimeService.subscribe(EVENTS.SCORE_UPDATE, (data) => {
        console.log('📈 Score update received:', data)
        setRealTimeData(prev => ({ ...prev, scoreUpdate: data }))
        setLastUpdated(new Date())
      }),

      realTimeService.subscribe(EVENTS.ANALYTICS_UPDATE, (data) => {
        console.log('📊 Analytics update received:', data)
        setRealTimeData(prev => ({ ...prev, analyticsUpdate: data }))
        setLastUpdated(new Date())
      }),

      // Global event listener for connection status
      realTimeService.subscribe('*', () => {
        setConnectionStatus(realTimeService.getStatus().connectionType)
      })
    ]

    // Polling fallback
    const interval = setInterval(() => {
      setLastUpdated(new Date())
      // Request updates if needed
      if (realTimeService.getStatus().connectionType === 'polling') {
        realTimeService.requestUpdate('trainer_analytics', { trainerId: user.id })
      }
    }, refreshInterval)

    return () => {
      unsubscribers.forEach(unsub => unsub())
      clearInterval(interval)
    }
  }, [isRealTime, user, refreshInterval])

  // Chart data preparation
  const chartData = useMemo(() => {
    if (!comprehensiveAnalysis) return null

    const { overall, byForm, trends } = comprehensiveAnalysis

    return {
      scoreDistribution: [
        { name: 'Numeric Scores', value: overall.breakdown.numeric.score, color: colors.primary },
        { name: 'Sentiment', value: overall.breakdown.sentiment.score, color: colors.success },
        { name: 'Engagement', value: overall.breakdown.engagement.score, color: colors.warning }
      ],
      performanceTrends: trends,
      formComparison: byForm.map(form => ({
        name: form.title.substring(0, 20) + (form.title.length > 20 ? '...' : ''),
        score: form.score.overallScore,
        responses: form.responseCount,
        sentiment: form.sentimentDistribution.positive || 0
      })),
      sentimentOverall: [
        { name: 'Positive', value: byForm.reduce((sum, f) => sum + f.sentimentDistribution.positive, 0), color: colors.success },
        { name: 'Neutral', value: byForm.reduce((sum, f) => sum + f.sentimentDistribution.neutral, 0), color: colors.warning },
        { name: 'Negative', value: byForm.reduce((sum, f) => sum + f.sentimentDistribution.negative, 0), color: colors.error }
      ],
      topicAnalysis: byForm.flatMap(f => f.topTopics).reduce((acc, topic) => {
        const existing = acc.find(t => t.topic === topic.topic)
        if (existing) {
          existing.score += topic.score
        } else {
          acc.push({ ...topic })
        }
        return acc
      }, []).sort((a, b) => b.score - a.score).slice(0, 8)
    }
  }, [comprehensiveAnalysis])

  if (!comprehensiveAnalysis || !chartData) {
    return (
      <div className="analytics-dashboard">
        <div className="loading-state">
          <motion.div
            className="loading-spinner"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
          <p>Analyzing responses...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="analytics-dashboard">
      {/* Header */}
      <div className="dashboard-header">
        <div className="header-content">
          <h2>📊 Real-Time Analytics Dashboard</h2>
          <div className="header-stats">
            <div className="stat-item">
              <span className="stat-value">{comprehensiveAnalysis.overall.overallScore}</span>
              <span className="stat-label">Overall Score</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{comprehensiveAnalysis.totalResponses}</span>
              <span className="stat-label">Total Responses</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{comprehensiveAnalysis.overall.confidence}%</span>
              <span className="stat-label">Confidence</span>
            </div>
          </div>
        </div>
        <div className="real-time-indicator">
          <motion.div
            className={`pulse-dot ${connectionStatus}`}
            animate={{
              scale: connectionStatus === 'websocket' ? [1, 1.2, 1] : [1, 1, 1],
              opacity: connectionStatus === 'error' ? [0.5, 0.5, 0.5] : [1, 0.7, 1]
            }}
            transition={{ duration: 2, repeat: Infinity }}
          />
          <span>
            {connectionStatus === 'websocket' && '🔴 Live WebSocket'}
            {connectionStatus === 'polling' && '🟡 Live Polling'}
            {connectionStatus === 'disconnected' && '⚫ Disconnected'}
            {connectionStatus === 'error' && '🔴 Connection Error'}
            {' • Updated ' + lastUpdated.toLocaleTimeString()}
          </span>
        </div>
      </div>

      {/* Main Charts Grid */}
      <div className="charts-grid">
        {/* Overall Score Breakdown */}
        <motion.div 
          className="chart-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <h3>Score Breakdown</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={chartData.scoreDistribution}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={120}
                paddingAngle={5}
                dataKey="value"
              >
                {chartData.scoreDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`${value}%`, 'Score']} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Performance Trends */}
        <motion.div 
          className="chart-container wide"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h3>Performance Trends</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={chartData.performanceTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis domain={[0, 100]} />
              <Tooltip />
              <Area 
                type="monotone" 
                dataKey="score" 
                stroke={colors.primary} 
                fill={colors.primary}
                fillOpacity={0.3}
              />
            </AreaChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Sentiment Analysis */}
        <motion.div 
          className="chart-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h3>Sentiment Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData.sentimentOverall}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill={colors.primary}>
                {chartData.sentimentOverall.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Topic Analysis */}
        <motion.div 
          className="chart-container"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h3>Key Topics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData.topicAnalysis} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, 100]} />
              <YAxis dataKey="topic" type="category" width={100} />
              <Tooltip />
              <Bar dataKey="score" fill={colors.success} />
            </BarChart>
          </ResponsiveContainer>
        </motion.div>

        {/* Form Comparison */}
        <motion.div 
          className="chart-container wide"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <h3>Form Performance Comparison</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={chartData.formComparison}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="responses" name="Responses" />
              <YAxis dataKey="score" name="Score" domain={[0, 100]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter dataKey="score" fill={colors.purple} />
            </ScatterChart>
          </ResponsiveContainer>
        </motion.div>
      </div>

      {/* Real-time Controls */}
      <div className="dashboard-controls">
        <div className="control-group">
          <label>Refresh Interval:</label>
          <select 
            value={refreshInterval} 
            onChange={(e) => setRefreshInterval(Number(e.target.value))}
          >
            <option value={10000}>10 seconds</option>
            <option value={30000}>30 seconds</option>
            <option value={60000}>1 minute</option>
            <option value={300000}>5 minutes</option>
          </select>
        </div>
        <div className="control-buttons">
          <button
            className="refresh-btn"
            onClick={() => {
              setLastUpdated(new Date())
              if (user) {
                realTimeService.requestUpdate('trainer_analytics', { trainerId: user.id })
              }
            }}
          >
            🔄 Refresh Now
          </button>

          <button
            className={`connection-btn ${connectionStatus}`}
            onClick={() => {
              if (connectionStatus === 'disconnected' || connectionStatus === 'error') {
                const token = localStorage.getItem('access_token')
                if (token) {
                  realTimeService.initialize(token)
                }
              } else {
                realTimeService.disconnect()
                setConnectionStatus('disconnected')
              }
            }}
          >
            {connectionStatus === 'websocket' && '🔌 Disconnect'}
            {connectionStatus === 'polling' && '🔌 Disconnect'}
            {(connectionStatus === 'disconnected' || connectionStatus === 'error') && '🔗 Reconnect'}
          </button>
        </div>
      </div>
    </div>
  )
}

export default TrainerAnalyticsDashboard
