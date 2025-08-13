import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  BarChart3, 
  PieChart, 
  TrendingUp, 
  Users,
  Star,
  Calendar,
  ChevronDown,
  Download
} from 'lucide-react'
import './ReportCharts.css'

const ReportCharts = ({ data, detailed = false }) => {
  const [selectedChart, setSelectedChart] = useState('responses')

  if (!data) {
    return (
      <div className="charts-container">
        <div className="no-data">
          <BarChart3 size={48} />
          <h3>No data available</h3>
          <p>Apply filters to generate charts</p>
        </div>
      </div>
    )
  }

  const chartTypes = [
    { id: 'responses', name: 'Response Trends', icon: TrendingUp },
    { id: 'ratings', name: 'Rating Distribution', icon: Star },
    { id: 'trainers', name: 'Trainer Performance', icon: Users },
    { id: 'department', name: 'Department Metrics', icon: BarChart3 },
    { id: 'comparison', name: 'Performance Comparison', icon: PieChart }
  ]

  const renderResponseTrendChart = () => {
    const maxResponses = Math.max(...data.responsesByDate.map(d => d.responses))
    
    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>Response Trends</h3>
          <button className="export-chart-btn">
            <Download size={16} />
          </button>
        </div>
        <div className="line-chart">
          <div className="chart-grid">
            {[...Array(5)].map((_, i) => (
              <div key={i} className="grid-line" style={{ bottom: `${(i * 20)}%` }}>
                <span className="grid-label">{Math.round((maxResponses / 4) * i)}</span>
              </div>
            ))}
          </div>
          <div className="chart-bars">
            {data.responsesByDate.map((item, index) => (
              <motion.div
                key={index}
                className="chart-bar"
                initial={{ height: 0 }}
                animate={{ height: `${(item.responses / maxResponses) * 100}%` }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <div className="bar-tooltip">
                  <span>{item.responses} responses</span>
                  <span>{new Date(item.date).toLocaleDateString()}</span>
                </div>
              </motion.div>
            ))}
          </div>
          <div className="chart-labels">
            {data.responsesByDate.map((item, index) => (
              <span key={index} className="chart-label">
                {new Date(item.date).toLocaleDateString('en-US', { weekday: 'short' })}
              </span>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const renderRatingDistributionChart = () => {
    const totalRatings = data.ratingDistribution.reduce((sum, item) => sum + item.count, 0)
    
    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>Rating Distribution</h3>
          <button className="export-chart-btn">
            <Download size={16} />
          </button>
        </div>
        <div className="rating-chart">
          <div className="rating-bars">
            {data.ratingDistribution.map((item, index) => (
              <motion.div
                key={index}
                className="rating-bar-container"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <div className="rating-bar">
                  <motion.div
                    className="rating-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${(item.count / totalRatings) * 100}%` }}
                    transition={{ delay: index * 0.1 + 0.3, duration: 0.8 }}
                  />
                  <div className="rating-info">
                    <span className="rating-stars">
                      {[...Array(item.rating)].map((_, i) => (
                        <Star key={i} size={16} fill="#ffc107" />
                      ))}
                    </span>
                    <span className="rating-count">{item.count}</span>
                    <span className="rating-percentage">
                      {Math.round((item.count / totalRatings) * 100)}%
                    </span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
          <div className="rating-summary">
            <div className="summary-item">
              <span className="summary-label">Total Ratings</span>
              <span className="summary-value">{totalRatings}</span>
            </div>
            <div className="summary-item">
              <span className="summary-label">Average Rating</span>
              <span className="summary-value">
                {(data.ratingDistribution.reduce((sum, item) => sum + (item.rating * item.count), 0) / totalRatings).toFixed(1)}
              </span>
            </div>
          </div>
        </div>
      </div>
    )
  }

  const renderTrainerPerformanceChart = () => {
    const maxResponses = Math.max(...data.trainerPerformance.map(t => t.responses))
    
    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>Trainer Performance</h3>
          <button className="export-chart-btn">
            <Download size={16} />
          </button>
        </div>
        <div className="trainer-chart">
          {data.trainerPerformance.map((trainer, index) => (
            <motion.div
              key={index}
              className="trainer-item"
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
            >
              <div className="trainer-info">
                <span className="trainer-name">{trainer.trainer}</span>
                <div className="trainer-stats">
                  <span className="stat">
                    <Star size={14} fill="#ffc107" />
                    {trainer.avgRating}
                  </span>
                  <span className="stat">
                    <Users size={14} />
                    {trainer.responses} responses
                  </span>
                </div>
              </div>
              <div className="trainer-bars">
                <div className="rating-bar">
                  <span className="bar-label">Rating</span>
                  <div className="bar-track">
                    <motion.div
                      className="bar-fill rating-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${(trainer.avgRating / 5) * 100}%` }}
                      transition={{ delay: index * 0.1 + 0.3, duration: 0.8 }}
                    />
                  </div>
                  <span className="bar-value">{trainer.avgRating}</span>
                </div>
                <div className="response-bar">
                  <span className="bar-label">Responses</span>
                  <div className="bar-track">
                    <motion.div
                      className="bar-fill response-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${(trainer.responses / maxResponses) * 100}%` }}
                      transition={{ delay: index * 0.1 + 0.5, duration: 0.8 }}
                    />
                  </div>
                  <span className="bar-value">{trainer.responses}</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    )
  }

  const renderChart = () => {
    switch (selectedChart) {
      case 'responses':
        return renderResponseTrendChart()
      case 'ratings':
        return renderRatingDistributionChart()
      case 'trainers':
        return renderTrainerPerformanceChart()
      case 'department':
        return renderDepartmentMetricsChart()
      case 'comparison':
        return renderPerformanceComparisonChart()
      default:
        return renderResponseTrendChart()
    }
  }

  const renderDepartmentMetricsChart = () => {
    if (!data.departmentMetrics) return null

    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>Department Metrics</h3>
          <button className="export-chart-btn">
            <Download size={16} />
          </button>
        </div>
        <div className="department-metrics-chart">
          <div className="metrics-grid">
            {data.departmentMetrics.map((dept, index) => (
              <motion.div
                key={index}
                className="metric-card"
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <div className="metric-header">
                  <h4>{dept.name}</h4>
                  <span className={`performance-indicator ${dept.performance}`}>
                    {dept.performance}
                  </span>
                </div>
                <div className="metric-stats">
                  <div className="stat">
                    <span className="label">Forms</span>
                    <span className="value">{dept.totalForms}</span>
                  </div>
                  <div className="stat">
                    <span className="label">Avg Rating</span>
                    <span className="value">{dept.avgRating}</span>
                  </div>
                  <div className="stat">
                    <span className="label">Responses</span>
                    <span className="value">{dept.totalResponses}</span>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  const renderPerformanceComparisonChart = () => {
    if (!data.performanceComparison) return null

    return (
      <div className="chart-container">
        <div className="chart-header">
          <h3>Performance Comparison</h3>
          <button className="export-chart-btn">
            <Download size={16} />
          </button>
        </div>
        <div className="comparison-chart">
          <div className="comparison-bars">
            {data.performanceComparison.map((item, index) => (
              <motion.div
                key={index}
                className="comparison-bar-container"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1, duration: 0.5 }}
              >
                <div className="comparison-info">
                  <span className="trainer-name">{item.name}</span>
                  <span className="rating-value">{item.rating}</span>
                </div>
                <div className="comparison-bar">
                  <motion.div
                    className="comparison-fill"
                    initial={{ width: 0 }}
                    animate={{ width: `${(item.rating / 5) * 100}%` }}
                    transition={{ delay: index * 0.1 + 0.3, duration: 0.8 }}
                    style={{
                      backgroundColor: item.rating >= 4 ? '#22c55e' :
                                     item.rating >= 3.5 ? '#3b82f6' :
                                     item.rating >= 3 ? '#fbbf24' : '#ef4444'
                    }}
                  />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="charts-container">
      {detailed && (
        <div className="chart-selector">
          <div className="selector-buttons">
            {chartTypes.map((chart) => (
              <button
                key={chart.id}
                className={`chart-btn ${selectedChart === chart.id ? 'active' : ''}`}
                onClick={() => setSelectedChart(chart.id)}
              >
                <chart.icon size={16} />
                {chart.name}
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="charts-grid">
        {detailed ? (
          renderChart()
        ) : (
          <>
            {renderResponseTrendChart()}
            {renderRatingDistributionChart()}
            {renderTrainerPerformanceChart()}
          </>
        )}
      </div>
    </div>
  )
}

export default ReportCharts
