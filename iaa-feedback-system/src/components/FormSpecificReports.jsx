import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import {
  FileText,
  Search,
  Download,
  Calendar,
  Users,
  RefreshCw,
  Filter
} from 'lucide-react'
import jsPDF from 'jspdf'
import autoTable from 'jspdf-autotable'
import { API_BASE_URL } from '../api/config'
import './FormSpecificReports.css'

const FormSpecificReports = ({ theme }) => {
  const [availableForms, setAvailableForms] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [loadingFormId, setLoadingFormId] = useState(null)
  const [searchTerm, setSearchTerm] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')
  const [departmentFilter, setDepartmentFilter] = useState('all')
  const [departments, setDepartments] = useState([])

  useEffect(() => {
    fetchAvailableForms()
    fetchDepartments()
  }, [])

  const fetchAvailableForms = async () => {
    try {
      const token = localStorage.getItem('iaa_token')
      console.log('Fetching forms with token:', token ? 'Token exists' : 'No token')
      console.log('API URL:', `${API_BASE_URL}/reports/forms`)

      const response = await fetch(`${API_BASE_URL}/reports/forms`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      console.log('Response status:', response.status)

      if (response.ok) {
        const data = await response.json()
        console.log('Forms data received:', data)
        setAvailableForms(data.forms || [])
      } else {
        console.error('Failed to fetch forms:', response.status, response.statusText)
        const errorData = await response.text()
        console.error('Error details:', errorData)
      }
    } catch (error) {
      console.error('Error fetching available forms:', error)
    }
  }

  const fetchDepartments = async () => {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/departments`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        const data = await response.json()
        setDepartments(data.departments || [])
      }
    } catch (error) {
      console.error('Error fetching departments:', error)
    }
  }



  const generateFormReport = async (formId) => {
    console.log('🎯 Generating form report for ID:', formId)

    setLoadingFormId(formId)
    setIsLoading(true)

    try {
      const token = localStorage.getItem('iaa_token')
      console.log('🔑 Token exists:', !!token)

      const response = await fetch(`${API_BASE_URL}/reports/generate/${formId}`, {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      console.log('📊 Report response status:', response.status)

      if (response.ok) {
        const data = await response.json()
        console.log('📈 Report data received:', data)

        // Find the form info
        const form = availableForms.find(f => f.id === formId)

        // Generate and download PDF
        await generateFormPDFReport(data.report, form)

      } else {
        console.error('❌ Failed to generate report:', response.status, response.statusText)
        const errorData = await response.text()
        console.error('Error details:', errorData)
        alert('Failed to generate report. Please try again.')
      }
    } catch (error) {
      console.error('❌ Error generating form report:', error)
      alert('Error generating report: ' + error.message)
    } finally {
      setIsLoading(false)
      setLoadingFormId(null)
    }
  }

  const generateFormPDFReport = async (reportData, form) => {
    try {
      console.log('📄 Generating Form PDF report...')

      const doc = new jsPDF()
      const pageWidth = doc.internal.pageSize.width
      const pageHeight = doc.internal.pageSize.height
      let yPosition = 20

      // IAA Colors
      const primaryColor = [33, 150, 243] // Blue
      const textColor = [33, 33, 33] // Dark gray

      // Cover Page
      doc.setFillColor(...primaryColor)
      doc.rect(0, 0, pageWidth, 60, 'F')

      // IAA Logo/Title
      doc.setTextColor(255, 255, 255)
      doc.setFontSize(24)
      doc.setFont('helvetica', 'bold')
      doc.text('Indian Aviation Academy', pageWidth / 2, 25, { align: 'center' })

      doc.setFontSize(18)
      doc.setFont('helvetica', 'normal')
      doc.text('Form-Specific Feedback Report', pageWidth / 2, 40, { align: 'center' })

      // Form Title
      yPosition = 80
      doc.setTextColor(...textColor)
      doc.setFontSize(20)
      doc.setFont('helvetica', 'bold')
      doc.text(`${form?.title || 'Unknown Form'}`, pageWidth / 2, yPosition, { align: 'center' })

      // Generation Date
      yPosition += 20
      doc.setFontSize(12)
      doc.setFont('helvetica', 'normal')
      const currentDate = new Date().toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric'
      })
      doc.text(`Generated on: ${currentDate}`, pageWidth / 2, yPosition, { align: 'center' })

      // Form Details Box
      yPosition += 30
      doc.setFillColor(245, 245, 245)
      doc.rect(20, yPosition, pageWidth - 40, 60, 'F')
      doc.setDrawColor(...primaryColor)
      doc.rect(20, yPosition, pageWidth - 40, 60, 'S')

      yPosition += 15
      doc.setFontSize(16)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...primaryColor)
      doc.text('Form Details', 30, yPosition)

      // Form Information
      yPosition += 15
      doc.setFontSize(10)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...textColor)

      const formDetails = [
        `Created by: ${form?.creator_name || 'Unknown'}`,
        `Department: ${form?.department_name || 'Unknown'}`,
        `Created on: ${form?.created_at ? new Date(form.created_at).toLocaleDateString() : 'Unknown'}`,
        `Status: ${form?.status || 'Unknown'}`
      ]

      formDetails.forEach((detail, index) => {
        doc.text(detail, 30, yPosition + (index * 8))
      })

      // Executive Summary Box
      yPosition += 50
      doc.setFillColor(245, 245, 245)
      doc.rect(20, yPosition, pageWidth - 40, 80, 'F')
      doc.setDrawColor(...primaryColor)
      doc.rect(20, yPosition, pageWidth - 40, 80, 'S')

      yPosition += 15
      doc.setFontSize(16)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...primaryColor)
      doc.text('Executive Summary', 30, yPosition)

      // Key Metrics
      yPosition += 20
      doc.setFontSize(12)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...textColor)

      const stats = reportData?.statistics || {}
      const metrics = [
        `Total Responses: ${stats.total_responses || 0}`,
        `Average Rating: ${stats.average_rating || 'N/A'}`,
        `Response Rate: ${stats.response_rate || 0}%`,
        `Completion Rate: ${stats.completion_rate || 0}%`
      ]

      metrics.forEach((metric, index) => {
        const x = 30 + (index % 2) * 80
        const y = yPosition + Math.floor(index / 2) * 15
        doc.text(metric, x, y)
      })

      // Add new page for detailed content
      doc.addPage()
      yPosition = 20

      // Response Analysis Section
      doc.setFontSize(16)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...primaryColor)
      doc.text('Response Analysis', 20, yPosition)

      yPosition += 15
      doc.setFontSize(10)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...textColor)

      // Response summary table
      const responseData = [
        ['Metric', 'Value'],
        ['Total Responses Received', (stats.total_responses || 0).toString()],
        ['Average Rating', (stats.average_rating || 'N/A').toString()],
        ['Highest Rating', (stats.highest_rating || 'N/A').toString()],
        ['Lowest Rating', (stats.lowest_rating || 'N/A').toString()],
        ['Response Rate', `${stats.response_rate || 0}%`]
      ]

      autoTable(doc, {
        startY: yPosition,
        head: [responseData[0]],
        body: responseData.slice(1),
        theme: 'grid',
        headStyles: { fillColor: primaryColor, textColor: [255, 255, 255] },
        alternateRowStyles: { fillColor: [248, 249, 250] },
        margin: { left: 20, right: 20 }
      })

      yPosition = doc.lastAutoTable.finalY + 20

      // Responses Section
      if (reportData?.responses && reportData.responses.length > 0) {
        doc.setFontSize(16)
        doc.setFont('helvetica', 'bold')
        doc.setTextColor(...primaryColor)
        doc.text('Individual Responses', 20, yPosition)

        yPosition += 15

        // Responses table
        const responseTableData = [
          ['Trainee', 'Rating', 'Submitted Date', 'Comments']
        ]

        reportData.responses.forEach(response => {
          responseTableData.push([
            response.trainee_name || 'Anonymous',
            response.rating ? response.rating.toString() : 'N/A',
            response.submitted_at ? new Date(response.submitted_at).toLocaleDateString() : 'N/A',
            response.comments ? (response.comments.length > 50 ? response.comments.substring(0, 50) + '...' : response.comments) : 'No comments'
          ])
        })

        autoTable(doc, {
          startY: yPosition,
          head: [responseTableData[0]],
          body: responseTableData.slice(1),
          theme: 'grid',
          headStyles: { fillColor: primaryColor, textColor: [255, 255, 255] },
          alternateRowStyles: { fillColor: [248, 249, 250] },
          margin: { left: 20, right: 20 },
          styles: { fontSize: 9 }
        })

        yPosition = doc.lastAutoTable.finalY + 20
      }

      // Insights and Recommendations
      if (yPosition > pageHeight - 60) {
        doc.addPage()
        yPosition = 20
      }

      doc.setFontSize(16)
      doc.setFont('helvetica', 'bold')
      doc.setTextColor(...primaryColor)
      doc.text('Key Insights & Recommendations', 20, yPosition)

      yPosition += 15
      doc.setFontSize(10)
      doc.setFont('helvetica', 'normal')
      doc.setTextColor(...textColor)

      // Insights
      if (reportData?.insights && reportData.insights.length > 0) {
        doc.setFont('helvetica', 'bold')
        doc.text('Key Insights:', 20, yPosition)
        yPosition += 10

        doc.setFont('helvetica', 'normal')
        reportData.insights.forEach((insight) => {
          const lines = doc.splitTextToSize(`• ${insight}`, pageWidth - 40)
          doc.text(lines, 25, yPosition)
          yPosition += lines.length * 5 + 5
        })

        yPosition += 10
      }

      // Recommendations
      if (reportData?.recommendations && reportData.recommendations.length > 0) {
        doc.setFont('helvetica', 'bold')
        doc.text('Recommendations:', 20, yPosition)
        yPosition += 10

        doc.setFont('helvetica', 'normal')
        reportData.recommendations.forEach((recommendation) => {
          const lines = doc.splitTextToSize(`• ${recommendation}`, pageWidth - 40)
          doc.text(lines, 25, yPosition)
          yPosition += lines.length * 5 + 5
        })
      }

      // Footer on all pages
      const pageCount = doc.internal.getNumberOfPages()
      for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i)
        doc.setFontSize(8)
        doc.setTextColor(128, 128, 128)
        doc.text(`Page ${i} of ${pageCount}`, pageWidth - 30, pageHeight - 10)
        doc.text('Confidential - Indian Aviation Academy', 20, pageHeight - 10)
        doc.text(`Generated: ${new Date().toLocaleString()}`, pageWidth / 2, pageHeight - 10, { align: 'center' })
      }

      // Download the PDF
      const fileName = `Form-Report-${form?.title?.replace(/[^a-zA-Z0-9]/g, '-') || 'Unknown'}-${new Date().toISOString().split('T')[0]}.pdf`
      doc.save(fileName)

      console.log('✅ PDF report generated successfully:', fileName)

    } catch (error) {
      console.error('❌ Error generating PDF:', error)
      alert('Error generating PDF report: ' + error.message)
    }
  }

  const filteredForms = availableForms.filter(form => {
    const matchesSearch = form.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         form.creator_name?.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesStatus = statusFilter === 'all' || form.status === statusFilter
    const matchesDepartment = departmentFilter === 'all' || form.department_name === departmentFilter

    return matchesSearch && matchesStatus && matchesDepartment
  })

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    })
  }

  return (
    <div className="form-specific-reports">
      <div className="reports-header">
        <h2>Form-Specific Reports</h2>
        <p>Generate comprehensive PDF reports with detailed analytics - Click any form to download</p>
      </div>

      <div className="form-selection">
        {/* Filters */}
        <div className="filters-section">
          <div className="search-filter">
            <Search size={20} />
            <input
              type="text"
              placeholder="Search forms..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div className="dropdown-filter">
            <Filter size={16} />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
            >
              <option value="all">All Status</option>
              <option value="published">Published</option>
              <option value="draft">Draft</option>
            </select>
          </div>

          <div className="dropdown-filter">
            <Users size={16} />
            <select
              value={departmentFilter}
              onChange={(e) => setDepartmentFilter(e.target.value)}
            >
              <option value="all">All Departments</option>
              {departments.map(dept => (
                <option key={dept.id} value={dept.name}>{dept.name}</option>
              ))}
            </select>
          </div>



          <button className="refresh-btn" onClick={fetchAvailableForms}>
            <RefreshCw size={16} />
            Refresh
          </button>
        </div>

        {/* Forms Grid */}
        <div className="forms-grid">
          {filteredForms.map((form, index) => (
            <motion.div
              key={form.id}
              className="form-card"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1, duration: 0.5 }}
              whileHover={{ scale: 1.02 }}
            >
              <div className="form-card-header">
                <FileText size={24} />
                <span className={`status-badge ${form.status}`}>
                  {form.status}
                </span>
              </div>

              <h3>{form.title}</h3>
              <p className="form-meta">
                <span>Created by: {form.creator_name}</span>
                <span>Department: {form.department_name}</span>
                <span>Created: {formatDate(form.created_at)}</span>
              </p>

              <div className="form-stats">
                <div className="stat">
                  <Users size={16} />
                  <span>{form.response_count} responses</span>
                </div>
              </div>

              <button
                className={`generate-btn ${loadingFormId === form.id ? 'loading' : ''}`}
                onClick={() => generateFormReport(form.id)}
                disabled={form.response_count === 0 || loadingFormId === form.id}
              >
                {loadingFormId === form.id ? (
                  <>
                    <RefreshCw size={16} className="spinning" />
                    Generating PDF...
                  </>
                ) : (
                  <>
                    <Download size={16} />
                    Download PDF Report
                  </>
                )}
              </button>
            </motion.div>
          ))}
        </div>

        {filteredForms.length === 0 && !isLoading && (
          <div className="no-forms">
            <FileText size={48} />
            <h3>No forms found</h3>
            <p>No forms match your current filters</p>
          </div>
        )}

        {isLoading && filteredForms.length === 0 && (
          <div className="loading-forms">
            <RefreshCw size={32} className="spinning" />
            <p>Loading forms...</p>
          </div>
        )}
      </div>
    </div>
  )
}

export default FormSpecificReports
