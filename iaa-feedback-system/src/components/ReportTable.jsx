import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Eye, 
  Download, 
  ChevronUp, 
  ChevronDown, 
  Search,
  Filter,
  Star,
  Calendar,
  Users,
  FileText
} from 'lucide-react'
import './ReportTable.css'

const ReportTable = ({ data, onViewDetails }) => {
  const [sortField, setSortField] = useState('date')
  const [sortDirection, setSortDirection] = useState('desc')
  const [searchTerm, setSearchTerm] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage] = useState(10)

  if (!data || data.length === 0) {
    return (
      <div className="report-table">
        <div className="table-header">
          <h3>Detailed Reports</h3>
        </div>
        <div className="no-data">
          <FileText size={48} />
          <h3>No data available</h3>
          <p>No forms match the current filters</p>
        </div>
      </div>
    )
  }

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const getSortIcon = (field) => {
    if (sortField !== field) return null
    return sortDirection === 'asc' ? <ChevronUp size={16} /> : <ChevronDown size={16} />
  }

  const filteredData = data.filter(item =>
    item.formTitle.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.trainer.toLowerCase().includes(searchTerm.toLowerCase()) ||
    item.session.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const sortedData = [...filteredData].sort((a, b) => {
    let aValue = a[sortField]
    let bValue = b[sortField]

    if (sortField === 'date') {
      aValue = new Date(aValue)
      bValue = new Date(bValue)
    }

    if (typeof aValue === 'string') {
      aValue = aValue.toLowerCase()
      bValue = bValue.toLowerCase()
    }

    if (sortDirection === 'asc') {
      return aValue > bValue ? 1 : -1
    } else {
      return aValue < bValue ? 1 : -1
    }
  })

  const totalPages = Math.ceil(sortedData.length / itemsPerPage)
  const startIndex = (currentPage - 1) * itemsPerPage
  const paginatedData = sortedData.slice(startIndex, startIndex + itemsPerPage)

  const getStatusBadge = (status) => {
    const statusConfig = {
      active: { class: 'status-active', label: 'Active' },
      completed: { class: 'status-completed', label: 'Completed' },
      draft: { class: 'status-draft', label: 'Draft' },
      archived: { class: 'status-archived', label: 'Archived' }
    }
    
    const config = statusConfig[status] || { class: 'status-unknown', label: status }
    
    return (
      <span className={`status-badge ${config.class}`}>
        {config.label}
      </span>
    )
  }

  const renderStars = (rating) => {
    return (
      <div className="rating-stars">
        {[...Array(5)].map((_, i) => (
          <Star
            key={i}
            size={14}
            fill={i < Math.floor(rating) ? '#ffc107' : 'none'}
            stroke={i < Math.floor(rating) ? '#ffc107' : '#e0e0e0'}
          />
        ))}
        <span className="rating-value">{rating}</span>
      </div>
    )
  }

  const handleExportTable = () => {
    // Mock export functionality
    console.log('Exporting table data:', sortedData)
    alert('Table data exported! (This is a demo)')
  }

  return (
    <div className="report-table">
      <div className="table-header">
        <div className="header-left">
          <h3>Detailed Reports</h3>
          <span className="data-count">
            {filteredData.length} of {data.length} forms
          </span>
        </div>
        <div className="header-actions">
          <div className="search-container">
            <Search size={16} />
            <input
              type="text"
              placeholder="Search forms, trainers, sessions..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>
          <button className="export-table-btn" onClick={handleExportTable}>
            <Download size={16} />
            Export Table
          </button>
        </div>
      </div>

      <div className="table-container">
        <table className="data-table">
          <thead>
            <tr>
              <th 
                className={`sortable ${sortField === 'formTitle' ? 'sorted' : ''}`}
                onClick={() => handleSort('formTitle')}
              >
                <div className="th-content">
                  <FileText size={16} />
                  Form Title
                  {getSortIcon('formTitle')}
                </div>
              </th>
              <th 
                className={`sortable ${sortField === 'trainer' ? 'sorted' : ''}`}
                onClick={() => handleSort('trainer')}
              >
                <div className="th-content">
                  <Users size={16} />
                  Trainer
                  {getSortIcon('trainer')}
                </div>
              </th>
              <th 
                className={`sortable ${sortField === 'session' ? 'sorted' : ''}`}
                onClick={() => handleSort('session')}
              >
                <div className="th-content">
                  Session
                  {getSortIcon('session')}
                </div>
              </th>
              <th 
                className={`sortable ${sortField === 'responses' ? 'sorted' : ''}`}
                onClick={() => handleSort('responses')}
              >
                <div className="th-content">
                  Responses
                  {getSortIcon('responses')}
                </div>
              </th>
              <th 
                className={`sortable ${sortField === 'avgRating' ? 'sorted' : ''}`}
                onClick={() => handleSort('avgRating')}
              >
                <div className="th-content">
                  <Star size={16} />
                  Avg Rating
                  {getSortIcon('avgRating')}
                </div>
              </th>
              <th 
                className={`sortable ${sortField === 'date' ? 'sorted' : ''}`}
                onClick={() => handleSort('date')}
              >
                <div className="th-content">
                  <Calendar size={16} />
                  Date
                  {getSortIcon('date')}
                </div>
              </th>
              <th>Status</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {paginatedData.map((item, index) => (
              <motion.tr
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05, duration: 0.3 }}
              >
                <td>
                  <div className="form-title">
                    {item.formTitle}
                  </div>
                </td>
                <td>
                  <div className="trainer-info">
                    {item.trainer}
                  </div>
                </td>
                <td>
                  <div className="session-info">
                    {item.session}
                  </div>
                </td>
                <td>
                  <div className="response-count">
                    {item.responses}
                  </div>
                </td>
                <td>
                  {renderStars(item.avgRating)}
                </td>
                <td>
                  <div className="date-info">
                    {new Date(item.date).toLocaleDateString()}
                  </div>
                </td>
                <td>
                  {getStatusBadge(item.status)}
                </td>
                <td>
                  <div className="action-buttons">
                    <button
                      className="action-btn view-btn"
                      onClick={() => onViewDetails(item.id)}
                      title="View Details"
                    >
                      <Eye size={16} />
                    </button>
                    <button
                      className="action-btn download-btn"
                      onClick={() => console.log('Download report for:', item.id)}
                      title="Download Report"
                    >
                      <Download size={16} />
                    </button>
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div className="table-pagination">
          <div className="pagination-info">
            Showing {startIndex + 1}-{Math.min(startIndex + itemsPerPage, sortedData.length)} of {sortedData.length} results
          </div>
          <div className="pagination-controls">
            <button
              className="page-btn"
              onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
              disabled={currentPage === 1}
            >
              Previous
            </button>
            
            {[...Array(totalPages)].map((_, i) => {
              const page = i + 1
              if (
                page === 1 ||
                page === totalPages ||
                (page >= currentPage - 1 && page <= currentPage + 1)
              ) {
                return (
                  <button
                    key={page}
                    className={`page-btn ${currentPage === page ? 'active' : ''}`}
                    onClick={() => setCurrentPage(page)}
                  >
                    {page}
                  </button>
                )
              } else if (page === currentPage - 2 || page === currentPage + 2) {
                return <span key={page} className="page-ellipsis">...</span>
              }
              return null
            })}
            
            <button
              className="page-btn"
              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
              disabled={currentPage === totalPages}
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

export default ReportTable
