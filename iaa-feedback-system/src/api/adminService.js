import { API_BASE_URL, getAuthHeaders } from './config'

class AdminService {
  // Get all departments
  async getDepartments() {
    try {
      const response = await fetch(`${API_BASE_URL}/departments`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch departments')
      }

      return data.departments
    } catch (error) {
      console.error('Error fetching departments:', error)
      throw error
    }
  }

  // Update user departments (for trainers)
  async updateUserDepartments(departmentIds) {
    try {
      console.log('🔍 Updating departments:', departmentIds)
      console.log('🔍 Auth headers:', getAuthHeaders())

      const response = await fetch(`${API_BASE_URL}/users/departments`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify({ department_ids: departmentIds }),
      })

      console.log('🔍 Response status:', response.status)
      const data = await response.json()
      console.log('🔍 Response data:', data)

      if (!response.ok) {
        throw new Error(data.detail || data.message || 'Failed to update departments')
      }

      return data
    } catch (error) {
      console.error('Error updating user departments:', error)
      throw error
    }
  }

  // Get all trainers
  async getTrainers() {
    try {
      const response = await fetch(`${API_BASE_URL}/trainers`)
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch trainers')
      }
      
      return data.trainers
    } catch (error) {
      console.error('Error fetching trainers:', error)
      throw error
    }
  }

  // Get all form requests
  async getFormRequests() {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/form-requests`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch form requests')
      }
      
      return data.requests
    } catch (error) {
      console.error('Error fetching form requests:', error)
      throw error
    }
  }

  // Create a new form request (from trainer)
  async createFormRequest(requestData) {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/form-requests`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(requestData)
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to create form request')
      }
      
      return data.request
    } catch (error) {
      console.error('Error creating form request:', error)
      throw error
    }
  }

  // Approve a form request
  async approveFormRequest(requestId) {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/form-requests/${requestId}/approve`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      })
      
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to approve form request')
      }
      
      return data.request
    } catch (error) {
      console.error('Error approving form request:', error)
      throw error
    }
  }

  // Reject a form request
  async rejectFormRequest(requestId, reason = 'Request rejected by admin') {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/form-requests/${requestId}/reject`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({
          reason: reason
        })
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to reject form request')
      }

      return data.request
    } catch (error) {
      console.error('Error rejecting form request:', error)
      throw error
    }
  }

  // Delete a form request (for trainers to delete their own rejected requests)
  async deleteFormRequest(requestId) {
    try {
      const response = await fetch(`${API_BASE_URL}/form-requests/${requestId}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to delete form request')
      }

      return data
    } catch (error) {
      console.error('Error deleting form request:', error)
      throw error
    }
  }

  // Mark a form request as processed (published)
  async markRequestAsProcessed(requestId) {
    try {
      const token = localStorage.getItem('iaa_token')
      const response = await fetch(`${API_BASE_URL}/form-requests/${requestId}/mark-processed`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        }
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to mark request as processed')
      }

      return data.request
    } catch (error) {
      console.error('Error marking request as processed:', error)
      throw error
    }
  }

  // Get default form template
  async getDefaultFormTemplate() {
    try {
      const response = await fetch(`${API_BASE_URL}/form-templates/default`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch default template')
      }

      return data.template
    } catch (error) {
      console.error('Error fetching default template:', error)
      throw error
    }
  }

  // Get trainer form template
  async getTrainerFormTemplate() {
    try {
      const response = await fetch(`${API_BASE_URL}/trainer/form-template`, {
        headers: getAuthHeaders()
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch trainer template')
      }

      return data.template
    } catch (error) {
      console.error('Error fetching trainer template:', error)
      throw error
    }
  }

  // Get all forms
  async getForms() {
    try {
      const response = await fetch(`${API_BASE_URL}/forms`, {
        headers: getAuthHeaders()
      })
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch forms')
      }
      
      return data.forms
    } catch (error) {
      console.error('Error fetching forms:', error)
      throw error
    }
  }

  // Create a new form
  async createForm(formData) {
    try {
      // Transform frontend data to backend format
      console.log('Original form data:', formData)
      console.log('departmentId:', formData.departmentId, 'type:', typeof formData.departmentId)
      console.log('trainerId:', formData.trainerId, 'type:', typeof formData.trainerId)

      // Helper function to safely parse integer
      const safeParseInt = (value, defaultValue = null) => {
        if (value === null || value === undefined || value === '') {
          return defaultValue
        }
        const parsed = parseInt(value)
        return isNaN(parsed) ? defaultValue : parsed
      }

      // Helper function to safely convert date to ISO string
      const safeDateToISO = (date) => {
        if (!date) return null
        if (typeof date === 'string') return date // Already a string
        if (date instanceof Date) return date.toISOString()
        // Try to parse as date
        try {
          return new Date(date).toISOString()
        } catch {
          return null
        }
      }

      const backendFormData = {
        title: formData.title || '',
        description: formData.description || '',
        department_id: safeParseInt(formData.departmentId, 1), // Default to department 1 if not specified
        trainer_id: safeParseInt(formData.trainerId, null),
        due_date: safeDateToISO(formData.dueDate),
        status: formData.status || 'draft',  // ✅ Include status field
        form_data: {
          questions: formData.questions || [],
          settings: formData.settings || {},
          session: formData.session || {}
        }
      }

      console.log('Sending form data to backend:', backendFormData)

      const response = await fetch(`${API_BASE_URL}/forms`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify(backendFormData)
      })

      const data = await response.json()

      if (!response.ok) {
        console.error('Backend error response:', data)
        console.error('Response status:', response.status)
        console.error('Response headers:', response.headers)

        // Handle validation errors (422)
        if (response.status === 422 && data.detail) {
          if (Array.isArray(data.detail)) {
            const validationErrors = data.detail.map(err => `${err.loc?.join('.')}: ${err.msg}`).join(', ')
            throw new Error(`Validation error: ${validationErrors}`)
          } else {
            throw new Error(`Validation error: ${JSON.stringify(data.detail)}`)
          }
        }

        throw new Error(data.detail || `Failed to create form (${response.status})`)
      }

      return data.form
    } catch (error) {
      console.error('Error creating form:', error)
      throw error
    }
  }

  // Update an existing form
  async updateForm(formId, formData) {
    try {
      // Helper function to safely convert date to ISO string
      const safeDateToISO = (date) => {
        if (!date) return null
        if (typeof date === 'string') return date // Already a string
        if (date instanceof Date) return date.toISOString()
        // Try to parse as date
        try {
          return new Date(date).toISOString()
        } catch {
          return null
        }
      }

      // Transform frontend data to backend format
      const backendFormData = {
        title: formData.title,
        description: formData.description,
        department_id: parseInt(formData.departmentId),
        trainer_id: formData.trainerId ? parseInt(formData.trainerId) : null,
        due_date: safeDateToISO(formData.dueDate),
        form_data: {
          questions: formData.questions || [],
          settings: formData.settings || {},
          session: formData.session || {}
        }
      }

      console.log('Updating form data:', backendFormData)

      const response = await fetch(`${API_BASE_URL}/forms/${formId}`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify(backendFormData)
      })

      const data = await response.json()

      if (!response.ok) {
        console.error('Backend error response:', data)
        throw new Error(data.detail || 'Failed to update form')
      }

      return data.form
    } catch (error) {
      console.error('Error updating form:', error)
      throw error
    }
  }

  // Publish a form
  async publishForm(formId) {
    try {
      const response = await fetch(`${API_BASE_URL}/forms/${formId}/publish`, {
        method: 'PUT',
        headers: getAuthHeaders()
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to publish form')
      }

      return data.form
    } catch (error) {
      console.error('Error publishing form:', error)
      throw error
    }
  }

  // Get a specific form
  async getForm(formId) {
    try {
      const response = await fetch(`${API_BASE_URL}/forms/${formId}`, {
        headers: getAuthHeaders()
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch form')
      }

      return data.form
    } catch (error) {
      console.error('Error fetching form:', error)
      throw error
    }
  }

  // Delete a form (admin only - direct deletion)
  async deleteForm(formId) {
    try {
      const response = await fetch(`${API_BASE_URL}/forms/${formId}`, {
        method: 'DELETE',
        headers: getAuthHeaders()
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to delete form')
      }

      return data
    } catch (error) {
      console.error('Error deleting form:', error)
      throw error
    }
  }

  // Form deletion request functions
  async createFormDeletionRequest(formId, reason = '') {
    try {
      const response = await fetch(`${API_BASE_URL}/forms/${formId}/deletion-request`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ form_id: formId, reason })
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to create deletion request')
      }

      return data.request
    } catch (error) {
      console.error('Error creating deletion request:', error)
      throw error
    }
  }

  async getFormDeletionRequests() {
    try {
      const response = await fetch(`${API_BASE_URL}/form-deletion-requests`, {
        headers: getAuthHeaders()
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch deletion requests')
      }

      return data.requests
    } catch (error) {
      console.error('Error fetching deletion requests:', error)
      throw error
    }
  }

  async approveFormDeletionRequest(requestId) {
    try {
      const response = await fetch(`${API_BASE_URL}/form-deletion-requests/${requestId}/approve`, {
        method: 'PUT',
        headers: getAuthHeaders()
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to approve deletion request')
      }

      return data.request
    } catch (error) {
      console.error('Error approving deletion request:', error)
      throw error
    }
  }

  async rejectFormDeletionRequest(requestId, reason = 'Request rejected by admin') {
    try {
      const response = await fetch(`${API_BASE_URL}/form-deletion-requests/${requestId}/reject`, {
        method: 'PUT',
        headers: getAuthHeaders(),
        body: JSON.stringify({ reason })
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to reject deletion request')
      }

      return data.request
    } catch (error) {
      console.error('Error rejecting deletion request:', error)
      throw error
    }
  }

  // Get form responses
  async getFormResponses(formId) {
    try {
      const response = await fetch(`${API_BASE_URL}/forms/${formId}/responses`, {
        headers: getAuthHeaders()
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch form responses')
      }

      return data
    } catch (error) {
      console.error('Error fetching form responses:', error)
      throw error
    }
  }

  // Get dashboard statistics
  async getDashboardStats() {
    try {
      const response = await fetch(`${API_BASE_URL}/dashboard/stats`, {
        headers: getAuthHeaders()
      })
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || 'Failed to fetch dashboard stats')
      }

      return data.stats
    } catch (error) {
      console.error('Error fetching dashboard stats:', error)
      throw error
    }
  }

  // Submit form response
  async submitFormResponse(formId, responses) {
    try {
      const response = await fetch(`${API_BASE_URL}/forms/${formId}/responses`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({ responses })
      })

      const data = await response.json()

      if (!response.ok) {
        let errorMessage = data.detail || `HTTP error! status: ${response.status}`

        // Provide user-friendly error messages
        if (response.status === 400) {
          if (data.detail?.includes('already responded')) {
            errorMessage = 'You have already submitted a response to this form.'
          }
        } else if (response.status === 404) {
          if (data.detail?.includes('not published')) {
            errorMessage = 'This form is not currently available for responses.'
          } else {
            errorMessage = 'Form not found or no longer available.'
          }
        } else if (response.status === 401) {
          errorMessage = 'Please log in to submit your response.'
        } else if (response.status === 403) {
          errorMessage = 'You do not have permission to respond to this form.'
        }

        throw new Error(errorMessage)
      }

      return data
    } catch (error) {
      console.error('Error submitting form response:', error)
      throw error
    }
  }
}

export default new AdminService()
