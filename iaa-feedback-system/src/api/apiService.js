/**
 * API Service for IAA Feedback System
 * 
 * This service provides methods for making API requests to the backend
 * with proper authentication and error handling.
 */

import authService from './authService';
import { API_BASE_URL, DEFAULT_HEADERS, handleApiError } from './config';

class ApiService {
  /**
   * Make a GET request to the API
   * @param {string} endpoint - API endpoint
   * @param {boolean} requiresAuth - Whether the request requires authentication
   * @returns {Promise<Object>} API response
   */
  async get(endpoint, requiresAuth = true) {
    try {
      const headers = requiresAuth ? authService.getAuthHeaders() : DEFAULT_HEADERS;

      console.log(`Making GET request to: ${API_BASE_URL}${endpoint}`);

      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'GET',
        headers,
      });

      console.log(`Response status: ${response.status}`);

      const data = await response.json();
      console.log('Response data:', data);

      if (!response.ok) {
        throw { response, data };
      }

      return data;
    } catch (error) {
      console.error('API Error:', error);
      return {
        success: false,
        message: error.message || 'Network error. Please check your connection.'
      };
    }
  }
  
  /**
   * Make a POST request to the API
   * @param {string} endpoint - API endpoint
   * @param {Object} body - Request body
   * @param {boolean} requiresAuth - Whether the request requires authentication
   * @returns {Promise<Object>} API response
   */
  async post(endpoint, body, requiresAuth = true) {
    try {
      const headers = requiresAuth ? authService.getAuthHeaders() : DEFAULT_HEADERS;
      
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(body),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw { response, data };
      }
      
      return data;
    } catch (error) {
      return handleApiError(error);
    }
  }
  
  /**
   * Make a PUT request to the API
   * @param {string} endpoint - API endpoint
   * @param {Object} body - Request body
   * @param {boolean} requiresAuth - Whether the request requires authentication
   * @returns {Promise<Object>} API response
   */
  async put(endpoint, body, requiresAuth = true) {
    try {
      const headers = requiresAuth ? authService.getAuthHeaders() : DEFAULT_HEADERS;
      
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'PUT',
        headers,
        body: JSON.stringify(body),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw { response, data };
      }
      
      return data;
    } catch (error) {
      return handleApiError(error);
    }
  }
  
  /**
   * Make a DELETE request to the API
   * @param {string} endpoint - API endpoint
   * @param {boolean} requiresAuth - Whether the request requires authentication
   * @returns {Promise<Object>} API response
   */
  async delete(endpoint, requiresAuth = true) {
    try {
      const headers = requiresAuth ? authService.getAuthHeaders() : DEFAULT_HEADERS;
      
      const response = await fetch(`${API_BASE_URL}${endpoint}`, {
        method: 'DELETE',
        headers,
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        throw { response, data };
      }
      
      return data;
    } catch (error) {
      return handleApiError(error);
    }
  }
  
  /**
   * Check backend health
   * @returns {Promise<Object>} Health check response
   */
  async checkHealth() {
    return this.get('/health', false);
  }
  
  /**
   * Get database information
   * @returns {Promise<Object>} Database info response
   */
  async getDatabaseInfo() {
    return this.get('/db-info', false);
  }
  
  /**
   * Setup database tables (admin only)
   * @returns {Promise<Object>} Setup response
   */
  async setupDatabase() {
    return this.post('/setup', {}, true);
  }
  
  /**
   * List database tables (admin only)
   * @returns {Promise<Object>} Tables list response
   */
  async listTables() {
    return this.get('/tables', true);
  }
}

// Create and export a singleton instance
const apiService = new ApiService();
export default apiService;
