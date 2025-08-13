/**
 * Authentication Service for IAA Feedback System
 * 
 * This service handles all authentication-related API calls
 * including login, register, token refresh, and user management.
 */

import { AUTH_ENDPOINTS, DEFAULT_HEADERS, handleApiError } from './config.js';

class AuthService {
  constructor() {
    this.token = localStorage.getItem('iaa_token');
    this.user = JSON.parse(localStorage.getItem('iaa_user') || 'null');
    this.refreshToken = localStorage.getItem('iaa_refresh_token');
  }

  /**
   * Register a new user
   * @param {Object} userData - User registration data
   * @returns {Promise<Object>} API response
   */
  async register(userData) {
    try {
      const response = await fetch(AUTH_ENDPOINTS.REGISTER, {
        method: 'POST',
        headers: DEFAULT_HEADERS,
        body: JSON.stringify({
          email: userData.email,
          password: userData.password,
          first_name: userData.firstName,
          last_name: userData.lastName,
          role: userData.role.toLowerCase(), // admin, trainer, trainee
          department_id: userData.departmentId || null,
          supervisor_email: userData.supervisorEmail || null,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Registration failed');
      }

      return {
        success: true,
        message: data.message || 'Registration successful',
        user: data.data,
      };
    } catch (error) {
      console.error('Registration error:', error);
      return {
        success: false,
        message: error.message || 'Registration failed',
      };
    }
  }

  /**
   * Login user with ID (preferred method)
   * @param {string} role - User role (admin/trainer/trainee)
   * @param {string} userId - User ID
   * @param {string} password - User password
   * @returns {Promise<Object>} API response with token and user data
   */
  async loginWithId(role, userId, password) {
    try {
      const response = await fetch(AUTH_ENDPOINTS.LOGIN, {
        method: 'POST',
        headers: DEFAULT_HEADERS,
        body: JSON.stringify({
          role,
          userId,
          password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        return {
          success: false,
          message: data.detail || data.message || 'Login failed',
        };
      }

      // Check if backend response indicates success
      if (data.success && data.data) {
        this.token = data.data.access_token;
        this.refreshToken = data.data.refresh_token;
        this.user = data.data.user;

        localStorage.setItem('iaa_token', this.token);
        localStorage.setItem('iaa_refresh_token', this.refreshToken);
        localStorage.setItem('iaa_user', JSON.stringify(this.user));

        console.log('Login successful:', this.user);

        return {
          success: true,
          message: data.message || 'Login successful',
          user: this.user,
          token: this.token,
        };
      } else {
        console.error('Login failed - Backend error:', data);
        return {
          success: false,
          message: data.message || 'Login failed',
        };
      }
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        message: error.message || 'Login failed',
      };
    }
  }

  /**
   * Login user with email (backward compatibility)
   * @param {string} email - User email
   * @param {string} password - User password
   * @returns {Promise<Object>} API response with token and user data
   */
  async login(email, password) {
    try {
      const response = await fetch(AUTH_ENDPOINTS.LOGIN, {
        method: 'POST',
        headers: DEFAULT_HEADERS,
        body: JSON.stringify({
          email,
          password,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        console.error('Login failed - HTTP error:', response.status, data);
        return {
          success: false,
          message: data.detail || data.message || 'Login failed',
        };
      }

      // Check if backend response indicates success
      if (data.success && data.data) {
        this.token = data.data.access_token;
        this.refreshToken = data.data.refresh_token;
        this.user = data.data.user;

        localStorage.setItem('iaa_token', this.token);
        localStorage.setItem('iaa_refresh_token', this.refreshToken);
        localStorage.setItem('iaa_user', JSON.stringify(this.user));

        console.log('Login successful:', this.user);

        return {
          success: true,
          message: data.message || 'Login successful',
          user: this.user,
          token: this.token,
        };
      } else {
        console.error('Login failed - Backend error:', data);
        return {
          success: false,
          message: data.message || 'Login failed',
        };
      }
    } catch (error) {
      console.error('Login error:', error);
      return {
        success: false,
        message: error.message || 'Login failed',
      };
    }
  }

  /**
   * Refresh authentication token
   * @returns {Promise<Object>} API response with new token
   */
  async refreshAuthToken() {
    if (!this.refreshToken) {
      return {
        success: false,
        message: 'No refresh token available',
      };
    }

    try {
      const response = await fetch(AUTH_ENDPOINTS.REFRESH_TOKEN, {
        method: 'POST',
        headers: DEFAULT_HEADERS,
        body: JSON.stringify({
          refresh_token: this.refreshToken,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.message || 'Token refresh failed');
      }

      // Update stored token
      if (data.success && data.data) {
        this.token = data.data.access_token;
        localStorage.setItem('iaa_token', this.token);
      }

      return {
        success: true,
        message: 'Token refreshed successfully',
        token: this.token,
      };
    } catch (error) {
      console.error('Token refresh error:', error);
      this.logout(); // Clear invalid tokens
      return {
        success: false,
        message: error.message || 'Token refresh failed',
      };
    }
  }

  /**
   * Logout user and clear stored data
   */
  logout() {
    this.token = null;
    this.refreshToken = null;
    this.user = null;

    localStorage.removeItem('iaa_token');
    localStorage.removeItem('iaa_refresh_token');
    localStorage.removeItem('iaa_user');
  }

  /**
   * Check if user is authenticated
   * @returns {boolean} Authentication status
   */
  isAuthenticated() {
    return !!this.token && !!this.user;
  }

  /**
   * Get current user data
   * @returns {Object|null} User data or null
   */
  getCurrentUser() {
    return this.user;
  }

  /**
   * Get current user role
   * @returns {string|null} User role or null
   */
  getUserRole() {
    return this.user?.role || null;
  }

  /**
   * Check if user has specific role
   * @param {string} role - Role to check
   * @returns {boolean} Role check result
   */
  hasRole(role) {
    return this.user?.role === role.toLowerCase();
  }

  /**
   * Check if user is admin
   * @returns {boolean} Admin status
   */
  isAdmin() {
    return this.hasRole('admin');
  }

  /**
   * Check if user is trainer
   * @returns {boolean} Trainer status
   */
  isTrainer() {
    return this.hasRole('trainer');
  }

  /**
   * Check if user is trainee
   * @returns {boolean} Trainee status
   */
  isTrainee() {
    return this.hasRole('trainee');
  }

  /**
   * Get authorization headers for API requests
   * @returns {Object} Headers with authorization token
   */
  getAuthHeaders() {
    return {
      ...DEFAULT_HEADERS,
      'Authorization': this.token ? `Bearer ${this.token}` : '',
    };
  }
}

// Create and export a singleton instance
const authService = new AuthService();
export default authService;
