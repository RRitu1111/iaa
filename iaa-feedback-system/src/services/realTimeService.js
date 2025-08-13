/**
 * Real-Time Service for Live Updates
 * Implements both WebSocket and polling mechanisms for real-time data updates
 */

class RealTimeService {
  constructor() {
    this.websocket = null
    this.pollingInterval = null
    this.subscribers = new Map()
    this.isConnected = false
    this.reconnectAttempts = 0
    this.maxReconnectAttempts = 5
    this.reconnectDelay = 1000
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001'
    this.wsUrl = this.baseUrl.replace('http', 'ws') + '/ws'
    this.pollingEnabled = true
    this.pollingIntervalMs = 30000 // 30 seconds default
  }

  /**
   * Initialize real-time connection
   */
  async initialize(authToken) {
    this.authToken = authToken

    // Try WebSocket first, fallback to polling
    try {
      await this.connectWebSocket()
    } catch (error) {
      // Silently fallback to polling - this is expected in development
      console.log('🔄 Using polling mode for real-time updates')
      this.startPolling()
    }
  }

  /**
   * Connect to WebSocket server
   */
  async connectWebSocket() {
    return new Promise((resolve, reject) => {
      try {
        this.websocket = new WebSocket(`${this.wsUrl}?token=${this.authToken}`)

        // Set a timeout for connection
        const connectionTimeout = setTimeout(() => {
          if (this.websocket.readyState !== WebSocket.OPEN) {
            this.websocket.close()
            reject(new Error('WebSocket connection timeout'))
          }
        }, 5000) // 5 second timeout

        this.websocket.onopen = () => {
          clearTimeout(connectionTimeout)
          console.log('✅ WebSocket connected')
          this.isConnected = true
          this.reconnectAttempts = 0
          this.stopPolling() // Stop polling if WebSocket works
          resolve()
        }

        this.websocket.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data)
            this.handleMessage(data)
          } catch (error) {
            console.error('Error parsing WebSocket message:', error)
          }
        }

        this.websocket.onclose = (event) => {
          console.log('🔌 WebSocket disconnected:', event.code, event.reason)
          this.isConnected = false
          
          if (event.code !== 1000) { // Not a normal closure
            this.handleReconnect()
          }
        }

        this.websocket.onerror = (error) => {
          console.warn('⚠️ WebSocket connection failed, will fallback to polling')
          this.isConnected = false
          clearTimeout(connectionTimeout)
          reject(error)
        }

        // Timeout for connection
        setTimeout(() => {
          if (!this.isConnected) {
            this.websocket.close()
            reject(new Error('WebSocket connection timeout'))
          }
        }, 5000)

      } catch (error) {
        reject(error)
      }
    })
  }

  /**
   * Handle WebSocket reconnection
   */
  handleReconnect() {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1)
      
      console.log(`🔄 Attempting to reconnect WebSocket (${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`)
      
      setTimeout(() => {
        this.connectWebSocket().catch(() => {
          if (this.reconnectAttempts >= this.maxReconnectAttempts) {
            console.log('🔄 Max reconnection attempts reached, falling back to polling')
            this.startPolling()
          }
        })
      }, delay)
    } else {
      console.log('🔄 Falling back to polling after failed WebSocket reconnections')
      this.startPolling()
    }
  }

  /**
   * Start polling for updates
   */
  startPolling() {
    if (this.pollingInterval) return // Already polling
    
    console.log('🔄 Starting polling for real-time updates')
    this.pollingEnabled = true
    
    this.pollingInterval = setInterval(async () => {
      if (!this.pollingEnabled) return
      
      try {
        await this.pollForUpdates()
      } catch (error) {
        console.error('Polling error:', error)
      }
    }, this.pollingIntervalMs)
  }

  /**
   * Stop polling
   */
  stopPolling() {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval)
      this.pollingInterval = null
      this.pollingEnabled = false
      console.log('⏹️ Stopped polling')
    }
  }

  /**
   * Poll for updates from the server
   */
  async pollForUpdates() {
    try {
      const response = await fetch(`${this.baseUrl}/realtime/updates`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json'
        }
      })

      if (response.ok) {
        const data = await response.json()
        this.handleMessage(data)
      }
    } catch (error) {
      console.error('Error polling for updates:', error)
    }
  }

  /**
   * Handle incoming messages/updates
   */
  handleMessage(data) {
    const { type, payload, timestamp } = data
    
    // Notify all subscribers for this message type
    if (this.subscribers.has(type)) {
      this.subscribers.get(type).forEach(callback => {
        try {
          callback(payload, timestamp)
        } catch (error) {
          console.error(`Error in subscriber callback for ${type}:`, error)
        }
      })
    }

    // Notify global subscribers
    if (this.subscribers.has('*')) {
      this.subscribers.get('*').forEach(callback => {
        try {
          callback({ type, payload, timestamp })
        } catch (error) {
          console.error('Error in global subscriber callback:', error)
        }
      })
    }
  }

  /**
   * Subscribe to real-time updates
   */
  subscribe(eventType, callback) {
    if (!this.subscribers.has(eventType)) {
      this.subscribers.set(eventType, new Set())
    }
    
    this.subscribers.get(eventType).add(callback)
    
    // Return unsubscribe function
    return () => {
      if (this.subscribers.has(eventType)) {
        this.subscribers.get(eventType).delete(callback)
        
        // Clean up empty subscriber sets
        if (this.subscribers.get(eventType).size === 0) {
          this.subscribers.delete(eventType)
        }
      }
    }
  }

  /**
   * Send message through WebSocket
   */
  send(type, payload) {
    if (this.isConnected && this.websocket) {
      const message = {
        type,
        payload,
        timestamp: new Date().toISOString()
      }
      
      this.websocket.send(JSON.stringify(message))
      return true
    }
    
    return false
  }

  /**
   * Set polling interval
   */
  setPollingInterval(intervalMs) {
    this.pollingIntervalMs = intervalMs
    
    if (this.pollingInterval) {
      this.stopPolling()
      this.startPolling()
    }
  }

  /**
   * Get connection status
   */
  getStatus() {
    return {
      isConnected: this.isConnected,
      connectionType: this.isConnected ? 'websocket' : (this.pollingEnabled ? 'polling' : 'disconnected'),
      subscriberCount: Array.from(this.subscribers.values()).reduce((total, set) => total + set.size, 0),
      reconnectAttempts: this.reconnectAttempts
    }
  }

  /**
   * Disconnect and cleanup
   */
  disconnect() {
    console.log('🔌 Disconnecting real-time service')
    
    if (this.websocket) {
      this.websocket.close(1000, 'Client disconnect')
      this.websocket = null
    }
    
    this.stopPolling()
    this.subscribers.clear()
    this.isConnected = false
    this.reconnectAttempts = 0
  }

  /**
   * Request immediate update for specific data
   */
  async requestUpdate(dataType, params = {}) {
    if (this.isConnected) {
      // Send request through WebSocket
      this.send('request_update', { dataType, params })
    } else {
      // Make direct API call
      try {
        const response = await fetch(`${this.baseUrl}/realtime/request-update`, {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${this.authToken}`,
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ dataType, params })
        })

        if (response.ok) {
          const data = await response.json()
          this.handleMessage({
            type: `${dataType}_update`,
            payload: data,
            timestamp: new Date().toISOString()
          })
        }
      } catch (error) {
        console.error('Error requesting update:', error)
      }
    }
  }
}

// Create singleton instance
const realTimeService = new RealTimeService()

// Event types constants
export const EVENTS = {
  FORM_RESPONSE: 'form_response',
  SCORE_UPDATE: 'score_update',
  ANALYTICS_UPDATE: 'analytics_update',
  FORM_STATUS_CHANGE: 'form_status_change',
  NEW_FORM: 'new_form',
  USER_ACTIVITY: 'user_activity',
  SYSTEM_ALERT: 'system_alert'
}

export default realTimeService
