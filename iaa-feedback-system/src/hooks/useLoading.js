import { useState, useCallback } from 'react'

/**
 * Custom hook for managing loading states with the LoadingPage component
 */
export const useLoading = () => {
  const [isLoading, setIsLoading] = useState(false)
  const [loadingMessage, setLoadingMessage] = useState('Loading...')
  const [loadingSubMessage, setLoadingSubMessage] = useState('Please wait while we prepare your experience')

  const showLoading = useCallback((message, subMessage) => {
    if (message) setLoadingMessage(message)
    if (subMessage) setLoadingSubMessage(subMessage)
    setIsLoading(true)
  }, [])

  const hideLoading = useCallback(() => {
    setIsLoading(false)
  }, [])

  const withLoading = useCallback(async (asyncFunction, message, subMessage) => {
    showLoading(message, subMessage)
    try {
      const result = await asyncFunction()
      return result
    } finally {
      hideLoading()
    }
  }, [showLoading, hideLoading])

  return {
    isLoading,
    loadingMessage,
    loadingSubMessage,
    showLoading,
    hideLoading,
    withLoading
  }
}

/**
 * Utility functions for common loading scenarios
 */
export const LoadingUtils = {
  // Simulate network delay for development
  delay: (ms = 2000) => new Promise(resolve => setTimeout(resolve, ms)),
  
  // Common loading messages
  messages: {
    login: {
      message: 'Authenticating...',
      subMessage: 'Verifying your credentials with IAA systems'
    },
    dashboard: {
      message: 'Loading Dashboard...',
      subMessage: 'Preparing your personalized aviation training interface'
    },
    forms: {
      message: 'Loading Forms...',
      subMessage: 'Fetching your training feedback forms'
    },
    reports: {
      message: 'Generating Reports...',
      subMessage: 'Analyzing training data and performance metrics'
    },
    formBuilder: {
      message: 'Loading Form Builder...',
      subMessage: 'Preparing the advanced form creation tools'
    },
    submission: {
      message: 'Submitting Form...',
      subMessage: 'Securely sending your feedback to the training system'
    },
    logout: {
      message: 'Signing Out...',
      subMessage: 'Safely logging you out of the IAA system'
    }
  }
}

export default useLoading
