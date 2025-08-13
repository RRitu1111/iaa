import React from 'react'
import ErrorPage from '../pages/ErrorPage'

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props)
    this.state = { 
      hasError: false, 
      error: null, 
      errorInfo: null 
    }
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true }
  }

  componentDidCatch(error, errorInfo) {
    // Log error details
    console.error('ErrorBoundary caught an error:', error, errorInfo)
    
    this.setState({
      error: error,
      errorInfo: errorInfo
    })

    // You can also log the error to an error reporting service here
    // logErrorToService(error, errorInfo)
  }

  handleRetry = () => {
    this.setState({ 
      hasError: false, 
      error: null, 
      errorInfo: null 
    })
  }

  handleGoHome = () => {
    window.location.href = '/'
  }

  render() {
    if (this.state.hasError) {
      const customActions = [
        {
          label: 'Try Again',
          onClick: this.handleRetry,
          variant: 'secondary',
          icon: null
        }
      ]

      return (
        <ErrorPage
          errorCode="500"
          errorTitle="Application Error"
          errorMessage="Something went wrong in the application. Our technical team has been notified."
          showRetry={false}
          showGoHome={true}
          showGoBack={false}
          customActions={customActions}
        />
      )
    }

    return this.props.children
  }
}

export default ErrorBoundary
