import { motion } from 'framer-motion'
import { ArrowRight, Users, FileText, Database, BarChart3, FileBarChart } from 'lucide-react'
import './DataFlowVisualization.css'

const DataFlowVisualization = () => {
  const flowSteps = [
    {
      id: 1,
      icon: Users,
      title: "User Input",
      description: "Trainees submit feedback",
      color: "#2196f3"
    },
    {
      id: 2,
      icon: FileText,
      title: "Form Processing",
      description: "Data validation & processing",
      color: "#4caf50"
    },
    {
      id: 3,
      icon: Database,
      title: "Data Storage",
      description: "Secure database storage",
      color: "#ff9800"
    },
    {
      id: 4,
      icon: BarChart3,
      title: "Analytics",
      description: "Real-time analysis",
      color: "#9c27b0"
    },
    {
      id: 5,
      icon: FileBarChart,
      title: "Reports",
      description: "Comprehensive insights",
      color: "#f44336"
    }
  ]

  return (
    <div className="data-flow-visualization">
      <div className="flow-container">
        {flowSteps.map((step, index) => (
          <motion.div
            key={step.id}
            className="flow-step"
            initial={{ opacity: 0, y: 50 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.2, duration: 0.6 }}
          >
            <motion.div
              className="step-icon"
              style={{ backgroundColor: step.color }}
              whileHover={{ scale: 1.1, rotate: 5 }}
              transition={{ duration: 0.3 }}
            >
              <step.icon size={32} color="white" />
            </motion.div>
            
            <div className="step-content">
              <h4>{step.title}</h4>
              <p>{step.description}</p>
            </div>

            {index < flowSteps.length - 1 && (
              <motion.div
                className="flow-arrow"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.2 + 0.3, duration: 0.4 }}
              >
                <ArrowRight size={24} />
              </motion.div>
            )}
          </motion.div>
        ))}
      </div>

      {/* Animated Data Flow Lines */}
      <svg className="flow-lines" viewBox="0 0 800 200">
        <defs>
          <linearGradient id="flowGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#2196f3" />
            <stop offset="25%" stopColor="#4caf50" />
            <stop offset="50%" stopColor="#ff9800" />
            <stop offset="75%" stopColor="#9c27b0" />
            <stop offset="100%" stopColor="#f44336" />
          </linearGradient>
        </defs>
        
        <motion.path
          d="M 50 100 Q 200 50 350 100 Q 500 150 650 100 Q 750 50 800 100"
          stroke="url(#flowGradient)"
          strokeWidth="3"
          fill="none"
          strokeDasharray="10,5"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
        />
        
        {/* Animated particles */}
        {[...Array(5)].map((_, i) => (
          <motion.circle
            key={i}
            r="4"
            fill="#2196f3"
            initial={{ x: 50, y: 100 }}
            animate={{
              x: [50, 200, 350, 500, 650, 800],
              y: [100, 50, 100, 150, 100, 100]
            }}
            transition={{
              duration: 4,
              repeat: Infinity,
              delay: i * 0.8,
              ease: "linear"
            }}
          />
        ))}
      </svg>
    </div>
  )
}

export default DataFlowVisualization
