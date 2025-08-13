import { motion } from 'framer-motion'
import { User, UserX, Shield, Lock, Database, ArrowRight, Eye, EyeOff } from 'lucide-react'
import './AnonymousVisualization.css'

const AnonymousVisualization = () => {
  return (
    <div className="anonymous-visualization">
      <div className="anonymity-flow">
        {/* Row 1: Step 1 and Step 2 */}
        {/* Step 1: User Input */}
        <motion.div
          className="anonymity-step step-1"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.8 }}
        >
          <div className="step-icon user-icon">
            <User size={32} color="white" />
          </div>
          <h4>User Feedback</h4>
          <p>Original feedback with personal data</p>
          <div className="data-representation">
            <div className="data-item visible">Name: Rajesh Kumar</div>
            <div className="data-item visible">ID: IAA2024</div>
            <div className="data-item visible">Feedback: Excellent training!</div>
          </div>
        </motion.div>

        {/* Arrow 1 - Horizontal */}
        <motion.div
          className="flow-arrow horizontal-arrow"
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.8, duration: 0.5 }}
        >
          <ArrowRight size={24} />
        </motion.div>

        {/* Step 2: Identity Removal */}
        <motion.div
          className="anonymity-step step-2"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 1.0, duration: 0.8 }}
        >
          <div className="step-icon removal-icon">
            <UserX size={32} color="white" />
          </div>
          <h4>Identity Removal</h4>
          <p>Personal identifiers stripped away</p>
          <div className="data-representation">
            <div className="data-item hidden">Name: ████████████</div>
            <div className="data-item hidden">ID: ███████</div>
            <div className="data-item visible">Feedback: Excellent training!</div>
          </div>
        </motion.div>

        {/* Arrow 2 - Vertical down */}
        <motion.div
          className="flow-arrow vertical-arrow"
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 1.6, duration: 0.5 }}
        >
          <ArrowRight size={24} style={{ transform: 'rotate(90deg)' }} />
        </motion.div>

        {/* Empty cell for grid alignment */}
        <div className="grid-spacer"></div>

        {/* Row 2: Step 4 and Step 3 (reversed order for flow) */}
        {/* Step 4: Secure Storage */}
        <motion.div
          className="anonymity-step step-4"
          initial={{ opacity: 0, x: -50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 2.6, duration: 0.8 }}
        >
          <div className="step-icon storage-icon">
            <Database size={32} color="white" />
          </div>
          <h4>Secure Storage</h4>
          <p>Anonymous data safely stored</p>
          <div className="data-representation">
            <div className="data-item secure">🔒 Anonymous Record #A1B2C3</div>
            <div className="data-item secure">🛡️ Encrypted Content</div>
            <div className="data-item secure">✅ Identity Protected</div>
          </div>
        </motion.div>

        {/* Arrow 3 - Horizontal (left arrow) */}
        <motion.div
          className="flow-arrow horizontal-arrow"
          initial={{ opacity: 0, scale: 0 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 2.4, duration: 0.5 }}
        >
          <ArrowRight size={24} style={{ transform: 'rotate(180deg)' }} />
        </motion.div>

        {/* Step 3: Encryption */}
        <motion.div
          className="anonymity-step step-3"
          initial={{ opacity: 0, x: 50 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 1.8, duration: 0.8 }}
        >
          <div className="step-icon encryption-icon">
            <Lock size={32} color="white" />
          </div>
          <h4>Data Encryption</h4>
          <p>Feedback encrypted for security</p>
          <div className="data-representation">
            <div className="data-item encrypted">████: ████████████</div>
            <div className="data-item encrypted">██: ███████</div>
            <div className="data-item encrypted">████████: █████████ ████████!</div>
          </div>
        </motion.div>
      </div>

      {/* Privacy Guarantee Banner */}
      <motion.div
        className="privacy-guarantee"
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 3.2, duration: 0.8 }}
      >
        <div className="guarantee-icon">
          <Shield size={24} />
        </div>
        <div className="guarantee-text">
          <h5>100% Anonymous</h5>
          <p>Your identity is completely protected throughout the entire process</p>
        </div>
        <div className="guarantee-indicators">
          <div className="indicator">
            <EyeOff size={16} />
            <span>No Tracking</span>
          </div>
          <div className="indicator">
            <Lock size={16} />
            <span>Encrypted</span>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

export default AnonymousVisualization
