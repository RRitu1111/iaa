import { motion } from 'framer-motion'
import { GraduationCap, UserCheck, Shield } from 'lucide-react'
import './RoleAvatar.css'

const RoleAvatar = ({ role, isSelected, onClick, t }) => {
  const avatarConfig = {
    trainee: {
      icon: GraduationCap,
      color: '#4caf50',
      bgColor: '#e8f5e8',
      darkBgColor: '#2d4a2d'
    },
    trainer: {
      icon: UserCheck,
      color: '#2196f3',
      bgColor: '#e3f2fd',
      darkBgColor: '#1a2d3d'
    },
    admin: {
      icon: Shield,
      color: '#ff9800',
      bgColor: '#fff3e0',
      darkBgColor: '#3d2d1a'
    }
  }

  const config = avatarConfig[role]
  const IconComponent = config.icon

  return (
    <motion.div
      className={`role-avatar ${isSelected ? 'selected' : ''}`}
      onClick={() => onClick(role)}
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      animate={{
        scale: isSelected ? 1.1 : 1,
        y: isSelected ? -10 : 0
      }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 20
      }}
      style={{
        '--avatar-color': config.color,
        '--avatar-bg': config.bgColor,
        '--avatar-dark-bg': config.darkBgColor
      }}
    >
      <div className="avatar-circle">
        <IconComponent size={48} className="avatar-icon" />
      </div>
      <div className="avatar-label">
        <h3>{t(`register.roles.${role}`)}</h3>
      </div>
      {isSelected && (
        <motion.div
          className="selection-indicator"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ delay: 0.1 }}
        />
      )}
    </motion.div>
  )
}

export default RoleAvatar
