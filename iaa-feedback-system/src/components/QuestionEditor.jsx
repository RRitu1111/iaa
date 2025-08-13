import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Trash2, 
  Copy, 
  ChevronUp, 
  ChevronDown, 
  Plus, 
  Minus,
  Settings,
  Type,
  AlignLeft,
  List,
  Star,
  ToggleLeft,
  CheckSquare,
  Hash,
  Mail,
  Sliders
} from 'lucide-react'
import './QuestionEditor.css'

const QuestionEditor = ({
  question,
  index,
  isActive,
  onUpdate,
  onDelete,
  onDuplicate,
  onMove,
  onActivate,
  canMoveUp,
  canMoveDown
}) => {
  const [showAdvanced, setShowAdvanced] = useState(false)

  const getQuestionIcon = (type) => {
    switch (type) {
      case 'text': return Type
      case 'textarea': return AlignLeft
      case 'multiple-choice': return List
      case 'rating': return Star
      case 'yes-no': return ToggleLeft
      case 'checkbox': return CheckSquare
      case 'number': return Hash
      case 'email': return Mail
      case 'slider': return Sliders
      default: return Type
    }
  }

  const getQuestionTypeName = (type) => {
    switch (type) {
      case 'text': return 'Short Text'
      case 'textarea': return 'Long Text'
      case 'multiple-choice': return 'Multiple Choice'
      case 'rating': return 'Rating Scale'
      case 'yes-no': return 'Yes/No'
      case 'checkbox': return 'Checkboxes'
      case 'number': return 'Number'
      case 'email': return 'Email'
      case 'slider': return 'Slider'
      default: return 'Unknown'
    }
  }

  const addOption = () => {
    const newOptions = [...question.options, `Option ${question.options.length + 1}`]
    onUpdate({ options: newOptions })
  }

  const updateOption = (index, value) => {
    const newOptions = [...question.options]
    newOptions[index] = value
    onUpdate({ options: newOptions })
  }

  const removeOption = (index) => {
    if (question.options.length > 2) {
      const newOptions = question.options.filter((_, i) => i !== index)
      onUpdate({ options: newOptions })
    }
  }

  const updateRatingScale = (field, value) => {
    onUpdate({
      scale: {
        ...question.scale,
        [field]: value
      }
    })
  }

  return (
    <motion.div
      className={`question-editor ${isActive ? 'active' : ''}`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.3 }}
      onClick={onActivate}
    >
      <div className="question-header">
        <div className="question-info">
          <div className="question-type">
            {(() => {
              const Icon = getQuestionIcon(question.type)
              return <Icon size={16} />
            })()}
            <span>Question {index + 1} - {getQuestionTypeName(question.type)}</span>
          </div>
          <div className="question-actions">
            <button
              className="action-btn"
              onClick={(e) => {
                e.stopPropagation()
                onMove('up')
              }}
              disabled={!canMoveUp}
              title="Move Up"
            >
              <ChevronUp size={16} />
            </button>
            <button
              className="action-btn"
              onClick={(e) => {
                e.stopPropagation()
                onMove('down')
              }}
              disabled={!canMoveDown}
              title="Move Down"
            >
              <ChevronDown size={16} />
            </button>
            <button
              className="action-btn"
              onClick={(e) => {
                e.stopPropagation()
                onDuplicate()
              }}
              title="Duplicate"
            >
              <Copy size={16} />
            </button>
            <button
              className="action-btn delete"
              onClick={(e) => {
                e.stopPropagation()
                onDelete()
              }}
              title="Delete"
            >
              <Trash2 size={16} />
            </button>
          </div>
        </div>
      </div>

      {isActive && (
        <motion.div
          className="question-content"
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          transition={{ duration: 0.3 }}
        >
          <div className="question-basic">
            <div className="field-group">
              <label>Question Title *</label>
              <input
                type="text"
                placeholder="Enter your question"
                value={question.title}
                onChange={(e) => onUpdate({ title: e.target.value })}
                className="question-title-input"
              />
            </div>

            <div className="field-group">
              <label>Description (Optional)</label>
              <textarea
                placeholder="Add additional context or instructions"
                value={question.description}
                onChange={(e) => onUpdate({ description: e.target.value })}
                rows={2}
              />
            </div>

            <div className="question-settings">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={question.required}
                  onChange={(e) => onUpdate({ required: e.target.checked })}
                />
                Required Question
              </label>
            </div>
          </div>

          {/* Question Type Specific Options */}
          {question.type === 'multiple-choice' && (
            <div className="question-options">
              <label>Answer Options</label>
              <div className="options-list">
                {question.options.map((option, index) => (
                  <div key={index} className="option-item">
                    <input
                      type="text"
                      value={option}
                      onChange={(e) => updateOption(index, e.target.value)}
                      placeholder={`Option ${index + 1}`}
                    />
                    <button
                      className="remove-option"
                      onClick={() => removeOption(index)}
                      disabled={question.options.length <= 2}
                    >
                      <Minus size={16} />
                    </button>
                  </div>
                ))}
              </div>
              <button className="add-option" onClick={addOption}>
                <Plus size={16} />
                Add Option
              </button>
            </div>
          )}

          {question.type === 'rating' && (
            <div className="rating-settings">
              <label>Rating Scale Settings</label>
              <div className="rating-config">
                <div className="scale-range">
                  <div className="field-group">
                    <label>Minimum</label>
                    <select
                      value={question.scale?.min || 1}
                      onChange={(e) => updateRatingScale('min', parseInt(e.target.value))}
                    >
                      <option value={1}>1</option>
                      <option value={0}>0</option>
                    </select>
                  </div>
                  <div className="field-group">
                    <label>Maximum</label>
                    <select
                      value={question.scale?.max || 5}
                      onChange={(e) => updateRatingScale('max', parseInt(e.target.value))}
                    >
                      <option value={5}>5</option>
                      <option value={10}>10</option>
                    </select>
                  </div>
                </div>
                <div className="scale-labels">
                  <div className="field-group">
                    <label>Low Label</label>
                    <input
                      type="text"
                      value={question.scale?.labels?.[0] || ''}
                      onChange={(e) => {
                        const labels = [...(question.scale?.labels || ['', ''])]
                        labels[0] = e.target.value
                        updateRatingScale('labels', labels)
                      }}
                      placeholder="e.g., Poor"
                    />
                  </div>
                  <div className="field-group">
                    <label>High Label</label>
                    <input
                      type="text"
                      value={question.scale?.labels?.[1] || ''}
                      onChange={(e) => {
                        const labels = [...(question.scale?.labels || ['', ''])]
                        labels[1] = e.target.value
                        updateRatingScale('labels', labels)
                      }}
                      placeholder="e.g., Excellent"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {question.type === 'text' && (
            <div className="text-settings">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={question.validation?.email || false}
                  onChange={(e) => onUpdate({
                    validation: { ...question.validation, email: e.target.checked }
                  })}
                />
                Validate as Email
              </label>
            </div>
          )}

          {question.type === 'checkbox' && (
            <div className="question-options">
              <label>Checkbox Options</label>
              <div className="options-list">
                {question.options.map((option, index) => (
                  <div key={index} className="option-item">
                    <input
                      type="text"
                      value={option}
                      onChange={(e) => updateOption(index, e.target.value)}
                      placeholder={`Option ${index + 1}`}
                    />
                    <button
                      className="remove-option"
                      onClick={() => removeOption(index)}
                      disabled={question.options.length <= 1}
                    >
                      <Minus size={16} />
                    </button>
                  </div>
                ))}
              </div>
              <button className="add-option" onClick={addOption}>
                <Plus size={16} />
                Add Option
              </button>
            </div>
          )}

          {question.type === 'number' && (
            <div className="number-settings">
              <label>Number Validation</label>
              <div className="number-config">
                <div className="field-group">
                  <label>Minimum Value</label>
                  <input
                    type="number"
                    value={question.validation?.min || ''}
                    onChange={(e) => onUpdate({
                      validation: { ...question.validation, min: e.target.value ? parseInt(e.target.value) : null }
                    })}
                    placeholder="No minimum"
                  />
                </div>
                <div className="field-group">
                  <label>Maximum Value</label>
                  <input
                    type="number"
                    value={question.validation?.max || ''}
                    onChange={(e) => onUpdate({
                      validation: { ...question.validation, max: e.target.value ? parseInt(e.target.value) : null }
                    })}
                    placeholder="No maximum"
                  />
                </div>
              </div>
            </div>
          )}

          {question.type === 'slider' && (
            <div className="slider-settings">
              <label>Slider Configuration</label>
              <div className="slider-config">
                <div className="slider-range">
                  <div className="field-group">
                    <label>Minimum</label>
                    <input
                      type="number"
                      value={question.slider?.min || 0}
                      onChange={(e) => onUpdate({
                        slider: { ...question.slider, min: parseInt(e.target.value) }
                      })}
                    />
                  </div>
                  <div className="field-group">
                    <label>Maximum</label>
                    <input
                      type="number"
                      value={question.slider?.max || 100}
                      onChange={(e) => onUpdate({
                        slider: { ...question.slider, max: parseInt(e.target.value) }
                      })}
                    />
                  </div>
                  <div className="field-group">
                    <label>Step</label>
                    <input
                      type="number"
                      value={question.slider?.step || 1}
                      onChange={(e) => onUpdate({
                        slider: { ...question.slider, step: parseInt(e.target.value) }
                      })}
                    />
                  </div>
                </div>
                <div className="slider-labels">
                  <div className="field-group">
                    <label>Low Label</label>
                    <input
                      type="text"
                      value={question.slider?.labels?.[0] || ''}
                      onChange={(e) => {
                        const labels = [...(question.slider?.labels || ['', ''])]
                        labels[0] = e.target.value
                        onUpdate({
                          slider: { ...question.slider, labels }
                        })
                      }}
                      placeholder="e.g., Low"
                    />
                  </div>
                  <div className="field-group">
                    <label>High Label</label>
                    <input
                      type="text"
                      value={question.slider?.labels?.[1] || ''}
                      onChange={(e) => {
                        const labels = [...(question.slider?.labels || ['', ''])]
                        labels[1] = e.target.value
                        onUpdate({
                          slider: { ...question.slider, labels }
                        })
                      }}
                      placeholder="e.g., High"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}

          {question.type === 'email' && (
            <div className="email-settings">
              <label className="checkbox-label">
                <input
                  type="checkbox"
                  checked={question.validation?.required || false}
                  onChange={(e) => onUpdate({
                    validation: { ...question.validation, required: e.target.checked }
                  })}
                />
                Require valid email format
              </label>
            </div>
          )}

          {/* Advanced Settings */}
          <div className="advanced-settings">
            <button
              className="advanced-toggle"
              onClick={() => setShowAdvanced(!showAdvanced)}
            >
              <Settings size={16} />
              Advanced Settings
              <ChevronDown 
                size={16} 
                style={{ 
                  transform: showAdvanced ? 'rotate(180deg)' : 'rotate(0deg)',
                  transition: 'transform 0.3s ease'
                }}
              />
            </button>

            {showAdvanced && (
              <motion.div
                className="advanced-content"
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <div className="field-group">
                  <label>Conditional Logic</label>
                  <select
                    value={question.conditional?.questionId || ''}
                    onChange={(e) => onUpdate({
                      conditional: e.target.value ? {
                        questionId: e.target.value,
                        condition: 'equals',
                        value: ''
                      } : null
                    })}
                  >
                    <option value="">No condition</option>
                    <option value="prev">Show based on previous answer</option>
                  </select>
                </div>

                {question.conditional && (
                  <div className="conditional-settings">
                    <div className="field-group">
                      <label>Show when answer</label>
                      <select
                        value={question.conditional.condition}
                        onChange={(e) => onUpdate({
                          conditional: { ...question.conditional, condition: e.target.value }
                        })}
                      >
                        <option value="equals">equals</option>
                        <option value="not-equals">does not equal</option>
                        <option value="contains">contains</option>
                      </select>
                    </div>
                    <div className="field-group">
                      <label>Value</label>
                      <input
                        type="text"
                        value={question.conditional.value}
                        onChange={(e) => onUpdate({
                          conditional: { ...question.conditional, value: e.target.value }
                        })}
                        placeholder="Enter value"
                      />
                    </div>
                  </div>
                )}

                <div className="field-group">
                  <label>Help Text</label>
                  <input
                    type="text"
                    value={question.helpText || ''}
                    onChange={(e) => onUpdate({ helpText: e.target.value })}
                    placeholder="Additional help or instructions"
                  />
                </div>
              </motion.div>
            )}
          </div>
        </motion.div>
      )}
    </motion.div>
  )
}

export default QuestionEditor
