import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import DatePicker from 'react-datepicker'
import 'react-datepicker/dist/react-datepicker.css'
import { 
  Plus, 
  Trash2, 
  Edit, 
  Save, 
  Eye, 
  Copy,
  ChevronUp,
  ChevronDown,
  Settings,
  Type,
  List,
  Star,
  ToggleLeft,
  AlignLeft,
  Calendar,
  Users,
  BookOpen,
  CheckSquare,
  Sliders,
  Hash,
  Mail
} from 'lucide-react'
import QuestionEditor from './QuestionEditor'
import './FormBuilder.css'

const FormBuilder = ({
  onSave,
  onPreview,
  initialForm = null,
  trainers = [],
  departments = [],
  onFormChange,
  currentView = 'builder',
  isTrainerMode = false,
  currentUser = null
}) => {
  // Initialize form with proper defaults
  const getInitialFormState = () => {
    if (initialForm) {
      return {
        ...initialForm,
        trainerId: initialForm.trainerId || (trainers.length > 0 ? trainers[0].id.toString() : ''),
        departmentId: initialForm.departmentId || (departments.length > 0 ? departments[0].id.toString() : ''),
        questions: initialForm.questions || [],
        settings: initialForm.settings || {
          allowAnonymous: true,
          requireAll: false,
          showProgress: true,
          randomizeQuestions: false
        }
      }
    }

    return {
      id: null,
      title: '',
      description: '',
      trainerId: trainers.length > 0 ? trainers[0].id.toString() : '',
      departmentId: departments.length > 0 ? departments[0].id.toString() : '',
      dueDate: null,
      session: {
        name: '',
        date: '',
        course: '',
        duration: ''
      },
      questions: [],
      settings: {
        allowAnonymous: true,
        requireAll: false,
        showProgress: true,
        randomizeQuestions: false
      },
      status: 'draft'
    }
  }

  const [form, setForm] = useState(getInitialFormState)
  const [draggedItem, setDraggedItem] = useState(null)
  const [activeQuestion, setActiveQuestion] = useState(null)
  const [showSettings, setShowSettings] = useState(false)
  const [showTemplates, setShowTemplates] = useState(false)

  // Only update form when initialForm actually changes (not on every render)
  useEffect(() => {
    if (initialForm && JSON.stringify(initialForm) !== JSON.stringify(form)) {
      console.log('FormBuilder: Updating form with new initialForm')
      setForm(getInitialFormState())
    }
  }, [initialForm?.id, initialForm?.title]) // Only depend on key identifying properties

  // Call onFormChange when form state changes (for trainer mode) - debounced
  useEffect(() => {
    if (onFormChange && isTrainerMode) {
      const timeoutId = setTimeout(() => {
        onFormChange(form)
      }, 100) // Small delay to prevent excessive calls

      return () => clearTimeout(timeoutId)
    }
  }, [form, onFormChange, isTrainerMode])

  // Helper function to get trainer's departments
  const getTrainerDepartments = () => {
    if (!isTrainerMode || !currentUser) {
      return departments
    }

    // If user has specific departments assigned, filter to those
    if (currentUser.departments && Array.isArray(currentUser.departments)) {
      return currentUser.departments
    }

    // If user has a single department_id, find that department
    if (currentUser.department_id) {
      const userDept = departments.find(dept => dept.id === currentUser.department_id)
      return userDept ? [userDept] : departments
    }

    // Fallback to all departments
    return departments
  }

  // Helper function to add a default question if none exist
  const ensureDefaultQuestion = () => {
    if (!form.questions || form.questions.length === 0) {
      const defaultQuestion = {
        id: Date.now(),
        type: 'rating',
        title: 'How would you rate this training session overall?',
        description: '',
        required: true,
        options: [],
        scale: { min: 1, max: 5, labels: ['Poor', 'Excellent'] },
        slider: null,
        validation: null,
        conditional: null,
        order: 0
      }

      setForm(prev => ({
        ...prev,
        questions: [defaultQuestion]
      }))
    }
  }

  const questionTypes = [
    {
      type: 'text',
      label: 'Short Text',
      icon: Type,
      description: 'Single line text input'
    },
    {
      type: 'textarea',
      label: 'Long Text',
      icon: AlignLeft,
      description: 'Multi-line text area'
    },
    {
      type: 'multiple-choice',
      label: 'Multiple Choice',
      icon: List,
      description: 'Select one option'
    },
    {
      type: 'rating',
      label: 'Rating Scale',
      icon: Star,
      description: '1-5 or 1-10 rating'
    },
    {
      type: 'yes-no',
      label: 'Yes/No',
      icon: ToggleLeft,
      description: 'Binary choice'
    },
    {
      type: 'checkbox',
      label: 'Checkboxes',
      icon: CheckSquare,
      description: 'Select multiple options'
    },
    {
      type: 'number',
      label: 'Number',
      icon: Hash,
      description: 'Numeric input'
    },
    {
      type: 'email',
      label: 'Email',
      icon: Mail,
      description: 'Email address input'
    },
    {
      type: 'slider',
      label: 'Slider',
      icon: Sliders,
      description: 'Range slider input'
    }
  ]

  // Form templates for quick start
  const formTemplates = [
    {
      id: 'flight-training',
      name: 'Flight Training Feedback',
      description: 'Standard feedback form for flight training sessions',
      questions: [
        {
          id: 1,
          type: 'rating',
          title: 'How would you rate the overall quality of the training session?',
          description: 'Consider the instructor\'s teaching methods, clarity of explanations, and session organization.',
          required: true,
          scale: { min: 1, max: 5, labels: ['Poor', 'Excellent'] }
        },
        {
          id: 2,
          type: 'multiple-choice',
          title: 'Which aspect of the training was most valuable?',
          required: true,
          options: ['Theoretical explanations', 'Practical demonstrations', 'Hands-on practice', 'Q&A sessions']
        },
        {
          id: 3,
          type: 'textarea',
          title: 'What specific topics would you like more focus on in future sessions?',
          required: false
        },
        {
          id: 4,
          type: 'yes-no',
          title: 'Would you recommend this training to other trainees?',
          required: true
        }
      ]
    },
    {
      id: 'instructor-evaluation',
      name: 'Instructor Evaluation',
      description: 'Comprehensive instructor performance evaluation',
      questions: [
        {
          id: 1,
          type: 'rating',
          title: 'Rate the instructor\'s knowledge of the subject matter',
          required: true,
          scale: { min: 1, max: 10, labels: ['Poor', 'Excellent'] }
        },
        {
          id: 2,
          type: 'rating',
          title: 'Rate the instructor\'s communication skills',
          required: true,
          scale: { min: 1, max: 10, labels: ['Poor', 'Excellent'] }
        },
        {
          id: 3,
          type: 'checkbox',
          title: 'Which teaching methods did the instructor use effectively?',
          required: false,
          options: ['Visual aids', 'Hands-on demonstrations', 'Interactive discussions', 'Real-world examples', 'Practice exercises']
        },
        {
          id: 4,
          type: 'slider',
          title: 'How well did the instructor manage the training pace?',
          required: true,
          slider: { min: 0, max: 100, step: 10, labels: ['Too slow', 'Perfect pace', 'Too fast'] }
        }
      ]
    },
    {
      id: 'course-feedback',
      name: 'Course Feedback',
      description: 'General course evaluation and feedback',
      questions: [
        {
          id: 1,
          type: 'text',
          title: 'Course Name',
          required: true
        },
        {
          id: 2,
          type: 'rating',
          title: 'Overall course satisfaction',
          required: true,
          scale: { min: 1, max: 5, labels: ['Very Dissatisfied', 'Very Satisfied'] }
        },
        {
          id: 3,
          type: 'number',
          title: 'How many hours per week did you spend on this course?',
          required: false,
          validation: { min: 0, max: 168 }
        },
        {
          id: 4,
          type: 'email',
          title: 'Email (optional for follow-up)',
          required: false
        }
      ]
    }
  ]

  useEffect(() => {
    if (initialForm) {
      setForm(prev => ({
        ...prev,
        ...initialForm,
        session: {
          ...prev.session,
          ...(initialForm.session || {})
        },
        settings: {
          ...prev.settings,
          ...(initialForm.settings || {})
        }
      }))
    }
  }, [initialForm])

  const addQuestion = (type) => {
    const newQuestion = {
      id: Date.now(),
      type,
      title: '',
      description: '',
      required: false,
      options: (type === 'multiple-choice' || type === 'checkbox') ? ['Option 1', 'Option 2'] : [],
      scale: type === 'rating' ? { min: 1, max: 5, labels: ['Poor', 'Excellent'] } : null,
      slider: type === 'slider' ? { min: 0, max: 100, step: 1, labels: ['Low', 'High'] } : null,
      validation: type === 'email' ? { email: true } : type === 'number' ? { numeric: true } : null,
      conditional: null,
      order: form.questions.length
    }

    setForm(prev => ({
      ...prev,
      questions: [...prev.questions, newQuestion]
    }))
    setActiveQuestion(newQuestion.id)
  }

  const updateQuestion = (questionId, updates) => {
    setForm(prev => ({
      ...prev,
      questions: prev.questions.map(q => 
        q.id === questionId ? { ...q, ...updates } : q
      )
    }))
  }

  const deleteQuestion = (questionId) => {
    setForm(prev => ({
      ...prev,
      questions: prev.questions.filter(q => q.id !== questionId)
    }))
    setActiveQuestion(null)
  }

  const duplicateQuestion = (questionId) => {
    const question = form.questions.find(q => q.id === questionId)
    if (question) {
      const newQuestion = {
        ...question,
        id: Date.now(),
        title: `${question.title} (Copy)`,
        order: form.questions.length
      }
      setForm(prev => ({
        ...prev,
        questions: [...prev.questions, newQuestion]
      }))
    }
  }

  const moveQuestion = (questionId, direction) => {
    const questions = [...form.questions]
    const index = questions.findIndex(q => q.id === questionId)
    
    if (direction === 'up' && index > 0) {
      [questions[index], questions[index - 1]] = [questions[index - 1], questions[index]]
    } else if (direction === 'down' && index < questions.length - 1) {
      [questions[index], questions[index + 1]] = [questions[index + 1], questions[index]]
    }

    setForm(prev => ({ ...prev, questions }))
  }

  const validateForm = () => {
    const errors = []

    console.log('FormBuilder validation - current form state:', form)

    if (!form.title || !form.title.trim()) {
      errors.push('Form title is required')
    }

    // Check trainer ID - should be a valid number or string
    if (!form.trainerId || form.trainerId === '' || form.trainerId === '0') {
      console.log('Trainer ID validation failed:', form.trainerId)
      errors.push('Trainer ID is required')
    }

    // Check department ID - should be a valid number or string
    if (!form.departmentId || form.departmentId === '' || form.departmentId === '0') {
      console.log('Department ID validation failed:', form.departmentId)
      errors.push('Department ID is required')
    }

    if (!form.session || !form.session.name || !form.session.name.trim()) {
      errors.push('Session name is required')
    }

    if (!form.questions || form.questions.length === 0) {
      errors.push('At least one question is required')
    }

    if (form.questions && form.questions.length > 0) {
      form.questions.forEach((question, index) => {
        if (!question.title || !question.title.trim()) {
          errors.push(`Question ${index + 1}: Title is required`)
        }

        if (question.type === 'multiple-choice' || question.type === 'checkbox') {
          if (!question.options || question.options.length < 2) {
            errors.push(`Question ${index + 1}: At least 2 options are required`)
          }
          if (question.options && question.options.some(opt => !opt || !opt.trim())) {
            errors.push(`Question ${index + 1}: All options must have text`)
          }
        }
      })
    }

    console.log('FormBuilder validation errors:', errors)
    return errors
  }

  const handleSave = () => {
    console.log('FormBuilder handleSave - starting validation')
    console.log('Current form state:', form)

    const errors = validateForm()
    if (errors.length > 0) {
      console.log('Validation failed with errors:', errors)
      alert('Please fix the following errors:\n\n' + errors.join('\n'))
      return
    }

    // Prepare form data with proper structure for backend
    const formData = {
      title: form.title || '',
      description: form.description || '',
      trainerId: form.trainerId ? parseInt(form.trainerId) : null,
      departmentId: form.departmentId ? parseInt(form.departmentId) : null,
      dueDate: form.dueDate || null,
      session: form.session || {
        name: '',
        date: '',
        course: '',
        duration: ''
      },
      questions: form.questions || [],
      settings: form.settings || {
        allowAnonymous: true,
        requireAll: false,
        showProgress: true,
        randomizeQuestions: false
      },
      status: form.status || 'draft'
    }

    console.log('FormBuilder handleSave - prepared form data:', formData)
    if (onSave) {
      onSave(formData)
    }
  }

  const handlePreview = () => {
    if (onPreview) {
      onPreview(form)
    }
  }

  const loadTemplate = (template) => {
    const newForm = {
      ...form,
      title: template.name,
      description: template.description,
      questions: template.questions.map((q, index) => ({
        ...q,
        id: Date.now() + index,
        order: index
      }))
    }
    setForm(newForm)
    setShowTemplates(false)
  }

  return (
    <div className="form-builder">
      <div className="form-builder-header">
        <div className="header-content">
          <h1>Form Builder</h1>
          <div className="header-actions">
            <button className="btn-secondary" onClick={() => setShowSettings(!showSettings)}>
              <Settings size={20} />
              Settings
            </button>
            <button className="btn-secondary" onClick={handlePreview}>
              <Eye size={20} />
              Preview
            </button>
            <button className="btn-primary" onClick={handleSave}>
              <Save size={20} />
              Save Form
            </button>
          </div>
        </div>
      </div>

      <div className="form-builder-content">
        <div className="builder-sidebar">
          <div className="sidebar-section">
            <h3>Form Templates</h3>
            <button
              className="template-toggle-btn"
              onClick={() => setShowTemplates(!showTemplates)}
            >
              {showTemplates ? 'Hide Templates' : 'Show Templates'}
            </button>
            {showTemplates && (
              <div className="form-templates">
                {formTemplates.map((template) => (
                  <motion.button
                    key={template.id}
                    className="template-btn"
                    onClick={() => loadTemplate(template)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div>
                      <span className="template-name">{template.name}</span>
                      <span className="template-description">{template.description}</span>
                    </div>
                  </motion.button>
                ))}
              </div>
            )}
          </div>

          <div className="sidebar-section">
            <h3>Question Types</h3>
            <div className="question-types">
              {questionTypes.map((type) => (
                <motion.button
                  key={type.type}
                  className="question-type-btn"
                  onClick={() => addQuestion(type.type)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  <type.icon size={20} />
                  <div>
                    <span className="type-label">{type.label}</span>
                    <span className="type-description">{type.description}</span>
                  </div>
                </motion.button>
              ))}
            </div>
          </div>

          {showSettings && (
            <motion.div
              className="sidebar-section"
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
            >
              <h3>Form Settings</h3>
              <div className="form-settings">
                <label className="setting-item">
                  <input
                    type="checkbox"
                    checked={form.settings.allowAnonymous}
                    onChange={(e) => setForm(prev => ({
                      ...prev,
                      settings: { ...prev.settings, allowAnonymous: e.target.checked }
                    }))}
                  />
                  Allow Anonymous Responses
                </label>
                <label className="setting-item">
                  <input
                    type="checkbox"
                    checked={form.settings.requireAll}
                    onChange={(e) => setForm(prev => ({
                      ...prev,
                      settings: { ...prev.settings, requireAll: e.target.checked }
                    }))}
                  />
                  Require All Questions
                </label>
                <label className="setting-item">
                  <input
                    type="checkbox"
                    checked={form.settings.showProgress}
                    onChange={(e) => setForm(prev => ({
                      ...prev,
                      settings: { ...prev.settings, showProgress: e.target.checked }
                    }))}
                  />
                  Show Progress Bar
                </label>
                <label className="setting-item">
                  <input
                    type="checkbox"
                    checked={form.settings.randomizeQuestions}
                    onChange={(e) => setForm(prev => ({
                      ...prev,
                      settings: { ...prev.settings, randomizeQuestions: e.target.checked }
                    }))}
                  />
                  Randomize Question Order
                </label>
              </div>
            </motion.div>
          )}
        </div>

        <div className="builder-main">
          <div className="form-header-editor">
            <div className="form-basic-info">
              <input
                type="text"
                placeholder="Form Title"
                value={form.title}
                onChange={(e) => setForm(prev => ({ ...prev, title: e.target.value }))}
                className="form-title-input"
              />
              <textarea
                placeholder="Form Description"
                value={form.description}
                onChange={(e) => setForm(prev => ({ ...prev, description: e.target.value }))}
                className="form-description-input"
                rows={3}
              />

              <div className="field-group">
                <label>
                  <Calendar size={16} />
                  Due Date (Optional)
                </label>
                <DatePicker
                  selected={form.dueDate}
                  onChange={(date) => setForm(prev => ({ ...prev, dueDate: date }))}
                  showTimeSelect
                  timeFormat="HH:mm"
                  timeIntervals={15}
                  dateFormat="MMMM d, yyyy h:mm aa"
                  placeholderText="Select due date and time"
                  className="form-due-date-input"
                  minDate={new Date(Date.now() + 24 * 60 * 60 * 1000)} // 24 hours from now
                  isClearable
                />
                <small className="field-hint">
                  Forms must be due at least 24 hours in the future
                </small>
              </div>
            </div>

            <div className="form-association">
              <div className="association-row">
                <div className="field-group">
                  <label>
                    <Users size={16} />
                    Trainer ID
                    {isTrainerMode && <span className="read-only-indicator">(Auto-assigned)</span>}
                  </label>
                  {isTrainerMode ? (
                    <input
                      type="text"
                      value={`${form.trainerId || currentUser?.id || ''} - ${currentUser?.first_name || ''} ${currentUser?.last_name || ''}`}
                      readOnly
                      className="form-input read-only"
                      title="Trainer ID is automatically assigned and cannot be changed"
                    />
                  ) : (
                    <select
                      value={form.trainerId || ''}
                      onChange={(e) => {
                        console.log('Trainer ID changed to:', e.target.value)
                        setForm(prev => ({ ...prev, trainerId: e.target.value }))
                      }}
                    >
                      <option value="">Select Trainer ID</option>
                      {trainers.map(trainer => (
                        <option key={trainer.id} value={trainer.id}>
                          {trainer.id} - {trainer.first_name} {trainer.last_name} ({trainer.department_name || 'No Department'})
                        </option>
                      ))}
                    </select>
                  )}
                </div>
                <div className="field-group">
                  <label>
                    <BookOpen size={16} />
                    Department ID
                    {isTrainerMode && (
                      <span className="read-only-indicator">
                        {getTrainerDepartments().length === 1 ? '(Auto-assigned)' : '(Select from your departments)'}
                      </span>
                    )}
                  </label>
                  {isTrainerMode ? (
                    getTrainerDepartments().length === 1 ? (
                      // Single department - read-only
                      <input
                        type="text"
                        value={`${getTrainerDepartments()[0].id} - ${getTrainerDepartments()[0].name}`}
                        readOnly
                        className="form-input read-only"
                        title="Department is automatically assigned based on your profile"
                      />
                    ) : (
                      // Multiple departments - dropdown selector
                      <select
                        value={form.departmentId || ''}
                        onChange={(e) => {
                          console.log('Department ID changed to:', e.target.value)
                          setForm(prev => ({ ...prev, departmentId: e.target.value }))
                        }}
                        className="form-select"
                      >
                        <option value="">Select Department</option>
                        {getTrainerDepartments().map(department => (
                          <option key={department.id} value={department.id}>
                            {department.id} - {department.name}
                          </option>
                        ))}
                      </select>
                    )
                  ) : (
                    <select
                      value={form.departmentId || ''}
                      onChange={(e) => {
                        console.log('Department ID changed to:', e.target.value)
                        setForm(prev => ({ ...prev, departmentId: e.target.value }))
                      }}
                    >
                      <option value="">Select Department ID</option>
                      {departments.map(department => (
                        <option key={department.id} value={department.id}>
                          {department.id} - {department.name}
                        </option>
                      ))}
                    </select>
                  )}
                </div>
              </div>

              <div className="association-row">
                <div className="field-group">
                  <label>
                    <BookOpen size={16} />
                    Session Name
                  </label>
                  <input
                    type="text"
                    placeholder="Training Session Name"
                    value={form.session.name}
                    onChange={(e) => setForm(prev => ({
                      ...prev,
                      session: { ...prev.session, name: e.target.value }
                    }))}
                  />
                </div>
                <div className="field-group">
                  <label>
                    <Calendar size={16} />
                    Session Date
                  </label>
                  <input
                    type="date"
                    value={form.session.date}
                    onChange={(e) => setForm(prev => ({
                      ...prev,
                      session: { ...prev.session, date: e.target.value }
                    }))}
                  />
                </div>
              </div>

              <div className="association-row">
                <div className="field-group">
                  <label>Course Name</label>
                  <input
                    type="text"
                    placeholder="Course Name"
                    value={form.session.course}
                    onChange={(e) => setForm(prev => ({
                      ...prev,
                      session: { ...prev.session, course: e.target.value }
                    }))}
                  />
                </div>
                <div className="field-group">
                  <label>Duration (minutes)</label>
                  <input
                    type="number"
                    placeholder="60"
                    min="1"
                    max="480"
                    value={form.session.duration}
                    onChange={(e) => setForm(prev => ({
                      ...prev,
                      session: { ...prev.session, duration: e.target.value }
                    }))}
                  />
                </div>
              </div>
            </div>
          </div>

          <div className="questions-container">
            <h3>Questions ({form.questions.length})</h3>
            
            {form.questions.length === 0 ? (
              <div className="empty-state">
                <Plus size={48} />
                <h4>No questions yet</h4>
                <p>Add your first question using the question types on the left</p>
                <button
                  className="add-default-question-btn"
                  onClick={ensureDefaultQuestion}
                  style={{
                    marginTop: '1rem',
                    padding: '0.5rem 1rem',
                    backgroundColor: '#007bff',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Add Default Question
                </button>
              </div>
            ) : (
              <div className="questions-list">
                <AnimatePresence>
                  {form.questions.map((question, index) => (
                    <QuestionEditor
                      key={question.id}
                      question={question}
                      index={index}
                      isActive={activeQuestion === question.id}
                      onUpdate={(updates) => updateQuestion(question.id, updates)}
                      onDelete={() => deleteQuestion(question.id)}
                      onDuplicate={() => duplicateQuestion(question.id)}
                      onMove={(direction) => moveQuestion(question.id, direction)}
                      onActivate={() => setActiveQuestion(question.id)}
                      canMoveUp={index > 0}
                      canMoveDown={index < form.questions.length - 1}
                    />
                  ))}
                </AnimatePresence>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default FormBuilder
