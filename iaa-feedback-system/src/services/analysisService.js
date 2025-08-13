/**
 * Analysis Service for Trainer Response Analysis
 * Implements NLP-based text analysis and weighted scoring for numeric responses
 */

class AnalysisService {
  constructor() {
    this.sentimentKeywords = {
      positive: [
        'excellent', 'great', 'amazing', 'wonderful', 'fantastic', 'outstanding',
        'good', 'nice', 'helpful', 'clear', 'effective', 'engaging', 'informative',
        'professional', 'knowledgeable', 'patient', 'thorough', 'well-organized',
        'inspiring', 'motivating', 'supportive', 'friendly', 'approachable'
      ],
      negative: [
        'terrible', 'awful', 'bad', 'poor', 'disappointing', 'confusing',
        'unclear', 'boring', 'ineffective', 'unprofessional', 'rude', 'impatient',
        'disorganized', 'unhelpful', 'difficult', 'frustrating', 'waste',
        'useless', 'inadequate', 'insufficient', 'lacking'
      ],
      neutral: [
        'okay', 'average', 'normal', 'standard', 'typical', 'regular',
        'moderate', 'fair', 'acceptable', 'adequate'
      ]
    }

    this.topicKeywords = {
      'Teaching Quality': [
        'teaching', 'instruction', 'explanation', 'clarity', 'understanding',
        'knowledge', 'expertise', 'methodology', 'approach', 'delivery'
      ],
      'Communication': [
        'communication', 'speaking', 'listening', 'interaction', 'discussion',
        'questions', 'answers', 'feedback', 'response', 'dialogue'
      ],
      'Course Content': [
        'content', 'material', 'curriculum', 'topics', 'subjects', 'lessons',
        'modules', 'chapters', 'information', 'knowledge', 'theory', 'practical'
      ],
      'Organization': [
        'organization', 'structure', 'planning', 'schedule', 'time', 'management',
        'preparation', 'arrangement', 'order', 'sequence'
      ],
      'Engagement': [
        'engagement', 'participation', 'interaction', 'involvement', 'activity',
        'discussion', 'collaboration', 'teamwork', 'group', 'active'
      ]
    }

    this.emojiSentiments = {
      positive: ['😊', '😃', '😄', '😁', '🙂', '👍', '👏', '🎉', '✅', '💯', '🌟', '⭐'],
      negative: ['😞', '😢', '😠', '😡', '👎', '❌', '😔', '😕', '🙁', '😤', '😒'],
      neutral: ['😐', '😑', '🤔', '😶', '🙄', '😯', '😮']
    }
  }

  /**
   * Analyze text responses using NLP techniques
   */
  analyzeTextResponse(text) {
    if (!text || typeof text !== 'string') {
      return {
        sentiment: { score: 0, label: 'neutral', confidence: 0 },
        topics: [],
        emojiAnalysis: { entropy: 0, sentiment: 'neutral', count: 0 },
        wordCount: 0,
        readabilityScore: 0
      }
    }

    const cleanText = text.toLowerCase().trim()
    const words = cleanText.split(/\s+/).filter(word => word.length > 0)

    return {
      sentiment: this.analyzeSentiment(cleanText),
      topics: this.extractTopics(cleanText),
      emojiAnalysis: this.analyzeEmojis(text),
      wordCount: words.length,
      readabilityScore: this.calculateReadability(words),
      keyPhrases: this.extractKeyPhrases(cleanText)
    }
  }

  /**
   * Sentiment analysis using keyword matching and scoring
   */
  analyzeSentiment(text) {
    let positiveScore = 0
    let negativeScore = 0
    let neutralScore = 0

    // Count sentiment keywords
    this.sentimentKeywords.positive.forEach(keyword => {
      const matches = (text.match(new RegExp(keyword, 'gi')) || []).length
      positiveScore += matches
    })

    this.sentimentKeywords.negative.forEach(keyword => {
      const matches = (text.match(new RegExp(keyword, 'gi')) || []).length
      negativeScore += matches
    })

    this.sentimentKeywords.neutral.forEach(keyword => {
      const matches = (text.match(new RegExp(keyword, 'gi')) || []).length
      neutralScore += matches
    })

    // Calculate overall sentiment
    const totalScore = positiveScore + negativeScore + neutralScore
    if (totalScore === 0) {
      return { score: 0, label: 'neutral', confidence: 0 }
    }

    const positiveRatio = positiveScore / totalScore
    const negativeRatio = negativeScore / totalScore

    let sentiment, score, confidence

    if (positiveScore > negativeScore && positiveScore > neutralScore) {
      sentiment = 'positive'
      score = positiveRatio
      confidence = Math.min(0.9, positiveRatio * 1.2)
    } else if (negativeScore > positiveScore && negativeScore > neutralScore) {
      sentiment = 'negative'
      score = -negativeRatio
      confidence = Math.min(0.9, negativeRatio * 1.2)
    } else {
      sentiment = 'neutral'
      score = 0
      confidence = Math.max(0.3, neutralScore / totalScore)
    }

    return { score, label: sentiment, confidence }
  }

  /**
   * Extract topics from text using keyword matching
   */
  extractTopics(text) {
    const topics = []

    Object.entries(this.topicKeywords).forEach(([topic, keywords]) => {
      let score = 0
      keywords.forEach(keyword => {
        const matches = (text.match(new RegExp(keyword, 'gi')) || []).length
        score += matches
      })

      if (score > 0) {
        topics.push({
          topic,
          relevance: Math.min(1, score / 3), // Normalize to 0-1
          mentions: score
        })
      }
    })

    return topics.sort((a, b) => b.relevance - a.relevance)
  }

  /**
   * Analyze emoji usage and calculate entropy
   */
  analyzeEmojis(text) {
    const emojiRegex = /[\u{1F600}-\u{1F64F}]|[\u{1F300}-\u{1F5FF}]|[\u{1F680}-\u{1F6FF}]|[\u{1F1E0}-\u{1F1FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu
    const emojis = text.match(emojiRegex) || []

    if (emojis.length === 0) {
      return { entropy: 0, sentiment: 'neutral', count: 0, diversity: 0 }
    }

    // Calculate emoji diversity (entropy)
    const emojiCounts = {}
    emojis.forEach(emoji => {
      emojiCounts[emoji] = (emojiCounts[emoji] || 0) + 1
    })

    const uniqueEmojis = Object.keys(emojiCounts).length
    const entropy = uniqueEmojis > 1 ? 
      -Object.values(emojiCounts).reduce((sum, count) => {
        const p = count / emojis.length
        return sum + p * Math.log2(p)
      }, 0) : 0

    // Determine emoji sentiment
    let positiveCount = 0, negativeCount = 0, neutralCount = 0

    emojis.forEach(emoji => {
      if (this.emojiSentiments.positive.includes(emoji)) positiveCount++
      else if (this.emojiSentiments.negative.includes(emoji)) negativeCount++
      else neutralCount++
    })

    let sentiment = 'neutral'
    if (positiveCount > negativeCount && positiveCount > neutralCount) sentiment = 'positive'
    else if (negativeCount > positiveCount && negativeCount > neutralCount) sentiment = 'negative'

    return {
      entropy: entropy,
      sentiment: sentiment,
      count: emojis.length,
      diversity: uniqueEmojis / emojis.length,
      breakdown: { positive: positiveCount, negative: negativeCount, neutral: neutralCount }
    }
  }

  /**
   * Calculate readability score (simplified)
   */
  calculateReadability(words) {
    if (words.length === 0) return 0

    const avgWordLength = words.reduce((sum, word) => sum + word.length, 0) / words.length
    const sentences = words.join(' ').split(/[.!?]+/).filter(s => s.trim().length > 0).length
    const avgSentenceLength = words.length / Math.max(1, sentences)

    // Simplified readability score (0-100, higher = more readable)
    const score = Math.max(0, Math.min(100, 
      100 - (avgWordLength * 5) - (avgSentenceLength * 2)
    ))

    return Math.round(score)
  }

  /**
   * Extract key phrases from text
   */
  extractKeyPhrases(text) {
    const words = text.split(/\s+/).filter(word => 
      word.length > 3 && 
      !['the', 'and', 'but', 'for', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 'will', 'would', 'could', 'should'].includes(word)
    )

    const phrases = []
    for (let i = 0; i < words.length - 1; i++) {
      const phrase = `${words[i]} ${words[i + 1]}`
      phrases.push(phrase)
    }

    // Count phrase frequency
    const phraseCounts = {}
    phrases.forEach(phrase => {
      phraseCounts[phrase] = (phraseCounts[phrase] || 0) + 1
    })

    return Object.entries(phraseCounts)
      .filter(([phrase, count]) => count > 1)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 5)
      .map(([phrase, count]) => ({ phrase, frequency: count }))
  }

  /**
   * Analyze numeric responses with weighted scoring
   */
  analyzeNumericResponse(value, type, weight = 1) {
    if (value === null || value === undefined || isNaN(value)) {
      return { score: 0, normalizedScore: 0, weight, type }
    }

    const numValue = parseFloat(value)
    let normalizedScore = 0

    switch (type) {
      case 'rating':
        // Assume rating is 1-5 or 1-10, normalize to 0-1
        const maxRating = numValue <= 5 ? 5 : 10
        normalizedScore = (numValue - 1) / (maxRating - 1)
        break
      case 'percentage':
        normalizedScore = numValue / 100
        break
      case 'scale':
        // Assume scale is 1-10, normalize to 0-1
        normalizedScore = (numValue - 1) / 9
        break
      default:
        normalizedScore = Math.min(1, Math.max(0, numValue / 10))
    }

    return {
      score: numValue,
      normalizedScore: Math.max(0, Math.min(1, normalizedScore)),
      weight,
      type,
      weightedScore: normalizedScore * weight
    }
  }

  /**
   * Calculate overall trainer score from all responses
   */
  calculateTrainerScore(responses) {
    if (!responses || responses.length === 0) {
      return {
        overallScore: 0,
        breakdown: {
          numeric: { score: 0, weight: 0.6 },
          sentiment: { score: 0, weight: 0.3 },
          engagement: { score: 0, weight: 0.1 }
        },
        totalResponses: 0,
        confidence: 0
      }
    }

    let numericScores = []
    let sentimentScores = []
    let engagementScores = []

    responses.forEach(response => {
      if (response.responses) {
        response.responses.forEach(item => {
          if (item.type === 'rating' || item.type === 'number') {
            const analysis = this.analyzeNumericResponse(item.value, 'rating')
            numericScores.push(analysis.normalizedScore)
          } else if (item.type === 'text' || item.type === 'textarea') {
            const textAnalysis = this.analyzeTextResponse(item.value)
            sentimentScores.push((textAnalysis.sentiment.score + 1) / 2) // Convert -1,1 to 0,1
            engagementScores.push(Math.min(1, textAnalysis.wordCount / 50)) // Engagement based on response length
          }
        })
      }
    })

    // Calculate weighted averages
    const avgNumeric = numericScores.length > 0 ? 
      numericScores.reduce((sum, score) => sum + score, 0) / numericScores.length : 0.5

    const avgSentiment = sentimentScores.length > 0 ? 
      sentimentScores.reduce((sum, score) => sum + score, 0) / sentimentScores.length : 0.5

    const avgEngagement = engagementScores.length > 0 ? 
      engagementScores.reduce((sum, score) => sum + score, 0) / engagementScores.length : 0.5

    // Weighted final score
    const weights = { numeric: 0.6, sentiment: 0.3, engagement: 0.1 }
    const overallScore = (avgNumeric * weights.numeric) + 
                        (avgSentiment * weights.sentiment) + 
                        (avgEngagement * weights.engagement)

    const confidence = Math.min(1, responses.length / 10) // Higher confidence with more responses

    return {
      overallScore: Math.round(overallScore * 100), // Convert to 0-100 scale
      breakdown: {
        numeric: { score: Math.round(avgNumeric * 100), weight: weights.numeric },
        sentiment: { score: Math.round(avgSentiment * 100), weight: weights.sentiment },
        engagement: { score: Math.round(avgEngagement * 100), weight: weights.engagement }
      },
      totalResponses: responses.length,
      confidence: Math.round(confidence * 100),
      dataPoints: {
        numericCount: numericScores.length,
        textCount: sentimentScores.length,
        avgResponseLength: sentimentScores.length > 0 ? 
          responses.reduce((sum, r) => sum + (r.responses?.filter(i => i.type === 'text').length || 0), 0) / responses.length : 0
      }
    }
  }
}

export default new AnalysisService()
