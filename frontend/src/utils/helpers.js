/**
 * Format a number as currency
 * @param {number} value - The value to format
 * @param {string} currency - The currency code (default: 'USD')
 * @returns {string} - Formatted currency string
 */
export const formatCurrency = (value, currency = 'USD') => {
  if (value === null || value === undefined || isNaN(value)) {
    return '$0.00'
  }

  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

/**
 * Format a large number with appropriate suffixes (K, M, B, T)
 * @param {number} value - The value to format
 * @returns {string} - Formatted number string
 */
export const formatLargeNumber = (value) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '0'
  }

  const absValue = Math.abs(value)
  const sign = value < 0 ? '-' : ''

  if (absValue >= 1e12) {
    return `${sign}${(absValue / 1e12).toFixed(2)}T`
  } else if (absValue >= 1e9) {
    return `${sign}${(absValue / 1e9).toFixed(2)}B`
  } else if (absValue >= 1e6) {
    return `${sign}${(absValue / 1e6).toFixed(2)}M`
  } else if (absValue >= 1e3) {
    return `${sign}${(absValue / 1e3).toFixed(2)}K`
  } else {
    return `${sign}${absValue.toFixed(2)}`
  }
}

/**
 * Format a percentage value
 * @param {number} value - The percentage value
 * @param {number} decimals - Number of decimal places (default: 2)
 * @returns {string} - Formatted percentage string
 */
export const formatPercentage = (value, decimals = 2) => {
  if (value === null || value === undefined || isNaN(value)) {
    return '0.00%'
  }

  return `${value.toFixed(decimals)}%`
}

/**
 * Format a date string
 * @param {string|Date} date - The date to format
 * @param {string} format - Format type ('short', 'long', 'time')
 * @returns {string} - Formatted date string
 */
export const formatDate = (date, format = 'short') => {
  if (!date) return ''

  const dateObj = typeof date === 'string' ? new Date(date) : date

  if (isNaN(dateObj.getTime())) {
    return 'Invalid Date'
  }

  const options = {
    short: {
      year: 'numeric',
      month: 'short',
      day: 'numeric'
    },
    long: {
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    },
    time: {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    }
  }

  return dateObj.toLocaleDateString('en-US', options[format] || options.short)
}

/**
 * Get the appropriate CSS class for price changes
 * @param {number} change - The price change value
 * @returns {string} - CSS class name
 */
export const getPriceChangeClass = (change) => {
  if (change > 0) return 'price-positive'
  if (change < 0) return 'price-negative'
  return 'price-neutral'
}

/**
 * Get the appropriate icon for price changes
 * @param {number} change - The price change value
 * @returns {string} - Unicode arrow character
 */
export const getPriceChangeIcon = (change) => {
  if (change > 0) return '↗'
  if (change < 0) return '↘'
  return '→'
}

/**
 * Debounce function to limit the rate of function calls
 * @param {Function} func - The function to debounce
 * @param {number} delay - The delay in milliseconds
 * @returns {Function} - Debounced function
 */
export const debounce = (func, delay) => {
  let timeoutId
  return (...args) => {
    clearTimeout(timeoutId)
    timeoutId = setTimeout(() => func.apply(null, args), delay)
  }
}

/**
 * Calculate the relative time from now
 * @param {string|Date} date - The date to compare
 * @returns {string} - Relative time string
 */
export const getRelativeTime = (date) => {
  if (!date) return ''

  const now = new Date()
  const dateObj = typeof date === 'string' ? new Date(date) : date
  const diffMs = now - dateObj
  const diffSecs = Math.round(diffMs / 1000)
  const diffMins = Math.round(diffSecs / 60)
  const diffHours = Math.round(diffMins / 60)
  const diffDays = Math.round(diffHours / 24)

  if (diffSecs < 60) {
    return 'just now'
  } else if (diffMins < 60) {
    return `${diffMins} minute${diffMins === 1 ? '' : 's'} ago`
  } else if (diffHours < 24) {
    return `${diffHours} hour${diffHours === 1 ? '' : 's'} ago`
  } else if (diffDays < 30) {
    return `${diffDays} day${diffDays === 1 ? '' : 's'} ago`
  } else {
    return formatDate(dateObj)
  }
}

/**
 * Validate email format
 * @param {string} email - Email to validate
 * @returns {boolean} - True if valid email format
 */
export const isValidEmail = (email) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
  return emailRegex.test(email)
}

/**
 * Generate a random color for charts
 * @param {number} alpha - Alpha transparency (0-1)
 * @returns {string} - RGBA color string
 */
export const generateRandomColor = (alpha = 1) => {
  const r = Math.floor(Math.random() * 255)
  const g = Math.floor(Math.random() * 255)
  const b = Math.floor(Math.random() * 255)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

/**
 * Get predefined colors for charts
 * @param {number} index - Index of the color
 * @param {number} alpha - Alpha transparency (0-1)
 * @returns {string} - RGBA color string
 */
export const getChartColor = (index, alpha = 1) => {
  const colors = [
    [59, 130, 246],   // Blue
    [16, 185, 129],   // Green
    [239, 68, 68],    // Red
    [245, 158, 11],   // Yellow
    [139, 92, 246],   // Purple
    [236, 72, 153],   // Pink
    [20, 184, 166],   // Teal
    [251, 146, 60],   // Orange
  ]

  const colorIndex = index % colors.length
  const [r, g, b] = colors[colorIndex]
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

/**
 * Truncate text to specified length
 * @param {string} text - Text to truncate
 * @param {number} maxLength - Maximum length
 * @returns {string} - Truncated text
 */
export const truncateText = (text, maxLength) => {
  if (!text || text.length <= maxLength) return text
  return text.substring(0, maxLength) + '...'
}

/**
 * Convert object to query string
 * @param {Object} params - Parameters object
 * @returns {string} - Query string
 */
export const toQueryString = (params) => {
  return Object.keys(params)
    .filter(key => params[key] !== undefined && params[key] !== null)
    .map(key => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
    .join('&')
}
