import axios from 'axios'

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',  // Use environment variable or fallback to proxy
  timeout: 60000, // Increased timeout for slower connections
  withCredentials: true, // Important for session-based authentication
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache'
  },
  validateStatus: function (status) {
    return status >= 200 && status < 500; // Don't reject if status is not 2xx
  },
  // Add retry configuration
  retry: 3,
  retryDelay: (retryCount) => {
    return retryCount * 1000; // time interval between retries
  }
})

// Request interceptor for adding common headers
api.interceptors.request.use(
  (config) => {
    // Add any common headers here if needed
    return config
  },
  (error) => {
    return Promise.reject(error)
  }
)

// Response interceptor for handling common errors
api.interceptors.response.use(
  (response) => {
    return response
  },
  async (error) => {
    const originalRequest = error.config;
    
    // Handle network errors
    if (!error.response) {
      console.error('Network error detected');
      return Promise.reject(new Error('Network error. Please check your connection.'));
    }

    // Handle unauthorized errors
    if (error.response.status === 401) {
      console.log('Unauthorized access');
      // Optionally redirect to login page or refresh token
    }

    // Retry failed requests (if retry configuration exists)
    if (originalRequest.retry) {
      if (!originalRequest._retry) {
        originalRequest._retry = 0;
      }

      if (originalRequest._retry < originalRequest.retry) {
        originalRequest._retry++;
        
        // Wait before retrying
        await new Promise(resolve => 
          setTimeout(resolve, originalRequest.retryDelay(originalRequest._retry))
        );
        
        // Retry the request
        return api(originalRequest);
      }
    }

    // Handle timeout errors
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timed out. Please try again.'));
    }

    // Handle server errors
    if (error.response.status >= 500) {
      return Promise.reject(new Error('Server error. Please try again later.'));
    }

    return Promise.reject(error);
  }
)

// Stock API endpoints
export const stockApi = {
  search: (query, limit = 10) => 
    api.get(`/search`, { params: { q: query, limit } }),

  getDetails: (symbol) => 
    api.get(`/stock/${symbol}`),

  getHistory: (symbol, period = '1y', interval = '1d') => 
    api.get(`/stock/${symbol}/history`, { params: { period, interval } }),

  compare: (symbols, period = '1y', interval = '1d') => 
    api.post('/stocks/compare', { symbols, period, interval }),

  getTrending: (count = 10) => 
    api.get('/trending', { params: { count } }),

  getMarketSummary: () => 
    api.get('/market/summary'),

  getTopGainers: (count = 10) => 
    api.get('/market/gainers', { params: { count } }),

  getTopLosers: (count = 10) => 
    api.get('/market/losers', { params: { count } }),

  getMarketMovers: (count = 10) => 
    api.get('/market/movers', { params: { count } }),
}

// Authentication API endpoints
export const authApi = {
  login: (credentials) => 
    api.post('/auth/login', credentials),

  register: (userData) => 
    api.post('/auth/register', userData),

  logout: () => 
    api.post('/auth/logout'),

  checkAuth: () => 
    api.get('/auth/check'),

  getProfile: () => 
    api.get('/auth/profile'),

  updateProfile: (profileData) => 
    api.put('/auth/profile', profileData),
}

// Portfolio API endpoints
export const portfolioApi = {
  getHoldings: () => 
    api.get('/portfolio/holdings'),

  addHolding: (holdingData) => 
    api.post('/portfolio/holdings', holdingData),

  updateHolding: (holdingId, holdingData) => 
    api.put(`/portfolio/holdings/${holdingId}`, holdingData),

  deleteHolding: (holdingId) => 
    api.delete(`/portfolio/holdings/${holdingId}`),

  getWatchlist: () => 
    api.get('/portfolio/watchlist'),

  addToWatchlist: (stockData) => 
    api.post('/portfolio/watchlist', stockData),

  removeFromWatchlist: (itemId) => 
    api.delete(`/portfolio/watchlist/${itemId}`),
}

// Prediction API endpoints
export const predictionApi = {
  getPrediction: (symbol, options = {}) => {
    const { days = 7, model = 'arima', confidence = 95 } = options;
    return api.get(`/predict/${symbol}`, { params: { days, model, confidence } });
  },

  getAvailableModels: () => 
    api.get('/predict/models'),

  getBatchPredictions: (symbols, days = 7, model = 'arima') => 
    api.post('/predict/batch', { symbols, days, model }),

  getAccuracy: (symbol, model = 'arima') => 
    api.get(`/predict/accuracy/${symbol}`, { params: { model } }),
}

// News API endpoints
export const newsApi = {
  getMarketNews: (limit = 10) => 
    api.get('/news/market', { params: { limit } }),

  getStockNews: (symbol, limit = 10) => 
    api.get(`/news/stock/${symbol}`, { params: { limit } }),

  getTrendingNews: (limit = 15) => 
    api.get('/news/trending', { params: { limit } }),

  getSectorNews: (limitPerSector = 3) => 
    api.get('/news/sectors', { params: { limit_per_sector: limitPerSector } }),
}

// AI API endpoints
export const aiApi = {
  search: (query, limit = 10) =>
    api.get('/ai/search', { params: { q: query, limit } }),
    
  getModelsStatus: () =>
    api.get('/ai/models/status'),
    
  analyze: (symbol, days = 7, model = 'auto') =>
    api.get(`/ai/analyze/${symbol}`, { params: { days, model } }),
    
  compareModels: (symbol, days = 7) =>
    api.get(`/ai/compare-models/${symbol}`, { 
      params: { days },
      timeout: 45000 // 45 seconds timeout for model comparison
    }),
}

export default api
