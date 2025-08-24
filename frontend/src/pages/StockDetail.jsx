import React, { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery } from 'react-query'
import { 
  ArrowUpRight, 
  ArrowDownRight, 
  TrendingUp, 
  BarChart3, 
  Eye, 
  EyeOff, 
  Plus,
  Star,
  StarOff,
  Activity,
  Calendar,
  DollarSign,
  Percent,
  Volume,
  Users,
  Globe,
  Building,
  RefreshCw
} from 'lucide-react'
import { stockApi, portfolioApi, predictionApi } from '../services/api'
import StockChart from '../components/StockChart'
import LoadingSpinner from '../components/LoadingSpinner'
import AddInvestmentModal from '../components/AddInvestmentModal'
import { useAuth } from '../context/AuthContext'
import toast from 'react-hot-toast'

const StockDetail = () => {
  const { symbol } = useParams()
  const navigate = useNavigate()
  const { user } = useAuth()
  const [timeframe, setTimeframe] = useState('1y')
  const [showVolume, setShowVolume] = useState(false)
  const [isInWatchlist, setIsInWatchlist] = useState(false)
  const [showAddModal, setShowAddModal] = useState(false)

  // Redirect if symbol is undefined or invalid
  useEffect(() => {
    if (!symbol || symbol === 'undefined' || symbol.trim() === '') {
      console.error('Invalid stock symbol:', symbol)
      navigate('/', { replace: true })
      return
    }
  }, [symbol, navigate])

  // Fetch stock details
  const { 
    data: stockData, 
    isLoading: stockLoading, 
    error: stockError,
    refetch: refetchStock 
  } = useQuery(
    ['stock', symbol],
    async () => {
      const response = await stockApi.getDetails(symbol)
      return response.data
    },
    {
      enabled: !!symbol,
      retry: 2,
      staleTime: 60000, // 1 minute
    }
  )

  // Fetch historical data
  const { 
    data: historyData, 
    isLoading: historyLoading,
    error: historyError 
  } = useQuery(
    ['stockHistory', symbol, timeframe],
    async () => {
      const response = await stockApi.getHistory(symbol, timeframe, '1d')
      return response.data
    },
    {
      enabled: !!symbol,
      retry: 2,
      staleTime: 300000, // 5 minutes
    }
  )

  // Fetch predictions with ARIMA as default per specification
  const { 
    data: predictionsData, 
    isLoading: predictionsLoading 
  } = useQuery(
    ['predictions', symbol],
    async () => {
      // Request ARIMA model specifically for 7-day forecast per requirements
      const response = await predictionApi.getPrediction(symbol, { 
        days: 7, 
        model: 'arima', 
        confidence: 95 
      })
      return response.data
    },
    {
      enabled: !!symbol,
      retry: 1,
      staleTime: 3600000, // 1 hour
    }
  )

  // Check if stock is in watchlist
  useEffect(() => {
    if (user && symbol) {
      portfolioApi.getWatchlist()
        .then(response => {
          const watchlist = response.data.watchlist || []
          const watchlistItem = watchlist.find(item => item.symbol === symbol || item.ticker === symbol)
          setIsInWatchlist(!!watchlistItem)
        })
        .catch(err => console.log('Watchlist check failed:', err))
    }
  }, [user, symbol])

  const handleWatchlistToggle = async () => {
    if (!user) {
      toast.error('Please login to manage your watchlist')
      navigate('/login')
      return
    }

    try {
      if (isInWatchlist) {
        // First, get the watchlist to find the item ID
        const response = await portfolioApi.getWatchlist()
        const watchlist = response.data.watchlist || []
        const watchlistItem = watchlist.find(item => item.symbol === symbol || item.ticker === symbol)
        
        if (watchlistItem) {
          await portfolioApi.removeFromWatchlist(watchlistItem.id)
          toast.success('Removed from watchlist')
        } else {
          toast.error('Stock not found in watchlist')
        }
      } else {
        await portfolioApi.addToWatchlist({ symbol })
        toast.success('Added to watchlist')
      }
      setIsInWatchlist(!isInWatchlist)
    } catch (error) {
      toast.error('Failed to update watchlist')
      console.error('Watchlist error:', error)
    }
  }

  const handleAddToPortfolio = () => {
    if (!user) {
      toast.error('Please login to manage your portfolio')
      navigate('/login')
      return
    }
    setShowAddModal(true)
  }

  const formatPrice = (price) => {
    if (!price) return '$0.00'
    return `$${parseFloat(price).toFixed(2)}`
  }

  const formatPercentage = (percent) => {
    if (!percent) return '0.00%'
    const sign = percent >= 0 ? '+' : ''
    return `${sign}${parseFloat(percent).toFixed(2)}%`
  }

  const formatLargeNumber = (num) => {
    if (!num || num === 0) return 'N/A'
    
    if (num >= 1e12) {
      return `$${(num / 1e12).toFixed(2)}T`
    } else if (num >= 1e9) {
      return `$${(num / 1e9).toFixed(2)}B`
    } else if (num >= 1e6) {
      return `$${(num / 1e6).toFixed(2)}M`
    } else if (num >= 1e3) {
      return `$${(num / 1e3).toFixed(2)}K`
    }
    
    return `$${num.toLocaleString()}`
  }

  const formatVolume = (volume) => {
    if (!volume || volume === 0) return 'N/A'
    
    if (volume >= 1e9) {
      return `${(volume / 1e9).toFixed(2)}B`
    } else if (volume >= 1e6) {
      return `${(volume / 1e6).toFixed(2)}M`
    } else if (volume >= 1e3) {
      return `${(volume / 1e3).toFixed(2)}K`
    }
    
    return volume.toLocaleString()
  }

  const getChangeColor = (change) => {
    if (change > 0) return 'text-green-600'
    if (change < 0) return 'text-red-600'
    return 'text-gray-600'
  }

  const getChangeIcon = (change) => {
    if (change > 0) return <ArrowUpRight className="h-4 w-4" />
    if (change < 0) return <ArrowDownRight className="h-4 w-4" />
    return null
  }

  const timeframes = [
    { key: '1d', label: '1D' },
    { key: '5d', label: '5D' },
    { key: '1mo', label: '1M' },
    { key: '3mo', label: '3M' },
    { key: '6mo', label: '6M' },
    { key: '1y', label: '1Y' },
    { key: '2y', label: '2Y' },
    { key: '5y', label: '5Y' }
  ]

  if (stockLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="large" />
      </div>
    )
  }

  if (stockError || !stockData?.data) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-8 text-center">
          <h1 className="text-2xl font-bold text-red-800 mb-2">Stock Not Found</h1>
          <p className="text-red-600 mb-4">
            Unable to find data for symbol "{symbol}". Please check the symbol and try again.
          </p>
          <button
            onClick={() => navigate('/')}
            className="bg-red-600 text-white px-6 py-2 rounded-lg hover:bg-red-700 transition-colors"
          >
            Back to Home
          </button>
        </div>
      </div>
    )
  }

  const stock = stockData.data

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <div className="flex items-center space-x-3 mb-2">
              <h1 className="text-3xl font-bold text-gray-900">{stock.symbol}</h1>
              <button
                onClick={handleWatchlistToggle}
                className={`p-2 rounded-full transition-colors ${
                  isInWatchlist 
                    ? 'text-yellow-500 hover:text-yellow-600 bg-yellow-50 hover:bg-yellow-100' 
                    : 'text-gray-400 hover:text-yellow-500 bg-gray-50 hover:bg-yellow-50'
                }`}
                title={isInWatchlist ? 'Remove from watchlist' : 'Add to watchlist'}
              >
                {isInWatchlist ? <Star className="h-5 w-5 fill-current" /> : <StarOff className="h-5 w-5" />}
              </button>
              <button
                onClick={refetchStock}
                className="p-2 rounded-full text-gray-400 hover:text-blue-500 bg-gray-50 hover:bg-blue-50 transition-colors"
                title="Refresh data"
              >
                <RefreshCw className="h-5 w-5" />
              </button>
            </div>
            <h2 className="text-xl text-gray-600 mb-2">{stock.name}</h2>
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              {stock.sector && (
                <span className="flex items-center">
                  <Building className="h-4 w-4 mr-1" />
                  {stock.sector}
                </span>
              )}
              {stock.website && (
                <a 
                  href={stock.website} 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="flex items-center hover:text-blue-600 transition-colors"
                >
                  <Globe className="h-4 w-4 mr-1" />
                  Website
                </a>
              )}
            </div>
          </div>

          <div className="text-right">
            <div className="text-4xl font-bold text-gray-900 mb-1">
              {formatPrice(stock.current_price)}
            </div>
            <div className={`flex items-center justify-end space-x-2 text-lg font-medium ${getChangeColor(stock.price_change_percent)}`}>
              {getChangeIcon(stock.price_change_percent)}
              <span>{formatPercentage(stock.price_change_percent)}</span>
              <span className="text-sm">
                ({stock.price_change >= 0 ? '+' : ''}{formatPrice(stock.price_change).replace('$', '')})
              </span>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex flex-wrap gap-3 mt-6">
          <button
            onClick={handleAddToPortfolio}
            className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Plus className="h-4 w-4 mr-2" />
            Add to Portfolio
          </button>
          
          <button
            onClick={() => navigate(`/compare?symbols=${symbol}`)}
            className="flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <BarChart3 className="h-4 w-4 mr-2" />
            Compare
          </button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center text-gray-500 text-sm mb-1">
            <DollarSign className="h-4 w-4 mr-1" />
            Market Cap
          </div>
          <div className="font-semibold text-gray-900">
            {formatLargeNumber(stock.market_cap)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center text-gray-500 text-sm mb-1">
            <Percent className="h-4 w-4 mr-1" />
            P/E Ratio
          </div>
          <div className="font-semibold text-gray-900">
            {stock.pe_ratio && stock.pe_ratio > 0 ? parseFloat(stock.pe_ratio).toFixed(2) : 'N/A'}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center text-gray-500 text-sm mb-1">
            <Volume className="h-4 w-4 mr-1" />
            Volume
          </div>
          <div className="font-semibold text-gray-900">
            {formatVolume(stock.volume)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center text-gray-500 text-sm mb-1">
            <TrendingUp className="h-4 w-4 mr-1" />
            52W High
          </div>
          <div className="font-semibold text-gray-900">
            {formatPrice(stock.fifty_two_week_high)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center text-gray-500 text-sm mb-1">
            <ArrowDownRight className="h-4 w-4 mr-1" />
            52W Low
          </div>
          <div className="font-semibold text-gray-900">
            {formatPrice(stock.fifty_two_week_low)}
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
          <div className="flex items-center text-gray-500 text-sm mb-1">
            <Activity className="h-4 w-4 mr-1" />
            Beta
          </div>
          <div className="font-semibold text-gray-900">
            {stock.beta && stock.beta > 0 ? parseFloat(stock.beta).toFixed(2) : 'N/A'}
          </div>
        </div>
      </div>

      {/* Chart Section */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-6">
          <h3 className="text-xl font-bold text-gray-900">Price Chart</h3>
          
          {/* Timeframe Selector */}
          <div className="flex flex-wrap gap-2">
            {timeframes.map(tf => (
              <button
                key={tf.key}
                onClick={() => setTimeframe(tf.key)}
                className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                  timeframe === tf.key
                    ? 'bg-blue-100 text-blue-700 border border-blue-200'
                    : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {tf.label}
              </button>
            ))}
          </div>
        </div>

        {historyLoading ? (
          <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
            <LoadingSpinner />
          </div>
        ) : historyError ? (
          <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
            <div className="text-center text-gray-500">
              <p className="mb-2">Unable to load chart data</p>
              <button 
                onClick={() => window.location.reload()}
                className="text-blue-600 hover:text-blue-700 font-medium"
              >
                Try again
              </button>
            </div>
          </div>
        ) : (
          <StockChart
            data={historyData}
            symbol={stock.symbol}
            timeframe={timeframe}
            showVolume={showVolume}
            height={500}
          />
        )}
      </div>

      {/* ARIMA Predictions Section */}
      {predictionsData && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
            <Activity className="h-5 w-5 mr-2 text-purple-600" />
            7-Day ARIMA(1,1,1) Price Predictions with 95% Confidence Intervals
          </h3>
          <div className="mb-4 text-sm text-blue-600 bg-blue-50 rounded-lg p-3">
            <strong>Model Specifications:</strong> ARIMA(1,1,1) trained on 1-year historical data from yfinance • 
            7-day forecast horizon • 95% confidence intervals • Model updated weekly
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {predictionsData.predictions?.map((prediction, index) => (
              <div key={index} className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-lg p-4 border border-purple-100">
                <div className="text-sm text-purple-600 font-medium mb-2">
                  Day {index + 1} • {new Date(prediction.date).toLocaleDateString()}
                </div>
                <div className="text-2xl font-bold text-gray-900 mb-1">
                  {formatPrice(prediction.predicted_price)}
                </div>
                <div className="text-xs text-gray-600 mb-2">
                  95% CI: {formatPrice(prediction.lower_bound)} - {formatPrice(prediction.upper_bound)}
                </div>
                <div className="text-sm text-gray-600">
                  Confidence: {prediction.confidence_percentage || 95}%
                </div>
              </div>
            )) || (
              <div className="col-span-full text-center text-gray-500 py-8">
                <Activity className="h-12 w-12 mx-auto text-gray-300 mb-2" />
                <p>ARIMA model is generating predictions...</p>
                <p className="text-sm">Training on 1-year historical data</p>
              </div>
            )}
          </div>
          
          {predictionsData && (
            <div className="mt-4 text-xs text-gray-500 bg-gray-50 rounded-lg p-3">
              <strong>Model:</strong> {predictionsData.model || 'ARIMA(1,1,1)'} • 
              <strong> Training Period:</strong> {predictionsData.model_training_period || '1 year'} •
              <strong> Accuracy:</strong> {predictionsData.accuracy_metrics?.mape ? 
                `MAPE ${predictionsData.accuracy_metrics.mape.toFixed(1)}%` : 
                'Calculating...'} •
              <strong> Generated:</strong> {new Date(predictionsData.generated_at).toLocaleString()} •
              <strong> Next Update:</strong> Weekly
            </div>
          )}
        </div>
      )}

      {/* Company Information */}
      {stock.description && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-xl font-bold text-gray-900 mb-4">About {stock.name}</h3>
          <p className="text-gray-700 leading-relaxed">{stock.description}</p>
          
          {(stock.employees || stock.sector || stock.industry) && (
            <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
              {stock.employees && (
                <div className="flex items-center text-sm text-gray-600">
                  <Users className="h-4 w-4 mr-2" />
                  <span>{stock.employees.toLocaleString()} employees</span>
                </div>
              )}
              {stock.sector && (
                <div className="flex items-center text-sm text-gray-600">
                  <Building className="h-4 w-4 mr-2" />
                  <span>{stock.sector}</span>
                </div>
              )}
              {stock.industry && (
                <div className="flex items-center text-sm text-gray-600">
                  <BarChart3 className="h-4 w-4 mr-2" />
                  <span>{stock.industry}</span>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Additional Metrics */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Additional Metrics</h3>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
          <div>
            <div className="text-sm text-gray-500 mb-1">Forward P/E</div>
            <div className="font-semibold text-gray-900">
              {stock.forward_pe && stock.forward_pe > 0 ? parseFloat(stock.forward_pe).toFixed(2) : 'N/A'}
            </div>
          </div>
          
          <div>
            <div className="text-sm text-gray-500 mb-1">Dividend Yield</div>
            <div className="font-semibold text-gray-900">
              {stock.dividend_yield && stock.dividend_yield > 0 ? (parseFloat(stock.dividend_yield) * 100).toFixed(2) + '%' : 'N/A'}
            </div>
          </div>
          
          <div>
            <div className="text-sm text-gray-500 mb-1">Avg Volume</div>
            <div className="font-semibold text-gray-900">
              {formatVolume(stock.avg_volume)}
            </div>
          </div>
          
          <div>
            <div className="text-sm text-gray-500 mb-1">Beta</div>
            <div className="font-semibold text-gray-900">
              {stock.beta && stock.beta > 0 ? parseFloat(stock.beta).toFixed(2) : 'N/A'}
            </div>
          </div>
        </div>
      </div>

      {/* Add Investment Modal */}
      <AddInvestmentModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
        stockSymbol={stock.symbol}
        stockPrice={stock.current_price}
        stockName={stock.name}
      />
    </div>
  )
}

export default StockDetail
