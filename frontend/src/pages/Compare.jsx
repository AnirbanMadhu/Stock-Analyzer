import React, { useState, useEffect, useMemo } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import { useQuery } from 'react-query'
import { 
  Plus, 
  X, 
  BarChart3, 
  TrendingUp, 
  ArrowUpRight, 
  ArrowDownRight,
  Search
} from 'lucide-react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js'
import { Line } from 'react-chartjs-2'
import { format } from 'date-fns'
import { stockApi } from '../services/api'
import SearchBar from '../components/SearchBar'
import LoadingSpinner from '../components/LoadingSpinner'
import toast from 'react-hot-toast'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
)

const Compare = () => {
  const [searchParams, setSearchParams] = useSearchParams()
  const navigate = useNavigate()
  const [selectedStocks, setSelectedStocks] = useState([])
  const [timeframe, setTimeframe] = useState('1y')
  const [showAddStock, setShowAddStock] = useState(false)

  // Parse URL parameters on component mount
  useEffect(() => {
    const symbolsParam = searchParams.get('symbols')
    if (symbolsParam) {
      const symbols = symbolsParam.split(',').filter(s => s.trim())
      setSelectedStocks(symbols.slice(0, 4)) // Max 4 stocks
    }
  }, [searchParams])

  // Update URL when stocks change
  useEffect(() => {
    if (selectedStocks.length > 0) {
      setSearchParams({ symbols: selectedStocks.join(',') })
    } else {
      setSearchParams({})
    }
  }, [selectedStocks, setSearchParams])

  // Fetch comparison data
  const { 
    data: comparisonData, 
    isLoading: comparisonLoading, 
    error: comparisonError 
  } = useQuery(
    ['stockComparison', selectedStocks, timeframe],
    async () => {
      console.log('Fetching comparison data for stocks:', selectedStocks, 'timeframe:', timeframe)
      try {
        const response = await stockApi.compare(selectedStocks, timeframe, '1d')
        console.log('Comparison API response:', response.data)
        return response.data
      } catch (error) {
        console.error('Comparison API error:', error)
        throw error
      }
    },
    {
      enabled: selectedStocks.length >= 2,
      staleTime: 300000, // 5 minutes
      retry: (failureCount, error) => {
        console.log(`Retry attempt ${failureCount} for comparison:`, error?.message)
        return failureCount < 2
      },
      onError: (error) => {
        console.error('Stock comparison query error:', error)
        toast.error(`Failed to load comparison data: ${error?.response?.data?.error || error?.message}`)
      },
    }
  )

  const addStock = (symbol) => {
    const upperSymbol = symbol.toUpperCase()
    
    if (selectedStocks.includes(upperSymbol)) {
      toast.error('Stock already added to comparison')
      return
    }
    
    if (selectedStocks.length >= 4) {
      toast.error('Maximum 4 stocks can be compared')
      return
    }
    
    setSelectedStocks(prev => [...prev, upperSymbol])
    setShowAddStock(false)
    toast.success(`${upperSymbol} added to comparison`)
  }

  const removeStock = (symbol) => {
    setSelectedStocks(prev => prev.filter(s => s !== symbol))
    toast.success(`${symbol} removed from comparison`)
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
    if (!num) return 'N/A'
    
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

  // Create chart data for performance comparison
  const performanceChartData = useMemo(() => {
    if (!comparisonData?.stocks_history) return null

    console.log('Processing chart data for stocks:', Object.keys(comparisonData.stocks_history))

    // Get all dates from all stocks and sort them
    const allDatesSet = new Set()
    const validStocks = []
    
    // First, filter out stocks with valid data
    Object.entries(comparisonData.stocks_history).forEach(([symbol, stockData]) => {
      if (stockData?.data && Array.isArray(stockData.data) && stockData.data.length > 0) {
        validStocks.push({ symbol, data: stockData.data })
        stockData.data.forEach(point => {
          if (point && point.date && point.close) {
            allDatesSet.add(point.date)
          }
        })
      }
    })

    console.log('Valid stocks found:', validStocks.length, validStocks.map(s => s.symbol))

    if (validStocks.length === 0) {
      console.log('No valid stocks found for chart')
      return null
    }

    const sortedDates = Array.from(allDatesSet).sort()
    
    if (sortedDates.length === 0) {
      console.log('No valid dates found')
      return null
    }
    
    console.log('Date range:', sortedDates.length, 'points from', sortedDates[0], 'to', sortedDates[sortedDates.length - 1])
    
    // Limit the number of data points for better performance
    const maxDataPoints = 500
    let processedDates = sortedDates
    if (sortedDates.length > maxDataPoints) {
      const step = Math.ceil(sortedDates.length / maxDataPoints)
      processedDates = sortedDates.filter((_, index) => index % step === 0)
      console.log('Reduced data points from', sortedDates.length, 'to', processedDates.length)
    }
    
    // Format labels based on timeframe
    const labels = processedDates.map(date => {
      try {
        const d = new Date(date)
        if (timeframe === '1d') return format(d, 'HH:mm')
        if (timeframe === '5d' || timeframe === '1mo') return format(d, 'MMM dd')
        return format(d, 'MMM yyyy')
      } catch (error) {
        console.warn('Date formatting error:', error, date)
        return date
      }
    })

    // Generate colors for each stock (support up to 8 stocks)
    const colors = [
      '#3B82F6', // blue
      '#EF4444', // red  
      '#10B981', // green
      '#F59E0B', // yellow
      '#8B5CF6', // purple
      '#F97316', // orange
      '#06B6D4', // cyan
      '#84CC16'  // lime
    ]

    // Create datasets for each valid stock
    const datasets = validStocks.map((stockInfo, index) => {
      try {
        const { symbol, data } = stockInfo
        
        // Create price map for this stock
        const priceMap = new Map()
        data.forEach(point => {
          if (point && point.date && typeof point.close === 'number') {
            priceMap.set(point.date, parseFloat(point.close))
          }
        })

        console.log(`Processing ${symbol}: ${priceMap.size} data points`)

        // Get prices for our processed dates
        const prices = processedDates.map(date => {
          const price = priceMap.get(date)
          return price !== undefined ? price : null
        })

        // Find the first valid price for normalization
        const firstPrice = prices.find(price => price !== null && price > 0)
        
        if (!firstPrice) {
          console.warn(`No valid first price found for ${symbol}`)
          return null
        }
        
        console.log(`${symbol} first price for normalization:`, firstPrice)
        
        // Calculate normalized data (percentage change from first value)
        const normalizedData = prices.map(price => {
          if (price === null || price <= 0) return null
          return ((price - firstPrice) / firstPrice) * 100
        })

        // Count valid data points
        const validPoints = normalizedData.filter(point => point !== null).length
        console.log(`${symbol} has ${validPoints} valid normalized points out of ${normalizedData.length}`)

        if (validPoints === 0) {
          console.warn(`No valid data points for ${symbol}`)
          return null
        }

        return {
          label: symbol,
          data: normalizedData,
          borderColor: colors[index % colors.length],
          backgroundColor: colors[index % colors.length] + '20',
          borderWidth: 3,
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          pointHoverRadius: 6,
          pointHoverBackgroundColor: colors[index % colors.length],
          pointHoverBorderColor: '#ffffff',
          pointHoverBorderWidth: 2,
          spanGaps: true, // Connect line through null values
        }
      } catch (error) {
        console.error(`Error processing data for ${stockInfo.symbol}:`, error)
        return null
      }
    }).filter(Boolean)

    console.log('Final datasets created:', datasets.length)

    if (datasets.length === 0) {
      console.log('No datasets created')
      return null
    }

    return { labels, datasets }
  }, [comparisonData, timeframe])

  const chartOptions = useMemo(() => ({
    responsive: true,
    maintainAspectRatio: false,
    interaction: {
      mode: 'index',
      intersect: false,
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          color: '#374151',
          usePointStyle: true,
          padding: 20,
          font: {
            size: 12,
            weight: '500'
          }
        },
      },
      title: {
        display: true,
        text: `Performance Comparison - ${timeframe.toUpperCase()}`,
        color: '#111827',
        font: {
          size: 16,
          weight: '600',
        },
        padding: {
          top: 10,
          bottom: 30
        }
      },
      tooltip: {
        backgroundColor: 'rgba(255, 255, 255, 0.95)',
        titleColor: '#111827',
        bodyColor: '#374151',
        borderColor: '#E5E7EB',
        borderWidth: 1,
        cornerRadius: 8,
        displayColors: true,
        callbacks: {
          label: function(context) {
            const value = context.parsed.y
            const symbol = context.dataset.label
            if (value === null) return `${symbol}: No data`
            return `${symbol}: ${value >= 0 ? '+' : ''}${parseFloat(value).toFixed(2)}%`
          },
        },
      },
    },
    scales: {
      x: {
        display: true,
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          maxTicksLimit: 8,
          color: '#6B7280',
        },
      },
      y: {
        display: true,
        position: 'right',
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          color: '#6B7280',
          callback: function(value) {
            return (value >= 0 ? '+' : '') + parseFloat(value).toFixed(1) + '%'
          },
        },
      },
    },
    elements: {
      point: {
        hoverRadius: 8,
      },
    },
  }), [timeframe])

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

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2 flex items-center">
              <BarChart3 className="h-8 w-8 mr-3 text-blue-600" />
              Stock Comparison
            </h1>
            <p className="text-gray-600">
              Compare up to 4 stocks side by side to make informed investment decisions
            </p>
          </div>
          
          <button
            onClick={() => setShowAddStock(true)}
            disabled={selectedStocks.length >= 4}
            className={`flex items-center px-6 py-3 rounded-lg font-medium transition-colors ${
              selectedStocks.length >= 4
                ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            <Plus className="h-5 w-5 mr-2" />
            Add Stock {selectedStocks.length > 0 && `(${selectedStocks.length}/4)`}
          </button>
        </div>
      </div>

      {/* Add Stock Modal */}
      {showAddStock && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between p-6 border-b border-gray-200">
              <div>
                <h3 className="text-xl font-bold text-gray-900">Add Stock to Compare</h3>
                <p className="text-sm text-gray-500 mt-1">
                  Search for any stock or choose from popular options below
                </p>
              </div>
              <button
                onClick={() => setShowAddStock(false)}
                className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <div className="p-6">
              <div className="mb-6">
                <SearchBar 
                  placeholder="Search for any stock symbol or company name..."
                  className="w-full"
                  onSelectStock={addStock}
                />
              </div>

              <div className="mb-6">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Or enter stock symbol directly:</h4>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="e.g., AAPL"
                    className="flex-1 px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter') {
                        const value = e.target.value.trim().toUpperCase()
                        if (value) {
                          addStock(value)
                          e.target.value = ''
                        }
                      }
                    }}
                  />
                  <button
                    onClick={(e) => {
                      const input = e.target.previousElementSibling
                      const value = input.value.trim().toUpperCase()
                      if (value) {
                        addStock(value)
                        input.value = ''
                      }
                    }}
                    className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                  >
                    Add
                  </button>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Press Enter or click Add to include any stock symbol
                </p>
              </div>
              
              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Popular Tech Stocks:</h4>
                  <div className="flex flex-wrap gap-2">
                    {['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'META', 'NFLX', 'NVDA'].map(symbol => (
                      <button
                        key={symbol}
                        onClick={() => addStock(symbol)}
                        disabled={selectedStocks.includes(symbol)}
                        className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                          selectedStocks.includes(symbol)
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            : 'bg-blue-100 text-blue-700 hover:bg-blue-200'
                        }`}
                      >
                        {symbol}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Financial & Industrial:</h4>
                  <div className="flex flex-wrap gap-2">
                    {['JPM', 'BAC', 'WFC', 'V', 'MA', 'BRK-B', 'JNJ', 'PG'].map(symbol => (
                      <button
                        key={symbol}
                        onClick={() => addStock(symbol)}
                        disabled={selectedStocks.includes(symbol)}
                        className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                          selectedStocks.includes(symbol)
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            : 'bg-green-100 text-green-700 hover:bg-green-200'
                        }`}
                      >
                        {symbol}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Consumer & Retail:</h4>
                  <div className="flex flex-wrap gap-2">
                    {['WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'DIS', 'KO', 'PEP'].map(symbol => (
                      <button
                        key={symbol}
                        onClick={() => addStock(symbol)}
                        disabled={selectedStocks.includes(symbol)}
                        className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                          selectedStocks.includes(symbol)
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            : 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                        }`}
                      >
                        {symbol}
                      </button>
                    ))}
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-medium text-gray-700 mb-2">Healthcare & Energy:</h4>
                  <div className="flex flex-wrap gap-2">
                    {['PFE', 'UNH', 'ABBV', 'BMY', 'XOM', 'CVX', 'COP', 'SLB'].map(symbol => (
                      <button
                        key={symbol}
                        onClick={() => addStock(symbol)}
                        disabled={selectedStocks.includes(symbol)}
                        className={`px-3 py-1 rounded-full text-sm font-medium transition-colors ${
                          selectedStocks.includes(symbol)
                            ? 'bg-gray-100 text-gray-400 cursor-not-allowed'
                            : 'bg-red-100 text-red-700 hover:bg-red-200'
                        }`}
                      >
                        {symbol}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Selected Stocks */}
      {selectedStocks.length > 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-bold text-gray-900 mb-4">Selected Stocks</h3>
          
          <div className="flex flex-wrap gap-3">
            {selectedStocks.map(symbol => (
              <div 
                key={symbol} 
                className="flex items-center bg-blue-50 border border-blue-200 rounded-lg px-3 py-2"
              >
                <span className="font-medium text-blue-900">{symbol}</span>
                <button
                  onClick={() => removeStock(symbol)}
                  className="ml-2 p-1 hover:bg-blue-200 rounded-full transition-colors"
                >
                  <X className="h-4 w-4 text-blue-600" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* No stocks selected */}
      {selectedStocks.length === 0 && (
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-12 text-center">
          <BarChart3 className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Start Comparing Stocks</h3>
          <p className="text-gray-600 mb-6">
            Add 2-4 stocks to compare their performance, metrics, and historical data
          </p>
          <button
            onClick={() => setShowAddStock(true)}
            className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors font-medium"
          >
            Add Your First Stock
          </button>
        </div>
      )}

      {/* Need at least 2 stocks */}
      {selectedStocks.length === 1 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-6 text-center">
          <p className="text-amber-800">
            Add at least one more stock to start comparing. 
            <button
              onClick={() => setShowAddStock(true)}
              className="ml-2 text-amber-600 hover:text-amber-700 font-medium underline"
            >
              Add another stock
            </button>
          </p>
        </div>
      )}

      {/* Comparison Results */}
      {selectedStocks.length >= 2 && (
        <>
          {/* Key Metrics Comparison */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-xl font-bold text-gray-900 mb-6">Key Metrics Comparison</h3>
            
            {comparisonLoading ? (
              <div className="flex flex-col items-center justify-center py-12">
                <LoadingSpinner size="large" />
                <p className="mt-4 text-gray-600">
                  Loading comparison data for {selectedStocks.length} stocks...
                </p>
                {selectedStocks.length >= 3 && (
                  <p className="mt-2 text-sm text-gray-500">
                    This may take a moment with multiple stocks
                  </p>
                )}
              </div>
            ) : comparisonError ? (
              <div className="text-center py-12">
                <p className="text-red-600 mb-2">Failed to load comparison data</p>
                <p className="text-sm text-gray-500 mb-4">
                  {comparisonError?.response?.data?.error || 
                   (selectedStocks.length >= 3 
                    ? 'Multiple stocks comparison may fail due to data availability' 
                    : 'Please check if the stock symbols are valid')}
                </p>
                <div className="space-x-2">
                  <button 
                    onClick={() => window.location.reload()}
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    Try again
                  </button>
                  <button
                    onClick={() => setSelectedStocks(selectedStocks.slice(0, 2))}
                    className="text-orange-600 hover:text-orange-700 font-medium"
                  >
                    Compare first 2 stocks only
                  </button>
                </div>
              </div>
            ) : comparisonData?.stocks_info ? (
              <div>
                {/* Show warning if some stocks failed to load */}
                {comparisonData?.failed_symbols && comparisonData.failed_symbols.length > 0 && (
                  <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-yellow-800">
                      <span className="font-medium">Partial data loaded:</span> Could not retrieve data for{' '}
                      {comparisonData.failed_symbols.join(', ')}. 
                      Showing comparison for {comparisonData.success_count || Object.keys(comparisonData.stocks_info).length} stocks.
                    </p>
                  </div>
                )}
                
                <div className="overflow-x-auto">
                  <table className="w-full">
                    <thead>
                      <tr className="border-b border-gray-200">
                        <th className="text-left py-3 px-4 font-semibold text-gray-900">Metric</th>
                        {Object.keys(comparisonData.stocks_info).filter(symbol => comparisonData.stocks_info[symbol]).map(symbol => (
                          <th key={symbol} className="text-center py-3 px-4 font-semibold text-gray-900">
                            {symbol}
                          </th>
                        ))}
                      </tr>
                    </thead>
                  <tbody>
                    <tr className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium text-gray-700">Current Price</td>
                      {Object.values(comparisonData.stocks_info).filter(Boolean).map((stock, index) => (
                        <td key={index} className="py-3 px-4 text-center font-semibold">
                          {formatPrice(stock?.current_price)}
                        </td>
                      ))}
                    </tr>
                    
                    <tr className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium text-gray-700">Change %</td>
                      {Object.values(comparisonData.stocks_info).filter(Boolean).map((stock, index) => (
                        <td key={index} className={`py-3 px-4 text-center font-semibold ${getChangeColor(stock?.price_change_percent)}`}>
                          <div className="flex items-center justify-center space-x-1">
                            {getChangeIcon(stock?.price_change_percent)}
                            <span>{formatPercentage(stock?.price_change_percent)}</span>
                          </div>
                        </td>
                      ))}
                    </tr>
                    
                    <tr className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium text-gray-700">Market Cap</td>
                      {Object.values(comparisonData.stocks_info).filter(Boolean).map((stock, index) => (
                        <td key={index} className="py-3 px-4 text-center">
                          {formatLargeNumber(stock?.market_cap)}
                        </td>
                      ))}
                    </tr>
                    
                    <tr className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium text-gray-700">P/E Ratio</td>
                      {Object.values(comparisonData.stocks_info).filter(Boolean).map((stock, index) => (
                        <td key={index} className="py-3 px-4 text-center">
                          {stock?.pe_ratio ? parseFloat(stock.pe_ratio).toFixed(2) : 'N/A'}
                        </td>
                      ))}
                    </tr>
                    
                    <tr className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium text-gray-700">Beta</td>
                      {Object.values(comparisonData.stocks_info).filter(Boolean).map((stock, index) => (
                        <td key={index} className="py-3 px-4 text-center">
                          {stock?.beta ? parseFloat(stock.beta).toFixed(2) : 'N/A'}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
            ) : null}
          </div>

          {/* Performance Chart */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-6">
              <h3 className="text-xl font-bold text-gray-900">Performance Comparison</h3>
              
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

            {comparisonLoading ? (
              <div className="h-96 bg-gray-50 rounded-lg flex flex-col items-center justify-center">
                <LoadingSpinner size="large" />
                <p className="mt-4 text-gray-600">
                  Processing historical data for {selectedStocks.length} stocks...
                </p>
                {selectedStocks.length >= 3 && (
                  <p className="mt-2 text-sm text-gray-500">
                    Comparing multiple stocks requires more data processing
                  </p>
                )}
              </div>
            ) : comparisonError ? (
              <div className="h-96 bg-gray-50 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <p className="text-red-600 mb-4">Failed to load chart data</p>
                  <button 
                    onClick={() => window.location.reload()}
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    Try again
                  </button>
                </div>
              </div>
            ) : performanceChartData ? (
              <div className="h-96 w-full">
                <Line data={performanceChartData} options={chartOptions} />
              </div>
            ) : (
              <div className="h-96 bg-gray-50 rounded-lg flex items-center justify-center">
                <div className="text-center">
                  <TrendingUp className="h-12 w-12 text-gray-300 mx-auto mb-2" />
                  <p className="text-gray-500">No chart data available</p>
                  <p className="text-sm text-gray-400">
                    Unable to load historical data for comparison
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Individual Stock Links */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-bold text-gray-900 mb-4">View Individual Details</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {selectedStocks.map(symbol => (
                <button
                  key={symbol}
                  onClick={() => navigate(`/stock/${symbol}`)}
                  className="p-4 border border-gray-200 rounded-lg hover:border-blue-200 hover:bg-blue-50 transition-all text-left"
                >
                  <div className="font-semibold text-gray-900 mb-1">{symbol}</div>
                  <div className="text-sm text-gray-500">View detailed analysis →</div>
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default Compare
