import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { useAuth } from '../context/AuthContext'
import { Link, useNavigate } from 'react-router-dom'
import { 
  Plus, 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  Percent, 
  Eye,
  Trash2,
  Edit,
  BarChart3,
  PieChart,
  RefreshCw
} from 'lucide-react'
import { portfolioApi, stockApi } from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'
import AddInvestmentModal from '../components/AddInvestmentModal'
import toast from 'react-hot-toast'

const Portfolio = () => {
  const { isAuthenticated, user } = useAuth()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState('holdings')
  const [showAddModal, setShowAddModal] = useState(false)

  // Fetch portfolio holdings
  const { 
    data: holdingsData, 
    isLoading: holdingsLoading, 
    error: holdingsError,
    refetch: refetchHoldings 
  } = useQuery(
    'portfolioHoldings',
    async () => {
      const response = await portfolioApi.getHoldings()
      return response.data
    },
    {
      enabled: !!isAuthenticated,
      retry: 2,
      staleTime: 300000, // 5 minutes
    }
  )

  // Fetch watchlist
  const { 
    data: watchlistData, 
    isLoading: watchlistLoading, 
    error: watchlistError,
    refetch: refetchWatchlist 
  } = useQuery(
    'portfolioWatchlist',
    async () => {
      const response = await portfolioApi.getWatchlist()
      return response.data
    },
    {
      enabled: !!isAuthenticated,
      retry: 2,
      staleTime: 300000, // 5 minutes
    }
  )

  // Remove from watchlist mutation
  const removeFromWatchlistMutation = useMutation(
    (itemId) => portfolioApi.removeFromWatchlist(itemId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('portfolioWatchlist')
        toast.success('Removed from watchlist')
      },
      onError: (error) => {
        toast.error('Failed to remove from watchlist')
        console.error('Remove watchlist error:', error)
      }
    }
  )

  // Delete holding mutation
  const deleteHoldingMutation = useMutation(
    (holdingId) => portfolioApi.deleteHolding(holdingId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('portfolioHoldings')
        toast.success('Holding removed')
      },
      onError: (error) => {
        toast.error('Failed to remove holding')
        console.error('Delete holding error:', error)
      }
    }
  )

  const formatPrice = (price) => {
    if (!price) return '$0.00'
    return `$${parseFloat(price).toFixed(2)}`
  }

  const formatPercentage = (percent) => {
    if (!percent) return '0.00%'
    const sign = percent >= 0 ? '+' : ''
    return `${sign}${parseFloat(percent).toFixed(2)}%`
  }

  const getChangeColor = (change) => {
    if (change > 0) return 'text-green-600'
    if (change < 0) return 'text-red-600'
    return 'text-gray-600'
  }

  const calculatePortfolioValue = (holdings) => {
    if (!holdings || holdings.length === 0) return 0
    return holdings.reduce((total, holding) => {
      return total + (holding.quantity * holding.current_price)
    }, 0)
  }

  const calculatePortfolioGain = (holdings) => {
    if (!holdings || holdings.length === 0) return { gain: 0, percentage: 0 }
    
    let totalValue = 0
    let totalCost = 0
    
    holdings.forEach(holding => {
      totalValue += holding.quantity * holding.current_price
      totalCost += holding.quantity * holding.purchase_price
    })
    
    const gain = totalValue - totalCost
    const percentage = totalCost > 0 ? (gain / totalCost) * 100 : 0
    
    return { gain, percentage }
  }

  const handleRemoveFromWatchlist = (itemId) => {
    removeFromWatchlistMutation.mutate(itemId)
  }

  const handleDeleteHolding = (holdingId) => {
    if (window.confirm('Are you sure you want to remove this holding?')) {
      deleteHoldingMutation.mutate(holdingId)
    }
  }

  if (!isAuthenticated) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="text-center bg-white rounded-xl shadow-sm border border-gray-200 p-8">
          <DollarSign className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-gray-900 mb-4">Portfolio Management</h1>
          <p className="text-gray-600 mb-6">
            Please log in to view and manage your investment portfolio.
          </p>
          <Link 
            to="/login" 
            className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            Sign In to Continue
          </Link>
        </div>
      </div>
    )
  }

  const holdings = holdingsData?.holdings || []
  const watchlist = watchlistData?.watchlist || []
  const portfolioValue = calculatePortfolioValue(holdings)
  const portfolioGain = calculatePortfolioGain(holdings)

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Portfolio Overview</h1>
            <p className="text-gray-600">Manage your investments and track performance</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => {
                refetchHoldings()
                refetchWatchlist()
              }}
              className="flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </button>
            
            <button
              onClick={() => setShowAddModal(true)}
              className="flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Plus className="h-4 w-4 mr-2" />
              Add Investment
            </button>
          </div>
        </div>
      </div>

      {/* Portfolio Summary */}
      {holdings.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold text-gray-900">Total Value</h3>
              <DollarSign className="h-5 w-5 text-blue-600" />
            </div>
            <div className="text-3xl font-bold text-gray-900">
              {formatPrice(portfolioValue)}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold text-gray-900">Total Gain/Loss</h3>
              {portfolioGain.gain >= 0 ? (
                <TrendingUp className="h-5 w-5 text-green-600" />
              ) : (
                <TrendingDown className="h-5 w-5 text-red-600" />
              )}
            </div>
            <div className={`text-3xl font-bold ${getChangeColor(portfolioGain.gain)}`}>
              {formatPrice(portfolioGain.gain)}
            </div>
            <div className={`text-sm font-medium ${getChangeColor(portfolioGain.gain)}`}>
              {formatPercentage(portfolioGain.percentage)}
            </div>
          </div>

          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-lg font-semibold text-gray-900">Holdings</h3>
              <BarChart3 className="h-5 w-5 text-purple-600" />
            </div>
            <div className="text-3xl font-bold text-gray-900">
              {holdings.length}
            </div>
            <div className="text-sm text-gray-500">
              Active positions
            </div>
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200">
        <div className="border-b border-gray-200">
          <nav className="-mb-px flex space-x-8 px-6">
            <button
              onClick={() => setActiveTab('holdings')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'holdings'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Holdings ({holdings.length})
            </button>
            <button
              onClick={() => setActiveTab('watchlist')}
              className={`py-4 px-1 border-b-2 font-medium text-sm transition-colors ${
                activeTab === 'watchlist'
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              Watchlist ({watchlist.length})
            </button>
          </nav>
        </div>

        {/* Holdings Tab */}
        {activeTab === 'holdings' && (
          <div className="p-6">
            {holdingsLoading ? (
              <div className="flex items-center justify-center py-12">
                <LoadingSpinner />
              </div>
            ) : holdingsError ? (
              <div className="text-center py-12">
                <p className="text-red-600 mb-4">Failed to load holdings</p>
                <button
                  onClick={() => refetchHoldings()}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Try again
                </button>
              </div>
            ) : holdings.length === 0 ? (
              <div className="text-center py-12">
                <PieChart className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">No Holdings Yet</h3>
                <p className="text-gray-600 mb-6">
                  Start building your portfolio by adding your first investment.
                </p>
                <button
                  onClick={() => setShowAddModal(true)}
                  className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Add First Investment
                </button>
              </div>
            ) : (
              <div className="space-y-4">
                {holdings.map((holding) => (
                  <div key={holding.id} className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div>
                          <div className="flex items-center space-x-2">
                            <h4 className="font-semibold text-gray-900">{holding.ticker}</h4>
                            <span className="text-sm text-gray-500">
                              {holding.quantity} shares
                            </span>
                          </div>
                          <div className="text-sm text-gray-600">
                            Bought at {formatPrice(holding.purchase_price)} on {new Date(holding.purchase_date).toLocaleDateString()}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <div className="font-semibold text-gray-900">
                            {formatPrice(holding.current_price)}
                          </div>
                          <div className={`text-sm ${getChangeColor(holding.current_price - holding.purchase_price)}`}>
                            {formatPrice((holding.current_price - holding.purchase_price) * holding.quantity)}
                          </div>
                        </div>

                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => navigate(`/stock/${holding.ticker}`)}
                            className="p-2 text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                            title="View details"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          
                          <button
                            onClick={() => handleDeleteHolding(holding.id)}
                            className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                            title="Remove holding"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Watchlist Tab */}
        {activeTab === 'watchlist' && (
          <div className="p-6">
            {watchlistLoading ? (
              <div className="flex items-center justify-center py-12">
                <LoadingSpinner />
              </div>
            ) : watchlistError ? (
              <div className="text-center py-12">
                <p className="text-red-600 mb-4">Failed to load watchlist</p>
                <button
                  onClick={() => refetchWatchlist()}
                  className="text-blue-600 hover:text-blue-700 font-medium"
                >
                  Try again
                </button>
              </div>
            ) : watchlist.length === 0 ? (
              <div className="text-center py-12">
                <Eye className="h-16 w-16 text-gray-300 mx-auto mb-4" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Empty Watchlist</h3>
                <p className="text-gray-600 mb-6">
                  Add stocks to your watchlist to monitor their performance.
                </p>
                <Link
                  to="/"
                  className="inline-flex items-center px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
                >
                  <Plus className="h-4 w-4 mr-2" />
                  Find Stocks to Watch
                </Link>
              </div>
            ) : (
              <div className="space-y-4">
                {watchlist.map((item) => (
                  <div key={item.id} className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div>
                          <h4 className="font-semibold text-gray-900">{item.ticker}</h4>
                          <div className="text-sm text-gray-600">
                            Added {new Date(item.added_at).toLocaleDateString()}
                          </div>
                        </div>
                      </div>

                      <div className="flex items-center space-x-4">
                        <div className="text-right">
                          <div className="font-semibold text-gray-900">
                            {formatPrice(item.current_price)}
                          </div>
                          {item.price_change_percent && (
                            <div className={`text-sm ${getChangeColor(item.price_change_percent)}`}>
                              {formatPercentage(item.price_change_percent)}
                            </div>
                          )}
                        </div>

                        <div className="flex items-center space-x-2">
                          <button
                            onClick={() => navigate(`/stock/${item.ticker}`)}
                            className="p-2 text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                            title="View details"
                          >
                            <Eye className="h-4 w-4" />
                          </button>
                          
                          <button
                            onClick={() => handleRemoveFromWatchlist(item.id)}
                            className="p-2 text-red-600 hover:text-red-700 hover:bg-red-50 rounded-lg transition-colors"
                            title="Remove from watchlist"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Add Investment Modal */}
      <AddInvestmentModal
        isOpen={showAddModal}
        onClose={() => setShowAddModal(false)}
      />
    </div>
  )
}

export default Portfolio
