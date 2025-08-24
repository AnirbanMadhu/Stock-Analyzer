import React, { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from 'react-query'
import { useAuth } from '../context/AuthContext'
import { Link, useNavigate } from 'react-router-dom'
import { 
  Plus, 
  TrendingUp, 
  TrendingDown, 
  Eye,
  Trash2,
  Star,
  RefreshCw,
  Search,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3
} from 'lucide-react'
import { portfolioApi } from '../services/api'
import LoadingSpinner from '../components/LoadingSpinner'
import SearchBar from '../components/SearchBar'
import toast from 'react-hot-toast'

const Watchlist = () => {
  const { isAuthenticated, user } = useAuth()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [sortBy, setSortBy] = useState('symbol') // symbol, price, change, date
  const [sortOrder, setSortOrder] = useState('asc')

  // Fetch watchlist
  const { 
    data: watchlistData, 
    isLoading: watchlistLoading, 
    error: watchlistError,
    refetch: refetchWatchlist 
  } = useQuery(
    'watchlist',
    async () => {
      const response = await portfolioApi.getWatchlist()
      return response.data
    },
    {
      enabled: !!isAuthenticated,
      retry: 2,
      refetchInterval: 300000, // Refresh every 5 minutes
      staleTime: 240000, // 4 minutes
    }
  )

  // Remove from watchlist mutation
  const removeFromWatchlistMutation = useMutation(
    (itemId) => portfolioApi.removeFromWatchlist(itemId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('watchlist')
        toast.success('Removed from watchlist')
      },
      onError: (error) => {
        toast.error('Failed to remove from watchlist')
        console.error('Remove watchlist error:', error)
      }
    }
  )

  // Add to watchlist mutation (for search bar)
  const addToWatchlistMutation = useMutation(
    (symbol) => portfolioApi.addToWatchlist({ symbol }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('watchlist')
        toast.success('Added to watchlist')
      },
      onError: (error) => {
        toast.error('Failed to add to watchlist')
        console.error('Add watchlist error:', error)
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

  const getChangeIcon = (change) => {
    if (change > 0) return <ArrowUpRight className="h-4 w-4" />
    if (change < 0) return <ArrowDownRight className="h-4 w-4" />
    return null
  }

  const handleRemoveFromWatchlist = (itemId) => {
    if (window.confirm('Are you sure you want to remove this stock from your watchlist?')) {
      removeFromWatchlistMutation.mutate(itemId)
    }
  }

  const handleAddStock = (symbol) => {
    addToWatchlistMutation.mutate(symbol)
  }

  const handleSort = (field) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(field)
      setSortOrder('asc')
    }
  }

  const sortWatchlist = (watchlist) => {
    if (!watchlist) return []
    
    return [...watchlist].sort((a, b) => {
      let aVal, bVal
      
      switch (sortBy) {
        case 'symbol':
          aVal = a.symbol
          bVal = b.symbol
          break
        case 'price':
          aVal = a.current_price || 0
          bVal = b.current_price || 0
          break
        case 'change':
          aVal = a.price_change_percent || 0
          bVal = b.price_change_percent || 0
          break
        case 'date':
          aVal = new Date(a.added_at)
          bVal = new Date(b.added_at)
          break
        default:
          return 0
      }
      
      if (sortOrder === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0
      }
    })
  }

  if (!isAuthenticated) {
    return (
      <div className="max-w-4xl mx-auto px-4 py-12">
        <div className="text-center bg-white rounded-xl shadow-sm border border-gray-200 p-8">
          <Star className="h-16 w-16 text-gray-300 mx-auto mb-4" />
          <h1 className="text-3xl font-bold text-gray-900 mb-4">Stock Watchlist</h1>
          <p className="text-gray-600 mb-6">
            Please log in to create and manage your personal stock watchlist.
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

  const watchlist = watchlistData?.watchlist || []
  const sortedWatchlist = sortWatchlist(watchlist)

  return (
    <div className="max-w-7xl mx-auto px-4 py-6 space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 mb-2">Stock Watchlist</h1>
            <p className="text-gray-600">Monitor your favorite stocks and track their performance</p>
          </div>
          
          <div className="flex items-center space-x-3">
            <button
              onClick={() => refetchWatchlist()}
              className="flex items-center px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors"
              disabled={watchlistLoading}
            >
              <RefreshCw className={`h-4 w-4 mr-2 ${watchlistLoading ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Add Stock Search */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Add Stock to Watchlist</h3>
        <div className="max-w-2xl">
          <SearchBar 
            placeholder="Search and add stocks to your watchlist..."
            onSelectStock={handleAddStock}
          />
        </div>
      </div>

      {/* Watchlist */}
      <div className="bg-white rounded-xl shadow-sm border border-gray-200">
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center justify-between">
            <h3 className="text-xl font-semibold text-gray-900">
              Your Watchlist ({watchlist.length})
            </h3>
            
            {watchlist.length > 0 && (
              <div className="flex items-center space-x-2">
                <span className="text-sm text-gray-500">Sort by:</span>
                <select
                  value={sortBy}
                  onChange={(e) => handleSort(e.target.value)}
                  className="text-sm border border-gray-300 rounded px-2 py-1 focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="symbol">Symbol</option>
                  <option value="price">Price</option>
                  <option value="change">Change</option>
                  <option value="date">Date Added</option>
                </select>
              </div>
            )}
          </div>
        </div>

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
                Add stocks to your watchlist to monitor their performance and stay updated with price changes.
              </p>
              <p className="text-sm text-gray-500">
                Use the search bar above to find and add stocks to your watchlist.
              </p>
            </div>
          ) : (
            <div className="space-y-3">
              {/* Header Row */}
              <div className="hidden md:grid grid-cols-7 gap-4 px-4 py-3 bg-gray-50 rounded-lg text-sm font-medium text-gray-700">
                <div className="col-span-2">Stock</div>
                <div>Price</div>
                <div>Change</div>
                <div>% Change</div>
                <div>Date Added</div>
                <div>Actions</div>
              </div>

              {/* Stock Rows */}
              {sortedWatchlist.map((item) => (
                <div key={item.id} className="bg-gray-50 rounded-lg p-4 hover:bg-gray-100 transition-colors">
                  {/* Mobile Layout */}
                  <div className="md:hidden">
                    <div className="flex items-center justify-between mb-2">
                      <div>
                        <h4 className="font-semibold text-gray-900">{item.symbol}</h4>
                        <div className="text-sm text-gray-600">
                          {item.name || item.symbol}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-semibold text-gray-900">
                          {formatPrice(item.current_price)}
                        </div>
                        {item.price_change_percent && (
                          <div className={`text-sm flex items-center justify-end ${getChangeColor(item.price_change_percent)}`}>
                            {getChangeIcon(item.price_change_percent)}
                            <span className="ml-1">{formatPercentage(item.price_change_percent)}</span>
                          </div>
                        )}
                      </div>
                    </div>
                    
                    <div className="flex items-center justify-between">
                      <div className="text-xs text-gray-500">
                        Added {new Date(item.added_at).toLocaleDateString()}
                      </div>
                      
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => navigate(`/stock/${item.symbol}`)}
                          className="p-2 text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                          title="View details"
                        >
                          <Eye className="h-4 w-4" />
                        </button>
                        
                        <button
                          onClick={() => navigate(`/compare?symbols=${item.symbol}`)}
                          className="p-2 text-purple-600 hover:text-purple-700 hover:bg-purple-50 rounded-lg transition-colors"
                          title="Compare"
                        >
                          <BarChart3 className="h-4 w-4" />
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

                  {/* Desktop Layout */}
                  <div className="hidden md:grid grid-cols-7 gap-4 items-center">
                    <div className="col-span-2">
                      <div className="font-semibold text-gray-900">{item.symbol}</div>
                      <div className="text-sm text-gray-600 truncate">
                        {item.name || item.symbol}
                      </div>
                    </div>
                    
                    <div className="font-semibold text-gray-900">
                      {formatPrice(item.current_price)}
                    </div>
                    
                    <div className={`font-medium ${getChangeColor(item.price_change)}`}>
                      {item.price_change ? formatPrice(item.price_change) : 'N/A'}
                    </div>
                    
                    <div className={`flex items-center font-medium ${getChangeColor(item.price_change_percent)}`}>
                      {getChangeIcon(item.price_change_percent)}
                      <span className="ml-1">{formatPercentage(item.price_change_percent)}</span>
                    </div>
                    
                    <div className="text-sm text-gray-600">
                      {new Date(item.added_at).toLocaleDateString()}
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => navigate(`/stock/${item.symbol}`)}
                        className="p-2 text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-lg transition-colors"
                        title="View details"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                      
                      <button
                        onClick={() => navigate(`/compare?symbols=${item.symbol}`)}
                        className="p-2 text-purple-600 hover:text-purple-700 hover:bg-purple-50 rounded-lg transition-colors"
                        title="Compare"
                      >
                        <BarChart3 className="h-4 w-4" />
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
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Watchlist
