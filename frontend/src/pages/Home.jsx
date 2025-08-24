import React, { useState, useEffect } from 'react'
import { useQuery } from 'react-query'
import { Link, useNavigate } from 'react-router-dom'
import { Search, TrendingUp, ArrowUpRight, ArrowDownRight, Eye, Briefcase, BarChart, Shield, Newspaper, Activity } from 'lucide-react'
import SearchBar from '../components/SearchBar'
import { stockApi } from '../services/api'

const Home = () => {
  const navigate = useNavigate()

  // Fetch market summary data
  const { data: marketData, isLoading: marketLoading, error: marketError } = useQuery(
    'marketSummary',
    async () => {
      const response = await stockApi.getMarketSummary()
      return response.data
    },
    { 
      refetchInterval: 300000, // Refresh every 5 minutes
      retry: 3,
      staleTime: 240000, // 4 minutes
    }
  )

  // Fetch trending stocks
  const { data: trendingData, isLoading: trendingLoading, error: trendingError } = useQuery(
    'trendingStocks',
    async () => {
      const response = await stockApi.getTrending(10)
      return response.data
    },
    { 
      refetchInterval: 300000,
      retry: 3,
      staleTime: 240000,
    }
  )

  // Fetch top gainers and losers from real-time market data
  const { data: gainersData, isLoading: gainersLoading } = useQuery(
    'topGainers',
    async () => {
      const response = await stockApi.getTopGainers(10)
      return response.data
    },
    { 
      refetchInterval: 300000,
      retry: 3,
      staleTime: 240000,
    }
  )

  const { data: losersData, isLoading: losersLoading } = useQuery(
    'topLosers',
    async () => {
      const response = await stockApi.getTopLosers(10)
      return response.data
    },
    { 
      refetchInterval: 300000,
      retry: 3,
      staleTime: 240000,
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

  return (
    <div className="space-y-8 pb-16">
      {/* Hero Section with Search */}
      <div className="text-center py-12 bg-gradient-to-r from-blue-600 to-blue-700 rounded-2xl text-white">
        <h1 className="text-4xl md:text-5xl font-bold mb-4">
          Track, Analyze, Invest
        </h1>
        <p className="text-xl md:text-2xl mb-8 opacity-90 max-w-2xl mx-auto">
          Your comprehensive platform for stock market analysis and portfolio management
        </p>
        
        {/* Enhanced Search Bar */}
        <div className="max-w-2xl mx-auto">
          <SearchBar />
        </div>
      </div>

      {/* Market Indices */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Market Overview</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {marketLoading ? (
            // Loading skeleton
            Array.from({ length: 4 }).map((_, i) => (
              <div key={i} className="bg-white rounded-lg shadow p-6 animate-pulse">
                <div className="h-4 bg-gray-200 rounded mb-2"></div>
                <div className="h-8 bg-gray-200 rounded mb-1"></div>
                <div className="h-4 bg-gray-200 rounded w-1/2"></div>
              </div>
            ))
          ) : marketError ? (
            <div className="col-span-full bg-red-50 border border-red-200 rounded-lg p-6 text-center">
              <p className="text-red-600">Unable to load market data. Please try again later.</p>
            </div>
          ) : (
            marketData?.market_indices?.map((index) => (
              <div 
                key={index.symbol} 
                className="bg-white rounded-lg shadow hover:shadow-md transition-shadow p-6 cursor-pointer"
                onClick={() => {
                  if (index.symbol) {
                    // URL encode the symbol to handle special characters like ^
                    const encodedSymbol = encodeURIComponent(index.symbol);
                    navigate(`/stock/${encodedSymbol}`);
                  }
                }}
              >
                <div className="flex items-center justify-between mb-2">
                  <h3 className="font-semibold text-gray-900">{index.name}</h3>
                  <span className="text-xs text-gray-500">{index.symbol}</span>
                </div>
                <div className="space-y-1">
                  <div className="text-2xl font-bold text-gray-900">
                    {formatPrice(index.current_price)}
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`flex items-center text-sm font-medium ${getChangeColor(index.price_change_percent)}`}>
                      {getChangeIcon(index.price_change_percent)}
                      {formatPercentage(index.price_change_percent)}
                    </span>
                    <span className={`text-xs ${getChangeColor(index.price_change)}`}>
                      ({index.price_change >= 0 ? '+' : ''}{formatPrice(Math.abs(index.price_change)).replace('$', '')})
                    </span>
                  </div>
                </div>
              </div>
            )) || (
              <div className="col-span-full bg-gray-50 rounded-lg p-8 text-center">
                <p className="text-gray-500">No market data available</p>
              </div>
            )
          )}
        </div>
      </section>

      {/* Top Gainers and Losers */}
      <section>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Top Gainers */}
          <div>
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
              <ArrowUpRight className="h-5 w-5 text-green-600 mr-2" />
              Top Gainers
            </h3>
            <div className="bg-white rounded-lg shadow overflow-hidden">
              {gainersLoading ? (
                <div className="p-4 animate-pulse">
                  {Array.from({ length: 10 }).map((_, i) => (
                    <div key={i} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
                      <div className="h-4 bg-gray-200 rounded w-1/3"></div>
                      <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                    </div>
                  ))}
                </div>
              ) : (
                gainersData?.gainers?.map((stock) => (
                  <div key={stock.symbol} 
                       className="flex justify-between items-center p-4 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                       onClick={() => {
                         if (stock.symbol && stock.symbol !== 'undefined') {
                           navigate(`/stock/${stock.symbol}`)
                         }
                       }}>
                    <div>
                      <div className="font-semibold text-gray-900">{stock.symbol}</div>
                      <div className="text-sm text-gray-500 truncate max-w-32">{stock.name}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold">{formatPrice(stock.current_price)}</div>
                      <div className="text-sm text-green-600 flex items-center">
                        <ArrowUpRight className="h-3 w-3 mr-1" />
                        {formatPercentage(stock.price_change_percent)}
                      </div>
                    </div>
                  </div>
                )) || (
                  <div className="p-8 text-center text-gray-500">
                    <p>No gainers data available</p>
                  </div>
                )
              )}
            </div>
          </div>

          {/* Top Losers */}
          <div>
            <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
              <ArrowDownRight className="h-5 w-5 text-red-600 mr-2" />
              Top Losers
            </h3>
            <div className="bg-white rounded-lg shadow overflow-hidden">
              {losersLoading ? (
                <div className="p-4 animate-pulse">
                  {Array.from({ length: 10 }).map((_, i) => (
                    <div key={i} className="flex justify-between items-center py-2 border-b border-gray-100 last:border-b-0">
                      <div className="h-4 bg-gray-200 rounded w-1/3"></div>
                      <div className="h-4 bg-gray-200 rounded w-1/4"></div>
                    </div>
                  ))}
                </div>
              ) : (
                losersData?.losers?.map((stock) => (
                  <div key={stock.symbol} 
                       className="flex justify-between items-center p-4 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                       onClick={() => {
                         if (stock.symbol && stock.symbol !== 'undefined') {
                           navigate(`/stock/${stock.symbol}`)
                         }
                       }}>
                    <div>
                      <div className="font-semibold text-gray-900">{stock.symbol}</div>
                      <div className="text-sm text-gray-500 truncate max-w-32">{stock.name}</div>
                    </div>
                    <div className="text-right">
                      <div className="font-semibold">{formatPrice(stock.current_price)}</div>
                      <div className="text-sm text-red-600 flex items-center">
                        <ArrowDownRight className="h-3 w-3 mr-1" />
                        {formatPercentage(stock.price_change_percent)}
                      </div>
                    </div>
                  </div>
                )) || (
                  <div className="p-8 text-center text-gray-500">
                    <p>No losers data available</p>
                  </div>
                )
              )}
            </div>
          </div>
        </div>
      </section>

      {/* Trending Stocks */}
      <section>
        <h3 className="text-xl font-bold text-gray-900 mb-4 flex items-center">
          <TrendingUp className="h-5 w-5 text-blue-600 mr-2" />
          Trending Stocks
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {trendingLoading ? (
            Array.from({ length: 3 }).map((_, i) => (
              <div key={i} className="bg-white rounded-lg shadow p-6 animate-pulse">
                <div className="h-4 bg-gray-200 rounded mb-2"></div>
                <div className="h-6 bg-gray-200 rounded mb-2"></div>
                <div className="h-4 bg-gray-200 rounded"></div>
              </div>
            ))
          ) : trendingError ? (
            <div className="col-span-full bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
              <p className="text-yellow-600">Unable to load trending stocks. Please try again later.</p>
            </div>
          ) : (
            trendingData?.trending_stocks?.slice(0, 3).map((stock) => (
              <div key={stock.symbol} 
                   className="bg-white rounded-lg shadow hover:shadow-md transition-shadow p-6 cursor-pointer"
                   onClick={() => {
                     if (stock.symbol && stock.symbol !== 'undefined') {
                       navigate(`/stock/${stock.symbol}`)
                     }
                   }}>
                <div className="flex justify-between items-start mb-2">
                  <div>
                    <h4 className="font-bold text-lg text-gray-900">{stock.symbol}</h4>
                    <p className="text-sm text-gray-600 truncate">{stock.name}</p>
                  </div>
                  <div className={`p-2 rounded-full ${stock.price_change_percent >= 0 ? 'bg-green-100' : 'bg-red-100'}`}>
                    {getChangeIcon(stock.price_change_percent)}
                  </div>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-2xl font-bold">{formatPrice(stock.current_price)}</span>
                  <span className={`text-sm font-medium ${getChangeColor(stock.price_change_percent)}`}>
                    {formatPercentage(stock.price_change_percent)}
                  </span>
                </div>
              </div>
            )) || (
              <div className="col-span-full bg-gray-50 rounded-lg p-8 text-center">
                <p className="text-gray-500">No trending stocks available</p>
              </div>
            )
          )}
        </div>
      </section>

      {/* Feature Boxes */}
      <section>
        <h2 className="text-2xl font-bold text-gray-900 text-center mb-8">
          Powerful Investment Tools
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Watchlist */}
          <Link to="/watchlist" className="group">
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-8 border-2 border-transparent hover:border-blue-200">
              <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-blue-200 transition-colors">
                <Eye className="h-8 w-8 text-blue-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2 text-center">Watchlist</h3>
              <p className="text-gray-600 text-center">Track your favorite stocks and monitor their performance in real-time</p>
            </div>
          </Link>

          {/* Portfolio Management */}
          <Link to="/portfolio" className="group">
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-8 border-2 border-transparent hover:border-green-200">
              <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-green-200 transition-colors">
                <Briefcase className="h-8 w-8 text-green-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2 text-center">Portfolio Management</h3>
              <p className="text-gray-600 text-center">Manage your investments and track portfolio performance</p>
            </div>
          </Link>

          {/* Stock Predictor */}
          <div className="group cursor-pointer" onClick={() => navigate('/ai-predictor')}>
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-8 border-2 border-transparent hover:border-purple-200">
              <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-purple-200 transition-colors">
                <Activity className="h-8 w-8 text-purple-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2 text-center">Stock Predictor</h3>
              <p className="text-gray-600 text-center">Get 7-day price predictions using advanced AI models</p>
            </div>
          </div>

          {/* Authentication */}
          <Link to="/login" className="group">
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-8 border-2 border-transparent hover:border-indigo-200">
              <div className="w-16 h-16 bg-indigo-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-indigo-200 transition-colors">
                <Shield className="h-8 w-8 text-indigo-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2 text-center">Authentication</h3>
              <p className="text-gray-600 text-center">Secure sign-in and account management</p>
            </div>
          </Link>

          {/* Stock Comparison */}
          <Link to="/compare" className="group">
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-8 border-2 border-transparent hover:border-orange-200">
              <div className="w-16 h-16 bg-orange-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-orange-200 transition-colors">
                <BarChart className="h-8 w-8 text-orange-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2 text-center">Stock Comparison</h3>
              <p className="text-gray-600 text-center">Compare multiple stocks side by side</p>
            </div>
          </Link>

          {/* Market News */}
          <Link to="/news" className="group">
            <div className="bg-white rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-8 border-2 border-transparent hover:border-red-200">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4 group-hover:bg-red-200 transition-colors">
                <Newspaper className="h-8 w-8 text-red-600" />
              </div>
              <h3 className="text-xl font-semibold text-gray-900 mb-2 text-center">Market News</h3>
              <p className="text-gray-600 text-center">Stay updated with the latest market news and trends</p>
            </div>
          </Link>
        </div>
      </section>
    </div>
  )
}

export default Home
