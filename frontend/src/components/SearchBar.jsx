import React, { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, X } from 'lucide-react'
import { useQuery } from 'react-query'
import { stockApi } from '../services/api'
import { debounce } from '../utils/helpers'

const SearchBar = ({ 
  placeholder = 'Search stocks...', 
  className = '', 
  onSelectStock = null // Optional callback for stock selection
}) => {
  const [query, setQuery] = useState('')
  const [isOpen, setIsOpen] = useState(false)
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const inputRef = useRef(null)
  const dropdownRef = useRef(null)
  const navigate = useNavigate()

  // Debounce search query - optimized for instant suggestions
  const debouncedSearch = debounce((value) => {
    setDebouncedQuery(value)
  }, 50) // Reduced from 100ms to 50ms for faster suggestions

  useEffect(() => {
    debouncedSearch(query)
  }, [query])

  // Close dropdown when query is empty and not focused
  useEffect(() => {
    if (!query && !debouncedQuery) {
      setIsOpen(false)
    }
  }, [query, debouncedQuery])

  // Search results query - optimized for faster responses
  const { data: searchResults, isLoading } = useQuery(
    ['search', debouncedQuery],
    () => stockApi.search(debouncedQuery, 8),
    {
      enabled: debouncedQuery.length >= 1,
      select: (response) => response.data.results,
      staleTime: 120 * 1000, // Increased to 2 minutes for better caching
      cacheTime: 10 * 60 * 1000, // 10 minutes cache
      refetchOnWindowFocus: false, // Don't refetch on window focus
      retry: 1, // Only retry once to avoid delays
    }
  )

  // Trending stocks query for when search is empty
  const { data: trendingStocks, isLoading: trendingLoading } = useQuery(
    ['trending'],
    () => stockApi.getTrending(8),
    {
      select: (response) => response.data.trending_stocks,
      staleTime: 5 * 60 * 1000, // 5 minutes
    }
  )

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target) &&
        !inputRef.current.contains(event.target)
      ) {
        setIsOpen(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  const handleInputChange = (e) => {
    const value = e.target.value
    setQuery(value)
    setIsOpen(true) // Always show on input change
  }

  const handleSelectStock = (stock) => {
    setQuery('')
    setDebouncedQuery('') // Clear debounced query to remove search results
    setIsOpen(false)
    
    // Remove focus from input to prevent dropdown from reopening
    if (inputRef.current) {
      inputRef.current.blur()
    }
    
    // Validate stock symbol before navigation
    const stockSymbol = typeof stock === 'string' ? stock : stock.symbol
    if (!stockSymbol || stockSymbol === 'undefined' || stockSymbol.trim() === '') {
      console.error('Invalid stock symbol:', stockSymbol)
      return
    }
    
    // If callback provided, use it instead of navigation
    if (onSelectStock) {
      onSelectStock(stockSymbol)
    } else {
      navigate(`/stock/${stockSymbol}`)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    if (query.trim()) {
      // If there are search results, select the first one
      if (searchResults && searchResults.length > 0) {
        handleSelectStock(searchResults[0])
      } else {
        // Otherwise, try to use the query as a symbol
        const stockSymbol = query.toUpperCase().trim()
        if (stockSymbol && stockSymbol !== 'UNDEFINED') {
          if (onSelectStock) {
            setQuery('')
            setDebouncedQuery('') // Clear debounced query to remove search results
            setIsOpen(false)
            onSelectStock(stockSymbol)
          } else {
            setQuery('')
            setDebouncedQuery('') // Clear debounced query to remove search results
            navigate(`/stock/${stockSymbol}`)
          }
        }
      }
    }
  }

  const clearSearch = () => {
    setQuery('')
    setDebouncedQuery('') // Clear debounced query to remove search results
    setIsOpen(false)
    inputRef.current?.focus()
  }

  return (
    <div className={`relative ${className}`}>
      <form onSubmit={handleSubmit}>
        <div className="relative">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Search className="h-5 w-5 text-gray-400" />
          </div>
          
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={handleInputChange}
            onFocus={() => setIsOpen(true)}
            placeholder={placeholder}
            className="w-full pl-10 pr-10 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white"
            autoComplete="off"
          />
          
          {query && (
            <button
              type="button"
              onClick={clearSearch}
              className="absolute inset-y-0 right-0 pr-3 flex items-center"
            >
              <X className="h-5 w-5 text-gray-400 hover:text-gray-600" />
            </button>
          )}
        </div>
      </form>

      {/* Search Results Dropdown */}
      {isOpen && (
        <div
          ref={dropdownRef}
          className="absolute z-10 w-full mt-1 bg-white border border-gray-200 rounded-lg shadow-lg max-h-96 overflow-y-auto"
        >
          {query.length > 0 ? (
            // Show search results when user is typing
            isLoading ? (
              <div className="px-4 py-3 text-center text-gray-500">
                <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                <span>Searching...</span>
              </div>
            ) : searchResults && searchResults.length > 0 ? (
              <ul className="py-2">
                {searchResults.map((stock) => (
                  <li key={stock.symbol}>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation()
                        handleSelectStock(stock)
                      }}
                      className="w-full px-4 py-3 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none transition-colors"
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center space-x-3">
                            <div className="font-semibold text-gray-900">
                              {stock.symbol}
                            </div>
                            <div className="text-sm text-gray-600 truncate">
                              {stock.name}
                            </div>
                          </div>
                          {stock.sector && (
                            <div className="text-xs text-gray-500 mt-1">
                              {stock.sector} • {stock.industry}
                            </div>
                          )}
                        </div>
                        {stock.current_price > 0 && (
                          <div className="text-right ml-4">
                            <div className="font-semibold text-gray-900">
                              ${stock.current_price.toFixed(2)}
                            </div>
                          </div>
                        )}
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            ) : debouncedQuery.length > 0 ? (
              <div className="px-4 py-3 text-center text-gray-500">
                No results found for "{debouncedQuery}"
                <div className="mt-2">
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation()
                      handleSelectStock({ symbol: debouncedQuery.toUpperCase() })
                    }}
                    className="text-blue-600 hover:text-blue-700 font-medium"
                  >
                    Search for "{debouncedQuery.toUpperCase()}" anyway
                  </button>
                </div>
              </div>
            ) : null
          ) : (
            // Show trending stocks when search is empty
            trendingLoading ? (
              <div className="px-4 py-3 text-center text-gray-500">
                <div className="inline-block animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600 mr-2"></div>
                <span>Loading trending stocks...</span>
              </div>
            ) : trendingStocks && trendingStocks.length > 0 ? (
              <div>
                <div className="px-4 py-2 border-b border-gray-100">
                  <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                    📈 Trending Stocks
                  </div>
                </div>
                <ul className="py-2">
                  {trendingStocks.map((stock) => (
                    <li key={stock.symbol}>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          handleSelectStock(stock)
                        }}
                        className="w-full px-4 py-3 text-left hover:bg-gray-50 focus:bg-gray-50 focus:outline-none transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-3">
                              <div className="font-semibold text-gray-900">
                                {stock.symbol}
                              </div>
                              <div className="text-sm text-gray-600 truncate">
                                {stock.name}
                              </div>
                            </div>
                            {stock.sector && (
                              <div className="text-xs text-gray-500 mt-1">
                                {stock.sector} • {stock.industry}
                              </div>
                            )}
                          </div>
                          <div className="text-right ml-4">
                            {stock.current_price > 0 && (
                              <div className="font-semibold text-gray-900">
                                ${stock.current_price.toFixed(2)}
                              </div>
                            )}
                            {stock.price_change_percent && (
                              <div className={`text-xs font-medium ${
                                stock.price_change_percent >= 0 
                                  ? 'text-green-600' 
                                  : 'text-red-600'
                              }`}>
                                {stock.price_change_percent >= 0 ? '+' : ''}
                                {stock.price_change_percent.toFixed(2)}%
                              </div>
                            )}
                          </div>
                        </div>
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            ) : (
              <div className="px-4 py-3 text-center text-gray-500">
                <div className="text-sm">Start typing to search for stocks</div>
              </div>
            )
          )}
        </div>
      )}
    </div>
  )
}

export default SearchBar
