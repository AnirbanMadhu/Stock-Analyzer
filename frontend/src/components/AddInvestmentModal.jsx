import React, { useState, useEffect } from 'react'
import { useMutation, useQueryClient, useQuery } from 'react-query'
import { useForm } from 'react-hook-form'
import { X, DollarSign, Hash, Calendar, TrendingUp, Search } from 'lucide-react'
import { portfolioApi, stockApi } from '../services/api'
import LoadingSpinner from './LoadingSpinner'
import toast from 'react-hot-toast'

const AddInvestmentModal = ({ isOpen, onClose, stockSymbol, stockPrice = 0, stockName = '' }) => {
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState('')
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [selectedStock, setSelectedStock] = useState(null)
  const [fetchingPrice, setFetchingPrice] = useState(false)
  const [priceDebounceTimeout, setPriceDebounceTimeout] = useState(null)
  
  const {
    register,
    handleSubmit,
    reset,
    watch,
    setValue,
    formState: { errors, isValid }
  } = useForm({
    defaultValues: {
      symbol: '',
      quantity: '',
      purchasePrice: '',
      purchaseDate: new Date().toISOString().split('T')[0]
    }
  })

  // Initialize form when modal opens with pre-selected stock
  useEffect(() => {
    if (isOpen && stockSymbol) {
      setSearchQuery(stockSymbol)
      setValue('symbol', stockSymbol, { shouldValidate: true })
      if (stockPrice) {
        setValue('purchasePrice', stockPrice.toString(), { shouldValidate: true })
      }
      if (stockName) {
        setSelectedStock({ symbol: stockSymbol, name: stockName, price: stockPrice })
      }
    } else if (isOpen) {
      // Reset form when opening without pre-selected stock
      setSearchQuery('')
      setSelectedStock(null)
      reset({
        symbol: '',
        quantity: '',
        purchasePrice: '',
        purchaseDate: new Date().toISOString().split('T')[0]
      })
    }
  }, [isOpen, stockSymbol, stockPrice, stockName, setValue, reset])

  // Watch for symbol changes from form
  const symbolValue = watch('symbol')
  useEffect(() => {
    if (symbolValue !== searchQuery) {
      setSearchQuery(symbolValue || '')
    }
  }, [symbolValue])

  const quantity = watch('quantity')
  const purchasePrice = watch('purchasePrice')
  const totalValue = quantity && purchasePrice ? (parseFloat(quantity) * parseFloat(purchasePrice)) : 0

  // Search suggestions query - optimized for instant results
  const { data: searchResults, isLoading: searchLoading } = useQuery(
    ['stockSearch', searchQuery],
    () => stockApi.search(searchQuery, 8),
    {
      enabled: searchQuery.length >= 1,
      staleTime: 300000, // 5 minutes
      cacheTime: 600000, // 10 minutes cache
      retry: 1,
      refetchOnWindowFocus: false,
    }
  )

  const addInvestmentMutation = useMutation(
    (investmentData) => portfolioApi.addHolding(investmentData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('portfolioHoldings')
        toast.success('Investment added to portfolio!')
        reset()
        setSearchQuery('')
        setSelectedStock(null)
        onClose()
      },
      onError: (error) => {
        console.error('Add investment error:', error)
        console.error('Error response:', error.response)
        const errorMessage = error.response?.data?.error || error.response?.data?.message || 'Failed to add investment'
        toast.error(errorMessage)
      }
    }
  )

  const handleSymbolChange = (e) => {
    const value = e.target.value.toUpperCase()
    console.log('Symbol changed to:', value)
    setSearchQuery(value)
    setValue('symbol', value, { shouldValidate: true })
    setShowSuggestions(value.length >= 1)
    setSelectedStock(null)
    
    // Clear existing timeout
    if (priceDebounceTimeout) {
      clearTimeout(priceDebounceTimeout)
    }
    
    // Auto-fetch price when user types a complete symbol (3+ characters) with debounce
    if (value.length >= 3) {
      console.log('Setting up debounce for:', value)
      const timeout = setTimeout(() => {
        console.log('Debounce triggered for:', value)
        fetchCurrentPrice(value)
      }, 800) // Wait 800ms after user stops typing
      setPriceDebounceTimeout(timeout)
    }
  }

  const fetchCurrentPrice = async (symbol) => {
    if (fetchingPrice) return // Prevent multiple simultaneous requests
    
    console.log('Fetching price for symbol:', symbol)
    setFetchingPrice(true)
    try {
      const stockDetails = await stockApi.getDetails(symbol)
      console.log('Stock details response:', stockDetails)
      
      // Access the nested data structure: response.data.data.current_price
      const stockData = stockDetails?.data?.data
      if (stockData?.current_price && stockData.current_price > 0) {
        const price = parseFloat(stockData.current_price).toFixed(2)
        console.log('Setting price to:', price)
        setValue('purchasePrice', price, { shouldValidate: true })
        // Update selected stock with price info
        setSelectedStock({
          symbol: symbol,
          name: stockData.name || symbol,
          current_price: stockData.current_price
        })
        toast.success(`Price updated: $${price}`)
      } else {
        console.log('No valid price found in response')
        toast.error('Could not fetch current price')
      }
    } catch (error) {
      console.error('Error fetching price:', error)
      toast.error('Failed to fetch current price')
      // Clear price if stock is invalid
      if (error.response?.status === 404) {
        setValue('purchasePrice', '', { shouldValidate: false })
      }
    } finally {
      setFetchingPrice(false)
    }
  }

  const handleSelectStock = async (stock) => {
    setSearchQuery(stock.symbol)
    setValue('symbol', stock.symbol, { shouldValidate: true })
    setSelectedStock(stock)
    setShowSuggestions(false)
    
    // Always try to fetch the latest current price for accuracy
    await fetchCurrentPrice(stock.symbol)
  }

  const handleSymbolBlur = () => {
    // Delay hiding suggestions to allow clicking on them
    setTimeout(() => setShowSuggestions(false), 150)
    
    // Fetch price when user finishes entering symbol (on blur)
    const currentSymbol = watch('symbol')
    if (currentSymbol && currentSymbol.length >= 3 && !selectedStock) {
      fetchCurrentPrice(currentSymbol)
    }
  }

  const onSubmit = (data) => {
    const investmentData = {
      symbol: data.symbol.toUpperCase(),
      quantity: parseFloat(data.quantity),
      purchase_price: parseFloat(data.purchasePrice),
      purchase_date: data.purchaseDate
    }

    console.log('Submitting investment data:', investmentData)
    addInvestmentMutation.mutate(investmentData)
  }

  const handleClose = () => {
    if (priceDebounceTimeout) {
      clearTimeout(priceDebounceTimeout)
    }
    reset()
    setSearchQuery('')
    setSelectedStock(null)
    setShowSuggestions(false)
    setFetchingPrice(false)
    setPriceDebounceTimeout(null)
    onClose()
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-xl shadow-2xl max-w-md w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-100 rounded-lg">
              <TrendingUp className="h-5 w-5 text-blue-600" />
            </div>
            <div>
              <h3 className="text-xl font-bold text-gray-900">Add Investment</h3>
              <p className="text-sm text-gray-500">Add a stock to your portfolio</p>
            </div>
          </div>
          <button
            onClick={handleClose}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-gray-400" />
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="p-6 space-y-6">
          {/* Stock Symbol */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="block text-sm font-medium text-gray-700">
                Stock Symbol *
              </label>
              {watch('symbol') && watch('symbol').length >= 3 && (
                <button
                  type="button"
                  onClick={() => fetchCurrentPrice(watch('symbol'))}
                  disabled={fetchingPrice}
                  className="text-blue-600 hover:text-blue-700 text-sm font-medium disabled:opacity-50"
                >
                  {fetchingPrice ? 'Fetching...' : 'Get Price'}
                </button>
              )}
            </div>
            <div className="relative">
              <Hash className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="text"
                placeholder="e.g., AAPL"
                onChange={handleSymbolChange}
                onFocus={() => searchQuery.length >= 1 && setShowSuggestions(true)}
                onBlur={handleSymbolBlur}
                className={`w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                  errors.symbol ? 'border-red-300' : 'border-gray-300'
                }`}
                {...register('symbol', {
                  required: 'Stock symbol is required',
                  pattern: {
                    value: /^[A-Za-z]+$/,
                    message: 'Please enter a valid stock symbol'
                  }
                })}
              />
              
              {/* Search Suggestions Dropdown */}
              {showSuggestions && searchQuery.length >= 1 && (
                <div className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto">
                  {searchLoading ? (
                    <div className="p-3 text-center">
                      <LoadingSpinner size="small" />
                    </div>
                  ) : searchResults?.data?.results?.length > 0 ? (
                    <>
                      {searchResults.data.results.map((stock, index) => (
                        <button
                          key={index}
                          type="button"
                          onClick={() => handleSelectStock(stock)}
                          className="w-full px-4 py-3 text-left hover:bg-gray-50 border-b border-gray-100 last:border-b-0 focus:bg-blue-50 focus:outline-none"
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <div className="font-semibold text-gray-900">{stock.symbol}</div>
                              <div className="text-sm text-gray-600 truncate">{stock.name}</div>
                            </div>
                            {stock.current_price && stock.current_price > 0 && (
                              <div className="text-right text-sm">
                                <div className="font-medium text-gray-900">${parseFloat(stock.current_price).toFixed(2)}</div>
                              </div>
                            )}
                          </div>
                        </button>
                      ))}
                    </>
                  ) : (
                    <div className="p-3 text-center text-gray-500 text-sm">
                      {searchQuery.length >= 1 ? 'No stocks found' : 'Type to search...'}
                    </div>
                  )}
                </div>
              )}
            </div>
            {errors.symbol && (
              <p className="mt-1 text-sm text-red-600">{errors.symbol.message}</p>
            )}
            {selectedStock && (
              <p className="mt-1 text-sm text-gray-500">{selectedStock.name}</p>
            )}
          </div>

          {/* Quantity */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Quantity (Shares) *
            </label>
            <div className="relative">
              <Hash className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="number"
                step="0.001"
                min="0"
                placeholder="0"
                className={`w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                  errors.quantity ? 'border-red-300' : 'border-gray-300'
                }`}
                {...register('quantity', {
                  required: 'Quantity is required',
                  min: {
                    value: 0.001,
                    message: 'Quantity must be greater than 0'
                  }
                })}
              />
            </div>
            {errors.quantity && (
              <p className="mt-1 text-sm text-red-600">{errors.quantity.message}</p>
            )}
          </div>

          {/* Purchase Price */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Purchase Price per Share * 
              {fetchingPrice && <span className="text-blue-600 text-xs ml-2">(Fetching current price...)</span>}
              {selectedStock && !fetchingPrice && (
                <button
                  type="button"
                  onClick={() => fetchCurrentPrice(selectedStock.symbol)}
                  className="text-blue-600 hover:text-blue-700 text-xs ml-2 underline"
                  title="Refresh current price"
                >
                  Refresh Price
                </button>
              )}
            </label>
            <div className="relative">
              <DollarSign className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              {fetchingPrice && (
                <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                  <LoadingSpinner size="small" />
                </div>
              )}
              <input
                type="number"
                step="0.01"
                min="0"
                placeholder={fetchingPrice ? "Fetching..." : "0.00"}
                disabled={fetchingPrice}
                className={`w-full pl-10 ${fetchingPrice ? 'pr-12' : 'pr-4'} py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                  errors.purchasePrice ? 'border-red-300' : 'border-gray-300'
                } ${fetchingPrice ? 'bg-gray-50' : ''}`}
                {...register('purchasePrice', {
                  required: 'Purchase price is required',
                  min: {
                    value: 0.01,
                    message: 'Price must be greater than 0'
                  }
                })}
              />
            </div>
            {errors.purchasePrice && (
              <p className="mt-1 text-sm text-red-600">{errors.purchasePrice.message}</p>
            )}
            {selectedStock && selectedStock.current_price && !fetchingPrice && (
              <p className="mt-1 text-sm text-green-600">
                ✓ Current market price: ${parseFloat(selectedStock.current_price).toFixed(2)}
              </p>
            )}
          </div>

          {/* Purchase Date */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Purchase Date *
            </label>
            <div className="relative">
              <Calendar className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
              <input
                type="date"
                max={new Date().toISOString().split('T')[0]}
                className={`w-full pl-10 pr-4 py-3 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent ${
                  errors.purchaseDate ? 'border-red-300' : 'border-gray-300'
                }`}
                {...register('purchaseDate', {
                  required: 'Purchase date is required'
                })}
              />
            </div>
            {errors.purchaseDate && (
              <p className="mt-1 text-sm text-red-600">{errors.purchaseDate.message}</p>
            )}
          </div>

          {/* Total Value Display */}
          {totalValue > 0 && (
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-blue-800">Total Investment</span>
                <span className="text-lg font-bold text-blue-900">
                  ${totalValue.toFixed(2)}
                </span>
              </div>
              <div className="text-xs text-blue-600 mt-1">
                {quantity} shares × ${purchasePrice}
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-3 pt-4">
            <button
              type="button"
              onClick={handleClose}
              className="flex-1 px-4 py-3 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors font-medium"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={!isValid || addInvestmentMutation.isLoading}
              className="flex-1 flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed transition-colors font-medium"
            >
              {addInvestmentMutation.isLoading ? (
                <>
                  <LoadingSpinner size="small" className="mr-2" />
                  Adding...
                </>
              ) : (
                'Add Investment'
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default AddInvestmentModal
