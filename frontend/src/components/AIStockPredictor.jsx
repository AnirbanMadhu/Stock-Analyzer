import React, { useState, useEffect, useRef } from 'react';
import { Search, TrendingUp, Brain, BarChart3, AlertCircle, CheckCircle, Clock } from 'lucide-react';
import StockPredictionChart from './StockPredictionChart';
import ModelComparisonChart from './ModelComparisonChart';
import { aiApi } from '../services/api';

const AIStockPredictor = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [selectedStock, setSelectedStock] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [loading, setLoading] = useState(false);
  const [comparingModels, setComparingModels] = useState(false);
  const [comparisonProgress, setComparisonProgress] = useState(0);
  const [aiModelsStatus, setAiModelsStatus] = useState(null);
  const [selectedModel, setSelectedModel] = useState('arima');
  const [predictionDays, setPredictionDays] = useState(7);
  const prevParamsRef = useRef({ selectedModel: 'arima', predictionDays: 7 });

  // Load AI models status on component mount
  useEffect(() => {
    loadAiModelsStatus();
  }, []);

  // Re-analyze when model or prediction days change for selected stock
  useEffect(() => {
    const prevParams = prevParamsRef.current;
    const currentParams = { selectedModel, predictionDays };
    
    // Check if parameters actually changed and we have a selected stock
    if (selectedStock && 
        (prevParams.selectedModel !== currentParams.selectedModel || 
         prevParams.predictionDays !== currentParams.predictionDays) &&
        !loading) {
      
      const reAnalyzeStock = async () => {
        try {
          setLoading(true);
          setAnalysis(null);
          setModelComparison(null);

          const response = await aiApi.analyze(selectedStock, predictionDays, selectedModel);
          if (response.status === 200) {
            setAnalysis(response.data);
          }
        } catch (error) {
          console.error('Error re-analyzing stock:', error);
        } finally {
          setLoading(false);
        }
      };
      
      reAnalyzeStock();
    }
    
    // Update previous params
    prevParamsRef.current = currentParams;
  }, [selectedModel, predictionDays, selectedStock, loading]);

  const loadAiModelsStatus = async () => {
    try {
      const response = await aiApi.getModelsStatus();
      if (response.status === 200) {
        setAiModelsStatus(response.data);
      }
    } catch (error) {
      console.error('Error loading AI models status:', error);
    }
  };

  const searchStocks = async (query) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      setLoading(true);
      const response = await aiApi.search(query);
      if (response.status === 200) {
        setSearchResults(response.data.results || []);
      }
    } catch (error) {
      console.error('Error searching stocks:', error);
    } finally {
      setLoading(false);
    }
  };

  const analyzeStock = async (symbol) => {
    try {
      setLoading(true);
      setSelectedStock(symbol);
      setAnalysis(null);
      setModelComparison(null);
      setSearchResults([]); // Clear search results when a stock is selected
      setSearchQuery(''); // Clear search query

      const response = await aiApi.analyze(symbol, predictionDays, selectedModel);
      if (response.status === 200) {
        setAnalysis(response.data);
        // Debug log to see what data we're getting
        console.log('Analysis data received:', response.data);
        console.log('Stock info:', response.data.stock_info);
      }
    } catch (error) {
      console.error('Error analyzing stock:', error);
    } finally {
      setLoading(false);
    }
  };

  const compareModels = async (symbol) => {
    try {
      setComparingModels(true);
      setModelComparison(null); // Clear previous comparison
      setComparisonProgress(0);
      
      // Progress simulation
      const progressInterval = setInterval(() => {
        setComparisonProgress(prev => {
          if (prev >= 90) return prev;
          return prev + Math.random() * 10;
        });
      }, 1000);
      
      // Reduce days for faster comparison (max 5 days for quick comparison)
      const comparisonDays = Math.min(predictionDays, 5);
      
      const response = await aiApi.compareModels(symbol, comparisonDays);
      
      clearInterval(progressInterval);
      setComparisonProgress(100);
      
      if (response.status === 200) {
        setModelComparison(response.data);
      } else {
        console.error('Model comparison failed:', response.data);
        // Show a user-friendly error
        alert('Model comparison timed out or failed. Please try again with fewer prediction days.');
      }
    } catch (error) {
      console.error('Error comparing models:', error);
      
      // Handle timeout specifically
      if (error.message?.includes('timeout') || error.code === 'ECONNABORTED') {
        alert('Model comparison is taking too long. Try reducing the prediction days or try again later.');
      } else {
        alert('Failed to compare models. Please try again.');
      }
    } finally {
      setComparingModels(false);
      setComparisonProgress(0);
    }
  };

  const handleSearchInputChange = (e) => {
    const query = e.target.value;
    setSearchQuery(query);
    
    // Clear results if query is empty
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    
    // Debounced search
    clearTimeout(window.searchTimeout);
    window.searchTimeout = setTimeout(() => {
      searchStocks(query);
    }, 500);
  };

  const getModelIcon = (modelName) => {
    if (modelName?.includes('LSTM')) return <Brain className="w-4 h-4" />;
    if (modelName?.includes('ARIMA')) return <TrendingUp className="w-4 h-4" />;
    return <BarChart3 className="w-4 h-4" />;
  };

  const getRecommendationColor = (recommendation) => {
    const rec = recommendation?.toLowerCase() || '';
    if (rec.includes('buy')) return 'text-green-600 bg-green-50';
    if (rec.includes('sell')) return 'text-red-600 bg-red-50';
    return 'text-yellow-600 bg-yellow-50';
  };

  const getRiskColor = (risk) => {
    const riskLevel = risk?.toLowerCase() || '';
    if (riskLevel === 'low') return 'text-green-600 bg-green-50';
    if (riskLevel === 'high') return 'text-red-600 bg-red-50';
    return 'text-yellow-600 bg-yellow-50';
  };

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-6">
      {/* Header */}
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">
          AI Stock Predictor
        </h1>
        <p className="text-gray-600">
          Search stocks and get AI-powered predictions using ARIMA and LSTM models
        </p>
      </div>

      {/* AI Models Status */}
      {aiModelsStatus && (
        <div className="bg-white rounded-lg shadow-md p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
            <Brain className="w-5 h-5" />
            Available AI Models
          </h2>
          <div className="grid md:grid-cols-3 gap-4">
            {Object.entries(aiModelsStatus.models || {}).map(([key, model]) => (
              <div key={key} className="border rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  {model.available ? (
                    <CheckCircle className="w-5 h-5 text-green-500" />
                  ) : (
                    <AlertCircle className="w-5 h-5 text-red-500" />
                  )}
                  <h3 className="font-semibold">{model.name}</h3>
                </div>
                <p className="text-sm text-gray-600 mb-2">{model.description}</p>
                <div className="text-xs space-y-1">
                  <div>Best for: {model.best_for}</div>
                  <div>Accuracy: {model.accuracy_grade}</div>
                  <div>Speed: {model.computational_cost}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Search Section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex gap-4 mb-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search for stocks (e.g., Apple, AAPL, Tesla)..."
              value={searchQuery}
              onChange={handleSearchInputChange}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            />
          </div>
          <select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value="arima">ARIMA</option>
            <option value="lstm">LSTM</option>
          </select>
          <select
            value={predictionDays}
            onChange={(e) => setPredictionDays(parseInt(e.target.value))}
            className="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          >
            <option value={3}>3 Days</option>
            <option value={7}>7 Days</option>
            <option value={14}>14 Days</option>
            <option value={30}>30 Days</option>
          </select>
        </div>

        {/* Search Results */}
        {searchResults.length > 0 && searchQuery.trim() && (
          <div className="space-y-2">
            <h3 className="font-semibold text-gray-700">Search Results:</h3>
            {searchResults.map((stock) => (
              <div
                key={stock.symbol}
                className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 cursor-pointer transition-colors"
                onClick={() => analyzeStock(stock.symbol)}
              >
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="font-bold text-blue-600">{stock.symbol}</span>
                      <span className="text-gray-700">{stock.name}</span>
                    </div>
                    <div className="text-sm text-gray-600 space-x-4">
                      <span>Price: ${stock.current_price || 'N/A'}</span>
                      <span>Change: {(() => {
                        const change = stock.price_change_percent || stock.change_percent || 0;
                        return `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                      })()}</span>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center gap-1 mb-1">
                      {stock.prediction_available ? (
                        <CheckCircle className="w-4 h-4 text-green-500" />
                      ) : (
                        <AlertCircle className="w-4 h-4 text-red-500" />
                      )}
                      <span className="text-sm">
                        {stock.prediction_available ? 'Predictable' : 'Limited Data'}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      Models: {stock.supported_models?.join(', ') || 'Basic'}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Loading Indicator for Model Comparison */}
      {comparingModels && (
        <div className="text-center py-8">
          <Clock className="w-8 h-8 animate-spin mx-auto mb-2" />
          <p className="text-gray-600">Comparing AI models... This may take a moment</p>
          <p className="text-sm text-gray-500">Running ARIMA and LSTM predictions for comparison</p>
          
          {/* Progress Bar */}
          <div className="w-full max-w-md mx-auto mt-4">
            <div className="bg-gray-200 rounded-full h-2">
              <div 
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${comparisonProgress}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-500 mt-1">
              {Math.round(comparisonProgress)}% complete
            </p>
          </div>
          
          <p className="text-xs text-gray-400 mt-2">
            Estimated time: 15-30 seconds
          </p>
        </div>
      )}

      {/* Loading Indicator */}
      {loading && !comparingModels && (
        <div className="text-center py-8">
          <Clock className="w-8 h-8 animate-spin mx-auto mb-2" />
          <p className="text-gray-600">Analyzing stock with AI models...</p>
        </div>
      )}

      {/* Analysis Results */}
      {analysis && !loading && !comparingModels && (
        <div className="space-y-6">
          {/* Stock Info Header */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="flex justify-between items-start mb-4">
              <div>
                <h2 className="text-2xl font-bold flex items-center gap-2">
                  <BarChart3 className="w-6 h-6" />
                  {analysis.symbol} Analysis
                </h2>
                <p className="text-gray-600">AI-powered stock prediction and analysis</p>
              </div>
              {analysis.prediction && !comparingModels && (
                <button
                  onClick={() => compareModels(analysis.symbol)}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
                  title="Compare ARIMA vs LSTM models (optimized for speed)"
                >
                  <Brain className="w-4 h-4" />
                  Compare Models
                </button>
              )}
              {comparingModels && (
                <button
                  disabled
                  className="px-4 py-2 bg-gray-400 text-white rounded-lg cursor-not-allowed flex items-center gap-2"
                >
                  <Clock className="w-4 h-4 animate-spin" />
                  Comparing...
                </button>
              )}
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <div className="text-sm text-gray-600 mb-1">Current Price</div>
                <div className="text-2xl font-bold text-blue-600">
                  ${analysis.stock_info?.current_price?.toFixed(2) || 'N/A'}
                </div>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg text-center">
                <div className="text-sm text-gray-600 mb-1">24h Change</div>
                <div className={`text-2xl font-bold ${(() => {
                  // Calculate change percent first to determine color
                  let changePercent = analysis.stock_info?.price_change_percent;
                  
                  // Only try alternatives if changePercent is null or undefined (not 0)
                  if (changePercent === undefined || changePercent === null) {
                    changePercent = analysis.stock_info?.change_percent ??
                                  analysis.stock_info?.regularMarketChangePercent ??
                                  analysis.prediction?.current_price_change_percent;
                  }
                  
                  // If still no percentage, try to calculate from price change and current price
                  if ((changePercent === undefined || changePercent === null) && 
                      analysis.stock_info?.price_change !== undefined && analysis.stock_info?.current_price) {
                    const priceChange = analysis.stock_info.price_change;
                    const currentPrice = analysis.stock_info.current_price;
                    const previousPrice = currentPrice - priceChange;
                    if (previousPrice > 0) {
                      changePercent = (priceChange / previousPrice) * 100;
                    }
                  }
                  
                  if (changePercent === undefined || changePercent === null) {
                    return 'text-gray-600'; // neutral color for N/A
                  }
                  
                  return changePercent >= 0 ? 'text-green-600' : 'text-red-600';
                })()}`}>
                  {(() => {
                    // Try multiple sources for price change percentage
                    let changePercent = analysis.stock_info?.price_change_percent;
                    
                    // Only look for alternatives if we don't have a value (not even 0)
                    if (changePercent === undefined || changePercent === null) {
                      changePercent = analysis.stock_info?.change_percent ??
                                    analysis.stock_info?.regularMarketChangePercent ??
                                    analysis.prediction?.current_price_change_percent;
                    }
                    
                    // If still no percentage, try to calculate from price change and current price
                    if ((changePercent === undefined || changePercent === null) && 
                        analysis.stock_info?.price_change !== undefined && analysis.stock_info?.current_price) {
                      const priceChange = analysis.stock_info.price_change;
                      const currentPrice = analysis.stock_info.current_price;
                      const previousPrice = currentPrice - priceChange;
                      if (previousPrice > 0) {
                        changePercent = (priceChange / previousPrice) * 100;
                      }
                    }
                    
                    // Only show N/A if we truly have no data (null/undefined), not if it's 0
                    if (changePercent === undefined || changePercent === null) {
                      return 'N/A';
                    }
                    
                    return `${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
                  })()}
                </div>
              </div>
              {analysis.analysis && (
                <>
                  <div className="bg-gray-50 p-4 rounded-lg text-center">
                    <div className="text-sm text-gray-600 mb-1">Recommendation</div>
                    <div className={`text-lg font-bold px-2 py-1 rounded ${
                      getRecommendationColor(analysis.analysis.recommendation)
                    }`}>
                      {analysis.analysis.recommendation}
                    </div>
                  </div>
                  <div className="bg-gray-50 p-4 rounded-lg text-center">
                    <div className="text-sm text-gray-600 mb-1">Risk Level</div>
                    <div className={`text-lg font-bold px-2 py-1 rounded ${
                      getRiskColor(analysis.analysis.risk_level)
                    }`}>
                      {analysis.analysis.risk_level}
                    </div>
                  </div>
                </>
              )}
            </div>

            {analysis.analysis?.key_insights && (
              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <h3 className="font-semibold text-blue-800 mb-2">Key Insights:</h3>
                <ul className="space-y-1">
                  {analysis.analysis.key_insights.map((insight, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-blue-700">
                      <span className="text-blue-500 mt-1">•</span>
                      <span>{insight}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {/* Prediction Chart */}
          {analysis.prediction ? (
            <div className="bg-white rounded-lg shadow-md p-6">
              <h3 className="text-xl font-bold mb-4 flex items-center gap-2">
                {getModelIcon(analysis.prediction?.model)}
                AI Prediction Chart
              </h3>
              <StockPredictionChart analysis={analysis} selectedModel={selectedModel} predictionDays={predictionDays} />
            </div>
          ) : (
            <div className="bg-white rounded-lg shadow-md p-6">
              <div className="text-center py-8 text-gray-500">
                <AlertCircle className="w-12 h-12 mx-auto mb-2" />
                <p>No prediction available for this stock</p>
                <p className="text-sm">Insufficient historical data or unsupported symbol</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Model Comparison */}
      {modelComparison && !loading && !comparingModels && (
        <div className="bg-white rounded-lg shadow-md p-6">
          <ModelComparisonChart 
            modelComparison={modelComparison} 
            symbol={selectedStock} 
          />
        </div>
      )}
    </div>
  );
};

export default AIStockPredictor;
