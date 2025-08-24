import React from 'react'
import { Newspaper, ExternalLink, TrendingUp } from 'lucide-react'

const News = () => {
  const handleTabClick = (tabId) => {
    if (tabId === 'market') {
      // Redirect directly to Yahoo Finance News
      window.open('https://finance.yahoo.com/news/', '_blank')
      return
    }
    if (tabId === 'trending') {
      // Redirect directly to Yahoo Finance Trending
      window.open('https://finance.yahoo.com/trending-tickers/', '_blank')
      return
    }
  }



  return (
    <div className="space-y-6">
      {/* Content */}
      <div className="min-h-96">
        <div className="bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg p-8 text-center">
          <div className="max-w-2xl mx-auto">
            <Newspaper className="h-16 w-16 text-blue-600 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-900 mb-4">
              Financial News & Market Insights
            </h2>
            <p className="text-gray-600 mb-6">
              Access the latest market news and trending financial information through our integrated external sources.
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <Newspaper className="h-8 w-8 text-green-600 mb-3" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Market News</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Stay updated with comprehensive market coverage and breaking financial news.
                </p>
                <button
                  onClick={() => handleTabClick('market')}
                  className="w-full bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 transition-colors flex items-center justify-center"
                >
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Market News
                </button>
              </div>
              <div className="bg-white rounded-lg p-6 shadow-sm border border-gray-200">
                <TrendingUp className="h-8 w-8 text-blue-600 mb-3" />
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Trending Stocks</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Discover the most active and trending stocks in the market right now.
                </p>
                <button
                  onClick={() => handleTabClick('trending')}
                  className="w-full bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center"
                >
                  <ExternalLink className="h-4 w-4 mr-2" />
                  View Trending Stocks
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default News
