import React from 'react'
import { AlertTriangle, TrendingDown, DollarSign, Clock, Shield } from 'lucide-react'

const RiskDisclosure = () => {
  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg p-8">
        {/* Header */}
        <div className="flex items-center space-x-3 mb-8">
          <AlertTriangle className="h-8 w-8 text-red-500" />
          <h1 className="text-3xl font-bold text-gray-900">Risk Disclosure</h1>
        </div>

        {/* Important Notice */}
        <div className="bg-red-50 border-l-4 border-red-500 p-6 mb-8">
          <div className="flex items-center space-x-2 mb-2">
            <AlertTriangle className="h-5 w-5 text-red-500" />
            <h2 className="text-lg font-semibold text-red-700">Important Notice</h2>
          </div>
          <p className="text-red-700">
            All investments involve risk, including the potential loss of principal. Past performance does not guarantee future results. 
            Please read this disclosure carefully before using our platform.
          </p>
        </div>

        {/* Risk Categories */}
        <div className="space-y-8">
          {/* Market Risk */}
          <section>
            <div className="flex items-center space-x-2 mb-4">
              <TrendingDown className="h-6 w-6 text-red-500" />
              <h2 className="text-2xl font-semibold text-gray-900">Market Risk</h2>
            </div>
            <div className="bg-gray-50 p-6 rounded-lg">
              <p className="text-gray-700 mb-4">
                Stock prices can fluctuate significantly due to various market factors including:
              </p>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li>Economic conditions and market sentiment</li>
                <li>Company-specific events and earnings reports</li>
                <li>Political events and regulatory changes</li>
                <li>Interest rate changes and inflation</li>
                <li>Global events and natural disasters</li>
              </ul>
            </div>
          </section>

          {/* Investment Risk */}
          <section>
            <div className="flex items-center space-x-2 mb-4">
              <DollarSign className="h-6 w-6 text-yellow-500" />
              <h2 className="text-2xl font-semibold text-gray-900">Investment Risk</h2>
            </div>
            <div className="bg-gray-50 p-6 rounded-lg">
              <p className="text-gray-700 mb-4">
                Investment risks include but are not limited to:
              </p>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li><strong>Loss of Principal:</strong> You may lose some or all of your initial investment</li>
                <li><strong>Volatility Risk:</strong> Stock prices may experience significant short-term fluctuations</li>
                <li><strong>Liquidity Risk:</strong> You may not be able to sell your investments when desired</li>
                <li><strong>Concentration Risk:</strong> Investing in a limited number of stocks increases risk</li>
                <li><strong>Currency Risk:</strong> Foreign investments may be affected by exchange rate fluctuations</li>
              </ul>
            </div>
          </section>

          {/* AI/ML Predictions Risk */}
          <section>
            <div className="flex items-center space-x-2 mb-4">
              <Clock className="h-6 w-6 text-blue-500" />
              <h2 className="text-2xl font-semibold text-gray-900">AI Prediction Limitations</h2>
            </div>
            <div className="bg-gray-50 p-6 rounded-lg">
              <p className="text-gray-700 mb-4">
                Our AI-powered predictions and analysis tools have limitations:
              </p>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li>Predictions are based on historical data and may not reflect future performance</li>
                <li>AI models cannot predict unforeseen events or market anomalies</li>
                <li>Technical analysis and predictions should not be the sole basis for investment decisions</li>
                <li>Models may be affected by data quality and market regime changes</li>
                <li>Past accuracy of predictions does not guarantee future accuracy</li>
              </ul>
            </div>
          </section>

          {/* Data and Information Risk */}
          <section>
            <div className="flex items-center space-x-2 mb-4">
              <Shield className="h-6 w-6 text-green-500" />
              <h2 className="text-2xl font-semibold text-gray-900">Data and Information Risks</h2>
            </div>
            <div className="bg-gray-50 p-6 rounded-lg">
              <p className="text-gray-700 mb-4">
                Information provided on our platform may have limitations:
              </p>
              <ul className="list-disc list-inside space-y-2 text-gray-700">
                <li>Market data may be delayed and not reflect real-time prices</li>
                <li>Financial information may contain errors or be outdated</li>
                <li>News and analysis are for informational purposes only</li>
                <li>Third-party data sources may have their own limitations and risks</li>
                <li>System outages may affect access to critical information</li>
              </ul>
            </div>
          </section>
        </div>

        {/* Recommendations */}
        <div className="bg-blue-50 border-l-4 border-blue-500 p-6 mt-8">
          <h2 className="text-lg font-semibold text-blue-700 mb-3">Recommendations</h2>
          <ul className="list-disc list-inside space-y-2 text-blue-700">
            <li>Diversify your investment portfolio across different asset classes</li>
            <li>Only invest money you can afford to lose</li>
            <li>Conduct your own research before making investment decisions</li>
            <li>Consider consulting with a qualified financial advisor</li>
            <li>Regularly review and rebalance your portfolio</li>
            <li>Stay informed about market conditions and economic factors</li>
          </ul>
        </div>

        {/* Footer */}
        <div className="mt-8 pt-6 border-t border-gray-200">
          <p className="text-sm text-gray-600">
            <strong>Last Updated:</strong> August 17, 2025
          </p>
          <p className="text-sm text-gray-600 mt-2">
            This risk disclosure is provided for informational purposes only and does not constitute legal or financial advice. 
            Please consult with qualified professionals for personalized guidance.
          </p>
        </div>
      </div>
    </div>
  )
}

export default RiskDisclosure
