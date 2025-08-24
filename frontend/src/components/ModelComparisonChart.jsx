import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const ModelComparisonChart = ({ modelComparison, symbol }) => {
  if (!modelComparison?.comparison_summary?.predictions_comparison) {
    return (
      <div className="text-center py-8 text-gray-500">
        <div className="animate-pulse">
          <div className="w-8 h-8 bg-gray-300 rounded-full mx-auto mb-2"></div>
          <p>Loading model comparison data...</p>
          <p className="text-sm">This may take a moment for complex predictions</p>
        </div>
      </div>
    );
  }

  const comparisons = modelComparison.comparison_summary.predictions_comparison;
  
  // Filter out hybrid, auto, sample, and simple models from comparison
  const filteredComparisons = Object.keys(comparisons)
    .filter(modelName => {
      const lowerName = modelName.toLowerCase();
      return !lowerName.includes('hybrid') && !lowerName.includes('auto') && !lowerName.includes('sample') && !lowerName.includes('simple');
    })
    .reduce((obj, key) => {
      obj[key] = comparisons[key];
      return obj;
    }, {});
  
  // Check if we have any models after filtering
  if (Object.keys(filteredComparisons).length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No compatible models available for comparison</p>
        <p className="text-sm">Only ARIMA and LSTM models are supported</p>
      </div>
    );
  }
  
  const modelNames = Object.keys(filteredComparisons).map(name => 
    name.replace('_', ' ').toUpperCase()
  );
  
  const finalPrices = Object.values(filteredComparisons).map(data => data.final_day_price || 0);
  const totalChanges = Object.values(filteredComparisons).map(data => data.total_change_percent || 0);
  const avgConfidences = Object.values(filteredComparisons).map(data => data.average_confidence || 0);

  // Color scheme for different models
  const colors = [
    'rgba(59, 130, 246, 0.8)',   // Blue
    'rgba(34, 197, 94, 0.8)',    // Green
    'rgba(249, 115, 22, 0.8)',   // Orange
    'rgba(168, 85, 247, 0.8)',   // Purple
  ];

  const borderColors = [
    'rgb(59, 130, 246)',
    'rgb(34, 197, 94)',
    'rgb(249, 115, 22)',
    'rgb(168, 85, 247)',
  ];

  // Final Prices Chart Data
  const priceData = {
    labels: modelNames,
    datasets: [
      {
        label: 'Predicted Final Price ($)',
        data: finalPrices,
        backgroundColor: colors,
        borderColor: borderColors,
        borderWidth: 2,
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  };

  // Price Change Chart Data
  const changeData = {
    labels: modelNames,
    datasets: [
      {
        label: 'Expected Change (%)',
        data: totalChanges,
        backgroundColor: totalChanges.map(change => 
          change >= 0 ? 'rgba(34, 197, 94, 0.8)' : 'rgba(239, 68, 68, 0.8)'
        ),
        borderColor: totalChanges.map(change => 
          change >= 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)'
        ),
        borderWidth: 2,
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  };

  // Confidence Chart Data
  const confidenceData = {
    labels: modelNames,
    datasets: [
      {
        label: 'Average Confidence (%)',
        data: avgConfidences,
        backgroundColor: 'rgba(99, 102, 241, 0.8)',
        borderColor: 'rgb(99, 102, 241)',
        borderWidth: 2,
        borderRadius: 4,
        borderSkipped: false,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          padding: 20,
          usePointStyle: true,
        }
      },
      tooltip: {
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            
            if (label.includes('Price')) {
              return `${label}: $${value.toFixed(2)}`;
            } else if (label.includes('Change')) {
              return `${label}: ${value.toFixed(2)}%`;
            } else if (label.includes('Confidence')) {
              return `${label}: ${value.toFixed(1)}%`;
            }
            return `${label}: ${value}`;
          }
        }
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: function(value, index, values) {
            const label = this.chart.data.datasets[0].label;
            if (label.includes('Price')) {
              return '$' + value.toFixed(2);
            } else if (label.includes('Change') || label.includes('Confidence')) {
              return value.toFixed(1) + '%';
            }
            return value;
          }
        }
      },
      x: {
        grid: {
          display: false
        }
      }
    },
  };

  return (
    <div className="space-y-6">
      <div className="text-center mb-4">
        <h3 className="text-xl font-bold text-gray-900">
          {symbol} - Model Comparison Analysis
        </h3>
        <p className="text-gray-600 text-sm">
          Compare predictions from different AI models
        </p>
      </div>

      {/* Charts Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Final Prices Chart */}
        <div className="bg-white p-4 rounded-lg border">
          <h4 className="text-lg font-semibold mb-4 text-center">Predicted Final Prices</h4>
          <div style={{ height: '300px' }}>
            <Bar data={priceData} options={{
              ...chartOptions,
              plugins: {
                ...chartOptions.plugins,
                title: {
                  display: false
                }
              }
            }} />
          </div>
        </div>

        {/* Price Change Chart */}
        <div className="bg-white p-4 rounded-lg border">
          <h4 className="text-lg font-semibold mb-4 text-center">Expected Change</h4>
          <div style={{ height: '300px' }}>
            <Bar data={changeData} options={{
              ...chartOptions,
              plugins: {
                ...chartOptions.plugins,
                title: {
                  display: false
                }
              }
            }} />
          </div>
        </div>

        {/* Confidence Chart */}
        <div className="bg-white p-4 rounded-lg border">
          <h4 className="text-lg font-semibold mb-4 text-center">Model Confidence</h4>
          <div style={{ height: '300px' }}>
            <Bar data={confidenceData} options={{
              ...chartOptions,
              plugins: {
                ...chartOptions.plugins,
                title: {
                  display: false
                }
              }
            }} />
          </div>
        </div>
      </div>

      {/* Summary Table */}
      <div className="bg-white rounded-lg border overflow-hidden">
        <div className="bg-gray-50 px-6 py-3 border-b">
          <h4 className="text-lg font-semibold">Detailed Comparison</h4>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-100">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Model
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Final Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Change
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Confidence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Recommendation
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {Object.entries(filteredComparisons).map(([modelName, data], index) => (
                <tr key={modelName} className="hover:bg-gray-50">
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center">
                      <div 
                        className="w-3 h-3 rounded-full mr-3"
                        style={{ backgroundColor: colors[index] }}
                      ></div>
                      <span className="text-sm font-medium text-gray-900">
                        {modelName.replace('_', ' ').toUpperCase()}
                      </span>
                    </div>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    ${data.final_day_price?.toFixed(2) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <span className={`font-medium ${
                      (data.total_change_percent || 0) >= 0 
                        ? 'text-green-600' 
                        : 'text-red-600'
                    }`}>
                      {(data.total_change_percent || 0) >= 0 ? '+' : ''}
                      {(data.total_change_percent || 0).toFixed(2)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    {(data.average_confidence || 0).toFixed(1)}%
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm">
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      (data.total_change_percent || 0) >= 5
                        ? 'bg-green-100 text-green-800'
                        : (data.total_change_percent || 0) <= -5
                        ? 'bg-red-100 text-red-800'
                        : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {(data.total_change_percent || 0) >= 5
                        ? 'Strong Buy'
                        : (data.total_change_percent || 0) <= -5
                        ? 'Sell'
                        : 'Hold'
                      }
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Recommendation Summary */}
      {modelComparison.recommendation && (
        <div className="bg-blue-50 p-6 rounded-lg border-l-4 border-blue-400">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <svg className="h-5 w-5 text-blue-400" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="ml-3">
              <h4 className="text-lg font-semibold text-blue-800">
                AI Recommendation: {modelComparison.recommendation.recommendation}
              </h4>
              <p className="text-blue-700 mt-2">
                {modelComparison.recommendation.reason}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ModelComparisonChart;
