import React from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Line } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

const StockPredictionChart = ({ analysis, selectedModel = 'arima', predictionDays = 7 }) => {
  if (!analysis?.prediction?.predictions || analysis.prediction.predictions.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        <p>No prediction data available for charting</p>
      </div>
    );
  }

  const predictions = analysis.prediction.predictions;
  const currentPrice = analysis.prediction.current_price || analysis.stock_info?.current_price || 0;

  // Create labels (dates) for the chart
  const labels = predictions.map(pred => {
    const date = new Date(pred.date);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  });

  // Add "Today" as the first point
  const allLabels = ['Today', ...labels];

  // Create datasets
  const predictionPrices = predictions.map(pred => pred.predicted_price);
  const allPredictionPrices = [currentPrice, ...predictionPrices];

  // Create confidence bounds using 95% confidence intervals as specified
  const upperBounds = predictions.map(pred => {
    // Use the actual confidence intervals if available, otherwise calculate
    if (pred.upper_bound !== undefined) {
      return pred.upper_bound;
    }
    // Fallback calculation for 95% confidence interval
    const confidence = pred.confidence_percentage || 95;
    const margin = pred.predicted_price * (confidence / 100 * 0.05);
    return pred.predicted_price + margin;
  });
  
  const lowerBounds = predictions.map(pred => {
    // Use the actual confidence intervals if available, otherwise calculate
    if (pred.lower_bound !== undefined) {
      return pred.lower_bound;
    }
    // Fallback calculation for 95% confidence interval
    const confidence = pred.confidence_percentage || 95;
    const margin = pred.predicted_price * (confidence / 100 * 0.05);
    return pred.predicted_price - margin;
  });

  const allUpperBounds = [currentPrice, ...upperBounds];
  const allLowerBounds = [currentPrice, ...lowerBounds];

  const data = {
    labels: allLabels,
    datasets: [
      {
        label: 'Predicted Price',
        data: allPredictionPrices,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        borderWidth: 3,
        pointRadius: 5,
        pointHoverRadius: 7,
        tension: 0.1,
        fill: false,
      },
      {
        label: '95% Confidence Upper Bound',
        data: allUpperBounds,
        borderColor: 'rgba(34, 197, 94, 0.6)',
        backgroundColor: 'rgba(34, 197, 94, 0.15)',
        borderWidth: 2,
        borderDash: [8, 4],
        pointRadius: 0,
        tension: 0.1,
        fill: '+1',
      },
      {
        label: '95% Confidence Lower Bound',
        data: allLowerBounds,
        borderColor: 'rgba(239, 68, 68, 0.6)',
        backgroundColor: 'rgba(239, 68, 68, 0.15)',
        borderWidth: 2,
        borderDash: [8, 4],
        pointRadius: 0,
        tension: 0.1,
        fill: false,
      }
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          usePointStyle: true,
          padding: 20,
        }
      },
      title: {
        display: true,
        text: `${analysis.symbol} - ${predictionDays}-Day ${selectedModel.toUpperCase()} Price Prediction with 95% Confidence Intervals`,
        font: {
          size: 16,
          weight: 'bold'
        }
      },
      tooltip: {
        mode: 'index',
        intersect: false,
        callbacks: {
          label: function(context) {
            const label = context.dataset.label || '';
            const value = context.parsed.y;
            const datasetIndex = context.datasetIndex;
            
            if (datasetIndex === 0) { // Main prediction line
              const predictionIndex = context.dataIndex - 1;
              if (predictionIndex >= 0 && predictions[predictionIndex]) {
                const confidence = predictions[predictionIndex].confidence_percentage || 95;
                const intervalType = predictions[predictionIndex].prediction_interval || 95;
                return `${label}: $${value.toFixed(2)} (${intervalType}% CI: ${confidence}% confidence)`;
              }
              return `${label}: $${value.toFixed(2)}`;
            }
            if (datasetIndex === 1) { // Upper bound
              return `95% CI Upper: $${value.toFixed(2)}`;
            }
            if (datasetIndex === 2) { // Lower bound
              return `95% CI Lower: $${value.toFixed(2)}`;
            }
            return `${label}: $${value.toFixed(2)}`;
          }
        }
      },
    },
    interaction: {
      mode: 'nearest',
      axis: 'x',
      intersect: false
    },
    scales: {
      x: {
        display: true,
        title: {
          display: true,
          text: 'Time Period',
          font: {
            weight: 'bold'
          }
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        }
      },
      y: {
        display: true,
        title: {
          display: true,
          text: 'Price ($)',
          font: {
            weight: 'bold'
          }
        },
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.1)'
        },
        ticks: {
          callback: function(value) {
            return '$' + value.toFixed(2);
          }
        }
      },
    },
  };

  // Calculate some statistics for display
  const lastPrediction = predictions[predictions.length - 1];
  const priceChange = lastPrediction.predicted_price - currentPrice;
  const percentChange = (priceChange / currentPrice) * 100;

  return (
    <div className="space-y-4">
      {/* Chart Container */}
      <div className="bg-white p-4 rounded-lg border" style={{ height: '400px' }}>
        <Line data={data} options={options} />
      </div>

      {/* Chart Summary */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div className="text-center">
            <div className="font-semibold text-gray-700">Current Price</div>
            <div className="text-lg font-bold text-blue-600">
              ${currentPrice.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="font-semibold text-gray-700">Predicted Price</div>
            <div className="text-lg font-bold text-blue-600">
              ${lastPrediction.predicted_price.toFixed(2)}
            </div>
          </div>
          <div className="text-center">
            <div className="font-semibold text-gray-700">Expected Change</div>
            <div className={`text-lg font-bold ${
              percentChange >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {percentChange >= 0 ? '+' : ''}${priceChange.toFixed(2)} ({percentChange.toFixed(2)}%)
            </div>
          </div>
          <div className="text-center">
            <div className="font-semibold text-gray-700">Model Used</div>
            <div className="text-lg font-bold text-gray-800">
              {analysis.prediction.model}
            </div>
          </div>
        </div>
      </div>

      {/* Model Performance Metrics */}
      {analysis.prediction.accuracy_metrics && (
        <div className="bg-blue-50 p-4 rounded-lg">
          <h4 className="font-semibold text-blue-800 mb-2">Model Performance</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            {analysis.prediction.accuracy_metrics.mape && (
              <div>
                <span className="text-blue-600 font-medium">MAPE:</span>
                <span className="ml-2">{analysis.prediction.accuracy_metrics.mape.toFixed(2)}%</span>
              </div>
            )}
            {analysis.prediction.accuracy_metrics.accuracy_grade && (
              <div>
                <span className="text-blue-600 font-medium">Grade:</span>
                <span className="ml-2">{analysis.prediction.accuracy_metrics.accuracy_grade}</span>
              </div>
            )}
            {analysis.prediction.accuracy_metrics.directional_accuracy && (
              <div>
                <span className="text-blue-600 font-medium">Direction Accuracy:</span>
                <span className="ml-2">{analysis.prediction.accuracy_metrics.directional_accuracy.toFixed(1)}%</span>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default StockPredictionChart;
