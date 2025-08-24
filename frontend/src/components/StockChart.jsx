import React, { useMemo, useState } from 'react'
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js'
import { Line, Bar } from 'react-chartjs-2'
import { format, parseISO } from 'date-fns'

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  Filler
)

const StockChart = ({ 
  data, 
  symbol, 
  chartType = 'line', 
  timeframe = '1y',
  showVolume = false,
  height = 400 
}) => {
  const [activeChart, setActiveChart] = useState(chartType)

  const chartData = useMemo(() => {
    if (!data || !data.data || data.data.length === 0) {
      return null
    }

    const sortedData = [...data.data].sort((a, b) => new Date(a.date) - new Date(b.date))
    
    const labels = sortedData.map(item => {
      try {
        const date = new Date(item.date)
        if (timeframe === '1d') {
          return format(date, 'HH:mm')
        } else if (timeframe === '5d' || timeframe === '1mo') {
          return format(date, 'MMM dd')
        } else {
          return format(date, 'MMM yyyy')
        }
      } catch {
        return item.date
      }
    })

    const prices = sortedData.map(item => parseFloat(item.close))
    const volumes = sortedData.map(item => parseInt(item.volume))
    const highs = sortedData.map(item => parseFloat(item.high))
    const lows = sortedData.map(item => parseFloat(item.low))
    const opens = sortedData.map(item => parseFloat(item.open))

    // Determine color based on overall trend
    const firstPrice = prices[0]
    const lastPrice = prices[prices.length - 1]
    const isPositive = lastPrice >= firstPrice

    const baseDataset = {
      label: `${symbol} Price`,
      data: prices,
      borderColor: isPositive ? '#10B981' : '#EF4444',
      backgroundColor: isPositive 
        ? 'rgba(16, 185, 129, 0.1)' 
        : 'rgba(239, 68, 68, 0.1)',
      borderWidth: 2,
      fill: activeChart === 'area',
      tension: 0.1,
      pointRadius: 0,
      pointHoverRadius: 5,
      pointHoverBackgroundColor: isPositive ? '#10B981' : '#EF4444',
      pointHoverBorderColor: '#ffffff',
      pointHoverBorderWidth: 2,
    }

    if (activeChart === 'candlestick') {
      // For candlestick, we'll use a combination of datasets
      return {
        labels,
        datasets: [
          {
            label: 'High',
            data: highs,
            borderColor: '#9CA3AF',
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0,
            type: 'line',
          },
          {
            label: 'Low',
            data: lows,
            borderColor: '#9CA3AF',
            backgroundColor: 'transparent',
            borderWidth: 1,
            pointRadius: 0,
            type: 'line',
          },
          {
            label: 'Open',
            data: opens,
            borderColor: '#3B82F6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            borderWidth: 2,
            pointRadius: 1,
            type: 'line',
          },
          {
            label: 'Close',
            data: prices,
            borderColor: isPositive ? '#10B981' : '#EF4444',
            backgroundColor: isPositive 
              ? 'rgba(16, 185, 129, 0.2)' 
              : 'rgba(239, 68, 68, 0.2)',
            borderWidth: 3,
            pointRadius: 0,
            pointHoverRadius: 5,
            type: 'line',
          }
        ]
      }
    }

    const datasets = [baseDataset]

    if (showVolume) {
      datasets.push({
        label: 'Volume',
        data: volumes,
        backgroundColor: 'rgba(99, 102, 241, 0.3)',
        borderColor: '#6366F1',
        borderWidth: 1,
        yAxisID: 'volume',
        type: 'bar',
        barThickness: 2,
      })
    }

    return { labels, datasets }
  }, [data, symbol, activeChart, timeframe, showVolume])

  const options = useMemo(() => {
    const scales = {
      x: {
        display: true,
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          maxTicksLimit: 8,
          color: '#6B7280',
        },
      },
      y: {
        display: true,
        position: 'right',
        grid: {
          display: true,
          color: 'rgba(0, 0, 0, 0.05)',
        },
        ticks: {
          color: '#6B7280',
          callback: function(value) {
            return '$' + parseFloat(value).toFixed(2)
          },
        },
      },
    }

    if (showVolume) {
      scales.volume = {
        type: 'linear',
        display: true,
        position: 'left',
        grid: {
          drawOnChartArea: false,
        },
        ticks: {
          color: '#6B7280',
          callback: function(value) {
            if (value >= 1000000) {
              return (value / 1000000).toFixed(1) + 'M'
            } else if (value >= 1000) {
              return (value / 1000).toFixed(1) + 'K'
            }
            return value
          },
        },
      }
    }

    return {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false,
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: '#374151',
            usePointStyle: true,
            filter: function(legendItem) {
              // Don't show volume in legend if it's not the main focus
              return showVolume || legendItem.text !== 'Volume'
            }
          },
        },
        title: {
          display: true,
          text: `${symbol} Stock Price - ${timeframe.toUpperCase()}`,
          color: '#111827',
          font: {
            size: 16,
            weight: '600',
          },
        },
        tooltip: {
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          titleColor: '#111827',
          bodyColor: '#374151',
          borderColor: '#E5E7EB',
          borderWidth: 1,
          cornerRadius: 8,
          displayColors: true,
          callbacks: {
            label: function(context) {
              const label = context.dataset.label || ''
              const value = context.parsed.y
              
              if (label === 'Volume') {
                if (value >= 1000000) {
                  return `${label}: ${(value / 1000000).toFixed(2)}M`
                } else if (value >= 1000) {
                  return `${label}: ${(value / 1000).toFixed(1)}K`
                }
                return `${label}: ${value.toLocaleString()}`
              }
              
              return `${label}: $${parseFloat(value).toFixed(2)}`
            },
          },
        },
      },
      scales,
      elements: {
        point: {
          hoverRadius: 8,
        },
      },
    }
  }, [symbol, timeframe, showVolume])

  if (!chartData) {
    return (
      <div className="flex items-center justify-center h-96 bg-gray-50 rounded-lg">
        <div className="text-center">
          <div className="text-gray-400 text-lg mb-2">No chart data available</div>
          <div className="text-gray-500 text-sm">
            Unable to load price data for {symbol}
          </div>
        </div>
      </div>
    )
  }

  const chartTypes = [
    { key: 'line', label: 'Line', icon: '📈' },
    { key: 'area', label: 'Area', icon: '📊' },
    { key: 'candlestick', label: 'Candlestick', icon: '🕯️' },
  ]

  return (
    <div className="w-full">
      {/* Chart Type Selector */}
      <div className="flex flex-wrap items-center justify-between mb-4 gap-4">
        <div className="flex items-center space-x-2">
          {chartTypes.map(type => (
            <button
              key={type.key}
              onClick={() => setActiveChart(type.key)}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                activeChart === type.key
                  ? 'bg-blue-100 text-blue-700 border border-blue-200'
                  : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
              }`}
            >
              <span className="mr-1">{type.icon}</span>
              {type.label}
            </button>
          ))}
        </div>

        <label className="flex items-center space-x-2 text-sm">
          <input
            type="checkbox"
            checked={showVolume}
            onChange={(e) => {
              // This would need to be controlled by parent component
              console.log('Toggle volume:', e.target.checked)
            }}
            className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
          />
          <span className="text-gray-700">Show Volume</span>
        </label>
      </div>

      {/* Chart Container */}
      <div 
        className="w-full bg-white rounded-lg border border-gray-200 p-4"
        style={{ height: `${height}px` }}
      >
        {activeChart === 'line' || activeChart === 'area' || activeChart === 'candlestick' ? (
          <Line data={chartData} options={options} />
        ) : (
          <Bar data={chartData} options={options} />
        )}
      </div>

      {/* Chart Info */}
      <div className="mt-2 text-xs text-gray-500 text-center">
        Data provided by Yahoo Finance • Updated every 15 minutes during market hours
      </div>
    </div>
  )
}

export default StockChart
