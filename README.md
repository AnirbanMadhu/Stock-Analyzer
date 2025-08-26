# 📈 Stock Analyzer

A comprehensive full-stack web application for real-time stock market analysis, portfolio management, and AI-powered price predictions. Built with React, Flask, and MongoDB.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com)

## ✨ Features

### 🔍 **Stock Analysis**
- **Real-time stock data** from Yahoo Finance API
- **Interactive charts** with technical indicators
- **AI-powered price predictions** using ARIMA, LSTM, and hybrid models
- **Historical data analysis** with customizable time periods
- **Company financial metrics** and performance indicators

### 💼 **Portfolio Management**
- **Track multiple portfolios** with real-time performance
- **Investment tracking** with profit/loss calculations
- **Portfolio analytics** and diversification insights
- **Historical performance** charts and reports

### 📊 **Market Intelligence**
- **Trending stocks** and market movers
- **Gainers/Losers** with percentage changes
- **Market indices** (S&P 500, NASDAQ, DOW)
- **Stock comparison** tools for analysis
- **Watchlist** management with personal notes

### 🤖 **AI Predictions**
- **Multiple ML models** (ARIMA, LSTM, Hybrid, Simple)
- **1-30 day forecasts** with confidence intervals
- **Model performance** metrics and accuracy tracking
- **Intelligent model selection** for optimal predictions

### 🔐 **User Management**
- **Secure authentication** with encrypted passwords
- **Session management** and user profiles
- **Personalized dashboards**
- **Data privacy** and security

## 🛠️ Tech Stack

### **Frontend**
- ![React](https://img.shields.io/badge/React-18-blue?logo=react) - Modern UI framework
- ![Vite](https://img.shields.io/badge/Vite-4-purple?logo=vite) - Fast build tool
- ![TailwindCSS](https://img.shields.io/badge/TailwindCSS-3-cyan?logo=tailwindcss) - Utility-first CSS
- ![Chart.js](https://img.shields.io/badge/Chart.js-4-orange?logo=chartdotjs) - Data visualization

### **Backend**
- ![Flask](https://img.shields.io/badge/Flask-3.1-green?logo=flask) - Python web framework
- ![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-brightgreen?logo=mongodb) - Cloud database
- ![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python) - Backend language
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange?logo=scikit-learn) - Machine learning

### **APIs & Data**
- **Yahoo Finance** - Real-time stock data
- **yfinance** - Python API wrapper
- **pandas & numpy** - Data processing

### **Deployment**
- ![Render](https://img.shields.io/badge/Render-Cloud-purple) - Deployment platform
- ![Gunicorn](https://img.shields.io/badge/Gunicorn-WSGI-green) - Production server

## 🚀 Quick Start

### Prerequisites
- **Node.js** 18+ and npm
- **Python** 3.9+
- **MongoDB Atlas** account (free tier available)
- **Git**

### 1. Clone Repository
```bash
git clone https://github.com/AnirbanMadhu/Stock-Analyzer.git
cd Stock-Analyzer
```

### 2. Environment Setup

Create environment files:

**Backend** (`.env` in root):
```env
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/stock_analyzer
SECRET_KEY=your-super-secret-key-here
FLASK_ENV=development
MONGO_DB=stock_analyzer
```

### 3. Install Dependencies
```bash
# Install all dependencies (frontend + backend)
npm install
```

### 4. Run Development Server
```bash
# Start both frontend and backend
npm run dev
```

The application will be available at:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5000

## 🌐 Deployment

### Deploy to Render (Recommended)

1. **Fork this repository** to your GitHub account
2. **Create MongoDB Atlas cluster** (free tier)
3. **Sign up for Render** and connect your GitHub
4. **Create a new Web Service** with these settings:

**Build Command:**
```bash
npm run build
```

**Start Command:**
```bash
cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --worker-class gthread --max-requests 1000 --preload app:app
```

5. **Set Environment Variables:**
```env
MONGODB_URI=your-mongodb-atlas-connection-string
SECRET_KEY=auto-generated-by-render
FLASK_ENV=production
NODE_ENV=production
MONGO_DB=stock_analyzer
```

6. **Deploy!** Your app will be live in minutes 🚀

## 📱 Usage

### Getting Started
1. **Register** for a new account or **login**
2. **Search stocks** using symbols (AAPL, GOOGL, TSLA) or company names
3. **View real-time data** with interactive charts
4. **Add stocks** to your watchlist for monitoring

### Portfolio Management
1. **Create portfolios** to track your investments
2. **Add holdings** with purchase price and quantity
3. **Monitor performance** with real-time profit/loss
4. **Analyze trends** with historical charts

### AI Predictions
1. **Select a stock** for analysis
2. **Choose prediction model** (Auto recommended)
3. **Set forecast period** (1-30 days)
4. **View predictions** with confidence intervals
5. **Compare models** for accuracy

## 🎯 API Endpoints

### Authentication
```http
POST /api/auth/register    # User registration
POST /api/auth/login       # User login
POST /api/auth/logout      # User logout
GET  /api/auth/profile     # Get user profile
```

### Stock Data
```http
GET  /api/stock/{symbol}           # Get stock data
GET  /api/stock/{symbol}/history   # Historical data
GET  /api/search?q={query}         # Search stocks
GET  /api/trending                 # Trending stocks
```

### Portfolio
```http
GET    /api/portfolio              # Get portfolios
POST   /api/portfolio              # Create portfolio
PUT    /api/portfolio/{id}         # Update portfolio
DELETE /api/portfolio/{id}         # Delete portfolio
```

### Predictions
```http
GET  /api/predict/{symbol}?days=7&model=auto  # Get predictions
GET  /api/predict/models                      # Available models
POST /api/predict/batch                       # Batch predictions
```

## 🏗️ Project Structure

```
Stock-Analyzer/
├── frontend/                    # React frontend application
│   ├── src/
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/              # Page components (Home, Portfolio, etc.)
│   │   ├── services/           # API service functions
│   │   ├── context/            # React context providers
│   │   └── utils/              # Utility functions
│   ├── public/                 # Static assets
│   └── package.json            # Frontend dependencies
├── backend/                    # Flask backend application
│   ├── routes/                 # API route handlers
│   │   ├── auth_routes.py      # Authentication endpoints
│   │   ├── stock_routes.py     # Stock data endpoints
│   │   ├── portfolio_routes.py # Portfolio management
│   │   └── prediction_routes.py # AI prediction endpoints
│   ├── models/                 # Database models
│   ├── services/               # Business logic services
│   ├── ml_models/              # Machine learning models
│   ├── app.py                  # Main Flask application
│   └── requirements.txt        # Python dependencies
├── package.json               # Root package.json for deployment
├── render.yaml               # Render deployment configuration
└── README.md                 # This file
```

## 🤖 Machine Learning Models

| Model | Description | Use Case | Accuracy |
|-------|-------------|----------|----------|
| **ARIMA** | Time series forecasting | Trend analysis | Good for stable stocks |
| **LSTM** | Neural network | Complex patterns | High for volatile stocks |

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `MONGODB_URI` | MongoDB connection string | Yes | - |
| `SECRET_KEY` | Flask secret key | Yes | - |
| `FLASK_ENV` | Flask environment | No | development |
| `NODE_ENV` | Node environment | No | development |
| `MONGO_DB` | Database name | No | stock_analyzer |

### MongoDB Setup
1. Create account at [MongoDB Atlas](https://cloud.mongodb.com/)
2. Create a new cluster (free tier available)
3. Create database user with read/write permissions
4. Get connection string and add to environment variables
5. Whitelist your IP address (or use 0.0.0.0/0 for development)

## 🧪 Testing

```bash
# Run frontend tests
cd frontend && npm test

# Run backend tests
cd backend && python -m pytest

# Run all tests
npm run test
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow existing code style and conventions
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This application is for educational and analytical purposes only. Stock predictions and analysis provided are **NOT financial advice**. Always do your own research and consult with financial professionals before making investment decisions.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data API
- **MongoDB Atlas** for cloud database hosting
- **Render** for seamless deployment platform
- **React & Flask** communities for excellent documentation
- **Open source community** for inspiration and tools

## 📞 Support & Contact

- **GitHub Issues** - [Report bugs and request features](../../issues)
- **Discussions** - [Community discussions](../../discussions)
- **Email** - [anirbanmadhu@example.com](mailto:anirbanmadhu@example.com)

## 🔄 Version History

- **v1.0.0** - Initial release with core features
- **v1.1.0** - Added portfolio management
- **v1.2.0** - Enhanced ML predictions with multiple models
- **v1.3.0** - Performance optimizations and deployment improvements

## 🌟 Show Your Support

Give a ⭐️ if this project helped you!

[![GitHub stars](https://img.shields.io/github/stars/AnirbanMadhu/Stock-Analyzer?style=social)](https://github.com/AnirbanMadhu/Stock-Analyzer/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/AnirbanMadhu/Stock-Analyzer?style=social)](https://github.com/AnirbanMadhu/Stock-Analyzer/network/members)

---

**Made with ❤️ by [AnirbanMadhu](https://github.com/AnirbanMadhu)**

[🔝 Back to top](#-stock-analyzer)
