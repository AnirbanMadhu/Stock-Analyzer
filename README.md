# Stock Analyzer

Comprehensive web platform for exploring stocks, generating short‑term AI price forecasts, tracking portfolios, and maintaining watchlists. Built for educational and analytical use (not investment advice).

---
## Table of Contents
1. Features
2. Architecture & Tech Stack
3. Project Structure
4. Quick Start (Development)
5. Environment Variables
6. Usage Guide
7. Core API Summary
8. Prediction Models
9. Database Schema
10. Development Status & Roadmap
11. Troubleshooting
12. Production Deployment Hints
13. Contributing
14. License & Disclaimer
15. Documentation Links

---
## 1. Features
| Category | Capabilities |
|----------|--------------|
| Search | Real-time symbol/company lookup with autocomplete |
| Market Data | Trending, gainers, losers, indices summary |
| Charts | Historical price (line & candlestick ready) |
| Portfolio | Holdings CRUD, gain/loss calculations |
| Watchlist | Add/remove symbols, optional notes |
| Comparison | Multi-symbol or multi-model comparison (up to 4 symbols) |
| Predictions | 1–30 day forecasts (default 7) with confidence intervals |
| Models | ARIMA / LSTM / Hybrid ensemble / Simple fallback |
| Auth | Registration, login, profile update, session persistence |

---
## 2. Architecture & Tech Stack
Backend: Flask (Python 3.11+ compatible) + SQLAlchemy + yfinance + statsmodels + optional TensorFlow.  
Frontend: React 18 (Vite) + Tailwind CSS + Axios + Chart.js (integration in progress).  
Database: SQLite (development default) with optional migration to PostgreSQL/MySQL for production.  
Auth: Flask-Login (session cookie).  
ML: Enhanced ARIMA, advanced LSTM ensemble, Hybrid weighted combination, Simple moving average fallback.

---
## 3. Project Structure
```
Stock-Analyzer/
├─ backend/
│  ├─ app.py                # Flask app + blueprint registration
│  ├─ models/               # ORM models (User, Portfolio, Watchlist)
│  ├─ routes/               # REST endpoints (auth, stocks, portfolio, predict, etc.)
│  ├─ services/             # Data fetch & business logic
│  └─ ml_models/            # Prediction engine (stock_predictor.py)
├─ frontend/
│  ├─ src/                  # React components/pages/context
│  ├─ public/               # Static assets
│  └─ package.json          # Frontend dependencies
├─ docs/                    # Extended documentation
└─ README.md
```

---
## 4. Quick Start (Development)
Prerequisites: Python 3.10+ (or 3.11/3.12), Node.js 18+, Git.

```powershell
# Clone
git clone <your-repo-url>
cd Stock-Analyzer

# Backend setup
cd backend
python -m venv venv; ./venv/Scripts/Activate.ps1
pip install -r requirements.txt
python init_db.py   # Creates tables; may seed demo user

# Frontend setup
cd ../frontend
npm install

# Start (two terminals) or use start-dev.bat from root
python ../backend/app.py
npm run dev
```

Access:  
Frontend (Vite dev): http://localhost:3002  
Backend API: http://localhost:3001  
Demo credentials available:
- Username: `demo`, Email: `demo@stockanalyzer.com`
- Username: `debjit`, Email: `debjitmitra251@gmail.com`

The application is configured with:
Backend:
- MySQL database connection working
- Authentication routes configured
- API endpoints ready
- Sessions enabled

Frontend:
- React application ready
- Connected to backend API
- CORS properly configured

---
## 5. Environment Variables
Create `.env` in `backend/` (optional for dev):
```
SECRET_KEY=change-me
DATABASE_URL=sqlite:///absolute/path/to/stock_analyzer.db   # Omit to use default
```
Production additions: `DATABASE_URL` (PostgreSQL), logging config, optional model toggles.

---
## 6. Usage Guide (High-Level)
1. Register or log in.
2. Search a symbol → open details or add to watchlist.
3. Navigate to AI Predictor → request 7‑day forecast (or adjust days).
4. Add holdings in Portfolio to track performance.
5. Use Compare to evaluate models or multiple symbols.
6. Review Trending / Gainers / Losers for discovery.

For detailed walkthroughs see `docs/USER_MANUAL.md`.

---
## 7. Core API Summary (Selected)
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/search?q=TSLA` | GET | Symbol/company search |
| `/api/stock/AAPL` | GET | Stock snapshot |
| `/api/stock/AAPL/history?period=1y&interval=1d` | GET | Historical OHLCV |
| `/api/stocks/compare` | POST | Multi-symbol comparison |
| `/api/trending` | GET | Trending symbols |
| `/api/market/summary` | GET | Indices summary |
| `/api/auth/register` | POST | User registration |
| `/api/portfolio/holdings` | GET/POST | Holdings list / add |
| `/api/predict/AAPL?days=7&model=auto` | GET | Price forecast |
| `/api/predict/models` | GET | Available model metadata |
| `/api/predict/batch` | POST | Batch predictions |

Full expanded reference: `docs/TECHNICAL_DOCUMENTATION.md`.

---
## 8. Prediction Models
| Key | Description | Strengths | Notes |
|-----|-------------|-----------|-------|
| arima | Enhanced ARIMA(1,1,1) w/ auto parameter fallback | Fast, stable trends | Uses 1–2y data |
| lstm | Multi-layer LSTM + ensemble | Captures non-linear patterns | Needs TensorFlow installed |
| hybrid | Weighted ARIMA + LSTM | Balanced performance | Requires both libs |
| simple | Moving averages + trend decay | Instant fallback | Always available |
| auto | Intelligent selector | Convenience | Chooses best candidate |

Output includes: predictions array (date, price, bounds, confidence), accuracy metrics (MAE, RMSE, MAPE, directional accuracy, grade), and model metadata.

---
## 9. Database Schema (Dev Default: SQLite)
Users (`users`): id, username, email, password_hash, created_at, is_active  
Portfolios (`portfolios`): id, user_id, ticker, company_name, quantity, purchase_price, purchase_date, created_at, updated_at  
Watchlists (`watchlists`): id, user_id, ticker, company_name, added_at, notes, UNIQUE(user_id,ticker)

Migrate to PostgreSQL/MySQL by setting `DATABASE_URL` and running auto table creation (or integrate Alembic for versioned migrations).

---
## 10. Development Status & Roadmap
Implemented: Search, portfolio CRUD, watchlist, predictions (ARIMA/LSTM/Hybrid/Simple), caching, auth, trending/gainers/losers, indices summary.  
In Progress / Planned:
* Enhanced chart visualizations (candlestick overlays, indicator layers)
* Advanced portfolio analytics (allocation, sector breakdown)
* Real-time streaming (WebSockets)
* Sentiment/news enrichment & model feature fusion
* Additional models (Prophet / XGBoost / Transformer time-series)
* Test suite (unit + integration + performance)

---
## 11. Troubleshooting (Quick Reference)
| Issue | Cause | Fix |
|-------|-------|-----|
| CORS error | Port mismatch | Align frontend port with CORS list in `app.py` |
| Slow first prediction | Model load | Subsequent calls are faster (warm cache) |
| Empty prediction | Upstream data gap | Retry; try liquid symbol |
| TensorFlow unavailable | Not installed / no GPU | Stick to ARIMA/auto/simple |
| SQLite lock | Concurrent writes | Move to PostgreSQL in prod |

---
## 12. Production Deployment (Render)

### 🚀 Render Deployment (Recommended)
Your Stock Analyzer is now fully configured for deployment on Render.com with MongoDB.

**Quick Deploy:**
1. **Set up MongoDB Atlas** (free tier available)
2. **Connect GitHub repository** to Render
3. **Use Blueprint deployment** with included `render.yaml`
4. **Set environment variables** in Render dashboard

**Deployment Files:**
- `render.yaml` - Render deployment configuration
- `RENDER_DEPLOYMENT.md` - Complete step-by-step guide
- `DEPLOYMENT_CHECKLIST.md` - Pre-deployment checklist
- `.env.example` - Required environment variables

**Features:**
- ✅ Single-service deployment (frontend + backend)
- ✅ MongoDB integration
- ✅ Automatic builds on push
- ✅ Free tier compatible
- ✅ HTTPS enabled
- ✅ Production-optimized

### Environment Variables Required
```
MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/stock_analyzer
SECRET_KEY=auto-generated-by-render
FLASK_ENV=production
NODE_ENV=production
```

### Alternative Deployment Options
| Platform | Guide | Notes |
|----------|-------|-------|
| **Render** | `RENDER_DEPLOYMENT.md` | Recommended - includes `render.yaml` |
| Heroku | `docs/DEPLOYMENT.md` | Traditional PaaS option |
| Docker | `docs/DEPLOYMENT.md` | Container deployment |
| VPS | `docs/DEPLOYMENT.md` | Self-hosted option |

---
## 13. Contributing
1. Fork & clone
2. Create feature branch: `feat/<topic>`
3. Add/update docs & (future) tests
4. Open PR with clear description & rationale

Style: Keep endpoints documented; maintain consistent JSON shapes; avoid breaking model output without version note.

---
## 14. License & Disclaimer
Educational project (no formal license file supplied). All forecasts are experimental and NOT financial advice.

---
## 15. Documentation Links
| Doc | Path |
|-----|------|
| **Render Deployment Guide** | `RENDER_DEPLOYMENT.md` |
| **Deployment Checklist** | `DEPLOYMENT_CHECKLIST.md` |
| **Environment Variables** | `.env.example` |
| User Manual | `docs/USER_MANUAL.md` |
| Technical Documentation | `docs/TECHNICAL_DOCUMENTATION.md` |
| Development Guide | `docs/DEVELOPMENT.md` |
| Advanced Deployment | `docs/DEPLOYMENT.md` |

---
Enjoy exploring the markets with Stock Analyzer.
