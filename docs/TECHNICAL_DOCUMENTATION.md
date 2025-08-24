# Technical Documentation – Stock Analyzer

Version: 1.0  
Last Updated: 2025-08-17

---
## 1. System Overview
Stock Analyzer is a full‑stack analytical platform providing stock lookup, portfolio tracking, watchlists, and short‑term AI price forecasting. It emphasizes modular architecture, extensibility, and model experimentation. The application is not a trading platform; it’s an educational analytical tool.

---
## 2. Architecture
| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | React (Vite) + Tailwind CSS | UI, routing, client‑side interactions |
| Backend | Flask | REST API, auth, prediction orchestration |
| Models | Python (statsmodels, TensorFlow optional) | ARIMA / LSTM / Hybrid forecasting |
| Data Fetch | yfinance + HTML scraping | Market data, trending lists, gainers/losers |
| Database | SQLite (dev) | Persistent user, portfolio, watchlist data |
| Auth | Flask-Login + werkzeug security | Session-based authentication |

### High-Level Flow
1. React sends API request → `/api/...`
2. Flask blueprint handles routing, invokes service layer or ML module.
3. Services fetch data (cache + yfinance + scraping) and/or query DB.
4. ML layer (ARIMA/LSTM/Hybrid/Simple) generates prediction (with caching).
5. Response serialized to JSON → consumed by frontend components.

### Key Backend Modules
| Path | Responsibility |
|------|---------------|
| `backend/app.py` | App factory configuration, blueprint registration, health checks |
| `backend/routes/*` | REST endpoints grouped by domain |
| `backend/models/*` | SQLAlchemy ORM models |
| `backend/services/*` | Market data retrieval, caching, business logic |
| `backend/ml_models/stock_predictor.py` | Unified prediction interface & model implementations |
| `backend/database.py` | DB configuration & initialization |

---
## 3. Environment & Configuration
| Variable | Description | Default |
|----------|-------------|---------|
| `SECRET_KEY` | Flask session secret | dev-secret-key-change-in-production |
| `DATABASE_URL` | Override SQLite DB | (unset → local SQLite) |

Sensitive values should be moved to environment variables for production deployment (see `DEPLOYMENT.md`).

---
## 4. Database Schema
Relational schema (SQLite, easily portable to PostgreSQL/MySQL).  
All timestamps stored in UTC.

### users
| Column | Type | Constraints |
|--------|------|-------------|
| id | Integer | PK |
| username | String(80) | UNIQUE, NOT NULL |
| email | String(120) | UNIQUE, NOT NULL |
| password_hash | String(255) | NOT NULL |
| created_at | DateTime | default=utcnow |
| is_active | Boolean | default=True |

### portfolios
| Column | Type | Constraints |
|--------|------|-------------|
| id | Integer | PK |
| user_id | Integer | FK → users.id (cascade delete) |
| ticker | String(10) | NOT NULL, upper-cased |
| company_name | String(255) | nullable |
| quantity | Float | NOT NULL |
| purchase_price | Numeric(10,2) | NOT NULL |
| purchase_date | DateTime | default=utcnow |
| created_at | DateTime | default=utcnow |
| updated_at | DateTime | autoupdate |

### watchlists
| Column | Type | Constraints |
|--------|------|-------------|
| id | Integer | PK |
| user_id | Integer | FK → users.id (cascade delete) |
| ticker | String(10) | NOT NULL |
| company_name | String(255) | nullable |
| added_at | DateTime | default=utcnow |
| notes | Text | nullable |
| (unique_user_ticker) | Composite | UNIQUE(user_id, ticker) |

### Relationships
* `User.portfolios` (1→many) and `User.watchlists` (1→many).  
* Cascade delete ensures cleanup when a user is removed.

---
## 5. Authentication & Authorization
Session-based authentication via Flask-Login:  
1. User registers: password hashed with Werkzeug’s `generate_password_hash` (PBKDF2).  
2. Login sets a secure session cookie.  
3. Protected endpoints are decorated with `@login_required` (returns 401 JSON instead of redirect).  
4. `current_user` proxy used to scope DB queries.  

Stateless (token/JWT) approach can be added easily by introducing an auth token blueprint or switching to Bearer tokens for SPA scaling.

---
## 6. Caching Strategy
| Layer | Mechanism | TTL |
|-------|-----------|-----|
| Prediction results | In-memory dict in `StockPredictor.cache` | 1h (standard), 7d for model objects |
| Search / market data | Likely via `services/search_cache.py` (in-memory) | Short-lived (implementation-specific) |

Production Recommendation: Replace with Redis or persistent caching for multi-instance deployments.

---
## 7. API Reference
Base URL (dev): `http://localhost:5000/api` (except `/health`).  
Auth endpoints: `http://localhost:5000/api/auth`  
Prediction endpoints: `http://localhost:5000/api/predict`  
Hybrid/AI endpoints: `http://localhost:5000/api/ai` (enhanced)  

### 7.1 Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Service status (root app) |
| GET | `/api/health` | API namespace health |

### 7.2 Authentication (`/api/auth`)
| Method | Endpoint | Payload | Notes |
|--------|----------|---------|-------|
| POST | `/register` | `{username,email,password}` | Auto-login on success |
| POST | `/login` | `{username,password}` (username or email) | Returns user object |
| POST | `/logout` | — | Auth required |
| GET | `/profile` | — | Returns current user |
| PUT | `/profile` | `{email?,current_password?,new_password?}` | Password change requires current_password |
| GET | `/check` | — | Auth state probe |

### 7.3 Stock Data (`/api` stock blueprint)
| Method | Endpoint | Query / Body | Description |
|--------|----------|-------------|-------------|
| GET | `/search` | `q, limit?` | Search ticker/company |
| GET | `/stock/<symbol>` | — | Stock fundamentals/price snapshot |
| GET | `/stock/<symbol>/history` | `period, interval` | Historical OHLCV series |
| POST | `/stocks/compare` | `{symbols[],period?,interval?}` | Multi-symbol compare (≤4) |
| GET | `/trending` | `count?` | Trending symbols from external source |
| GET | `/market/summary` | — | Major indices summary |
| GET | `/market/gainers` | `count?` | Top gainers |
| GET | `/market/losers` | `count?` | Top losers |
| GET | `/market/movers` | `count?` | Combined gainers/losers |

### 7.4 Portfolio (`/api/portfolio`)
Auth required.
| Method | Endpoint | Payload | Description |
|--------|----------|---------|-------------|
| GET | `/` or `/holdings` | — | List holdings + summary |
| POST | `/` or `/holdings` | `{ticker,quantity,purchase_price,purchase_date?}` | Add holding |
| PUT | `/holdings/<id>` | `{quantity?,purchase_price?,purchase_date?}` | Update holding |
| DELETE | `/holdings/<id>` | — | Remove holding |

### 7.5 Watchlist (`/api/watchlist` OR `/api/portfolio/watchlist`)
| Method | Endpoint | Payload | Description |
|--------|----------|---------|-------------|
| GET | `/api/watchlist/` | — | User watchlist (root alias) |
| POST | `/api/watchlist/` | `{ticker|symbol,notes?}` | Add item |
| GET | `/api/portfolio/watchlist` | — | Alternate route (historic) |
| POST | `/api/portfolio/watchlist` | `{ticker|symbol,notes?}` | Alternate add |
| DELETE | `/api/portfolio/watchlist/<item_id>` | — | Remove item |
| DELETE | `/api/watchlist/<item_id>` | — | Remove item (root alias) |

### 7.6 Predictions (`/api/predict`)
| Method | Endpoint | Query / Body | Description |
|--------|----------|--------------|-------------|
| GET | `/<symbol>` | `days?,model?,confidence?` | Forecast using specified / default model (ARIMA default) |
| POST | `/stock` | `{symbol,days?,model?}` | POST variant |
| POST | `/batch` | `{symbols[],days?,model?}` | Batch predict (≤10 symbols) |
| GET | `/models` | — | Available model capabilities |
| GET | `/accuracy/<symbol>` | `model?` | Recent accuracy metrics |

### 7.7 Enhanced AI (`/api/ai`)
If additional advanced endpoints exist (not listed in snippet), they would mirror prediction endpoints but may return enriched hybrid analytics.

### Status Codes & Error Shape
| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Resource created |
| 400 | Validation error / bad request |
| 401 | Authentication required / invalid credentials |
| 404 | Not found |
| 500 | Internal server error |

Error JSON shape: `{ "error": "Message" }`

---
## 8. Machine Learning Subsystem
File: `backend/ml_models/stock_predictor.py`

### Supported Models
| Key | Description | Library | Availability Flag |
|-----|-------------|---------|-------------------|
| arima | ARIMA(1,1,1) with auto fallback search | statsmodels | `ARIMA_AVAILABLE` |
| lstm | Multi-layer LSTM + ensemble uncertainty | TensorFlow | `LSTM_AVAILABLE` |
| hybrid | Weighted ARIMA + LSTM ensemble | Both | Requires both libs |
| simple | Moving average + trend fallback | Built-in | Always |

### Core Features
* Automatic parameter selection preferring (1,1,1)
* Dynamic model selection (data length, volatility, trend)
* Ensemble weighting based on MAPE, R², directional accuracy
* Confidence % decays over horizon; intervals widen with volatility
* Caching with different TTL for predictions vs. model training artifacts
* Backtesting metrics: MAE, RMSE, MAPE, directional accuracy
* Grade mapping (A–F) for quick UI comprehension

### Data Pipeline
1. Fetch ~1–2 years historical OHLCV (auto-adjusted) via yfinance.
2. Derive engineered features: SMA(5/10/20), EMA(12/26), MACD, RSI, Bollinger Bands, volatility, price change.
3. Clean & drop incomplete rows.
4. Model-specific preparation:
	 * ARIMA: price series (with differencing handled by order selection)
	 * LSTM: scaled multi-feature sequences (adaptive sequence length)
	 * Hybrid: fuse ARIMA & LSTM predictions using performance-weighted averaging
5. Return JSON with predictions & metadata.

### Accuracy Calculation
* Backtesting: final 20% of series used for validation (ARIMA) / holdout test set (LSTM).
* Metrics computed: MAE, RMSE, MAPE, Directional Accuracy, R².
* Grade thresholds (simplified): MAPE≤5 & dir≥70 → A; ≤10 & dir≥60 → B; etc.

### Extending Models
Add a new function (e.g., `predict_with_prophet`) then register in `get_prediction` selection logic & availability map. Maintain similar output shape:  
```
{
	symbol, model, current_price, prediction_days,
	predictions: [ { date, predicted_price, lower_bound, upper_bound, confidence_percentage, ... } ],
	accuracy_metrics: { ... },
	model_performance: { ... },
	generated_at
}
```

---
## 9. Services Layer
`services/stock_service.py` (not shown fully here) likely provides:
* Symbol search (possibly cached)
* Ticker fundamental snapshot
* Historical price retrieval (period/interval mapping)
* Trending / gainers / losers scraping via `market_data_service`

`search_cache.py` caches search results to reduce upstream calls.

---
## 10. Error Handling & Logging
* Global 404 & 500 handlers in `app.py` return JSON (no HTML traces).
* Logging via Python `logging` module (INFO level configured).
* Prediction errors degrade gracefully to fallback models.
* DB operations rollback on exceptions inside transactional contexts.

Production Suggestion: integrate structured logging (JSON) and a monitoring stack (e.g., Prometheus + Loki or ELK).

---
## 11. Security Considerations
Implemented:
* Password hashing (PBKDF2 via Werkzeug)
* Session-based auth with `SECRET_KEY`
* CORS restricted to explicit localhost origins

Recommended Enhancements:
* HTTPS termination (reverse proxy: Nginx/Caddy)
* CSRF protection for form endpoints (if migrating to server-rendered) or same-site cookie flags
* Rate limiting (Flask-Limiter or reverse proxy) on login & prediction endpoints
* Input sanitization on free-text fields (notes)
* Security headers (via Flask-Talisman or manual config)

---
## 12. Performance Notes
| Aspect | Current | Optimization Paths |
|--------|---------|-------------------|
| Prediction first-call latency | Moderate (model load + data fetch) | Warm-up job / persistent model objects |
| Batch predictions | ThreadPoolExecutor (≤5 workers) | Async event loop or task queue |
| Data fetch | yfinance per symbol | Bulk retrieval or caching layer |
| DB | SQLite single file | Move to PostgreSQL + connection pooling |

---
## 13. Deployment Guidance (Summary)
See `DEPLOYMENT.md` for full detail. Key adaptations for production:
1. Switch DB to PostgreSQL: set `DATABASE_URL`.
2. Run behind WSGI server (gunicorn / uvicorn with ASGI adapter if migrated).
3. Externalize secrets & model artifacts.
4. Add Redis for caching & model store.
5. Enable build pipeline for frontend (static assets served via CDN or reverse proxy).

---
## 14. Testing Strategy (Suggested)
Current repository shows limited explicit tests. Recommended additions:
* Unit tests: model selection logic, fallback chain, accuracy grading.
* Integration tests: auth flow, portfolio CRUD, prediction endpoint with mocked yfinance.
* Load tests: prediction concurrency (locust / k6) for scaling insights.

---
## 15. Observability Roadmap
| Layer | Tooling |
|-------|---------|
| Metrics | Prometheus exporters (custom for prediction counts) |
| Logging | Structured JSON + ELK/ Loki |
| Tracing | OpenTelemetry instrumentation (Flask + model inference spans) |

---
## 16. Data Integrity & Migration
Migration Path: adopt Alembic for schema evolution once moving beyond prototype.  
Strategy: generate baseline migration from existing models → add new columns/models with revision scripts (ensuring backward compatibility with historical data where possible).

---
## 17. Known Limitations
| Area | Limitation | Mitigation |
|------|-----------|------------|
| Data Source | yfinance rate limits / intermittent failures | Retry & cache layer |
| Model Accuracy | Short-term predictions only (≤30 days) | Display disclaimers & accuracy grades |
| Persistence | SQLite local file locking under concurrency | Migrate to PostgreSQL |
| Scaling | Single-process Flask dev server | Use gunicorn + workers / container orchestration |
| Security | No 2FA / password reset flow | Add token-based reset + optional MFA |

---
## 18. Extension Opportunities
* Add sentiment analysis (news & social signals) into feature set
* Introduce Prophet / XGBoost time-series models for benchmarking
* Portfolio analytics: Sharpe ratio, sector diversification, risk metrics
* Option chain & implied volatility overlays (if data source added)
* Multi-tenancy & organization-level portfolios
* WebSockets for streaming price updates

---
## 19. Example Prediction Response (ARIMA)
```
{
	"symbol": "AAPL",
	"model": "Enhanced ARIMA(1, 1, 1)",
	"current_price": 213.42,
	"prediction_days": 7,
	"predictions": [
		{"date": "2025-08-18", "predicted_price": 214.10, "lower_bound": 210.9, "upper_bound": 217.2, "confidence_percentage": 91.2, "day": 1},
		...
	],
	"accuracy_metrics": {"mae": 1.23, "rmse": 1.85, "mape": 0.65, "directional_accuracy": 71.4, "accuracy_grade": "A (Excellent)"},
	"model_performance": {"order": [1,1,1], "aic": 1234.56, "bic": 1245.67, "r_squared": 0.82},
	"generated_at": "2025-08-17T10:22:11.512930"
}
```

---
## 20. Output Contract Summary
All prediction endpoints should return (on success):
```
symbol: string
model: string
current_price: number
prediction_days: int
predictions: [ { date, predicted_price, lower_bound?, upper_bound?, confidence_percentage?, day, ... } ]
accuracy_metrics: object (may include mae, rmse, mape, directional_accuracy, r_squared, accuracy_grade)
model_performance: object (model-specific attributes)
generated_at: ISO8601 timestamp
fallback_used?: string (if primary failed)
```

---
## 21. Maintenance Checklist
| Frequency | Task |
|-----------|------|
| Weekly | Refresh dependencies, check security advisories |
| Weekly | Review model accuracy drift vs. backtest logs |
| Monthly | Evaluate adding new engineered features |
| Quarterly | Run load tests & capacity planning |

---
## 22. Change Log
| Date | Change |
|------|--------|
| 2025-08-17 | Comprehensive rewrite; added full API & model details |

---
## 23. Contact
For architecture decisions, extension proposals, or incident escalation contact the project maintainers (see repository README). Contributions welcome via pull requests with clear test coverage.

---
End of Technical Documentation.
