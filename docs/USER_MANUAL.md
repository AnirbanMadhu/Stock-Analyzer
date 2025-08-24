# Stock Analyzer – User Manual

Welcome to Stock Analyzer. This guide walks you through everything you need to effectively explore stocks, generate AI predictions, manage a portfolio, and track a watchlist. (Educational use only – not financial advice.)

---
## 1. Quick Start
1. Open the site (development): http://localhost:3000
2. Click Register to create an account (or log in with any existing user).
3. Use the top navigation bar to move between: Home, AI Predictor, Portfolio, Watchlist, Compare, News (if enabled), Login/Logout.
4. Start by searching a ticker (e.g., AAPL) in the search bar.
5. Open the AI Predictor to view short‑term forecasts (default 7 days) with confidence ranges.
6. Add holdings in Portfolio to see gain/loss metrics update in real time.
7. Add interesting symbols to your Watchlist for quick monitoring.

---
## 2. Navigation Overview
| Section | Purpose |
|---------|---------|
| Home | High‑level entry point and quick links |
| AI Predictor | Generate model forecasts (ARIMA / LSTM / Hybrid / Simple) |
| Portfolio | Add, edit, remove holdings and view aggregated performance |
| Watchlist | Track symbols without entering cost basis |
| Compare | Side‑by‑side model output or stock comparisons (depending on UI) |
| News (optional) | Market or symbol-specific news items |
| Login / Register | Account creation and authentication |

---
## 3. Account & Authentication
### Creating an Account
Provide username, email, password (≥6 chars). Upon successful registration you are auto‑logged in.

### Logging In
Use either your username or email plus password. Optionally enable “remember me” (if exposed in UI) to persist your session.

### Profile Management
If a profile page is present you can update email and (with current password) set a new password.

### Logging Out
Use the Logout option in the nav bar (session cookie cleared). Some pages require authentication—if unauthorized you’ll receive a 401 and may be redirected or shown a prompt.

---
## 4. Stock Search & Exploration
1. Enter at least one character of a ticker or company name.
2. Select from autocomplete results (if implemented) or press Enter.
3. View returned details: current price, change %, volume, and basic metadata.
4. From a stock detail page (or card) you can: Add to Watchlist, Open Predictions, Add to Portfolio.

### Tips
* Use widely traded symbols first if testing (AAPL, MSFT, NVDA, TSLA, GOOGL).
* If data appears blank, retry—the upstream data provider (Yahoo Finance) can rate‑limit occasionally.

---
## 5. AI Predictions
Open the AI Predictor page, then:
1. Enter a valid ticker.
2. (Optional) Select forecast horizon (1–30 days) – default 7.
3. (Optional) Choose model: arima, lstm, hybrid, simple, or auto (auto selects best available).
4. Submit to receive:
	* Daily predicted price
	* Lower & upper bounds (approx. 95% interval)
	* Confidence % (dynamic based on model quality & horizon)
	* Model metadata & accuracy grade

### Choosing a Model
| Model | Strengths | When to Use |
|-------|-----------|------------|
| ARIMA | Fast, good for linear trends | Stable or trending stocks |
| LSTM | Captures non-linear patterns | High volatility / complex signals |
| Hybrid | Weighted ARIMA + LSTM | When both are available (best balance) |
| Simple | Fallback moving averages | Library unavailable / quick estimate |
| Auto | Intelligent selection | Most users—lets system decide |

### Interpretation Notes
* Predictions are experimental and not investment advice.
* Wider intervals imply higher uncertainty; short horizons are generally more reliable.
* Accuracy grade (A–F) is derived from backtesting metrics (MAPE & directional accuracy).

---
## 6. Portfolio Management
1. Navigate to Portfolio.
2. Click Add (or “+”) to create a holding: ticker, quantity, purchase price, (optional) date.
3. Holdings list displays: symbol, quantity, cost basis, current price, gain/loss $, gain/loss %.
4. Edit or Delete actions let you maintain accuracy over time.
5. Summary panel aggregates total market value and overall unrealized gain/loss.

### Best Practices
* Enter splits or consolidations as separate manual adjustments (edit quantity & price if necessary).
* Use consistent currency (USD assumed throughout).

---
## 7. Watchlist
1. From search or stock detail, choose Add to Watchlist (or use Watchlist page Add form).
2. Watchlist items show current price & intraday change.
3. Remove items via the delete icon/button.
4. Add notes (if UI includes a notes field) to annotate a thesis or reminder.

Use Watchlist for discovery before committing capital in Portfolio.

---
## 8. Model Comparison (If Enabled)
On Compare page:
1. Input a symbol and select horizon.
2. The system runs ARIMA, LSTM, Hybrid, and Simple (as available).
3. A comparison table / chart shows each model’s forecast path, average confidence, predicted change %, and accuracy grade.
4. A recommendation card suggests the most suitable model with reasoning.

---
## 9. News & Market Overview (Optional Sections)
* Trending / Gainers / Losers pages list popular or extreme movers.
* Market Summary shows major indices (S&P 500, Dow Jones, NASDAQ, Russell 2000).
* Use these to identify symbols to add to Watchlist or analyze with the predictor.

---
## 10. Error Handling & Messages
| Situation | Typical Message | Action |
|-----------|-----------------|--------|
| Invalid symbol | Stock not found | Verify ticker |
| Prediction failure | Could not generate prediction | Retry / choose different model |
| Not logged in | 401 / auth required | Log in and retry |
| Rate/Network issue | Internal server error | Wait & refresh |

---
## 11. FAQs
**Q: Are predictions real-time?**  
Historical data is fetched close to real time but may have slight delays. Forecasts are generated on demand and cached briefly.

**Q: Why do bounds widen over time?**  
Uncertainty grows with forecast horizon; models scale intervals accordingly.

**Q: Why does Hybrid sometimes fall back to a single model?**  
If one model fails or is unavailable, the hybrid returns the successful component with a fallback label.

**Q: Can I reset my password?**  
Currently password change is via Profile (requires current password). A full reset workflow may be a future enhancement.

**Q: Is my data public?**  
No. Portfolio and watchlist data are scoped to your authenticated session on the server.

---
## 12. Troubleshooting Quick Reference
| Problem | Resolution |
|---------|------------|
| Predictions empty | Try a widely traded symbol; check console/network tab |
| Portfolio values zero | Data provider temporarily unavailable; refresh later |
| Login not persisting | Ensure cookies enabled; clear site data and retry |
| Slow first prediction | Initial model load (ARIMA or TensorFlow) – subsequent calls faster |

---
## 13. Glossary
* **Ticker** – Short symbol representing a traded asset (e.g., AAPL).
* **Confidence Interval** – Range within which the model expects the true price to fall (approx. probability stated).
* **MAPE** – Mean Absolute Percentage Error; lower is better.
* **Directional Accuracy** – % of times model predicts correct direction of movement.
* **Hybrid Model** – Weighted combination of ARIMA & LSTM outputs.

---
## 14. Data & Privacy
All market data pulled from public market data sources (via yfinance and scraped pages for trending/gainers/losers). No personal financial account integrations are used. Only minimal user data (username, email, hashed password) and entered portfolio/watchlist records are stored locally (SQLite in development).

---
## 15. Disclaimers
Stock Analyzer is an educational tool. Forecasts are probabilistic and may be inaccurate. Do not make investment decisions solely based on the application’s outputs.

---
## 16. Feedback
For feature requests or issues, open a ticket in the repository or contact the development team. Your input helps prioritize enhancements such as: real‑time streaming, advanced analytics, sentiment integration, and mobile support.

Enjoy exploring the markets with Stock Analyzer!
