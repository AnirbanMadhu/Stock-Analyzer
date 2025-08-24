# Database Setup Documentation

## Overview
The Stock Analyzer backend uses SQLite as its database with SQLAlchemy as the ORM. The database connection is properly configured and all tables are created automatically.

## Database Configuration

### Location
- **Database File**: `backend/instance/stock_analyzer.db`
- **Configuration**: `backend/database.py`
- **Models**: `backend/models/`

### Connection Details
- **Engine**: SQLite
- **URI**: `sqlite:///stock_analyzer.db`
- **Pool Settings**: Pre-ping enabled, 300s recycle time
- **Auto-creation**: Tables created automatically on app startup

## Database Models

### 1. Users Table
```sql
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    username VARCHAR(80) UNIQUE NOT NULL,
    email VARCHAR(120) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at DATETIME,
    is_active BOOLEAN
);
```
**Purpose**: User authentication and account management

### 2. Portfolios Table
```sql
CREATE TABLE portfolios (
    id INTEGER PRIMARY KEY,
    user_id INTEGER FOREIGN KEY REFERENCES users(id),
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    quantity FLOAT NOT NULL,
    purchase_price NUMERIC(10,2) NOT NULL,
    purchase_date DATETIME,
    created_at DATETIME,
    updated_at DATETIME
);
```
**Purpose**: Track user stock holdings and calculate P&L

### 3. Watchlists Table
```sql
CREATE TABLE watchlists (
    id INTEGER PRIMARY KEY,
    user_id INTEGER FOREIGN KEY REFERENCES users(id),
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    added_at DATETIME,
    notes TEXT,
    UNIQUE(user_id, ticker)
);
```
**Purpose**: Monitor stocks of interest

## Database Operations

### Initialization
```bash
# Initialize database with tables
python backend/init_database.py

# Verify database connection
python backend/init_database.py --verify

# Run comprehensive tests
python backend/test_database.py
```

### Demo Data
- **Demo User**: username: `demo`, password: `demo123`
- **Email**: `demo@example.com`

## Connection Status ✅

- **Database Connected**: ✅ YES
- **Tables Created**: ✅ YES (users, portfolios, watchlists)
- **Models Working**: ✅ YES (Create, Read, Update, Delete)
- **Foreign Keys**: ✅ YES (User relationships working)
- **Flask Integration**: ✅ YES (Auto-initialization on startup)

## API Endpoints Using Database

### Authentication Endpoints
- `POST /api/auth/register` - Create new user
- `POST /api/auth/login` - Authenticate user
- `GET /api/auth/profile` - Get user profile
- `PUT /api/auth/profile` - Update user profile

### Portfolio Endpoints
- `GET /api/portfolio/holdings` - Get user portfolio
- `POST /api/portfolio/add` - Add stock to portfolio
- `PUT /api/portfolio/update/{id}` - Update portfolio holding
- `DELETE /api/portfolio/remove/{id}` - Remove from portfolio

### Watchlist Endpoints
- `GET /api/portfolio/watchlist` - Get user watchlist
- `POST /api/portfolio/watchlist/add` - Add to watchlist
- `DELETE /api/portfolio/watchlist/remove/{id}` - Remove from watchlist

## Troubleshooting

### Common Issues
1. **Database file not found**: Run `python init_database.py`
2. **Tables missing**: Check that `db.create_all()` is called in app context
3. **Foreign key errors**: Ensure user exists before creating portfolios/watchlists

### Verification Commands
```bash
# Check database file exists
ls backend/instance/stock_analyzer.db

# Test database connection
python backend/test_database.py

# View database structure
sqlite3 backend/instance/stock_analyzer.db ".schema"
```

## Security Notes
- Passwords are hashed using bcrypt
- Session management via Flask-Login
- CORS configured for frontend integration
- Database queries use parameterized statements (SQLAlchemy ORM)

---

**Status**: ✅ Database fully connected and operational!
