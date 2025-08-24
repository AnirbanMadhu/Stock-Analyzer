@echo off
REM Startup script for Stock Analyzer - Production-like environment (Windows)

echo 🚀 Starting Stock Analyzer in production mode...

REM Set environment variables for production-like testing
set FLASK_ENV=production
set NODE_ENV=production
set PORT=3001

REM Check if MongoDB URI is set
if "%MONGODB_URI%"=="" (
    echo ⚠️  MONGODB_URI not set. Using local MongoDB...
    set MONGO_HOST=localhost
    set MONGO_PORT=27017
    set MONGO_DB=stock_analyzer
)

REM Build frontend
echo 📦 Building frontend...
cd frontend
call npm ci
call npm run build
cd ..

REM Install backend dependencies
echo 📦 Installing backend dependencies...
cd backend
pip install -r requirements.txt

REM Start the application
echo 🌐 Starting application on port %PORT%...
gunicorn --bind 0.0.0.0:%PORT% --workers 1 --timeout 120 app:app
