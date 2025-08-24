@echo off
REM Production start script for Stock Analyzer on Windows
REM This script starts the application using gunicorn in production mode

echo 🚀 Starting Stock Analyzer in production mode...
echo ===============================================

REM Check if we're in the right directory
if not exist "backend\app.py" (
    echo ❌ Error: backend\app.py not found
    echo Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if frontend is built
if not exist "frontend\dist\index.html" (
    echo ⚠️  Warning: Frontend not built
    echo Running build first...
    call build.bat
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Build failed
        pause
        exit /b 1
    )
)

REM Set production environment variables
set FLASK_ENV=production
set NODE_ENV=production
if "%PORT%"=="" set PORT=3001

echo 🔧 Configuration:
echo    FLASK_ENV: %FLASK_ENV%
echo    NODE_ENV: %NODE_ENV%
echo    PORT: %PORT%

REM Check if MongoDB URI is set
if "%MONGODB_URI%"=="" (
    echo ⚠️  MONGODB_URI not set - using local MongoDB fallback
    if "%MONGO_HOST%"=="" set MONGO_HOST=localhost
    if "%MONGO_PORT%"=="" set MONGO_PORT=27017
    if "%MONGO_DB%"=="" set MONGO_DB=stock_analyzer
    echo    MongoDB: %MONGO_HOST%:%MONGO_PORT%/%MONGO_DB%
) else (
    echo    MongoDB: Connected via MONGODB_URI
)

REM Navigate to backend directory
cd backend

REM Check if gunicorn is installed
where gunicorn >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Gunicorn not found. Installing...
    pip install gunicorn
    if %ERRORLEVEL% NEQ 0 (
        echo ❌ Failed to install gunicorn
        cd ..
        pause
        exit /b 1
    )
)

REM Test app import before starting
echo 🧪 Testing Flask app import...
python -c "import app; print('✅ Flask app imports successfully')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Flask app import failed
    cd ..
    pause
    exit /b 1
)

echo ===============================================
echo 🌐 Starting application on port %PORT%...
echo 🔗 URL: http://localhost:%PORT%
echo 🛑 Press Ctrl+C to stop
echo ===============================================

REM Start with gunicorn
gunicorn --bind 0.0.0.0:%PORT% --workers 1 --threads 2 --timeout 120 --worker-class gthread --max-requests 1000 --max-requests-jitter 100 --access-logfile - --error-logfile - --log-level info --preload app:app
