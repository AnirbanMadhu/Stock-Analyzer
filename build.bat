@echo off
REM Optimized build script for Windows development testing
REM This script builds both frontend and backend for production testing

echo 🚀 Starting Stock Analyzer build process...
echo ======================================

REM Check if Node.js is installed
where node >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Node.js not found. Please install Node.js 18+ first.
    echo Download from: https://nodejs.org/
    pause
    exit /b 1
)

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Python not found. Please install Python 3.9+ first.
    echo Download from: https://python.org/
    pause
    exit /b 1
)

REM Verify installations
echo 🔍 Checking versions...
node --version
npm --version
python --version

REM Clean any existing builds
echo 🧹 Cleaning previous builds...
if exist "frontend\dist" rmdir /s /q "frontend\dist"
if exist "frontend\node_modules\.cache" rmdir /s /q "frontend\node_modules\.cache"

REM Build frontend
echo ⚛️ Building React frontend...
cd frontend
call npm ci --only=production --silent
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Frontend npm install failed
    cd ..
    pause
    exit /b 1
)

call npm run build
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Frontend build failed
    cd ..
    pause
    exit /b 1
)

echo ✅ Frontend build completed

REM Verify frontend build
if not exist "dist\index.html" (
    echo ❌ Frontend build failed - index.html not found
    cd ..
    pause
    exit /b 1
)

echo 📁 Frontend build contents:
dir dist\

cd ..

REM Install backend dependencies
echo 🐍 Installing Python dependencies...
cd backend
pip install --upgrade pip --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Pip upgrade failed
    cd ..
    pause
    exit /b 1
)

pip install -r requirements.txt --quiet
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Backend dependencies installation failed
    cd ..
    pause
    exit /b 1
)

echo ✅ Backend dependencies installed

REM Test Flask app import
echo 🧪 Testing Flask app...
python -c "import app; print('✅ Flask app imports successfully')"
if %ERRORLEVEL% NEQ 0 (
    echo ❌ Flask app import failed
    cd ..
    pause
    exit /b 1
)

cd ..

echo ======================================
echo 🎉 Build completed successfully!
echo 🌐 Ready for deployment testing
echo ======================================
pause
