@echo off
echo 🧪 Testing merged deployment locally...

REM Build frontend
echo 📦 Building frontend...
cd frontend
call npm install
call npm run build

if not exist "dist" (
    echo ❌ Frontend build failed - dist directory not found
    pause
    exit /b 1
)

echo ✅ Frontend built successfully
cd ..

REM Test backend serving static files
echo 🚀 Starting backend server...
cd backend
echo 🌐 Server will be available at: http://localhost:3001
echo 📊 API health check: http://localhost:3001/health
echo 🎯 Frontend: http://localhost:3001
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py
pause
