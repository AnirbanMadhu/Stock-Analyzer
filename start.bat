@echo off
REM Stock Analyzer Startup Script for Windows
echo ====================================
echo Starting Stock Analyzer Application
echo ====================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if Node.js is installed
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Node.js is not installed or not in PATH
    pause
    exit /b 1
)

REM Install backend dependencies
echo Installing backend dependencies...
cd backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo Error: Failed to install backend dependencies
    pause
    exit /b 1
)

REM Install frontend dependencies and build
echo Installing frontend dependencies...
cd ..\frontend
call npm install
if %errorlevel% neq 0 (
    echo Error: Failed to install frontend dependencies
    pause
    exit /b 1
)

echo Building frontend...
call npm run build
if %errorlevel% neq 0 (
    echo Error: Failed to build frontend
    pause
    exit /b 1
)

REM Start the application
echo Starting Stock Analyzer...
cd ..\backend
python app.py

pause
