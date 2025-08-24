#!/bin/bash
# Stock Analyzer Startup Script for Unix/Linux/MacOS

echo "===================================="
echo "Starting Stock Analyzer Application"
echo "===================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed or not in PATH"
    exit 1
fi

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend || exit 1
$PYTHON_CMD -m pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install backend dependencies"
    exit 1
fi

# Install frontend dependencies and build
echo "Installing frontend dependencies..."
cd ../frontend || exit 1
npm install
if [ $? -ne 0 ]; then
    echo "Error: Failed to install frontend dependencies"
    exit 1
fi

echo "Building frontend..."
npm run build
if [ $? -ne 0 ]; then
    echo "Error: Failed to build frontend"
    exit 1
fi

# Start the application
echo "Starting Stock Analyzer..."
cd ../backend || exit 1
$PYTHON_CMD app.py
