#!/bin/bash
# Startup script for Stock Analyzer - Production-like environment

echo "🚀 Starting Stock Analyzer in production mode..."

# Set environment variables for production-like testing
export FLASK_ENV=production
export NODE_ENV=production
export PORT=3001

# Check if MongoDB URI is set
if [ -z "$MONGODB_URI" ]; then
    echo "⚠️  MONGODB_URI not set. Using local MongoDB..."
    export MONGO_HOST=localhost
    export MONGO_PORT=27017
    export MONGO_DB=stock_analyzer
fi

# Build frontend
echo "📦 Building frontend..."
cd frontend
npm ci
npm run build
cd ..

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd backend
pip install -r requirements.txt

# Start the application
echo "🌐 Starting application on port $PORT..."
gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
