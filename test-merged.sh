#!/bin/bash

# Local test script for merged deployment
echo "🧪 Testing merged deployment locally..."

# Build frontend
echo "📦 Building frontend..."
cd frontend
npm install
npm run build

if [ ! -d "dist" ]; then
    echo "❌ Frontend build failed - dist directory not found"
    exit 1
fi

echo "✅ Frontend built successfully"
cd ..

# Test backend serving static files
echo "🚀 Starting backend server..."
cd backend
echo "🌐 Server will be available at: http://localhost:3001"
echo "📊 API health check: http://localhost:3001/health"
echo "🎯 Frontend: http://localhost:3001"
echo ""
echo "Press Ctrl+C to stop the server"

python app.py
