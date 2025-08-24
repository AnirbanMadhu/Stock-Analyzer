#!/bin/bash
# Optimized build script for Render deployment
# This script builds both frontend and backend for production

set -e  # Exit on any error

echo "🚀 Starting Stock Analyzer build process..."
echo "======================================"

# Install Node.js 18 (LTS)
echo "📦 Installing Node.js..."
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installations
echo "🔍 Checking versions..."
echo "Node.js: $(node --version)"
echo "NPM: $(npm --version)"  
echo "Python: $(python --version)"

# Clean any existing builds
echo "🧹 Cleaning previous builds..."
rm -rf frontend/dist
rm -rf frontend/node_modules/.cache

# Build frontend
echo "⚛️ Building React frontend..."
cd frontend
npm ci --only=production --silent
npm run build
echo "✅ Frontend build completed"

# Verify frontend build
if [ ! -f "dist/index.html" ]; then
    echo "❌ Frontend build failed - index.html not found"
    exit 1
fi

echo "📁 Frontend build contents:"
ls -la dist/

cd ..

# Install backend dependencies
echo "🐍 Installing Python dependencies..."
cd backend
pip install --no-cache-dir --upgrade pip --quiet
pip install --no-cache-dir -r requirements.txt --quiet
echo "✅ Backend dependencies installed"

# Test Flask app import
echo "🧪 Testing Flask app..."
python -c "
try:
    import app
    print('✅ Flask app imports successfully')
except Exception as e:
    print(f'❌ Flask app import failed: {e}')
    exit(1)
"

cd ..

echo "======================================"
echo "🎉 Build completed successfully!"
echo "🌐 Ready for deployment on Render"
echo "======================================"
