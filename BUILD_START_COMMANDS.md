# Build and Start Commands for Stock Analyzer

This document provides optimized build and start commands for deploying Stock Analyzer on various platforms.

## 🚀 Render Deployment Commands

### Build Command (Optimized)
```bash
# Install Node.js 18 (LTS)
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installations
echo "=== Checking Versions ==="
node --version
npm --version
python --version

# Clean any existing builds
echo "=== Cleaning Previous Builds ==="
rm -rf frontend/dist frontend/node_modules/.cache

# Build frontend with production optimizations
echo "=== Building Frontend ==="
cd frontend
npm ci --only=production
npm run build
echo "Frontend build completed ✓"
cd ..

# Install backend dependencies with optimizations
echo "=== Installing Backend Dependencies ==="
cd backend
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir -r requirements.txt
echo "Backend dependencies installed ✓"
cd ..

# Verify build outputs
echo "=== Build Verification ==="
ls -la frontend/dist/
echo "Frontend files:"
find frontend/dist -type f -name "*.html" -o -name "*.js" -o -name "*.css" | head -10

# Check Python app can import
echo "=== Testing App Import ==="
cd backend && python -c "import app; print('Flask app imports successfully ✓')" && cd ..

echo "=== Build Complete ==="
```

### Start Command (Production)
```bash
cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --worker-class gthread --max-requests 1000 --max-requests-jitter 100 --preload app:app
```

### Alternative Start Command (Simpler)
```bash
cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
```

## 📋 Command Breakdown

### Build Command Explanation

1. **Node.js Installation**:
   ```bash
   curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
   sudo apt-get install -y nodejs
   ```
   - Installs Node.js 18 (LTS version)
   - Required for building React frontend

2. **Version Verification**:
   ```bash
   node --version && npm --version && python --version
   ```
   - Confirms all tools are properly installed
   - Helps debug build issues

3. **Frontend Build**:
   ```bash
   cd frontend
   npm ci --only=production
   npm run build
   cd ..
   ```
   - `npm ci` is faster and more reliable than `npm install`
   - `--only=production` skips dev dependencies
   - Creates optimized build in `frontend/dist/`

4. **Backend Dependencies**:
   ```bash
   cd backend
   pip install --no-cache-dir -r requirements.txt
   cd ..
   ```
   - `--no-cache-dir` prevents cache issues on Render
   - Installs all Python packages

5. **Build Verification**:
   ```bash
   ls -la frontend/dist/
   ```
   - Confirms frontend build succeeded
   - Lists generated files

### Start Command Explanation

```bash
cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
```

- `cd backend` - Navigate to backend directory
- `gunicorn` - Production WSGI server
- `--bind 0.0.0.0:$PORT` - Bind to all interfaces on Render's port
- `--workers 1` - Single worker (sufficient for free tier)
- `--timeout 120` - 2-minute timeout for long requests
- `app:app` - Import `app` from `app.py`

## 🛠️ Platform-Specific Commands

### For Render (render.yaml)
```yaml
buildCommand: |
  curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
  sudo apt-get install -y nodejs
  cd frontend && npm ci && npm run build && cd ..
  cd backend && pip install --no-cache-dir -r requirements.txt && cd ..
startCommand: cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
```

### For Heroku (Procfile)
```
web: cd backend && gunicorn --bind 0.0.0.0:$PORT app:app
release: cd backend && python -c "from app import app; from database import init_database; init_database(app)"
```

### For Railway
```json
{
  "build": {
    "builder": "nixpacks"
  },
  "deploy": {
    "startCommand": "cd backend && gunicorn --bind 0.0.0.0:$PORT app:app",
    "restartPolicyType": "ON_FAILURE"
  }
}
```

### For Digital Ocean App Platform
```yaml
services:
- name: stock-analyzer
  source_dir: /
  github:
    repo: your-username/stock-analyzer
    branch: main
  run_command: cd backend && gunicorn --bind 0.0.0.0:$PORT app:app
  build_command: |
    cd frontend && npm ci && npm run build && cd ..
    cd backend && pip install -r requirements.txt
```

## 🔧 Development Commands

### Local Development Build
```bash
# Build frontend for testing
cd frontend
npm install
npm run build
cd ..

# Test backend with production settings
cd backend
pip install -r requirements.txt
FLASK_ENV=production python app.py
```

### Local Production Simulation
```bash
# Windows
set FLASK_ENV=production
set NODE_ENV=production
set PORT=3001
cd frontend && npm run build && cd ..
cd backend && gunicorn --bind 0.0.0.0:3001 app:app

# Linux/Mac
export FLASK_ENV=production
export NODE_ENV=production
export PORT=3001
cd frontend && npm run build && cd ..
cd backend && gunicorn --bind 0.0.0.0:3001 app:app
```

## 🚨 Troubleshooting Commands

### Debug Build Issues
```bash
# Check Node.js installation
which node && node --version

# Check npm installation
which npm && npm --version

# Check Python packages
cd backend && pip list

# Check frontend build output
ls -la frontend/dist/

# Test Flask app import
cd backend && python -c "import app; print('Success')"

# Check gunicorn can start
cd backend && gunicorn --check-config app:app
```

### Test Frontend Build
```bash
cd frontend
npm run build
python -m http.server 3000 --directory dist
# Visit http://localhost:3000
```

### Test Backend Standalone
```bash
cd backend
python app.py
# Visit http://localhost:3001
```

## ⚡ Performance-Optimized Commands

### High-Performance Build (for paid plans)
```bash
# Multi-threaded build
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
cd frontend
npm ci --prefer-offline --no-audit
npm run build
cd ../backend
pip install --no-cache-dir --compile -r requirements.txt
```

### High-Performance Start (for paid plans)
```bash
cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 2 --threads 4 --worker-class gthread --worker-connections 1000 --max-requests 1000 --max-requests-jitter 100 --timeout 120 --keep-alive 2 --preload app:app
```

## 📝 Command Templates

### Render Blueprint (render.yaml)
```yaml
services:
  - type: web
    name: stock-analyzer
    runtime: python
    buildCommand: |
      curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
      sudo apt-get install -y nodejs
      cd frontend && npm ci && npm run build && cd ..
      cd backend && pip install --no-cache-dir -r requirements.txt
    startCommand: cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
    plan: free
```

### Docker Commands
```dockerfile
# Build stage
FROM node:18-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci --only=production
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS backend
WORKDIR /app
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY --from=frontend-build /app/frontend/dist ./static

EXPOSE $PORT
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "--workers", "1", "--timeout", "120", "app:app"]
```

## ✅ Verification Commands

After deployment, test these endpoints:

```bash
# Health check
curl https://your-app.onrender.com/health

# API health
curl https://your-app.onrender.com/api/health

# Frontend (should return HTML)
curl https://your-app.onrender.com/

# Test API endpoint
curl "https://your-app.onrender.com/api/search?q=AAPL"
```

## 🎯 Quick Deploy Checklist

- [ ] MongoDB connection string ready
- [ ] GitHub repository updated
- [ ] Environment variables prepared
- [ ] Build command tested locally
- [ ] Start command tested locally
- [ ] Ready to deploy! 🚀

---

**These commands will deploy your Stock Analyzer with all features intact!** 📈
