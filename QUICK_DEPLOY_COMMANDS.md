# Quick Deploy Commands - Stock Analyzer

## 🚀 For Render Deployment

### Build Command (Copy this to Render)
```bash
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash - && sudo apt-get install -y nodejs && cd frontend && npm ci --only=production && npm run build && cd ../backend && pip install --no-cache-dir -r requirements.txt
```

### Start Command (Copy this to Render)
```bash
cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 1 --timeout 120 app:app
```

### Optimized Start Command (For better performance)
```bash
cd backend && gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --worker-class gthread --max-requests 1000 --preload app:app
```

## 🔧 Environment Variables for Render

Add these in your Render service Environment tab:

```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/stock_analyzer
SECRET_KEY=auto-generated-by-render
FLASK_ENV=production
NODE_ENV=production
MONGO_DB=stock_analyzer
```

## 🧪 Local Testing Commands

### Test Build Locally (Windows)
```cmd
build.bat
```

### Test Build Locally (Linux/Mac)
```bash
chmod +x build.sh
./build.sh
```

### Test Production Start (Windows)
```cmd
start_prod.bat
```

### Test Production Start (Linux/Mac)
```bash
chmod +x start_production.sh
./start_production.sh
```

## 📋 Quick Deployment Checklist

- [ ] MongoDB Atlas cluster created
- [ ] Connection string ready
- [ ] GitHub repository updated
- [ ] Build command tested locally
- [ ] Environment variables prepared
- [ ] Ready to deploy on Render!

## 🔗 Quick Links

- **MongoDB Atlas**: https://cloud.mongodb.com/
- **Render**: https://dashboard.render.com/
- **GitHub**: https://github.com/

## 🆘 Troubleshooting

### Build Fails
```bash
# Check versions
node --version && npm --version && python --version

# Test frontend build
cd frontend && npm ci && npm run build

# Test backend
cd backend && pip install -r requirements.txt
```

### App Won't Start
```bash
# Test Flask import
cd backend && python -c "import app; print('Success')"

# Check gunicorn
cd backend && gunicorn --check-config app:app
```

### Database Issues
- Verify MONGODB_URI format
- Check MongoDB Atlas network access (allow 0.0.0.0/0)
- Ensure database user has read/write permissions

---

**Copy and paste these commands directly into Render! 🚀**
