# Quick Deploy Commands - Stock Analyzer

## 🚀 For Render Deployment

### Build Command (Copy this to Render)
```bash
npm run build
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

⚠️ **Note for Windows users:** The start_prod.bat script uses gunicorn which requires Unix-like systems. On Windows, use `python app.py` for local testing instead.

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

### Common Render Build Errors

**Error: `sudo: command not found`**
- **Solution:** Remove `sudo` and Node.js installation from build command
- **Use:** `npm run build && cd backend && pip install --no-cache-dir -r requirements.txt`

**Error: `Failed writing body`**
- **Solution:** Render already has Node.js, don't try to install it
- **Use the simplified build command above**

**Error: `vite: not found`**
- **Solution:** Need to install devDependencies for build tools
- **Fixed:** Changed from `npm ci` to `npm install` in build script
- **Root cause:** Vite is in devDependencies but `npm ci --only=production` excludes it

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
