# Deployment Guide

## Local Development (Quick Start)

### Windows
```cmd
cd Stock-Analyzer
start-dev.bat
```

### macOS/Linux
```bash
cd Stock-Analyzer
chmod +x start-dev.sh
./start-dev.sh
```

## Manual Setup

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python init_db.py
python app.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Environment Variables

Create `.env` file in backend directory:
```
FLASK_ENV=development
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///stock_analyzer.db
```

## Production Deployment

### Backend (Flask)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Frontend (React)
```bash
npm run build
# Serve dist/ folder with nginx or Apache
```

## Docker Deployment (Optional)

Create `Dockerfile` for containerized deployment:

```dockerfile
# Backend Dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
```

## Database Setup

### SQLite (Development)
- Automatic setup with `python init_db.py`
- Database file: `stock_analyzer.db`

### MySQL (Production)
```sql
CREATE DATABASE stock_analyzer;
CREATE USER 'stock_user'@'localhost' IDENTIFIED BY 'password';
GRANT ALL PRIVILEGES ON stock_analyzer.* TO 'stock_user'@'localhost';
```

Update `.env`:
```
DATABASE_URL=mysql://stock_user:password@localhost/stock_analyzer
```
