#!/usr/bin/env python3
"""
Health check script for Stock Analyzer deployment
This script verifies that all components are working correctly
"""

import os
import sys
import requests
import json
from datetime import datetime

def print_status(message, status="INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {status}: {message}")

def check_environment_variables():
    """Check if required environment variables are set"""
    print_status("Checking environment variables...")
    
    required_vars = ['SECRET_KEY', 'MONGODB_URI', 'FLASK_ENV']
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print_status(f"Missing environment variables: {', '.join(missing_vars)}", "ERROR")
        return False
    else:
        print_status("All required environment variables are set", "SUCCESS")
        return True

def check_mongodb_connection():
    """Check MongoDB connection"""
    print_status("Checking MongoDB connection...")
    
    try:
        from pymongo import MongoClient
        mongo_uri = os.getenv('MONGODB_URI')
        
        if not mongo_uri:
            print_status("MONGODB_URI not set", "ERROR")
            return False
        
        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        print_status("MongoDB connection successful", "SUCCESS")
        return True
        
    except Exception as e:
        print_status(f"MongoDB connection failed: {e}", "ERROR")
        return False

def check_flask_app():
    """Check if Flask app can be imported and initialized"""
    print_status("Checking Flask app...")
    
    try:
        # Add backend to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
        
        from app import app
        with app.test_client() as client:
            response = client.get('/health')
            if response.status_code == 200:
                print_status("Flask app health check passed", "SUCCESS")
                return True
            else:
                print_status(f"Flask app health check failed: {response.status_code}", "ERROR")
                return False
                
    except Exception as e:
        print_status(f"Flask app check failed: {e}", "ERROR")
        return False

def check_frontend_build():
    """Check if frontend is built"""
    print_status("Checking frontend build...")
    
    dist_path = os.path.join(os.path.dirname(__file__), 'frontend', 'dist')
    index_path = os.path.join(dist_path, 'index.html')
    
    if os.path.exists(index_path):
        print_status("Frontend build found", "SUCCESS")
        return True
    else:
        print_status("Frontend build not found. Run 'npm run build' in frontend directory.", "ERROR")
        return False

def check_dependencies():
    """Check if all Python dependencies are installed"""
    print_status("Checking Python dependencies...")
    
    try:
        requirements_path = os.path.join(os.path.dirname(__file__), 'backend', 'requirements.txt')
        
        if not os.path.exists(requirements_path):
            print_status("requirements.txt not found", "ERROR")
            return False
        
        with open(requirements_path, 'r') as f:
            requirements = f.read().splitlines()
        
        missing_packages = []
        for req in requirements:
            if req.strip() and not req.startswith('#'):
                package_name = req.split('>=')[0].split('==')[0].split('<')[0].strip()
                try:
                    __import__(package_name.replace('-', '_'))
                except ImportError:
                    missing_packages.append(package_name)
        
        if missing_packages:
            print_status(f"Missing packages: {', '.join(missing_packages)}", "ERROR")
            return False
        else:
            print_status("All required packages are installed", "SUCCESS")
            return True
            
    except Exception as e:
        print_status(f"Dependency check failed: {e}", "ERROR")
        return False

def main():
    """Run all health checks"""
    print_status("Starting Stock Analyzer health check...")
    print("=" * 50)
    
    checks = [
        ("Environment Variables", check_environment_variables),
        ("MongoDB Connection", check_mongodb_connection),
        ("Flask Application", check_flask_app),
        ("Frontend Build", check_frontend_build),
        ("Python Dependencies", check_dependencies)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 30)
        if check_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print_status(f"Health check completed: {passed}/{total} checks passed")
    
    if passed == total:
        print_status("🎉 All checks passed! Your application is ready for deployment.", "SUCCESS")
        return 0
    else:
        print_status("❌ Some checks failed. Please address the issues above.", "ERROR")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
