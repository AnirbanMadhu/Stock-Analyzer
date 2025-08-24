#!/usr/bin/env python3
"""
MongoDB Connection Test Script
This script tests the MongoDB connection and ensures the database is properly created
"""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

# Load environment variables
load_dotenv()

MONGODB_URI = os.getenv('MONGODB_URI')
MONGO_DB = os.getenv('MONGO_DB')

if not MONGODB_URI or not MONGO_DB:
    print("❌ Error: Missing MONGODB_URI or MONGO_DB environment variables")
    exit(1)

print("MongoDB Connection Test")
print("=" * 50)
print(f"Database URI: {MONGODB_URI}")
print(f"Database Name: {MONGO_DB}")
print()

try:
    # Create MongoDB client
    client = MongoClient(MONGODB_URI)
    
    # Test connection by pinging the server
    print("Testing connection...")
    client.admin.command('ping')
    print("✅ Successfully connected to MongoDB!")
    
    # Get the database
    db = client[MONGO_DB]
    print(f"✅ Database '{MONGO_DB}' accessed successfully!")
    
    # List all collections in the database
    collections = db.list_collection_names()
    print(f"📁 Collections in database: {collections}")
    
    # If no collections exist, create them with sample data
    if not collections:
        print("⚠️  No collections found. Creating initial collections...")
        
        # Create a test user
        users_collection = db.users
        test_user = {
            "username": "test_user",
            "email": "test@example.com",
            "password_hash": "hashed_password",
            "created_at": "2025-01-01T00:00:00Z"
        }
        users_collection.insert_one(test_user)
        print("✅ Created 'users' collection with test data")
        
        # Create a test portfolio
        portfolios_collection = db.portfolios
        test_portfolio = {
            "user_id": "test_user_id",
            "investments": [
                {
                    "symbol": "AAPL",
                    "quantity": 10,
                    "purchase_price": 150.00,
                    "purchase_date": "2025-01-01"
                }
            ],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        }
        portfolios_collection.insert_one(test_portfolio)
        print("✅ Created 'portfolios' collection with test data")
        
        # Create a test watchlist
        watchlists_collection = db.watchlists
        test_watchlist = {
            "user_id": "test_user_id",
            "stocks": ["AAPL", "GOOGL", "MSFT"],
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z"
        }
        watchlists_collection.insert_one(test_watchlist)
        print("✅ Created 'watchlists' collection with test data")
    
    # Show updated collections list
    collections = db.list_collection_names()
    print(f"📁 Updated collections in database: {collections}")
    
    # Show document counts for each collection
    for collection_name in collections:
        collection = db[collection_name]
        count = collection.count_documents({})
        print(f"📊 '{collection_name}' has {count} documents")
        
        # Show sample documents
        if count > 0:
            sample = collection.find_one()
            print(f"📄 Sample document from '{collection_name}': {sample}")
            print()
    
    # Test database stats
    stats = db.command("dbStats")
    print(f"📈 Database stats:")
    print(f"   - Data size: {stats.get('dataSize', 0)} bytes")
    print(f"   - Storage size: {stats.get('storageSize', 0)} bytes")
    print(f"   - Collections: {stats.get('collections', 0)}")
    print(f"   - Objects: {stats.get('objects', 0)}")
    
    print("\n✅ MongoDB connection test completed successfully!")
    print("🔍 The database should now be visible in MongoDB Compass")
    
    # Close connection
    client.close()
    
except Exception as e:
    print(f"❌ Error connecting to MongoDB: {e}")
    import traceback
    traceback.print_exc()
