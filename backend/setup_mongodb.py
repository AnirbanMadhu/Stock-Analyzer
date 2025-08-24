"""
MongoDB setup and initialization script for Stock Analyzer

This script demonstrates the MongoDB collections and indexes structure.
The database will be automatically created when the application runs.

Collections:
1. users - User accounts
2. portfolios - User stock holdings
3. watchlists - User stock watchlists

Indexes:
- users: username (unique), email (unique)
- portfolios: user_id + ticker
- watchlists: user_id + ticker (unique)
"""

from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_mongodb():
    """Setup MongoDB collections and indexes"""
    try:
        # Get MongoDB URI from environment
        mongo_uri = os.getenv('MONGODB_URI')
        if not mongo_uri:
            print("❌ MONGODB_URI not found in environment variables")
            return False
        
        # Connect to MongoDB
        client = MongoClient(mongo_uri)
        db_name = os.getenv('MONGO_DB', 'stock_analyzer')
        db = client[db_name]
        
        # Test connection
        db.command('ping')
        print(f"✅ Connected to MongoDB database: {db_name}")
        
        # Create collections and indexes
        print("📊 Setting up collections and indexes...")
        
        # Users collection
        users_collection = db.users
        users_collection.create_index("username", unique=True)
        users_collection.create_index("email", unique=True)
        print("  ✅ Users collection configured")
        
        # Portfolios collection
        portfolios_collection = db.portfolios
        portfolios_collection.create_index([("user_id", 1), ("ticker", 1)])
        print("  ✅ Portfolios collection configured")
        
        # Watchlists collection
        watchlists_collection = db.watchlists
        watchlists_collection.create_index([("user_id", 1), ("ticker", 1)], unique=True)
        print("  ✅ Watchlists collection configured")
        
        print(f"🎉 MongoDB setup completed successfully for database: {db_name}")
        
        # Show database stats
        stats = db.command("dbstats")
        print(f"📈 Database size: {stats.get('dataSize', 0)} bytes")
        print(f"📊 Collections: {len(db.list_collection_names())}")
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB setup failed: {e}")
        return False
    finally:
        try:
            client.close()
        except:
            pass

if __name__ == "__main__":
    print("🚀 Starting MongoDB setup for Stock Analyzer...")
    setup_mongodb()
