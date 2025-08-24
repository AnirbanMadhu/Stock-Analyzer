"""
MongoDB Database configuration and connection settings for Stock Analyzer
"""
import os
from pymongo import MongoClient
from flask_pymongo import PyMongo
from bson import ObjectId
import logging

logger = logging.getLogger(__name__)

class DatabaseConfig:
    """MongoDB Database configuration class"""
    
    @classmethod
    def get_database_uri(cls):
        """Get the MongoDB URI from environment"""
        # Get MongoDB URI from environment
        mongo_uri = os.getenv('MONGODB_URI')
        
        if not mongo_uri:
            # Fallback to local MongoDB if no URI provided
            mongo_user = os.getenv('MONGO_USER', '')
            mongo_password = os.getenv('MONGO_PASSWORD', '')
            mongo_host = os.getenv('MONGO_HOST', 'localhost')
            mongo_port = os.getenv('MONGO_PORT', '27017')
            mongo_db = os.getenv('MONGO_DB', 'stock_analyzer')
            
            if mongo_user and mongo_password:
                mongo_uri = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}/{mongo_db}"
            else:
                mongo_uri = f"mongodb://{mongo_host}:{mongo_port}/{mongo_db}"
        
        return mongo_uri
    
    @classmethod
    def get_database_name(cls):
        """Get the database name from URI or environment"""
        db_name = os.getenv('MONGO_DB', 'stock_analyzer')
        
        # Extract database name from URI if it contains one
        mongo_uri = cls.get_database_uri()
        if mongo_uri and '/' in mongo_uri:
            # Extract database name from the end of URI
            uri_parts = mongo_uri.split('/')
            if len(uri_parts) > 3:
                # Remove query parameters if any
                db_from_uri = uri_parts[-1].split('?')[0]
                if db_from_uri:
                    db_name = db_from_uri
        
        return db_name
    
    @classmethod
    def configure_app(cls, app):
        """Configure Flask app with MongoDB settings"""
        app.config['MONGO_URI'] = cls.get_database_uri()
        return app
    
    @classmethod
    def get_connection_info(cls):
        """Get database connection information"""
        return {
            'type': 'MongoDB',
            'database_uri': cls.get_database_uri(),
            'database_name': cls.get_database_name()
        }

# MongoDB instance
mongo = PyMongo()

def init_database(app):
    """Initialize MongoDB with Flask app"""
    try:
        # Initialize PyMongo with app
        mongo.init_app(app)
        
        # Test connection
        with app.app_context():
            # Try to ping the database
            if mongo.db is not None:
                mongo.db.command('ping')
                
                # Create indexes for better performance
                create_indexes()
                
                print("✅ MongoDB connection established successfully")
            else:
                print("❌ MongoDB database instance is None")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ MongoDB initialization failed: {e}")
        return False

def create_indexes():
    """Create indexes for MongoDB collections"""
    try:
        # Check if mongo.db is available
        if mongo.db is None:
            logger.warning("⚠️ MongoDB database instance is None, cannot create indexes")
            return
            
        # Create indexes for users collection
        mongo.db.users.create_index("username", unique=True)
        mongo.db.users.create_index("email", unique=True)
        
        # Create indexes for portfolio collection
        mongo.db.portfolios.create_index([("user_id", 1), ("ticker", 1)])
        
        # Create compound unique index for watchlist
        mongo.db.watchlists.create_index([("user_id", 1), ("ticker", 1)], unique=True)
        
        logger.info("✅ MongoDB indexes created successfully")
        
    except Exception as e:
        logger.warning(f"⚠️ Some indexes may already exist or failed to create: {e}")

def test_database_connection(app):
    """Test MongoDB connection"""
    try:
        with app.app_context():
            # Check if mongo.db is available
            if mongo.db is None:
                return False, "MongoDB database instance is None"
                
            # Try to ping the database
            result = mongo.db.command('ping')
            if result.get('ok') == 1.0:
                return True, "MongoDB connection successful"
            else:
                return False, "MongoDB ping failed"
    except Exception as e:
        return False, f"MongoDB connection failed: {e}"

def get_db_info():
    """Get database connection information"""
    return {
        'database_uri': DatabaseConfig.get_database_uri(),
        'database_name': DatabaseConfig.get_database_name()
    }

# Helper function to convert ObjectId to string for JSON serialization
def serialize_doc(doc):
    """Convert MongoDB document to JSON serializable format"""
    if doc is None:
        return None
    
    if isinstance(doc, list):
        return [serialize_doc(item) for item in doc]
    
    if isinstance(doc, dict):
        serialized = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                serialized[key] = str(value)
            elif isinstance(value, dict):
                serialized[key] = serialize_doc(value)
            elif isinstance(value, list):
                serialized[key] = serialize_doc(value)
            else:
                serialized[key] = value
        return serialized
    
    return doc
