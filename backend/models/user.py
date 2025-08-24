from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from bson import ObjectId

def get_mongo():
    """Get mongo instance to avoid circular imports"""
    from database import mongo
    return mongo

class User(UserMixin):
    """User model for authentication and user data storage using MongoDB"""
    
    def __init__(self, username=None, email=None, password=None, user_id=None, **kwargs):
        if user_id:
            # Load existing user
            self._load_user(user_id)
        else:
            # Create new user
            self.id = None
            self.username = username
            self.email = email
            self.password_hash = None
            self.created_at = datetime.utcnow()
            self._is_active = True
            
            if password:
                self.set_password(password)
            
            # Handle additional fields from kwargs
            for key, value in kwargs.items():
                if key == 'is_active':
                    self._is_active = value
                else:
                    setattr(self, key, value)
    
    @property
    def is_active(self):
        """Get user active status"""
        return getattr(self, '_is_active', True)
    
    @is_active.setter
    def is_active(self, value):
        """Set user active status"""
        self._is_active = value
    
    def _load_user(self, user_id):
        """Load user data from MongoDB"""
        try:
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
            
            mongo = get_mongo()
            if mongo.db is None:
                raise RuntimeError("Database not initialized")
                
            user_data = mongo.db.users.find_one({'_id': user_id})
            if user_data:
                self.id = str(user_data['_id'])
                self.username = user_data.get('username')
                self.email = user_data.get('email')
                self.password_hash = user_data.get('password_hash')
                self.created_at = user_data.get('created_at', datetime.utcnow())
                self._is_active = user_data.get('is_active', True)
            else:
                raise ValueError(f"User with id {user_id} not found")
        except Exception as e:
            raise ValueError(f"Failed to load user: {e}")
    
    def get_id(self):
        """Return user ID for Flask-Login"""
        return str(self.id) if self.id else None
    
    def set_password(self, password):
        """Hash and set the user password"""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check if the provided password matches the stored hash"""
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)
    
    def save(self):
        """Save user to MongoDB"""
        mongo = get_mongo()
        if mongo.db is None:
            raise RuntimeError("Database not initialized")
            
        user_data = {
            'username': self.username,
            'email': self.email,
            'password_hash': self.password_hash,
            'created_at': self.created_at,
            'is_active': self.is_active
        }
        
        if self.id:
            # Update existing user
            mongo.db.users.update_one(
                {'_id': ObjectId(self.id)},
                {'$set': user_data}
            )
        else:
            # Create new user
            result = mongo.db.users.insert_one(user_data)
            self.id = str(result.inserted_id)
        
        return self
    
    def delete(self):
        """Delete user from MongoDB"""
        if self.id:
            mongo = get_mongo()
            if mongo.db is None:
                raise RuntimeError("Database not initialized")
            mongo.db.users.delete_one({'_id': ObjectId(self.id)})
            # Also delete related data
            mongo.db.portfolios.delete_many({'user_id': self.id})
            mongo.db.watchlists.delete_many({'user_id': self.id})
    
    def to_dict(self):
        """Convert user object to dictionary for JSON serialization"""
        try:
            return {
                'id': self.id,
                'username': self.username,
                'email': self.email,
                'created_at': self.created_at.isoformat() if self.created_at else None,
                'is_active': self.is_active
            }
        except Exception as e:
            # Fallback to basic user info if there's an error
            return {
                'id': self.id,
                'username': self.username,
                'email': self.email
            }
    
    @staticmethod
    def find_by_username(username):
        """Find user by username"""
        mongo = get_mongo()
        if mongo.db is None:
            return None
            
        user_data = mongo.db.users.find_one({'username': username})
        if user_data:
            user = User()
            user.id = str(user_data['_id'])
            user.username = user_data.get('username')
            user.email = user_data.get('email')
            user.password_hash = user_data.get('password_hash')
            user.created_at = user_data.get('created_at', datetime.utcnow())
            user._is_active = user_data.get('is_active', True)
            return user
        return None
    
    @staticmethod
    def find_by_email(email):
        """Find user by email"""
        mongo = get_mongo()
        if mongo.db is None:
            return None
            
        user_data = mongo.db.users.find_one({'email': email})
        if user_data:
            user = User()
            user.id = str(user_data['_id'])
            user.username = user_data.get('username')
            user.email = user_data.get('email')
            user.password_hash = user_data.get('password_hash')
            user.created_at = user_data.get('created_at', datetime.utcnow())
            user._is_active = user_data.get('is_active', True)
            return user
        return None
    
    @staticmethod
    def find_by_id(user_id):
        """Find user by ID"""
        try:
            if isinstance(user_id, str):
                user_id = ObjectId(user_id)
            
            mongo = get_mongo()
            if mongo.db is None:
                return None
                
            user_data = mongo.db.users.find_one({'_id': user_id})
            if user_data:
                user = User()
                user.id = str(user_data['_id'])
                user.username = user_data.get('username')
                user.email = user_data.get('email')
                user.password_hash = user_data.get('password_hash')
                user.created_at = user_data.get('created_at', datetime.utcnow())
                user._is_active = user_data.get('is_active', True)
                return user
            return None
        except Exception:
            return None
    
    def __repr__(self):
        return f'<User {self.username}>'
