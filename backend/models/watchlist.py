from datetime import datetime
from bson import ObjectId

def get_mongo():
    """Get mongo instance to avoid circular imports"""
    from database import mongo
    return mongo

class Watchlist:
    """Watchlist model for tracking stocks user wants to monitor using MongoDB"""
    
    def __init__(self, user_id=None, ticker=None, company_name=None, notes=None, watchlist_id=None, **kwargs):
        if watchlist_id:
            # Load existing watchlist
            self._load_watchlist(watchlist_id)
        else:
            # Create new watchlist entry
            self.id = None
            self.user_id = user_id
            self.ticker = ticker.upper() if ticker else None
            self.company_name = company_name
            self.notes = notes
            self.added_at = datetime.utcnow()
            
            # Handle additional fields from kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def _load_watchlist(self, watchlist_id):
        """Load watchlist data from MongoDB"""
        try:
            if isinstance(watchlist_id, str):
                watchlist_id = ObjectId(watchlist_id)
            
            mongo = get_mongo()
            if mongo.db is None:
                raise RuntimeError("Database not initialized")
                
            watchlist_data = mongo.db.watchlists.find_one({'_id': watchlist_id})
            if watchlist_data:
                self.id = str(watchlist_data['_id'])
                self.user_id = watchlist_data.get('user_id')
                self.ticker = watchlist_data.get('ticker')
                self.company_name = watchlist_data.get('company_name')
                self.notes = watchlist_data.get('notes')
                self.added_at = watchlist_data.get('added_at', datetime.utcnow())
            else:
                raise ValueError(f"Watchlist with id {watchlist_id} not found")
        except Exception as e:
            raise ValueError(f"Failed to load watchlist: {e}")
    
    def save(self):
        """Save watchlist to MongoDB"""
        mongo = get_mongo()
        if mongo.db is None:
            raise RuntimeError("Database not initialized")
            
        watchlist_data = {
            'user_id': self.user_id,
            'ticker': self.ticker,
            'company_name': self.company_name,
            'notes': self.notes,
            'added_at': self.added_at
        }
        
        if self.id:
            # Update existing watchlist
            mongo.db.watchlists.update_one(
                {'_id': ObjectId(self.id)},
                {'$set': watchlist_data}
            )
        else:
            # Create new watchlist entry
            result = mongo.db.watchlists.insert_one(watchlist_data)
            self.id = str(result.inserted_id)
        
        return self
    
    def delete(self):
        """Delete watchlist from MongoDB"""
        if self.id:
            mongo = get_mongo()
            if mongo.db is None:
                raise RuntimeError("Database not initialized")
            mongo.db.watchlists.delete_one({'_id': ObjectId(self.id)})
    
    def to_dict(self, current_price=None, price_change=None, price_change_percent=None):
        """Convert watchlist object to dictionary for JSON serialization"""
        result = {
            'id': self.id,
            'ticker': self.ticker,
            'symbol': self.ticker,  # Frontend expects 'symbol'
            'company_name': self.company_name,
            'name': self.company_name,  # Frontend expects 'name'
            'added_at': self.added_at.isoformat() if self.added_at else None,
            'notes': self.notes
        }
        
        # Add current market data if provided
        if current_price is not None:
            result['current_price'] = current_price
        if price_change is not None:
            result['price_change'] = price_change
        if price_change_percent is not None:
            result['price_change_percent'] = price_change_percent
        
        return result
    
    @staticmethod
    def get_user_watchlist(user_id):
        """Get all watchlist items for a specific user"""
        mongo = get_mongo()
        if mongo.db is None:
            return []
            
        watchlists = []
        for watchlist_data in mongo.db.watchlists.find({'user_id': user_id}).sort('added_at', -1):
            watchlist = Watchlist()
            watchlist.id = str(watchlist_data['_id'])
            watchlist.user_id = watchlist_data.get('user_id')
            watchlist.ticker = watchlist_data.get('ticker')
            watchlist.company_name = watchlist_data.get('company_name')
            watchlist.notes = watchlist_data.get('notes')
            watchlist.added_at = watchlist_data.get('added_at', datetime.utcnow())
            watchlists.append(watchlist)
        return watchlists
    
    @staticmethod
    def is_in_watchlist(user_id, ticker):
        """Check if a ticker is already in user's watchlist"""
        mongo = get_mongo()
        if mongo.db is None:
            return False
        return mongo.db.watchlists.find_one({'user_id': user_id, 'ticker': ticker.upper()}) is not None
    
    @staticmethod
    def find_by_id(watchlist_id):
        """Find watchlist by ID"""
        try:
            if isinstance(watchlist_id, str):
                watchlist_id = ObjectId(watchlist_id)
            
            mongo = get_mongo()
            if mongo.db is None:
                return None
                
            watchlist_data = mongo.db.watchlists.find_one({'_id': watchlist_id})
            if watchlist_data:
                watchlist = Watchlist()
                watchlist.id = str(watchlist_data['_id'])
                watchlist.user_id = watchlist_data.get('user_id')
                watchlist.ticker = watchlist_data.get('ticker')
                watchlist.company_name = watchlist_data.get('company_name')
                watchlist.notes = watchlist_data.get('notes')
                watchlist.added_at = watchlist_data.get('added_at', datetime.utcnow())
                return watchlist
            return None
        except Exception:
            return None
    
    @staticmethod
    def find_by_user_and_ticker(user_id, ticker):
        """Find watchlist entry by user and ticker"""
        mongo = get_mongo()
        if mongo.db is None:
            return None
            
        watchlist_data = mongo.db.watchlists.find_one({'user_id': user_id, 'ticker': ticker.upper()})
        if watchlist_data:
            watchlist = Watchlist()
            watchlist.id = str(watchlist_data['_id'])
            watchlist.user_id = watchlist_data.get('user_id')
            watchlist.ticker = watchlist_data.get('ticker')
            watchlist.company_name = watchlist_data.get('company_name')
            watchlist.notes = watchlist_data.get('notes')
            watchlist.added_at = watchlist_data.get('added_at', datetime.utcnow())
            return watchlist
        return None
    
    def __repr__(self):
        return f'<Watchlist {self.ticker} for User {self.user_id}>'
