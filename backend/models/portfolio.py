from datetime import datetime
from decimal import Decimal
from bson import ObjectId

def get_mongo():
    """Get mongo instance to avoid circular imports"""
    from database import mongo
    return mongo

class Portfolio:
    """Portfolio model for tracking user stock holdings using MongoDB"""
    
    def __init__(self, user_id=None, ticker=None, company_name=None, quantity=None, 
                 purchase_price=None, purchase_date=None, portfolio_id=None, **kwargs):
        if portfolio_id:
            # Load existing portfolio
            self._load_portfolio(portfolio_id)
        else:
            # Create new portfolio
            self.id = None
            self.user_id = user_id
            self.ticker = ticker.upper() if ticker else None
            self.company_name = company_name
            self.quantity = float(quantity) if quantity is not None else None
            self.purchase_price = Decimal(str(purchase_price)) if purchase_price is not None else None
            self.purchase_date = purchase_date or datetime.utcnow()
            self.created_at = datetime.utcnow()
            self.updated_at = datetime.utcnow()
            
            # Handle additional fields from kwargs
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    def _load_portfolio(self, portfolio_id):
        """Load portfolio data from MongoDB"""
        try:
            if isinstance(portfolio_id, str):
                portfolio_id = ObjectId(portfolio_id)
            
            mongo = get_mongo()
            if mongo.db is None:
                raise RuntimeError("Database not initialized")
                
            portfolio_data = mongo.db.portfolios.find_one({'_id': portfolio_id})
            if portfolio_data:
                self.id = str(portfolio_data['_id'])
                self.user_id = portfolio_data.get('user_id')
                self.ticker = portfolio_data.get('ticker')
                self.company_name = portfolio_data.get('company_name')
                self.quantity = portfolio_data.get('quantity')
                self.purchase_price = Decimal(str(portfolio_data.get('purchase_price', 0)))
                self.purchase_date = portfolio_data.get('purchase_date', datetime.utcnow())
                self.created_at = portfolio_data.get('created_at', datetime.utcnow())
                self.updated_at = portfolio_data.get('updated_at', datetime.utcnow())
            else:
                raise ValueError(f"Portfolio with id {portfolio_id} not found")
        except Exception as e:
            raise ValueError(f"Failed to load portfolio: {e}")
    
    def save(self):
        """Save portfolio to MongoDB"""
        mongo = get_mongo()
        if mongo.db is None:
            raise RuntimeError("Database not initialized")
            
        portfolio_data = {
            'user_id': self.user_id,
            'ticker': self.ticker,
            'company_name': self.company_name,
            'quantity': self.quantity,
            'purchase_price': float(self.purchase_price) if self.purchase_price is not None else 0,
            'purchase_date': self.purchase_date,
            'created_at': self.created_at,
            'updated_at': datetime.utcnow()
        }
        
        if self.id:
            # Update existing portfolio
            mongo.db.portfolios.update_one(
                {'_id': ObjectId(self.id)},
                {'$set': portfolio_data}
            )
            self.updated_at = datetime.utcnow()
        else:
            # Create new portfolio
            result = mongo.db.portfolios.insert_one(portfolio_data)
            self.id = str(result.inserted_id)
        
        return self
    
    def delete(self):
        """Delete portfolio from MongoDB"""
        if self.id:
            mongo = get_mongo()
            if mongo.db is None:
                raise RuntimeError("Database not initialized")
            mongo.db.portfolios.delete_one({'_id': ObjectId(self.id)})
    
    def calculate_current_value(self, current_price):
        """Calculate current value of the holding"""
        if self.quantity is None or current_price is None:
            return 0.0
        return float(self.quantity) * float(current_price)
    
    def calculate_gain_loss(self, current_price):
        """Calculate gain/loss for this holding"""
        if self.quantity is None or self.purchase_price is None or current_price is None:
            return 0.0
        current_value = self.calculate_current_value(current_price)
        purchase_value = float(self.quantity) * float(self.purchase_price)
        return current_value - purchase_value
    
    def calculate_gain_loss_percentage(self, current_price):
        """Calculate gain/loss percentage for this holding"""
        if self.quantity is None or self.purchase_price is None or current_price is None:
            return 0.0
        purchase_value = float(self.quantity) * float(self.purchase_price)
        if purchase_value == 0:
            return 0
        gain_loss = self.calculate_gain_loss(current_price)
        return (gain_loss / purchase_value) * 100
    
    def to_dict(self, current_price=None):
        """Convert portfolio object to dictionary for JSON serialization"""
        result = {
            'id': self.id,
            'ticker': self.ticker,
            'symbol': self.ticker,  # Frontend expects 'symbol'
            'company_name': self.company_name,
            'quantity': self.quantity,
            'purchase_price': float(self.purchase_price) if self.purchase_price is not None else 0,
            'purchase_date': self.purchase_date.isoformat() if self.purchase_date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
        
        if current_price is not None:
            result.update({
                'current_price': current_price,
                'current_value': self.calculate_current_value(current_price),
                'gain_loss': self.calculate_gain_loss(current_price),
                'gain_loss_percentage': self.calculate_gain_loss_percentage(current_price)
            })
        
        return result
    
    @staticmethod
    def find_by_id(portfolio_id):
        """Find portfolio by ID"""
        try:
            if isinstance(portfolio_id, str):
                portfolio_id = ObjectId(portfolio_id)
            
            mongo = get_mongo()
            if mongo.db is None:
                return None
                
            portfolio_data = mongo.db.portfolios.find_one({'_id': portfolio_id})
            if portfolio_data:
                portfolio = Portfolio()
                portfolio.id = str(portfolio_data['_id'])
                portfolio.user_id = portfolio_data.get('user_id')
                portfolio.ticker = portfolio_data.get('ticker')
                portfolio.company_name = portfolio_data.get('company_name')
                portfolio.quantity = portfolio_data.get('quantity')
                portfolio.purchase_price = Decimal(str(portfolio_data.get('purchase_price', 0)))
                portfolio.purchase_date = portfolio_data.get('purchase_date', datetime.utcnow())
                portfolio.created_at = portfolio_data.get('created_at', datetime.utcnow())
                portfolio.updated_at = portfolio_data.get('updated_at', datetime.utcnow())
                return portfolio
            return None
        except Exception:
            return None
    
    @staticmethod
    def find_by_user_id(user_id):
        """Find all portfolios for a user"""
        mongo = get_mongo()
        if mongo.db is None:
            return []
            
        portfolios = []
        for portfolio_data in mongo.db.portfolios.find({'user_id': user_id}):
            portfolio = Portfolio()
            portfolio.id = str(portfolio_data['_id'])
            portfolio.user_id = portfolio_data.get('user_id')
            portfolio.ticker = portfolio_data.get('ticker')
            portfolio.company_name = portfolio_data.get('company_name')
            portfolio.quantity = portfolio_data.get('quantity')
            portfolio.purchase_price = Decimal(str(portfolio_data.get('purchase_price', 0)))
            portfolio.purchase_date = portfolio_data.get('purchase_date', datetime.utcnow())
            portfolio.created_at = portfolio_data.get('created_at', datetime.utcnow())
            portfolio.updated_at = portfolio_data.get('updated_at', datetime.utcnow())
            portfolios.append(portfolio)
        return portfolios
    
    def __repr__(self):
        return f'<Portfolio {self.ticker}: {self.quantity} shares at ${self.purchase_price}>'
