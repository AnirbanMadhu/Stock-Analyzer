from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
import logging
from models import Portfolio, Watchlist
from database import mongo
from services.stock_service import stock_service

logger = logging.getLogger(__name__)

# Create blueprint for portfolio management routes
portfolio_bp = Blueprint('portfolio', __name__)

# Root endpoint for portfolio (redirects to holdings)
@portfolio_bp.route('/', methods=['GET'])
@login_required
def get_portfolio_root():
    """
    Get user's portfolio (root endpoint)
    """
    return get_portfolio()

@portfolio_bp.route('/', methods=['POST'])
@login_required
def add_holding_root():
    """
    Add holding to portfolio (root endpoint)
    """
    return add_holding()

@portfolio_bp.route('/holdings', methods=['GET'])
@login_required
def get_portfolio():
    """
    Get user's portfolio holdings
    """
    try:
        holdings = Portfolio.find_by_user_id(current_user.id)
        
        # Get current prices for all holdings
        portfolio_data = []
        total_value = 0
        total_gain_loss = 0
        
        for holding in holdings:
            try:
                stock_info = stock_service.get_stock_info(holding.ticker)
                current_price = stock_info['current_price'] if stock_info else 0
                
                holding_dict = holding.to_dict(current_price)
                portfolio_data.append(holding_dict)
                
                if current_price > 0:
                    total_value += holding_dict['current_value']
                    total_gain_loss += holding_dict['gain_loss']
                    
            except Exception as e:
                logger.warning(f"Could not fetch current price for {holding.ticker}: {e}")
                # Include holding without current price data
                portfolio_data.append(holding.to_dict())
        
        return jsonify({
            'holdings': portfolio_data,
            'summary': {
                'total_holdings': len(portfolio_data),
                'total_value': total_value,
                'total_gain_loss': total_gain_loss,
                'total_gain_loss_percent': (total_gain_loss / (total_value - total_gain_loss)) * 100 if (total_value - total_gain_loss) > 0 else 0
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching portfolio: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@portfolio_bp.route('/holdings', methods=['POST'])
@login_required
def add_holding():
    """
    Add a new holding to portfolio
    Expected JSON payload:
    {
        "ticker": "AAPL",
        "quantity": 10,
        "purchase_price": 150.50,
        "purchase_date": "2023-01-01" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['ticker', 'quantity', 'purchase_price']
        
        # Handle both 'ticker' and 'symbol' field names
        if 'symbol' in data and 'ticker' not in data:
            data['ticker'] = data['symbol']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'{field} is required'}), 400
        
        ticker = data['ticker'].upper().strip()
        quantity = float(data['quantity'])
        purchase_price = float(data['purchase_price'])
        purchase_date = data.get('purchase_date')
        
        # Validate values
        if quantity <= 0:
            return jsonify({'error': 'Quantity must be greater than 0'}), 400
        
        if purchase_price <= 0:
            return jsonify({'error': 'Purchase price must be greater than 0'}), 400
        
        # Get company name from stock service
        stock_info = stock_service.get_stock_info(ticker)
        company_name = stock_info['name'] if stock_info else ticker
        
        # Create new holding
        holding = Portfolio(
            user_id=current_user.id,
            ticker=ticker,
            company_name=company_name,
            quantity=quantity,
            purchase_price=purchase_price
        )
        
        if purchase_date:
            from datetime import datetime
            try:
                holding.purchase_date = datetime.strptime(purchase_date, '%Y-%m-%d')
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        holding.save()
        
        logger.info(f"Added holding for user {current_user.username}: {ticker} x {quantity}")
        
        return jsonify({
            'message': 'Holding added successfully',
            'holding': holding.to_dict()
        }), 201
        
    except ValueError as e:
        return jsonify({'error': 'Invalid numeric value provided'}), 400
    except Exception as e:
        logger.error(f"Error adding holding: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@portfolio_bp.route('/holdings/<int:holding_id>', methods=['PUT'])
@login_required
def update_holding(holding_id):
    """
    Update a portfolio holding
    """
    try:
        # Find holding by converting the int ID to string for MongoDB
        holdings = Portfolio.find_by_user_id(current_user.id)
        holding = None
        for h in holdings:
            if h.id == str(holding_id):
                holding = h
                break
        
        if not holding:
            return jsonify({'error': 'Holding not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update fields if provided
        if 'quantity' in data:
            quantity = float(data['quantity'])
            if quantity <= 0:
                return jsonify({'error': 'Quantity must be greater than 0'}), 400
            holding.quantity = quantity
        
        if 'purchase_price' in data:
            purchase_price = float(data['purchase_price'])
            if purchase_price <= 0:
                return jsonify({'error': 'Purchase price must be greater than 0'}), 400
            holding.purchase_price = purchase_price
        
        if 'purchase_date' in data:
            from datetime import datetime
            try:
                holding.purchase_date = datetime.strptime(data['purchase_date'], '%Y-%m-%d')
            except ValueError:
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
        
        holding.save()
        
        logger.info(f"Updated holding {holding_id} for user {current_user.username}")
        
        return jsonify({
            'message': 'Holding updated successfully',
            'holding': holding.to_dict()
        }), 200
        
    except ValueError as e:
        return jsonify({'error': 'Invalid numeric value provided'}), 400
    except Exception as e:
        logger.error(f"Error updating holding: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@portfolio_bp.route('/holdings/<int:holding_id>', methods=['DELETE'])
@login_required
def delete_holding(holding_id):
    """
    Delete a portfolio holding
    """
    try:
        # Find holding by converting the int ID to string for MongoDB
        holdings = Portfolio.find_by_user_id(current_user.id)
        holding = None
        for h in holdings:
            if h.id == str(holding_id):
                holding = h
                break
        
        if not holding:
            return jsonify({'error': 'Holding not found'}), 404
        
        ticker = holding.ticker
        holding.delete()
        
        logger.info(f"Deleted holding {holding_id} ({ticker}) for user {current_user.username}")
        
        return jsonify({
            'message': 'Holding deleted successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error deleting holding: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@portfolio_bp.route('/watchlist', methods=['GET'])
@login_required
def get_watchlist():
    """
    Get user's watchlist
    """
    try:
        watchlist_items = Watchlist.get_user_watchlist(current_user.id)
        
        # Get current prices for watchlist items
        watchlist_data = []
        for item in watchlist_items:
            try:
                stock_info = stock_service.get_stock_info(item.ticker)
                if stock_info:
                    item_dict = item.to_dict(
                        current_price=stock_info['current_price'],
                        price_change=stock_info['price_change'],
                        price_change_percent=stock_info['price_change_percent']
                    )
                else:
                    item_dict = item.to_dict()
                
                watchlist_data.append(item_dict)
                
            except Exception as e:
                logger.warning(f"Could not fetch current price for {item.ticker}: {e}")
                watchlist_data.append(item.to_dict())
        
        return jsonify({
            'watchlist': watchlist_data,
            'count': len(watchlist_data)
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching watchlist: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@portfolio_bp.route('/watchlist', methods=['POST'])
@login_required
def add_to_watchlist():
    """
    Add a stock to watchlist
    Expected JSON payload:
    {
        "ticker": "AAPL",
        "notes": "Optional notes about the stock"
    }
    """
    try:
        data = request.get_json()
        
        if not data or ('ticker' not in data and 'symbol' not in data):
            return jsonify({'error': 'Ticker symbol is required'}), 400
        
        # Handle both 'ticker' and 'symbol' field names
        ticker = data.get('ticker') or data.get('symbol')
        ticker = ticker.upper().strip()
        notes = data.get('notes', '').strip()
        
        # Check if already in watchlist
        existing = Watchlist.is_in_watchlist(current_user.id, ticker)
        if existing:
            return jsonify({'error': 'Stock is already in your watchlist'}), 400
        
        # Get company name from stock service
        stock_info = stock_service.get_stock_info(ticker)
        if not stock_info:
            return jsonify({'error': 'Invalid stock symbol'}), 400
        
        company_name = stock_info['name']
        
        # Create watchlist item
        watchlist_item = Watchlist(
            user_id=current_user.id,
            ticker=ticker,
            company_name=company_name,
            notes=notes if notes else None
        )
        
        watchlist_item.save()
        
        logger.info(f"Added {ticker} to watchlist for user {current_user.username}")
        
        return jsonify({
            'message': 'Stock added to watchlist successfully',
            'watchlist_item': watchlist_item.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@portfolio_bp.route('/watchlist/<int:item_id>', methods=['DELETE'])
@login_required
def remove_from_watchlist(item_id):
    """
    Remove a stock from watchlist
    """
    try:
        # Find watchlist item by converting the int ID to string for MongoDB
        watchlist_items = Watchlist.get_user_watchlist(current_user.id)
        watchlist_item = None
        for item in watchlist_items:
            if item.id == str(item_id):
                watchlist_item = item
                break
        
        if not watchlist_item:
            return jsonify({'error': 'Watchlist item not found'}), 404
        
        ticker = watchlist_item.ticker
        watchlist_item.delete()
        
        logger.info(f"Removed {ticker} from watchlist for user {current_user.username}")
        
        return jsonify({
            'message': 'Stock removed from watchlist successfully'
        }), 200
        
    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}")
        return jsonify({'error': 'Internal server error'}), 500
