from flask import Blueprint, request, jsonify
import logging
from services.stock_service import stock_service
from services.market_data_service import market_data_service

logger = logging.getLogger(__name__)

# Create blueprint for stock-related routes
stock_bp = Blueprint('stock', __name__)

@stock_bp.route('/health', methods=['GET'])
def api_health():
    """
    API health check endpoint
    """
    return jsonify({
        'status': 'healthy',
        'message': 'Stock Analyzer API is running',
        'version': '1.0.0'
    })

@stock_bp.route('/search', methods=['GET'])
def search_stocks():
    """
    Search for stocks by ticker symbol or company name
    Query parameters:
    - q: search query
    - limit: maximum number of results (default: 10)
    """
    try:
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', 10, type=int)
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        if len(query) < 1:
            return jsonify({'error': 'Search query must be at least 1 character long'}), 400
        
        if limit > 50:
            limit = 50  # Limit maximum results
        
        results = stock_service.search_stocks(query, limit)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error in stock search: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@stock_bp.route('/stock/<symbol>', methods=['GET'])
def get_stock_details(symbol):
    """
    Get detailed information for a specific stock
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        # URL decode the symbol to handle special characters like ^ in indices
        from urllib.parse import unquote
        decoded_symbol = unquote(symbol)
        
        stock_info = stock_service.get_stock_info(decoded_symbol)
        
        if not stock_info:
            return jsonify({'error': f'Stock {decoded_symbol} not found'}), 404
        
        return jsonify({
            'symbol': decoded_symbol.upper(),
            'data': stock_info
        })
        
    except Exception as e:
        logger.error(f"Error fetching stock details for {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@stock_bp.route('/stock/<symbol>/history', methods=['GET'])
def get_stock_history(symbol):
    """
    Get historical price data for a stock
    Query parameters:
    - period: time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
    - interval: data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        # URL decode the symbol to handle special characters like ^ in indices
        from urllib.parse import unquote
        decoded_symbol = unquote(symbol)
        
        period = request.args.get('period', '1y')
        interval = request.args.get('interval', '1d')
        
        # Validate period
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if period not in valid_periods:
            return jsonify({'error': f'Invalid period. Must be one of: {", ".join(valid_periods)}'}), 400
        
        # Validate interval
        valid_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo']
        if interval not in valid_intervals:
            return jsonify({'error': f'Invalid interval. Must be one of: {", ".join(valid_intervals)}'}), 400
        
        history_data = stock_service.get_stock_history(decoded_symbol, period, interval)
        
        if not history_data:
            return jsonify({'error': f'No historical data found for {decoded_symbol}'}), 404
        
        return jsonify(history_data)
        
    except Exception as e:
        logger.error(f"Error fetching stock history for {decoded_symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@stock_bp.route('/stocks/compare', methods=['POST'])
def compare_stocks():
    """
    Compare multiple stocks
    Request body should contain:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "period": "1y",
        "interval": "1d"
    }
    """
    try:
        data = request.get_json()
        logger.info(f"Stock comparison request: {data}")
        
        if not data or 'symbols' not in data:
            logger.warning("Missing symbols in comparison request")
            return jsonify({'error': 'Symbols array is required'}), 400
        
        symbols = data['symbols']
        if not isinstance(symbols, list) or len(symbols) == 0:
            logger.warning(f"Invalid symbols format: {symbols}")
            return jsonify({'error': 'At least one symbol is required'}), 400
        
        if len(symbols) > 4:
            logger.warning(f"Too many symbols requested: {len(symbols)}")
            return jsonify({'error': 'Maximum 4 stocks can be compared at once'}), 400
        
        period = data.get('period', '1y')
        interval = data.get('interval', '1d')
        
        logger.info(f"Comparing stocks: {symbols} with period={period}, interval={interval}")
        
        # Get basic info for all stocks
        logger.info("Fetching stocks info...")
        stocks_info = stock_service.get_multiple_stocks_info(symbols)
        logger.info(f"Stocks info retrieved for {len(stocks_info)} symbols")
        
        # Get historical data for all stocks
        logger.info("Fetching historical data...")
        stocks_history = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching history for {symbol}...")
                history = stock_service.get_stock_history(symbol, period, interval)
                if history:
                    stocks_history[symbol.upper()] = history
                    logger.info(f"Successfully retrieved {len(history.get('data', []))} data points for {symbol}")
                else:
                    logger.warning(f"No history data returned for {symbol}")
                    failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Failed to fetch history for {symbol}: {e}")
                failed_symbols.append(symbol)
        
        logger.info(f"Historical data retrieved for {len(stocks_history)} out of {len(symbols)} symbols")
        
        if failed_symbols:
            logger.warning(f"Failed to get data for symbols: {failed_symbols}")
        
        result = {
            'symbols': [s.upper() for s in symbols],
            'period': period,
            'interval': interval,
            'stocks_info': stocks_info,
            'stocks_history': stocks_history,
            'failed_symbols': failed_symbols,
            'success_count': len(stocks_history)
        }
        
        logger.info(f"Comparison completed successfully. Returning data for {len(stocks_history)} stocks")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error comparing stocks: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

@stock_bp.route('/trending', methods=['GET'])
def get_trending_stocks():
    """
    Get trending stocks using market data service with real data
    Query parameters:
    - count: number of stocks to return (default: 10, max: 20)
    """
    try:
        count = request.args.get('count', 10, type=int)
        
        if count > 20:
            count = 20
        elif count < 1:
            count = 10
        
        # Use the market data service for trending stocks
        trending_stocks = market_data_service.fetch_trending_stocks(count)
        
        # If market data service fails, fall back to stock service
        if not trending_stocks:
            trending_stocks = stock_service.get_trending_stocks(count)
        
        import pandas as pd
        
        return jsonify({
            'trending_stocks': trending_stocks,
            'count': len(trending_stocks),
            'source': 'Stock Analysis & yFinance',
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.exception(f"Error fetching trending stocks: {e}")
        import pandas as pd
        return jsonify({
            'trending_stocks': [],
            'count': 0,
            'error': str(e),
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@stock_bp.route('/market/summary', methods=['GET'])
def get_market_summary():
    """
    Get market summary with major indices using yfinance directly
    """
    try:
        # Major market indices - expand to include 4 main ones
        indices = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow Jones, NASDAQ, Russell 2000
        indices_names = ['S&P 500', 'Dow Jones', 'NASDAQ', 'Russell 2000']
        
        import yfinance as yf
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def fetch_index_data(symbol, name):
            try:
                ticker = yf.Ticker(symbol)
                today_data = ticker.history(period='1d', interval='1h')
                
                if not today_data.empty:
                    current_price = float(today_data['Close'].iloc[-1])
                    # Use simpler calculation for price change
                    price_change = float(today_data['Close'].iloc[-1] - today_data['Open'].iloc[0])
                    price_change_percent = (price_change / today_data['Open'].iloc[0]) * 100
                    
                    return {
                        'name': name,
                        'symbol': symbol,
                        'current_price': current_price,
                        'price_change': price_change,
                        'price_change_percent': price_change_percent
                    }
            except Exception as e:
                logger.warning(f"Could not fetch data for index {symbol}: {e}")
                return None
        
        # Use ThreadPoolExecutor for parallel requests
        market_data = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_to_index = {
                executor.submit(fetch_index_data, symbol, name): (symbol, name)
                for symbol, name in zip(indices, indices_names)
            }
            
            for future in as_completed(future_to_index):
                symbol, name = future_to_index[future]
                try:
                    data = future.result()
                    if data:
                        market_data.append(data)
                    else:
                        # Add placeholder data if fetching fails
                        market_data.append({
                            'name': name,
                            'symbol': symbol,
                            'current_price': 0,
                            'price_change': 0,
                            'price_change_percent': 0
                        })
                except Exception as e:
                    logger.warning(f"Error processing {symbol}: {e}")
                    # Add placeholder data
                    market_data.append({
                        'name': name,
                        'symbol': symbol,
                        'current_price': 0,
                        'price_change': 0,
                        'price_change_percent': 0
                    })
        
        return jsonify({
            'market_indices': market_data,
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.exception(f"Error fetching market summary: {e}")
        return jsonify({
            'market_indices': [],
            'error': str(e),
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@stock_bp.route('/market/gainers', methods=['GET'])
def get_top_gainers():
    """
    Get top gaining stocks using market data service with real data
    Query parameters:
    - count: number of stocks to return (default: 10, max: 20)
    """
    try:
        count = request.args.get('count', 10, type=int)
        
        if count > 20:
            count = 20
        elif count < 1:
            count = 10
        
        # Use the market data service for real gainers data
        gainers = market_data_service.fetch_top_gainers(count)
        
        import pandas as pd
        
        return jsonify({
            'gainers': gainers,
            'count': len(gainers),
            'source': 'Stock Analysis & yFinance',
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.exception(f"Error fetching top gainers: {e}")
        import pandas as pd
        return jsonify({
            'gainers': [],
            'count': 0,
            'error': str(e),
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@stock_bp.route('/market/losers', methods=['GET'])
def get_top_losers():
    """
    Get top losing stocks using market data service with real data
    Query parameters:
    - count: number of stocks to return (default: 10, max: 20)
    """
    try:
        count = request.args.get('count', 10, type=int)
        
        if count > 20:
            count = 20
        elif count < 1:
            count = 10
        
        # Use the market data service for real losers data
        losers = market_data_service.fetch_top_losers(count)
        
        import pandas as pd
        
        return jsonify({
            'losers': losers,
            'count': len(losers),
            'source': 'Stock Analysis & yFinance',
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.exception(f"Error fetching top losers: {e}")
        import pandas as pd
        return jsonify({
            'losers': [],
            'count': 0,
            'error': str(e),
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }), 500

@stock_bp.route('/market/movers', methods=['GET'])
def get_market_movers():
    """
    Get both top gainers and losers in one call
    Query parameters:
    - count: number of stocks to return for each category (default: 10, max: 20)
    """
    try:
        count = request.args.get('count', 10, type=int)
        
        if count > 20:
            count = 20
        elif count < 1:
            count = 10
        
        # Fetch both gainers and losers
        market_movers = market_data_service.fetch_market_movers(count)
        
        return jsonify(market_movers)
        
    except Exception as e:
        logger.error(f"Error fetching market movers: {e}")
        return jsonify({'error': 'Internal server error'}), 500
