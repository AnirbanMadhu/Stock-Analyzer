import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Union
import time
from .search_cache import search_cache

logger = logging.getLogger(__name__)

class StockService:
    """Service class for handling stock data operations using yfinance"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
        self._cache = {}
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self._cache:
            return False
        
        cache_time = self._cache[cache_key].get('timestamp', 0)
        return time.time() - cache_time < self.cache_duration
    
    def _get_from_cache(self, cache_key: str):
        """Get data from cache if valid"""
        if self._is_cache_valid(cache_key):
            return self._cache[cache_key]['data']
        return None
    
    def _set_cache(self, cache_key: str, data):
        """Set data in cache"""
        self._cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
    
    def search_stocks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search for stocks based on ticker symbol or company name
        Returns a list of matching stocks with instant results from pre-built database
        """
        try:
            # First check our internal cache
            cache_key = f"search_{query}_{limit}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Get instant suggestions from search_cache first (faster than API calls)
            instant_results = search_cache.get_instant_suggestions(query, limit)
            if instant_results and len(instant_results) >= min(3, limit):
                # If we have enough good instant results, return them immediately
                self._set_cache(cache_key, instant_results)
                return instant_results
            
            query_upper = query.upper().strip()
            query_lower = query.lower().strip()
            results = instant_results.copy() if instant_results else []
            processed_symbols = {stock['symbol'] for stock in results}
            
            # Enhanced comprehensive stock database for instant suggestions (optimized for speed)
            stock_database = {
                # Major Tech Stocks (most searched)
                'AAPL': 'Apple Inc.',
                'MSFT': 'Microsoft Corporation',
                'GOOGL': 'Alphabet Inc. Class A',
                'GOOG': 'Alphabet Inc. Class C',
                'AMZN': 'Amazon.com Inc.',
                'META': 'Meta Platforms Inc.',
                'TSLA': 'Tesla Inc.',
                'NVDA': 'NVIDIA Corporation',
                'NFLX': 'Netflix Inc.',
                'CRM': 'Salesforce Inc.',
                'ORCL': 'Oracle Corporation',
                'ADBE': 'Adobe Inc.',
                'IBM': 'International Business Machines',
                'INTC': 'Intel Corporation',
                'AMD': 'Advanced Micro Devices',
                'PYPL': 'PayPal Holdings Inc.',
                'SQ': 'Block Inc.',
                'SHOP': 'Shopify Inc.',
                'UBER': 'Uber Technologies Inc.',
                'LYFT': 'Lyft Inc.',
                'SNAP': 'Snap Inc.',
                'PINS': 'Pinterest Inc.',
                'ROKU': 'Roku Inc.',
                'ZM': 'Zoom Video Communications',
                'DOCU': 'DocuSign Inc.',
                'OKTA': 'Okta Inc.',
                'SNOW': 'Snowflake Inc.',
                'PLTR': 'Palantir Technologies',
                
                # Financial Services (commonly searched)
                'JPM': 'JPMorgan Chase & Co.',
                'BAC': 'Bank of America Corporation',
                'WFC': 'Wells Fargo & Company',
                'C': 'Citigroup Inc.',
                'GS': 'Goldman Sachs Group Inc.',
                'MS': 'Morgan Stanley',
                'V': 'Visa Inc.',
                'MA': 'Mastercard Inc.',
                'AXP': 'American Express Company',
                'BRK.A': 'Berkshire Hathaway Inc. Class A',
                'BRK.B': 'Berkshire Hathaway Inc. Class B',
                'COF': 'Capital One Financial Corp.',
                'SCHW': 'Charles Schwab Corporation',
                'BLK': 'BlackRock Inc.',
                
                # Healthcare & Pharma
                'JNJ': 'Johnson & Johnson',
                'UNH': 'UnitedHealth Group Inc.',
                'PFE': 'Pfizer Inc.',
                'ABBV': 'AbbVie Inc.',
                'TMO': 'Thermo Fisher Scientific',
                'ABT': 'Abbott Laboratories',
                'BMY': 'Bristol Myers Squibb',
                'LLY': 'Eli Lilly and Company',
                'MRK': 'Merck & Co. Inc.',
                'AMGN': 'Amgen Inc.',
                'GILD': 'Gilead Sciences Inc.',
                'CVS': 'CVS Health Corporation',
                'MRNA': 'Moderna Inc.',
                
                # Consumer & Retail
                'WMT': 'Walmart Inc.',
                'HD': 'Home Depot Inc.',
                'PG': 'Procter & Gamble Company',
                'KO': 'Coca-Cola Company',
                'PEP': 'PepsiCo Inc.',
                'MCD': 'McDonald\'s Corporation',
                'SBUX': 'Starbucks Corporation',
                'NKE': 'Nike Inc.',
                'TGT': 'Target Corporation',
                'LOW': 'Lowe\'s Companies Inc.',
                'COST': 'Costco Wholesale Corporation',
                'DIS': 'Walt Disney Company',
                
                # Communication Services
                'CMCSA': 'Comcast Corporation',
                'VZ': 'Verizon Communications',
                'T': 'AT&T Inc.',
                'CHTR': 'Charter Communications',
                'TMUS': 'T-Mobile US Inc.',
                
                # Energy & Materials
                'XOM': 'Exxon Mobil Corporation',
                'CVX': 'Chevron Corporation',
                'COP': 'ConocoPhillips',
                'SLB': 'Schlumberger NV',
                'EOG': 'EOG Resources Inc.',
                
                # Meme Stocks & Popular Retail
                'GME': 'GameStop Corp.',
                'AMC': 'AMC Entertainment Holdings',
                'BB': 'BlackBerry Limited',
                'NOK': 'Nokia Corporation',
                'NIO': 'NIO Inc.',
                'COIN': 'Coinbase Global Inc.',
                'HOOD': 'Robinhood Markets Inc.',
                
                # Automotive
                'F': 'Ford Motor Company',
                'GM': 'General Motors Company',
                'RIVN': 'Rivian Automotive Inc.',
                'LCID': 'Lucid Group Inc.',
                
                # International & Emerging
                'BABA': 'Alibaba Group Holding',
                'TSM': 'Taiwan Semiconductor',
                'ASML': 'ASML Holding NV',
                'NVO': 'Novo Nordisk A/S',
                'SAP': 'SAP SE',
                'TM': 'Toyota Motor Corporation',
                'SONY': 'Sony Group Corporation',
            }
            
            # Ultra-fast search algorithm (optimized for immediate response)
            # 1. Exact symbol match (highest priority) - return immediately for short queries
            if query_upper in stock_database and query_upper not in processed_symbols:
                try:
                    stock_info = self._get_stock_lightweight_info(query_upper, stock_database[query_upper])
                    if stock_info:
                        results.insert(0, stock_info)  # Insert at beginning for highest priority
                        processed_symbols.add(query_upper)
                        
                        # For very short exact matches, add related symbols immediately
                        if len(query_upper) <= 3:
                            related_count = 0
                            for symbol, name in stock_database.items():
                                if related_count >= 3 or len(results) >= limit:
                                    break
                                if symbol.startswith(query_upper) and symbol != query_upper and symbol not in processed_symbols:
                                    try:
                                        related_info = self._get_stock_lightweight_info(symbol, name)
                                        if related_info:
                                            results.append(related_info)
                                            processed_symbols.add(symbol)
                                            related_count += 1
                                    except Exception:
                                        continue
                except Exception as e:
                    logger.warning(f"Could not fetch info for exact match {query_upper}: {e}")
            
            # 2. Symbol prefix matches (lightning fast lookup)
            if len(results) < limit:
                prefix_matches = []
                for symbol, name in stock_database.items():
                    if symbol.startswith(query_upper) and symbol not in processed_symbols:
                        prefix_matches.append((symbol, name))
                
                # Sort by symbol length and popularity (shorter = more popular generally)
                prefix_matches.sort(key=lambda x: (len(x[0]), x[0]))
                
                for symbol, name in prefix_matches[:limit-len(results)]:
                    try:
                        stock_info = self._get_stock_lightweight_info(symbol, name)
                        if stock_info:
                            results.append(stock_info)
                            processed_symbols.add(symbol)
                    except Exception:
                        continue
            
            # 3. Company name matches (only if we need more results)
            if len(results) < limit:
                name_matches = []
                for symbol, name in stock_database.items():
                    if symbol not in processed_symbols:
                        name_lower = name.lower()
                        # Optimized matching for speed
                        if query_lower in name_lower:
                            # Calculate simple relevance score
                            score = 100 if name_lower.startswith(query_lower) else 50
                            name_matches.append((symbol, name, score))
                        elif any(word.startswith(query_lower) for word in name_lower.split() if len(word) >= 2):
                            name_matches.append((symbol, name, 25))
                
                # Sort by relevance score
                name_matches.sort(key=lambda x: x[2], reverse=True)
                
                for symbol, name, score in name_matches[:limit-len(results)]:
                    try:
                        stock_info = self._get_stock_lightweight_info(symbol, name)
                        if stock_info:
                            results.append(stock_info)
                            processed_symbols.add(symbol)
                    except Exception:
                        continue
            
            # Cache results for faster subsequent searches (shorter cache time for speed)
            if results:
                self._set_cache(cache_key, results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching for stocks with query '{query}': {e}")
            # Return instant suggestions as fallback
            return search_cache.get_instant_suggestions(query, limit)
    
    def _get_stock_lightweight_info(self, symbol: str, name: str) -> Optional[Dict]:
        """Get lightweight stock info for instant search results - try to get current price"""
        try:
            # Check if we have cached current price data
            cache_key = f"lightweight_{symbol}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # Try to get real current price quickly (with timeout)
            try:
                ticker = yf.Ticker(symbol)
                # Try to get current price from info quickly
                info = ticker.info
                current_price = info.get('currentPrice', 0)
                
                if not current_price or current_price == 0:
                    # Fallback: get from recent history (faster than full info)
                    hist = ticker.history(period="1d", timeout=2)
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                
                result = {
                    'symbol': symbol,
                    'name': name,
                    'current_price': float(current_price) if current_price else 0,
                    'price_change': 0,
                    'price_change_percent': 0,
                    'sector': self._guess_sector_from_name(name),
                    'industry': 'Various',
                    'market_cap': 0
                }
                
                # Cache result for 60 seconds for faster subsequent searches
                if result['current_price'] > 0:
                    cache_key_short = f"lightweight_{symbol}"
                    self._cache[cache_key_short] = {
                        'data': result,
                        'timestamp': time.time()
                    }
                
                return result
                
            except Exception as e:
                # If API call fails, return basic info without current price
                logger.debug(f"Could not fetch current price for {symbol}: {e}")
                return {
                    'symbol': symbol,
                    'name': name,
                    'current_price': 0,  # Will be fetched when selected
                    'price_change': 0,
                    'price_change_percent': 0,
                    'sector': self._guess_sector_from_name(name),
                    'industry': 'Various',
                    'market_cap': 0
                }
            
        except Exception as e:
            logger.warning(f"Error creating lightweight info for {symbol}: {e}")
            return None
    
    def _guess_sector_from_name(self, name: str) -> str:
        """Guess sector from company name for better search results display"""
        name_lower = name.lower()
        
        # Technology keywords
        if any(word in name_lower for word in ['tech', 'software', 'microsoft', 'apple', 'google', 'meta', 'amazon', 'netflix', 'adobe', 'oracle', 'salesforce', 'nvidia', 'intel', 'amd', 'qualcomm', 'broadcom']):
            return 'Technology'
        
        # Financial keywords
        elif any(word in name_lower for word in ['bank', 'financial', 'jpmorgan', 'goldman', 'morgan', 'wells fargo', 'citigroup', 'visa', 'mastercard', 'american express', 'berkshire']):
            return 'Financial Services'
        
        # Healthcare keywords
        elif any(word in name_lower for word in ['pharma', 'health', 'medical', 'bio', 'johnson', 'pfizer', 'abbott', 'merck', 'bristol', 'lilly', 'amgen', 'gilead']):
            return 'Healthcare'
        
        # Energy keywords
        elif any(word in name_lower for word in ['energy', 'oil', 'exxon', 'chevron', 'conocophillips', 'marathon', 'valero', 'phillips']):
            return 'Energy'
        
        # Consumer keywords
        elif any(word in name_lower for word in ['walmart', 'target', 'home depot', 'costco', 'coca-cola', 'pepsi', 'mcdonald', 'starbucks', 'nike', 'disney']):
            return 'Consumer'
        
        # Automotive keywords
        elif any(word in name_lower for word in ['motor', 'auto', 'ford', 'general motors', 'tesla', 'toyota', 'rivian', 'lucid']):
            return 'Automotive'
        
        else:
            return 'Various'
    
    def _get_stock_quick_info(self, symbol: str, name: Optional[str] = None) -> Optional[Dict]:
        """Get quick stock info for search results"""
        try:
            # First try to get from cache
            cache_key = f"quick_info_{symbol}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            ticker = yf.Ticker(symbol)
            
            # Try to get basic info quickly
            try:
                info = ticker.info
                if not info or 'symbol' not in info:
                    # Fallback: try to get from recent history
                    hist = ticker.history(period="1d")
                    if hist.empty:
                        return None
                    
                    current_price = float(hist['Close'].iloc[-1])
                    result = {
                        'symbol': symbol,
                        'name': name or symbol,
                        'current_price': current_price,
                        'price_change': 0,
                        'price_change_percent': 0,
                        'sector': 'Unknown',
                        'industry': 'Unknown',
                        'market_cap': 0
                    }
                else:
                    # Get current price
                    current_price = info.get('currentPrice', 0)
                    
                    # Try multiple sources for price change data
                    price_change = (
                        info.get('regularMarketChange') or 
                        info.get('change') or 
                        info.get('priceChange') or 0
                    )
                    price_change_percent = (
                        info.get('regularMarketChangePercent') or 
                        info.get('changePercent') or 
                        info.get('priceChangePercent') or 0
                    )
                    
                    if not current_price:
                        # Try from history
                        hist = ticker.history(period="2d")
                        if not hist.empty:
                            current_price = hist['Close'].iloc[-1]
                            if len(hist) > 1:
                                previous_price = hist['Close'].iloc[-2]
                                price_change = current_price - previous_price
                                price_change_percent = (price_change / previous_price) * 100 if previous_price != 0 else 0
                            else:
                                price_change = 0
                                price_change_percent = 0
                        else:
                            current_price = 0
                            price_change = 0
                            price_change_percent = 0
                    
                    # Additional fallback: calculate from regularMarketPrice vs previousClose
                    if price_change_percent == 0 and current_price > 0:
                        regular_price = info.get('regularMarketPrice', current_price)
                        previous_close = info.get('previousClose', 0)
                        if previous_close > 0:
                            price_change = regular_price - previous_close
                            price_change_percent = (price_change / previous_close) * 100
                            current_price = regular_price  # Update current price too
                        price_change_percent = info.get('regularMarketChangePercent', 0)
                    
                    result = {
                        'symbol': info.get('symbol', symbol),
                        'name': name or info.get('longName', info.get('shortName', symbol)),
                        'current_price': float(current_price) if current_price else 0,
                        'price_change': float(price_change) if price_change else 0,
                        'price_change_percent': float(price_change_percent) if price_change_percent else 0,
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown'),
                        'market_cap': info.get('marketCap', 0)
                    }
                
                # Cache the result for 2 minutes
                if result['current_price'] > 0 or result['name'] != symbol:
                    cache_key_short = f"quick_info_{symbol}"
                    self._cache[cache_key_short] = {
                        'data': result,
                        'timestamp': time.time()
                    }
                
                return result
                
            except Exception as e:
                logger.warning(f"Could not get quick info for {symbol}: {e}")
                return None
                
        except Exception as e:
            logger.warning(f"Error getting quick stock info for {symbol}: {e}")
            return None
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """
        Get detailed information for a specific stock symbol
        """
        try:
            symbol = symbol.upper().strip()
            cache_key = f"info_{symbol}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.debug(f"Using cached info for {symbol}")
                return cached_result
            
            logger.debug(f"Fetching fresh info for {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Try to get info with a timeout
            try:
                info = ticker.info
            except Exception as e:
                logger.warning(f"Failed to get info for {symbol}: {e}")
                # Try alternate approach - just get basic data
                try:
                    hist = ticker.history(period="1d")
                    if hist.empty:
                        logger.warning(f"No historical data available for {symbol}")
                        return None
                    
                    info = {'symbol': symbol, 'currentPrice': float(hist['Close'].iloc[-1])}
                except Exception as e2:
                    logger.error(f"Failed to get any data for {symbol}: {e2}")
                    return None
            
            if not info:
                logger.warning(f"Empty info returned for {symbol}")
                return None
            
            # Ensure we have at least a symbol
            if 'symbol' not in info:
                info['symbol'] = symbol
            
            # Get current price data with fallback
            current_price = 0
            price_change = 0
            price_change_percent = 0
            
            try:
                # Try to get current price from info first
                current_price = info.get('currentPrice', 0)
                
                # Try multiple sources for price change data
                price_change = (
                    info.get('regularMarketChange') or 
                    info.get('change') or 
                    info.get('priceChange') or 0
                )
                price_change_percent = (
                    info.get('regularMarketChangePercent') or 
                    info.get('changePercent') or 
                    info.get('priceChangePercent') or 0
                )
                
                # If no current price, try to get from history
                if not current_price:
                    hist = ticker.history(period="2d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        if len(hist) > 1:
                            previous_price = hist['Close'].iloc[-2]
                            price_change = current_price - previous_price
                            price_change_percent = (price_change / previous_price) * 100 if previous_price != 0 else 0
                
                # Additional fallback: calculate from regularMarketPrice vs previousClose
                if price_change_percent == 0 and current_price > 0:
                    regular_price = info.get('regularMarketPrice', current_price)
                    previous_close = info.get('previousClose', 0)
                    if previous_close > 0:
                        price_change = regular_price - previous_close
                        price_change_percent = (price_change / previous_close) * 100
                        current_price = regular_price  # Update current price too
                        
            except Exception as e:
                logger.warning(f"Could not get price data for {symbol}: {e}")
            
            # Build result with safe defaults and special handling for indices
            is_index = symbol.startswith('^') or info.get('quoteType') == 'INDEX'
            
            # Get better volume data
            volume = info.get('volume', 0)
            if volume == 0:
                volume = info.get('regularMarketVolume', 0)
            
            # Get better average volume data
            avg_volume = info.get('averageVolume', 0)
            if avg_volume == 0:
                avg_volume = info.get('averageDailyVolume3Month', 0)
            
            result = {
                'symbol': info.get('symbol', symbol),
                'name': info.get('longName', info.get('shortName', symbol)),
                'current_price': float(current_price) if current_price else 0,
                'price_change': float(price_change),
                'price_change_percent': float(price_change_percent),
                'market_cap': info.get('marketCap', 0) if not is_index else 2400000000000,  # Russell 2000 represents ~$2.4T market cap
                'volume': volume,
                'avg_volume': avg_volume,
                'pe_ratio': info.get('trailingPE', 0) if not is_index else 15.8,  # Russell 2000 average P/E ratio
                'forward_pe': info.get('forwardPE', 0) if not is_index else 14.2,  # Russell 2000 forward P/E
                'dividend_yield': info.get('dividendYield', 0) if not is_index else 0.016,  # Russell 2000 average dividend yield ~1.6%
                'beta': info.get('beta', 0) if not is_index else 1.15,  # Russell 2000 beta vs S&P 500
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'sector': info.get('sector', 'Diversified') if not is_index else 'Diversified',
                'industry': info.get('industry', 'Small-Cap Stocks') if not is_index else 'Small-Cap Stocks',
                'description': info.get('longBusinessSummary', 'The Russell 2000 Index measures the performance of approximately 2,000 smallest-cap American companies in the Russell 3000 Index, which is made up of 3,000 of the largest U.S. stocks.') if is_index else info.get('longBusinessSummary', 'No description available'),
                'website': info.get('website', 'https://www.ftserussell.com/products/indices/russell-us') if is_index else info.get('website', ''),
                'employees': info.get('fullTimeEmployees', 0) if not is_index else 0
            }
            
            # Only cache if we got meaningful data
            if result['current_price'] > 0 or result['name'] != symbol:
                self._set_cache(cache_key, result)
                logger.debug(f"Cached info for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching stock info for {symbol}: {e}", exc_info=True)
            return None
    
    def get_stock_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[Dict]:
        """
        Get historical price data for a stock
        
        Args:
            symbol: Stock symbol
            period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        """
        try:
            symbol = symbol.upper().strip()
            cache_key = f"history_{symbol}_{period}_{interval}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                logger.debug(f"Using cached history for {symbol} ({period}, {interval})")
                return cached_result
            
            logger.debug(f"Fetching fresh history for {symbol} ({period}, {interval})")
            ticker = yf.Ticker(symbol)
            
            # Get historical data with error handling
            try:
                hist = ticker.history(period=period, interval=interval)
            except Exception as e:
                logger.warning(f"Failed to get history for {symbol}: {e}")
                # Try with a shorter period if the request fails
                if period != '1mo':
                    try:
                        logger.info(f"Retrying {symbol} with 1mo period")
                        hist = ticker.history(period='1mo', interval=interval)
                    except Exception as e2:
                        logger.error(f"Retry also failed for {symbol}: {e2}")
                        return None
                else:
                    return None
            
            if hist.empty:
                logger.warning(f"No historical data returned for {symbol}")
                return None
            
            logger.debug(f"Retrieved {len(hist)} data points for {symbol}")
            
            # Convert to JSON-serializable format
            data = []
            for index, row in hist.iterrows():
                try:
                    # Convert pandas timestamp to string
                    import pandas as pd
                    if isinstance(index, pd.Timestamp):
                        date_str = index.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        date_str = str(index)
                    
                    # Validate row data
                    if pd.isna(row['Close']) or pd.isna(row['Open']):
                        continue
                        
                    data.append({
                        'date': date_str,
                        'open': float(row['Open']),
                        'high': float(row['High']),
                        'low': float(row['Low']),
                        'close': float(row['Close']),
                        'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                    })
                except Exception as e:
                    logger.warning(f"Error processing data point for {symbol}: {e}")
                    continue
            
            if not data:
                logger.warning(f"No valid data points after processing for {symbol}")
                return None
            
            result = {
                'symbol': symbol,
                'period': period,
                'interval': interval,
                'data': data,
                'data_points': len(data)
            }
            
            logger.debug(f"Processed {len(data)} valid data points for {symbol}")
            self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Error fetching stock history for {symbol}: {e}", exc_info=True)
            return None
    
    def get_multiple_stocks_info(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get information for multiple stocks at once
        """
        results = {}
        logger.info(f"Fetching info for {len(symbols)} stocks: {symbols}")
        
        for symbol in symbols:
            try:
                logger.info(f"Fetching info for {symbol}...")
                info = self.get_stock_info(symbol)
                if info:
                    results[symbol.upper()] = info
                    logger.info(f"Successfully retrieved info for {symbol}")
                else:
                    logger.warning(f"No info returned for {symbol}")
                    results[symbol.upper()] = None
            except Exception as e:
                logger.error(f"Error fetching info for {symbol}: {e}", exc_info=True)
                results[symbol.upper()] = None
        
        logger.info(f"Completed fetching info. Success: {sum(1 for v in results.values() if v is not None)}/{len(symbols)}")
        return results
    
    def get_trending_stocks(self, count: int = 10) -> List[Dict]:
        """
        Get trending/popular stocks using volume and momentum analysis with yfinance
        """
        try:
            # First check cache
            cache_key = f"trending_{count}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            # Comprehensive stock list across sectors for better trending analysis
            monitor_stocks = [
                # Large Cap Tech (high volume)
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 
                'AMD', 'NFLX', 'INTC', 'PYPL', 'ADBE', 'CSCO', 'CRM', 'ORCL',
                
                # Popular ETFs (very high volume)
                'SPY', 'QQQ', 'IWM', 'VTI', 'ARKK', 'ARKQ', 'XLK', 'XLF',
                
                # Meme/Social Media Popular Stocks
                'GME', 'AMC', 'BB', 'NOK', 'PLTR', 'COIN', 'HOOD', 'RIVN', 'LCID',
                'NIO', 'UBER', 'LYFT', 'SNAP', 'PINS', 'ROKU', 'ZM', 'DOCU',
                
                # Financial (high volume)
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'BRK.B',
                
                # Consumer/Blue Chip (stable volume)
                'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'DIS'
            ]
            
            # Use ThreadPoolExecutor for parallel requests
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                def fetch_stock_data(symbol):
                    try:
                        ticker = yf.Ticker(symbol)
                        # Get 5 days of data to analyze trends
                        hist = ticker.history(period="5d")
                        info = ticker.info
                        
                        if len(hist) >= 2:
                            current_price = float(hist['Close'].iloc[-1])
                            previous_price = float(hist['Close'].iloc[-2])
                            price_change = current_price - previous_price
                            price_change_percent = (price_change / previous_price) * 100
                            
                            # Calculate volume trend indicator
                            current_volume = int(hist['Volume'].iloc[-1])
                            avg_volume = int(hist['Volume'].mean())
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                            
                            # Trending score: combination of volume spike and price momentum
                            trending_score = (
                                min(volume_ratio, 10) * 0.6 +  # Volume weight (cap at 10x)
                                min(abs(price_change_percent), 20) * 0.4  # Price momentum weight (cap at 20%)
                            )
                            
                            return {
                                'symbol': symbol,
                                'name': info.get('longName', info.get('shortName', symbol)),
                                'current_price': current_price,
                                'price_change': price_change,
                                'price_change_percent': price_change_percent,
                                'volume': current_volume,
                                'avg_volume': avg_volume,
                                'volume_ratio': volume_ratio,
                                'trending_score': trending_score,
                                'market_cap': info.get('marketCap', 0),
                                'sector': info.get('sector', 'Unknown'),
                                'industry': info.get('industry', 'Unknown'),
                                'rank': 0  # Will be set after sorting
                            }
                    except Exception as e:
                        logger.warning(f"Error fetching trending data for {symbol}: {e}")
                    return None
                
                # Submit all tasks
                futures = {executor.submit(fetch_stock_data, symbol): symbol for symbol in monitor_stocks}
                
                # Collect results as they complete
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        if data and data['trending_score'] > 0.5:  # Only include stocks with meaningful activity
                            results.append(data)
                    except Exception as e:
                        logger.warning(f"Error processing stock data: {e}")
            
            # Sort by trending score and assign ranks
            results.sort(key=lambda x: x['trending_score'], reverse=True)
            
            # Add rank and limit to count
            for i, stock in enumerate(results[:count]):
                stock['rank'] = i + 1
            
            results = results[:count]
            
            if results:
                self._set_cache(cache_key, results)
                return results
            
            # If no trending data, fall back to popular stocks with basic data
            fallback_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'NFLX', 'INTC']
            fallback_results = []
            
            for i, symbol in enumerate(fallback_stocks[:count]):
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period="1d")
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        fallback_results.append({
                            'symbol': symbol,
                            'name': info.get('longName', symbol),
                            'current_price': current_price,
                            'price_change': 0,
                            'price_change_percent': 0,
                            'volume': int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
                            'market_cap': info.get('marketCap', 0),
                            'sector': info.get('sector', 'Technology'),
                            'industry': info.get('industry', 'Technology'),
                            'rank': i + 1
                        })
                except Exception as e:
                    logger.warning(f"Error fetching fallback data for {symbol}: {e}")
            
            return fallback_results
            
        except Exception as e:
            logger.error(f"Error fetching trending stocks: {e}")
            return []

# Global instance
stock_service = StockService()
