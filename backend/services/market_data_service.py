import requests
import logging
import time
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class MarketDataService:
    """Service class for fetching market data from external sources"""
    
    def __init__(self):
        self.cache_duration = 300  # 5 minutes cache
        self._cache = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
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
    
    def _clean_number_string(self, value_str: str) -> float:
        """Clean and convert a number string to float"""
        try:
            if not value_str or value_str.strip() == '-':
                return 0.0
            
            # Remove any non-numeric characters except for decimal point, minus sign, and common suffixes
            value_str = value_str.strip()
            
            # Handle percentage values
            if '%' in value_str:
                return float(value_str.replace('%', '').replace(',', ''))
            
            # Handle currency values
            value_str = value_str.replace('$', '').replace(',', '')
            
            # Handle market cap suffixes (B for billions, M for millions, K for thousands)
            multiplier = 1
            if value_str.endswith('B'):
                multiplier = 1_000_000_000
                value_str = value_str[:-1]
            elif value_str.endswith('M'):
                multiplier = 1_000_000
                value_str = value_str[:-1]
            elif value_str.endswith('K'):
                multiplier = 1_000
                value_str = value_str[:-1]
            
            return float(value_str) * multiplier
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse number: {value_str}")
            return 0.0
    
    def _parse_table_data(self, html_content: str, data_type: str) -> List[Dict]:
        """Parse HTML table data using BeautifulSoup"""
        try:
            # Import BeautifulSoup locally to avoid type checking issues
            from bs4 import BeautifulSoup, Tag
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the table
            table = soup.find('table')
            if not table or not isinstance(table, Tag):
                logger.warning(f"Could not find {data_type} table")
                return []
            
            results = []
            
            # Find all table rows
            rows = table.find_all('tr')
            
            # Skip header row
            data_rows = rows[1:] if len(rows) > 1 else []
            
            for i, row in enumerate(data_rows):
                try:
                    if not isinstance(row, Tag):
                        continue
                    cells = row.find_all('td')
                    if len(cells) < 6:  # Need at least 6 columns
                        continue
                    
                    # Based on the website structure:
                    # Column 0: Rank
                    # Column 1: Symbol 
                    # Column 2: Company Name
                    # Column 3: Change %
                    # Column 4: Price
                    # Column 5: Volume
                    # Column 6: Market Cap (if available)
                    
                    rank = self._clean_number_string(cells[0].get_text(strip=True))
                    symbol = cells[1].get_text(strip=True) if cells[1] else ""
                    name = cells[2].get_text(strip=True) if cells[2] else symbol
                    
                    # Change percentage (remove % sign)
                    change_percent_text = cells[3].get_text(strip=True)
                    change_percent = self._clean_number_string(change_percent_text)
                    
                    # Price
                    price = self._clean_number_string(cells[4].get_text(strip=True))
                    
                    # Volume
                    volume = self._clean_number_string(cells[5].get_text(strip=True))
                    
                    # Market cap (if available)
                    market_cap = 0
                    if len(cells) > 6:
                        market_cap = self._clean_number_string(cells[6].get_text(strip=True))
                    
                    # Calculate price change from percentage and current price
                    price_change = 0
                    if price > 0 and change_percent != 0:
                        price_change = (price * change_percent) / (100 + change_percent)
                    
                    if symbol and symbol.isalnum():  # Only add if we have a valid symbol
                        stock_data = {
                            'symbol': symbol,
                            'name': name,
                            'current_price': price,
                            'price_change': price_change,
                            'price_change_percent': change_percent,
                            'volume': volume,
                            'market_cap': market_cap,
                            'rank': int(rank) if rank > 0 else i + 1
                        }
                        results.append(stock_data)
                    
                except Exception as e:
                    logger.warning(f"Error parsing {data_type} row {i}: {e}")
                    continue
            
            return results
            
        except ImportError:
            logger.error("BeautifulSoup4 is not installed. Please install it: pip install beautifulsoup4")
            return []
        except Exception as e:
            logger.error(f"Error parsing {data_type} table: {e}")
            return []
    
    def fetch_top_gainers(self, count: int = 10) -> List[Dict]:
        """
        Fetch top gaining stocks using expanded stock universe with real-time data
        """
        try:
            cache_key = f"gainers_{count}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            # First try web scraping
            try:
                url = "https://stockanalysis.com/markets/gainers/"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                gainers = self._parse_table_data(response.text, "gainers")
                if gainers and len(gainers) >= count:
                    self._set_cache(cache_key, gainers[:count])
                    return gainers[:count]
            except Exception as e:
                logger.warning(f"Failed to fetch gainers from primary source: {e}")

            # Enhanced fallback with broader stock universe
            import yfinance as yf
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Expanded list covering multiple sectors and market caps
            monitor_stocks = [
                # Large Cap Tech
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 
                'AMD', 'NFLX', 'INTC', 'PYPL', 'ADBE', 'CSCO', 'CRM', 'ORCL',
                
                # Financial Services
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'COF',
                'SCHW', 'BLK', 'USB', 'PNC', 'TFC',
                
                # Healthcare & Pharma
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'BMY', 'LLY', 'MRK', 
                'AMGN', 'GILD', 'CVS', 'MRNA', 'MDT', 'DHR',
                
                # Consumer & Retail
                'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'TGT', 
                'LOW', 'COST', 'DIS', 'CMCSA', 'VZ', 'T',
                
                # Energy & Industrials
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'BA', 'CAT', 'GE', 'MMM', 
                'UPS', 'FDX', 'RTX', 'LMT', 'NOC',
                
                # Mid-Cap Growth
                'UBER', 'LYFT', 'SNAP', 'PINS', 'ROKU', 'ZM', 'DOCU', 'OKTA', 
                'SNOW', 'PLTR', 'COIN', 'HOOD', 'SQ', 'SHOP',
                
                # Meme/Popular Stocks
                'GME', 'AMC', 'BB', 'NOK', 'NIO', 'RIVN', 'LCID', 'F', 'GM',
                
                # ETFs and Index Funds
                'SPY', 'QQQ', 'IWM', 'VTI', 'ARKK', 'ARKQ', 'XLK', 'XLF', 'XLE'
            ]

            gainers = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                def fetch_stock_data(symbol):
                    try:
                        ticker = yf.Ticker(symbol)
                        # Get last 2 days to calculate change
                        hist = ticker.history(period="2d")
                        
                        if len(hist) >= 2:
                            current_price = float(hist['Close'].iloc[-1])
                            previous_price = float(hist['Close'].iloc[-2])
                            price_change = current_price - previous_price
                            price_change_percent = (price_change / previous_price) * 100
                            
                            if price_change_percent > 0.1:  # Only significant gainers (>0.1%)
                                # Get additional info
                                info = ticker.info
                                return {
                                    'symbol': symbol,
                                    'name': info.get('longName', info.get('shortName', symbol)),
                                    'current_price': current_price,
                                    'price_change': price_change,
                                    'price_change_percent': price_change_percent,
                                    'volume': int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
                                    'market_cap': info.get('marketCap', 0),
                                    'sector': info.get('sector', 'Unknown'),
                                    'industry': info.get('industry', 'Unknown')
                                }
                    except Exception as e:
                        logger.debug(f"Error fetching {symbol}: {e}")
                        return None

                futures = {executor.submit(fetch_stock_data, symbol): symbol for symbol in monitor_stocks}
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        if data:
                            gainers.append(data)
                    except Exception as e:
                        logger.warning(f"Error processing stock data: {e}")

            # Sort by percentage gain and limit to count
            gainers.sort(key=lambda x: x['price_change_percent'], reverse=True)
            gainers = gainers[:count]
            
            if gainers:
                self._set_cache(cache_key, gainers)
                return gainers
            
            # Return empty list if no data could be fetched
            return []
            
        except Exception as e:
            logger.error(f"Error fetching top gainers: {e}")
            return []
    
    def fetch_top_losers(self, count: int = 10) -> List[Dict]:
        """
        Fetch top losing stocks using expanded stock universe with real-time data
        """
        try:
            cache_key = f"losers_{count}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result

            # First try web scraping
            try:
                url = "https://stockanalysis.com/markets/losers/"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                losers = self._parse_table_data(response.text, "losers")
                if losers and len(losers) >= count:
                    self._set_cache(cache_key, losers[:count])
                    return losers[:count]
            except Exception as e:
                logger.warning(f"Failed to fetch losers from primary source: {e}")

            # Enhanced fallback with broader stock universe
            import yfinance as yf
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Use same expanded list as gainers for consistency
            monitor_stocks = [
                # Large Cap Tech
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 
                'AMD', 'NFLX', 'INTC', 'PYPL', 'ADBE', 'CSCO', 'CRM', 'ORCL',
                
                # Financial Services
                'JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'V', 'MA', 'AXP', 'COF',
                'SCHW', 'BLK', 'USB', 'PNC', 'TFC',
                
                # Healthcare & Pharma
                'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'ABT', 'BMY', 'LLY', 'MRK', 
                'AMGN', 'GILD', 'CVS', 'MRNA', 'MDT', 'DHR',
                
                # Consumer & Retail
                'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'SBUX', 'NKE', 'TGT', 
                'LOW', 'COST', 'DIS', 'CMCSA', 'VZ', 'T',
                
                # Energy & Industrials
                'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'BA', 'CAT', 'GE', 'MMM', 
                'UPS', 'FDX', 'RTX', 'LMT', 'NOC',
                
                # Mid-Cap Growth
                'UBER', 'LYFT', 'SNAP', 'PINS', 'ROKU', 'ZM', 'DOCU', 'OKTA', 
                'SNOW', 'PLTR', 'COIN', 'HOOD', 'SQ', 'SHOP',
                
                # Meme/Popular Stocks
                'GME', 'AMC', 'BB', 'NOK', 'NIO', 'RIVN', 'LCID', 'F', 'GM',
                
                # ETFs and Index Funds
                'SPY', 'QQQ', 'IWM', 'VTI', 'ARKK', 'ARKQ', 'XLK', 'XLF', 'XLE'
            ]

            losers = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                def fetch_stock_data(symbol):
                    try:
                        ticker = yf.Ticker(symbol)
                        # Get last 2 days to calculate change
                        hist = ticker.history(period="2d")
                        
                        if len(hist) >= 2:
                            current_price = float(hist['Close'].iloc[-1])
                            previous_price = float(hist['Close'].iloc[-2])
                            price_change = current_price - previous_price
                            price_change_percent = (price_change / previous_price) * 100
                            
                            if price_change_percent < -0.1:  # Only significant losers (<-0.1%)
                                # Get additional info
                                info = ticker.info
                                return {
                                    'symbol': symbol,
                                    'name': info.get('longName', info.get('shortName', symbol)),
                                    'current_price': current_price,
                                    'price_change': price_change,
                                    'price_change_percent': price_change_percent,
                                    'volume': int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
                                    'market_cap': info.get('marketCap', 0),
                                    'sector': info.get('sector', 'Unknown'),
                                    'industry': info.get('industry', 'Unknown')
                                }
                    except Exception as e:
                        logger.debug(f"Error fetching {symbol}: {e}")
                        return None

                futures = {executor.submit(fetch_stock_data, symbol): symbol for symbol in monitor_stocks}
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        if data:
                            losers.append(data)
                    except Exception as e:
                        logger.warning(f"Error processing stock data: {e}")

            # Sort by percentage loss (most negative first) and limit to count
            losers.sort(key=lambda x: x['price_change_percent'])
            losers = losers[:count]
            
            if losers:
                self._set_cache(cache_key, losers)
                return losers
            
            # Return empty list if no data could be fetched
            return []
            
        except Exception as e:
            logger.error(f"Error fetching top losers: {e}")
            return []
    
    def fetch_market_movers(self, count: int = 10) -> Dict[str, Any]:
        """
        Fetch both top gainers and losers in one call
        """
        try:
            gainers = self.fetch_top_gainers(count)
            losers = self.fetch_top_losers(count)
            
            return {
                'gainers': gainers,
                'losers': losers,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            logger.error(f"Error fetching market movers: {e}")
            return {
                'gainers': [],
                'losers': [],
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _parse_trending_table_data(self, html_content: str) -> List[Dict]:
        """Parse trending stocks HTML table data from stockanalysis.com/trending/"""
        try:
            # Import BeautifulSoup locally to avoid type checking issues
            from bs4 import BeautifulSoup, Tag
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Find the table - look for table with trending data
            table = soup.find('table')
            if not table or not isinstance(table, Tag):
                logger.warning("Could not find trending stocks table")
                return []
            
            results = []
            
            # Find all table rows
            rows = table.find_all('tr')
            
            # Skip header row
            data_rows = rows[1:] if len(rows) > 1 else []
            
            for i, row in enumerate(data_rows):
                try:
                    if not isinstance(row, Tag):
                        continue
                    cells = row.find_all('td')
                    if len(cells) < 6:  # Need at least 6 columns
                        continue
                    
                    # Based on stockanalysis.com trending page structure:
                    # Column 0: Rank
                    # Column 1: Symbol 
                    # Column 2: Company Name
                    # Column 3: Pageviews
                    # Column 4: Market Cap
                    # Column 5: Change %
                    # Column 6: Volume (if available)
                    
                    rank = self._clean_number_string(cells[0].get_text(strip=True))
                    symbol = cells[1].get_text(strip=True) if cells[1] else ""
                    name = cells[2].get_text(strip=True) if cells[2] else symbol
                    pageviews = self._clean_number_string(cells[3].get_text(strip=True))
                    market_cap = self._clean_number_string(cells[4].get_text(strip=True))
                    
                    # Change percentage (remove % sign)
                    change_percent_text = cells[5].get_text(strip=True)
                    change_percent = self._clean_number_string(change_percent_text)
                    
                    # Volume (if available)
                    volume = 0
                    if len(cells) > 6:
                        volume = self._clean_number_string(cells[6].get_text(strip=True))
                    
                    if symbol and len(symbol) >= 1 and len(symbol) <= 5:  # Valid stock symbol format
                        stock_data = {
                            'symbol': symbol,
                            'name': name,
                            'pageviews': pageviews,
                            'market_cap': market_cap,
                            'price_change_percent': change_percent,
                            'volume': volume,
                            'rank': int(rank) if rank > 0 else i + 1
                        }
                        results.append(stock_data)
                    
                except Exception as e:
                    logger.warning(f"Error parsing trending row {i}: {e}")
                    continue
            
            return results
            
        except ImportError:
            logger.error("BeautifulSoup4 is not installed. Please install it: pip install beautifulsoup4")
            return []
        except Exception as e:
            logger.error(f"Error parsing trending table: {e}")
            return []
    
    def fetch_trending_stocks(self, count: int = 10) -> List[Dict]:
        """
        Fetch trending stocks from multiple sources with enhanced data
        """
        try:
            cache_key = f"trending_{count}"
            cached_result = self._get_from_cache(cache_key)
            if cached_result:
                return cached_result
            
            # First try web scraping from stockanalysis.com trending
            try:
                url = "https://stockanalysis.com/trending/"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                trending_stocks = self._parse_trending_table_data(response.text)[:count]
                
                if trending_stocks and len(trending_stocks) >= min(5, count):
                    # Enhance with current price data from yfinance
                    enhanced_stocks = self._enhance_trending_stocks(trending_stocks)
                    if enhanced_stocks:
                        self._set_cache(cache_key, enhanced_stocks)
                        return enhanced_stocks
            except Exception as e:
                logger.warning(f"Failed to fetch trending from primary source: {e}")

            # Fallback: Use volume-based trending analysis
            import yfinance as yf
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Expanded universe for better trending analysis
            monitor_stocks = [
                # High volume tech stocks
                'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA', 
                'AMD', 'NFLX', 'INTC', 'PYPL', 'ADBE', 'CSCO', 'CRM', 'ORCL',
                
                # Popular trading stocks
                'SPY', 'QQQ', 'IWM', 'VTI', 'GME', 'AMC', 'BB', 'NOK', 'PLTR',
                'COIN', 'HOOD', 'RIVN', 'LCID', 'NIO', 'UBER', 'LYFT',
                
                # Meme and social media favorites
                'SNAP', 'PINS', 'ROKU', 'ZM', 'DOCU', 'OKTA', 'SNOW', 'SQ', 'SHOP',
                
                # Financial and blue chips
                'JPM', 'BAC', 'WFC', 'C', 'V', 'MA', 'BRK.B', 'WMT', 'DIS', 'KO'
            ]

            trending_stocks = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                def fetch_trending_data(symbol):
                    try:
                        ticker = yf.Ticker(symbol)
                        
                        # Get 2 days of data to analyze volume and price trends
                        hist = ticker.history(period="5d")
                        
                        if len(hist) >= 2:
                            current_price = float(hist['Close'].iloc[-1])
                            previous_price = float(hist['Close'].iloc[-2])
                            price_change = current_price - previous_price
                            price_change_percent = (price_change / previous_price) * 100
                            
                            # Calculate volume trend (compare recent vs average)
                            current_volume = int(hist['Volume'].iloc[-1])
                            avg_volume = int(hist['Volume'].mean()) if len(hist) > 1 else current_volume
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                            
                            # Calculate trending score based on volume and price momentum
                            trending_score = (abs(price_change_percent) * 0.3 + 
                                            min(volume_ratio, 5) * 0.7)  # Cap volume ratio at 5x
                            
                            if trending_score > 0.5:  # Only stocks with meaningful activity
                                info = ticker.info
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
                                    'industry': info.get('industry', 'Unknown')
                                }
                    except Exception as e:
                        logger.debug(f"Error fetching trending data for {symbol}: {e}")
                        return None

                futures = {executor.submit(fetch_trending_data, symbol): symbol for symbol in monitor_stocks}
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        if data:
                            trending_stocks.append(data)
                    except Exception as e:
                        logger.warning(f"Error processing trending stock data: {e}")

            # Sort by trending score (volume + momentum) and return top results
            trending_stocks.sort(key=lambda x: x['trending_score'], reverse=True)
            trending_stocks = trending_stocks[:count]
            
            if trending_stocks:
                self._set_cache(cache_key, trending_stocks)
                return trending_stocks
            
            return []
            
        except Exception as e:
            logger.error(f"Error fetching trending stocks: {e}")
            return []

    def _enhance_trending_stocks(self, trending_stocks: List[Dict]) -> List[Dict]:
        """Enhance trending stocks with real-time price data"""
        try:
            import yfinance as yf
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            enhanced_stocks = []
            
            def enhance_stock(stock):
                try:
                    ticker = yf.Ticker(stock['symbol'])
                    hist = ticker.history(period="2d")
                    info = ticker.info
                    
                    if not hist.empty:
                        current_price = float(hist['Close'].iloc[-1])
                        if len(hist) > 1:
                            previous_price = float(hist['Close'].iloc[-2])
                            price_change = current_price - previous_price
                            price_change_percent = (price_change / previous_price) * 100
                        else:
                            price_change = 0
                            price_change_percent = 0
                        
                        return {
                            **stock,
                            'current_price': current_price,
                            'price_change': price_change,
                            'price_change_percent': price_change_percent,
                            'volume': int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
                            'sector': info.get('sector', stock.get('sector', 'Unknown')),
                            'industry': info.get('industry', stock.get('industry', 'Unknown')),
                            'market_cap': info.get('marketCap', stock.get('market_cap', 0))
                        }
                except Exception as e:
                    logger.warning(f"Could not enhance stock data for {stock['symbol']}: {e}")
                    return stock  # Return original if enhancement fails
            
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {executor.submit(enhance_stock, stock): stock for stock in trending_stocks}
                for future in as_completed(futures):
                    try:
                        enhanced_stock = future.result()
                        if enhanced_stock:
                            enhanced_stocks.append(enhanced_stock)
                    except Exception as e:
                        logger.warning(f"Error enhancing stock: {e}")
            
            return enhanced_stocks
            
        except Exception as e:
            logger.error(f"Error enhancing trending stocks: {e}")
            return trending_stocks  # Return original if enhancement fails

# Global instance
market_data_service = MarketDataService()
