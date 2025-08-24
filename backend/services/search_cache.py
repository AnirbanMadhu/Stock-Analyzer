"""
Search cache service for optimizing stock search performance
Provides in-memory caching for search results and stock data
"""
import time
import logging
from typing import Dict, List, Optional, Any
from threading import Lock

logger = logging.getLogger(__name__)

class SearchCache:
    """High-performance search cache with TTL support"""
    
    def __init__(self, default_ttl: int = 300):  # 5 minutes default
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl
        self.max_cache_size = 1000  # Prevent memory overflow
        
        # Pre-built popular stocks for instant results
        self.popular_stocks = {
            'AAPL': {'name': 'Apple Inc.', 'sector': 'Technology'},
            'GOOGL': {'name': 'Alphabet Inc. Class A', 'sector': 'Technology'},
            'MSFT': {'name': 'Microsoft Corporation', 'sector': 'Technology'},
            'AMZN': {'name': 'Amazon.com Inc.', 'sector': 'Technology'},
            'TSLA': {'name': 'Tesla Inc.', 'sector': 'Automotive'},
            'META': {'name': 'Meta Platforms Inc.', 'sector': 'Technology'},
            'NVDA': {'name': 'NVIDIA Corporation', 'sector': 'Technology'},
            'NFLX': {'name': 'Netflix Inc.', 'sector': 'Technology'},
            'JPM': {'name': 'JPMorgan Chase & Co.', 'sector': 'Financial Services'},
            'V': {'name': 'Visa Inc.', 'sector': 'Financial Services'},
            'JNJ': {'name': 'Johnson & Johnson', 'sector': 'Healthcare'},
            'WMT': {'name': 'Walmart Inc.', 'sector': 'Consumer'},
            'PG': {'name': 'Procter & Gamble Company', 'sector': 'Consumer'},
            'UNH': {'name': 'UnitedHealth Group Inc.', 'sector': 'Healthcare'},
            'HD': {'name': 'Home Depot Inc.', 'sector': 'Consumer'},
            'MA': {'name': 'Mastercard Inc.', 'sector': 'Financial Services'},
            'DIS': {'name': 'Walt Disney Company', 'sector': 'Consumer'},
            'PYPL': {'name': 'PayPal Holdings Inc.', 'sector': 'Financial Services'},
            'ADBE': {'name': 'Adobe Inc.', 'sector': 'Technology'},
            'CRM': {'name': 'Salesforce Inc.', 'sector': 'Technology'},
            'BAC': {'name': 'Bank of America Corporation', 'sector': 'Financial Services'},
            'XOM': {'name': 'Exxon Mobil Corporation', 'sector': 'Energy'},
            'CVX': {'name': 'Chevron Corporation', 'sector': 'Energy'},
            'PFE': {'name': 'Pfizer Inc.', 'sector': 'Healthcare'},
            'KO': {'name': 'Coca-Cola Company', 'sector': 'Consumer'},
            'PEP': {'name': 'PepsiCo Inc.', 'sector': 'Consumer'},
            'TMO': {'name': 'Thermo Fisher Scientific', 'sector': 'Healthcare'},
            'COST': {'name': 'Costco Wholesale Corporation', 'sector': 'Consumer'},
            'ABBV': {'name': 'AbbVie Inc.', 'sector': 'Healthcare'},
        }
    
    def get(self, key: str, ttl: Optional[int] = None) -> Optional[Any]:
        """Get cached value if not expired"""
        with self._lock:
            if key not in self._cache:
                return None
            
            cache_entry = self._cache[key]
            cache_time = cache_entry.get('timestamp', 0)
            used_ttl = ttl or cache_entry.get('ttl', self.default_ttl)
            
            if time.time() - cache_time > used_ttl:
                # Expired, remove from cache
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                return None
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            return cache_entry['data']
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set cached value with TTL"""
        with self._lock:
            # Clean up old entries if cache is getting too large
            if len(self._cache) >= self.max_cache_size:
                self._cleanup_old_entries()
            
            used_ttl = ttl or self.default_ttl
            self._cache[key] = {
                'data': value,
                'timestamp': time.time(),
                'ttl': used_ttl
            }
            self._access_times[key] = time.time()
    
    def get_instant_suggestions(self, query: str, limit: int = 8) -> List[Dict]:
        """Get instant suggestions from popular stocks without API calls"""
        query_upper = query.upper().strip()
        query_lower = query.lower().strip()
        results = []
        
        if not query or len(query) < 1:
            return []
        
        # Search through popular stocks for instant results
        for symbol, info in self.popular_stocks.items():
            if len(results) >= limit:
                break
                
            name_lower = info['name'].lower()
            
            # Exact symbol match (highest priority)
            if symbol == query_upper:
                results.insert(0, {
                    'symbol': symbol,
                    'name': info['name'],
                    'sector': info['sector'],
                    'current_price': 0,  # Will be fetched later
                    'price_change': 0,
                    'price_change_percent': 0,
                    'industry': 'Various',
                    'market_cap': 0,
                    'priority': 100
                })
            # Symbol starts with query
            elif symbol.startswith(query_upper):
                results.append({
                    'symbol': symbol,
                    'name': info['name'],
                    'sector': info['sector'],
                    'current_price': 0,
                    'price_change': 0,
                    'price_change_percent': 0,
                    'industry': 'Various',
                    'market_cap': 0,
                    'priority': 90
                })
            # Company name contains query
            elif (query_lower in name_lower or 
                  any(word.startswith(query_lower) for word in name_lower.split())):
                priority = 80 if name_lower.startswith(query_lower) else 70
                results.append({
                    'symbol': symbol,
                    'name': info['name'],
                    'sector': info['sector'],
                    'current_price': 0,
                    'price_change': 0,
                    'price_change_percent': 0,
                    'industry': 'Various',
                    'market_cap': 0,
                    'priority': priority
                })
        
        # Sort by priority and return
        results.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        # Remove priority field before returning
        for result in results:
            result.pop('priority', None)
        
        return results[:limit]
    
    def _cleanup_old_entries(self) -> None:
        """Remove old entries to prevent memory overflow"""
        current_time = time.time()
        
        # Remove expired entries first
        expired_keys = []
        for key, entry in self._cache.items():
            cache_time = entry.get('timestamp', 0)
            ttl = entry.get('ttl', self.default_ttl)
            if current_time - cache_time > ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
        
        # If still too large, remove least recently used entries
        if len(self._cache) >= self.max_cache_size * 0.8:
            # Sort by access time and remove oldest 20%
            sorted_by_access = sorted(
                self._access_times.items(), 
                key=lambda x: x[1]
            )
            
            num_to_remove = int(len(self._cache) * 0.2)
            for key, _ in sorted_by_access[:num_to_remove]:
                if key in self._cache:
                    del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
    
    def clear(self) -> None:
        """Clear all cached data"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            return {
                'total_entries': len(self._cache),
                'max_size': self.max_cache_size,
                'default_ttl': self.default_ttl,
                'popular_stocks_count': len(self.popular_stocks)
            }

# Global cache instance
search_cache = SearchCache()
