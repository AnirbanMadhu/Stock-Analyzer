from flask import Blueprint, request, jsonify
import logging
import yfinance as yf
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Create blueprint for news-related routes
news_bp = Blueprint('news', __name__)

@news_bp.route('/market', methods=['GET'])
def get_market_news():
    """
    Get general market news from Yahoo Finance Markets
    Fetches from major market indices and ETFs that represent the overall market
    Query parameters:
    - limit: number of articles to return (default: 10, max: 20)
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        
        if limit > 20:
            limit = 20
        elif limit < 1:
            limit = 10
        
        # Major market tickers representing the overall market (from finance.yahoo.com/markets/)
        market_tickers = ['^GSPC', '^DJI', '^IXIC', 'SPY', 'QQQ', 'DIA', '^RUT', 'IWM']
        all_news = []
        
        for ticker in market_tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                for article in news[:3]:  # Limit per ticker
                    if len(all_news) >= limit:
                        break
                    
                    # Skip articles without essential fields
                    if not article.get('title') or not article.get('link'):
                        continue
                    
                    # Handle different timestamp formats
                    try:
                        if 'providerPublishTime' in article:
                            published_date = datetime.fromtimestamp(article['providerPublishTime']).isoformat()
                        elif 'publishTime' in article:
                            published_date = datetime.fromtimestamp(article['publishTime']).isoformat()
                        else:
                            published_date = datetime.now().isoformat()
                    except (KeyError, ValueError, TypeError):
                        published_date = datetime.now().isoformat()
                    
                    # Check for duplicates
                    if not any(existing['url'] == article['link'] for existing in all_news):
                        all_news.append({
                            'title': article.get('title', 'No title available'),
                            'summary': article.get('summary', ''),
                            'url': article.get('link', ''),
                            'source': article.get('publisher', 'Unknown source'),
                            'published_date': published_date,
                            'thumbnail': article.get('thumbnail', {}).get('resolutions', [{}])[-1].get('url', '') if article.get('thumbnail') else '',
                            'related_tickers': []
                        })
                
                if len(all_news) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Could not fetch market news for {ticker}: {e}")
                continue
        
        # Sort by date (newest first)
        all_news.sort(key=lambda x: x['published_date'], reverse=True)
        
        return jsonify({
            'articles': all_news[:limit],
            'count': len(all_news[:limit]),
            'source': 'Yahoo Finance Markets',
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching market news: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@news_bp.route('/trending', methods=['GET'])
def get_trending_news():
    """
    Get trending financial news from Yahoo Finance Markets
    Fetches from popular and trending stocks that represent market trends
    Query parameters:
    - limit: number of articles to return (default: 15, max: 30)
    """
    try:
        limit = request.args.get('limit', 15, type=int)
        
        if limit > 30:
            limit = 30
        elif limit < 1:
            limit = 15
        
        # Popular trending stocks from Yahoo Finance markets
        trending_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'CRM', 'BABA', 'UBER']
        all_news = []
        
        for ticker in trending_tickers:
            try:
                stock = yf.Ticker(ticker)
                news = stock.news
                
                for article in news[:2]:  # Limit per ticker to get variety
                    if len(all_news) >= limit:
                        break
                    
                    # Skip articles without essential fields
                    if not article.get('title') or not article.get('link'):
                        continue
                    
                    # Handle different timestamp formats
                    try:
                        if 'providerPublishTime' in article:
                            published_date = datetime.fromtimestamp(article['providerPublishTime']).isoformat()
                        elif 'publishTime' in article:
                            published_date = datetime.fromtimestamp(article['publishTime']).isoformat()
                        else:
                            published_date = datetime.now().isoformat()
                    except (KeyError, ValueError, TypeError):
                        published_date = datetime.now().isoformat()
                    
                    # Check for duplicates
                    if not any(existing['url'] == article['link'] for existing in all_news):
                        all_news.append({
                            'title': article.get('title', 'No title available'),
                            'summary': article.get('summary', ''),
                            'url': article.get('link', ''),
                            'source': article.get('publisher', 'Unknown source'),
                            'published_date': published_date,
                            'thumbnail': article.get('thumbnail', {}).get('resolutions', [{}])[-1].get('url', '') if article.get('thumbnail') else '',
                            'related_tickers': [ticker]
                        })
                
                if len(all_news) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Could not fetch trending news for {ticker}: {e}")
                continue
        
        # Sort by date (newest first)
        all_news.sort(key=lambda x: x['published_date'], reverse=True)
        
        return jsonify({
            'articles': all_news[:limit],
            'count': len(all_news[:limit]),
            'source': 'Yahoo Finance Markets',
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching trending news: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@news_bp.route('/stock/<symbol>', methods=['GET'])
def get_stock_news(symbol):
    """
    Get news for a specific stock symbol
    Query parameters:
    - limit: number of articles to return (default: 10, max: 20)
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        limit = request.args.get('limit', 10, type=int)
        
        if limit > 20:
            limit = 20
        elif limit < 1:
            limit = 10
        
        # Get stock-specific news using yfinance
        stock = yf.Ticker(symbol.upper())
        news = stock.news
        
        news_articles = []
        for article in news[:limit]:
            # Skip articles without essential fields
            if not article.get('title') or not article.get('link'):
                continue
                
            # Handle different timestamp formats
            try:
                if 'providerPublishTime' in article:
                    published_date = datetime.fromtimestamp(article['providerPublishTime']).isoformat()
                elif 'publishTime' in article:
                    published_date = datetime.fromtimestamp(article['publishTime']).isoformat()
                else:
                    published_date = datetime.now().isoformat()
            except (KeyError, ValueError, TypeError):
                published_date = datetime.now().isoformat()
            
            news_articles.append({
                'title': article.get('title', 'No title available'),
                'summary': article.get('summary', ''),
                'url': article.get('link', ''),
                'source': article.get('publisher', 'Unknown source'),
                'published_date': published_date,
                'thumbnail': article.get('thumbnail', {}).get('resolutions', [{}])[-1].get('url', '') if article.get('thumbnail') else '',
                'related_tickers': [symbol.upper()]
            })
        
        return jsonify({
            'symbol': symbol.upper(),
            'articles': news_articles,
            'count': len(news_articles),
            'source': 'Yahoo Finance',
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error fetching news for {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500


