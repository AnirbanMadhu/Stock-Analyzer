from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from flask_login import LoginManager, login_required, current_user
from dotenv import load_dotenv
import os
import logging

# Load environment variables
load_dotenv()

# Import database and models
from models import User, Portfolio, Watchlist
from database import DatabaseConfig, init_database, test_database_connection, mongo

# Initialize Flask app with static folder pointing to frontend build
app = Flask(__name__, 
           static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend', 'dist'),
           static_url_path='')

# Configure database and app settings
DatabaseConfig.configure_app(app)

# Security and session configuration
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production'),
    SESSION_COOKIE_SECURE=os.getenv('FLASK_ENV') == 'production',  # Use HTTPS in production
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    REMEMBER_COOKIE_SAMESITE='Lax',
    REMEMBER_COOKIE_HTTPONLY=True,
    REMEMBER_COOKIE_DURATION=2592000,  # 30 days in seconds
    SESSION_TYPE='filesystem',
    PERMANENT_SESSION_LIFETIME=2592000,  # 30 days in seconds
    SESSION_COOKIE_NAME='stock_analyzer_session',
    SESSION_FILE_DIR=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'flask_session'),
    SESSION_FILE_THRESHOLD=500
)

# Initialize Flask-Session
from flask_session import Session
Session(app)

# Initialize extensions
mongo.init_app(app)

# Initialize and configure Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.session_protection = 'strong'

# Enable CORS for React frontend
CORS(app, 
     resources={
         r"/api/*": {
             "origins": [
                 "http://localhost:5173",
                 "http://127.0.0.1:5173",
                 "http://localhost:3001",
                 "http://127.0.0.1:3001",
                 # Production domains for Render
                 "https://*.onrender.com",
                 # For merged deployment, allow same origin
                 "*",  # This allows same-origin requests in production
                 # Development origins
                 "http://localhost:*",
                 "http://127.0.0.1:*"
             ],
             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
             "allow_headers": [
                 "Content-Type", 
                 "Authorization", 
                 "Access-Control-Allow-Credentials",
                 "X-Requested-With",
                 "Accept",
                 "Origin"
             ],
             "supports_credentials": True,
             "expose_headers": ["Set-Cookie", "Authorization"],
             "max_age": 600,  # Cache preflight requests for 10 minutes
             "allow_credentials": True
         }
     },
     supports_credentials=True)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# User loader for Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.find_by_id(user_id)

# Import and register blueprints
from routes.stock_routes import stock_bp
from routes.auth_routes import auth_bp
from routes.portfolio_routes import portfolio_bp
from routes.prediction_routes import prediction_bp
from routes.enhanced_prediction_routes import enhanced_prediction_bp
from routes.news_routes import news_bp

# Configure enhanced prediction routes with additional paths
@enhanced_prediction_bp.route('/compare-models/<symbol>', methods=['GET'])
def compare_models(symbol):
    """Compare different prediction models for a stock"""
    try:
        days = request.args.get('days', 7, type=int)
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
            
        symbol = symbol.upper().strip()
        
        from ml_models.stock_predictor import stock_predictor
        comparison = stock_predictor.compare_models(symbol, days)
        
        if comparison:
            return jsonify(comparison), 200
        else:
            return jsonify({'error': 'Could not generate model comparison'}), 404
    except Exception as e:
        logger.error(f"Error comparing models for {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

app.register_blueprint(stock_bp, url_prefix='/api')
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(portfolio_bp, url_prefix='/api/portfolio')
app.register_blueprint(prediction_bp, url_prefix='/api/predict')
app.register_blueprint(enhanced_prediction_bp, url_prefix='/api/ai')
app.register_blueprint(news_bp, url_prefix='/api/news')

# Configure login manager after blueprints are registered
login_manager.login_view = 'auth.login'  # Use auth blueprint's login route  # type: ignore
login_manager.login_message = 'Please log in to access this page.'
login_manager.login_message_category = 'info'

@login_manager.unauthorized_handler
def unauthorized():
    if request.blueprint == 'auth' and request.endpoint == 'auth.check':
        # For auth check endpoint, return 200 with authenticated=false
        return jsonify({
            'authenticated': False,
            'message': 'Not authenticated'
        }), 200
    else:
        # For other endpoints that require authentication
        return jsonify({
            'error': 'Unauthorized',
            'message': 'Please log in to access this resource',
            'authenticated': False
        }), 401

# Add watchlist routes as a separate blueprint
from flask import Blueprint
watchlist_bp = Blueprint('watchlist', __name__)

@watchlist_bp.route('/', methods=['GET'])
@login_required
def get_watchlist_root():
    """Get user's watchlist"""
    from routes.portfolio_routes import get_watchlist
    return get_watchlist()

@watchlist_bp.route('/', methods=['POST'])
@login_required
def add_to_watchlist_root():
    """Add to user's watchlist"""
    from routes.portfolio_routes import add_to_watchlist
    return add_to_watchlist()

app.register_blueprint(watchlist_bp, url_prefix='/api/watchlist')

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': 'Stock Analyzer API is running'})

# Serve React frontend
@app.route('/')
def serve_react_app():
    """Serve the React frontend"""
    if not app.static_folder or not os.path.exists(app.static_folder):
        return jsonify({'error': 'Frontend not built. Please run npm run build in frontend directory.'}), 404
    
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except FileNotFoundError:
        return jsonify({'error': 'Frontend not built. Please run npm run build in frontend directory.'}), 404

@app.route('/<path:path>')
def serve_static_files(path):
    """Serve static files from React build"""
    if not app.static_folder or not os.path.exists(app.static_folder):
        return jsonify({'error': 'Frontend not built. Please run npm run build in frontend directory.'}), 404
    
    try:
        # Try to serve the requested file
        return send_from_directory(app.static_folder, path)
    except FileNotFoundError:
        # For client-side routing, serve index.html for non-API routes
        if not path.startswith('api/'):
            try:
                return send_from_directory(app.static_folder, 'index.html')
            except FileNotFoundError:
                return jsonify({'error': 'Frontend not built. Please run npm run build in frontend directory.'}), 404
        else:
            return jsonify({'error': 'API endpoint not found'}), 404

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Create database tables and test connection
with app.app_context():
    # Test database connection first
    success, message = test_database_connection(app)
    if success:
        logger.info(f"✅ {message}")
        # Initialize database
        if init_database(app):
            logger.info("✅ Database tables created/verified successfully")
        else:
            logger.error("❌ Database initialization failed")
    else:
        logger.error(f"❌ {message}")
        
    # Log connection info
    conn_info = DatabaseConfig.get_connection_info()
    logger.info(f"📊 Using {conn_info['type']} database")

if __name__ == '__main__':
    try:
        # Get port from environment variable (Render sets this automatically)
        port = int(os.getenv('PORT', 3001))
        
        # Determine if running in production
        is_production = os.getenv('RENDER') == 'true' or os.getenv('FLASK_ENV') == 'production'
        
        if is_production:
            logger.info("🚀 Starting Flask production server...")
            logger.info(f"🌐 Server will be available on port: {port}")
            # In production, gunicorn will handle the server
            # This block is mainly for logging purposes
            app.run(debug=False, host='0.0.0.0', port=port, threaded=True)
        else:
            logger.info("🚀 Starting Flask development server...")
            logger.info(f"🌐 Server will be available at: http://localhost:{port}")
            logger.info(f"🌐 Server will be available at: http://127.0.0.1:{port}")
            app.run(debug=True, host='127.0.0.1', port=port, threaded=True, use_reloader=False)
    except Exception as e:
        logger.error(f"❌ Failed to start Flask server: {e}")
        import traceback
        traceback.print_exc()
