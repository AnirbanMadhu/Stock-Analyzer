from flask import Blueprint, request, jsonify, current_app
from flask_login import login_user, logout_user, login_required, current_user
import logging
from models import User
from database import mongo

logger = logging.getLogger(__name__)

# Create blueprint for authentication routes
auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/register', methods=['POST'])
def register():
    """
    User registration endpoint
    Expected JSON payload:
    {
        "username": "string",
        "email": "string",
        "password": "string"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        required_fields = ['username', 'email', 'password']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'{field.capitalize()} is required'}), 400
        
        username = data['username'].strip()
        email = data['email'].strip().lower()
        password = data['password']
        
        # Validate username length
        if len(username) < 3 or len(username) > 50:
            return jsonify({'error': 'Username must be between 3 and 50 characters'}), 400
        
        # Validate password strength
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters long'}), 400
        
        # Check if user already exists
        if User.find_by_username(username):
            return jsonify({'error': 'Username already exists'}), 400
        
        if User.find_by_email(email):
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        user = User(username=username, email=email, password=password)
        user.save()
        
        # Log the user in
        login_user(user)
        
        logger.info(f"New user registered: {username}")
        
        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error during registration: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/login', methods=['POST'])
def login():
    """
    User login endpoint
    Expected JSON payload:
    {
        "username": "string",  # Can be username or email
        "password": "string"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            logger.warning("No data provided in login request")
            return jsonify({'error': 'No data provided'}), 400
        
        # Debug logging
        logger.debug(f"Login attempt - data received: {data}")
        
        username_or_email = data.get('username', '').strip()
        password = data.get('password', '')
        
        # Debug logging
        logger.debug(f"Login attempt - username: '{username_or_email}', password length: {len(password)}")
        
        if not username_or_email or not password:
            logger.warning("Missing username/email or password")
            return jsonify({'error': 'Username/email and password are required'}), 400
        
        try:
            # Find user by username or email
            user = User.find_by_username(username_or_email)
            if not user:
                user = User.find_by_email(username_or_email.lower())
            
            if not user:
                logger.warning(f"User not found: {username_or_email}")
                return jsonify({'error': 'Invalid username/email or password'}), 401
            
            if not user.check_password(password):
                logger.warning(f"Invalid password for user: {username_or_email}")
                return jsonify({'error': 'Invalid username/email or password'}), 401
            
            if not user.is_active:
                logger.warning(f"Inactive account: {username_or_email}")
                return jsonify({'error': 'Account is deactivated'}), 401
            
            # Log the user in
            login_user(user, remember=data.get('remember', False))
            
            # Create session data
            from flask import session
            session['user_id'] = user.id
            session['username'] = user.username
            session.permanent = True
            
            logger.info(f"User logged in successfully: {user.username}")
            
            return jsonify({
                'message': 'Login successful',
                'user': user.to_dict()
            }), 200
            
        except Exception as db_error:
            logger.error(f"Database error during login: {str(db_error)}")
            raise
        
    except Exception as e:
        logger.exception(f"Error during login: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred during login. Please try again.',
            'debug_info': str(e) if current_app.debug else None
        }), 500

@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """
    User logout endpoint
    """
    try:
        username = current_user.username
        logout_user()
        
        logger.info(f"User logged out: {username}")
        
        return jsonify({'message': 'Logout successful'}), 200
        
    except Exception as e:
        logger.error(f"Error during logout: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/profile', methods=['GET'])
@login_required
def get_profile():
    """
    Get current user profile
    """
    try:
        return jsonify({
            'user': current_user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching profile: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/profile', methods=['PUT'])
@login_required
def update_profile():
    """
    Update user profile
    Expected JSON payload:
    {
        "email": "string" (optional),
        "current_password": "string" (required if changing password),
        "new_password": "string" (optional)
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        user = current_user
        
        # Update email if provided
        if 'email' in data:
            new_email = data['email'].strip().lower()
            if new_email != user.email:
                # Check if email is already taken
                existing_user = User.find_by_email(new_email)
                if existing_user and existing_user.id != user.id:
                    return jsonify({'error': 'Email already registered'}), 400
                user.email = new_email
        
        # Update password if provided
        if 'new_password' in data:
            current_password = data.get('current_password', '')
            new_password = data['new_password']
            
            if not current_password:
                return jsonify({'error': 'Current password is required to change password'}), 400
            
            if not user.check_password(current_password):
                return jsonify({'error': 'Current password is incorrect'}), 400
            
            if len(new_password) < 6:
                return jsonify({'error': 'New password must be at least 6 characters long'}), 400
            
            user.set_password(new_password)
        
        user.save()
        
        logger.info(f"Profile updated for user: {user.username}")
        
        return jsonify({
            'message': 'Profile updated successfully',
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Error updating profile: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@auth_bp.route('/check', methods=['GET'])
def check_auth():
    """
    Check if user is authenticated
    """
    logger.debug("Starting authentication check")
    try:
        # Set CORS headers for this route specifically
        headers = {
            'Access-Control-Allow-Origin': request.headers.get('Origin', '*'),
            'Access-Control-Allow-Credentials': 'true'
        }
        
        # Log the current_user state
        logger.debug(f"Current user object: {current_user}")
        logger.debug(f"Current user type: {type(current_user)}")
        
        # First check if current_user exists and has the required attribute
        if not hasattr(current_user, 'is_authenticated'):
            logger.warning("current_user does not have is_authenticated attribute")
            return jsonify({
                'authenticated': False,
                'message': 'No valid session found'
            }), 200, headers
        
        # Check authentication status
        authenticated = current_user.is_authenticated
        logger.debug(f"Authentication status: {authenticated}")
        
        response = {
            'authenticated': authenticated,
            'message': 'Authentication check successful'
        }
        
        # If authenticated, try to get user data
        if authenticated:
            try:
                # Try to get user data from database to ensure session is valid
                user = User.find_by_id(current_user.id)
                if user is None:
                    logger.warning(f"User not found in database: {current_user.id}")
                    return jsonify({
                        'authenticated': False,
                        'message': 'User session invalid'
                    }), 200
                
                # Get user data using to_dict if available
                if hasattr(user, 'to_dict'):
                    response['user'] = user.to_dict()
                else:
                    response['user'] = {
                        'id': user.id,
                        'username': user.username,
                        'email': user.email
                    }
                logger.debug(f"User data successfully serialized: {response['user']}")
                
            except Exception as e:
                logger.exception(f"Error serializing user data: {str(e)}")
                response['user'] = {'id': current_user.id}
        
        logger.debug(f"Returning response: {response}")
        return jsonify(response), 200, headers
            
    except Exception as e:
        logger.exception(f"Unexpected error in authentication check: {str(e)}")
        # Try to reset the session on error
        try:
            from flask import session
            session.clear()
        except Exception as clear_error:
            logger.error(f"Error clearing session: {str(clear_error)}")
        
        return jsonify({
            'authenticated': False,
            'message': 'Session check failed',
            'debug_info': str(e) if current_app.debug else None
        }), 200
