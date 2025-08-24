from flask import Blueprint, request, jsonify
import logging
from ml_models.stock_predictor import stock_predictor, ARIMA_AVAILABLE, LSTM_AVAILABLE

logger = logging.getLogger(__name__)

# Create blueprint for prediction routes
prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/stock', methods=['POST'])
def predict_stock_price_post():
    """
    Get stock price predictions via POST request
    Expected JSON payload:
    {
        "symbol": "AAPL",
        "days": 30,
        "model": "lstm"
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'details': 'Request body must be a JSON object with symbol, optional days (1-30), and optional model'
            }), 400
        
        if not isinstance(data, dict):
            return jsonify({
                'error': 'Invalid request format',
                'details': 'Request body must be a JSON object'
            }), 400
            
        # Get and validate symbol
        symbol = data.get('symbol', '')
        if not isinstance(symbol, str):
            return jsonify({
                'error': 'Invalid symbol format',
                'details': 'Symbol must be a string'
            }), 400
        
        symbol = symbol.strip().upper()
        if not symbol:
            return jsonify({
                'error': 'Stock symbol is required',
                'details': 'Please provide a valid stock symbol (e.g., AAPL, GOOGL)'
            }), 400
        
        # Get and validate days
        days = data.get('days', 7)
        if not isinstance(days, (int, float)):
            return jsonify({
                'error': 'Invalid days format',
                'details': 'Days must be a number between 1 and 30'
            }), 400
        
        days = int(days)
        if days < 1 or days > 30:
            return jsonify({
                'error': 'Invalid days value',
                'details': 'Days must be between 1 and 30'
            }), 400
        
        # Get and validate model
        model = data.get('model', 'auto')
        if not isinstance(model, str):
            return jsonify({
                'error': 'Invalid model format',
                'details': 'Model must be a string'
            }), 400
        
        model = model.lower()
        valid_models = ['arima', 'lstm', 'hybrid', 'auto']
        if model not in valid_models:
            return jsonify({
                'error': 'Invalid model',
                'details': f'Model must be one of: {", ".join(valid_models)}'
            }), 400        # Get prediction
        logger.info(f"Requesting {model} prediction for {symbol} ({days} days)")
        prediction = stock_predictor.get_prediction(symbol, days, model)
        
        if not prediction:
            # Check if it's a model availability issue
            if not ARIMA_AVAILABLE and model in ['arima', 'hybrid']:
                return jsonify({
                    'error': 'Model unavailable',
                    'details': 'ARIMA model is not available. Please install statsmodels package or use a different model.'
                }), 503  # Service Unavailable
            elif not LSTM_AVAILABLE and model in ['lstm', 'hybrid']:
                return jsonify({
                    'error': 'Model unavailable',
                    'details': 'LSTM model is not available. Please install TensorFlow package or use a different model.'
                }), 503
            else:
                return jsonify({
                    'error': 'Prediction failed',
                    'details': f'Could not generate prediction for {symbol}. This may be due to insufficient historical data or invalid stock symbol.'
                }), 404
        
        logger.info(f"Successfully generated {model} prediction for {symbol} ({days} days)")
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'model_used': prediction.get('model', model),
            'timestamp': prediction.get('generated_at')
        }), 200
        
    except ValueError as e:
        logger.error(f"Validation error for {symbol}: {e}")
        return jsonify({
            'error': 'Invalid input',
            'details': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {e}")
        return jsonify({
            'error': 'Internal server error',
            'details': f'An unexpected error occurred while processing your request. Error: {str(e)}'
        }), 500

@prediction_bp.route('/<symbol>', methods=['GET'])
def predict_stock_price(symbol):
    """
    Get stock price predictions for a given symbol with ARIMA(1,1,1) and LSTM models
    
    This endpoint provides 7-day forecasts with 95% confidence intervals using:
    - ARIMA(1,1,1) model trained on 1-year historical data
    - LSTM neural network for non-linear pattern recognition
    - Automatic model selection based on data characteristics
    
    Query parameters:
    - days: number of days to predict (default: 7, max: 30)
    - model: model to use ("arima", "lstm", "hybrid", or "auto", default: "arima")
    - confidence: confidence level for intervals (default: 95)
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        symbol = symbol.upper().strip()
        
        # Get query parameters with ARIMA as default per specifications
        days = request.args.get('days', 7, type=int)
        model = request.args.get('model', 'arima').lower()  # Default to ARIMA per spec
        confidence_level = request.args.get('confidence', 95, type=int)
        
        # Validate parameters
        if days < 1 or days > 30:
            return jsonify({'error': 'Days must be between 1 and 30'}), 400
        
        valid_models = ['arima', 'lstm', 'hybrid', 'auto']
        if model not in valid_models:
            return jsonify({'error': f'Invalid model. Must be one of: {", ".join(valid_models)}'}), 400
        
        if confidence_level < 80 or confidence_level > 99:
            return jsonify({'error': 'Confidence level must be between 80 and 99'}), 400
        
        # Get prediction with specified confidence level
        prediction = stock_predictor.get_prediction(symbol, days, model)
        
        if not prediction:
            return jsonify({'error': f'Could not generate prediction for {symbol}'}), 404
        
        # Add confidence level metadata
        prediction['confidence_level_requested'] = confidence_level
        prediction['model_training_period'] = '1 year historical data'
        prediction['forecast_horizon'] = f'{days} days'
        prediction['primary_model'] = 'ARIMA(1,1,1)' if 'arima' in prediction.get('model', '').lower() else prediction.get('model')
        
        logger.info(f"Generated {model} prediction for {symbol} ({days} days, {confidence_level}% confidence)")
        
        return jsonify(prediction), 200
        
    except Exception as e:
        logger.error(f"Error generating prediction for {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@prediction_bp.route('/models', methods=['GET'])
def get_available_models():
    """
    Get information about available prediction models
    """
    try:
        from ml_models.stock_predictor import ARIMA_AVAILABLE, LSTM_AVAILABLE
        
        models = {
            'arima': {
                'name': 'ARIMA(1,1,1)',
                'description': 'AutoRegressive Integrated Moving Average model for time series forecasting',
                'available': ARIMA_AVAILABLE,
                'accuracy': 'Good for linear trends and seasonal patterns',
                'speed': 'Fast'
            },
            'lstm': {
                'name': 'LSTM Neural Network',
                'description': 'Long Short-Term Memory neural network for non-linear pattern recognition',
                'available': LSTM_AVAILABLE,
                'accuracy': 'Good for complex patterns and non-linear trends',
                'speed': 'Slower (requires training)'
            },
            'simple': {
                'name': 'Simple Moving Average',
                'description': 'Fallback model using moving averages and trend analysis',
                'available': True,
                'accuracy': 'Basic trend following',
                'speed': 'Very fast'
            }
        }
        
        return jsonify({
            'available_models': models,
            'default_model': 'arima',
            'recommendation': 'Use "arima" for reliable stock price predictions'
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching model information: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@prediction_bp.route('/batch', methods=['POST'])
def predict_multiple_stocks():
    """
    Get predictions for multiple stocks
    Expected JSON payload:
    {
        "symbols": ["AAPL", "GOOGL", "MSFT"],
        "days": 7,
        "model": "auto"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'symbols' not in data:
            return jsonify({'error': 'Symbols array is required'}), 400
        
        symbols = data['symbols']
        if not isinstance(symbols, list) or len(symbols) == 0:
            return jsonify({'error': 'At least one symbol is required'}), 400
        
        if len(symbols) > 10:
            return jsonify({'error': 'Maximum 10 symbols allowed per batch request'}), 400
        
        days = data.get('days', 7)
        model = data.get('model', 'arima').lower()
        
        # Validate days
        if not isinstance(days, int):
            return jsonify({
                'error': 'Invalid days format',
                'details': 'Days must be a number between 1 and 30'
            }), 400
            
        if days < 1 or days > 30:
            return jsonify({
                'error': 'Invalid days value',
                'details': 'Days must be between 1 and 30'
            }), 400
        
        # Validate model
        if not isinstance(model, str):
            return jsonify({
                'error': 'Invalid model format',
                'details': 'Model must be a string'
            }), 400
            
        valid_models = ['arima', 'lstm', 'hybrid', 'auto']
        if model not in valid_models:
            return jsonify({
                'error': 'Invalid model',
                'details': f'Model must be one of: {", ".join(valid_models)}'
            }), 400
        
        # Generate predictions for all symbols
        predictions = {}
        successful_predictions = 0
        
        for symbol in symbols:
            try:
                symbol = symbol.upper().strip()
                prediction = stock_predictor.get_prediction(symbol, days, model)
                
                if prediction:
                    predictions[symbol] = prediction
                    successful_predictions += 1
                else:
                    predictions[symbol] = {
                        'error': f'Could not generate prediction for {symbol}'
                    }
                    
            except Exception as e:
                logger.warning(f"Error predicting {symbol}: {e}")
                predictions[symbol] = {
                    'error': f'Prediction failed for {symbol}'
                }
        
        logger.info(f"Batch prediction completed: {successful_predictions}/{len(symbols)} successful")
        
        return jsonify({
            'predictions': predictions,
            'summary': {
                'total_symbols': len(symbols),
                'successful_predictions': successful_predictions,
                'failed_predictions': len(symbols) - successful_predictions,
                'model_used': model,
                'prediction_days': days
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@prediction_bp.route('/accuracy/<symbol>', methods=['GET'])
def get_prediction_accuracy(symbol):
    """
    Get historical accuracy information for predictions of a specific stock
    Query parameters:
    - model: model to check ("arima", "lstm", default: "auto")
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        symbol = symbol.upper().strip()
        model = request.args.get('model', 'auto').lower()
        
        # Generate a prediction to get accuracy metrics
        prediction = stock_predictor.get_prediction(symbol, 7, model)
        
        if not prediction:
            return jsonify({'error': f'Could not analyze accuracy for {symbol}'}), 404
        
        accuracy_info = {
            'symbol': symbol,
            'model_used': prediction.get('model', 'Unknown'),
            'accuracy_metrics': prediction.get('accuracy_metrics'),
            'last_updated': prediction.get('generated_at'),
            'note': 'Accuracy metrics are based on backtesting with recent historical data'
        }
        
        return jsonify(accuracy_info), 200
        
    except Exception as e:
        logger.error(f"Error getting accuracy for {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500
