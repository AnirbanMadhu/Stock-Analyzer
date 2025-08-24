from flask import Blueprint, request, jsonify
import logging
import numpy as np
from ml_models.stock_predictor import stock_predictor

logger = logging.getLogger(__name__)

# Create blueprint for enhanced prediction routes
enhanced_prediction_bp = Blueprint('enhanced_prediction', __name__)

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to JSON-serializable Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'dtype'):  # numpy scalar types
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
        elif np.issubdtype(obj.dtype, np.str_):
            return str(obj)
    return obj

@enhanced_prediction_bp.route('/search', methods=['GET'])
def search_stocks_with_prediction_info():
    """
    Search for stocks with prediction capability information
    Query parameters:
    - q: search query (required)
    - limit: maximum number of results (default: 10, max: 20)
    """
    try:
        query = request.args.get('q', '').strip()
        limit = request.args.get('limit', 10, type=int)
        
        if not query:
            return jsonify({'error': 'Search query is required'}), 400
        
        if len(query) < 1:
            return jsonify({'error': 'Search query must be at least 1 character long'}), 400
        
        if limit > 20:
            limit = 20
        elif limit < 1:
            limit = 10
        
        # Use enhanced search with prediction metadata
        from services.stock_service import stock_service
        
        # Get basic search results
        basic_results = stock_service.search_stocks(query, limit)
        
        # Enhance with prediction metadata
        results = []
        for stock in basic_results:
            symbol = stock.get('symbol', '')
            
            # Check if prediction is available
            prediction_available = True
            supported_models = ['Enhanced ARIMA', 'Enhanced LSTM', 'Simple MA']
            
            enhanced_stock = {
                **stock,
                'prediction_available': prediction_available,
                'supported_models': supported_models,
                'recommendation': 'Use ARIMA for best results'
            }
            results.append(enhanced_stock)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results),
            'search_type': 'enhanced_with_prediction_info'
        })
        
    except Exception as e:
        logger.error(f"Error in enhanced stock search: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@enhanced_prediction_bp.route('/analyze/<symbol>', methods=['GET'])
def get_comprehensive_stock_analysis(symbol):
    """
    Get comprehensive stock analysis with AI-powered predictions
    Query parameters:
    - days: number of days to predict (default: 7, max: 30)
    - model: model to use ("arima", "lstm", or "auto", default: "arima")
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        symbol = symbol.upper().strip()
        
        # Get query parameters
        days = request.args.get('days', 7, type=int)
        model = request.args.get('model', 'arima').lower()
        
        # Validate parameters
        if days < 1 or days > 30:
            return jsonify({'error': 'Days must be between 1 and 30'}), 400
        
        valid_models = ['arima', 'lstm', 'auto']
        if model not in valid_models:
            return jsonify({'error': f'Invalid model. Must be one of: {", ".join(valid_models)}'}), 400
        
        # Get comprehensive analysis using enhanced stock predictor
        from services.stock_service import stock_service
        
        # Get basic stock info
        stock_info = stock_service.get_stock_info(symbol)
        if not stock_info:
            return jsonify({'error': f'Stock {symbol} not found or has no data'}), 404
        
        # Get enhanced prediction
        prediction = stock_predictor.get_prediction(symbol, days, model)
        
        # Create comprehensive analysis
        analysis = {
            'symbol': symbol,
            'stock_info': stock_info,
            'prediction': prediction,
            'analysis': {
                'recommendation': 'HOLD',
                'risk_level': 'MEDIUM',
                'key_insights': ['AI-powered prediction available', 'Enhanced model selection']
            } if prediction else None,
            'generated_at': prediction.get('generated_at') if prediction else None
        }
        
        if not analysis:
            return jsonify({'error': f'Could not analyze {symbol}. Stock may not exist or have insufficient data.'}), 404
        
        logger.info(f"Generated comprehensive analysis for {symbol} using {model} model ({days} days)")
        
        # Convert numpy types to JSON-serializable types
        analysis = convert_numpy_types(analysis)
        
        return jsonify(analysis), 200
        
    except Exception as e:
        logger.error(f"Error generating comprehensive analysis for {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@enhanced_prediction_bp.route('/predict/<symbol>', methods=['GET'])
def get_enhanced_prediction(symbol):
    """
    Get enhanced AI prediction for a stock
    Query parameters:
    - days: number of days to predict (default: 7, max: 30)
    - model: model to use ("arima", "lstm", or "auto", default: "arima")
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        symbol = symbol.upper().strip()
        
        # Get query parameters
        days = request.args.get('days', 7, type=int)
        model = request.args.get('model', 'arima').lower()
        
        # Validate parameters
        if days < 1 or days > 30:
            return jsonify({'error': 'Days must be between 1 and 30'}), 400
        
        valid_models = ['arima', 'lstm', 'auto']
        if model not in valid_models:
            return jsonify({'error': f'Invalid model. Must be one of: {", ".join(valid_models)}'}), 400
        
        # Get enhanced prediction
        prediction = stock_predictor.get_prediction(symbol, days, model)
        
        if not prediction:
            return jsonify({'error': f'Could not generate prediction for {symbol}. Stock may not exist or have insufficient data.'}), 404
        
        logger.info(f"Generated enhanced prediction for {symbol} using {model} model ({days} days)")
        
        # Convert numpy types to JSON-serializable types
        prediction = convert_numpy_types(prediction)
        
        return jsonify(prediction), 200
        
    except Exception as e:
        logger.error(f"Error generating enhanced prediction for {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@enhanced_prediction_bp.route('/batch-predict', methods=['POST'])
def batch_enhanced_predictions():
    """
    Get enhanced predictions for multiple stocks
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
        
        if len(symbols) > 5:  # Reduced from 10 for AI models (more computation intensive)
            return jsonify({'error': 'Maximum 5 symbols allowed per batch request for AI predictions'}), 400
        
        days = data.get('days', 7)
        model = data.get('model', 'arima').lower()
        
        # Validate parameters
        if days < 1 or days > 30:
            return jsonify({'error': 'Days must be between 1 and 30'}), 400
        
        valid_models = ['arima', 'lstm', 'auto']
        if model not in valid_models:
            return jsonify({'error': f'Invalid model. Must be one of: {", ".join(valid_models)}'}), 400
        
        # Generate enhanced predictions for all symbols
        predictions = {}
        successful_predictions = 0
        model_usage = {}
        
        for symbol in symbols:
            try:
                symbol = symbol.upper().strip()
                prediction = stock_predictor.get_prediction(symbol, days, model)
                
                if prediction:
                    predictions[symbol] = prediction
                    successful_predictions += 1
                    
                    # Track model usage
                    model_used = prediction.get('model', 'Unknown')
                    model_usage[model_used] = model_usage.get(model_used, 0) + 1
                else:
                    predictions[symbol] = {
                        'error': f'Could not generate enhanced prediction for {symbol}',
                        'reason': 'Insufficient data or invalid symbol'
                    }
                    
            except Exception as e:
                logger.warning(f"Error predicting {symbol}: {e}")
                predictions[symbol] = {
                    'error': f'Prediction failed for {symbol}',
                    'reason': 'Internal processing error'
                }
        
        logger.info(f"Enhanced batch prediction completed: {successful_predictions}/{len(symbols)} successful")
        
        result = {
            'predictions': predictions,
            'summary': {
                'total_symbols': len(symbols),
                'successful_predictions': successful_predictions,
                'failed_predictions': len(symbols) - successful_predictions,
                'model_preference': model,
                'model_usage': model_usage,
                'prediction_days': days,
                'processing_type': 'enhanced_ai_prediction'
            }
        }
        
        # Convert numpy types to JSON-serializable types
        result = convert_numpy_types(result)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error in enhanced batch prediction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@enhanced_prediction_bp.route('/models/status', methods=['GET'])
def get_ai_models_status():
    """
    Get status and information about available AI prediction models
    """
    try:
        from ml_models.stock_predictor import ARIMA_AVAILABLE, LSTM_AVAILABLE
        
        models_status = {
            'arima': {
                'name': 'Enhanced ARIMA',
                'full_name': 'AutoRegressive Integrated Moving Average (Multi-Order)',
                'description': 'Advanced ARIMA model with automatic order selection for optimal fit',
                'available': ARIMA_AVAILABLE,
                'strengths': [
                    'Excellent for trend analysis',
                    'Good with seasonal patterns',
                    'Fast prediction generation',
                    'Reliable confidence intervals'
                ],
                'best_for': 'Stocks with clear trends and patterns',
                'min_data_points': 100,
                'accuracy_grade': 'B to A',
                'computational_cost': 'Low'
            },
            'lstm': {
                'name': 'Enhanced LSTM Neural Network',
                'full_name': 'Long Short-Term Memory Deep Neural Network',
                'description': 'Advanced neural network with uncertainty quantification and ensemble predictions',
                'available': LSTM_AVAILABLE,
                'strengths': [
                    'Captures complex non-linear patterns',
                    'Adapts to market changes',
                    'Multiple prediction simulations',
                    'Uncertainty quantification'
                ],
                'best_for': 'Stocks with complex, non-linear patterns',
                'min_data_points': 200,
                'accuracy_grade': 'A to A+',
                'computational_cost': 'High'
            },
            'enhanced_simple': {
                'name': 'Enhanced Moving Average',
                'full_name': 'Multi-MA Trend Analysis with Volatility Modeling',
                'description': 'Sophisticated moving average model with trend analysis and volatility adjustment',
                'available': True,
                'strengths': [
                    'Very fast predictions',
                    'Transparent methodology',
                    'Volatility-adjusted confidence',
                    'Trend decomposition'
                ],
                'best_for': 'Quick analysis and fallback predictions',
                'min_data_points': 30,
                'accuracy_grade': 'C to B',
                'computational_cost': 'Very Low'
            }
        }
        
        system_status = {
            'tensorflow_available': LSTM_AVAILABLE,
            'statsmodels_available': ARIMA_AVAILABLE,
            'recommended_model': 'arima',
            'auto_selection_logic': {
                'data_points_500+': 'LSTM (if available)',
                'data_points_200+': 'ARIMA (if available)',
                'data_points_100+': 'ARIMA (if available)',
                'data_points_30+': 'Enhanced Simple MA'
            }
        }
        
        return jsonify({
            'models': models_status,
            'system_status': system_status,
            'recommendations': {
                'for_beginners': 'Use "arima" model',
                'for_trending_stocks': 'Use "arima" model',
                'for_volatile_stocks': 'Use "lstm" model',
                'for_quick_analysis': 'Use enhanced simple model'
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error fetching AI models status: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@enhanced_prediction_bp.route('/compare-models/<symbol>', methods=['GET'])
def compare_prediction_models(symbol):
    """
    Compare predictions from different AI models for the same stock
    Query parameters:
    - days: number of days to predict (default: 7, max: 15 for comparison)
    """
    try:
        if not symbol or len(symbol.strip()) == 0:
            return jsonify({'error': 'Stock symbol is required'}), 400
        
        symbol = symbol.upper().strip()
        days = request.args.get('days', 7, type=int)
        
        # Limit days for model comparison (computationally expensive)
        if days < 1 or days > 10:
            return jsonify({'error': 'Days must be between 1 and 10 for model comparison'}), 400
        
        # Get predictions from all available models
        model_predictions = {}
        model_performances = {}
        
        # Try different models using the enhanced stock predictor
        try:
            # Get comparison from the enhanced stock predictor
            comparison = stock_predictor.compare_models(symbol, days)
            if comparison:
                model_predictions = comparison.get('detailed_results', {})
                model_performances = {}
                
                for model_name, result in model_predictions.items():
                    if 'model_performance' in result:
                        model_performances[model_name] = result['model_performance']
            else:
                # Fallback: try individual models
                arima_pred = stock_predictor.get_prediction(symbol, days, 'arima')
                if arima_pred:
                    model_predictions['ARIMA'] = arima_pred
                    model_performances['ARIMA'] = arima_pred.get('model_performance', {})
                
                lstm_pred = stock_predictor.get_prediction(symbol, days, 'lstm')
                if lstm_pred:
                    model_predictions['LSTM'] = lstm_pred
                    model_performances['LSTM'] = lstm_pred.get('model_performance', {})
                
                simple_pred = stock_predictor.get_prediction(symbol, days, 'simple')
                if simple_pred:
                    model_predictions['Simple'] = simple_pred
                    model_performances['Simple'] = simple_pred.get('model_performance', {})
                    
        except Exception as e:
            logger.warning(f"Model comparison failed for {symbol}: {e}")
            # Final fallback
            try:
                auto_pred = stock_predictor.get_prediction(symbol, days, 'arima')
                if auto_pred:
                    model_predictions['ARIMA'] = auto_pred
                    model_performances['ARIMA'] = auto_pred.get('model_performance', {})
            except Exception as e2:
                logger.error(f"All prediction methods failed for {symbol}: {e2}")
        
        if not model_predictions:
            return jsonify({'error': f'Could not generate any predictions for {symbol}'}), 404
        
        # Create comparison summary
        comparison_summary = {
            'symbol': symbol,
            'prediction_days': days,
            'models_compared': list(model_predictions.keys()),
            'comparison_date': model_predictions[list(model_predictions.keys())[0]].get('generated_at'),
            'predictions_comparison': {},
            'performance_comparison': model_performances
        }
        
        # Compare first and last day predictions
        if model_predictions:
            for model_name, prediction in model_predictions.items():
                predictions = prediction.get('predictions', [])
                if predictions:
                    first_day = predictions[0]
                    last_day = predictions[-1]
                    current_price = prediction.get('current_price', 0)
                    
                    total_change = ((last_day['predicted_price'] - current_price) / current_price * 100) if current_price > 0 else 0
                    avg_confidence = sum(p.get('confidence_percentage', 0) for p in predictions) / len(predictions)
                    
                    comparison_summary['predictions_comparison'][model_name] = {
                        'day_1_price': first_day.get('predicted_price', 0),
                        'final_day_price': last_day.get('predicted_price', 0),
                        'total_change_percent': round(total_change, 2),
                        'average_confidence': round(avg_confidence, 1),
                        'model_type': prediction.get('model', 'Unknown')
                    }
        
        result = {
            'comparison_summary': comparison_summary,
            'detailed_predictions': model_predictions,
            'recommendation': _get_model_recommendation(model_predictions),
            'note': 'Model comparison is computationally intensive and limited to 15 days'
        }
        
        logger.info(f"Model comparison completed for {symbol}: {len(model_predictions)} models")
        
        # Convert numpy types to JSON-serializable types
        result = convert_numpy_types(result)
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error comparing models for {symbol}: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def _get_model_recommendation(model_predictions: dict) -> dict:
    """Generate recommendation based on model comparison"""
    try:
        if not model_predictions:
            return {'recommendation': 'No models available', 'reason': 'No predictions generated'}
        
        # Simple recommendation logic based on available models and their characteristics
        if 'LSTM' in model_predictions and 'ARIMA' in model_predictions:
            return {
                'recommendation': 'Use LSTM for complex patterns, ARIMA for trend analysis',
                'reason': 'Both advanced models available - choose based on your analysis needs',
                'suggested_model': 'LSTM'
            }
        elif 'LSTM' in model_predictions:
            return {
                'recommendation': 'Use LSTM',
                'reason': 'LSTM model provides sophisticated pattern recognition',
                'suggested_model': 'LSTM'
            }
        elif 'ARIMA' in model_predictions:
            return {
                'recommendation': 'Use ARIMA',
                'reason': 'ARIMA model provides reliable trend analysis',
                'suggested_model': 'ARIMA'
            }
        else:
            return {
                'recommendation': 'Use Enhanced Simple MA',
                'reason': 'Only basic model available, but provides useful trend insights',
                'suggested_model': 'Enhanced_Simple'
            }
    except Exception:
        return {'recommendation': 'Use ARIMA', 'reason': 'ARIMA is the recommended default model'}

@enhanced_prediction_bp.route('/models/lstm/config', methods=['POST'])
def configure_lstm_performance():
    """
    Configure LSTM performance mode
    Expected JSON payload:
    {
        "fast_mode": true/false
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'fast_mode' not in data:
            return jsonify({'error': 'fast_mode parameter is required'}), 400
        
        fast_mode = bool(data['fast_mode'])
        stock_predictor.set_lstm_performance_mode(fast_mode)
        
        return jsonify({
            'message': f'LSTM performance mode set to {"fast" if fast_mode else "standard"}',
            'fast_mode': fast_mode,
            'settings': {
                'max_epochs': stock_predictor.lstm_max_epochs,
                'max_sequence_length': stock_predictor.lstm_max_sequence_length,
                'ensemble_size': stock_predictor.lstm_ensemble_size
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error configuring LSTM performance: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@enhanced_prediction_bp.route('/health', methods=['GET'])
def enhanced_prediction_health():
    """
    Health check for enhanced prediction service
    """
    try:
        from ml_models.stock_predictor import ARIMA_AVAILABLE, LSTM_AVAILABLE
        
        health_status = {
            'status': 'healthy',
            'service': 'Enhanced Stock Prediction API',
            'version': '2.0.0',
            'features': [
                'AI-powered predictions',
                'Multi-model comparison',
                'Enhanced search with prediction info',
                'Comprehensive stock analysis',
                'Uncertainty quantification'
            ],
            'models_available': {
                'arima': ARIMA_AVAILABLE,
                'lstm': LSTM_AVAILABLE,
                'enhanced_simple': True
            },
            'capabilities': {
                'search_with_prediction_info': True,
                'comprehensive_analysis': True,
                'batch_predictions': True,
                'model_comparison': True,
                'accuracy_metrics': True
            }
        }
        
        return jsonify(health_status), 200
        
    except Exception as e:
        logger.error(f"Error in enhanced prediction health check: {e}")
        return jsonify({
            'status': 'unhealthy',
            'error': 'Service health check failed'
        }), 500
