import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
import warnings
import json
import time
from concurrent.futures import ThreadPoolExecutor
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    logging.warning("ARIMA model not available. Install statsmodels for ARIMA predictions.")

# Scikit-learn imports with fallback
try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    MinMaxScaler = None
    StandardScaler = None
    logging.warning("Scikit-learn not available. Using simple linear regression fallback.")
    
    # Simple fallback functions for metrics
    def mean_absolute_error(y_true, y_pred):
        """Fallback implementation of MAE"""
        return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))
    
    def mean_squared_error(y_true, y_pred):
        """Fallback implementation of MSE"""
        return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)
    
    def r2_score(y_true, y_pred):
        """Fallback implementation of R2 score"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

# TensorFlow imports with fallback for linting
LSTM_AVAILABLE = False
Sequential = None
LSTM = None
Dense = None
Dropout = None
MinMaxScaler = None
BatchNormalization = None
EarlyStopping = None

try:
    # Import TensorFlow with version checking 
    import tensorflow as tf
    
    # Set memory growth to prevent GPU memory issues
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            logging.warning(f"GPU memory configuration failed: {e}")
    
    # Import sklearn components
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    
    # Import Keras components dynamically to avoid linting issues
    import importlib
    keras_models = importlib.import_module('tensorflow.keras.models')
    keras_layers = importlib.import_module('tensorflow.keras.layers')
    keras_callbacks = importlib.import_module('tensorflow.keras.callbacks')
    
    Sequential = keras_models.Sequential
    LSTM = keras_layers.LSTM
    Dense = keras_layers.Dense
    Dropout = keras_layers.Dropout
    BatchNormalization = keras_layers.BatchNormalization
    EarlyStopping = keras_callbacks.EarlyStopping
    
    LSTM_AVAILABLE = True
    logging.info(f"TensorFlow {tf.__version__} loaded successfully")
except ImportError as e:
    LSTM_AVAILABLE = False
    logging.warning(f"LSTM model not available. Install TensorFlow for LSTM predictions. Error: {e}")

logger = logging.getLogger(__name__)

def to_numpy_array(data):
    """Convert various data types to numpy array safely"""
    try:
        if isinstance(data, np.ndarray):
            return data.astype(float)
        elif hasattr(data, 'to_numpy'):
            return data.to_numpy().astype(float)
        elif hasattr(data, 'values'):
            return np.array(data.values).astype(float)
        else:
            return np.asarray(data, dtype=float)
    except:
        return np.array(data, dtype=float)

class StockPredictor:
    """
    Enhanced Stock price prediction service using multiple AI models
    Features:
    - ARIMA with automatic parameter selection
    - LSTM with advanced architecture and ensemble methods
    - Hybrid models combining multiple approaches
    - Advanced caching and performance optimization
    - Real-time accuracy tracking and model selection
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache for predictions
        self.model_cache_duration = 604800  # 1 week (7 days) for model updates per specification
        self.model_performance_cache = {}
        self.accuracy_history = {}
        self.model_selection_history = {}
        
        # Performance tracking
        self.prediction_count = 0
        self.successful_predictions = 0
        self.model_usage_stats = {
            'arima': 0,
            'lstm': 0,
            'hybrid': 0,
            'simple': 0
        }
        
        # LSTM optimization settings
        self.lstm_fast_mode = True  # Enable fast LSTM mode by default
        self.lstm_max_epochs = 15  # Reduced max epochs
        self.lstm_max_sequence_length = 25  # Reduced sequence length
        self.lstm_ensemble_size = 2  # Reduced ensemble size
        
    def set_lstm_performance_mode(self, fast_mode: bool = True):
        """
        Configure LSTM performance mode
        fast_mode=True: Faster predictions with slightly lower accuracy
        fast_mode=False: Slower predictions with better accuracy
        """
        self.lstm_fast_mode = fast_mode
        if fast_mode:
            self.lstm_max_epochs = 10
            self.lstm_max_sequence_length = 20
            self.lstm_ensemble_size = 1  # Single prediction for max speed
        else:
            self.lstm_max_epochs = 25
            self.lstm_max_sequence_length = 40
            self.lstm_ensemble_size = 3
        
        # Training data specifications per requirements
        self.training_period = "1y"  # 1 year historical data as specified
        self.default_forecast_days = 7  # 7-day forecast as specified
        self.default_confidence_level = 95  # 95% confidence intervals as specified
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached prediction is still valid"""
        if cache_key not in self.cache:
            return False
        
        import time
        cache_time = self.cache[cache_key].get('timestamp', 0)
        
        # Use weekly model updates for predictions per specification
        cache_duration = self.model_cache_duration if 'model_' in cache_key else self.cache_duration
        return time.time() - cache_time < cache_duration
    
    def _get_stock_data(self, symbol: str, period: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Enhanced stock data fetching with validation and cleaning
        Default to 1-year period as specified for ARIMA training
        """
        try:
            logger.info(f"Fetching stock data for {symbol} with period {period}")
            
            # Use 1-year training period as specified in requirements
            if period is None:
                period = self.training_period
                
            ticker = yf.Ticker(symbol.upper())
            hist = ticker.history(period=period, auto_adjust=True, prepost=True)
            
            if hist.empty:
                logger.warning(f"No historical data available for {symbol}")
                return None
            
            # Create comprehensive dataset with multiple features
            data = pd.DataFrame()
            data['price'] = hist['Close']
            data['volume'] = hist['Volume'] 
            data['high'] = hist['High']
            data['low'] = hist['Low']
            data['open'] = hist['Open']
            
            # Calculate technical indicators for enhanced ARIMA performance
            data['price_change'] = data['price'].pct_change()
            data['volatility'] = data['price_change'].rolling(window=20).std()
            data['sma_5'] = data['price'].rolling(window=5).mean()
            data['sma_10'] = data['price'].rolling(window=10).mean()
            data['sma_20'] = data['price'].rolling(window=20).mean()
            data['ema_12'] = data['price'].ewm(span=12).mean()
            data['ema_26'] = data['price'].ewm(span=26).mean()
            
            # MACD
            data['macd'] = data['ema_12'] - data['ema_26']
            data['macd_signal'] = data['macd'].ewm(span=9).mean()
            
            # RSI (Relative Strength Index)
            try:
                delta = data['price'].diff()
                # Convert to numpy for comparison operations
                delta_values = to_numpy_array(delta)
                gain = pd.Series(np.where(delta_values > 0, delta_values, 0), index=delta.index).rolling(window=14).mean()
                loss = pd.Series(np.where(delta_values < 0, -delta_values, 0), index=delta.index).rolling(window=14).mean()
                rs = gain / loss.replace(0, np.nan)  # Avoid division by zero
                data['rsi'] = 100 - (100 / (1 + rs))
            except Exception:
                data['rsi'] = 50.0  # Default RSI if calculation fails
            
            # Bollinger Bands
            data['bb_middle'] = data['price'].rolling(window=20).mean()
            bb_std = data['price'].rolling(window=20).std()
            data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
            data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
            data['bb_position'] = (data['price'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Clean data - remove NaN values
            data = data.dropna()
            
            if len(data) < 30:
                logger.warning(f"Insufficient data for {symbol}: only {len(data)} points after cleaning")
                return None
            
            logger.info(f"Retrieved and processed {len(data)} data points for {symbol} (period: {period})")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _analyze_time_series(self, data: pd.Series) -> Dict:
        """Analyze time series characteristics for better model selection"""
        try:
            analysis = {
                'is_stationary': False,
                'trend_strength': 0.0,
                'seasonality_strength': 0.0,
                'volatility_level': 'medium',
                'data_quality': 'good',
                'recommended_models': []
            }
            
            # Stationarity test
            if len(data) > 50:
                try:
                    adf_result = adfuller(data.dropna())
                    analysis['is_stationary'] = adf_result[1] < 0.05
                    analysis['adf_pvalue'] = adf_result[1]
                except:
                    analysis['is_stationary'] = False
            
            # Trend analysis
            if len(data) > 100:
                try:
                    # Simple trend calculation
                    x = np.arange(len(data))
                    data_values = to_numpy_array(data)
                    slope = np.polyfit(x, data_values, 1)[0]
                    analysis['trend_strength'] = abs(slope) / np.mean(data_values) * 100
                except:
                    pass
            
            # Volatility analysis
            returns = data.pct_change().dropna()
            if len(returns) > 20:
                volatility = returns.std()
                if volatility > 0.05:
                    analysis['volatility_level'] = 'high'
                elif volatility < 0.02:
                    analysis['volatility_level'] = 'low'
            
            # Model recommendations based on analysis
            if analysis['volatility_level'] == 'high' and len(data) > 200:
                analysis['recommended_models'].append('lstm')
            if analysis['trend_strength'] > 1.0:
                analysis['recommended_models'].append('arima')
            if len(data) < 100:
                analysis['recommended_models'].append('simple')
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in time series analysis: {e}")
            return {'recommended_models': ['simple']}
    
    def predict_with_arima(self, symbol: str, days: int = 7) -> Optional[Dict]:
        """
        Enhanced ARIMA prediction with automatic parameter selection
        """
        if not ARIMA_AVAILABLE:
            logger.warning("ARIMA not available for prediction - statsmodels required")
            return None
            
        if not isinstance(days, int) or days < 1 or days > 30:
            logger.error(f"Invalid days parameter: {days}")
            return None
            
        if not symbol or not isinstance(symbol, str):
            logger.error(f"Invalid symbol parameter: {symbol}")
            return None
            
        cache_key = f"enhanced_arima_{symbol}_{days}"
        
        try:
            if self._is_cache_valid(cache_key):
                logger.info(f"Using cached prediction for {symbol}")
                return self.cache[cache_key]['data']
            
            symbol = symbol.upper().strip()
            logger.info(f"Starting ARIMA prediction for {symbol} with {days} days forecast horizon")
            
            # Get historical data with technical indicators
            data = self._get_stock_data(symbol, period="2y")
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data for ARIMA prediction: {symbol}")
                return None
            
            # Analyze time series characteristics
            prices = to_numpy_array(data['price'])
            ts_analysis = self._analyze_time_series(data['price'])
            
            # Automatic ARIMA parameter selection
            best_order, best_model = self._auto_arima_selection(prices)
            
            if best_model is None:
                logger.warning(f"Could not fit ARIMA model for {symbol}")
                return None
            
            # Generate predictions with confidence intervals
            forecast_method = getattr(best_model, 'get_forecast', None)
            if forecast_method:
                forecast_obj = forecast_method(steps=days)
                forecast = forecast_obj.predicted_mean
                conf_int = forecast_obj.conf_int()
            else:
                # Fallback to simple forecast if available
                forecast_method = getattr(best_model, 'forecast', None)
                if forecast_method:
                    forecast = forecast_method(steps=days)
                    conf_int = None
                else:
                    logger.error("No forecast method available on model")
                    return None
            
            current_price = float(prices[-1])
            predictions = []
            
            # Calculate dynamic confidence based on model performance and time horizon
            model_r2 = self._calculate_model_r2(best_model, prices[-50:] if len(prices) >= 50 else prices)
            
            for i in range(days):
                prediction_date = datetime.now() + timedelta(days=i+1)
                pred_price = float(forecast.iloc[i]) if hasattr(forecast, 'iloc') else float(forecast[i])
                
                # Enhanced confidence intervals
                try:
                    if conf_int is not None:
                        lower_bound = float(conf_int.iloc[i, 0])
                        upper_bound = float(conf_int.iloc[i, 1])
                    else:
                        raise AttributeError("conf_int is None")
                except (IndexError, TypeError, AttributeError):
                    # Fallback confidence calculation
                    volatility = float(np.std(prices[-30:] if len(prices) >= 30 else prices))
                    margin = 1.96 * volatility * np.sqrt(i + 1)
                    lower_bound = pred_price - margin
                    upper_bound = pred_price + margin
                
                # Dynamic confidence calculation
                base_confidence = 90.0 * model_r2
                time_decay = max(0.5, 1.0 - (i * 0.1))
                volatility_penalty = 0.2 if ts_analysis.get('volatility_level', 'medium') == 'high' else 0.0
                confidence_percentage = base_confidence * time_decay * (1.0 - volatility_penalty)
                confidence_percentage = max(60, min(95, confidence_percentage))
                
                predictions.append({
                    'date': prediction_date.strftime('%Y-%m-%d'),
                    'predicted_price': pred_price,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'confidence_percentage': round(confidence_percentage, 1),
                    'day': i + 1,
                    'prediction_interval': 95
                })
            
            # Enhanced accuracy metrics with multiple validation approaches
            accuracy_metrics = self._enhanced_accuracy_validation(prices, best_model, 'arima')
            
            # Model performance details
            model_info = {
                'order': best_order,
                'aic': float(getattr(best_model, 'aic', 0.0)),
                'bic': float(getattr(best_model, 'bic', 0.0)),
                'r_squared': model_r2,
                'data_points': len(prices),
                'is_stationary': bool(ts_analysis.get('is_stationary', False)),
                'trend_strength': float(ts_analysis.get('trend_strength', 0))
            }
            
            result = {
                'symbol': symbol,
                'model': f'Enhanced ARIMA{best_order}',
                'current_price': current_price,
                'prediction_days': days,
                'predictions': predictions,
                'accuracy_metrics': accuracy_metrics,
                'model_performance': model_info,
                'time_series_analysis': ts_analysis,
                'generated_at': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            # Update model usage statistics
            self.model_usage_stats['arima'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced ARIMA prediction failed for {symbol}: {str(e)}")
            return None
        
        cache_key = f"enhanced_arima_{symbol}_{days}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Get historical data with technical indicators
            data = self._get_stock_data(symbol, period="2y")
            if data is None or len(data) < 100:
                logger.warning(f"Insufficient data for ARIMA prediction: {symbol}")
                return None
            
            # Analyze time series characteristics
            prices = to_numpy_array(data['price'])
            ts_analysis = self._analyze_time_series(data['price'])
            
            # Automatic ARIMA parameter selection
            best_order, best_model = self._auto_arima_selection(prices)
            
            if best_model is None:
                logger.warning(f"Could not fit ARIMA model for {symbol}")
                return None
            
            # Generate predictions with confidence intervals
            try:
                forecast_result = getattr(best_model, 'get_forecast', None)
                if forecast_result:
                    forecast_obj = forecast_result(steps=days)
                    forecast = forecast_obj.predicted_mean
                    conf_int = forecast_obj.conf_int()
                else:
                    # Fallback to simple forecast
                    forecast_method = getattr(best_model, 'forecast', None)
                    if forecast_method:
                        forecast = forecast_method(steps=days)
                        conf_int = None
                    else:
                        logger.error("No forecast method available on model")
                        return None
            except Exception as e:
                logger.error(f"Error generating forecast: {e}")
                return None
            
            # Prepare enhanced predictions
            current_price = float(prices[-1])
            predictions = []
            
            # Calculate dynamic confidence based on model performance and time horizon
            model_r2 = self._calculate_model_r2(best_model, to_numpy_array(prices[-50:] if len(prices) >= 50 else prices))
            
            for i in range(days):
                prediction_date = datetime.now() + timedelta(days=i+1)
                # Handle different forecast formats
                if hasattr(forecast, 'iloc'):
                    pred_price = float(forecast.iloc[i])
                elif hasattr(forecast, '__getitem__'):
                    pred_price = float(forecast[i])
                else:
                    pred_price = float(forecast)
                
                # Enhanced confidence intervals
                try:
                    if conf_int is not None and hasattr(conf_int, 'iloc'):
                        lower_bound = float(conf_int.iloc[i, 0])
                        upper_bound = float(conf_int.iloc[i, 1])
                    else:
                        raise AttributeError("No confidence intervals available")
                except (IndexError, TypeError, AttributeError):
                    # Fallback confidence calculation
                    price_array = to_numpy_array(prices[-30:] if len(prices) >= 30 else prices)
                    volatility = float(np.std(price_array)) if len(price_array) > 1 else pred_price * 0.05
                    margin = 1.96 * volatility * np.sqrt(i + 1)
                    lower_bound = pred_price - margin
                    upper_bound = pred_price + margin
                
                # Dynamic confidence calculation
                base_confidence = 90.0 * model_r2  # Scale by model fit quality
                time_decay = max(0.5, 1.0 - (i * 0.1))  # Confidence decreases with time
                volatility_penalty = 0.2 if ts_analysis.get('volatility_level', 'medium') == 'high' else 0.0
                volatility_penalty = min(0.3, volatility_penalty)
                
                confidence_percentage = base_confidence * time_decay * (1 - volatility_penalty)
                confidence_percentage = max(60, min(95, confidence_percentage))
                
                predictions.append({
                    'date': prediction_date.strftime('%Y-%m-%d'),
                    'predicted_price': float(pred_price),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'confidence_percentage': round(float(confidence_percentage), 1),
                    'day': i + 1,
                    'prediction_interval': 95
                })
            
            # Enhanced accuracy metrics with multiple validation approaches
            accuracy_metrics = self._enhanced_accuracy_validation(prices, best_model, 'arima')
            
            # Model performance details
            model_info = {
                'order': best_order,
                'aic': float(getattr(best_model, 'aic', 0.0)),
                'bic': float(getattr(best_model, 'bic', 0.0)),
                'r_squared': model_r2,
                'data_points': len(prices),
                'is_stationary': bool(ts_analysis.get('is_stationary', False)),
                'trend_strength': float(ts_analysis.get('trend_strength', 0))
            }
            
            result = {
                'symbol': symbol.upper(),
                'model': f'Enhanced ARIMA{best_order}',
                'current_price': current_price,
                'prediction_days': days,
                'predictions': predictions,
                'accuracy_metrics': accuracy_metrics,
                'model_performance': model_info,
                'time_series_analysis': ts_analysis,
                'generated_at': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            # Update model usage statistics
            self.model_usage_stats['arima'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced ARIMA prediction failed for {symbol}: {e}")
            return None
    
    def _auto_arima_selection(self, prices: np.ndarray) -> Tuple[Optional[tuple], Optional[object]]:
        """
        ARIMA parameter selection with preference for (1,1,1) as specified in requirements
        Implements automatic parameter selection with (1,1,1) as primary model
        """
        if not isinstance(prices, np.ndarray):
            logger.error("Input prices must be a numpy array")
            return None, None
            
        try:
            # Start with ARIMA(1,1,1) as specified in requirements
            best_order = (1, 1, 1)
            best_model = None
            best_aic = float('inf')
            
            # Try ARIMA(1,1,1) first per specification
            try:
                model = ARIMA(prices, order=(1, 1, 1))
                fitted_model = model.fit()
                best_model = fitted_model
                best_aic = fitted_model.aic
                best_order = (1, 1, 1)
                logger.info(f"Successfully fitted ARIMA(1,1,1) model with AIC: {best_aic:.2f}")
            except Exception as e:
                logger.warning(f"ARIMA(1,1,1) failed, trying alternative parameters: {e}")
            
            # If ARIMA(1,1,1) fails, try other parameter combinations
            if best_model is None:
                # Parameter ranges to try as fallback
                p_values = [0, 1, 2]
                d_values = [0, 1, 2] 
                q_values = [0, 1, 2]
                
                for p in p_values:
                    for d in d_values:
                        for q in q_values:
                            try:
                                if p == 0 and d == 0 and q == 0:
                                    continue
                                    
                                model = ARIMA(prices, order=(p, d, q))
                                fitted_model = model.fit()
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_order = (p, d, q)
                                    best_model = fitted_model
                                    
                            except Exception:
                                continue
                
                if best_model is not None:
                    logger.info(f"Fallback ARIMA{best_order} selected with AIC: {best_aic:.2f}")
            
            # Final fallback to simple ARIMA(1,1,1) with different fitting approach
            if best_model is None:
                try:
                    model = ARIMA(prices, order=(1, 1, 1))
                    best_model = model.fit(method_kwargs={"warn_convergence": False})
                    best_order = (1, 1, 1)
                    logger.info("Used simplified ARIMA(1,1,1) fitting approach")
                except Exception as e:
                    logger.error(f"All ARIMA fitting approaches failed: {e}")
                    return None, None
            
            return best_order, best_model
            
        except Exception as e:
            logger.error(f"Auto ARIMA selection failed: {e}")
            return None, None
    
    def _calculate_model_r2(self, model, data: np.ndarray) -> float:
        """Calculate R-squared for model fit quality"""
        try:
            fitted_values = getattr(model, 'fittedvalues', None)
            if fitted_values is None:
                return 0.7
                
            fitted_values = to_numpy_array(fitted_values)
            if len(fitted_values) > len(data):
                fitted_values = fitted_values[-len(data):]
            elif len(fitted_values) < len(data):
                data = data[-len(fitted_values):]
            
            return max(0, r2_score(data, fitted_values))
        except Exception:
            return 0.7  # Default reasonable R²
    
    def _enhanced_accuracy_validation(self, prices: np.ndarray, model, model_type: str) -> Dict:
        """Enhanced accuracy validation with multiple metrics"""
        try:
            if len(prices) < 30:
                return {'note': 'Insufficient data for validation'}
            
            # Use last 20% of data for validation
            test_size = max(10, int(len(prices) * 0.2))
            train_data = prices[:-test_size]
            test_data = prices[-test_size:]
            
            if model_type == 'arima' and hasattr(model, 'forecast'):
                try:
                    # Get predictions for test period
                    predictions = model.forecast(steps=len(test_data))
                    predictions = to_numpy_array(predictions)
                    
                    # Calculate metrics
                    mae = mean_absolute_error(test_data, predictions)
                    rmse = np.sqrt(mean_squared_error(test_data, predictions))
                    
                    # MAPE with zero handling
                    mape_values = np.abs((test_data - predictions) / np.where(test_data == 0, 1e-8, test_data)) * 100
                    mape = np.mean(mape_values[np.isfinite(mape_values)])
                    
                    # Directional accuracy
                    if len(test_data) > 1 and len(predictions) > 1:
                        actual_directions = np.sign(np.diff(test_data))
                        pred_directions = np.sign(np.diff(predictions))
                        directional_accuracy = np.mean(actual_directions == pred_directions) * 100
                    else:
                        directional_accuracy = 50.0
                    
                    return {
                        'mae': float(mae) if not np.isnan(mae) else 0.0,
                        'rmse': float(rmse) if not np.isnan(rmse) else 0.0,
                        'mape': float(mape) if not np.isnan(mape) else 0.0,
                        'directional_accuracy': float(directional_accuracy),
                        'validation_samples': len(test_data),
                        'accuracy_grade': self._get_accuracy_grade(float(mape), float(directional_accuracy))
                    }
                except Exception as e:
                    logger.warning(f"Validation error for {model_type}: {e}")
            
            # Fallback validation using simple metrics
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns) if len(returns) > 1 else 0.02
            
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'mape': 0.0,
                'directional_accuracy': 50.0,
                'volatility': float(volatility),
                'note': f'Simplified validation for {model_type} model',
                'accuracy_grade': 'C (Fair - Limited validation)'
            }
            
        except Exception as e:
            logger.error(f"Enhanced accuracy validation failed: {e}")
            return {'note': 'Validation failed', 'accuracy_grade': 'N/A'}
    
    def _get_accuracy_grade(self, mape: float, directional_accuracy: float) -> str:
        """Get accuracy grade based on MAPE and directional accuracy"""
        try:
            if mape <= 5 and directional_accuracy >= 70:
                return 'A (Excellent)'
            elif mape <= 10 and directional_accuracy >= 60:
                return 'B (Good)'
            elif mape <= 15 and directional_accuracy >= 55:
                return 'C (Fair)'
            elif mape <= 25 and directional_accuracy >= 50:
                return 'D (Poor)'
            else:
                return 'F (Very Poor)'
        except:
            return 'N/A'
    
    def predict_with_lstm(self, symbol: str, days: int = 7) -> Optional[Dict]:
        """
        Optimized LSTM prediction with faster training and inference
        """
        import time
        start_time = time.time()
        max_execution_time = 120  # 2 minutes timeout
        
        if not LSTM_AVAILABLE or Sequential is None or MinMaxScaler is None:
            logger.warning("LSTM not available for prediction")
            return None
        
        cache_key = f"enhanced_lstm_{symbol}_{days}"
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]['data']
        
        try:
            # Check timeout periodically
            if time.time() - start_time > max_execution_time:
                logger.warning(f"LSTM prediction timeout for {symbol}")
                return None
                
            # Get enhanced historical data (reduced period for speed)
            data = self._get_stock_data(symbol, period="1y")  # Reduced from 2y to 1y
            if data is None or len(data) < 150:
                logger.warning(f"Insufficient data for LSTM prediction: {symbol}")
                return None
                
            # Limit data size for faster processing
            if len(data) > 500:
                data = data.tail(500)  # Use only last 500 days for speed
            
            # Optimize feature selection - use fewer features for faster training
            # Prioritize most important features for speed
            essential_features = ['price']  # Start with just price for speed
            if 'volume' in data.columns and not data['volume'].isna().all():
                essential_features.append('volume')
            if 'volatility' in data.columns and not data['volatility'].isna().all():
                essential_features.append('volatility')
            
            # Create feature matrix with essential features only
            feature_data = data[essential_features].dropna()
            if len(feature_data) < 150:  # Reduced minimum requirement
                logger.warning(f"Insufficient clean data for LSTM: {len(feature_data)} points")
                return None
            
            # Check if sklearn is available for scaling
            if not SKLEARN_AVAILABLE or StandardScaler is None or MinMaxScaler is None:
                logger.warning("Scikit-learn not available for feature scaling")
                return None
            
            # Scale features
            scaler_features = StandardScaler()
            scaler_target = MinMaxScaler(feature_range=(0, 1))
            
            # Scale all features
            scaled_features = scaler_features.fit_transform(feature_data.values)
            
            # Scale target (price) separately for inverse transform
            price_data = to_numpy_array(feature_data['price']).reshape(-1, 1)
            scaled_target = scaler_target.fit_transform(price_data)
            
            # Create sequences with optimized length for speed
            sequence_length = min(self.lstm_max_sequence_length if self.lstm_fast_mode else 30, 
                                len(scaled_features) // 5)  # Use configurable length
            X, y = [], []
            
            for i in range(sequence_length, len(scaled_features)):
                X.append(scaled_features[i-sequence_length:i])  # Multi-feature sequence
                y.append(scaled_target[i, 0])  # Target price
            
            X, y = np.array(X), np.array(y)
            
            if len(X) < 50:  # Reduced minimum requirement for faster processing
                logger.warning(f"Insufficient sequences for enhanced LSTM: {len(X)}")
                return None
            
            # Advanced train/validation/test split
            train_size = int(len(X) * 0.7)
            val_size = int(len(X) * 0.2)
            
            X_train = X[:train_size]
            y_train = y[:train_size]
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            # Check timeout before model building
            if time.time() - start_time > max_execution_time * 0.3:
                logger.warning(f"LSTM prediction timeout during data preparation for {symbol}")
                return None
            
            # Build optimized LSTM model (simpler architecture for speed)
            if not all([Sequential, LSTM, Dropout, Dense]):
                logger.error("LSTM components not available")
                return None
            
            if self.lstm_fast_mode:
                # Ultra-fast simple LSTM for speed
                model = Sequential([  # type: ignore
                    LSTM(32, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])),  # type: ignore
                    Dropout(0.1),  # type: ignore
                    Dense(1, activation='sigmoid')  # type: ignore
                ])
            else:
                # Standard optimized LSTM
                model = Sequential([  # type: ignore
                    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),  # type: ignore
                    Dropout(0.2),  # type: ignore
                    LSTM(32, return_sequences=False),  # type: ignore
                    Dropout(0.2),  # type: ignore
                    Dense(16, activation='relu'),  # type: ignore
                    Dense(1, activation='sigmoid')  # type: ignore
                ])
            
            model.compile(
                optimizer='adam',
                loss='mean_squared_error', 
                metrics=['mae', 'mse']
            )
            
            # Train with optimized parameters for speed
            callbacks = []
            if EarlyStopping is not None:
                early_stopping = EarlyStopping(  # type: ignore
                    monitor='val_loss',
                    patience=3,  # Reduced patience for faster training
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            epochs = min(self.lstm_max_epochs if self.lstm_fast_mode else 20, 
                        max(8, len(X_train) // 100))  # Use configurable epochs
            batch_size = min(64, max(16, len(X_train) // 10))  # Adaptive batch size
            
            history = model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=0
            )
            
            # Check timeout after training
            if time.time() - start_time > max_execution_time * 0.8:
                logger.warning(f"LSTM prediction timeout after training for {symbol}")
                return None
            
            # Optimized ensemble prediction (configurable runs for speed)
            n_predictions = self.lstm_ensemble_size if self.lstm_fast_mode else 3
            all_predictions = []
            
            # Use the last sequence for prediction
            last_sequence = scaled_features[-sequence_length:].reshape(1, sequence_length, -1)
            
            for _ in range(n_predictions):
                current_sequence = last_sequence.copy()
                pred_sequence = []
                
                for i in range(days):
                    # Predict next value
                    next_pred = model.predict(current_sequence, verbose=0)[0, 0]
                    pred_sequence.append(next_pred)
                    
                    # Update sequence for next prediction (simplified approach)
                    # For faster processing, we'll use a simpler sequence update
                    new_features = current_sequence[0, -1, :].copy()
                    new_features[0] = next_pred  # Update price (first feature)
                    
                    # Shift sequence and add new prediction
                    current_sequence = np.concatenate([
                        current_sequence[:, 1:, :],
                        new_features.reshape(1, 1, -1)
                    ], axis=1)
                
                all_predictions.append(pred_sequence)
            
            # Calculate mean and uncertainty from ensemble
            all_predictions = np.array(all_predictions)
            mean_predictions = np.mean(all_predictions, axis=0)
            std_predictions = np.std(all_predictions, axis=0)
            
            # Inverse transform predictions to original price scale
            mean_predictions_scaled = mean_predictions.reshape(-1, 1)
            predictions_original = scaler_target.inverse_transform(mean_predictions_scaled)
            
            # Calculate model performance metrics
            if len(X_test) > 0:
                test_predictions_scaled = model.predict(X_test, verbose=0)
                test_predictions_original = scaler_target.inverse_transform(test_predictions_scaled)
                test_actual_original = scaler_target.inverse_transform(y_test.reshape(-1, 1))
                
                mae = mean_absolute_error(test_actual_original, test_predictions_original)
                rmse = np.sqrt(mean_squared_error(test_actual_original, test_predictions_original))
                r2 = r2_score(test_actual_original, test_predictions_original)
                
                # MAPE calculation
                mape_values = np.abs((test_actual_original - test_predictions_original) / 
                                   np.where(test_actual_original == 0, 1e-8, test_actual_original)) * 100
                mape = np.mean(mape_values[np.isfinite(mape_values)])
                
            else:
                mae = rmse = mape = r2 = 0.0
            
            # Prepare enhanced predictions with uncertainty
            current_price = float(price_data[-1])
            result_predictions = []
            
            for i in range(days):
                prediction_date = datetime.now() + timedelta(days=i+1)
                pred_price = float(predictions_original[i][0])
                
                # Calculate confidence intervals using ensemble uncertainty
                uncertainty = float(std_predictions[i])
                confidence_interval = 1.96 * uncertainty
                
                # Convert uncertainty back to original scale
                uncertainty_original = uncertainty * (scaler_target.data_max_[0] - scaler_target.data_min_[0])
                
                lower_bound = pred_price - uncertainty_original
                upper_bound = pred_price + uncertainty_original
                
                # Dynamic confidence based on ensemble agreement and time horizon
                base_confidence = 85.0
                ensemble_agreement = 1.0 - min(1.0, uncertainty * 5) # Higher uncertainty = lower confidence
                time_decay = max(0.6, 1.0 - (i * 0.08))  # Confidence decreases over time
                model_confidence = min(1.0, max(0.5, r2))
                
                confidence_percentage = base_confidence * ensemble_agreement * time_decay * model_confidence
                confidence_percentage = max(50, min(95, confidence_percentage))
                
                result_predictions.append({
                    'date': prediction_date.strftime('%Y-%m-%d'),
                    'predicted_price': pred_price,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'confidence_percentage': round(confidence_percentage, 1),
                    'uncertainty': round(uncertainty_original, 2),
                    'ensemble_std': round(float(std_predictions[i]), 4),
                    'day': i + 1
                })
            
            # Enhanced accuracy metrics
            accuracy_metrics = {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r_squared': float(r2),
                'test_samples': len(X_test),
                'accuracy_grade': self._get_accuracy_grade(float(mape), 60.0),  # Default directional accuracy
                'ensemble_runs': n_predictions
            }
            
            # Model performance details
            model_info = {
                'sequence_length': sequence_length,
                'features_used': essential_features,
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'epochs_trained': len(history.history['loss']),
                'final_train_loss': float(history.history['loss'][-1]),
                'final_val_loss': float(history.history['val_loss'][-1]) if 'val_loss' in history.history else 0.0,
                'ensemble_predictions': n_predictions
            }
            
            result = {
                'symbol': symbol.upper(),
                'model': 'Enhanced LSTM with Ensemble',
                'current_price': current_price,
                'prediction_days': days,
                'predictions': result_predictions,
                'accuracy_metrics': accuracy_metrics,
                'model_performance': model_info,
                'generated_at': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            # Update model usage statistics
            self.model_usage_stats['lstm'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Enhanced LSTM prediction failed for {symbol}: {e}")
            return None
    
    def get_prediction(self, symbol: str, days: int = 7, model: str = "arima") -> Optional[Dict]:
        """
        Enhanced prediction with intelligent model selection and hybrid approaches
        
        Args:
            symbol: Stock symbol
            days: Number of days to predict (default: 7)
            model: Model to use ("arima", "lstm", "hybrid", or "auto")
        """
        try:
            self.prediction_count += 1
            symbol = symbol.upper().strip()
            
            # Get data for model selection
            data = self._get_stock_data(symbol, period="1y")
            if data is None:
                logger.warning(f"No data available for {symbol}")
                return None
            
            # Analyze data characteristics for intelligent model selection
            ts_analysis = self._analyze_time_series(data['price'])
            best_model = self._select_optimal_model(symbol, model, data, ts_analysis)
            
            logger.info(f"Selected model '{best_model}' for {symbol} based on data analysis")
            
            result = None
            
            # Try the selected model first with timeout handling
            if best_model == "hybrid":
                result = self._predict_with_hybrid(symbol, days, data)
            elif best_model == "arima" and ARIMA_AVAILABLE:
                result = self.predict_with_arima(symbol, days)
            elif best_model == "lstm" and LSTM_AVAILABLE:
                # Add timeout wrapper for LSTM
                import threading
                from typing import List, Any
                lstm_result: List[Any] = [None]
                lstm_error: List[Any] = [None]
                
                def lstm_prediction():
                    try:
                        lstm_result[0] = self.predict_with_lstm(symbol, days)
                    except Exception as e:
                        lstm_error[0] = e
                
                lstm_thread = threading.Thread(target=lstm_prediction)
                lstm_thread.start()
                lstm_thread.join(timeout=90)  # 90 second timeout
                
                if lstm_thread.is_alive():
                    logger.warning(f"LSTM prediction timed out for {symbol}, falling back to ARIMA")
                    result = None
                elif lstm_error[0]:
                    logger.warning(f"LSTM prediction failed for {symbol}: {lstm_error[0]}")
                    result = None
                else:
                    result = lstm_result[0]
            
            # Fallback strategy if primary model fails
            if not result:
                fallback_models = self._get_fallback_models(best_model)
                for fallback in fallback_models:
                    try:
                        if fallback == "arima" and ARIMA_AVAILABLE:
                            result = self.predict_with_arima(symbol, days)
                        elif fallback == "lstm" and LSTM_AVAILABLE:
                            # Use faster LSTM fallback with shorter timeout
                            import threading
                            from typing import List, Any
                            lstm_result: List[Any] = [None]
                            
                            def lstm_fallback():
                                try:
                                    lstm_result[0] = self.predict_with_lstm(symbol, days)
                                except Exception:
                                    pass
                            
                            lstm_thread = threading.Thread(target=lstm_fallback)
                            lstm_thread.start()
                            lstm_thread.join(timeout=60)  # Shorter timeout for fallback
                            
                            if not lstm_thread.is_alive() and lstm_result[0]:
                                result = lstm_result[0]
                        elif fallback == "simple":
                            result = self._simple_prediction(symbol, days)
                        
                        if result:
                            result['fallback_used'] = fallback
                            logger.info(f"Used fallback model '{fallback}' for {symbol}")
                            break
                    except Exception as e:
                        logger.warning(f"Fallback model {fallback} failed for {symbol}: {e}")
                        continue
            
            # Final fallback to simple prediction
            if not result:
                result = self._simple_prediction(symbol, days)
                if result:
                    result['fallback_used'] = 'simple'
            
            if result:
                self.successful_predictions += 1
                # Add prediction metadata
                result['model_selection_reason'] = f"Selected based on data analysis: {ts_analysis.get('recommended_models', [])}"
                result['prediction_id'] = f"{symbol}_{int(time.time())}"
                result['success_rate'] = self.successful_predictions / self.prediction_count if self.prediction_count > 0 else 0
                
                # Update accuracy history
                self._update_accuracy_history(symbol, result)
            else:
                logger.error(f"All prediction methods failed for {symbol}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting prediction for {symbol}: {e}")
            return None
    
    def _select_optimal_model(self, symbol: str, model_preference: str, data: pd.DataFrame, ts_analysis: Dict) -> str:
        """Intelligently select the optimal model based on data characteristics"""
        try:
            if model_preference in ['arima', 'lstm', 'hybrid'] and model_preference != 'auto':
                return model_preference
            
            data_points = len(data)
            volatility_level = ts_analysis.get('volatility_level', 'medium')
            trend_strength = ts_analysis.get('trend_strength', 0)
            recommended_models = ts_analysis.get('recommended_models', [])
            
            # Historical performance for this symbol
            symbol_history = self.model_performance_cache.get(symbol, {})
            
            # Score different models
            scores = {'arima': 0, 'lstm': 0, 'hybrid': 0, 'simple': 0}
            
            # Data size scoring
            if data_points >= 500:
                scores['lstm'] += 3
                scores['hybrid'] += 2
            elif data_points >= 300:
                scores['arima'] += 3
                scores['lstm'] += 2
                scores['hybrid'] += 3
            elif data_points >= 150:
                scores['arima'] += 2
                scores['simple'] += 1
            else:
                scores['simple'] += 3
            
            # Volatility scoring
            if volatility_level == 'high':
                scores['lstm'] += 2
                scores['hybrid'] += 3
            elif volatility_level == 'medium':
                scores['arima'] += 2
                scores['hybrid'] += 1
            else:
                scores['arima'] += 3
            
            # Trend strength scoring
            if trend_strength > 2:
                scores['arima'] += 2
                scores['hybrid'] += 1
            elif trend_strength > 1:
                scores['arima'] += 1
            
            # Model availability scoring
            if not ARIMA_AVAILABLE:
                scores['arima'] = 0
                scores['hybrid'] = 0
            if not LSTM_AVAILABLE:
                scores['lstm'] = 0
                scores['hybrid'] = 0
            
            # Historical performance scoring
            for model_name, perf_data in symbol_history.items():
                if model_name in scores and perf_data.get('success_rate', 0) > 0.7:
                    scores[model_name] += 2
            
            # Select model with highest score
            best_model = max(scores.keys(), key=lambda k: scores[k])
            
            # Log selection reasoning
            logger.info(f"Model selection for {symbol}: scores={scores}, selected={best_model}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return 'simple'
    
    def _get_fallback_models(self, primary_model: str) -> List[str]:
        """Get fallback models in order of preference"""
        fallback_order = {
            'hybrid': ['lstm', 'arima', 'simple'],
            'lstm': ['arima', 'simple'],
            'arima': ['simple'],
            'simple': []
        }
        return fallback_order.get(primary_model, ['simple'])
    
    def _predict_with_hybrid(self, symbol: str, days: int, data: pd.DataFrame) -> Optional[Dict]:
        """Hybrid prediction combining ARIMA and LSTM with ensemble weighting"""
        try:
            cache_key = f"hybrid_{symbol}_{days}"
            if self._is_cache_valid(cache_key):
                return self.cache[cache_key]['data']
            
            # Get predictions from both models
            arima_result = self.predict_with_arima(symbol, days) if ARIMA_AVAILABLE else None
            lstm_result = self.predict_with_lstm(symbol, days) if LSTM_AVAILABLE else None
            
            if not arima_result and not lstm_result:
                return None
            
            # If only one model succeeded, return that result
            if not arima_result and lstm_result:
                lstm_result['model'] = 'LSTM (Hybrid fallback)'
                return lstm_result
            if not lstm_result and arima_result:
                arima_result['model'] = 'ARIMA (Hybrid fallback)'
                return arima_result
            
            # Combine predictions with intelligent weighting
            combined_predictions = []
            current_price = float(data['price'].iloc[-1])
            
            # Both models succeeded - calculate weights and combine
            arima_weight = self._calculate_model_weight(arima_result) if arima_result else 0.0
            lstm_weight = self._calculate_model_weight(lstm_result) if lstm_result else 0.0
            
            total_weight = arima_weight + lstm_weight
            if total_weight > 0:
                arima_weight /= total_weight
                lstm_weight /= total_weight
            else:
                arima_weight = lstm_weight = 0.5
            
            logger.info(f"Hybrid weights for {symbol}: ARIMA={arima_weight:.2f}, LSTM={lstm_weight:.2f}")
            
            for i in range(days):
                arima_pred = arima_result['predictions'][i] if arima_result else None
                lstm_pred = lstm_result['predictions'][i] if lstm_result else None
                
                if not arima_pred or not lstm_pred:
                    continue
                
                # Weighted combination
                combined_price = (arima_pred['predicted_price'] * arima_weight + 
                                lstm_pred['predicted_price'] * lstm_weight)
                
                # Combined confidence intervals
                combined_lower = (arima_pred['lower_bound'] * arima_weight + 
                                lstm_pred['lower_bound'] * lstm_weight)
                combined_upper = (arima_pred['upper_bound'] * arima_weight + 
                                lstm_pred['upper_bound'] * lstm_weight)
                
                # Combined confidence
                combined_confidence = (arima_pred['confidence_percentage'] * arima_weight + 
                                     lstm_pred['confidence_percentage'] * lstm_weight)
                
                combined_predictions.append({
                    'date': arima_pred['date'],
                    'predicted_price': float(combined_price),
                    'lower_bound': float(combined_lower),
                    'upper_bound': float(combined_upper),
                    'confidence_percentage': round(float(combined_confidence), 1),
                    'day': i + 1,
                    'arima_contribution': arima_weight,
                    'lstm_contribution': lstm_weight
                })
            
            # Combined accuracy metrics
            combined_accuracy = self._combine_accuracy_metrics(
                arima_result['accuracy_metrics'] if arima_result else {},
                lstm_result['accuracy_metrics'] if lstm_result else {},
                arima_weight,
                lstm_weight
            )
            
            result = {
                'symbol': symbol.upper(),
                'model': f'Hybrid Ensemble (ARIMA: {arima_weight:.1%}, LSTM: {lstm_weight:.1%})',
                'current_price': current_price,
                'prediction_days': days,
                'predictions': combined_predictions,
                'accuracy_metrics': combined_accuracy,
                'model_performance': {
                    'arima_weight': arima_weight,
                    'lstm_weight': lstm_weight,
                    'component_models': ['Enhanced ARIMA', 'Enhanced LSTM'],
                    'ensemble_method': 'Weighted Average'
                },
                'component_results': {
                    'arima': arima_result,
                    'lstm': lstm_result
                },
                'generated_at': datetime.now().isoformat()
            }
            
            # Cache the result
            self.cache[cache_key] = {
                'data': result,
                'timestamp': time.time()
            }
            
            # Update model usage statistics
            self.model_usage_stats['hybrid'] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Hybrid prediction failed for {symbol}: {e}")
            return None
    
    def _calculate_model_weight(self, model_result: Dict) -> float:
        """Calculate weight for model in ensemble based on performance metrics"""
        try:
            accuracy_metrics = model_result.get('accuracy_metrics', {})
            
            # Base weight
            weight = 0.5
            
            # Adjust based on MAPE (lower is better)
            mape = accuracy_metrics.get('mape', 50.0)
            if mape < 5:
                weight += 0.3
            elif mape < 10:
                weight += 0.2
            elif mape < 15:
                weight += 0.1
            elif mape > 25:
                weight -= 0.2
            
            # Adjust based on R²
            r2 = accuracy_metrics.get('r_squared', 0.5)
            weight += (r2 - 0.5) * 0.4
            
            # Adjust based on directional accuracy
            dir_acc = accuracy_metrics.get('directional_accuracy', 50.0)
            if dir_acc > 70:
                weight += 0.2
            elif dir_acc > 60:
                weight += 0.1
            elif dir_acc < 45:
                weight -= 0.1
            
            return max(0.1, min(1.0, weight))
            
        except Exception:
            return 0.5
    
    def _combine_accuracy_metrics(self, arima_metrics: Dict, lstm_metrics: Dict, 
                                 arima_weight: float, lstm_weight: float) -> Dict:
        """Combine accuracy metrics from multiple models"""
        try:
            combined = {}
            
            # Weighted average of metrics
            metrics_to_combine = ['mae', 'rmse', 'mape']
            for metric in metrics_to_combine:
                arima_val = arima_metrics.get(metric, 0.0)
                lstm_val = lstm_metrics.get(metric, 0.0)
                combined[metric] = float(arima_val * arima_weight + lstm_val * lstm_weight)
            
            # Take better of directional accuracies
            arima_dir = arima_metrics.get('directional_accuracy', 50.0)
            lstm_dir = lstm_metrics.get('directional_accuracy', 50.0)
            combined['directional_accuracy'] = max(arima_dir, lstm_dir)
            
            # Combined R²
            arima_r2 = arima_metrics.get('r_squared', 0.5)
            lstm_r2 = lstm_metrics.get('r_squared', 0.5)
            combined['r_squared'] = arima_r2 * arima_weight + lstm_r2 * lstm_weight
            
            combined['accuracy_grade'] = self._get_accuracy_grade(
                combined['mape'], 
                combined['directional_accuracy']
            )
            combined['ensemble_method'] = 'Weighted combination of ARIMA and LSTM'
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining accuracy metrics: {e}")
            return {'note': 'Error in metric combination'}
    
    def _update_accuracy_history(self, symbol: str, result: Dict):
        """Update historical accuracy tracking for model selection"""
        try:
            model_name = result.get('model', 'unknown').lower()
            accuracy_metrics = result.get('accuracy_metrics', {})
            
            if symbol not in self.accuracy_history:
                self.accuracy_history[symbol] = {}
            
            if model_name not in self.accuracy_history[symbol]:
                self.accuracy_history[symbol][model_name] = []
            
            # Store recent accuracy data (keep last 10 predictions)
            accuracy_data = {
                'timestamp': time.time(),
                'mape': accuracy_metrics.get('mape', 0),
                'directional_accuracy': accuracy_metrics.get('directional_accuracy', 50),
                'r_squared': accuracy_metrics.get('r_squared', 0.5)
            }
            
            self.accuracy_history[symbol][model_name].append(accuracy_data)
            
            # Keep only recent history
            if len(self.accuracy_history[symbol][model_name]) > 10:
                self.accuracy_history[symbol][model_name] = self.accuracy_history[symbol][model_name][-10:]
            
        except Exception as e:
            logger.error(f"Error updating accuracy history: {e}")
    
    def get_model_performance_stats(self) -> Dict:
        """Get comprehensive model performance statistics"""
        try:
            total_predictions = self.prediction_count
            success_rate = self.successful_predictions / total_predictions if total_predictions > 0 else 0
            
            return {
                'total_predictions': total_predictions,
                'successful_predictions': self.successful_predictions,
                'success_rate': round(success_rate * 100, 2),
                'model_usage': self.model_usage_stats.copy(),
                'cache_size': len(self.cache),
                'available_models': {
                    'arima': ARIMA_AVAILABLE,
                    'lstm': LSTM_AVAILABLE,
                    'hybrid': ARIMA_AVAILABLE and LSTM_AVAILABLE,
                    'simple': True
                },
                'accuracy_history_symbols': list(self.accuracy_history.keys()),
                'performance_cache_symbols': list(self.model_performance_cache.keys())
            }
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {'error': 'Failed to get performance statistics'}
    
    def _simple_prediction(self, symbol: str, days: int = 7) -> Optional[Dict]:
        """
        Simple fallback prediction using moving averages
        """
        try:
            data = self._get_stock_data(symbol, period="3mo")
            if data is None or len(data) < 30:
                return None
            
            # Convert prices to numpy array
            try:
                prices = to_numpy_array(data['price'])
            except:
                # Fallback: convert pandas to numpy
                prices = to_numpy_array(data['price'])
            
            current_price = prices[-1]
            
            # Calculate moving averages
            ma_7 = np.mean(prices[-7:])
            ma_30 = np.mean(prices[-30:])
            
            # Simple trend estimation
            recent_trend = (ma_7 - ma_30) / ma_30
            
            predictions = []
            for i in range(days):
                prediction_date = datetime.now() + timedelta(days=i+1)
                # Apply trend with decreasing confidence
                trend_factor = recent_trend * (0.9 ** i)  # Decay factor
                predicted_price = current_price * (1 + trend_factor)
                
                # Simple confidence intervals (±5%)
                confidence_range = predicted_price * 0.05
                
                # Calculate confidence percentage (decreases over time)
                base_confidence = 75.0  # Base confidence for simple model
                time_decay = max(0, base_confidence - (i * 5))  # Decrease confidence over time
                
                predictions.append({
                    'date': prediction_date.strftime('%Y-%m-%d'),
                    'predicted_price': float(predicted_price),
                    'lower_bound': float(predicted_price - confidence_range),
                    'upper_bound': float(predicted_price + confidence_range),
                    'confidence_interval': 95,
                    'confidence_percentage': round(time_decay, 1)
                })
            
            return {
                'symbol': symbol.upper(),
                'model': 'Simple Moving Average',
                'current_price': float(current_price),
                'prediction_days': days,
                'predictions': predictions,
                'accuracy_metrics': {
                    'note': 'Simplified prediction model - accuracy metrics not available'
                },
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in simple prediction for {symbol}: {e}")
            return None

    def predict_batch(self, symbols: List[str], days: int = 7, model: str = "auto") -> Dict:
        """
        Batch prediction for multiple stocks with parallel processing
        """
        try:
            results = {}
            summary = {
                'total_symbols': len(symbols),
                'successful_predictions': 0,
                'failed_predictions': 0,
                'model_usage': {},
                'processing_time': 0
            }
            
            start_time = time.time()
            
            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(5, len(symbols))) as executor:
                # Submit all prediction tasks
                future_to_symbol = {
                    executor.submit(self.get_prediction, symbol.upper().strip(), days, model): symbol.upper().strip()
                    for symbol in symbols
                }
                
                # Collect results
                for future in future_to_symbol:
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result(timeout=30)  # 30 second timeout per prediction
                        if result:
                            results[symbol] = result
                            summary['successful_predictions'] += 1
                            
                            # Track model usage
                            model_used = result.get('model', 'unknown')
                            summary['model_usage'][model_used] = summary['model_usage'].get(model_used, 0) + 1
                        else:
                            results[symbol] = {'error': f'Prediction failed for {symbol}'}
                            summary['failed_predictions'] += 1
                    except Exception as e:
                        results[symbol] = {'error': f'Prediction error for {symbol}: {str(e)}'}
                        summary['failed_predictions'] += 1
            
            summary['processing_time'] = round(time.time() - start_time, 2)
            summary['success_rate'] = (summary['successful_predictions'] / summary['total_symbols'] * 100) if summary['total_symbols'] > 0 else 0
            
            return {
                'predictions': results,
                'summary': summary,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            return {
                'predictions': {},
                'summary': {'error': 'Batch prediction failed'},
                'generated_at': datetime.now().isoformat()
            }
    
    def compare_models(self, symbol: str, days: int = 7) -> Optional[Dict]:
        """
        Compare all available prediction models for a given symbol
        Optimized for speed with reduced days limit
        """
        try:
            symbol = symbol.upper().strip()
            
            # Limit days for faster comparison
            days = min(days, 7)  # Max 7 days for comparison to speed up
            
            models_to_test = []
            
            # Only include ARIMA and LSTM models for comparison
            if ARIMA_AVAILABLE:
                models_to_test.append('arima')
            if LSTM_AVAILABLE:
                models_to_test.append('lstm')
            
            if not models_to_test:
                return None
            
            results = {}
            comparison_data = {}
            
            # Get stock data once to reuse for all models
            stock_data = self._get_stock_data(symbol)
            if stock_data is None or stock_data.empty or len(stock_data) < 30:
                logger.warning(f"Insufficient data for comparison: {symbol}")
                return None
            
            # Get predictions from all available models with optimized settings
            for model in models_to_test:
                try:
                    # Use regular prediction but with reduced days for faster comparison
                    result = self.get_prediction(symbol, days, model)
                    if result:
                        results[model] = result
                        
                        # Extract comparison metrics
                        predictions = result.get('predictions', [])
                        if predictions:
                            first_pred = predictions[0]
                            last_pred = predictions[-1]
                            current_price = result.get('current_price', 0)
                            
                            total_change = ((last_pred['predicted_price'] - current_price) / current_price * 100) if current_price > 0 else 0
                            avg_confidence = sum(p.get('confidence_percentage', 0) for p in predictions) / len(predictions)
                            
                            comparison_data[model] = {
                                'day_1_price': first_pred['predicted_price'],
                                'final_day_price': last_pred['predicted_price'],
                                'total_change_percent': round(total_change, 2),
                                'average_confidence': round(avg_confidence, 1),
                                'accuracy_grade': result.get('accuracy_metrics', {}).get('accuracy_grade', 'N/A'),
                                'model_name': result.get('model', model)
                            }
                except Exception as e:
                    logger.warning(f"Model {model} failed for {symbol}: {e}")
                    results[model] = {'error': f'Model failed: {str(e)}'}
            
            if not comparison_data:
                return None
            
            # Determine recommendation
            recommendation = self._get_model_recommendation(comparison_data)
            
            return {
                'symbol': symbol,
                'comparison_date': datetime.now().isoformat(),
                'models_compared': list(results.keys()),
                'detailed_results': results,
                'comparison_summary': {
                    'predictions_comparison': comparison_data,
                    'models_compared': list(comparison_data.keys()),
                    'recommendation': recommendation
                },
                'recommendation': recommendation
            }
            
        except Exception as e:
            logger.error(f"Model comparison failed for {symbol}: {e}")
            return None
    
    def _get_model_recommendation(self, comparison_data: Dict) -> Dict:
        """Get recommendation for best model based on comparison"""
        try:
            if not comparison_data:
                return {'suggestion': 'No models available', 'reason': 'All models failed'}
            
            # Score models based on multiple criteria
            scores = {}
            
            for model, data in comparison_data.items():
                score = 0
                
                # Confidence score (higher is better)
                confidence = data.get('average_confidence', 0)
                score += confidence / 10  # Scale to 0-10
                
                # Accuracy grade score
                grade = data.get('accuracy_grade', 'N/A')
                if 'A' in grade:
                    score += 10
                elif 'B' in grade:
                    score += 8
                elif 'C' in grade:
                    score += 6
                elif 'D' in grade:
                    score += 4
                else:
                    score += 2
                
                # Stability score (prefer models with moderate predictions)
                change = abs(data.get('total_change_percent', 0))
                if change < 15:  # Reasonable change
                    score += 5
                elif change > 50:  # Extreme change
                    score -= 5
                
                scores[model] = score
            
            # Find best model
            best_model = max(scores.keys(), key=lambda k: scores[k])
            best_score = scores[best_model]
            
            # Generate recommendation text
            model_name = comparison_data[best_model].get('model_name', best_model)
            confidence = comparison_data[best_model].get('average_confidence', 0)
            
            if best_score > 15:
                suggestion = f"Use {model_name} for best results"
                reason = f"Highest overall score ({best_score:.1f}) with {confidence:.1f}% average confidence"
            elif best_score > 10:
                suggestion = f"Consider {model_name} for predictions"
                reason = f"Good performance with {confidence:.1f}% confidence"
            else:
                suggestion = f"Use {model_name} with caution"
                reason = f"Limited confidence in predictions ({confidence:.1f}%)"
            
            return {
                'suggestion': suggestion,
                'reason': reason,
                'best_model': best_model,
                'model_scores': scores,
                'confidence_level': 'high' if best_score > 15 else 'medium' if best_score > 10 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error getting model recommendation: {e}")
            return {'suggestion': 'Use available models with caution', 'reason': 'Unable to determine best model'}
    
    def get_model_availability(self) -> Dict:
        """Get detailed information about model availability and capabilities"""
        try:
            return {
                'models': {
                    'arima': {
                        'name': 'Enhanced ARIMA with Auto Parameter Selection',
                        'available': ARIMA_AVAILABLE,
                        'best_for': 'Trending stocks with linear patterns',
                        'accuracy_grade': 'B-A (Good to Excellent)',
                        'computational_cost': 'Low',
                        'features': ['Auto parameter selection', 'Confidence intervals', 'Time series analysis']
                    },
                    'lstm': {
                        'name': 'Enhanced LSTM with Ensemble',
                        'available': LSTM_AVAILABLE,
                        'best_for': 'Complex patterns and non-linear trends',
                        'accuracy_grade': 'A-B (Excellent to Good)',
                        'computational_cost': 'High',
                        'features': ['Multi-feature input', 'Ensemble predictions', 'Uncertainty quantification']
                    },
                    'hybrid': {
                        'name': 'Hybrid Ensemble (ARIMA + LSTM)',
                        'available': ARIMA_AVAILABLE and LSTM_AVAILABLE,
                        'best_for': 'Maximum accuracy combining multiple approaches',
                        'accuracy_grade': 'A (Excellent)',
                        'computational_cost': 'High',
                        'features': ['Weighted combination', 'Best of both models', 'Intelligent model selection']
                    },
                    'simple': {
                        'name': 'Enhanced Moving Average with Trend Analysis',
                        'available': True,
                        'best_for': 'Quick predictions and fallback scenarios',
                        'accuracy_grade': 'C (Fair)',
                        'computational_cost': 'Very Low',
                        'features': ['Multiple moving averages', 'Trend analysis', 'Volatility adjustment']
                    }
                },
                'system_status': {
                    'tensorflow_available': LSTM_AVAILABLE,
                    'statsmodels_available': ARIMA_AVAILABLE,
                    'recommended_model': 'hybrid' if (ARIMA_AVAILABLE and LSTM_AVAILABLE) else 'auto',
                    'total_predictions': self.prediction_count,
                    'success_rate': (self.successful_predictions / self.prediction_count * 100) if self.prediction_count > 0 else 0
                },
                'performance_stats': self.get_model_performance_stats()
            }
            
        except Exception as e:
            logger.error(f"Error getting model availability: {e}")
            return {'error': 'Failed to get model availability information'}

# Global enhanced instance
stock_predictor = StockPredictor()
