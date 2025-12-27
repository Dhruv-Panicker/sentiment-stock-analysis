"""
Prediction Service Module
Handles model loading, scaling, and inference
"""

import pickle
import numpy as np
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


class PredictionService:
    """
    Service for managing model inference and predictions
    Loads pre-trained Random Forest models and scalers
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        # Initialize prediction service by loading models and scalers
        
        Args:
            models_dir: Directory containing pickled models and scalers
        """
        self.models_dir = Path(models_dir)
        self.models = {}
        self.scalers = {}
        self.available_tickers = ["TSLA", "AMD", "NIO"]
        
        # Load models and scalers
        self._load_models()
        self._load_scalers()
        
        if self.is_ready():
            logger.info(f"✅ PredictionService initialized with {len(self.models)} models")
        else:
            logger.error("❌ PredictionService initialization failed - models not loaded")
    
    
    def _load_models(self):
        #Load pre-trained Random Forest models from pickle files
        logger.info(f"Loading models from {self.models_dir}...")
        
        for ticker in self.available_tickers:
            model_path = self.models_dir / f"rf_model_{ticker}.pkl"
            
            try:
                with open(model_path, 'rb') as f:
                    self.models[ticker] = pickle.load(f)
                logger.info(f"Loaded {ticker} model")
            except FileNotFoundError:
                logger.warning(f" Model not found for {ticker}: {model_path}")
            except Exception as e:
                logger.error(f" Error loading {ticker} model: {e}")
    
    
    def _load_scalers(self):
        #Load StandardScaler objects from pickle files
        logger.info(f"Loading scalers from {self.models_dir}...")
        
        for ticker in self.available_tickers:
            scaler_path = self.models_dir / f"scaler_{ticker}.pkl"
            
            try:
                with open(scaler_path, 'rb') as f:
                    self.scalers[ticker] = pickle.load(f)
                logger.info(f" Loaded {ticker} scaler")
            except FileNotFoundError:
                logger.warning(f" Scaler not found for {ticker}: {scaler_path}")
            except Exception as e:
                logger.error(f" Error loading {ticker} scaler: {e}")
    
    
    def is_ready(self) -> bool:
        # Check if all models and scalers are loaded
        return (
            len(self.models) == len(self.available_tickers) and
            len(self.scalers) == len(self.available_tickers)
        )
    
    
    def validate_input(self, ticker: str, sentiment_mean: float, sentiment_momentum: float):
        """
        Validate input parameters
        
        Args:
            ticker: Stock ticker symbol
            sentiment_mean: Average sentiment score
            sentiment_momentum: Sentiment momentum/change
            
        Raises:
            ValueError: If inputs are invalid
        """
        # Validate ticker
        if ticker not in self.available_tickers:
            raise ValueError(f"Invalid ticker '{ticker}'. Must be one of: {self.available_tickers}")
        
        # Validate sentiment_mean
        if not isinstance(sentiment_mean, (int, float)):
            raise ValueError(f"sentiment_mean must be a number, got {type(sentiment_mean)}")
        if sentiment_mean < -1.0 or sentiment_mean > 1.0:
            raise ValueError(f"sentiment_mean must be between -1.0 and 1.0, got {sentiment_mean}")
        
        # Validate sentiment_momentum
        if not isinstance(sentiment_momentum, (int, float)):
            raise ValueError(f"sentiment_momentum must be a number, got {type(sentiment_momentum)}")
        if sentiment_momentum < -1.0 or sentiment_momentum > 1.0:
            raise ValueError(f"sentiment_momentum must be between -1.0 and 1.0, got {sentiment_momentum}")
    
    
    def predict(self, ticker: str, sentiment_mean: float, sentiment_momentum: float) -> Tuple[float, str]:
        """
        Make prediction for given inputs
        
        Args:
            ticker: Stock ticker (TSLA, AMD, NIO)
            sentiment_mean: Average sentiment (-1 to 1)
            sentiment_momentum: Sentiment momentum/change (-1 to 1)
            
        Returns:
            Tuple of (prediction, confidence_level)
            prediction: Predicted price change percentage
            confidence_level: "low", "medium", "high" based on magnitude
            
        Raises:
            ValueError: If inputs are invalid
            KeyError: If model or scaler not found for ticker
        """
        # Validate inputs
        self.validate_input(ticker, sentiment_mean, sentiment_momentum)
        
        # Check if model exists
        if ticker not in self.models:
            raise KeyError(f"Model not found for ticker {ticker}")
        if ticker not in self.scalers:
            raise KeyError(f"Scaler not found for ticker {ticker}")
        
        # Prepare input features
        X = np.array([[sentiment_mean, sentiment_momentum]])
        
        # Scale features using ticker-specific scaler
        scaler = self.scalers[ticker]
        X_scaled = scaler.transform(X)
        
        # Get model and make prediction
        model = self.models[ticker]
        prediction = model.predict(X_scaled)[0]
        
        # Determine confidence level based on prediction magnitude
        abs_prediction = abs(prediction)
        if abs_prediction > 2.0:
            confidence = "high"
        elif abs_prediction > 1.0:
            confidence = "medium"
        else:
            confidence = "low"
        
        return prediction, confidence
    
    
    def predict_batch(self, predictions_list: list) -> list:
        """
        Make multiple predictions
        
        Args:
            predictions_list: List of dicts with keys: ticker, sentiment_mean, sentiment_momentum
            
        Returns:
            List of tuples (prediction, confidence)
        """
        results = []
        
        for pred_data in predictions_list:
            ticker = pred_data.get('ticker')
            sentiment_mean = pred_data.get('sentiment_mean')
            sentiment_momentum = pred_data.get('sentiment_momentum')
            
            try:
                prediction, confidence = self.predict(ticker, sentiment_mean, sentiment_momentum)
                results.append({
                    'ticker': ticker,
                    'prediction': prediction,
                    'confidence': confidence,
                    'error': None
                })
            except Exception as e:
                logger.error(f"Error predicting for {pred_data}: {e}")
                results.append({
                    'ticker': pred_data.get('ticker'),
                    'prediction': None,
                    'confidence': None,
                    'error': str(e)
                })
        
        return results
    
    
    @staticmethod
    def get_prediction_interpretation(prediction: float) -> str:
        """
        Convert raw prediction to human-readable interpretation
        
        Args:
            prediction: Predicted price change percentage
            confidence: Confidence level (low, medium, high)
            
        Returns:
            Human-readable interpretation string with emoji and details
        """
        abs_pred = abs(prediction)
        
        if prediction > 0:
            direction = "Price Expected to Rise"
        else:
            direction = "Price Expected to Fall"
        
        if abs_pred < 0.5:
            magnitude = "minimal change"
        elif abs_pred < 1.5:
            magnitude = "small change"
        elif abs_pred < 3.0:
            magnitude = "moderate change"
        else:
            magnitude = "significant change"
        
        return f"{direction} | Expected: {prediction:+.2f}% ({magnitude})"
