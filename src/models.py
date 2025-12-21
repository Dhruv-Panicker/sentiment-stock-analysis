"""
Model training and evaluation module.

Implements baseline models (Linear, Polynomial) and advanced models (XGBoost, LSTM).
Includes proper time-series validation strategies.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

logger = logging.getLogger(__name__)


class TimeSeriesValidator:
    """Implements proper time-series validation strategies."""
    
    @staticmethod
    def walk_forward_validate(
        X: np.ndarray, 
        y: np.ndarray, 
        train_size: int = 60, 
        test_size: int = 5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Walk-forward validation for time series.
        
        Args:
            X: Feature matrix (time-ordered)
            y: Target vector (time-ordered)
            train_size: Training window size (days)
            test_size: Testing window size (days)
            
        Yields:
            (X_train, X_test, y_train, y_test) tuples
        """
        n = len(X)
        
        for start in range(0, n - train_size - test_size, test_size):
            X_train = X[start:start + train_size]
            y_train = y[start:start + train_size]
            X_test = X[start + train_size:start + train_size + test_size]
            y_test = y[start + train_size:start + train_size + test_size]
            
            yield X_train, X_test, y_train, y_test
    
    @staticmethod
    def time_series_split(n_splits: int = 5):
        """Return TimeSeriesSplit validator."""
        return TimeSeriesSplit(n_splits=n_splits)


class SentimentStockModel:
    """Train and evaluate stock prediction models."""
    
    def __init__(self, model_type: str = "xgboost"):
        """
        Initialize model.
        
        Args:
            model_type: "linear", "polynomial", "xgboost", or "ridge"
        """
        self.model_type = model_type
        self.model = self._init_model()
        self.scaler = StandardScaler()
        self.poly_features = None
        
    def _init_model(self):
        """Initialize the underlying model."""
        if self.model_type == "linear":
            return LinearRegression()
        elif self.model_type == "ridge":
            return Ridge(alpha=1.0)
        elif self.model_type == "lasso":
            return Lasso(alpha=0.01)
        elif self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
                verbosity=0
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def prepare_features(self, X: np.ndarray, y: np.ndarray = None, fit: bool = True) -> np.ndarray:
        """
        Scale and prepare features.
        
        Args:
            X: Raw feature matrix
            y: Optional target (for fit=True)
            fit: Whether to fit the scaler
            
        Returns:
            Processed feature matrix
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        if self.model_type == "polynomial":
            if fit:
                self.poly_features = PolynomialFeatures(degree=3)
                X_scaled = self.poly_features.fit_transform(X_scaled)
            else:
                X_scaled = self.poly_features.transform(X_scaled)
        
        return X_scaled
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train model on data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Training metrics dictionary
        """
        logger.info(f"Training {self.model_type} model on {len(X)} samples")
        
        X_prepared = self.prepare_features(X, y, fit=True)
        self.model.fit(X_prepared, y)
        
        y_pred = self.model.predict(X_prepared)
        metrics = self._calculate_metrics(y, y_pred, "train")
        
        logger.info(f"Training R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")
        return metrics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Evaluation metrics dictionary
        """
        X_prepared = self.prepare_features(X, fit=False)
        y_pred = self.model.predict(X_prepared)
        metrics = self._calculate_metrics(y, y_pred, "test")
        
        logger.info(f"Test R²: {metrics['r2']:.4f}, MAE: {metrics['mae']:.4f}")
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        X_prepared = self.prepare_features(X, fit=False)
        return self.model.predict(X_prepared)
    
    @staticmethod
    def _calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, stage: str) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'stage': stage
        }


class ModelEnsemble:
    """Ensemble multiple models for better predictions."""
    
    def __init__(self, models: list = None):
        """
        Initialize ensemble.
        
        Args:
            models: List of initialized model objects
        """
        self.models = models or [
            SentimentStockModel("ridge"),
            SentimentStockModel("xgboost")
        ]
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train all models."""
        for model in self.models:
            model.train(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average predictions from all models."""
        predictions = np.array([model.predict(X) for model in self.models])
        return np.mean(predictions, axis=0)
