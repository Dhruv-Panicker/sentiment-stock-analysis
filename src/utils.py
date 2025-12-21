"""
Utility functions for the sentiment-stock analysis project.

Includes helpers for logging, visualization, and data manipulation.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd


def setup_logging(log_file: str = "logs/analysis.log") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
    """
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def save_results(results: Dict[str, Any], filepath: str) -> None:
    """
    Save results to JSON file.
    
    Args:
        results: Dictionary of results
        filepath: Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    with open(filepath, 'w') as f:
        json.dump(results, f, default=convert_types, indent=2)


def load_results(filepath: str) -> Dict[str, Any]:
    """
    Load results from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_ticker_stats(df: pd.DataFrame, ticker: str) -> Dict[str, float]:
    """
    Calculate statistics for a specific ticker.
    
    Args:
        df: DataFrame with data
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary of statistics
    """
    ticker_data = df[df['ticker'] == ticker]
    
    return {
        'ticker': ticker,
        'n_records': len(ticker_data),
        'n_days': len(ticker_data['date'].unique()),
        'avg_sentiment': float(ticker_data['sentiment_mean'].mean()),
        'std_sentiment': float(ticker_data['sentiment_mean'].std()),
        'avg_price': float(ticker_data['close_price'].mean()),
        'min_price': float(ticker_data['close_price'].min()),
        'max_price': float(ticker_data['close_price'].max()),
        'price_volatility': float(ticker_data['close_price'].std() / ticker_data['close_price'].mean())
    }


def calculate_correlation_matrix(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Calculate correlation matrix between features.
    
    Args:
        df: Input DataFrame
        features: List of feature columns
        
    Returns:
        Correlation matrix
    """
    return df[features].corr()


def detect_outliers(data: np.ndarray, method: str = "iqr", threshold: float = 3.0) -> np.ndarray:
    """
    Detect outliers in data.
    
    Args:
        data: Input array
        method: "iqr" or "zscore"
        threshold: Threshold for outlier detection
        
    Returns:
        Boolean array marking outliers
    """
    if method == "iqr":
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        return (data < Q1 - 1.5 * IQR) | (data > Q3 + 1.5 * IQR)
    elif method == "zscore":
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold
    else:
        raise ValueError(f"Unknown method: {method}")
