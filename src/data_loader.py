"""
Data loading and preprocessing module for sentiment-stock analysis.

Handles:
- Loading tweet and stock price data
- Merging datasets by date and ticker
- Cleaning and normalizing data
- Feature engineering for temporal analysis
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess sentiment and stock price data."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize DataLoader.
        
        Args:
            data_dir: Directory containing raw data files
        """
        self.data_dir = data_dir
        self.tweets_df = None
        self.prices_df = None
        self.merged_df = None
        
    def load_tweets(self, filepath: str) -> pd.DataFrame:
        """
        Load tweet sentiment data.
        
        Args:
            filepath: Path to tweets CSV file
            
        Returns:
            DataFrame with columns: date, ticker, sentiment_score
        """
        logger.info(f"Loading tweets from {filepath}")
        self.tweets_df = pd.read_csv(filepath)
        self.tweets_df['date'] = pd.to_datetime(self.tweets_df['date'])
        return self.tweets_df
    
    def load_stock_prices(self, filepath: str) -> pd.DataFrame:
        """
        Load stock price data.
        
        Args:
            filepath: Path to stock prices CSV file
            
        Returns:
            DataFrame with columns: date, ticker, close_price
        """
        logger.info(f"Loading stock prices from {filepath}")
        self.prices_df = pd.read_csv(filepath)
        self.prices_df['date'] = pd.to_datetime(self.prices_df['date'])
        return self.prices_df
    
    def merge_data(self, tweets_df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge sentiment and price data by date and ticker.
        
        Args:
            tweets_df: DataFrame with tweet sentiment
            prices_df: DataFrame with stock prices
            
        Returns:
            Merged DataFrame with both sentiment and prices
        """
        logger.info("Merging sentiment and price data")
        
        # Group tweets by date and ticker
        daily_sentiment = tweets_df.groupby(['date', 'ticker'])['sentiment_score'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        daily_sentiment.columns = ['date', 'ticker', 'sentiment_mean', 'sentiment_std', 'tweet_count']
        
        # Merge with prices
        merged = prices_df.merge(daily_sentiment, on=['date', 'ticker'], how='left')
        merged = merged.sort_values(['ticker', 'date'])
        
        self.merged_df = merged
        return merged
    
    def clean_data(self, df: pd.DataFrame, min_tweets: int = 5) -> pd.DataFrame:
        """
        Clean merged data by removing NaNs and low-signal records.
        
        Args:
            df: Merged DataFrame
            min_tweets: Minimum tweets per day per ticker to keep
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Cleaning data (min tweets: {min_tweets})")
        
        df = df.copy()
        
        # Remove rows with missing sentiment or price
        df = df.dropna(subset=['sentiment_mean', 'close_price'])
        
        # Remove days with too few tweets
        df = df[df['tweet_count'] >= min_tweets]
        
        logger.info(f"Cleaned data: {len(df)} records")
        return df


def normalize_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Normalize features to [0, 1] scale.
    
    Args:
        df: Input DataFrame
        feature_cols: Columns to normalize
        
    Returns:
        DataFrame with normalized features
    """
    df = df.copy()
    for col in feature_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        df[f"{col}_normalized"] = (df[col] - min_val) / (max_val - min_val + 1e-8)
    return df


def create_lag_features(df: pd.DataFrame, target_col: str, lags: List[int] = [1, 2, 7]) -> pd.DataFrame:
    """
    Create lagged features for time series analysis.
    
    Args:
        df: Input DataFrame (must be sorted by date)
        target_col: Column to create lags for
        lags: List of lag values (in days)
        
    Returns:
        DataFrame with lag features added
    """
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = df.groupby('ticker')[target_col].shift(lag)
    return df
