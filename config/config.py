"""
Configuration settings for sentiment-stock analysis.
"""

# Data paths
DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
RESULTS_DIR = "results"

# Model parameters
TRAIN_SIZE = 60  # days for walk-forward validation
TEST_SIZE = 5    # days for testing
MIN_TWEETS_PER_DAY = 5

# Feature parameters
LAG_PERIODS = [1, 2, 7]  # days of lag features
ROLLING_WINDOW = 7       # days for rolling average

# Model parameters
RANDOM_STATE = 42
TEST_SIZE_RATIO = 0.2

# Stock tickers to analyze
TICKERS = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']

# Sentiment model
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32
MAX_TOKENS = 512
