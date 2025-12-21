"""
Main entry point for the sentiment-stock analysis project.

This script orchestrates the entire pipeline:
1. Load data
2. Analyze sentiment
3. Train models
4. Evaluate and save results
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data_loader import DataLoader, normalize_features, create_lag_features
from src.sentiment_analyzer import SentimentAnalyzer
from src.models import SentimentStockModel, TimeSeriesValidator
from src.utils import setup_logging, save_results, get_ticker_stats
from config.config import *

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


def main():
    """Run the complete analysis pipeline."""
    logger.info("=" * 60)
    logger.info("Starting Sentiment-Driven Stock Analysis Pipeline")
    logger.info("=" * 60)
    
    # TODO: Implement full pipeline
    # 1. Load data
    # 2. Analyze sentiment
    # 3. Create features
    # 4. Train models
    # 5. Evaluate and save results
    
    logger.info("Pipeline execution completed")


if __name__ == "__main__":
    main()
