"""
CLI Tool for Batch Predictions
Make predictions from command line using CSV input
"""

import argparse
import csv
import json
import logging
from pathlib import Path
from prediction_service import PredictionService


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def predict_from_csv(input_file: str, output_file: str, models_dir: str = "models"):
    """
    Read predictions from CSV and write results to output file
    
    Args:
        input_file: Path to input CSV with columns: ticker, sentiment_mean, sentiment_momentum
        output_file: Path to output JSON file
        models_dir: Directory containing models
    """
    logger.info(f"Loading prediction service from {models_dir}...")
    service = PredictionService(models_dir=models_dir)
    
    if not service.is_ready():
        logger.error("❌ Models not ready")
        return
    
    logger.info(f"Reading predictions from {input_file}...")
    
    predictions_list = []
    
    try:
        with open(input_file, 'r') as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                try:
                    ticker = row['ticker'].strip().upper()
                    sentiment_mean = float(row['sentiment_mean'])
                    sentiment_momentum = float(row['sentiment_momentum'])
                    
                    prediction, confidence = service.predict(
                        ticker=ticker,
                        sentiment_mean=sentiment_mean,
                        sentiment_momentum=sentiment_momentum
                    )
                    
                    predictions_list.append({
                        'row': idx + 1,
                        'ticker': ticker,
                        'sentiment_mean': sentiment_mean,
                        'sentiment_momentum': sentiment_momentum,
                        'prediction': round(prediction, 4),
                        'confidence': confidence,
                        'status': 'success'
                    })
                    
                    logger.info(f"✅ Row {idx + 1}: {ticker} prediction = {prediction:.4f}%")
                
                except Exception as e:
                    logger.error(f"❌ Row {idx + 1}: {e}")
                    predictions_list.append({
                        'row': idx + 1,
                        'status': 'error',
                        'error': str(e)
                    })
    
    except FileNotFoundError:
        logger.error(f"❌ Input file not found: {input_file}")
        return
    
    # Write results
    logger.info(f"Writing results to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump({
            'total_predictions': len(predictions_list),
            'successful': sum(1 for p in predictions_list if p['status'] == 'success'),
            'failed': sum(1 for p in predictions_list if p['status'] == 'error'),
            'predictions': predictions_list
        }, f, indent=2)
    
    logger.info(f"✅ Results saved to {output_file}")
    logger.info(f"Total: {len(predictions_list)}, Success: {sum(1 for p in predictions_list if p['status'] == 'success')}")


def predict_single(ticker: str, sentiment_mean: float, sentiment_momentum: float, models_dir: str = "models"):
    """
    Make a single prediction and print result
    
    Args:
        ticker: Stock ticker
        sentiment_mean: Average sentiment
        sentiment_momentum: Sentiment momentum
        models_dir: Directory containing models
    """
    logger.info("Loading prediction service...")
    service = PredictionService(models_dir=models_dir)
    
    if not service.is_ready():
        logger.error("❌ Models not ready")
        return
    
    try:
        prediction, confidence = service.predict(
            ticker=ticker,
            sentiment_mean=sentiment_mean,
            sentiment_momentum=sentiment_momentum
        )
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"Ticker:              {ticker}")
        print(f"Sentiment Mean:      {sentiment_mean:.4f}")
        print(f"Sentiment Momentum:  {sentiment_momentum:.4f}")
        print(f"Predicted Change:    {prediction:.4f}%")
        print(f"Confidence:          {confidence}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Stock Price Prediction CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single prediction
  python inference.py --single TSLA 0.5 0.1
  
  # Batch prediction from CSV
  python inference.py --batch input.csv output.json
        """
    )
    
    parser.add_argument('--models', type=str, default='models', help='Path to models directory')
    
    # Single prediction mode
    parser.add_argument('--single', nargs=3, metavar=('TICKER', 'SENTIMENT_MEAN', 'SENTIMENT_MOMENTUM'),
                       help='Make single prediction')
    
    # Batch prediction mode
    parser.add_argument('--batch', nargs=2, metavar=('INPUT_CSV', 'OUTPUT_JSON'),
                       help='Make batch predictions from CSV')
    
    args = parser.parse_args()
    
    if args.single:
        try:
            ticker = args.single[0]
            sentiment_mean = float(args.single[1])
            sentiment_momentum = float(args.single[2])
            predict_single(ticker, sentiment_mean, sentiment_momentum, args.models)
        except ValueError as e:
            logger.error(f"Invalid input: {e}")
    
    elif args.batch:
        predict_from_csv(args.batch[0], args.batch[1], args.models)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
