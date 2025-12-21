"""
Sentiment analysis module for processing tweets.

Uses Twitter-roBERTa fine-tuned model for extracting sentiment scores.
"""

import torch
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analyze tweet sentiment using transformer models."""
    
    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"):
        """
        Initialize sentiment analyzer with pre-trained model.
        
        Args:
            model_name: HuggingFace model ID for sentiment analysis
        """
        logger.info(f"Loading sentiment model: {model_name}")
        
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
        except ImportError:
            logger.error("transformers library required. Install with: pip install transformers")
            raise
    
    def analyze(self, tweets: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Analyze sentiment for a batch of tweets.
        
        Args:
            tweets: List of tweet texts
            batch_size: Number of tweets to process at once
            
        Returns:
            Array of sentiment scores in range [-1, 1]
        """
        sentiments = []
        
        for i in range(0, len(tweets), batch_size):
            batch = tweets[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
            
            # Convert to sentiment score: negative=-1, neutral=0, positive=1
            for prob in probabilities:
                sentiment_score = (prob[2] - prob[0])  # positive - negative
                sentiments.append(sentiment_score)
        
        return np.array(sentiments)
    
    def analyze_with_confidence(self, tweets: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment with confidence scores.
        
        Args:
            tweets: List of tweet texts
            
        Returns:
            List of dicts with 'sentiment' and 'confidence' keys
        """
        results = []
        
        inputs = self.tokenizer(
            tweets,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        
        for prob in probabilities:
            sentiment_score = (prob[2] - prob[0])
            confidence = max(prob)
            results.append({
                'sentiment': sentiment_score,
                'confidence': float(confidence),
                'negative': float(prob[0]),
                'neutral': float(prob[1]),
                'positive': float(prob[2])
            })
        
        return results
