"""
API Testing Module
Unit and integration tests for the FastAPI application
"""

import pytest
import json
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app
from src.prediction_service import PredictionService


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for health check endpoint"""
    
    def test_health_check_success(self, client):
        """Test health check returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
    
    def test_health_check_schema(self, client):
        """Test health check response has required fields"""
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "service" in data
        assert "version" in data
        assert "models_loaded" in data


class TestModelInfoEndpoint:
    """Tests for model info endpoint"""
    
    def test_model_info_success(self, client):
        """Test model info endpoint returns 200"""
        response = client.get("/model_info")
        assert response.status_code == 200
    
    def test_model_info_schema(self, client):
        """Test model info has required fields"""
        response = client.get("/model_info")
        data = response.json()
        assert "available_tickers" in data
        assert "model_type" in data
        assert "feature_names" in data
        assert "model_version" in data
    
    def test_available_tickers(self, client):
        """Test available tickers are correct"""
        response = client.get("/model_info")
        data = response.json()
        assert set(data["available_tickers"]) == {"TSLA", "AMD", "NIO"}


class TestPredictEndpoint:
    """Tests for single prediction endpoint"""
    
    def test_predict_valid_input(self, client):
        """Test prediction with valid input"""
        payload = {
            "ticker": "TSLA",
            "sentiment_mean": 0.5,
            "sentiment_momentum": 0.1
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "TSLA"
        assert isinstance(data["prediction"], float)
        assert data["confidence"] in ["low", "medium", "high"]
        assert "timestamp" in data
    
    def test_predict_invalid_ticker(self, client):
        """Test prediction with invalid ticker"""
        payload = {
            "ticker": "INVALID",
            "sentiment_mean": 0.5,
            "sentiment_momentum": 0.1
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
    
    def test_predict_sentiment_out_of_range(self, client):
        """Test prediction with sentiment out of range"""
        payload = {
            "ticker": "TSLA",
            "sentiment_mean": 1.5,  # Invalid: > 1.0
            "sentiment_momentum": 0.1
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 400
    
    def test_predict_missing_field(self, client):
        """Test prediction with missing required field"""
        payload = {
            "ticker": "TSLA",
            "sentiment_mean": 0.5
            # Missing sentiment_momentum
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_predict_all_tickers(self, client):
        """Test prediction for all available tickers"""
        for ticker in ["TSLA", "AMD", "NIO"]:
            payload = {
                "ticker": ticker,
                "sentiment_mean": 0.3,
                "sentiment_momentum": 0.05
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["ticker"] == ticker


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint"""
    
    def test_batch_predict_valid(self, client):
        """Test batch prediction with valid inputs"""
        payload = {
            "predictions": [
                {
                    "ticker": "TSLA",
                    "sentiment_mean": 0.5,
                    "sentiment_momentum": 0.1
                },
                {
                    "ticker": "AMD",
                    "sentiment_mean": 0.3,
                    "sentiment_momentum": -0.05
                },
                {
                    "ticker": "NIO",
                    "sentiment_mean": 0.7,
                    "sentiment_momentum": 0.15
                }
            ]
        }
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] == 3
        assert len(data["predictions"]) == 3
        for result in data["predictions"]:
            assert "ticker" in result
            assert "prediction" in result
            assert "confidence" in result
    
    def test_batch_predict_empty(self, client):
        """Test batch prediction with empty list"""
        payload = {"predictions": []}
        response = client.post("/predict_batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_predictions"] == 0


class TestRootEndpoint:
    """Tests for root endpoint"""
    
    def test_root_success(self, client):
        """Test root endpoint returns 200"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


class TestPredictionService:
    """Tests for PredictionService class"""
    
    def test_service_initialization(self):
        """Test service initializes correctly"""
        service = PredictionService()
        assert service is not None
        assert len(service.available_tickers) == 3
    
    def test_service_validation_invalid_ticker(self):
        """Test validation with invalid ticker"""
        service = PredictionService()
        with pytest.raises(ValueError):
            service.validate_input("INVALID", 0.5, 0.1)
    
    def test_service_validation_sentiment_range(self):
        """Test validation for sentiment range"""
        service = PredictionService()
        with pytest.raises(ValueError):
            service.validate_input("TSLA", 1.5, 0.1)  # Invalid range
    
    def test_service_validation_valid_input(self):
        """Test validation passes with valid input"""
        service = PredictionService()
        # Should not raise exception
        service.validate_input("TSLA", 0.5, 0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
