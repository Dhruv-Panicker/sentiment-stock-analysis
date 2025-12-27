"""
FastAPI Application for Stock Price Prediction
Serves pre-trained Random Forest models for TSLA, AMD, NIO
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List
import numpy as np
import logging
from datetime import datetime
import sys
from pathlib import Path
import uvicorn

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.prediction_service import PredictionService
from src.config import Settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load settings
settings = Settings()

# Initialize FastAPI app
app = FastAPI(
    title="Stock Sentiment Prediction API",
    description="ML model API for predicting stock price movements based on sentiment analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
try:
    prediction_service = PredictionService(models_dir=settings.MODELS_DIR)
    logger.info(" Prediction service initialized successfully")
except Exception as e:
    logger.error(f" Failed to initialize prediction service: {e}")
    prediction_service = None


# Pydantic Models (Request/Response Schemas)


class PredictionRequest(BaseModel):
    """Request schema for making predictions"""
    ticker: str = Field(..., description="Stock ticker (TSLA, AMD, NIO)")
    sentiment_mean: float = Field(..., description="Average sentiment score (-1 to 1)")
    sentiment_momentum: float = Field(..., description="Day-to-day sentiment change")


class PredictionResponse(BaseModel):
    """Response schema with prediction and interpretation"""
    ticker: str = Field(..., description="Stock ticker symbol")
    prediction: float = Field(..., description="Predicted price change percentage (%)")
    prediction_interpretation: str = Field(..., description="Human-readable interpretation (e.g., 'Price Expected to Rise | Expected: +3.50% (significant change)')")
    confidence: str = Field(..., description="Prediction confidence level (low, medium, high)")
    timestamp: str = Field(..., description="ISO 8601 timestamp of prediction")
    model_version: str = Field(default="1.0", description="Model version used for prediction")


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions"""
    predictions: List[PredictionRequest]


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions"""
    predictions: List[PredictionResponse] = Field(..., description="List of individual predictions with interpretations")
    total_predictions: int = Field(..., description="Total number of predictions made")
    timestamp: str = Field(..., description="ISO 8601 timestamp of batch request")


class HealthResponse(BaseModel):
    #Health check response
    status: str
    service: str = "Stock Prediction API"
    version: str = "1.0.0"
    models_loaded: bool
    timestamp: str


class ModelInfoResponse(BaseModel):
    #Model metadata response
    available_tickers: List[str]
    model_type: str
    feature_names: List[str]
    model_version: str
    trained_on: str
    input_requirements: Dict


# Endpoints

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint to verify API is running and models are loaded
    """
    logger.info("Health check request received")
    
    models_loaded = prediction_service is not None and prediction_service.is_ready()
    
    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        timestamp=datetime.now().isoformat()
    )


@app.get("/model_info", response_model=ModelInfoResponse, tags=["System"])
async def get_model_info():
    """
    Get information about loaded models and requirements
    """
    logger.info("Model info request received")
    
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")
    
    return ModelInfoResponse(
        available_tickers=prediction_service.available_tickers,
        model_type="Random Forest Regressor",
        feature_names=["sentiment_mean", "sentiment_momentum"],
        model_version="1.0",
        trained_on="Phase 4 - Multi-Stock Ensemble",
        input_requirements={
            "ticker": "One of: TSLA, AMD, NIO",
            "sentiment_mean": "Float between -1.0 and 1.0",
            "sentiment_momentum": "Float between -1.0 and 1.0"
        }
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Make a single prediction based on sentiment features
    
    **Parameters:**
    - `ticker`: Stock symbol (TSLA, AMD, NIO)
    - `sentiment_mean`: Average sentiment (-1 to 1)
    - `sentiment_momentum`: Sentiment momentum/change
    
    **Returns:**
    - Predicted price change percentage
    - Confidence level
    """
    logger.info(f"Prediction request for {request.ticker}")
    
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")
    
    try:
        # Make prediction
        prediction, confidence = prediction_service.predict(
            ticker=request.ticker,
            sentiment_mean=request.sentiment_mean,
            sentiment_momentum=request.sentiment_momentum
        )
        
        interpretation = PredictionService.get_prediction_interpretation(prediction)
        
        logger.info(f" Prediction made for {request.ticker}: {prediction:.4f}%")
        
        response_data = {
            "ticker": request.ticker,
            "prediction": round(float(prediction), 2),
            "prediction_interpretation": interpretation,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat(),
            "model_version": "1.0"
        }
        return JSONResponse(content=response_data)
    
    except ValueError as e:
        logger.error(f"❌ Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Make multiple predictions in a single request
    
    **Parameters:**
    - `predictions`: List of PredictionRequest objects
    
    **Returns:**
    - List of PredictionResponse objects
    """
    logger.info(f"Batch prediction request for {len(request.predictions)} samples")
    
    if prediction_service is None:
        raise HTTPException(status_code=503, detail="Prediction service not initialized")
    
    results = []
    
    try:
        for pred_request in request.predictions:
            prediction, confidence = prediction_service.predict(
                ticker=pred_request.ticker,
                sentiment_mean=pred_request.sentiment_mean,
                sentiment_momentum=pred_request.sentiment_momentum
            )
            
            interpretation = PredictionService.get_prediction_interpretation(prediction)
            
            results.append({
                "ticker": pred_request.ticker,
                "prediction": round(float(prediction), 2),
                "prediction_interpretation": interpretation,
                "confidence": confidence,
                "timestamp": datetime.now().isoformat(),
                "model_version": "1.0"
            })
        
        return JSONResponse(content={
            "predictions": results,
            "total_predictions": len(results),
            "timestamp": datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"❌ Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail="Batch prediction failed")


@app.get("/", tags=["Root"])
async def root():
    """Welcome endpoint with API documentation"""
    return {
        "message": "Stock Sentiment Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "model_info": "/model_info"
    }



# Error Handlers

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle validation errors"""
    logger.error(f"Value error: {exc}")
    return {"detail": str(exc), "status": "error"}


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected errors"""
    logger.error(f"Unexpected error: {exc}")
    return {"detail": "Internal server error", "status": "error"}


if __name__ == "__main__":
    logger.info(f"🚀 Starting API server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info"
    )
