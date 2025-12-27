"""
Configuration Module
Settings for the FastAPI application
"""

from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or .env file
    """
    
    # API Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Model Configuration
    MODELS_DIR: str = "models"
    
    # API Documentation
    API_TITLE: str = "Stock Sentiment Prediction API"
    API_VERSION: str = "1.0.0"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
