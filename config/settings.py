# config/settings.py
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", env_file_encoding="utf-8")
    
    # API Configuration
    alpha_vantage_api_key: str = "test_key"  # Default for testing
    financial_modeling_prep_api_key: Optional[str] = None
    api_base_url: str = "https://www.alphavantage.co/query"
    
    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    
    # Application Settings
    default_stock_symbol: str = "AAPL"
    prediction_threshold: float = 0.01
    cache_ttl: int = 300  # 5 minutes
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/stock_alerter.log"

settings = Settings()