# src/core/data_ingestion.py
from typing import Optional, Dict, Any
import pandas as pd
from datetime import datetime, timedelta
from loguru import logger
from src.core.api_client import AlphaVantageClient
from src.models.data_models import StockDataset, StockDataPoint
from src.utils.security import SecurityManager

class DataIngestionService:
    def __init__(self):
        self.api_client = AlphaVantageClient()
        self.security_manager = SecurityManager()
    
    def fetch_stock_data(self, symbol: str, interval: str = "5min") -> Optional[StockDataset]:
        """Fetch and validate stock data from API"""
        clean_symbol = self.security_manager.sanitize_symbol(symbol)
        if not clean_symbol:
            logger.error(f"Invalid symbol: {symbol}")
            return None
        
        raw_data = self.api_client.get_intraday_data(clean_symbol, interval)
        if not raw_data:
            logger.error(f"Failed to fetch data for {clean_symbol}")
            return None
        
        try:
            return self._parse_alpha_vantage_data(raw_data, clean_symbol, interval)
        except Exception as e:
            logger.error(f"Failed to parse data for {clean_symbol}: {e}")
            return None
    
    def _parse_alpha_vantage_data(self, raw_data: Dict[str, Any], symbol: str, interval: str) -> StockDataset:
        """Parse Alpha Vantage API response into structured data"""
        meta_data = raw_data.get('Meta Data', {})
        time_series_key = f'Time Series ({interval})'
        time_series = raw_data.get(time_series_key, {})
        
        if not time_series:
            raise ValueError(f"No time series data found for {symbol}")
        
        last_refreshed = datetime.fromisoformat(
            meta_data.get('3. Last Refreshed', datetime.now().isoformat())
        )
        
        data_points = []
        for timestamp_str, values in time_series.items():
            try:
                data_point = StockDataPoint(
                    timestamp=datetime.fromisoformat(timestamp_str),
                    open_price=float(values['1. open']),
                    high_price=float(values['2. high']),
                    low_price=float(values['3. low']),
                    close_price=float(values['4. close']),
                    volume=int(values['5. volume'])
                )
                data_points.append(data_point)
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping invalid data point {timestamp_str}: {e}")
                continue
        
        if not data_points:
            raise ValueError(f"No valid data points found for {symbol}")
        
        data_points.sort(key=lambda x: x.timestamp, reverse=True)
        
        return StockDataset(
            symbol=symbol,
            interval=interval,
            last_refreshed=last_refreshed,
            data_points=data_points
        )