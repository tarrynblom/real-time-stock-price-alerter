# src/models/data_models.py
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


class StockDataPoint(BaseModel):
    timestamp: datetime
    open_price: float = Field(..., gt=0)
    high_price: float = Field(..., gt=0)
    low_price: float = Field(..., gt=0)
    close_price: float = Field(..., gt=0)
    volume: int = Field(..., ge=0)

    @field_validator("high_price")
    @classmethod
    def validate_high_price(cls, v, info):
        if info.data and "low_price" in info.data and v < info.data["low_price"]:
            raise ValueError("High price must be >= low price")
        return v


class StockDataset(BaseModel):
    symbol: str
    interval: str
    last_refreshed: datetime
    data_points: List[StockDataPoint]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis"""
        data = []
        for point in self.data_points:
            data.append(
                {
                    "timestamp": point.timestamp,
                    "open": point.open_price,
                    "high": point.high_price,
                    "low": point.low_price,
                    "close": point.close_price,
                    "volume": point.volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df.sort_index()


class PredictionRequest(BaseModel):
    symbol: str
    lookback_periods: int = Field(default=20, ge=5, le=100)
    prediction_horizon: int = Field(default=1, ge=1, le=5)

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v):
        return v.upper().strip()
