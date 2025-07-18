from pydantic import BaseModel
from typing import List, Dict, Any
from enum import Enum


class AlertTypeEnum(str, Enum):
    PRICE_INCREASE = "price_increase"
    PRICE_DECREASE = "price_decrease"
    VOLATILITY_SPIKE = "volatility_spike"


class AlertSeverityEnum(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    predicted_price: float
    price_change: float
    price_change_pct: float
    prediction_time: str
    confidence: str


class AlertResponse(BaseModel):
    type: AlertTypeEnum
    severity: AlertSeverityEnum
    message: str
    timestamp: str


class AlertCheckResponse(BaseModel):
    success: bool
    symbol: str
    prediction: PredictionResponse
    alerts_triggered: int
    alerts: List[AlertResponse]
    notification_results: Dict[str, Any]
