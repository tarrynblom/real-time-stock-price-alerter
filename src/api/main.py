from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import datetime
from loguru import logger
import uvicorn

from src.core.prediction_service import PredictionService
from src.core.alert_service import AlertService
from src.core.alerting_engine import AlertType, AlertSeverity, AlertRule
from config.settings import settings

# Initialize FastAPI app
app = FastAPI(
    title="Stock Price Alerter API",
    description="Real-time stock price prediction and alerting microservice",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
prediction_service = PredictionService()
alert_service = AlertService(prediction_service)


# Request/Response Models
class PredictionRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=5, description="Stock symbol")
    interval: str = Field(default="5min", description="Data interval")


class TrainingRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=5, description="Stock symbol")
    interval: str = Field(default="5min", description="Data interval")


class AlertRuleConfig(BaseModel):
    alert_type: str = Field(..., description="Alert type")
    threshold_pct: float = Field(..., description="Threshold percentage")
    severity: str = Field(..., description="Alert severity")
    enabled: bool = Field(default=True, description="Whether rule is enabled")


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_trained: bool
    version: str
    services: Dict[str, Any]
    metrics: Dict[str, Any]


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Stock Price Alerter API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Simple health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        model_trained=prediction_service.model_trainer.is_trained,
        version="1.0.0",
        services={"api": {"status": "healthy"}},
        metrics={
            "models_loaded": 1 if prediction_service.model_trainer.is_trained else 0
        },
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    # This will be enhanced when Prometheus integration is added
    # For now, return basic text metrics
    metrics_text = f"""# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{{method="GET",endpoint="/health",status_code="200"}} 0

# HELP model_trained_status Model training status
# TYPE model_trained_status gauge
model_trained_status {1 if prediction_service.model_trainer.is_trained else 0}

# HELP api_health_status API health status (1=healthy, 0=unhealthy)
# TYPE api_health_status gauge
api_health_status 1
"""
    from fastapi.responses import PlainTextResponse

    return PlainTextResponse(metrics_text)


@app.post("/train")
async def train_model(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Train ML model for a specific stock symbol"""
    try:
        symbol = request.symbol.upper().strip()

        # Run training in background for better UX
        def train_model_task():
            result = prediction_service.train_model(symbol, request.interval)
            logger.info(f"Training completed for {symbol}: {result}")

        background_tasks.add_task(train_model_task)

        return {
            "message": f"Model training started for {symbol}",
            "symbol": symbol,
            "status": "training_in_progress",
        }

    except Exception as e:
        logger.error(f"Training request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict")
async def predict_price(request: PredictionRequest):
    """Get price prediction for a stock symbol"""
    try:
        symbol = request.symbol.upper().strip()

        if not prediction_service.model_trainer.is_trained:
            raise HTTPException(
                status_code=400, detail="Model not trained. Call /train endpoint first."
            )

        result = prediction_service.predict_next_price(symbol, request.interval)

        if not result.get("prediction_successful", True):
            raise HTTPException(
                status_code=500, detail=result.get("error", "Prediction failed")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/alert")
async def check_alerts(request: PredictionRequest):
    """Check for alerts based on price prediction"""
    try:
        symbol = request.symbol.upper().strip()

        if not prediction_service.model_trainer.is_trained:
            raise HTTPException(
                status_code=400, detail="Model not trained. Call /train endpoint first."
            )

        result = alert_service.check_and_alert(symbol, request.interval)

        if not result.get("success"):
            raise HTTPException(
                status_code=500, detail=result.get("error", "Alert check failed")
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alert request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
