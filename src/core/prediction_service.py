# src/core/prediction_service.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger

from src.core.data_ingestion import DataIngestionService
from src.core.data_preprocessing import DataPreprocessor
from src.core.feature_engineering import FeatureEngineer
from src.core.model_trainer import ModelTrainer
from src.utils.feature_selection import prepare_ml_data
from src.utils.feature_validation import validate_features


class PredictionService:
    """Orchestrates the complete prediction pipeline"""

    def __init__(self, lookback_periods: int = 5):
        self.data_ingestion = DataIngestionService()
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer(lookback_periods)
        self.model_trainer = ModelTrainer()
        self.lookback_periods = lookback_periods

    def train_model(self, symbol: str, interval: str = "5min") -> Dict[str, Any]:
        """Complete training pipeline for a stock symbol"""
        try:
            logger.info(f"Starting model training for {symbol}")
            dataset = self.data_ingestion.fetch_stock_data(symbol, interval)
            if not dataset:
                raise ValueError(f"Failed to fetch data for {symbol}")

            df = self.preprocessor.preprocess_dataset(dataset)
            if df is None:
                raise ValueError(f"Failed to preprocess data for {symbol}")

            features_df = self.feature_engineer.create_features(df)

            X, y = prepare_ml_data(features_df)

            validation_results = validate_features(X, y)
            if not validation_results["is_valid"]:
                raise ValueError(
                    f"Feature validation failed: {validation_results['errors']}"
                )

            metrics = self.model_trainer.train(X, y)

            logger.info(f"Model training completed for {symbol}")
            logger.info(f"Performance metrics: {metrics}")

            return {
                "symbol": symbol,
                "training_completed": True,
                "metrics": metrics,
                "feature_info": validation_results["stats"],
            }

        except Exception as e:
            logger.error(f"Model training failed for {symbol}: {e}")
            return {"symbol": symbol, "training_completed": False, "error": str(e)}

    def predict_next_price(self, symbol: str, interval: str = "5min") -> Dict[str, Any]:
        """Predict next price for a stock symbol"""
        try:
            if not self.model_trainer.is_trained:
                raise ValueError("Model must be trained before making predictions")

            dataset = self.data_ingestion.fetch_stock_data(symbol, interval)
            if not dataset:
                raise ValueError(f"Failed to fetch data for {symbol}")

            df = self.preprocessor.preprocess_dataset(dataset)
            if df is None:
                raise ValueError(f"Failed to preprocess data for {symbol}")

            features_df = self.feature_engineer.create_features(df)

            X, _ = prepare_ml_data(features_df)
            latest_features = X.iloc[-1:]  # Get most recent features

            prediction = self.model_trainer.predict(latest_features)[0]
            current_price = df["close"].iloc[-1]

            price_change = prediction - current_price
            price_change_pct = (price_change / current_price) * 100

            prediction_result = {
                "symbol": symbol,
                "current_price": current_price,
                "predicted_price": prediction,
                "price_change": price_change,
                "price_change_pct": price_change_pct,
                "prediction_time": datetime.now().isoformat(),
                "confidence": "medium",  # Simple confidence level
            }

            logger.info(
                f"Prediction for {symbol}: {current_price:.2f} -> {prediction:.2f} ({price_change_pct:+.2f}%)"
            )
            return prediction_result

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            return {"symbol": symbol, "prediction_successful": False, "error": str(e)}
