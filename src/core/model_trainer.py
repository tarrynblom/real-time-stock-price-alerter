# src/core/model_trainer.py
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from loguru import logger
import joblib
import os

class ModelTrainer:
    """Simple, focused model training for stock price prediction"""
    
    def __init__(self, model_type: str = 'linear', test_size: float = 0.2, random_state: int = 42):
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.training_metrics = {}
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Train the model and return performance metrics"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )
        
        self.model = self._get_model()
        
        logger.info(f"Training {self.model_type} model with {len(X_train)} samples")
        self.model.fit(X_train, y_train)
        
        metrics = self._evaluate_model(X_train, X_test, y_train, y_test)
        self.training_metrics = metrics
        self.is_trained = True
        
        logger.info(f"Model training completed. R² Score: {metrics['r2_score']:.4f}")
        return metrics
    
    def _get_model(self):
        """Get the appropriate model based on model_type"""
        if self.model_type == 'linear':
            return LinearRegression()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _evaluate_model(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                       y_train: pd.Series, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate model performance"""
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_test_pred),
            'mse': mean_squared_error(y_test, y_test_pred),
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'n_features': len(X_train.columns)
        }
        
        actual_direction = np.sign(y_test.diff().dropna())
        predicted_direction = np.sign(pd.Series(y_test_pred).diff().dropna())
        metrics['directional_accuracy'] = (actual_direction == predicted_direction).mean()
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions with the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'metrics': self.training_metrics,
            'model_type': self.model_type
        }, filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.training_metrics = model_data['metrics']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        logger.info(f"Model loaded from {filepath}") 