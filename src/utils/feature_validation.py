# src/utils/feature_validation.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

def validate_features(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Validate feature quality for ML training"""
    validation_results = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }
    
    if len(X) < 20:
        validation_results['errors'].append(f"Insufficient data: {len(X)} samples")
        validation_results['is_valid'] = False
    
    missing_features = X.isnull().sum()
    if missing_features.any():
        validation_results['errors'].append(f"Missing values in features: {missing_features[missing_features > 0]}")
        validation_results['is_valid'] = False
    
    inf_features = np.isinf(X).sum()
    if inf_features.any():
        validation_results['errors'].append(f"Infinite values in features: {inf_features[inf_features > 0]}")
        validation_results['is_valid'] = False
    
    validation_results['stats'] = {
        'n_samples': len(X),
        'n_features': len(X.columns),
        'target_mean': y.mean(),
        'target_std': y.std()
    }
    
    if validation_results['errors']:
        for error in validation_results['errors']:
            logger.error(f"Feature validation error: {error}")
    
    return validation_results 