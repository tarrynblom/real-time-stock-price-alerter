# src/utils/feature_selection.py
import pandas as pd
from typing import List, Tuple

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Extract feature columns for ML training"""
    feature_columns = [
        col for col in df.columns 
        if col.startswith(('close_lag_', 'volume_lag_', 'price_change_lag_')) 
        or col in ['volatility', 'price_to_sma']
    ]
    return feature_columns

def get_target_column() -> str:
    """Get target column name"""
    return 'close'

def prepare_ml_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target for ML training"""
    feature_cols = get_feature_columns(df)
    target_col = get_target_column()
    
    if not feature_cols:
        raise ValueError("No feature columns found")
    
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    X = df[feature_cols]
    y = df[target_col]
    
    return X, y 