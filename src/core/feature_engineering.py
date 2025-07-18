# src/core/feature_engineering.py
import pandas as pd
import numpy as np

from loguru import logger


class FeatureEngineer:
    """Simple, focused feature engineering for time-series prediction"""

    def __init__(self, lookback_periods: int = 5):
        self.lookback_periods = lookback_periods

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lagged features for time-series prediction"""
        if df.empty or len(df) < self.lookback_periods:
            raise ValueError(
                f"Insufficient data: need at least {self.lookback_periods} periods"
            )

        features_df = df.copy()
        features_df = self._add_lagged_features(features_df)
        features_df = self._add_basic_indicators(features_df)
        features_df = self._clean_features(features_df)

        logger.info(
            f"Created {len(features_df.columns)} features from {len(df)} data points"
        )
        return features_df

    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged price features (primary ML features)"""
        for i in range(1, self.lookback_periods + 1):
            df[f"close_lag_{i}"] = df["close"].shift(i)
            df[f"volume_lag_{i}"] = df["volume"].shift(i)
        return df

    def _add_basic_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add essential technical indicators"""
        # Price momentum
        df["price_change"] = df["close"].pct_change()
        df["price_change_lag_1"] = df["price_change"].shift(1)

        # Volatility
        df["volatility"] = df["close"].rolling(window=5).std()

        # Simple moving average
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["price_to_sma"] = df["close"] / df["sma_5"]

        return df

    def _clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare features for ML"""
        df_clean = df.dropna()

        df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()

        if df_clean.empty:
            raise ValueError("No valid data remaining after feature cleaning")

        return df_clean
