# src/core/data_preprocessing.py
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from loguru import logger
from src.models.data_models import StockDataset

class DataPreprocessor:
    def __init__(self):
        self.required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    def preprocess_dataset(self, dataset: StockDataset) -> Optional[pd.DataFrame]:
        """Complete preprocessing pipeline"""
        try:
            df = dataset.to_dataframe()
            
            if not self._validate_data_quality(df):
                logger.error(f"Data quality validation failed for {dataset.symbol}")
                return None
            
            df = self._clean_data(df)
            df = self._add_technical_features(df)
            df = self._handle_missing_values(df)
            
            logger.info(f"Preprocessed {len(df)} data points for {dataset.symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Preprocessing failed for {dataset.symbol}: {e}")
            return None
    
    def _validate_data_quality(self, df: pd.DataFrame) -> bool:
        """Validate data quality and completeness"""
        if df.empty:
            logger.error("Empty DataFrame")
            return False
        
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        if len(df) < 20:
            logger.error(f"Insufficient data points: {len(df)}")
            return False
        
        if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
            logger.error("Invalid price data found (negative or zero values)")
            return False
        
        return True
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data"""
        df_clean = df.copy()
        
        df_clean = df_clean[~df_clean.index.duplicated(keep='first')]
        
        invalid_rows = (
            (df_clean['high'] < df_clean['low']) |
            (df_clean['high'] < df_clean['open']) |
            (df_clean['high'] < df_clean['close']) |
            (df_clean['low'] > df_clean['open']) |
            (df_clean['low'] > df_clean['close'])
        )
        
        if invalid_rows.any():
            logger.warning(f"Removing {invalid_rows.sum()} invalid price relationships")
            df_clean = df_clean[~invalid_rows]
        
        return df_clean
    
    def _add_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis features"""
        df_enhanced = df.copy()
        
        # Price-based features
        df_enhanced['price_change'] = df_enhanced['close'].pct_change()
        df_enhanced['price_volatility'] = df_enhanced['close'].rolling(window=5).std()
        df_enhanced['high_low_pct'] = (df_enhanced['high'] - df_enhanced['low']) / df_enhanced['close']
        
        # Volume-based features
        df_enhanced['volume_ma'] = df_enhanced['volume'].rolling(window=10).mean()
        df_enhanced['volume_ratio'] = df_enhanced['volume'] / df_enhanced['volume_ma']
        
        # Moving averages
        df_enhanced['ma_5'] = df_enhanced['close'].rolling(window=5).mean()
        df_enhanced['ma_10'] = df_enhanced['close'].rolling(window=10).mean()
        df_enhanced['ma_20'] = df_enhanced['close'].rolling(window=20).mean()
        
        # Technical indicators
        df_enhanced['rsi'] = self._calculate_rsi(df_enhanced['close'])
        df_enhanced['macd'] = self._calculate_macd(df_enhanced['close'])
        
        return df_enhanced
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """Calculate MACD indicator"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        return macd
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        df_filled = df.copy()
        
        price_columns = ['open', 'high', 'low', 'close']
        df_filled[price_columns] = df_filled[price_columns].fillna(method='ffill')
        
        technical_columns = ['rsi', 'macd', 'price_volatility']
        for col in technical_columns:
            if col in df_filled.columns:
                df_filled[col] = df_filled[col].interpolate(method='linear')
        
        numeric_columns = df_filled.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            df_filled[col] = df_filled[col].fillna(df_filled[col].median())
        
        return df_filled