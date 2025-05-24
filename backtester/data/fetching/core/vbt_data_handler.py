#!/usr/bin/env python3
"""VBT Data Handler - Standardized interface for VBT data structures.

This module provides a consistent way to work with VBT data, handling all the
various formats and structures that VBT can produce.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import vectorbtpro as vbt

logger = logging.getLogger(__name__)


class VBTDataHandler:
    """Standardized handler for VBT data access and manipulation."""
    
    @staticmethod
    def extract_ohlcv(data: vbt.Data) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Extract OHLCV data from VBT Data object using the most appropriate method.
        
        Returns:
            Dict mapping feature names to DataFrames with symbols as columns
            Returns None if extraction fails
        """
        if data is None:
            return None
            
        # Try direct attribute access first (most common case)
        if hasattr(data, 'open') and data.open is not None:
            try:
                return {
                    'open': data.open,
                    'high': data.high,
                    'low': data.low,
                    'close': data.close,
                    'volume': data.volume if hasattr(data, 'volume') else None
                }
            except Exception as e:
                logger.debug(f"Direct attribute access failed: {e}")
        
        # Try capitalized attributes
        if hasattr(data, 'Open') and data.Open is not None:
            try:
                return {
                    'open': data.Open,
                    'high': data.High,
                    'low': data.Low,
                    'close': data.Close,
                    'volume': data.Volume if hasattr(data, 'Volume') else None
                }
            except Exception as e:
                logger.debug(f"Capitalized attribute access failed: {e}")
        
        # Try extracting from underlying DataFrame
        try:
            df = data.get()
            if isinstance(df.columns, pd.MultiIndex):
                return VBTDataHandler._extract_from_multiindex(df)
            else:
                return VBTDataHandler._extract_from_simple_columns(df)
        except Exception as e:
            logger.debug(f"DataFrame extraction failed: {e}")
            
        return None
    
    @staticmethod
    def _extract_from_multiindex(df: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
        """Extract OHLCV from MultiIndex DataFrame."""
        try:
            # Detect which level contains features
            level_0_values = set(df.columns.get_level_values(0))
            level_1_values = set(df.columns.get_level_values(1))
            
            feature_names = ['open', 'high', 'low', 'close', 'volume', 
                           'Open', 'High', 'Low', 'Close', 'Volume']
            
            features_in_level_0 = any(name in level_0_values for name in feature_names)
            features_in_level_1 = any(name in level_1_values for name in feature_names)
            
            ohlcv_data = {}
            
            if features_in_level_1:
                # Features in level 1, symbols in level 0
                for feature in ['open', 'high', 'low', 'close', 'volume']:
                    for feature_case in [feature, feature.capitalize()]:
                        if feature_case in level_1_values:
                            ohlcv_data[feature] = df.xs(feature_case, axis=1, level=1)
                            break
            elif features_in_level_0:
                # Features in level 0, symbols in level 1
                for feature in ['open', 'high', 'low', 'close', 'volume']:
                    for feature_case in [feature, feature.capitalize()]:
                        if feature_case in level_0_values:
                            ohlcv_data[feature] = df.xs(feature_case, axis=1, level=0)
                            break
            
            return ohlcv_data if len(ohlcv_data) >= 4 else None
            
        except Exception as e:
            logger.debug(f"MultiIndex extraction failed: {e}")
            return None
    
    @staticmethod
    def _extract_from_simple_columns(df: pd.DataFrame) -> Optional[Dict[str, pd.DataFrame]]:
        """Extract OHLCV from simple column structure."""
        try:
            column_mapping = {}
            for col in df.columns:
                col_lower = str(col).lower()
                if 'open' in col_lower and 'open' not in column_mapping:
                    column_mapping['open'] = col
                elif 'high' in col_lower and 'high' not in column_mapping:
                    column_mapping['high'] = col
                elif 'low' in col_lower and 'low' not in column_mapping:
                    column_mapping['low'] = col
                elif 'close' in col_lower and 'close' not in column_mapping:
                    column_mapping['close'] = col
                elif ('volume' in col_lower or 'vol' in col_lower) and 'volume' not in column_mapping:
                    column_mapping['volume'] = col
            
            if len(column_mapping) >= 4:  # At least OHLC
                ohlcv_data = {}
                for feature, col_name in column_mapping.items():
                    ohlcv_data[feature] = df[[col_name]]
                return ohlcv_data
                
            return None
            
        except Exception as e:
            logger.debug(f"Simple column extraction failed: {e}")
            return None
    
    @staticmethod
    def get_symbol_data(data: vbt.Data, symbol: str, feature: str = 'close') -> Optional[pd.Series]:
        """
        Get data for a specific symbol and feature.
        
        Args:
            data: VBT Data object
            symbol: Symbol to extract
            feature: Feature to extract (default: 'close')
            
        Returns:
            Series with the symbol's data or None
        """
        try:
            # Try direct access
            feature_data = getattr(data, feature, None)
            if feature_data is None:
                feature_data = getattr(data, feature.capitalize(), None)
                
            if feature_data is not None:
                if len(data.symbols) > 1:
                    return feature_data[symbol].dropna()
                else:
                    return feature_data.dropna()
                    
            # Try through OHLCV extraction
            ohlcv = VBTDataHandler.extract_ohlcv(data)
            if ohlcv and feature in ohlcv:
                df = ohlcv[feature]
                if symbol in df.columns:
                    return df[symbol].dropna()
                    
        except Exception as e:
            logger.debug(f"Error getting symbol data: {e}")
            
        return None
    
    @staticmethod
    def get_date_range(data: vbt.Data, symbol: Optional[str] = None) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get the date range for the data or a specific symbol."""
        try:
            if symbol:
                symbol_data = VBTDataHandler.get_symbol_data(data, symbol)
                if symbol_data is not None and len(symbol_data) > 0:
                    return symbol_data.index[0], symbol_data.index[-1]
            
            # Fallback to wrapper index
            if hasattr(data, 'wrapper') and hasattr(data.wrapper, 'index'):
                return data.wrapper.index[0], data.wrapper.index[-1]
                
        except Exception as e:
            logger.debug(f"Error getting date range: {e}")
            
        raise ValueError("Could not determine date range")
    
    @staticmethod
    def create_from_dict(symbol_dict: Dict[str, pd.DataFrame]) -> Optional[vbt.Data]:
        """
        Create VBT Data from a dictionary of symbol DataFrames.
        
        Args:
            symbol_dict: Dict mapping symbol names to DataFrames with OHLCV columns
            
        Returns:
            VBT Data object or None
        """
        try:
            # Ensure all DataFrames have consistent column names (capitalized)
            normalized_dict = {}
            
            for symbol, df in symbol_dict.items():
                normalized_df = pd.DataFrame(index=df.index)
                
                # Map columns to standard names
                for col in df.columns:
                    col_lower = str(col).lower()
                    if 'open' in col_lower:
                        normalized_df['Open'] = df[col]
                    elif 'high' in col_lower:
                        normalized_df['High'] = df[col]
                    elif 'low' in col_lower:
                        normalized_df['Low'] = df[col]
                    elif 'close' in col_lower:
                        normalized_df['Close'] = df[col]
                    elif 'volume' in col_lower or 'vol' in col_lower:
                        normalized_df['Volume'] = df[col]
                
                if len(normalized_df.columns) >= 4:  # At least OHLC
                    normalized_dict[symbol] = normalized_df
            
            if normalized_dict:
                return vbt.Data.from_data(normalized_dict)
                
        except Exception as e:
            logger.error(f"Error creating VBT Data from dict: {e}")
            
        return None 