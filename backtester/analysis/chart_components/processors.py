"""
Data and Indicator Processors for Chart Components

This module contains processors responsible for extracting, cleaning, and preparing
data from VectorBT objects for chart rendering. It implements the IDataProcessor
interface defined in base.py.

Module Structure:
================
1. DataProcessor: Extracts and cleans OHLCV data from VectorBT Data objects
2. IndicatorProcessor: Categorizes and processes technical indicators

Integration with Chart System:
=============================
These processors are used by the TradingChartsEngine to prepare data before
passing it to renderers. They ensure data quality and consistency across
all chart types.

Data Flow:
==========
VectorBT Objects → DataProcessor → Clean OHLCV Data → Renderers
                                 ↘
                                   Portfolio Equity/Returns
                                   
Strategy Indicators → IndicatorProcessor → Categorized Indicators → Renderers
                                         ↘
                                           Price Overlays
                                           Volume Overlays  
                                           Separate Subplots

Related Modules:
===============
- base.py: Defines IDataProcessor interface and BaseDataProcessor
- renderers.py: Uses processed data for visualization
- trading_charts.py: Orchestrates processors in the main engine
"""

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum

from backtester.analysis.chart_components.base import BaseDataProcessor, ProcessorConfig
from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


class IndicatorType(Enum):
    """
    Enumeration for indicator placement categories.
    
    This enum determines where indicators are rendered on the chart:
    - PRICE_OVERLAY: Indicators that overlay on the main price chart (MA, Bollinger Bands)
    - VOLUME_OVERLAY: Indicators that overlay on the volume subplot (OBV, Volume MA)
    - SEPARATE_SUBPLOT: Indicators that need their own subplot (RSI, MACD, Stochastic)
    """
    PRICE_OVERLAY = "price_overlay"
    VOLUME_OVERLAY = "volume_overlay"
    SEPARATE_SUBPLOT = "separate_subplot"


class DataProcessor(BaseDataProcessor):
    """
    Processes VectorBT portfolio and data objects for chart rendering.
    
    This processor is responsible for:
    1. Extracting OHLCV data from VectorBT Data objects
    2. Cleaning and validating price/volume data
    3. Extracting portfolio equity curves and returns
    4. Handling data alignment and reindexing
    
    The processor ensures all data is clean and properly formatted before
    being passed to chart renderers, preventing common issues like:
    - NaN/Inf values causing chart scaling problems
    - Negative prices or volumes
    - Invalid OHLC relationships
    - Misaligned timestamps
    
    Usage:
        processor = DataProcessor(portfolio, data)
        ohlcv_data = processor.get_ohlcv_data()
        equity = processor.get_portfolio_equity()
    
    Integration Points:
    - Used by: TradingChartsEngine, ChartBuilder, all Renderers
    - Requires: VectorBT Portfolio and Data objects
    - Provides: Clean, validated data ready for plotting
    """
    
    def __init__(self, portfolio: vbt.Portfolio, data: vbt.Data, 
                 config: Optional[ProcessorConfig] = None):
        """
        Initialize the data processor.
        
        Args:
            portfolio: VectorBT Portfolio object containing trade results
            data: VectorBT Data object containing market data
            config: Optional processor configuration
        """
        super().__init__(config or ProcessorConfig(name="DataProcessor"))
        self.portfolio = portfolio
        self.data = data
        self.ohlcv_data = None  # Cached processed OHLCV data
        
    def get_ohlcv_data(self) -> pd.DataFrame:
        """
        Get processed OHLCV data.
        
        This is the main entry point for getting clean market data.
        Data is cached after first processing for efficiency.
        
        Returns:
            DataFrame with columns: Open, High, Low, Close, [Volume]
            Index: DatetimeIndex aligned with original data
        """
        if self.ohlcv_data is None:
            self.ohlcv_data = self.process()
        return self.ohlcv_data
    
    def _extract_data(self) -> pd.DataFrame:
        """
        Extract OHLCV data from VectorBT Data object.
        
        Handles various data formats and structures that VectorBT might provide.
        Ensures consistent column naming and data types.
        """
        try:
            ohlcv = pd.DataFrame({
                'Open': self._extract_series(self.data.get('open')),
                'High': self._extract_series(self.data.get('high')),
                'Low': self._extract_series(self.data.get('low')),
                'Close': self._extract_series(self.data.get('close')),
            })
            
            # Add volume if available (not all data sources provide volume)
            try:
                volume = self.data.get('volume')
                if volume is not None:
                    ohlcv['Volume'] = self._extract_series(volume)
            except (AttributeError, KeyError):
                logger.debug("Volume data not available")
            
            logger.debug(f"Extracted OHLCV data: {ohlcv.shape[0]} rows, {ohlcv.shape[1]} columns")
            return ohlcv
            
        except Exception as e:
            logger.error(f"Failed to extract OHLCV data: {e}")
            raise
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate OHLCV data for consistency and quality.
        
        Checks:
        - Data is not empty
        - Required columns exist (OHLC)
        - No full-NaN columns
        - Reasonable data ranges
        """
        if data.empty:
            logger.error("OHLCV data is empty")
            return False
        
        required_columns = ['Open', 'High', 'Low', 'Close']
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Missing required column: {col}")
                return False
            
            if data[col].isna().all():
                logger.error(f"Column {col} contains only NaN values")
                return False
        
        return True
    
    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform OHLCV data for chart compatibility.
        
        Transformations:
        1. Remove NaN/Inf values
        2. Filter out invalid prices (negative, zero)
        3. Validate OHLC relationships
        4. Remove extreme outliers
        5. Handle zero volume appropriately
        """
        original_count = len(data)
        
        # Replace infinity values with NaN for consistent handling
        data = data.replace([np.inf, -np.inf], np.nan)
        
        # Remove rows with any NaN in OHLC columns
        data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        if len(data) == 0:
            raise ValueError("No valid OHLCV data after removing NaN values")
        
        # Remove rows with non-positive prices
        price_cols = ['Open', 'High', 'Low', 'Close']
        for col in price_cols:
            data = data[data[col] > 0]
        
        # Validate OHLC relationships
        # High should be >= max(Open, Close) and >= Low
        # Low should be <= min(Open, Close) and <= High
        valid_ohlc = (
            (data['High'] >= data['Open']) &
            (data['High'] >= data['Close']) &
            (data['Low'] <= data['Open']) &
            (data['Low'] <= data['Close']) &
            (data['High'] >= data['Low'])
        )
        data = data[valid_ohlc]
        
        if self.config.remove_outliers:
            data = self._remove_outliers(data, price_cols)
        
        # Clean volume data if present
        if 'Volume' in data.columns:
            data = self._clean_volume_data(data)
        
        cleaned_count = len(data)
        if cleaned_count < original_count:
            logger.info(f"Data cleaning: removed {original_count - cleaned_count} invalid rows, "
                       f"kept {cleaned_count} valid rows")
        
        if len(data) == 0:
            raise ValueError("No valid data remaining after cleaning")
            
        return data
    
    def _remove_outliers(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Remove extreme outliers that could skew chart scaling.
        
        Uses a generous IQR-based method to only remove truly extreme values
        while preserving legitimate price movements.
        """
        for col in columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.01)
                Q3 = data[col].quantile(0.99)
                IQR = Q3 - Q1
                
                # Use configurable threshold (default 10x IQR)
                lower_bound = Q1 - self.config.outlier_threshold * IQR
                upper_bound = Q3 + self.config.outlier_threshold * IQR
                
                data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def _clean_volume_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean volume data for chart compatibility.
        
        Handles:
        - Negative volume (data errors)
        - Zero volume (replace with small value for log scale compatibility)
        """
        # Remove negative volume
        data = data[data['Volume'] >= 0]
        
        # Replace zero volume with small positive value
        # This prevents issues with log scale volume charts
        zero_volume_mask = data['Volume'] == 0
        if zero_volume_mask.any():
            min_positive_volume = data.loc[data['Volume'] > 0, 'Volume'].min()
            if pd.notna(min_positive_volume):
                replacement_value = float(min_positive_volume / 10)
                data.loc[zero_volume_mask, 'Volume'] = replacement_value
                logger.debug(f"Replaced {zero_volume_mask.sum()} zero volume values with {replacement_value}")
        
        return data
    
    def _extract_series(self, data: Any) -> pd.Series:
        """
        Extract pandas Series from various VectorBT data formats.
        
        VectorBT can return data in different formats depending on the
        data source and configuration. This method handles all cases.
        """
        if data is None:
            return pd.Series(dtype=float)
        
        if isinstance(data, pd.Series):
            return data
        elif isinstance(data, pd.DataFrame):
            # Multi-column data - use first column
            if data.shape[1] == 1:
                return data.iloc[:, 0]
            else:
                logger.debug(f"Multi-column data detected ({data.shape[1]} columns), using first column")
                return data.iloc[:, 0]
        elif hasattr(data, 'values') and hasattr(data, 'index'):
            # VectorBT wrapper objects
            return pd.Series(data.values, index=data.index)
        else:
            # Fallback for unknown types
            logger.debug(f"Converting unknown data type to Series: {type(data)}")
            return pd.Series(data)
    
    def get_portfolio_equity(self) -> pd.Series:
        """
        Get portfolio equity curve aligned with OHLCV data.
        
        Extracts the portfolio value over time and ensures it's properly
        aligned with the market data index for synchronized plotting.
        
        Returns:
            Series of portfolio values indexed by datetime
        """
        try:
            # Ensure OHLCV data is processed first
            if self.ohlcv_data is None:
                self.get_ohlcv_data()
            
            # Extract portfolio value
            portfolio_value = self.portfolio.value
            if hasattr(portfolio_value, 'values'):
                equity_values = portfolio_value.values.flatten()
                portfolio_index = portfolio_value.index
            else:
                equity_values = portfolio_value
                portfolio_index = self.portfolio.wrapper.index
            
            # Create equity series
            equity_series = pd.Series(equity_values, index=portfolio_index)
            
            # Align with OHLCV data index
            equity_series = equity_series.reindex(self.ohlcv_data.index, method='ffill')
            
            # Clean equity data
            equity_series = equity_series.replace([np.inf, -np.inf], np.nan)
            equity_series = equity_series.dropna()
            
            # Validate positive values
            if len(equity_series) > 0 and equity_series.min() <= 0:
                logger.warning("Portfolio equity contains non-positive values, filtering...")
                equity_series = equity_series[equity_series > 0]
            
            return equity_series
            
        except Exception as e:
            logger.error(f"Failed to extract portfolio equity: {e}")
            # Return empty series with correct index
            return pd.Series([], index=self.ohlcv_data.index if self.ohlcv_data is not None else [])
    
    def get_portfolio_returns(self) -> pd.Series:
        """
        Get portfolio returns aligned with OHLCV data.
        
        Returns:
            Series of portfolio returns (percentage) indexed by datetime
        """
        try:
            returns = self.portfolio.returns
            if hasattr(returns, 'values'):
                returns_series = pd.Series(returns.values.flatten(), index=self.ohlcv_data.index)
            else:
                returns_series = returns
            
            # Clean returns data
            returns_series = returns_series.replace([np.inf, -np.inf], np.nan)
            returns_series = returns_series.dropna()
            
            return returns_series
            
        except Exception as e:
            logger.error(f"Failed to extract portfolio returns: {e}")
            return pd.Series([], index=self.ohlcv_data.index if self.ohlcv_data is not None else [])
    
    def _calculate_data_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate statistics and metadata about the processed data.
        
        Used for debugging and validation.
        """
        info = {
            "rows": len(data),
            "columns": list(data.columns),
            "date_range": (data.index[0], data.index[-1]) if len(data) > 0 else (None, None),
            "has_volume": 'Volume' in data.columns,
        }
        
        # Add price statistics
        if len(data) > 0:
            info["price_range"] = {
                "min": data['Low'].min(),
                "max": data['High'].max(),
                "mean": data['Close'].mean()
            }
        
        return info


class IndicatorProcessor(BaseDataProcessor):
    """
    Processes and categorizes technical indicators for optimal chart placement.
    
    This processor analyzes indicators provided by trading strategies and
    determines the best location for rendering them on the chart:
    1. Price overlays (MAs, Bollinger Bands)
    2. Volume overlays (OBV, Volume MAs)
    3. Separate subplots (RSI, MACD, Stochastic)
    
    The categorization is based on:
    - Indicator name patterns
    - Value ranges compared to price data
    - Common indicator conventions
    
    Usage:
        processor = IndicatorProcessor(indicators, ohlcv_data)
        categorized = processor.get_categorized_indicators()
        
    Integration Points:
    - Used by: TradingChartsEngine, ChartBuilder (for subplot calculation)
    - Requires: Dictionary of indicators, OHLCV data for range comparison
    - Provides: Categorized indicators ready for rendering
    """
    
    def __init__(self, indicators: Dict[str, Any], ohlcv_data: pd.DataFrame,
                 config: Optional[ProcessorConfig] = None):
        """
        Initialize the indicator processor.
        
        Args:
            indicators: Dictionary of indicator name -> data (Series or array)
            ohlcv_data: Cleaned OHLCV data for range comparison
            config: Optional processor configuration
        """
        super().__init__(config or ProcessorConfig(name="IndicatorProcessor"))
        self.indicators = indicators
        self.ohlcv_data = ohlcv_data
        self.categorized = None  # Cached categorization results
    
    def get_categorized_indicators(self) -> Dict[IndicatorType, Dict[str, pd.Series]]:
        """
        Get indicators organized by placement category.
        
        Returns:
            Dictionary mapping IndicatorType to indicators in that category
        """
        if self.categorized is None:
            self.categorized = self.process()
        return self.categorized
    
    def _extract_data(self) -> Dict[str, pd.Series]:
        """
        Extract and standardize indicator data.
        
        Converts various indicator formats to pandas Series.
        """
        extracted = {}
        
        for name, indicator in self.indicators.items():
            try:
                series = self._extract_series(indicator)
                if len(series) > 0:
                    extracted[name] = series
            except Exception as e:
                logger.warning(f"Could not process indicator {name}: {e}")
        
        return extracted
    
    def _validate_data(self, data: Dict[str, pd.Series]) -> bool:
        """Validate extracted indicators."""
        if not data:
            logger.warning("No valid indicators to process")
            return True  # Empty is valid, just nothing to do
        
        return True
    
    def _transform_data(self, data: Dict[str, pd.Series]) -> Dict[IndicatorType, Dict[str, pd.Series]]:
        """
        Categorize indicators by placement type.
        
        Returns indicators organized by where they should be rendered.
        """
        categorized = {
            IndicatorType.PRICE_OVERLAY: {},
            IndicatorType.VOLUME_OVERLAY: {},
            IndicatorType.SEPARATE_SUBPLOT: {}
        }
        
        for name, series in data.items():
            # Clean the series
            series = series.replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(series) > 0:
                indicator_type = self._categorize_indicator(name, series)
                categorized[indicator_type][name] = series
        
        # Log categorization results
        logger.debug(f"Categorized indicators: "
                    f"{len(categorized[IndicatorType.PRICE_OVERLAY])} price overlay, "
                    f"{len(categorized[IndicatorType.SEPARATE_SUBPLOT])} subplots, "
                    f"{len(categorized[IndicatorType.VOLUME_OVERLAY])} volume")
        
        return categorized
    
    def _categorize_indicator(self, name: str, series: pd.Series) -> IndicatorType:
        """
        Determine the best placement for an indicator.
        
        Decision process:
        1. Check if it's a volume indicator by name
        2. Check if it's a known price overlay by name
        3. Compare value range with price data
        4. Default to separate subplot for unknowns
        """
        name_lower = name.lower()
        
        # Volume indicators first (most specific)
        if self._is_volume_indicator(name_lower):
            return IndicatorType.VOLUME_OVERLAY
        
        # Price overlay indicators
        if self._is_price_overlay_indicator(name_lower, series):
            return IndicatorType.PRICE_OVERLAY
        
        # Everything else gets its own subplot
        return IndicatorType.SEPARATE_SUBPLOT
    
    def _is_volume_indicator(self, name: str) -> bool:
        """Check if indicator is volume-related based on name."""
        volume_keywords = [
            'volume', 'vol_', 'obv',  # On-Balance Volume
            'ad', 'adl',              # Accumulation/Distribution
            'cmf',                     # Chaikin Money Flow
            'mfi',                     # Money Flow Index  
            'vwap',                    # Volume Weighted Average Price
            'pvt',                     # Price Volume Trend
            'vpt',                     # Volume Price Trend
            'nvi', 'pvi'              # Negative/Positive Volume Index
        ]
        return any(keyword in name for keyword in volume_keywords)
    
    def _is_price_overlay_indicator(self, name: str, series: pd.Series) -> bool:
        """
        Check if indicator should overlay on price chart.
        
        Uses both name patterns and value range analysis.
        """
        # Known price overlay patterns
        overlay_patterns = [
            'ma', 'sma', 'ema', 'wma',      # Moving averages
            'bollinger', 'bb_',              # Bollinger Bands
            'keltner', 'kc_',                # Keltner Channels
            'envelope',                       # Envelopes
            'atr_bands',                      # ATR Bands
            'pivot', 'support', 'resistance', # Support/Resistance
            'vwap'                           # Can be price overlay too
        ]
        
        # Check name patterns
        if any(pattern in name for pattern in overlay_patterns):
            return True
        
        # Check value range similarity to price
        if len(series) > 0 and len(self.ohlcv_data) > 0:
            try:
                # Calculate ranges
                price_range = self.ohlcv_data['Close'].max() - self.ohlcv_data['Close'].min()
                price_mean = self.ohlcv_data['Close'].mean()
                
                indicator_range = series.max() - series.min()
                indicator_mean = series.mean()
                
                # Check if indicator is in similar range as price
                # (within order of magnitude)
                if price_range > 0:
                    range_ratio = indicator_range / price_range
                    mean_ratio = indicator_mean / price_mean
                    
                    # If both range and mean are similar to price, likely an overlay
                    if (0.1 <= range_ratio <= 10) and (0.5 <= mean_ratio <= 2):
                        return True
                        
            except (ZeroDivisionError, ValueError):
                pass
        
        return False
    
    def _extract_series(self, indicator: Any) -> pd.Series:
        """
        Extract pandas Series from various indicator formats.
        
        Handles:
        - Pandas Series/DataFrame
        - NumPy arrays
        - VectorBT indicator objects
        - Lists and other iterables
        """
        if isinstance(indicator, pd.Series):
            return indicator
        elif isinstance(indicator, pd.DataFrame):
            # For DataFrames, use first column
            return indicator.iloc[:, 0]
        elif hasattr(indicator, 'values') and hasattr(indicator, 'index'):
            # VectorBT objects
            return pd.Series(indicator.values, index=indicator.index)
        elif isinstance(indicator, np.ndarray):
            # NumPy arrays - try to align with OHLCV index
            if len(indicator) == len(self.ohlcv_data):
                return pd.Series(indicator, index=self.ohlcv_data.index)
            else:
                return pd.Series(indicator)
        else:
            # Try to convert to Series
            return pd.Series(indicator)
    
    def _calculate_data_info(self, data: Dict[IndicatorType, Dict[str, pd.Series]]) -> Dict[str, Any]:
        """Calculate indicator statistics."""
        total_indicators = sum(len(indicators) for indicators in data.values())
        
        return {
            "total_indicators": total_indicators,
            "price_overlays": list(data[IndicatorType.PRICE_OVERLAY].keys()),
            "volume_overlays": list(data[IndicatorType.VOLUME_OVERLAY].keys()),
            "separate_subplots": list(data[IndicatorType.SEPARATE_SUBPLOT].keys()),
            "categorization_complete": True
        }


# Module exports
__all__ = [
    'DataProcessor',
    'IndicatorProcessor', 
    'IndicatorType'
]