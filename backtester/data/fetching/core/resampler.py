#!/usr/bin/env python3
"""Data Resampler - Handle timeframe resampling operations.

This module provides clean resampling functionality for converting data
from lower to higher timeframes.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import vectorbtpro as vbt
from .vbt_data_handler import VBTDataHandler

logger = logging.getLogger(__name__)

# Timeframe hierarchy for resampling
TIMEFRAME_HIERARCHY = [
    "1m",
    "2m",
    "3m",
    "5m",
    "15m",
    "30m",  # Minutes
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",  # Hours
    "1d",
    "3d",
    "1w",
    "2w",  # Days and weeks
    "1M",
    "3M",
    "6M",
    "1y",  # Months and years
]


class DataResampler:
    """Handle data resampling operations."""

    @staticmethod
    def can_resample(source_tf: str, target_tf: str) -> bool:
        """Check if we can resample from source to target timeframe."""
        try:
            source_idx = TIMEFRAME_HIERARCHY.index(source_tf)
            target_idx = TIMEFRAME_HIERARCHY.index(target_tf)
            # Can only resample from lower to higher timeframes
            return source_idx < target_idx
        except ValueError:
            return False

    @staticmethod
    def get_available_sources(target_tf: str) -> List[str]:
        """Get list of timeframes that can be resampled to target."""
        try:
            target_idx = TIMEFRAME_HIERARCHY.index(target_tf)
            return TIMEFRAME_HIERARCHY[:target_idx]
        except ValueError:
            return []

    @staticmethod
    def resample_data(data: vbt.Data, target_timeframe: str) -> Optional[vbt.Data]:
        """
        Resample VBT data to target timeframe.

        Args:
            data: Source VBT Data object
            target_timeframe: Target timeframe string

        Returns:
            Resampled VBT Data object or None
        """
        try:
            logger.debug(f"Resampling data to {target_timeframe}")

            # Extract OHLCV data
            ohlcv_data = VBTDataHandler.extract_ohlcv(data)
            if not ohlcv_data:
                logger.error("Failed to extract OHLCV data for resampling")
                return None

            # Get symbols
            symbols = list(data.symbols) if hasattr(data, "symbols") else []
            if not symbols:
                logger.error("No symbols found in data")
                return None

            # Convert timeframe to pandas format
            pandas_tf = DataResampler._convert_timeframe_to_pandas(target_timeframe)
            logger.debug(f"Using pandas timeframe: {pandas_tf}")

            # Resample each symbol individually to preserve date ranges
            symbol_dict = {}

            for symbol in symbols:
                resampled_symbol = DataResampler._resample_symbol(
                    symbol, ohlcv_data, pandas_tf
                )
                if resampled_symbol is not None:
                    symbol_dict[symbol] = resampled_symbol

            if not symbol_dict:
                logger.error("No valid symbol data after resampling")
                return None

            # Create VBT Data from resampled symbols
            return VBTDataHandler.create_from_dict(symbol_dict)

        except Exception as e:
            logger.error(f"Error in resample_data: {e}")
            return None

    @staticmethod
    def _convert_timeframe_to_pandas(timeframe: str) -> str:
        """Convert timeframe to pandas-compatible format."""
        tf = timeframe.lower()

        if tf.endswith("m") and not tf.endswith("M"):
            # Minutes: use 'min' to distinguish from months
            return tf.replace("m", "min")
        elif tf.endswith("d"):
            # Days: use 'D'
            return tf.replace("d", "D")
        elif tf.endswith("w"):
            # Weeks: use 'W'
            return tf.replace("w", "W")
        elif tf.endswith("h"):
            # Hours: keep 'h' (new pandas standard)
            return tf
        else:
            # Keep as is (months 'M', years 'y')
            return timeframe

    @staticmethod
    def _resample_symbol(
        symbol: str, ohlcv_data: Dict[str, pd.DataFrame], pandas_timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Resample a single symbol's OHLCV data."""
        try:
            # Extract symbol data
            symbol_data = {}

            for feature in ["open", "high", "low", "close", "volume"]:
                if feature in ohlcv_data and ohlcv_data[feature] is not None:
                    df = ohlcv_data[feature]
                    if symbol in df.columns:
                        series = df[symbol].dropna()
                        if len(series) > 0:
                            symbol_data[feature] = series

            # Need at least OHLC
            if len(symbol_data) < 4:
                logger.warning(f"Insufficient OHLCV data for {symbol}")
                return None

            # Create DataFrame
            df = pd.DataFrame(symbol_data)

            if df.empty:
                return None

            # Log original range
            logger.debug(
                f"Resampling {symbol}: {df.index[0]} to {df.index[-1]} "
                f"({len(df)} rows)"
            )

            # Resample with appropriate aggregation
            resampled = pd.DataFrame()

            aggregation = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }

            for feature, method in aggregation.items():
                if feature in df.columns:
                    resampled[feature.capitalize()] = (
                        df[feature].resample(pandas_timeframe).agg(method)
                    )

            # Remove rows with any NaN in OHLC
            critical_cols = ["Open", "High", "Low", "Close"]
            available_critical = [
                col for col in critical_cols if col in resampled.columns
            ]

            if available_critical:
                valid_mask = resampled[available_critical].notna().all(axis=1)
                resampled = resampled[valid_mask]

            if resampled.empty:
                return None

            logger.debug(
                f"Resampled {symbol}: {resampled.index[0]} to {resampled.index[-1]} "
                f"({len(resampled)} rows)"
            )

            return resampled

        except Exception as e:
            logger.warning(f"Error resampling {symbol}: {e}")
            return None
