"""
Multi-Timeframe Alignment Engine

Handles proper alignment of multi-timeframe data using vectorbtpro's
realign methods to prevent look-ahead bias.
"""

import logging
from typing import Dict, Optional, Union, Tuple
import pandas as pd
import numpy as np
import vectorbtpro as vbt

logger = logging.getLogger(__name__)


class MTFAlignmentEngine:
    """
    Engine for aligning multi-timeframe data safely.
    
    Uses vectorbtpro's realign_opening and realign_closing methods
    to ensure proper time alignment without look-ahead bias.
    """
    
    def __init__(self, base_timeframe: str = "1h"):
        """
        Initialize alignment engine.
        
        Args:
            base_timeframe: Base timeframe for alignment
        """
        self.base_timeframe = base_timeframe
        logger.info(f"Initialized MTFAlignmentEngine with base: {base_timeframe}")
    
    def align_data(
        self,
        source_data: vbt.Data,
        target_data: vbt.Data,
        method: str = "auto"
    ) -> vbt.Data:
        """
        Align source data to target data's timeframe.
        
        Args:
            source_data: Data to align
            target_data: Target timeframe data
            method: Alignment method ('auto', 'opening', 'closing')
            
        Returns:
            Aligned vbt.Data object
        """
        try:
            # Get target index
            target_index = target_data.wrapper.index
            
            # Create aligned data dictionary
            aligned_dict = {}
            
            # Align each OHLCV component appropriately
            if hasattr(source_data, 'open'):
                # Open price is available at bar open
                aligned_dict['open'] = self._align_series(
                    source_data.open, target_index, 'opening'
                )
            
            if hasattr(source_data, 'high'):
                # High price is known only at bar close
                aligned_dict['high'] = self._align_series(
                    source_data.high, target_index, 'closing'
                )
            
            if hasattr(source_data, 'low'):
                # Low price is known only at bar close
                aligned_dict['low'] = self._align_series(
                    source_data.low, target_index, 'closing'
                )
            
            if hasattr(source_data, 'close'):
                # Close price is known only at bar close
                aligned_dict['close'] = self._align_series(
                    source_data.close, target_index, 'closing'
                )
            
            if hasattr(source_data, 'volume'):
                # Volume is accumulated throughout the bar
                aligned_dict['volume'] = self._align_series(
                    source_data.volume, target_index, 'closing'
                )
            
            # Create new vbt.Data object with aligned data
            aligned_data = vbt.Data.from_data(aligned_dict)
            
            return aligned_data
            
        except Exception as e:
            logger.error(f"Failed to align data: {e}")
            raise
    
    def _align_series(
        self,
        series: Union[pd.Series, pd.DataFrame],
        target_index: pd.DatetimeIndex,
        method: str = 'closing'
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Align a series to target index using appropriate method.
        
        Args:
            series: Series to align
            target_index: Target index
            method: 'opening' or 'closing'
            
        Returns:
            Aligned series
        """
        if isinstance(series, pd.DataFrame):
            # Handle multi-column data
            aligned_cols = {}
            for col in series.columns:
                aligned_cols[col] = self._align_single_series(
                    series[col], target_index, method
                )
            return pd.DataFrame(aligned_cols, index=target_index)
        else:
            # Single series
            return self._align_single_series(series, target_index, method)
    
    def _align_single_series(
        self,
        series: pd.Series,
        target_index: pd.DatetimeIndex,
        method: str
    ) -> pd.Series:
        """
        Align a single series using vectorbtpro's methods.
        
        Args:
            series: Series to align
            target_index: Target index
            method: 'opening' or 'closing'
            
        Returns:
            Aligned series
        """
        try:
            if method == 'opening':
                # Use realign_opening for data available at bar open
                aligned = series.vbt.realign_opening(target_index)
            else:
                # Use realign_closing for data available only at bar close
                aligned = series.vbt.realign_closing(target_index)
            
            return aligned
            
        except Exception as e:
            logger.warning(f"VBT realign failed, using fallback: {e}")
            # Fallback to manual alignment
            return self._manual_align(series, target_index, method)
    
    def _manual_align(
        self,
        series: pd.Series,
        target_index: pd.DatetimeIndex,
        method: str
    ) -> pd.Series:
        """
        Manual alignment fallback when VBT methods fail.
        
        Args:
            series: Series to align
            target_index: Target index
            method: 'opening' or 'closing'
            
        Returns:
            Aligned series
        """
        # Create resampler
        resampler = vbt.Resampler(
            source_index=series.index,
            target_index=target_index,
            source_freq=pd.infer_freq(series.index),
            target_freq=pd.infer_freq(target_index)
        )
        
        # Perform alignment based on method
        if method == 'opening':
            # Forward fill from previous value
            aligned = series.reindex(target_index, method='ffill')
        else:
            # For closing data, shift by one to avoid look-ahead
            shifted = series.shift(1)
            aligned = shifted.reindex(target_index, method='ffill')
        
        return aligned
    
    def check_look_ahead_bias(
        self,
        signals: pd.Series,
        mtf_data: Dict[str, vbt.Data]
    ) -> bool:
        """
        Check if signals potentially use future information.
        
        Args:
            signals: Signal series to check
            mtf_data: Multi-timeframe data used
            
        Returns:
            True if potential look-ahead bias detected
        """
        # Get signal timestamps
        signal_times = signals[signals].index
        
        if len(signal_times) == 0:
            return False
        
        # Check each timeframe's data availability
        for tf, data in mtf_data.items():
            if tf == self.base_timeframe:
                continue
            
            # Get data timestamps
            data_times = data.wrapper.index
            
            # Check if signals use data not yet available
            for signal_time in signal_times:
                # Find the last available data point before signal
                available_data = data_times[data_times <= signal_time]
                
                if len(available_data) == 0:
                    logger.warning(f"Signal at {signal_time} has no prior {tf} data")
                    return True
                
                # Check if the gap is suspiciously large
                last_data_time = available_data[-1]
                time_diff = signal_time - last_data_time
                
                # Convert timeframe to expected timedelta
                expected_delay = self._timeframe_to_timedelta(tf)
                
                if time_diff < expected_delay * 0.9:  # 90% threshold
                    logger.warning(
                        f"Signal at {signal_time} may use future {tf} data "
                        f"(gap: {time_diff} < expected: {expected_delay})"
                    )
                    return True
        
        return False
    
    def correct_signal_timing(
        self,
        signals: pd.Series,
        mtf_data: Dict[str, vbt.Data]
    ) -> pd.Series:
        """
        Correct signal timing to prevent look-ahead bias.
        
        Args:
            signals: Signal series to correct
            mtf_data: Multi-timeframe data used
            
        Returns:
            Corrected signal series
        """
        # Find the highest timeframe used
        max_timeframe = self._get_max_timeframe(list(mtf_data.keys()))
        max_delay = self._timeframe_to_timedelta(max_timeframe)
        
        # Shift signals by the maximum delay
        corrected_signals = signals.shift(periods=1, freq=max_delay)
        
        logger.info(f"Shifted signals by {max_delay} to prevent look-ahead bias")
        
        return corrected_signals
    
    def _timeframe_to_timedelta(self, timeframe: str) -> pd.Timedelta:
        """Convert timeframe string to timedelta."""
        # Handle common timeframes
        mappings = {
            '1m': pd.Timedelta(minutes=1),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4),
            '1d': pd.Timedelta(days=1),
            '1w': pd.Timedelta(weeks=1),
        }
        
        if timeframe in mappings:
            return mappings[timeframe]
        
        # Try to parse with pandas
        try:
            return pd.Timedelta(timeframe)
        except:
            logger.warning(f"Unknown timeframe {timeframe}, defaulting to 1h")
            return pd.Timedelta(hours=1)
    
    def _get_max_timeframe(self, timeframes: list) -> str:
        """Get the highest timeframe from a list."""
        max_delta = pd.Timedelta(0)
        max_tf = timeframes[0]
        
        for tf in timeframes:
            delta = self._timeframe_to_timedelta(tf)
            if delta > max_delta:
                max_delta = delta
                max_tf = tf
        
        return max_tf
    
    def create_aligned_features(
        self,
        mtf_data: Dict[str, vbt.Data],
        feature_columns: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Create aligned features from multiple timeframes.
        
        Args:
            mtf_data: Multi-timeframe data
            feature_columns: Mapping of feature names to data columns
            
        Returns:
            DataFrame with aligned features
        """
        features = {}
        
        # Get base index
        base_index = mtf_data[self.base_timeframe].wrapper.index
        
        for tf, data in mtf_data.items():
            for feature_name, column in feature_columns.items():
                # Extract the column data
                if hasattr(data, column):
                    col_data = getattr(data, column)
                    
                    # Determine alignment method based on column type
                    if column == 'open':
                        method = 'opening'
                    else:
                        method = 'closing'
                    
                    # Align to base timeframe
                    if tf != self.base_timeframe:
                        aligned = self._align_series(col_data, base_index, method)
                    else:
                        aligned = col_data
                    
                    # Store with timeframe suffix
                    features[f"{feature_name}_{tf}"] = aligned
        
        return pd.DataFrame(features, index=base_index) 