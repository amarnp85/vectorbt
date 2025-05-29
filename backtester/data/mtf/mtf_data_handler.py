"""
Multi-Timeframe Data Handler

Provides comprehensive functionality for handling multi-timeframe data,
including fetching, alignment, and caching with look-ahead bias prevention.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import vectorbtpro as vbt
from ..fetching.data_fetcher_new import fetch_data
from .mtf_alignment import MTFAlignmentEngine
from .mtf_utils import get_timeframe_hierarchy, validate_timeframe_compatibility

logger = logging.getLogger(__name__)


class MTFDataHandler:
    """
    Handles multi-timeframe data operations with proper alignment.
    
    This class manages fetching, caching, and aligning data across multiple
    timeframes while preventing look-ahead bias.
    """
    
    def __init__(self, base_timeframe: str = "1h"):
        """
        Initialize MTF data handler.
        
        Args:
            base_timeframe: Base timeframe for alignment (default: '1h')
        """
        self.base_timeframe = base_timeframe
        self.alignment_engine = MTFAlignmentEngine(base_timeframe)
        self.data_cache: Dict[str, Dict[str, vbt.Data]] = {}
        
        logger.info(f"Initialized MTFDataHandler with base timeframe: {base_timeframe}")
    
    def fetch_mtf_data(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: Optional[Union[str, pd.Timestamp]] = None,
        end_date: Optional[Union[str, pd.Timestamp]] = None,
        exchange_id: str = "binance",
        use_cache: bool = True,
        align_to_base: bool = True
    ) -> Dict[str, vbt.Data]:
        """
        Fetch data for multiple timeframes with optional alignment.
        
        Args:
            symbols: List of trading symbols
            timeframes: List of timeframes to fetch
            start_date: Start date for data
            end_date: End date for data
            exchange_id: Exchange identifier
            use_cache: Whether to use caching
            align_to_base: Whether to align all timeframes to base
            
        Returns:
            Dictionary mapping timeframe to vbt.Data object
        """
        logger.info(f"Fetching MTF data for {len(symbols)} symbols across {len(timeframes)} timeframes")
        
        # Validate timeframes
        if not validate_timeframe_compatibility(timeframes, self.base_timeframe):
            logger.warning("Some timeframes may not be compatible for alignment")
        
        # Sort timeframes by hierarchy
        sorted_timeframes = get_timeframe_hierarchy(timeframes)
        
        # Fetch data for each timeframe
        mtf_data = {}
        
        for tf in sorted_timeframes:
            logger.info(f"Fetching data for timeframe: {tf}")
            
            # Check cache first
            cache_key = f"{exchange_id}_{tf}_{'_'.join(sorted(symbols))}"
            if use_cache and cache_key in self.data_cache:
                logger.info(f"Using cached data for {tf}")
                mtf_data[tf] = self.data_cache[cache_key]
                continue
            
            # Fetch fresh data
            data = fetch_data(
                symbols=symbols,
                exchange_id=exchange_id,
                timeframe=tf,
                start_date=start_date,
                end_date=end_date,
                use_cache=use_cache
            )
            
            if data is not None:
                mtf_data[tf] = data
                # Cache the data
                if use_cache:
                    self.data_cache[cache_key] = data
            else:
                logger.error(f"Failed to fetch data for timeframe {tf}")
        
        # Align to base timeframe if requested
        if align_to_base and len(mtf_data) > 1:
            mtf_data = self.align_all_timeframes(mtf_data)
        
        return mtf_data
    
    def align_all_timeframes(self, mtf_data: Dict[str, vbt.Data]) -> Dict[str, vbt.Data]:
        """
        Align all timeframes to the base timeframe.
        
        Args:
            mtf_data: Dictionary of timeframe to data
            
        Returns:
            Dictionary with aligned data
        """
        if self.base_timeframe not in mtf_data:
            logger.warning(f"Base timeframe {self.base_timeframe} not in data, using lowest timeframe")
            # Use the lowest timeframe as base
            sorted_tfs = get_timeframe_hierarchy(list(mtf_data.keys()))
            self.base_timeframe = sorted_tfs[0]
        
        base_data = mtf_data[self.base_timeframe]
        aligned_data = {self.base_timeframe: base_data}
        
        for tf, data in mtf_data.items():
            if tf != self.base_timeframe:
                logger.info(f"Aligning {tf} to {self.base_timeframe}")
                aligned = self.alignment_engine.align_data(data, base_data)
                aligned_data[tf] = aligned
        
        return aligned_data
    
    def create_mtf_indicators(
        self,
        mtf_data: Dict[str, vbt.Data],
        indicator_func: callable,
        **indicator_params
    ) -> Dict[str, pd.DataFrame]:
        """
        Calculate indicators across multiple timeframes.
        
        Args:
            mtf_data: Dictionary of timeframe to data
            indicator_func: Indicator function to apply
            **indicator_params: Parameters for the indicator
            
        Returns:
            Dictionary mapping timeframe to indicator results
        """
        mtf_indicators = {}
        
        for tf, data in mtf_data.items():
            logger.debug(f"Calculating indicator for {tf}")
            try:
                indicator = indicator_func(data, **indicator_params)
                mtf_indicators[tf] = indicator
            except Exception as e:
                logger.error(f"Failed to calculate indicator for {tf}: {e}")
        
        return mtf_indicators
    
    def get_aligned_close_prices(
        self,
        mtf_data: Dict[str, vbt.Data],
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get aligned close prices across all timeframes.
        
        Args:
            mtf_data: Dictionary of timeframe to data
            symbol: Specific symbol to extract (None for all)
            
        Returns:
            DataFrame with aligned close prices
        """
        close_prices = {}
        
        for tf, data in mtf_data.items():
            if hasattr(data, 'close'):
                close = data.close
                if symbol and isinstance(close, pd.DataFrame) and symbol in close.columns:
                    close = close[symbol]
                elif isinstance(close, pd.DataFrame) and close.shape[1] == 1:
                    close = close.iloc[:, 0]
                
                close_prices[f"close_{tf}"] = close
        
        # Create DataFrame with all close prices
        df = pd.DataFrame(close_prices)
        
        # Forward fill to handle NaN values from alignment
        df = df.ffill()
        
        return df
    
    def create_mtf_features(
        self,
        mtf_data: Dict[str, vbt.Data],
        feature_funcs: Dict[str, callable],
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a feature matrix from multiple timeframes.
        
        Args:
            mtf_data: Dictionary of timeframe to data
            feature_funcs: Dictionary of feature name to function
            symbol: Specific symbol to process
            
        Returns:
            DataFrame with all MTF features
        """
        all_features = {}
        
        for tf, data in mtf_data.items():
            for feature_name, func in feature_funcs.items():
                try:
                    feature = func(data)
                    if symbol and isinstance(feature, pd.DataFrame) and symbol in feature.columns:
                        feature = feature[symbol]
                    elif isinstance(feature, pd.DataFrame) and feature.shape[1] == 1:
                        feature = feature.iloc[:, 0]
                    
                    all_features[f"{feature_name}_{tf}"] = feature
                except Exception as e:
                    logger.error(f"Failed to create feature {feature_name} for {tf}: {e}")
        
        # Create feature DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Forward fill NaN values
        features_df = features_df.ffill()
        
        return features_df
    
    def validate_mtf_signals(
        self,
        signals: Dict[str, pd.Series],
        mtf_data: Dict[str, vbt.Data]
    ) -> Dict[str, pd.Series]:
        """
        Validate signals against MTF data to prevent look-ahead bias.
        
        Args:
            signals: Dictionary of signal series
            mtf_data: Dictionary of timeframe to data
            
        Returns:
            Validated signals
        """
        validated_signals = {}
        
        for signal_name, signal_series in signals.items():
            # Check if signal uses future information
            if self.alignment_engine.check_look_ahead_bias(signal_series, mtf_data):
                logger.warning(f"Potential look-ahead bias detected in {signal_name}")
                # Apply correction if needed
                signal_series = self.alignment_engine.correct_signal_timing(
                    signal_series, mtf_data
                )
            
            validated_signals[signal_name] = signal_series
        
        return validated_signals
    
    def get_timeframe_info(self) -> Dict[str, any]:
        """
        Get information about current MTF setup.
        
        Returns:
            Dictionary with MTF configuration info
        """
        return {
            'base_timeframe': self.base_timeframe,
            'cached_timeframes': list(set(
                key.split('_')[1] for key in self.data_cache.keys()
            )),
            'cache_size': len(self.data_cache),
            'alignment_engine': str(self.alignment_engine)
        }
    
    def clear_cache(self):
        """Clear the data cache."""
        self.data_cache.clear()
        logger.info("MTF data cache cleared") 