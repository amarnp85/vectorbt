"""
Multi-Timeframe Strategy Base Class

Provides a foundation for strategies that use multiple timeframes,
leveraging VectorBTPro's native multi-timeframe capabilities.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, List, Optional, Union
from abc import abstractmethod
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MTFStrategy(BaseStrategy):
    """
    Base class for multi-timeframe trading strategies.
    
    This class extends BaseStrategy to handle multiple timeframes using
    VectorBTPro's native multi-timeframe support. VBT automatically handles
    data alignment and prevents look-ahead bias.
    
    Note: Position sizing and portfolio management are handled at the portfolio level.
    """
    
    def __init__(
        self,
        data: Union[vbt.Data, Dict[str, vbt.Data]],
        params: Dict[str, Any],
        base_timeframe: str = "1h"
    ):
        """
        Initialize multi-timeframe strategy.
        
        Args:
            data: Either a single vbt.Data object with multiple timeframes
                  or dict of timeframe to data
            params: Strategy parameters
            base_timeframe: Base timeframe for signal generation
        """
        # Handle different data formats
        if isinstance(data, dict):
            # Multiple vbt.Data objects
            self.mtf_data = data
            base_data = data.get(base_timeframe, list(data.values())[0])
        else:
            # Single vbt.Data with multiple timeframes
            # Check if data has multiple timeframes
            if hasattr(data, 'timeframe') and isinstance(data.timeframe, dict):
                self.mtf_data = data
                base_data = data
            else:
                # Single timeframe data
                self.mtf_data = {base_timeframe: data}
                base_data = data
        
        # Initialize parent with base data
        super().__init__(base_data, params)
        
        self.base_timeframe = base_timeframe
        self.timeframes = self._extract_timeframes()
        
        # MTF-specific parameters
        self.use_mtf_confirmation = params.get('use_mtf_confirmation', True)
        self.mtf_weights = params.get('mtf_weights', {})
        
        # Storage for MTF indicators
        self.mtf_indicators = {}
        
        logger.info(
            f"Initialized MTFStrategy with timeframes: {self.timeframes}, "
            f"base: {self.base_timeframe}"
        )
    
    def _extract_timeframes(self) -> List[str]:
        """Extract available timeframes from data."""
        if isinstance(self.mtf_data, dict):
            return list(self.mtf_data.keys())
        elif hasattr(self.mtf_data, 'timeframe'):
            if isinstance(self.mtf_data.timeframe, dict):
                return list(self.mtf_data.timeframe.keys())
            else:
                return [str(self.mtf_data.timeframe)]
        else:
            return [self.base_timeframe]
    
    def init_indicators(self) -> Dict[str, Any]:
        """
        Initialize indicators across all timeframes using VBT's native support.
        
        Returns:
            Dictionary of indicators including MTF indicators
        """
        logger.info("Calculating MTF indicators")
        
        # If using dict of data objects
        if isinstance(self.mtf_data, dict):
            for tf, tf_data in self.mtf_data.items():
                tf_indicators = self._calculate_timeframe_indicators(tf_data, tf)
                self.mtf_indicators[tf] = tf_indicators
        else:
            # Single data object with multiple timeframes
            # VBT handles multi-timeframe data natively
            all_indicators = self._calculate_all_timeframe_indicators(self.mtf_data)
            self.mtf_indicators = all_indicators
        
        # Store base timeframe indicators in parent class format
        if self.base_timeframe in self.mtf_indicators:
            self.indicators = self.mtf_indicators[self.base_timeframe]
        else:
            self.indicators = self.mtf_indicators
        
        # Add cross-timeframe indicators
        if self.use_mtf_confirmation and len(self.timeframes) > 1:
            cross_tf_indicators = self._calculate_cross_timeframe_indicators()
            self.indicators['mtf'] = cross_tf_indicators
        
        return self.indicators
    
    @abstractmethod
    def _calculate_timeframe_indicators(
        self,
        data: vbt.Data,
        timeframe: str
    ) -> Dict[str, Any]:
        """
        Calculate indicators for a specific timeframe.
        
        Args:
            data: Data for the timeframe
            timeframe: Timeframe string
            
        Returns:
            Dictionary of indicators
        """
        pass
    
    def _calculate_all_timeframe_indicators(
        self,
        data: vbt.Data
    ) -> Dict[str, Dict[str, Any]]:
        """
        Calculate indicators for all timeframes in a single data object.
        
        Args:
            data: Multi-timeframe data object
            
        Returns:
            Dictionary of timeframe to indicators
        """
        indicators = {}
        
        # VBT can calculate indicators across multiple timeframes efficiently
        # Example using VBT's native MTF support
        for tf in self.timeframes:
            # Select timeframe data
            if hasattr(data, 'select_timeframe'):
                tf_data = data.select_timeframe(tf)
            else:
                tf_data = data
            
            indicators[tf] = self._calculate_timeframe_indicators(tf_data, tf)
        
        return indicators
    
    def _calculate_cross_timeframe_indicators(self) -> Dict[str, Any]:
        """
        Calculate indicators that combine multiple timeframes.
        
        Returns:
            Dictionary of cross-timeframe indicators
        """
        cross_indicators = {}
        
        # Trend alignment across timeframes
        if len(self.timeframes) > 1:
            trend_alignment = self._calculate_trend_alignment()
            cross_indicators['trend_alignment'] = trend_alignment
        
        # Momentum confluence
        momentum_confluence = self._calculate_momentum_confluence()
        if momentum_confluence is not None:
            cross_indicators['momentum_confluence'] = momentum_confluence
        
        return cross_indicators
    
    def _calculate_trend_alignment(self) -> pd.Series:
        """
        Calculate trend alignment score across timeframes using VBT.
        
        Returns:
            Series with trend alignment scores (-1 to 1)
        """
        trends = {}
        base_index = None
        
        # Calculate trend for each timeframe
        for tf in self.timeframes:
            if tf in self.mtf_indicators:
                tf_indicators = self.mtf_indicators[tf]
                
                # Use SMA comparison for trend
                if 'sma_fast' in tf_indicators and 'sma_slow' in tf_indicators:
                    fast_ma = tf_indicators['sma_fast']
                    slow_ma = tf_indicators['sma_slow']
                    
                    # Trend: 1 for uptrend, -1 for downtrend
                    trend = (fast_ma > slow_ma).astype(int) * 2 - 1
                    
                    # Align to base timeframe using VBT
                    if base_index is None:
                        base_index = trend.index
                    elif not trend.index.equals(base_index):
                        # Use VBT's resampling for alignment
                        if hasattr(trend, 'vbt'):
                            trend = trend.vbt.resample_closing(base_index)
                        else:
                            trend = trend.reindex(base_index, method='ffill')
                    
                    trends[tf] = trend
        
        if not trends:
            # Return neutral if no trends calculated
            return pd.Series(0, index=self.data.wrapper.index)
        
        # Calculate alignment score (average of all timeframe trends)
        trend_df = pd.DataFrame(trends)
        alignment_score = trend_df.mean(axis=1)
        
        return alignment_score
    
    def _calculate_momentum_confluence(self) -> Optional[pd.Series]:
        """
        Calculate momentum confluence across timeframes.
        
        Returns:
            Series with momentum confluence scores or None
        """
        momentum_scores = {}
        base_index = None
        
        # Calculate momentum for each timeframe
        for tf in self.timeframes:
            if tf in self.mtf_indicators:
                tf_indicators = self.mtf_indicators[tf]
                
                # Use RSI for momentum
                if 'rsi' in tf_indicators:
                    rsi = tf_indicators['rsi']
                    
                    # Normalize RSI to -1 to 1 scale
                    momentum = (rsi - 50) / 50
                    
                    # Align to base timeframe
                    if base_index is None:
                        base_index = momentum.index
                    elif not momentum.index.equals(base_index):
                        if hasattr(momentum, 'vbt'):
                            momentum = momentum.vbt.resample_closing(base_index)
                        else:
                            momentum = momentum.reindex(base_index, method='ffill')
                    
                    momentum_scores[tf] = momentum
        
        if not momentum_scores:
            return None
        
        # Calculate confluence (weighted average if weights provided)
        momentum_df = pd.DataFrame(momentum_scores)
        
        if self.mtf_weights:
            # Apply custom weights
            weights = pd.Series(
                [self.mtf_weights.get(tf, 1.0) for tf in momentum_df.columns]
            )
            weights = weights / weights.sum()  # Normalize
            confluence = (momentum_df * weights).sum(axis=1)
        else:
            # Equal weighting
            confluence = momentum_df.mean(axis=1)
        
        return confluence
    
    def generate_signals(self) -> Dict[str, Any]:
        """
        Generate trading signals using MTF analysis.
        
        Returns:
            Dictionary of signal arrays
        """
        logger.info("Generating MTF signals")
        
        # Generate base signals from primary timeframe
        base_signals = self._generate_base_signals()
        
        # Apply MTF confirmation if enabled
        if self.use_mtf_confirmation and len(self.timeframes) > 1:
            confirmed_signals = self._apply_mtf_confirmation(base_signals)
        else:
            confirmed_signals = base_signals
        
        # Clean signals using VBT
        if 'long_entries' in confirmed_signals and 'long_exits' in confirmed_signals:
            if hasattr(confirmed_signals['long_entries'], 'vbt'):
                confirmed_signals['long_entries'], confirmed_signals['long_exits'] = \
                    confirmed_signals['long_entries'].vbt.signals.clean(
                        confirmed_signals['long_exits']
                    )
        
        if 'short_entries' in confirmed_signals and 'short_exits' in confirmed_signals:
            if hasattr(confirmed_signals['short_entries'], 'vbt'):
                confirmed_signals['short_entries'], confirmed_signals['short_exits'] = \
                    confirmed_signals['short_entries'].vbt.signals.clean(
                        confirmed_signals['short_exits']
                    )
        
        self.signals = confirmed_signals
        return self.signals
    
    @abstractmethod
    def _generate_base_signals(self) -> Dict[str, pd.Series]:
        """
        Generate base signals from primary timeframe.
        
        Returns:
            Dictionary of base signals
        """
        pass
    
    def _apply_mtf_confirmation(
        self,
        base_signals: Dict[str, pd.Series]
    ) -> Dict[str, pd.Series]:
        """
        Apply multi-timeframe confirmation to base signals.
        
        Args:
            base_signals: Base signals to confirm
            
        Returns:
            Confirmed signals
        """
        confirmed_signals = base_signals.copy()
        
        # Get MTF indicators
        if 'mtf' not in self.indicators:
            return confirmed_signals
        
        mtf_indicators = self.indicators['mtf']
        
        # Apply trend alignment filter
        if 'trend_alignment' in mtf_indicators:
            trend_score = mtf_indicators['trend_alignment']
            
            # Confirm long signals when trend alignment is positive
            if 'long_entries' in confirmed_signals:
                confirmed_signals['long_entries'] = (
                    confirmed_signals['long_entries'] & (trend_score > 0.3)
                )
            
            # Confirm short signals when trend alignment is negative
            if 'short_entries' in confirmed_signals:
                confirmed_signals['short_entries'] = (
                    confirmed_signals['short_entries'] & (trend_score < -0.3)
                )
        
        # Apply momentum confluence filter
        if 'momentum_confluence' in mtf_indicators:
            momentum = mtf_indicators['momentum_confluence']
            
            # Additional confirmation based on momentum
            if 'long_entries' in confirmed_signals:
                confirmed_signals['long_entries'] = (
                    confirmed_signals['long_entries'] & (momentum > 0)
                )
            
            if 'short_entries' in confirmed_signals:
                confirmed_signals['short_entries'] = (
                    confirmed_signals['short_entries'] & (momentum < 0)
                )
        
        return confirmed_signals
    
    def get_mtf_summary(self) -> Dict[str, Any]:
        """
        Get summary of MTF setup and indicators.
        
        Returns:
            Dictionary with MTF information
        """
        summary = {
            'base_timeframe': self.base_timeframe,
            'timeframes': self.timeframes,
            'mtf_confirmation': self.use_mtf_confirmation,
            'mtf_weights': self.mtf_weights,
            'indicators_per_timeframe': {}
        }
        
        # Add indicator summary
        for tf, indicators in self.mtf_indicators.items():
            summary['indicators_per_timeframe'][tf] = list(indicators.keys())
        
        # Add MTF indicator info
        if 'mtf' in self.indicators:
            summary['cross_timeframe_indicators'] = list(self.indicators['mtf'].keys())
        
        return summary
    
    def __repr__(self) -> str:
        """String representation of MTF strategy."""
        return (
            f"{self.__class__.__name__}("
            f"timeframes={self.timeframes}, "
            f"base={self.base_timeframe}, "
            f"mtf_confirmation={self.use_mtf_confirmation})"
        ) 