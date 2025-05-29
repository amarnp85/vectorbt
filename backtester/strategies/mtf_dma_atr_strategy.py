"""
Multi-Timeframe DMA ATR Strategy

Extends the basic DMA ATR strategy with multi-timeframe analysis
for improved signal confirmation and trend alignment.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional, Union
import logging

from .mtf_strategy_base import MTFStrategy
from ..indicators.simple_indicators import sma, atr

logger = logging.getLogger(__name__)


class MTF_DMA_ATR_Strategy(MTFStrategy):
    """
    Multi-timeframe version of DMA ATR strategy.
    
    Uses multiple timeframes to confirm trends and filter signals,
    reducing false entries and improving overall performance.
    
    Note: Position sizing and risk management are handled at the portfolio level.
    """
    
    def __init__(
        self,
        data: Union[vbt.Data, Dict[str, vbt.Data]],
        params: Dict[str, Any],
        base_timeframe: str = "1h"
    ):
        """
        Initialize MTF DMA ATR strategy.
        
        Args:
            data: Either multi-timeframe vbt.Data or dict of timeframe to vbt.Data
            params: Strategy parameters including:
                - fast_window: Fast MA period (default: 10)
                - slow_window: Slow MA period (default: 30)
                - atr_window: ATR calculation period (default: 14)
                - atr_multiplier_sl: ATR multiplier for stop loss (default: 2.0)
                - atr_multiplier_tp: ATR multiplier for take profit (default: 3.0)
                - use_mtf_confirmation: Whether to use MTF confirmation (default: True)
                - trend_alignment_threshold: Min alignment score for entry (default: 0.3)
                - mtf_weights: Weights for different timeframes
            base_timeframe: Base timeframe for signal generation
        """
        # Set default parameters
        default_params = {
            'fast_window': 10,
            'slow_window': 30,
            'atr_window': 14,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'use_mtf_confirmation': True,
            'trend_alignment_threshold': 0.3,
            'mtf_weights': {'1h': 1.0, '4h': 1.5, '1d': 2.0}
        }
        
        # Merge with provided params
        params = {**default_params, **params}
        
        # Initialize parent MTF strategy
        super().__init__(data, params, base_timeframe)
        
        # Store MTF-specific parameters
        self.trend_alignment_threshold = params['trend_alignment_threshold']
        
        logger.info(
            f"Initialized MTF_DMA_ATR_Strategy with {len(self.timeframes)} timeframes"
        )
    
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
        indicators = {}
        
        # Extract price data
        close = data.close
        high = data.high
        low = data.low
        
        # Handle multi-symbol data
        if isinstance(close, pd.DataFrame) and close.shape[1] == 1:
            close = close.iloc[:, 0]
            high = high.iloc[:, 0]
            low = low.iloc[:, 0]
        
        # Calculate moving averages
        indicators['sma_fast'] = sma(close, window=self.params['fast_window'])
        indicators['sma_slow'] = sma(close, window=self.params['slow_window'])
        
        # Calculate ATR
        indicators['atr'] = atr(high, low, close, window=self.params['atr_window'])
        
        # Calculate trend strength (normalized by ATR)
        indicators['trend_strength'] = (
            indicators['sma_fast'] - indicators['sma_slow']
        ) / indicators['atr'].where(indicators['atr'] > 0, 1)  # Avoid division by zero
        
        # Store close for reference
        indicators['close'] = close
        
        # Calculate RSI for momentum confluence
        indicators['rsi'] = vbt.talib("RSI").run(close, timeperiod=14).real
        
        return indicators
    
    def _generate_base_signals(self) -> Dict[str, pd.Series]:
        """
        Generate base signals from primary timeframe.
        
        Returns:
            Dictionary of base signals
        """
        # Get base timeframe indicators
        base_indicators = self.mtf_indicators[self.base_timeframe]
        
        fast_ma = base_indicators['sma_fast']
        slow_ma = base_indicators['sma_slow']
        atr_values = base_indicators['atr']
        close = base_indicators['close']
        
        # Generate crossover signals
        long_entries = (
            (fast_ma > slow_ma) &
            (fast_ma.shift(1) <= slow_ma.shift(1))
        )
        
        long_exits = (
            (fast_ma < slow_ma) &
            (fast_ma.shift(1) >= slow_ma.shift(1))
        )
        
        # Short signals (opposite)
        short_entries = long_exits.copy()
        short_exits = long_entries.copy()
        
        # Apply base trend filter
        trend_filter = base_indicators['trend_strength'] > 0.2
        long_entries = long_entries & trend_filter
        short_entries = short_entries & ~trend_filter
        
        # Calculate stop levels
        sl_levels = pd.Series(np.nan, index=close.index)
        tp_levels = pd.Series(np.nan, index=close.index)
        
        # Long stop levels
        if long_entries.any():
            sl_levels[long_entries] = (
                close[long_entries] - 
                self.params['atr_multiplier_sl'] * atr_values[long_entries]
            )
            tp_levels[long_entries] = (
                close[long_entries] + 
                self.params['atr_multiplier_tp'] * atr_values[long_entries]
            )
        
        # Short stop levels
        if short_entries.any():
            sl_levels[short_entries] = (
                close[short_entries] + 
                self.params['atr_multiplier_sl'] * atr_values[short_entries]
            )
            tp_levels[short_entries] = (
                close[short_entries] - 
                self.params['atr_multiplier_tp'] * atr_values[short_entries]
            )
        
        return {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'sl_levels': sl_levels if not sl_levels.isna().all() else None,
            'tp_levels': tp_levels if not tp_levels.isna().all() else None
        }
    
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
            
            # Only take long entries when trend alignment is positive
            if 'long_entries' in confirmed_signals:
                confirmed_signals['long_entries'] = (
                    confirmed_signals['long_entries'] & 
                    (trend_score > self.trend_alignment_threshold)
                )
            
            # Only take short entries when trend alignment is negative
            if 'short_entries' in confirmed_signals:
                confirmed_signals['short_entries'] = (
                    confirmed_signals['short_entries'] & 
                    (trend_score < -self.trend_alignment_threshold)
                )
            
            # Exit longs when trend turns strongly negative
            if 'long_exits' in confirmed_signals:
                trend_exit = trend_score < -0.5
                confirmed_signals['long_exits'] = (
                    confirmed_signals['long_exits'] | trend_exit
                )
            
            # Exit shorts when trend turns strongly positive
            if 'short_exits' in confirmed_signals:
                trend_exit = trend_score > 0.5
                confirmed_signals['short_exits'] = (
                    confirmed_signals['short_exits'] | trend_exit
                )
        
        # Apply momentum confluence filter
        if 'momentum_confluence' in mtf_indicators:
            momentum = mtf_indicators['momentum_confluence']
            
            # Additional confirmation for entries
            if 'long_entries' in confirmed_signals:
                confirmed_signals['long_entries'] = (
                    confirmed_signals['long_entries'] & (momentum > 0.1)
                )
            
            if 'short_entries' in confirmed_signals:
                confirmed_signals['short_entries'] = (
                    confirmed_signals['short_entries'] & (momentum < -0.1)
                )
        
        # Check higher timeframe agreement
        higher_tf_confirmation = self._check_higher_timeframe_trends()
        if higher_tf_confirmation is not None:
            if 'long_entries' in confirmed_signals:
                confirmed_signals['long_entries'] = (
                    confirmed_signals['long_entries'] & higher_tf_confirmation
                )
            if 'short_entries' in confirmed_signals:
                confirmed_signals['short_entries'] = (
                    confirmed_signals['short_entries'] & ~higher_tf_confirmation
                )
        
        return confirmed_signals
    
    def _check_higher_timeframe_trends(self) -> Optional[pd.Series]:
        """
        Check if higher timeframes are in agreement with the trend.
        
        Returns:
            Boolean series indicating bullish confirmation
        """
        # Get base index
        base_index = self.data.wrapper.index
        confirmation_score = pd.Series(0.0, index=base_index)
        weight_sum = 0.0
        
        # Check each timeframe
        for tf in self.timeframes:
            if tf in self.mtf_indicators:
                tf_indicators = self.mtf_indicators[tf]
                
                # Check if this timeframe is trending
                if 'trend_strength' in tf_indicators:
                    trend_strength = tf_indicators['trend_strength']
                    
                    # Align to base index if needed
                    if not trend_strength.index.equals(base_index):
                        if hasattr(trend_strength, 'vbt'):
                            trend_strength = trend_strength.vbt.resample_closing(base_index)
                        else:
                            trend_strength = trend_strength.reindex(base_index, method='ffill')
                    
                    # Weight by timeframe importance
                    weight = self.mtf_weights.get(tf, 1.0)
                    confirmation_score += trend_strength * weight
                    weight_sum += weight
        
        # Normalize by total weight
        if weight_sum > 0:
            confirmation_score /= weight_sum
        
        # Return boolean confirmation (positive trend across timeframes)
        return confirmation_score > 0.2
    
    def get_strategy_description(self) -> str:
        """Get a detailed description of the strategy configuration."""
        desc = f"""
        {self.__class__.__name__} Configuration:
        
        Base Parameters:
        - Fast MA Window: {self.params['fast_window']}
        - Slow MA Window: {self.params['slow_window']}
        - ATR Window: {self.params['atr_window']}
        - Stop Loss: {self.params['atr_multiplier_sl']}x ATR
        - Take Profit: {self.params['atr_multiplier_tp']}x ATR
        
        Multi-Timeframe Settings:
        - Timeframes: {', '.join(self.timeframes)}
        - Base Timeframe: {self.base_timeframe}
        - MTF Confirmation: {'Enabled' if self.use_mtf_confirmation else 'Disabled'}
        - Trend Alignment Threshold: {self.trend_alignment_threshold}
        - MTF Weights: {self.mtf_weights}
        
        Strategy Logic:
        - Enter long when fast MA crosses above slow MA with MTF confirmation
        - Enter short when fast MA crosses below slow MA with MTF confirmation
        - Dynamic stops based on ATR volatility
        - Multi-timeframe trend alignment required for entries
        - Higher timeframe trends provide additional confirmation
        """
        
        return desc
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"timeframes={len(self.timeframes)}, "
            f"base={self.base_timeframe}, "
            f"fast={self.params['fast_window']}, "
            f"slow={self.params['slow_window']})"
        ) 