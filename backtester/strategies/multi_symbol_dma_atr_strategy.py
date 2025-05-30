"""
Multi-Symbol DMA ATR Strategy

A multi-symbol implementation of the DMA ATR strategy that leverages VectorBT Pro's
native multi-symbol support for efficient portfolio-level signal generation.

This strategy applies the DMA ATR logic across multiple symbols simultaneously
while adding cross-symbol filters and portfolio-level position management.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional
import logging

from .multi_symbol_strategy_base import MultiSymbolStrategy
from ..indicators.simple_indicators import sma, atr

logger = logging.getLogger(__name__)


class MultiSymbolDMAATRStrategy(MultiSymbolStrategy):
    """
    Multi-symbol DMA ATR strategy with cross-symbol analysis.
    
    Features:
    - Fast/slow MA crossover for trend detection across all symbols
    - ATR-based dynamic stop losses for each symbol
    - Cross-symbol correlation filtering
    - Market regime detection
    - Symbol ranking and selection
    - Portfolio-level position sizing
    """
    
    def __init__(self, data: vbt.Data, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the multi-symbol DMA ATR strategy.
        
        Args:
            data: VectorBT Data object with multiple symbols
            params: Strategy parameters including:
                - fast_period: Fast MA period (default: 20)
                - slow_period: Slow MA period (default: 50)
                - atr_period: ATR calculation period (default: 14)
                - atr_multiplier_sl: ATR multiplier for stop loss (default: 2.0)
                - atr_multiplier_tp: ATR multiplier for take profit (default: 3.0)
                - use_volume_filter: Whether to use volume confirmation (default: False)
                - max_correlation: Max correlation for position sizing (default: 0.8)
                - min_market_breadth: Min market breadth for long signals (default: 0.3)
        """
        # Set default strategy-specific parameters
        default_strategy_params = {
            'fast_period': 20,
            'slow_period': 50,
            'atr_period': 14,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'use_volume_filter': False,
            'volume_period': 20,
            'trend_strength_threshold': 0.005,  # 0.5% separation for trend confirmation
            'max_correlation': 0.8,
            'min_market_breadth': 0.3
        }
        
        # Merge with provided params
        if params is None:
            params = {}
        params = {**default_strategy_params, **params}
        
        super().__init__(data, params)
        
        # Strategy-specific parameters
        self.fast_period = params['fast_period']
        self.slow_period = params['slow_period']
        self.atr_period = params['atr_period']
        self.atr_multiplier_sl = params['atr_multiplier_sl']
        self.atr_multiplier_tp = params['atr_multiplier_tp']
        self.use_volume_filter = params['use_volume_filter']
        self.volume_period = params.get('volume_period', 20)
        self.trend_strength_threshold = params['trend_strength_threshold']
        
        logger.info(f"Initialized MultiSymbolDMAATRStrategy with {len(self.symbols)} symbols")
        logger.info(f"Parameters: fast={self.fast_period}, slow={self.slow_period}, atr={self.atr_period}")
    
    def _calculate_custom_indicators(self) -> Dict[str, Any]:
        """
        Calculate DMA ATR specific indicators for all symbols.
        
        Returns:
            Dictionary of custom indicators
        """
        close = self.data.close
        high = self.data.high
        low = self.data.low
        volume = self.data.volume if hasattr(self.data, 'volume') else None
        
        indicators = {}
        
        # Moving averages for all symbols (VBT broadcasts automatically)
        indicators['fast_ma'] = vbt.talib("SMA").run(close, timeperiod=self.fast_period).real
        indicators['slow_ma'] = vbt.talib("SMA").run(close, timeperiod=self.slow_period).real
        
        # ATR for all symbols
        indicators['atr'] = vbt.talib("ATR").run(high, low, close, timeperiod=self.atr_period).real
        
        # Volume indicators if enabled
        if self.use_volume_filter and volume is not None:
            indicators['volume_ma'] = vbt.talib("SMA").run(volume, timeperiod=self.volume_period).real
            indicators['volume_ratio'] = volume / indicators['volume_ma']
        
        # Trend strength for all symbols
        indicators['trend_strength'] = (indicators['fast_ma'] - indicators['slow_ma']) / indicators['slow_ma']
        
        return indicators
    
    def _generate_base_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Generate base DMA ATR signals for all symbols.
        
        Returns:
            Dictionary with signal DataFrames
        """
        fast_ma = self.indicators['fast_ma']
        slow_ma = self.indicators['slow_ma']
        atr_values = self.indicators['atr']
        trend_strength = self.indicators['trend_strength']
        close = self.data.close
        
        # Generate crossover signals for all symbols at once
        fast_above_slow = fast_ma > slow_ma
        fast_crosses_above = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        fast_crosses_below = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # Trend confirmation
        bullish_trend = trend_strength > self.trend_strength_threshold
        bearish_trend = trend_strength < -self.trend_strength_threshold
        
        # Base entry conditions
        long_entry_conditions = fast_crosses_above & bullish_trend
        short_entry_conditions = fast_crosses_below & bearish_trend
        
        # Traditional exit conditions
        long_exit_conditions = fast_crosses_below
        short_exit_conditions = fast_crosses_above
        
        # Volume filter if enabled
        if self.use_volume_filter and 'volume_ratio' in self.indicators:
            volume_confirmation = self.indicators['volume_ratio'] > 1.2
            long_entry_conditions = long_entry_conditions & volume_confirmation
            short_entry_conditions = short_entry_conditions & volume_confirmation
        
        # Clean signals for each symbol
        cleaned_long_entries = pd.DataFrame(index=long_entry_conditions.index)
        cleaned_long_exits = pd.DataFrame(index=long_exit_conditions.index)
        cleaned_short_entries = pd.DataFrame(index=short_entry_conditions.index)
        cleaned_short_exits = pd.DataFrame(index=short_exit_conditions.index)
        
        for symbol in self.symbols:
            if symbol in long_entry_conditions.columns:
                # Clean long signals
                clean_long_entry, clean_long_exit = long_entry_conditions[symbol].vbt.signals.clean(
                    long_exit_conditions[symbol]
                )
                cleaned_long_entries[symbol] = clean_long_entry
                cleaned_long_exits[symbol] = clean_long_exit
                
                # Clean short signals
                clean_short_entry, clean_short_exit = short_entry_conditions[symbol].vbt.signals.clean(
                    short_exit_conditions[symbol]
                )
                cleaned_short_entries[symbol] = clean_short_entry
                cleaned_short_exits[symbol] = clean_short_exit
        
        # Calculate stop levels
        sl_levels = pd.DataFrame(index=close.index)
        tp_levels = pd.DataFrame(index=close.index)
        
        for symbol in self.symbols:
            if symbol in close.columns:
                # Stop loss levels (percentage-based)
                sl_pct = atr_values[symbol] / close[symbol] * self.atr_multiplier_sl
                tp_pct = atr_values[symbol] / close[symbol] * self.atr_multiplier_tp
                
                sl_levels[symbol] = sl_pct
                tp_levels[symbol] = tp_pct
        
        signals = {
            'long_entries': cleaned_long_entries,
            'long_exits': cleaned_long_exits,
            'short_entries': cleaned_short_entries,
            'short_exits': cleaned_short_exits,
            'sl_levels': sl_levels,
            'tp_levels': tp_levels
        }
        
        return signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'atr_period': self.atr_period,
            'atr_multiplier_sl': self.atr_multiplier_sl,
            'atr_multiplier_tp': self.atr_multiplier_tp,
            'use_volume_filter': self.use_volume_filter,
            'volume_period': self.volume_period,
            'trend_strength_threshold': self.trend_strength_threshold,
            'correlation_lookback': self.correlation_lookback,
            'relative_strength_lookback': self.relative_strength_lookback,
            'max_active_symbols': self.max_active_symbols
        }
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        if self.fast_period >= self.slow_period:
            logger.error("Fast period must be less than slow period")
            return False
        
        if self.atr_period <= 0:
            logger.error("ATR period must be positive")
            return False
        
        if self.atr_multiplier_sl <= 0 or self.atr_multiplier_tp <= 0:
            logger.error("ATR multipliers must be positive")
            return False
        
        if len(self.symbols) < 2:
            logger.error("Multi-symbol strategy requires at least 2 symbols")
            return False
        
        return True
    
    def get_strategy_description(self) -> str:
        """Get strategy description."""
        return (
            f"Multi-Symbol DMA ATR Strategy: "
            f"Fast MA({self.fast_period}) vs Slow MA({self.slow_period}) crossover "
            f"with ATR({self.atr_period}) stops across {len(self.symbols)} symbols. "
            f"Includes cross-symbol correlation filtering and market regime detection."
        )