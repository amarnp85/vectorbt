"""
Mean Reversion Strategy

A strategy that trades on the assumption that prices will revert to their mean.
Uses Bollinger Bands and RSI for entry signals with regime-based filters.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional, Union
import logging

from .base_strategy import BaseStrategy
from ..indicators.simple_indicators import bollinger_bands, rsi, atr

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI.
    
    Features:
    - Bollinger Bands for overbought/oversold detection
    - RSI for confirmation
    - Regime filter to avoid trending markets
    - ATR-based stop levels
    
    Note: Position sizing and risk management are handled at the portfolio level.
    """
    
    def __init__(
        self,
        data: vbt.Data,
        params: Dict[str, Any],
        **kwargs
    ):
        """
        Initialize mean reversion strategy.
        
        Args:
            data: VBT Data object with price data
            params: Strategy parameters including:
                - bb_period: Bollinger Bands period (default: 20)
                - bb_std: Number of standard deviations (default: 2)
                - rsi_period: RSI period (default: 14)
                - rsi_oversold: RSI oversold threshold (default: 30)
                - rsi_overbought: RSI overbought threshold (default: 70)
                - regime_filter: Whether to use regime filter (default: True)
                - atr_multiplier_sl: ATR multiplier for stop loss (default: 2.0)
                - atr_multiplier_tp: ATR multiplier for take profit (default: 3.0)
        """
        # Set default parameters
        default_params = {
            'bb_period': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'regime_filter': True,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0
        }
        
        # Merge with provided params
        self.params = {**default_params, **params}
        
        super().__init__(data, self.params, **kwargs)
    
    def init_indicators(self) -> Dict[str, Any]:
        """Calculate mean reversion indicators using the new simple indicators."""
        close = self.data.close
        high = self.data.high
        low = self.data.low
        
        # Handle multi-symbol data
        if isinstance(close, pd.DataFrame) and close.shape[1] > 1:
            # For multi-symbol, use first symbol or implement symbol-specific logic
            close = close.iloc[:, 0]
            high = high.iloc[:, 0]
            low = low.iloc[:, 0]
        
        # Bollinger Bands - using new simple indicator
        bb_upper, bb_middle, bb_lower = bollinger_bands(
            close,
            window=self.params['bb_period'],
            std_dev=self.params['bb_std']
        )
        
        self.indicators['bb_upper'] = bb_upper
        self.indicators['bb_middle'] = bb_middle
        self.indicators['bb_lower'] = bb_lower
        self.indicators['bb_width'] = bb_upper - bb_lower
        self.indicators['bb_percent'] = (close - bb_lower) / (bb_upper - bb_lower)
        
        # RSI - using new simple indicator
        self.indicators['rsi'] = rsi(close, window=self.params['rsi_period'])
        
        # Price position relative to bands
        self.indicators['distance_from_middle'] = (close - bb_middle) / bb_middle
        
        # ATR for stop levels
        self.indicators['atr'] = atr(high, low, close, window=14)
        
        # Market regime detection using simple trend detection
        if self.params['regime_filter']:
            # Simple trend detection using SMA
            sma_short = close.rolling(window=20).mean()
            sma_long = close.rolling(window=50).mean()
            
            # Trend strength
            trend_strength = (sma_short - sma_long) / sma_long
            
            # Classify regime
            self.indicators['trend_regime'] = pd.Series('ranging', index=close.index)
            self.indicators['trend_regime'][trend_strength > 0.02] = 'uptrend'
            self.indicators['trend_regime'][trend_strength < -0.02] = 'downtrend'
        
        return self.indicators
    
    def generate_signals(self) -> Dict[str, pd.Series]:
        """Generate mean reversion signals."""
        close = self.data.close
        if isinstance(close, pd.DataFrame) and close.shape[1] > 1:
            close = close.iloc[:, 0]
        
        # Extract indicators
        bb_upper = self.indicators['bb_upper']
        bb_lower = self.indicators['bb_lower']
        bb_percent = self.indicators['bb_percent']
        rsi_values = self.indicators['rsi']
        
        # Entry conditions
        # Long: Price near lower band AND RSI oversold
        long_condition = (
            (bb_percent < 0.2) &  # Near lower band
            (rsi_values < self.params['rsi_oversold']) &
            (close > bb_lower * 0.98)  # Not too far below band
        )
        
        # Short: Price near upper band AND RSI overbought
        short_condition = (
            (bb_percent > 0.8) &  # Near upper band
            (rsi_values > self.params['rsi_overbought']) &
            (close < bb_upper * 1.02)  # Not too far above band
        )
        
        # Exit conditions
        # Exit long: Price crosses middle band or RSI > 50
        exit_long = (
            (close > self.indicators['bb_middle']) |
            (rsi_values > 50)
        )
        
        # Exit short: Price crosses middle band or RSI < 50
        exit_short = (
            (close < self.indicators['bb_middle']) |
            (rsi_values < 50)
        )
        
        # Apply regime filter if enabled
        if self.params['regime_filter'] and 'trend_regime' in self.indicators:
            trend_regime = self.indicators['trend_regime']
            # Only trade in ranging markets
            regime_filter = trend_regime == 'ranging'
            long_condition = long_condition & regime_filter
            short_condition = short_condition & regime_filter
        
        # Generate clean entry/exit signals
        long_entries = pd.Series(False, index=close.index)
        long_exits = pd.Series(False, index=close.index)
        short_entries = pd.Series(False, index=close.index)
        short_exits = pd.Series(False, index=close.index)
        
        # Track position state to generate clean signals
        position = 0
        for i in range(len(close)):
            if position == 0:
                if long_condition.iloc[i]:
                    long_entries.iloc[i] = True
                    position = 1
                elif short_condition.iloc[i]:
                    short_entries.iloc[i] = True
                    position = -1
            elif position == 1:
                if exit_long.iloc[i]:
                    long_exits.iloc[i] = True
                    position = 0
            elif position == -1:
                if exit_short.iloc[i]:
                    short_exits.iloc[i] = True
                    position = 0
        
        # Calculate stop loss levels
        sl_levels = pd.Series(np.nan, index=close.index)
        tp_levels = pd.Series(np.nan, index=close.index)
        
        # Calculate stop losses for entries
        atr_values = self.indicators['atr']
        
        # Long stop levels
        long_entry_mask = long_entries
        sl_levels[long_entry_mask] = close[long_entry_mask] - (
            self.params['atr_multiplier_sl'] * atr_values[long_entry_mask]
        )
        tp_levels[long_entry_mask] = close[long_entry_mask] + (
            self.params['atr_multiplier_tp'] * atr_values[long_entry_mask]
        )
        
        # Short stop levels
        short_entry_mask = short_entries
        sl_levels[short_entry_mask] = close[short_entry_mask] + (
            self.params['atr_multiplier_sl'] * atr_values[short_entry_mask]
        )
        tp_levels[short_entry_mask] = close[short_entry_mask] - (
            self.params['atr_multiplier_tp'] * atr_values[short_entry_mask]
        )
        
        # Store signals in expected format
        self.signals = {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'sl_levels': sl_levels if not sl_levels.isna().all() else None,
            'tp_levels': tp_levels if not tp_levels.isna().all() else None
        }
        
        return self.signals 