"""
Momentum Strategy

A trend-following strategy that enters positions based on momentum indicators
and filters trades based on market regime.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional
import logging

from .base_strategy import BaseStrategy
from ..indicators.simple_indicators import sma, rsi, atr, macd, adx

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy with regime filters and dynamic risk management.
    
    Features:
    - Multiple momentum indicators (ROC, RSI, MACD)
    - Market regime filtering
    - Trend strength confirmation
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
        Initialize momentum strategy.
        
        Args:
            data: VBT Data object with price data
            params: Strategy parameters including:
                - roc_period: Rate of Change period
                - rsi_period: RSI period
                - macd_fast: MACD fast period
                - macd_slow: MACD slow period
                - macd_signal: MACD signal period
                - momentum_threshold: Minimum momentum for entry
                - regime_filter: Whether to use regime filter
                - atr_multiplier_sl: ATR multiplier for stop loss
                - atr_multiplier_tp: ATR multiplier for take profit
        """
        # Set default parameters
        default_params = {
            'roc_period': 20,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'momentum_threshold': 0.02,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'regime_filter': True,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'adx_threshold': 25
        }
        
        # Merge with provided params
        self.params = {**default_params, **params}
        
        super().__init__(data, self.params, **kwargs)
    
    def init_indicators(self) -> Dict[str, Any]:
        """Calculate momentum indicators using the new simple indicators."""
        close = self.data.close
        high = self.data.high
        low = self.data.low
        
        # Handle multi-symbol data
        if isinstance(close, pd.DataFrame) and close.shape[1] > 1:
            # For multi-symbol, use first symbol or implement symbol-specific logic
            close = close.iloc[:, 0]
            high = high.iloc[:, 0]
            low = low.iloc[:, 0]
        
        # Rate of Change (ROC) - using VBT directly as it's not in simple_indicators
        self.indicators['roc'] = vbt.talib('ROC').run(
            close,
            timeperiod=self.params['roc_period']
        ).real / 100  # Convert to decimal
        
        # RSI - using new simple indicator
        self.indicators['rsi'] = rsi(close, window=self.params['rsi_period'])
        
        # MACD - using new simple indicator
        macd_line, signal_line, histogram = macd(
            close,
            fast_window=self.params['macd_fast'],
            slow_window=self.params['macd_slow'],
            signal_window=self.params['macd_signal']
        )
        self.indicators['macd'] = macd_line
        self.indicators['macd_signal'] = signal_line
        self.indicators['macd_hist'] = histogram
        
        # ADX for trend strength - using new simple indicator
        self.indicators['adx'] = adx(high, low, close, period=14)
        
        # Moving averages for trend - using new simple indicators
        self.indicators['sma_20'] = sma(close, window=20)
        self.indicators['sma_50'] = sma(close, window=50)
        self.indicators['sma_200'] = sma(close, window=200)
        
        # ATR for volatility - using new simple indicator
        self.indicators['atr'] = atr(high, low, close, window=14)
        
        # Momentum score (composite)
        self.indicators['momentum_score'] = self._calculate_momentum_score()
        
        # Simple trend detection
        self.indicators['trend_direction'] = np.where(
            close > self.indicators['sma_50'], 1, -1
        )
        
        return self.indicators
    
    def _calculate_momentum_score(self) -> pd.Series:
        """
        Calculate composite momentum score.
        
        Combines multiple momentum indicators into a single score.
        """
        close = self.data.close
        if isinstance(close, pd.DataFrame) and close.shape[1] > 1:
            close = close.iloc[:, 0]
        
        # Components of momentum score
        scores = []
        
        # 1. ROC component (normalized)
        roc_score = self.indicators['roc'] / self.indicators['roc'].rolling(50).std()
        scores.append(roc_score.fillna(0))
        
        # 2. RSI component (normalized to -1 to 1)
        rsi_score = (self.indicators['rsi'] - 50) / 50
        scores.append(rsi_score)
        
        # 3. MACD component (normalized)
        macd_score = self.indicators['macd_hist'] / close * 100
        macd_score = macd_score / macd_score.rolling(50).std()
        scores.append(macd_score.fillna(0))
        
        # 4. Price vs MA component
        ma_score = (close - self.indicators['sma_50']) / self.indicators['sma_50']
        scores.append(ma_score)
        
        # Combine scores
        momentum_score = pd.concat(scores, axis=1).mean(axis=1)
        
        return momentum_score
    
    def generate_signals(self) -> Dict[str, pd.Series]:
        """Generate momentum-based signals."""
        close = self.data.close
        if isinstance(close, pd.DataFrame) and close.shape[1] > 1:
            close = close.iloc[:, 0]
        
        # Extract indicators
        momentum_score = self.indicators['momentum_score']
        rsi_values = self.indicators['rsi']
        macd_line = self.indicators['macd']
        macd_signal = self.indicators['macd_signal']
        adx_values = self.indicators['adx']
        atr_values = self.indicators['atr']
        
        # Entry conditions
        # Long: Strong positive momentum + trend confirmation
        long_momentum = momentum_score > self.params['momentum_threshold']
        long_macd = macd_line > macd_signal
        long_trend = close > self.indicators['sma_50']
        strong_trend = adx_values > self.params['adx_threshold']
        
        long_condition = (
            long_momentum &
            long_macd &
            long_trend &
            strong_trend &
            (rsi_values < self.params['rsi_overbought'])
        )
        
        # Short: Strong negative momentum + trend confirmation
        short_momentum = momentum_score < -self.params['momentum_threshold']
        short_macd = macd_line < macd_signal
        short_trend = close < self.indicators['sma_50']
        
        short_condition = (
            short_momentum &
            short_macd &
            short_trend &
            strong_trend &
            (rsi_values > self.params['rsi_oversold'])
        )
        
        # Exit conditions
        # Exit long: Momentum reversal or RSI overbought
        exit_long = (
            (momentum_score < 0) |
            (rsi_values > self.params['rsi_overbought'] + 10) |
            (macd_line < macd_signal)
        )
        
        # Exit short: Momentum reversal or RSI oversold
        exit_short = (
            (momentum_score > 0) |
            (rsi_values < self.params['rsi_oversold'] - 10) |
            (macd_line > macd_signal)
        )
        
        # Apply regime filter if enabled
        if self.params['regime_filter']:
            # Simple regime: only trade in trending markets (high ADX)
            regime_filter = adx_values > 20
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
        
        # Calculate stop loss and take profit levels
        sl_levels = pd.Series(np.nan, index=close.index)
        tp_levels = pd.Series(np.nan, index=close.index)
        
        # Long stop losses and take profits
        long_entry_mask = long_entries
        sl_levels[long_entry_mask] = close[long_entry_mask] - (
            self.params['atr_multiplier_sl'] * atr_values[long_entry_mask]
        )
        tp_levels[long_entry_mask] = close[long_entry_mask] + (
            self.params['atr_multiplier_tp'] * atr_values[long_entry_mask]
        )
        
        # Short stop losses and take profits
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
            'sl_levels': sl_levels,
            'tp_levels': tp_levels
        }
        
        return self.signals
    
    def get_strategy_metadata(self) -> Dict[str, Any]:
        """Get strategy-specific metadata.
        
        Returns:
            Dictionary with strategy configuration and characteristics
        """
        return {
            'strategy_type': 'momentum',
            'indicators_used': ['ROC', 'RSI', 'MACD', 'ADX', 'SMA'],
            'regime_filter_enabled': self.params['regime_filter'],
            'risk_parameters': {
                'atr_multiplier_sl': self.params['atr_multiplier_sl'],
                'atr_multiplier_tp': self.params['atr_multiplier_tp']
            },
            'entry_thresholds': {
                'momentum_threshold': self.params['momentum_threshold'],
                'adx_threshold': self.params['adx_threshold'],
                'rsi_overbought': self.params['rsi_overbought'],
                'rsi_oversold': self.params['rsi_oversold']
            }
        } 