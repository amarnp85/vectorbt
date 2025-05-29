"""
Stop Loss Management Module

Implements various stop loss methods including:
- Dynamic stop loss based on market conditions
- ATR-based stop loss
- Trailing stop loss
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Union, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class DynamicStopLoss:
    """
    Dynamic stop loss that adjusts based on market conditions.
    
    Tightens stops in volatile markets and loosens in calm markets.
    """
    
    def __init__(
        self,
        base_stop_pct: float = 0.02,
        volatility_lookback: int = 20,
        volatility_adjustment: float = 1.0,
        min_stop_pct: float = 0.01,
        max_stop_pct: float = 0.05
    ):
        """
        Initialize dynamic stop loss.
        
        Args:
            base_stop_pct: Base stop loss percentage
            volatility_lookback: Period for volatility calculation
            volatility_adjustment: How much to adjust for volatility
            min_stop_pct: Minimum stop loss percentage
            max_stop_pct: Maximum stop loss percentage
        """
        self.base_stop_pct = base_stop_pct
        self.volatility_lookback = volatility_lookback
        self.volatility_adjustment = volatility_adjustment
        self.min_stop_pct = min_stop_pct
        self.max_stop_pct = max_stop_pct
    
    def calculate_stop_loss(
        self,
        entry_price: pd.Series,
        price_data: pd.DataFrame,
        position_type: str = 'long'
    ) -> pd.Series:
        """
        Calculate dynamic stop loss levels.
        
        Args:
            entry_price: Entry prices
            price_data: DataFrame with OHLC data
            position_type: 'long' or 'short'
            
        Returns:
            Series of stop loss prices
        """
        # Calculate historical volatility
        returns = price_data['close'].pct_change()
        volatility = returns.rolling(self.volatility_lookback).std()
        
        # Normalize volatility (use percentile rank)
        volatility_rank = volatility.rolling(252).rank(pct=True)
        
        # Adjust stop percentage based on volatility
        # Higher volatility = tighter stops
        adjusted_stop_pct = self.base_stop_pct * (1 + self.volatility_adjustment * volatility_rank)
        adjusted_stop_pct = adjusted_stop_pct.clip(self.min_stop_pct, self.max_stop_pct)
        
        # Calculate stop loss price
        if position_type == 'long':
            stop_loss = entry_price * (1 - adjusted_stop_pct)
        else:
            stop_loss = entry_price * (1 + adjusted_stop_pct)
        
        return stop_loss


class ATRStopLoss:
    """
    ATR-based stop loss.
    
    Uses Average True Range to set stop loss distance.
    """
    
    def __init__(
        self,
        atr_period: int = 14,
        atr_multiplier: float = 2.0,
        use_close: bool = False
    ):
        """
        Initialize ATR stop loss.
        
        Args:
            atr_period: Period for ATR calculation
            atr_multiplier: ATR multiplier for stop distance
            use_close: Use close price instead of low/high for stops
        """
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.use_close = use_close
    
    def calculate_stop_loss(
        self,
        entry_price: pd.Series,
        price_data: pd.DataFrame,
        position_type: str = 'long'
    ) -> pd.Series:
        """
        Calculate ATR-based stop loss levels.
        
        Args:
            entry_price: Entry prices
            price_data: DataFrame with OHLC data
            position_type: 'long' or 'short'
            
        Returns:
            Series of stop loss prices
        """
        # Calculate ATR
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(self.atr_period).mean()
        
        # Calculate stop loss
        stop_distance = atr * self.atr_multiplier
        
        if position_type == 'long':
            if self.use_close:
                stop_loss = entry_price - stop_distance
            else:
                # Use swing low
                swing_low = low.rolling(self.atr_period).min()
                stop_loss = pd.Series(
                    np.minimum(entry_price - stop_distance, swing_low),
                    index=entry_price.index
                )
        else:
            if self.use_close:
                stop_loss = entry_price + stop_distance
            else:
                # Use swing high
                swing_high = high.rolling(self.atr_period).max()
                stop_loss = pd.Series(
                    np.maximum(entry_price + stop_distance, swing_high),
                    index=entry_price.index
                )
        
        return stop_loss


class TrailingStopLoss:
    """
    Trailing stop loss that follows price movement.
    
    Adjusts stop loss to lock in profits as price moves favorably.
    """
    
    def __init__(
        self,
        trail_pct: float = 0.02,
        trail_activation: float = 0.01,
        use_atr: bool = False,
        atr_period: int = 14,
        atr_multiplier: float = 2.0
    ):
        """
        Initialize trailing stop loss.
        
        Args:
            trail_pct: Trailing stop percentage
            trail_activation: Profit percentage to activate trailing
            use_atr: Use ATR for trailing distance
            atr_period: Period for ATR calculation
            atr_multiplier: ATR multiplier for trail distance
        """
        self.trail_pct = trail_pct
        self.trail_activation = trail_activation
        self.use_atr = use_atr
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    def calculate_stop_loss(
        self,
        entry_price: pd.Series,
        price_data: pd.DataFrame,
        position_type: str = 'long'
    ) -> pd.DataFrame:
        """
        Calculate trailing stop loss levels.
        
        Args:
            entry_price: Entry prices
            price_data: DataFrame with OHLC data
            position_type: 'long' or 'short'
            
        Returns:
            DataFrame with columns: 'stop_loss', 'is_trailing'
        """
        close = price_data['close']
        
        # Initialize stop loss tracking
        stop_loss = pd.Series(index=close.index, dtype=float)
        is_trailing = pd.Series(False, index=close.index)
        
        # Calculate trail distance
        if self.use_atr:
            high = price_data['high']
            low = price_data['low']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.atr_period).mean()
            trail_distance = atr * self.atr_multiplier
        else:
            trail_distance = close * self.trail_pct
        
        # Track highest/lowest price since entry
        if position_type == 'long':
            # Initial stop loss
            initial_stop = entry_price * (1 - self.trail_pct)
            
            # Track highest price
            highest_price = close.expanding().max()
            
            # Activate trailing when profit threshold reached
            profit_pct = (close - entry_price) / entry_price
            trailing_active = profit_pct >= self.trail_activation
            
            # Calculate trailing stop
            trailing_stop = highest_price - trail_distance
            
            # Use trailing stop when active, otherwise initial stop
            stop_loss = pd.Series(
                np.where(trailing_active, trailing_stop, initial_stop),
                index=close.index
            )
            is_trailing = trailing_active
            
        else:  # short position
            # Initial stop loss
            initial_stop = entry_price * (1 + self.trail_pct)
            
            # Track lowest price
            lowest_price = close.expanding().min()
            
            # Activate trailing when profit threshold reached
            profit_pct = (entry_price - close) / entry_price
            trailing_active = profit_pct >= self.trail_activation
            
            # Calculate trailing stop
            trailing_stop = lowest_price + trail_distance
            
            # Use trailing stop when active, otherwise initial stop
            stop_loss = pd.Series(
                np.where(trailing_active, trailing_stop, initial_stop),
                index=close.index
            )
            is_trailing = trailing_active
        
        return pd.DataFrame({
            'stop_loss': stop_loss,
            'is_trailing': is_trailing
        })


class AdaptiveStopLoss:
    """
    Adaptive stop loss that combines multiple methods.
    
    Switches between different stop loss methods based on market conditions.
    """
    
    def __init__(
        self,
        methods: Optional[Dict[str, Any]] = None,
        regime_lookback: int = 20
    ):
        """
        Initialize adaptive stop loss.
        
        Args:
            methods: Dictionary of stop loss methods to use
            regime_lookback: Period for regime detection
        """
        if methods is None:
            # Default methods
            self.methods = {
                'volatile': ATRStopLoss(atr_multiplier=3.0),
                'trending': TrailingStopLoss(trail_pct=0.02),
                'ranging': DynamicStopLoss(base_stop_pct=0.015)
            }
        else:
            self.methods = methods
        
        self.regime_lookback = regime_lookback
    
    def detect_regime(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Detect market regime (volatile, trending, ranging).
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            Series with regime labels
        """
        close = price_data['close']
        returns = close.pct_change()
        
        # Calculate metrics
        volatility = returns.rolling(self.regime_lookback).std()
        trend_strength = abs(close.rolling(self.regime_lookback).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        ))
        
        # Normalize metrics
        vol_rank = volatility.rolling(252).rank(pct=True)
        trend_rank = trend_strength.rolling(252).rank(pct=True)
        
        # Classify regime
        regime = pd.Series('ranging', index=close.index)
        regime[vol_rank > 0.7] = 'volatile'
        regime[trend_rank > 0.7] = 'trending'
        
        return regime
    
    def calculate_stop_loss(
        self,
        entry_price: pd.Series,
        price_data: pd.DataFrame,
        position_type: str = 'long'
    ) -> pd.DataFrame:
        """
        Calculate adaptive stop loss levels.
        
        Args:
            entry_price: Entry prices
            price_data: DataFrame with OHLC data
            position_type: 'long' or 'short'
            
        Returns:
            DataFrame with stop loss and regime info
        """
        # Detect regime
        regime = self.detect_regime(price_data)
        
        # Calculate stop loss for each regime
        stop_losses = {}
        for regime_type, method in self.methods.items():
            sl = method.calculate_stop_loss(entry_price, price_data, position_type)
            if isinstance(sl, pd.DataFrame):
                stop_losses[regime_type] = sl['stop_loss']
            else:
                stop_losses[regime_type] = sl
        
        # Select appropriate stop loss based on regime
        stop_loss = pd.Series(index=price_data.index, dtype=float)
        for regime_type in self.methods.keys():
            mask = regime == regime_type
            stop_loss[mask] = stop_losses[regime_type][mask]
        
        return pd.DataFrame({
            'stop_loss': stop_loss,
            'regime': regime
        }) 