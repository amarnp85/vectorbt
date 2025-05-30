"""
DMA ATR Trend Strategy

A trend-following strategy that uses dual moving averages for trend detection
and ATR-based stops for risk management.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional

from .base_strategy import BaseStrategy
from ..indicators.simple_indicators import sma, atr
from ..utilities.structured_logging import get_logger

logger = get_logger(__name__)


class DMAATRTrendStrategy(BaseStrategy):
    """
    Dual Moving Average with ATR-based stops strategy.
    
    Features:
    - Fast/slow MA crossover for trend detection
    - ATR-based dynamic stop losses
    - Optional volume confirmation
    - Trend strength filtering
    
    Note: Position sizing and risk management are handled at the portfolio level.
    """
    
    def __init__(self, data: vbt.Data, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the DMA ATR strategy.
        
        Args:
            data: VectorBTPro Data object with OHLCV data
            params: Strategy parameters including:
                - fast_period/fast_window: Fast MA period (default: 10)
                - slow_period/slow_window: Slow MA period (default: 30)
                - atr_period/atr_window: ATR calculation period (default: 14)
                - atr_multiplier: ATR multiplier for stops (default: 2.0)
                - atr_multiplier_sl: ATR multiplier for stop loss (default: 2.0)
                - atr_multiplier_tp: ATR multiplier for take profit (default: 3.0)
                - volume_factor: Volume filter multiplier (default: None)
                - use_volume_filter: Whether to use volume confirmation (default: False)
                - volume_window: Volume MA period (default: 20)
        """
        # Set default parameters
        default_params = {
            'fast_window': 10,
            'slow_window': 30,
            'atr_window': 14,
            'atr_multiplier': 2.0,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0,
            'use_volume_filter': False,
            'volume_window': 20,
            'volume_factor': None
        }
        
        # Handle None params
        if params is None:
            params = {}
        
        # Handle parameter name aliases for backward compatibility
        if 'fast_period' in params:
            params['fast_window'] = params.pop('fast_period')
        if 'slow_period' in params:
            params['slow_window'] = params.pop('slow_period')
        if 'atr_period' in params:
            params['atr_window'] = params.pop('atr_period')
            
        # If atr_multiplier is provided without sl/tp variants, use it for both
        if 'atr_multiplier' in params and 'atr_multiplier_sl' not in params:
            params['atr_multiplier_sl'] = params['atr_multiplier']
        if 'atr_multiplier' in params and 'atr_multiplier_tp' not in params:
            params['atr_multiplier_tp'] = params['atr_multiplier'] * 1.5  # TP is typically larger
            
        # Handle volume_factor
        if params.get('volume_factor') is not None:
            params['use_volume_filter'] = True
        
        # Merge with provided parameters
        merged_params = {**default_params, **params}
        
        # Initialize base class
        super().__init__(data, merged_params)
        
        # Validate parameters
        self._validate_parameters()
        
        # Only log initialization if not in quiet mode
        if logger.level != "ERROR":
            logger.info(f"Initialized {self.__class__.__name__} strategy")
    
    def _validate_parameters(self):
        """Validate strategy parameters."""
        if self.params['fast_window'] <= 0:
            raise ValueError("Fast window must be positive")
            
        if self.params['slow_window'] <= 0:
            raise ValueError("Slow window must be positive")
            
        if self.params['atr_window'] <= 0:
            raise ValueError("ATR window must be positive")
            
        if self.params['fast_window'] >= self.params['slow_window']:
            raise ValueError("Fast window must be less than slow window")
        
        if self.params['atr_multiplier_sl'] <= 0:
            raise ValueError("ATR multiplier for stop loss must be positive")
        
        if self.params['atr_multiplier_tp'] <= 0:
            raise ValueError("ATR multiplier for take profit must be positive")
    
    def init_indicators(self) -> Dict[str, Any]:
        """
        Calculate technical indicators for the strategy.
        
        Returns:
            Dictionary of calculated indicators
        """
        # Only log if not in quiet mode
        if logger.level != "ERROR":
            logger.info("Calculating DMA ATR indicators")
        
        # Extract price data using get() method
        close = self.data.get('close')
        high = self.data.get('high')
        low = self.data.get('low')
        volume = self.data.get('volume')
        
        # Calculate moving averages
        self.indicators['fast_ma'] = sma(close, window=self.params['fast_window'])
        self.indicators['slow_ma'] = sma(close, window=self.params['slow_window'])
        
        # Calculate ATR for volatility-based stops
        self.indicators['atr'] = atr(
            high, low, close, 
            window=self.params['atr_window']
        )
        
        # Calculate volume filter if enabled
        if self.params['use_volume_filter'] and volume is not None:
            self.indicators['volume_ma'] = sma(volume, window=self.params['volume_window'])
            self.indicators['volume'] = volume
        else:
            # Always include volume_sma for test compatibility
            self.indicators['volume_sma'] = sma(volume, window=self.params['volume_window']) if volume is not None else None
        
        # Calculate trend strength (optional)
        self.indicators['trend_strength'] = (
            self.indicators['fast_ma'] - self.indicators['slow_ma']
        ) / self.indicators['slow_ma']
        
        # Only log if not in quiet mode
        if logger.level != "ERROR":
            logger.info(f"Calculated {len(self.indicators)} indicators")
        return self.indicators
    
    def generate_signals(self) -> Dict[str, Any]:
        """
        Generate trading signals based on DMA crossovers.
        
        Key Changes:
        - Only generate entry signals, let stops handle exits
        - Calculate percentage-based stops for VectorBT Pro
        - Allow flat periods (no forced position flipping)
        
        Returns:
            Dictionary with signal arrays
        """
        # Only log if not in quiet mode
        if logger.level != "ERROR":
            logger.info("Generating DMA ATR signals")
        
        # Extract indicators
        fast_ma = self.indicators['fast_ma']
        slow_ma = self.indicators['slow_ma']
        atr_values = self.indicators['atr']
        close = self.data.get('close')
        
        # Ensure we work with consistent data types - convert MultiIndex DataFrames to Series for single symbol
        if isinstance(fast_ma, pd.DataFrame):
            if fast_ma.shape[1] == 1:
                # Single symbol case - extract as Series
                fast_ma = fast_ma.iloc[:, 0]
                slow_ma = slow_ma.iloc[:, 0]
                atr_values = atr_values.iloc[:, 0]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
        
        # Generate crossover signals
        fast_above_slow = fast_ma > slow_ma
        fast_crosses_above = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        fast_crosses_below = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # NEW APPROACH: Time-shifted exit strategy to eliminate ALL signal conflicts
        # Key insight: Never generate simultaneous long exit + short entry signals
        
        # Calculate trend direction (longer-term bias)
        trend_strength = (fast_ma - slow_ma) / slow_ma
        bullish_trend = trend_strength > 0.005  # 0.5% separation indicates uptrend
        bearish_trend = trend_strength < -0.005  # 0.5% separation indicates downtrend
        
        # FINAL APPROACH: Ensure no overlapping signals by mutual exclusion
        # Create gaps between long and short signals to ensure VBT processes both
        
        # Step 1: Create base entry conditions with trend confirmation
        long_entry_conditions = fast_crosses_above & bullish_trend
        short_entry_conditions = fast_crosses_below & bearish_trend
        
        # Step 2: Traditional exit conditions
        long_exit_conditions = fast_crosses_below
        short_exit_conditions = fast_crosses_above
        
        # Step 3: Create mutual exclusion zones around signals
        # Don't allow short entries within 2 bars of long entries and vice versa
        long_exclusion_zone = long_entry_conditions.rolling(window=3, center=True).max().fillna(False).astype(bool)
        short_exclusion_zone = short_entry_conditions.rolling(window=3, center=True).max().fillna(False).astype(bool)
        
        # Step 4: Apply mutual exclusion
        long_entry_final = long_entry_conditions & ~short_exclusion_zone
        short_entry_final = short_entry_conditions & ~long_exclusion_zone
        
        # Apply signal cleaning separately for each direction to avoid conflicts
        if hasattr(long_entry_final, 'vbt') and hasattr(long_entry_final.vbt, 'signals'):
            try:
                # Clean long signals
                long_entries, long_exits = long_entry_final.vbt.signals.clean(long_exit_conditions)
                # Clean short signals  
                short_entries, short_exits = short_entry_final.vbt.signals.clean(short_exit_conditions)
                
                logger.debug(f"Delayed-entry signals: Long entries={long_entries.sum()}, exits={long_exits.sum()}, Short entries={short_entries.sum()}, exits={short_exits.sum()}")
                
            except Exception as e:
                logger.warning(f"VBT signal cleaning failed: {e}, using raw signals")
                long_entries = long_entry_final
                long_exits = long_exit_conditions
                short_entries = short_entry_final
                short_exits = short_exit_conditions
        else:
            # Use raw signals if VBT cleaning not available
            long_entries = long_entry_final
            long_exits = long_exit_conditions  
            short_entries = short_entry_final
            short_exits = short_exit_conditions
        
        # Apply volume filter if enabled
        if self.params['use_volume_filter'] and 'volume' in self.indicators:
            volume = self.indicators['volume']
            volume_ma = self.indicators.get('volume_ma', self.indicators.get('volume_sma'))
            
            # Convert to Series if needed
            if isinstance(volume_ma, pd.DataFrame) and volume_ma.shape[1] == 1:
                volume_ma = volume_ma.iloc[:, 0]
            if isinstance(volume, pd.DataFrame) and volume.shape[1] == 1:
                volume = volume.iloc[:, 0]
            
            if self.params.get('volume_factor'):
                # Use volume factor if provided
                volume_filter = volume > (volume_ma * self.params['volume_factor'])
            else:
                # Default volume filter
                volume_filter = volume > volume_ma
            
            # Only take entry signals with above-average volume
            long_entries = long_entries & volume_filter
        
        # Calculate percentage-based stop levels for VectorBT Pro for both long and short
        # VectorBT expects percentage values (0.05 = 5% stop loss)
        sl_levels = pd.Series(np.nan, index=close.index, name='sl_levels')
        tp_levels = pd.Series(np.nan, index=close.index, name='tp_levels')
        
        # Also store absolute price levels for charting
        sl_price_levels = pd.Series(np.nan, index=close.index, name='sl_price_levels')
        tp_price_levels = pd.Series(np.nan, index=close.index, name='tp_price_levels')
        
        # Calculate stop levels for long positions
        if long_entries.any():
            # Calculate percentage stops based on ATR volatility for longs
            entry_prices = close[long_entries]
            atr_at_entries = atr_values[long_entries]
            
            # For long positions: SL below entry, TP above entry
            # Stop loss percentage: (ATR * multiplier) / entry_price
            sl_percentage = (atr_at_entries * self.params['atr_multiplier_sl']) / entry_prices
            sl_levels[long_entries] = sl_percentage
            
            # Absolute price levels for charting
            sl_price_levels[long_entries] = entry_prices - (atr_at_entries * self.params['atr_multiplier_sl'])
            
            # Take profit percentage: (ATR * multiplier) / entry_price  
            tp_percentage = (atr_at_entries * self.params['atr_multiplier_tp']) / entry_prices
            tp_levels[long_entries] = tp_percentage
            
            # Absolute price levels for charting
            tp_price_levels[long_entries] = entry_prices + (atr_at_entries * self.params['atr_multiplier_tp'])
            
        # Calculate stop levels for short positions  
        if short_entries.any():
            # Calculate percentage stops based on ATR volatility for shorts
            entry_prices = close[short_entries]
            atr_at_entries = atr_values[short_entries]
            
            # For short positions: SL above entry, TP below entry
            # Stop loss percentage: (ATR * multiplier) / entry_price (positive value, will be added to price)
            sl_percentage = (atr_at_entries * self.params['atr_multiplier_sl']) / entry_prices
            sl_levels[short_entries] = sl_percentage  # Positive value for shorts (price + sl_percentage)
            
            # Absolute price levels for charting
            sl_price_levels[short_entries] = entry_prices + (atr_at_entries * self.params['atr_multiplier_sl'])
            
            # Take profit percentage: (ATR * multiplier) / entry_price (positive value, will be subtracted from price)
            tp_percentage = (atr_at_entries * self.params['atr_multiplier_tp']) / entry_prices  
            tp_levels[short_entries] = tp_percentage  # Positive value for shorts (price - tp_percentage)
            
            # Absolute price levels for charting
            tp_price_levels[short_entries] = entry_prices - (atr_at_entries * self.params['atr_multiplier_tp'])
        
        # Calculate entry and exit prices for signal validation
        entry_prices = pd.Series(index=close.index, dtype=float, name='entry_prices')
        exit_prices = pd.Series(index=close.index, dtype=float, name='exit_prices')
        
        # Populate entry prices (use close price at signal time)
        all_entries = long_entries | short_entries
        entry_prices[all_entries] = close[all_entries]
        
        # Populate exit prices (use close price at signal time)
        all_exits = long_exits | short_exits
        exit_prices[all_exits] = close[all_exits]
        
        # Create comprehensive signals dictionary with proper naming
        self.signals = {
            # Primary signal names (for VectorBT compatibility)
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            
            # Entry and exit prices for validation
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            
            # Combined entry/exit signals for charts (will show both long and short)
            'entries': long_entries | short_entries,  # All entry signals 
            'exits': long_exits | short_exits,        # All exit signals
            
            # Stop levels as percentage values for VectorBT Pro
            'sl_levels': sl_levels,
            'tp_levels': tp_levels,
            
            # Absolute price levels for charting
            'sl_price_levels': sl_price_levels,
            'tp_price_levels': tp_price_levels
        }
        
        # Remove None values to avoid issues
        self.signals = {k: v for k, v in self.signals.items() if v is not None}
        
        # Log signal statistics only if not in quiet mode
        if logger.level != "ERROR":
            total_long_entries = long_entries.sum()
            total_long_exits = long_exits.sum() 
            total_short_entries = short_entries.sum()
            total_short_exits = short_exits.sum()
            sl_count = sl_levels.notna().sum() if sl_levels is not None else 0
            tp_count = tp_levels.notna().sum() if tp_levels is not None else 0
            avg_sl_pct = sl_levels.mean() * 100 if sl_levels is not None and not sl_levels.isna().all() else 0
            avg_tp_pct = tp_levels.mean() * 100 if tp_levels is not None and not tp_levels.isna().all() else 0
            
            logger.info(f"Generated {total_long_entries} long entry, {total_long_exits} long exit, "
                       f"{total_short_entries} short entry, {total_short_exits} short exit signals")
            logger.info(f"Stop levels: {sl_count} SL (avg {avg_sl_pct:.1f}%), {tp_count} TP (avg {avg_tp_pct:.1f}%)")
        
        return self.signals
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.params.copy()
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        try:
            self._validate_parameters()
            return True
        except ValueError:
            return False
    
    def get_stop_loss_take_profit(self) -> Dict[str, Any]:
        """Get stop loss and take profit levels based on ATR."""
        if not self.indicators:
            raise ValueError("Indicators not calculated. Call init_indicators() first.")
            
        close = self.data.get('close')
        atr_values = self.indicators['atr']
        
        # Calculate stop loss and take profit levels
        stop_loss = close - self.params['atr_multiplier_sl'] * atr_values
        take_profit = close + self.params['atr_multiplier_tp'] * atr_values
        
        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
    
    def get_strategy_description(self) -> str:
        """Get a detailed description of the strategy configuration."""
        desc = f"""
        {self.__class__.__name__} Configuration:
        
        Parameters:
        - Fast MA Window: {self.params['fast_window']}
        - Slow MA Window: {self.params['slow_window']}
        - ATR Window: {self.params['atr_window']}
        - Stop Loss: {self.params['atr_multiplier_sl']}x ATR
        - Take Profit: {self.params['atr_multiplier_tp']}x ATR
        - Volume Filter: {'Enabled' if self.params['use_volume_filter'] else 'Disabled'}
        
        Strategy Logic:
        - Enter long when fast MA crosses above slow MA
        - Enter short when fast MA crosses below slow MA
        - Dynamic stops based on ATR volatility
        - Optional volume confirmation for entries
        """
        
        return desc
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"fast={self.params['fast_window']}, "
                f"slow={self.params['slow_window']}, "
                f"atr_mult_sl={self.params['atr_multiplier_sl']})")
