# Strategy Development Template

This template provides the standard structure for developing trading strategies in the backtesting system. Follow this pattern to ensure consistency and compatibility with the portfolio simulation infrastructure.

## Key Principles

1. **Separation of Concerns**: Strategies focus ONLY on:
   - Calculating indicators
   - Generating signals
   - Providing metadata

2. **VectorBTPro Native**: Use VBT's built-in functions for:
   - Indicator calculation
   - Signal cleaning
   - Multi-symbol broadcasting
   - Parameter optimization

3. **No Portfolio Logic**: The following should NOT be in strategies:
   - Portfolio simulation
   - Position sizing
   - Risk management components
   - Trade execution

## Basic Strategy Template

```python
"""
Strategy Name

Brief description of the strategy logic and approach.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional

from .base_strategy import BaseStrategy


class YourStrategy(BaseStrategy):
    """
    Detailed strategy description.
    
    Features:
    - List key features
    - Indicators used
    - Signal generation logic
    
    Note: Position sizing and risk management are handled at the portfolio level.
    """
    
    def __init__(self, data: vbt.Data, params: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            data: VectorBTPro Data object with OHLCV data
            params: Strategy parameters
        """
        # Set default parameters
        default_params = {
            'param1': 20,
            'param2': 50,
            'param3': 2.0
        }
        
        # Merge with provided parameters
        params = {**default_params, **params}
        
        # Initialize base class
        super().__init__(data, params)
    
    def init_indicators(self) -> Dict[str, Any]:
        """
        Calculate indicators using VectorBTPro.
        
        Returns:
            Dictionary of calculated indicators
        """
        # Get price data - works with single and multi-symbol
        close = self.data.close
        high = self.data.high
        low = self.data.low
        volume = self.data.volume
        
        # Calculate indicators using VBT's native functions
        # VBT handles multi-symbol broadcasting automatically
        
        # Example: Moving averages
        self.indicators['sma_fast'] = vbt.talib("SMA").run(
            close, timeperiod=self.params['param1']
        ).real
        
        self.indicators['sma_slow'] = vbt.talib("SMA").run(
            close, timeperiod=self.params['param2']
        ).real
        
        # Example: RSI
        self.indicators['rsi'] = vbt.talib("RSI").run(
            close, timeperiod=14
        ).real
        
        # Example: ATR for stop levels
        self.indicators['atr'] = vbt.talib("ATR").run(
            high, low, close, timeperiod=14
        ).real
        
        # Custom calculations
        self.indicators['custom_metric'] = self._calculate_custom_metric()
        
        return self.indicators
    
    def _calculate_custom_metric(self) -> pd.Series:
        """
        Calculate custom indicator or metric.
        
        Returns:
            Series or DataFrame with custom values
        """
        close = self.data.close
        
        # Your custom logic here
        # Remember: VBT handles multi-symbol automatically
        custom_value = close.pct_change().rolling(20).std()
        
        return custom_value
    
    def generate_signals(self) -> Dict[str, Any]:
        """
        Generate trading signals from indicators.
        
        Returns:
            Dictionary with signal arrays
        """
        # Extract indicators
        fast_ma = self.indicators['sma_fast']
        slow_ma = self.indicators['sma_slow']
        rsi = self.indicators['rsi']
        atr = self.indicators['atr']
        
        # Generate entry signals
        long_entries = (
            (fast_ma > slow_ma) & 
            (fast_ma.shift(1) <= slow_ma.shift(1)) &
            (rsi < 70)
        )
        
        # Generate exit signals
        long_exits = (
            (fast_ma < slow_ma) & 
            (fast_ma.shift(1) >= slow_ma.shift(1))
        )
        
        # Clean signals using VBT utilities
        if hasattr(long_entries, 'vbt'):
            long_entries, long_exits = long_entries.vbt.signals.clean(
                long_exits,
                conflict_mode='sequential'
            )
        
        # Short signals (if applicable)
        short_entries = (
            (fast_ma < slow_ma) & 
            (fast_ma.shift(1) >= slow_ma.shift(1)) &
            (rsi > 30)
        )
        
        short_exits = (
            (fast_ma > slow_ma) & 
            (fast_ma.shift(1) <= slow_ma.shift(1))
        )
        
        if hasattr(short_entries, 'vbt'):
            short_entries, short_exits = short_entries.vbt.signals.clean(
                short_exits,
                conflict_mode='sequential'
            )
        
        # Calculate stop levels (optional)
        close = self.data.close
        sl_levels = pd.Series(np.nan, index=close.index)
        tp_levels = pd.Series(np.nan, index=close.index)
        
        # Set stop levels at entry points
        if isinstance(long_entries, pd.DataFrame):
            # Multi-symbol case
            for col in long_entries.columns:
                long_mask = long_entries[col]
                sl_levels.loc[long_mask, col] = (
                    close.loc[long_mask, col] - 
                    self.params['param3'] * atr.loc[long_mask, col]
                )
                tp_levels.loc[long_mask, col] = (
                    close.loc[long_mask, col] + 
                    self.params['param3'] * 2 * atr.loc[long_mask, col]
                )
        else:
            # Single symbol case
            sl_levels[long_entries] = (
                close[long_entries] - 
                self.params['param3'] * atr[long_entries]
            )
            tp_levels[long_entries] = (
                close[long_entries] + 
                self.params['param3'] * 2 * atr[long_entries]
            )
        
        # Store signals
        self.signals = {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'sl_levels': sl_levels if not sl_levels.isna().all() else None,
            'tp_levels': tp_levels if not tp_levels.isna().all() else None
        }
        
        return self.signals
```

## Multi-Symbol Strategy Example

```python
class MultiSymbolStrategy(BaseStrategy):
    """
    Example showing VBT's native multi-symbol support.
    """
    
    def init_indicators(self) -> Dict[str, Any]:
        """Calculate indicators for all symbols at once."""
        # VBT handles multiple symbols natively
        close = self.data.close  # DataFrame with columns for each symbol
        
        # All calculations work on all symbols simultaneously
        self.indicators['returns'] = close.pct_change()
        self.indicators['volatility'] = self.indicators['returns'].rolling(20).std()
        
        # Cross-symbol calculations
        self.indicators['relative_strength'] = close / close.mean(axis=1).values[:, None]
        self.indicators['correlation_score'] = self._calculate_correlation_score()
        
        return self.indicators
    
    def _calculate_correlation_score(self) -> pd.DataFrame:
        """Calculate rolling correlation between symbols."""
        returns = self.indicators['returns']
        
        # Rolling correlation matrix
        rolling_corr = returns.rolling(30).corr()
        
        # Extract average correlation for each symbol
        corr_score = pd.DataFrame(index=returns.index, columns=returns.columns)
        
        for symbol in returns.columns:
            # Average correlation with other symbols
            symbol_corr = rolling_corr.loc[:, symbol, :].mean(axis=1)
            corr_score[symbol] = symbol_corr
        
        return corr_score
```

## Parameter Optimization Support

```python
class OptimizableStrategy(BaseStrategy):
    """
    Example with built-in parameter optimization support.
    """
    
    def __init__(self, data: vbt.Data, params: Dict[str, Any]):
        # Support both single values and vbt.Param objects
        default_params = {
            'window': 20,  # Can be vbt.Param([10, 20, 30])
            'threshold': 2.0  # Can be vbt.Param([1.5, 2.0, 2.5])
        }
        params = {**default_params, **params}
        super().__init__(data, params)
    
    def init_indicators(self) -> Dict[str, Any]:
        """Indicators work with parameter combinations."""
        close = self.data.close
        window = self.params['window']  # Works with vbt.Param
        
        # VBT handles parameter combinations automatically
        self.indicators['sma'] = vbt.talib("SMA").run(
            close, timeperiod=window
        ).real
        
        return self.indicators

# Usage:
# params = {
#     'window': vbt.Param([10, 20, 30, 40]),
#     'threshold': vbt.Param(np.arange(1.0, 3.0, 0.5))
# }
# strategy = OptimizableStrategy(data, params)
# results = strategy.run_signal_generation()
```

## Best Practices

### 1. Use VectorBTPro's Native Functions
```python
# ✅ GOOD: Use VBT's indicators
self.indicators['rsi'] = vbt.talib("RSI").run(close, timeperiod=14).real

# ❌ BAD: Manual calculations when VBT has built-in
rsi = self.calculate_rsi_manually(close, 14)
```

### 2. Let VBT Handle Broadcasting
```python
# ✅ GOOD: Write once, works for single and multi-symbol
long_entries = (fast_ma > slow_ma) & (rsi < 70)

# ❌ BAD: Manual symbol iteration
for symbol in symbols:
    long_entries[symbol] = (fast_ma[symbol] > slow_ma[symbol])
```

### 3. Use Signal Cleaning
```python
# ✅ GOOD: Clean signals to avoid conflicts
long_entries, long_exits = long_entries.vbt.signals.clean(
    long_exits, conflict_mode='sequential'
)

# ❌ BAD: Raw signals without cleaning
return {'long_entries': long_entries, 'long_exits': long_exits}
```

### 4. Proper Signal Format
```python
# ✅ GOOD: Complete signal dictionary
self.signals = {
    'long_entries': long_entries,
    'long_exits': long_exits,
    'short_entries': short_entries,
    'short_exits': short_exits,
    'sl_levels': sl_levels,  # Optional
    'tp_levels': tp_levels   # Optional
}

# ❌ BAD: Incomplete or wrong format
self.signals = {
    'signals': combined_signals,  # Wrong structure
    'positions': positions  # Portfolio concern
}
```

### 5. No Portfolio Logic
```python
# ❌ BAD: Portfolio simulation in strategy
def backtest(self):
    portfolio = vbt.Portfolio.from_signals(...)
    return portfolio

# ❌ BAD: Position sizing in strategy
def calculate_position_size(self, capital):
    return capital * 0.1

# ✅ GOOD: Only signals and metadata
def generate_signals(self):
    return {'long_entries': entries, 'long_exits': exits}
```

## Testing Your Strategy

```python
# 1. Test with single symbol
data = vbt.YFData.fetch("AAPL", start="2023-01-01", end="2023-12-31")
strategy = YourStrategy(data, params)
result = strategy.run_signal_generation()

# 2. Test with multiple symbols
data = vbt.YFData.fetch(["AAPL", "MSFT"], start="2023-01-01", end="2023-12-31")
strategy = YourStrategy(data, params)
result = strategy.run_signal_generation()

# 3. Test with parameter optimization
params = {
    'param1': vbt.Param([10, 20, 30]),
    'param2': vbt.Param([40, 50, 60])
}
strategy = YourStrategy(data, params)
result = strategy.run_signal_generation()

# 4. Verify signal format
assert 'long_entries' in result['signals']
assert 'long_exits' in result['signals']
assert 'short_entries' in result['signals']
assert 'short_exits' in result['signals']
```

## Common Patterns

### 1. Trend Following
```python
# Moving average crossover
long_entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
long_exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
```

### 2. Mean Reversion
```python
# Bollinger Bands
long_entries = (close < lower_band) & (rsi < 30)
long_exits = (close > middle_band) | (rsi > 70)
```

### 3. Momentum
```python
# Momentum with filters
momentum = close.pct_change(20)
long_entries = (momentum > threshold) & (adx > 25) & (volume > volume_ma)
```

### 4. Multi-Timeframe
```python
# Use VBT's resampling
daily_data = self.data
weekly_data = daily_data.resample('W')

daily_rsi = vbt.talib("RSI").run(daily_data.close).real
weekly_trend = vbt.talib("SMA").run(weekly_data.close, timeperiod=10).real

# Align to daily timeframe
weekly_trend_daily = weekly_trend.reindex(daily_rsi.index, method='ffill')
```

Remember: Focus on the strategy logic. Let the portfolio module handle execution, sizing, and risk management. 