# Strategy Module API Documentation

## Overview

The strategy module provides a flexible framework for implementing trading strategies. It includes base classes, concrete implementations, and utilities for signal generation.

## Base Classes

### `BaseStrategy`

The abstract base class that all strategies must inherit from.

```python
from backtester.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Implementation required
        pass
```

**Required Methods:**
- `generate_signals(data: pd.DataFrame) -> pd.Series`: Generate buy/sell signals
  - Input: DataFrame with OHLCV data
  - Output: Series with values: 1 (buy), -1 (sell), 0 (hold)

**Optional Methods:**
- `validate_parameters() -> bool`: Validate strategy parameters
- `get_required_indicators() -> List[str]`: List required indicators
- `calculate_position_size() -> float`: Custom position sizing

### `MTFStrategy`

Base class for multi-timeframe strategies.

```python
from backtester.strategies.mtf_strategy import MTFStrategy

class MyMTFStrategy(MTFStrategy):
    def generate_mtf_signals(self, mtf_data: Dict[str, pd.DataFrame]) -> pd.Series:
        # Implementation required
        pass
```

**Additional Features:**
- Automatic timeframe alignment
- Cross-timeframe signal confirmation
- MTF-specific indicator calculations

## Concrete Strategies

### `DMAATRTrendStrategy`

Dual Moving Average with ATR-based risk management and trend filtering.

```python
from backtester.strategies import DMAATRTrendStrategy

strategy = DMAATRTrendStrategy(
    short_ma_window=20,
    long_ma_window=50,
    atr_period=14,
    sl_atr_multiplier=2.0,
    tp_atr_multiplier=4.0,
    trend_ma_window=200,
    use_trend_filter=True
)

signals = strategy.generate_signals(data)
```

**Parameters:**
- `short_ma_window` (int): Fast moving average period (default: 20)
- `long_ma_window` (int): Slow moving average period (default: 50)
- `atr_period` (int): ATR calculation period (default: 14)
- `sl_atr_multiplier` (float): Stop-loss distance in ATR units (default: 2.0)
- `tp_atr_multiplier` (float): Take-profit distance in ATR units (default: 4.0)
- `trend_ma_window` (int): Trend filter MA period (default: 200)
- `use_trend_filter` (bool): Enable trend filtering (default: True)

**Signal Logic:**
- Long: Short MA > Long MA AND price > Trend MA
- Short: Short MA < Long MA AND price < Trend MA
- Exit: Stop-loss or take-profit hit

### `MeanReversionStrategy`

Bollinger Bands and RSI based mean reversion with regime filtering.

```python
from backtester.strategies import MeanReversionStrategy

strategy = MeanReversionStrategy(
    bb_period=20,
    bb_std=2.0,
    rsi_period=14,
    rsi_oversold=30,
    rsi_overbought=70,
    regime_filter=True,
    position_sizer='volatility'
)
```

**Parameters:**
- `bb_period` (int): Bollinger Bands period (default: 20)
- `bb_std` (float): Standard deviation multiplier (default: 2.0)
- `rsi_period` (int): RSI calculation period (default: 14)
- `rsi_oversold` (float): Oversold threshold (default: 30)
- `rsi_overbought` (float): Overbought threshold (default: 70)
- `regime_filter` (bool): Use market regime filtering (default: True)
- `position_sizer` (str): Position sizing method (default: 'volatility')

**Signal Logic:**
- Long: Price < Lower BB AND RSI < oversold AND ranging/bullish regime
- Short: Price > Upper BB AND RSI > overbought AND ranging/bearish regime

### `MomentumStrategy`

Trend-following momentum strategy with regime detection.

```python
from backtester.strategies import MomentumStrategy

strategy = MomentumStrategy(
    fast_ma=10,
    slow_ma=30,
    momentum_period=20,
    volume_ma=20,
    regime_ma=50,
    atr_period=14,
    risk_per_trade=0.02
)
```

**Parameters:**
- `fast_ma` (int): Fast MA period (default: 10)
- `slow_ma` (int): Slow MA period (default: 30)
- `momentum_period` (int): Momentum calculation period (default: 20)
- `volume_ma` (int): Volume MA period (default: 20)
- `regime_ma` (int): Regime detection MA (default: 50)
- `atr_period` (int): ATR period for stops (default: 14)
- `risk_per_trade` (float): Risk per trade (default: 0.02)

**Features:**
- Momentum strength filtering
- Volume confirmation
- Dynamic stop-loss adjustment
- Regime-based position sizing

### `PairsTradingStrategy`

Statistical arbitrage strategy for cointegrated pairs.

```python
from backtester.strategies import PairsTradingStrategy

strategy = PairsTradingStrategy(
    lookback_period=60,
    entry_z_score=2.0,
    exit_z_score=0.5,
    stop_loss_z_score=3.0,
    min_half_life=1,
    max_half_life=30,
    cointegration_pvalue=0.05
)
```

**Parameters:**
- `lookback_period` (int): Period for spread calculation (default: 60)
- `entry_z_score` (float): Z-score threshold for entry (default: 2.0)
- `exit_z_score` (float): Z-score threshold for exit (default: 0.5)
- `stop_loss_z_score` (float): Stop-loss z-score (default: 3.0)
- `min_half_life` (int): Minimum acceptable half-life (default: 1)
- `max_half_life` (int): Maximum acceptable half-life (default: 30)
- `cointegration_pvalue` (float): P-value threshold (default: 0.05)

**Features:**
- Automatic cointegration testing
- Hedge ratio calculation
- Spread mean reversion
- Half-life based position sizing

### `MTF_DMA_ATR_Strategy`

Multi-timeframe implementation of DMA-ATR strategy.

```python
from backtester.strategies import MTF_DMA_ATR_Strategy

strategy = MTF_DMA_ATR_Strategy(
    timeframes=['15m', '1h', '4h'],
    short_ma_window=20,
    long_ma_window=50,
    atr_period=14,
    confirmation_timeframes=2,
    use_volume_filter=True
)
```

**Additional Parameters:**
- `timeframes` (List[str]): Timeframes to analyze
- `confirmation_timeframes` (int): Required confirmations (default: 2)
- `use_volume_filter` (bool): Volume confirmation (default: True)

## Strategy Development

### Creating Custom Strategies

```python
from backtester.strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np

class CustomStrategy(BaseStrategy):
    def __init__(self, param1=10, param2=20):
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # Calculate indicators
        indicator1 = data['close'].rolling(self.param1).mean()
        indicator2 = data['close'].rolling(self.param2).mean()
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        signals[indicator1 > indicator2] = 1  # Buy
        signals[indicator1 < indicator2] = -1  # Sell
        
        return signals
    
    def validate_parameters(self) -> bool:
        return self.param1 > 0 and self.param2 > self.param1
```

### Strategy Configuration

Strategies can be configured via JSON files:

```json
{
    "strategy_name": "DMAATRTrendStrategy",
    "parameters": {
        "short_ma_window": 20,
        "long_ma_window": 50,
        "atr_period": 14,
        "sl_atr_multiplier": 2.0,
        "tp_atr_multiplier": 4.0,
        "trend_ma_window": 200,
        "use_trend_filter": true
    },
    "risk_management": {
        "position_size": 0.1,
        "max_positions": 1,
        "use_stops": true
    }
}
```

## Integration with Risk Management

Strategies can integrate with risk management modules:

```python
from backtester.strategies import DMAATRTrendStrategy
from backtester.risk_management import VolatilityPositionSizer

strategy = DMAATRTrendStrategy()
position_sizer = VolatilityPositionSizer(
    target_volatility=0.15,
    lookback_period=20
)

# In backtesting loop
signals = strategy.generate_signals(data)
position_sizes = position_sizer.calculate_position_size(
    data, 
    signals, 
    current_portfolio_value
)
```

## Performance Considerations

### Vectorized Operations
All built-in strategies use vectorized pandas/numpy operations for performance:

```python
# Good - Vectorized
signals = (short_ma > long_ma).astype(int) - (short_ma < long_ma).astype(int)

# Bad - Loop
signals = pd.Series(0, index=data.index)
for i in range(len(data)):
    if short_ma[i] > long_ma[i]:
        signals[i] = 1
```

### Indicator Caching
Strategies cache calculated indicators to avoid redundant computations:

```python
class OptimizedStrategy(BaseStrategy):
    def __init__(self):
        super().__init__()
        self._indicator_cache = {}
        
    def _get_or_calculate_indicator(self, name, calculation_func):
        if name not in self._indicator_cache:
            self._indicator_cache[name] = calculation_func()
        return self._indicator_cache[name]
```

## Testing Strategies

### Unit Testing
```python
import pytest
from backtester.strategies import DMAATRTrendStrategy

def test_strategy_signals():
    strategy = DMAATRTrendStrategy()
    
    # Create test data
    test_data = pd.DataFrame({
        'close': [100, 102, 101, 103, 105, 104, 106],
        'high': [101, 103, 102, 104, 106, 105, 107],
        'low': [99, 101, 100, 102, 104, 103, 105],
        'volume': [1000] * 7
    })
    
    signals = strategy.generate_signals(test_data)
    
    assert isinstance(signals, pd.Series)
    assert len(signals) == len(test_data)
    assert all(s in [-1, 0, 1] for s in signals)
```

### Backtesting
```python
from backtester.portfolio import PortfolioManager
from backtester.strategies import MomentumStrategy

strategy = MomentumStrategy()
portfolio = PortfolioManager(initial_cash=100000)

# Run backtest
results = portfolio.run_backtest(
    data=price_data,
    strategy=strategy,
    commission=0.001
)
```

## Best Practices

1. **Parameter Validation**: Always validate parameters in `__init__`
2. **Signal Quality**: Ensure signals are -1, 0, or 1 only
3. **Null Handling**: Handle NaN values in indicators gracefully
4. **Documentation**: Document signal logic and parameter effects
5. **Testing**: Test with various market conditions
6. **Performance**: Use vectorized operations whenever possible

## Common Patterns

### Signal Smoothing
```python
# Reduce signal noise
signals = signals.rolling(3).mean().round()
```

### Signal Confirmation
```python
# Require multiple conditions
condition1 = short_ma > long_ma
condition2 = rsi < 70
condition3 = volume > volume_ma

signals[condition1 & condition2 & condition3] = 1
```

### Dynamic Parameters
```python
# Adjust parameters based on market conditions
if volatility > high_volatility_threshold:
    self.sl_atr_multiplier = 3.0  # Wider stops in volatile markets
else:
    self.sl_atr_multiplier = 2.0
``` 