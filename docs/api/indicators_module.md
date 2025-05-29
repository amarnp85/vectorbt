# Indicators Module API Documentation

## Overview

The indicators module provides a comprehensive set of technical indicators optimized for vectorized operations using pandas and numpy. All indicators are designed to work seamlessly with vectorbtpro.

## Core Indicators

### Moving Averages

#### `sma(data, window, min_periods=None)`
Simple Moving Average

```python
from backtester.indicators import sma

# Calculate 20-period SMA
sma_20 = sma(close_prices, window=20)

# With minimum periods requirement
sma_20 = sma(close_prices, window=20, min_periods=10)
```

**Parameters:**
- `data` (pd.Series): Input price series
- `window` (int): Lookback period
- `min_periods` (int, optional): Minimum periods required

**Returns:**
- `pd.Series`: Simple moving average values

#### `ema(data, window, adjust=True)`
Exponential Moving Average

```python
from backtester.indicators import ema

# Calculate 12-period EMA
ema_12 = ema(close_prices, window=12)

# Without bias adjustment
ema_12 = ema(close_prices, window=12, adjust=False)
```

**Parameters:**
- `data` (pd.Series): Input price series
- `window` (int): Lookback period
- `adjust` (bool): Use bias adjustment (default: True)

**Returns:**
- `pd.Series`: Exponential moving average values

#### `wma(data, window)`
Weighted Moving Average

```python
from backtester.indicators import wma

# Calculate 10-period WMA
wma_10 = wma(close_prices, window=10)
```

**Parameters:**
- `data` (pd.Series): Input price series
- `window` (int): Lookback period

**Returns:**
- `pd.Series`: Weighted moving average values

### Volatility Indicators

#### `atr(high, low, close, period=14)`
Average True Range

```python
from backtester.indicators import atr

# Calculate 14-period ATR
atr_values = atr(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    period=14
)
```

**Parameters:**
- `high` (pd.Series): High prices
- `low` (pd.Series): Low prices
- `close` (pd.Series): Close prices
- `period` (int): ATR period (default: 14)

**Returns:**
- `pd.Series`: ATR values

#### `bollinger_bands(data, window=20, num_std=2)`
Bollinger Bands

```python
from backtester.indicators import bollinger_bands

# Calculate Bollinger Bands
upper, middle, lower = bollinger_bands(
    close_prices,
    window=20,
    num_std=2
)
```

**Parameters:**
- `data` (pd.Series): Input price series
- `window` (int): MA period (default: 20)
- `num_std` (float): Standard deviations (default: 2)

**Returns:**
- `tuple`: (upper_band, middle_band, lower_band)

#### `keltner_channels(high, low, close, ema_period=20, atr_period=10, multiplier=2)`
Keltner Channels

```python
from backtester.indicators import keltner_channels

# Calculate Keltner Channels
upper, middle, lower = keltner_channels(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    ema_period=20,
    atr_period=10,
    multiplier=2
)
```

**Parameters:**
- `high`, `low`, `close` (pd.Series): OHLC data
- `ema_period` (int): EMA period (default: 20)
- `atr_period` (int): ATR period (default: 10)
- `multiplier` (float): ATR multiplier (default: 2)

**Returns:**
- `tuple`: (upper_channel, middle_line, lower_channel)

### Momentum Indicators

#### `rsi(data, period=14)`
Relative Strength Index

```python
from backtester.indicators import rsi

# Calculate 14-period RSI
rsi_values = rsi(close_prices, period=14)
```

**Parameters:**
- `data` (pd.Series): Input price series
- `period` (int): RSI period (default: 14)

**Returns:**
- `pd.Series`: RSI values (0-100)

#### `macd(data, fast_period=12, slow_period=26, signal_period=9)`
MACD (Moving Average Convergence Divergence)

```python
from backtester.indicators import macd

# Calculate MACD
macd_line, signal_line, histogram = macd(
    close_prices,
    fast_period=12,
    slow_period=26,
    signal_period=9
)
```

**Parameters:**
- `data` (pd.Series): Input price series
- `fast_period` (int): Fast EMA period (default: 12)
- `slow_period` (int): Slow EMA period (default: 26)
- `signal_period` (int): Signal EMA period (default: 9)

**Returns:**
- `tuple`: (macd_line, signal_line, histogram)

#### `stochastic(high, low, close, k_period=14, d_period=3)`
Stochastic Oscillator

```python
from backtester.indicators import stochastic

# Calculate Stochastic
k_values, d_values = stochastic(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    k_period=14,
    d_period=3
)
```

**Parameters:**
- `high`, `low`, `close` (pd.Series): OHLC data
- `k_period` (int): %K period (default: 14)
- `d_period` (int): %D smoothing (default: 3)

**Returns:**
- `tuple`: (k_values, d_values)

#### `momentum(data, period=10)`
Momentum Indicator

```python
from backtester.indicators import momentum

# Calculate 10-period momentum
mom_values = momentum(close_prices, period=10)
```

**Parameters:**
- `data` (pd.Series): Input price series
- `period` (int): Lookback period (default: 10)

**Returns:**
- `pd.Series`: Momentum values

### Trend Indicators

#### `adx(high, low, close, period=14)`
Average Directional Index

```python
from backtester.indicators import adx

# Calculate ADX with directional indicators
adx_values, plus_di, minus_di = adx(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    period=14
)
```

**Parameters:**
- `high`, `low`, `close` (pd.Series): OHLC data
- `period` (int): ADX period (default: 14)

**Returns:**
- `tuple`: (adx, plus_di, minus_di)

#### `cci(high, low, close, period=20)`
Commodity Channel Index

```python
from backtester.indicators import cci

# Calculate CCI
cci_values = cci(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    period=20
)
```

**Parameters:**
- `high`, `low`, `close` (pd.Series): OHLC data
- `period` (int): CCI period (default: 20)

**Returns:**
- `pd.Series`: CCI values

#### `aroon(high, low, period=25)`
Aroon Indicator

```python
from backtester.indicators import aroon

# Calculate Aroon
aroon_up, aroon_down = aroon(
    high=df['high'],
    low=df['low'],
    period=25
)
```

**Parameters:**
- `high`, `low` (pd.Series): High and low prices
- `period` (int): Lookback period (default: 25)

**Returns:**
- `tuple`: (aroon_up, aroon_down)

### Volume Indicators

#### `obv(close, volume)`
On Balance Volume

```python
from backtester.indicators import obv

# Calculate OBV
obv_values = obv(
    close=df['close'],
    volume=df['volume']
)
```

**Parameters:**
- `close` (pd.Series): Close prices
- `volume` (pd.Series): Volume data

**Returns:**
- `pd.Series`: OBV values

#### `vwap(high, low, close, volume)`
Volume Weighted Average Price

```python
from backtester.indicators import vwap

# Calculate VWAP
vwap_values = vwap(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    volume=df['volume']
)
```

**Parameters:**
- `high`, `low`, `close`, `volume` (pd.Series): OHLCV data

**Returns:**
- `pd.Series`: VWAP values

#### `mfi(high, low, close, volume, period=14)`
Money Flow Index

```python
from backtester.indicators import mfi

# Calculate MFI
mfi_values = mfi(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    volume=df['volume'],
    period=14
)
```

**Parameters:**
- `high`, `low`, `close`, `volume` (pd.Series): OHLCV data
- `period` (int): MFI period (default: 14)

**Returns:**
- `pd.Series`: MFI values (0-100)

### Support/Resistance Indicators

#### `pivot_points(high, low, close)`
Pivot Points

```python
from backtester.indicators import pivot_points

# Calculate pivot points
pp, r1, r2, r3, s1, s2, s3 = pivot_points(
    high=df['high'],
    low=df['low'],
    close=df['close']
)
```

**Parameters:**
- `high`, `low`, `close` (pd.Series): OHLC data

**Returns:**
- `tuple`: (pivot, resistance1, resistance2, resistance3, support1, support2, support3)

#### `fibonacci_retracements(high, low)`
Fibonacci Retracement Levels

```python
from backtester.indicators import fibonacci_retracements

# Calculate Fibonacci levels
levels = fibonacci_retracements(
    high=df['high'].max(),
    low=df['low'].min()
)
```

**Parameters:**
- `high` (float): High price
- `low` (float): Low price

**Returns:**
- `dict`: Fibonacci levels (0%, 23.6%, 38.2%, 50%, 61.8%, 100%)

## Multi-Timeframe Indicators

### `mtf_indicator(data, indicator_func, timeframes, **kwargs)`
Calculate indicator across multiple timeframes

```python
from backtester.indicators import mtf_indicator, rsi

# Calculate RSI on multiple timeframes
mtf_rsi = mtf_indicator(
    data=close_prices,
    indicator_func=rsi,
    timeframes=['1h', '4h', '1d'],
    period=14
)
```

**Parameters:**
- `data` (pd.Series): Input data
- `indicator_func` (callable): Indicator function
- `timeframes` (list): Target timeframes
- `**kwargs`: Arguments for indicator function

**Returns:**
- `dict`: {timeframe: indicator_values}

## Custom Indicators

### Creating Custom Indicators

```python
from backtester.indicators.base import BaseIndicator
import pandas as pd
import numpy as np

class CustomIndicator(BaseIndicator):
    def __init__(self, period=20):
        self.period = period
        
    def calculate(self, data):
        # Your custom calculation
        result = data.rolling(self.period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        return result
        
# Usage
custom_ind = CustomIndicator(period=30)
values = custom_ind.calculate(close_prices)
```

## Performance Optimization

### Vectorized Calculations
All indicators use vectorized operations for performance:

```python
# Efficient RSI calculation
def rsi_vectorized(data, period=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))
```

### Caching Results
For repeated calculations:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_indicator(data_hash, period):
    # Expensive calculation
    return result
```

## Integration with VectorBT

### Using with VectorBT Indicators
```python
import vectorbtpro as vbt

# Use VBT's built-in indicators
close_prices = data.get('Close')
rsi = vbt.RSI.run(close_prices, window=14)
bb = vbt.BB.run(close_prices, window=20, alpha=2)

# Combine with custom indicators
from backtester.indicators import keltner_channels
kc_upper, kc_middle, kc_lower = keltner_channels(
    data.get('High'),
    data.get('Low'),
    data.get('Close')
)
```

## Best Practices

1. **Handle NaN Values**: Always handle initial NaN values appropriately
2. **Parameter Validation**: Validate input parameters
3. **Memory Efficiency**: Use views instead of copies when possible
4. **Type Consistency**: Ensure consistent return types
5. **Documentation**: Document expected inputs and outputs

## Common Patterns

### Indicator Combinations
```python
# Trend confirmation
sma_50 = sma(close, 50)
sma_200 = sma(close, 200)
trend_up = sma_50 > sma_200

# Momentum confirmation
rsi_14 = rsi(close, 14)
macd_line, signal, _ = macd(close)
momentum_up = (rsi_14 > 50) & (macd_line > signal)

# Combined signal
buy_signal = trend_up & momentum_up
```

### Adaptive Indicators
```python
# Adaptive moving average based on volatility
def adaptive_ma(data, base_period=20):
    volatility = data.rolling(base_period).std()
    adaptive_period = base_period * (1 + volatility / volatility.mean())
    return data.rolling(int(adaptive_period.mean())).mean()
```

## Error Handling

All indicators include error handling:

```python
try:
    result = atr(high, low, close, period=14)
except ValueError as e:
    logger.error(f"ATR calculation failed: {e}")
    result = pd.Series(index=close.index)  # Return empty series
``` 