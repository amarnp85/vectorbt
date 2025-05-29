# Strategies Module

This module contains trading strategy implementations for the backtesting system. All strategies follow a consistent interface and focus exclusively on indicator calculation and signal generation.

## 📋 Overview

The strategies module provides:
- Abstract base classes for different strategy types
- Ready-to-use strategy implementations
- Clean examples demonstrating best practices
- Full integration with VectorBTPro for optimal performance

**Important**: Strategies do NOT handle:
- Portfolio simulation (handled by portfolio module)
- Position sizing (handled by portfolio module)
- Risk management (handled by risk_management module)
- Trade execution (handled by portfolio module)

## 🏗️ Architecture

### Base Classes

#### 1. **BaseStrategy** (`base_strategy.py`)
The core abstract base class that all strategies inherit from.

**Key Methods**:
- `init_indicators()` - Calculate technical indicators
- `generate_signals()` - Generate trading signals
- `run_signal_generation()` - Main entry point
- `clean_signals()` - Clean signals using VBT utilities
- `validate_signals()` - Validate signal format

**Usage**:
```python
from backtester.strategies import BaseStrategy

class MyStrategy(BaseStrategy):
    def init_indicators(self):
        # Calculate indicators
        pass
    
    def generate_signals(self):
        # Generate signals
        pass
```

#### 2. **MTFStrategy** (`mtf_strategy_base.py`)
Base class for multi-timeframe strategies.

**Features**:
- Native VectorBTPro multi-timeframe support
- Automatic data alignment
- Cross-timeframe indicators
- Trend alignment scoring

**Usage**:
```python
from backtester.strategies import MTFStrategy

class MyMTFStrategy(MTFStrategy):
    def _calculate_timeframe_indicators(self, data, timeframe):
        # Calculate indicators for specific timeframe
        pass
```

#### 3. **MultiSymbolStrategy** (`multi_symbol_strategy_base.py`)
Base class for strategies that trade multiple symbols simultaneously.

**Features**:
- Cross-symbol correlation analysis
- Relative strength indicators
- Symbol ranking and selection
- Market regime detection

**Usage**:
```python
from backtester.strategies import MultiSymbolStrategy

class MyMultiSymbolStrategy(MultiSymbolStrategy):
    def _calculate_custom_indicators(self):
        # Calculate strategy-specific indicators
        pass
```

## 📊 Available Strategies

### 1. **DMA ATR Trend Strategy** (`dma_atr_trend_strategy.py`)
A trend-following strategy using dual moving averages with ATR-based stops.

**Parameters**:
- `fast_window`: Fast MA period (default: 10)
- `slow_window`: Slow MA period (default: 30)
- `atr_window`: ATR calculation period (default: 14)
- `atr_multiplier_sl`: Stop loss multiplier (default: 2.0)
- `atr_multiplier_tp`: Take profit multiplier (default: 3.0)

**Example**:
```python
from backtester.strategies import DMAATRTrendStrategy

strategy = DMAATRTrendStrategy(data, {
    'fast_window': 10,
    'slow_window': 30,
    'atr_multiplier_sl': 2.0
})
result = strategy.run_signal_generation()
```

### 2. **Mean Reversion Strategy** (`mean_reversion_strategy.py`)
Trades on the assumption that prices revert to their mean using Bollinger Bands and RSI.

**Parameters**:
- `bb_period`: Bollinger Bands period (default: 20)
- `bb_std`: Number of standard deviations (default: 2)
- `rsi_period`: RSI period (default: 14)
- `rsi_oversold`: Oversold threshold (default: 30)
- `rsi_overbought`: Overbought threshold (default: 70)

**Example**:
```python
from backtester.strategies import MeanReversionStrategy

strategy = MeanReversionStrategy(data, {
    'bb_period': 20,
    'bb_std': 2,
    'rsi_oversold': 30
})
```

### 3. **Momentum Strategy** (`momentum_strategy.py`)
A trend-following strategy based on multiple momentum indicators.

**Features**:
- ROC, RSI, MACD indicators
- Composite momentum scoring
- ADX trend strength filter
- Market regime detection

**Parameters**:
- `roc_period`: Rate of Change period (default: 20)
- `momentum_threshold`: Entry threshold (default: 0.02)
- `adx_threshold`: Minimum ADX for entry (default: 25)

### 4. **Pairs Trading Strategy** (`pairs_trading_strategy.py`)
Statistical arbitrage strategy for trading cointegrated pairs.

**Features**:
- Automatic pair selection via cointegration testing
- Z-score based entry/exit
- OLS-based spread calculation
- Support for multiple pairs

**Parameters**:
- `window`: Lookback for z-score (default: 30)
- `entry_z`: Entry z-score threshold (default: 2.0)
- `exit_z`: Exit z-score threshold (default: 0.0)

**Example**:
```python
from backtester.strategies import PairsTradingStrategy

# Auto-select best pair
strategy = PairsTradingStrategy(multi_symbol_data, {
    'window': 30,
    'entry_z': 2.0
})

# Or specify pair
strategy = PairsTradingStrategy(data, params, pair=('AAPL', 'MSFT'))
```

### 5. **MTF DMA ATR Strategy** (`mtf_dma_atr_strategy.py`)
Multi-timeframe version of the DMA ATR strategy.

**Features**:
- Multiple timeframe confirmation
- Trend alignment across timeframes
- Momentum confluence
- Higher timeframe trend validation

**Example**:
```python
from backtester.strategies import MTF_DMA_ATR_Strategy

# With multiple timeframe data
strategy = MTF_DMA_ATR_Strategy(
    mtf_data,  # Dict of timeframe -> vbt.Data
    params,
    base_timeframe="1h"
)
```

### 6. **Simple MA Crossover Example** (`simple_ma_crossover_example.py`)
A clean example demonstrating VectorBTPro best practices.

**Purpose**:
- Reference implementation
- Shows proper VBT usage
- Multi-symbol support
- Parameter optimization ready

**Example**:
```python
from backtester.strategies import SimpleMAStrategy

# Single symbol
strategy = SimpleMAStrategy(data, {
    'fast_window': 10,
    'slow_window': 30,
    'ma_type': 'EMA'
})

# Parameter optimization
params = {
    'fast_window': vbt.Param([5, 10, 15, 20]),
    'slow_window': vbt.Param([20, 30, 40, 50])
}
strategy = SimpleMAStrategy(data, params)
```

## 🚀 Quick Start

### Basic Usage

```python
import vectorbtpro as vbt
from backtester.strategies import MomentumStrategy

# 1. Load data
data = vbt.YFData.fetch("AAPL", start="2023-01-01", end="2023-12-31")

# 2. Create strategy
strategy = MomentumStrategy(data, {
    'roc_period': 20,
    'momentum_threshold': 0.02
})

# 3. Generate signals
result = strategy.run_signal_generation()

# 4. Access components
signals = result['signals']
indicators = result['indicators']
metadata = result['metadata']
```

### Multi-Symbol Usage

```python
# Load multiple symbols
symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
data = vbt.YFData.fetch(symbols, start="2023-01-01", end="2023-12-31")

# Same strategy code works!
strategy = DMAATRTrendStrategy(data, params)
result = strategy.run_signal_generation()

# Signals are DataFrames with columns for each symbol
long_entries = result['signals']['long_entries']  # DataFrame
```

### Parameter Optimization

```python
# Define parameter ranges
params = {
    'fast_window': vbt.Param([5, 10, 15, 20]),
    'slow_window': vbt.Param([20, 30, 40, 50]),
    'atr_multiplier_sl': vbt.Param([1.5, 2.0, 2.5])
}

# Create strategy with parameter ranges
strategy = DMAATRTrendStrategy(data, params)
result = strategy.run_signal_generation()

# Result contains signals for all parameter combinations
```

## 📝 Creating New Strategies

See [STRATEGY_TEMPLATE.md](STRATEGY_TEMPLATE.md) for a detailed guide on creating new strategies.

### Key Points:
1. Inherit from appropriate base class
2. Implement `init_indicators()` and `generate_signals()`
3. Use VectorBTPro's native functions
4. Return standardized signal format
5. No portfolio logic in strategies

### Signal Format

All strategies must return signals in this format:
```python
{
    'long_entries': pd.Series/DataFrame,   # Boolean
    'long_exits': pd.Series/DataFrame,     # Boolean
    'short_entries': pd.Series/DataFrame,  # Boolean
    'short_exits': pd.Series/DataFrame,   # Boolean
    'sl_levels': pd.Series/DataFrame,      # Numeric (optional)
    'tp_levels': pd.Series/DataFrame       # Numeric (optional)
}
```

## 🧪 Testing Strategies

```python
# Test single symbol
data = vbt.YFData.fetch("SPY", period="1y")
strategy = YourStrategy(data, params)
result = strategy.run_signal_generation()

# Validate signals
is_valid = strategy.validate_signals()

# Check signal statistics
long_entries = result['signals']['long_entries']
print(f"Total long signals: {long_entries.sum()}")

# Test multi-symbol
multi_data = vbt.YFData.fetch(["SPY", "QQQ"], period="1y")
strategy = YourStrategy(multi_data, params)
result = strategy.run_signal_generation()
```

## 🔧 Best Practices

1. **Use VectorBTPro Native Functions**
   ```python
   # Good
   self.indicators['sma'] = vbt.talib("SMA").run(close, timeperiod=20).real
   
   # Bad
   self.indicators['sma'] = close.rolling(20).mean()
   ```

2. **Let VBT Handle Broadcasting**
   ```python
   # Works for both single and multi-symbol
   long_entries = (fast_ma > slow_ma) & (rsi < 70)
   ```

3. **Clean Signals**
   ```python
   # Always clean entry/exit signals
   long_entries, long_exits = long_entries.vbt.signals.clean(long_exits)
   ```

4. **No Portfolio Logic**
   ```python
   # Bad - Don't do this in strategies
   def calculate_position_size(self, capital):
       return capital * 0.1
   
   # Good - Only return signals
   def generate_signals(self):
       return {'long_entries': entries, 'long_exits': exits}
   ```

## 📚 Additional Resources

- [STRATEGY_TEMPLATE.md](STRATEGY_TEMPLATE.md) - Detailed template for new strategies
- VectorBTPro documentation for indicator references
- Portfolio module documentation for backtesting signals
- Risk management module for position sizing

## 🤝 Contributing

When adding new strategies:
1. Follow the established patterns
2. Include comprehensive docstrings
3. Add parameter validation
4. Test with single and multi-symbol data
5. Update this README with the new strategy 