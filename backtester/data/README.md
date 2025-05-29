# Data Module

The data module provides a simple, unified interface for fetching cryptocurrency OHLCV data with automatic caching, resampling, and exchange management.

## Quick Start

```python
from backtester.data import fetch_data, quick_fetch

# Simplest usage - get BTC data
btc_data = quick_fetch('BTC/USDT')

# Get multiple symbols  
data = fetch_data(['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])

# Access VectorBT features
close_prices = data.close
returns = data.returns
rsi = data.run('talib:RSI', 14)
```

## Main Entry Points

The module provides 5 simple functions that handle everything automatically:

1. **`fetch_data()`** - Primary interface for getting OHLCV data
2. **`fetch_top_symbols()`** - Get top symbols by volume
3. **`update_data()`** - Update cached data to latest
4. **`quick_fetch()`** - Simple single-symbol fetch
5. **`load_cached()`** - Load previously cached data

That's it! These functions automatically handle:
- ✅ Local caching for fast access
- ✅ Intelligent resampling from lower timeframes
- ✅ Exchange API calls only when necessary
- ✅ Metadata updates (symbol lists, volumes)
- ✅ Multi-exchange support
- ✅ VectorBT integration

## How It Works

### Automatic Caching

```python
# First call - fetches from exchange and caches
data = fetch_data(['BTC/USDT'], start_date='2024-01-01')

# Second call - loads from cache (instant!)
data = fetch_data(['BTC/USDT'], start_date='2024-01-01')
```

### Smart Resampling

```python
# If you have 1h data cached...
hourly_data = fetch_data(['BTC/USDT'], timeframe='1h')

# This resamples from cache instead of fetching!
daily_data = fetch_data(['BTC/USDT'], timeframe='1d')
```

### Multi-Exchange Support

```python
# Each exchange works the same way
binance_data = fetch_data(['BTC/USDT'], exchange='binance')
bybit_data = fetch_data(['BTCUSDT'], exchange='bybit', market_type='swap')
```

## Examples

### Basic Usage

```python
from backtester.data import fetch_data

# Fetch daily data for multiple symbols
data = fetch_data(
    symbols=['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
    timeframe='1d',
    start_date='2023-01-01'
)

# Work with VectorBT data
print(f"Symbols: {data.symbols}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"Latest BTC close: ${data.close['BTC/USDT'].iloc[-1]:,.2f}")
```

### Get Top Symbols

```python
from backtester.data import fetch_top_symbols

# Get top 20 USDT pairs by volume
top_data = fetch_top_symbols(limit=20)
print(f"Top symbols: {top_data.symbols}")

# Analyze them
returns = top_data.returns
correlation = returns.corr()
```

### Quick Prototyping

```python
from backtester.data import quick_fetch

# Get last 30 days of hourly BTC data
data = quick_fetch('BTC/USDT', days=30, timeframe='1h')

# Run quick analysis
sma_20 = data.run('talib:SMA', 20)
rsi_14 = data.run('talib:RSI', 14)
```

### Keep Data Updated

```python
from backtester.data import update_data

# Update all cached daily data
update_data(timeframe='1d')

# Update specific symbols
update_data(timeframe='1h', symbols=['BTC/USDT', 'ETH/USDT'])
```

## Architecture (For Developers)

While the interface is simple, the module has a sophisticated architecture:

```
backtester/data/
├── simple_interface.py    # 🎯 Main entry point - start here!
├── __init__.py           # Module exports
├── fetching/             # Core fetching logic
│   ├── data_fetcher_new.py
│   └── core/            # Modular components
├── storage/              # Data persistence
├── cache_system/         # Metadata caching
└── exchange_config.py    # Exchange settings
```

The complexity is hidden behind the simple interface. Users only need to know about the 5 main functions.

## Storage

Data is stored in an organized hierarchy:

```
vbt_data/
├── binance/
│   ├── spot/
│   │   ├── 1d/
│   │   │   ├── BTC_USDT.h5
│   │   │   └── ETH_USDT.h5
│   │   └── 1h/
│   └── swap/
└── cache/
    └── metadata.json
```

## Best Practices

1. **Use the simple interface** - Don't dig into internal modules unless necessary
2. **Let caching work** - The default settings are optimized
3. **Fetch multiple symbols together** - More efficient than individual calls
4. **Use `quick_fetch()` for experiments** - Perfect for Jupyter notebooks
5. **Update incrementally** - Use `update_data()` instead of re-fetching everything

## Troubleshooting

```python
from backtester.data import get_cache_info

# See what's cached
info = get_cache_info()
print(info)

# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)
```

## See Also

- [USAGE_GUIDE.md](USAGE_GUIDE.md) - Detailed usage examples
- [simple_interface.py](simple_interface.py) - Main entry point source code
- VectorBT Pro documentation for data object methods

## Summary

The data module makes data fetching simple:
- Import `fetch_data` and related functions
- Call them with your requirements
- Everything else happens automatically

No need to understand the internal architecture - just use the simple interface! 