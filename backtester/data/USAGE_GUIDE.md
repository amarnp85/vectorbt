# Data Module Usage Guide

## Overview

The data module provides a simple interface for fetching cryptocurrency OHLCV data. It automatically handles caching, resampling, and exchange fetching, so you don't need to worry about the details.

## Quick Start

```python
from backtester.data import fetch_data, quick_fetch

# Simplest usage - get BTC data for the last year
btc_data = quick_fetch('BTC/USDT')

# Get multiple symbols
data = fetch_data(['BTC/USDT', 'ETH/USDT', 'BNB/USDT'])

# Access the data (VectorBT format)
close_prices = data.close
returns = data.returns
rsi = data.run('talib:RSI', 14)
```

## Main Functions

### 1. `fetch_data()` - Primary Interface

This is your main entry point for getting data. It automatically:
- Checks local cache first (fastest)
- Tries to resample from lower timeframes if available
- Only fetches from exchange when necessary
- Updates metadata (symbol lists, volumes) as needed

```python
data = fetch_data(
    symbols=['BTC/USDT', 'ETH/USDT'],
    exchange='binance',      # 'binance', 'bybit', 'hyperliquid'
    timeframe='1h',          # '1m', '5m', '15m', '1h', '4h', '1d'
    start_date='2024-01-01', # Optional: defaults to inception
    end_date='2024-12-31',   # Optional: defaults to now
    market_type='spot'       # 'spot' or 'swap' (futures)
)
```

### 2. `quick_fetch()` - Simple Single Symbol

For quick prototyping with a single symbol:

```python
# Get last 30 days of hourly BTC data
data = quick_fetch('BTC/USDT', days=30, timeframe='1h')
```

### 3. `fetch_top_symbols()` - Get Popular Symbols

Automatically discover and fetch the most traded symbols:

```python
# Get top 20 USDT pairs by volume
top_data = fetch_top_symbols(limit=20, quote_currency='USDT')
print(f"Fetched: {top_data.symbols}")
```

### 4. `update_data()` - Keep Data Fresh

Update your cached data to the latest:

```python
# Update all cached daily data
update_data(timeframe='1d')

# Update specific symbols only
update_data(timeframe='1h', symbols=['BTC/USDT', 'ETH/USDT'])
```

### 5. `load_cached()` - Offline Access

Load previously cached data without fetching:

```python
# Load all cached daily data
data = load_cached()

# Load specific symbols
data = load_cached(symbols=['BTC/USDT', 'ETH/USDT'])
```

## How It Works (Automatic Features)

### 1. Smart Caching

When you call `fetch_data()`, it automatically:

1. **Checks Cache First**: Looks for existing data in `vbt_data/` directory
2. **Validates Freshness**: Ensures cached data is up-to-date
3. **Fills Gaps**: Only fetches missing data from exchange

Example flow:
```python
# First call - fetches from exchange and caches
data = fetch_data(['BTC/USDT'], start_date='2024-01-01')

# Second call - loads from cache (instant!)
data = fetch_data(['BTC/USDT'], start_date='2024-01-01')

# Third call - only fetches new data
data = fetch_data(['BTC/USDT'], start_date='2024-01-01', end_date='2024-12-31')
```

### 2. Intelligent Resampling

The module can create higher timeframes from lower ones:

```python
# If you have 1h data cached...
hourly_data = fetch_data(['BTC/USDT'], timeframe='1h')

# This will resample from 1h instead of fetching from exchange!
daily_data = fetch_data(['BTC/USDT'], timeframe='1d')
```

Resampling hierarchy:
- 1m → 5m → 15m → 1h → 4h → 1d

### 3. Metadata Management

The module automatically maintains:
- Symbol lists per exchange
- 24h volumes for ranking
- Symbol blacklists (delisted/invalid)
- Last update timestamps

This happens transparently when you fetch data.

### 4. Multi-Exchange Support

Each exchange has its own cache and configuration:

```python
# Binance spot market
binance_data = fetch_data(['BTC/USDT'], exchange='binance')

# Bybit futures
bybit_data = fetch_data(['BTCUSDT'], exchange='bybit', market_type='swap')

# Different exchanges use different symbol formats
# The module handles this automatically
```

## Working with VectorBT Data

The returned data is a VectorBT Data object with rich functionality:

```python
data = fetch_data(['BTC/USDT', 'ETH/USDT'])

# Access OHLCV data
open_prices = data.open
high_prices = data.high
low_prices = data.low
close_prices = data.close
volume = data.volume

# Calculate returns
returns = data.returns
log_returns = data.log_returns

# Run indicators
sma_20 = data.run('talib:SMA', 20)
rsi_14 = data.run('talib:RSI', 14)
bbands = data.run('talib:BBANDS', 20, 2, 2)

# Select specific symbols
btc_data = data.select('BTC/USDT')
eth_data = data.select('ETH/USDT')

# Resample to different timeframe
if data.wrapper.freq:  # If timeframe info available
    daily_data = data.resample('1D')
```

## Storage Structure

Data is organized in a clear hierarchy:

```
vbt_data/
├── binance/
│   ├── spot/
│   │   ├── 1m/
│   │   │   ├── BTC_USDT.h5
│   │   │   └── ETH_USDT.h5
│   │   ├── 1h/
│   │   └── 1d/
│   └── swap/
├── bybit/
└── cache/
    ├── binance/
    │   ├── volume.json
    │   └── timestamps.json
    └── blacklist.json
```

## Best Practices

1. **Always use caching** (default behavior) unless you need real-time data
2. **Fetch multiple symbols together** - more efficient than individual calls
3. **Use `quick_fetch()` for prototyping** - simple and fast
4. **Update incrementally** with `update_data()` instead of re-fetching
5. **Check cache info** with `get_cache_info()` to see what's available

## Common Patterns

### Strategy Development
```python
# Get data for strategy testing
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
data = fetch_data(symbols, timeframe='4h', start_date='2023-01-01')

# Run strategy with the data
# ... strategy code ...
```

### Multi-Timeframe Analysis
```python
# Fetch base timeframe
h1_data = fetch_data(['BTC/USDT'], timeframe='1h')

# The module will resample these automatically from cache
h4_data = fetch_data(['BTC/USDT'], timeframe='4h')
d1_data = fetch_data(['BTC/USDT'], timeframe='1d')
```

### Portfolio Selection
```python
# Get top symbols for portfolio
top_symbols = fetch_top_symbols(limit=20)
print(f"Selected: {top_symbols.symbols}")

# Analyze them
returns = top_symbols.returns
correlation = returns.corr()
```

## Troubleshooting

### Check What's Cached
```python
from backtester.data import get_cache_info
info = get_cache_info()
print(info)
```

### Force Fresh Fetch
```python
# Clear cache for specific symbol/timeframe
import os
cache_file = 'vbt_data/binance/spot/1d/BTC_USDT.h5'
if os.path.exists(cache_file):
    os.remove(cache_file)

# Now fetch will get fresh data
data = fetch_data(['BTC/USDT'])
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now you'll see detailed logs of what the module is doing
data = fetch_data(['BTC/USDT'])
```

## Summary

The data module handles all the complexity of data management for you:
- ✅ Automatic caching
- ✅ Smart resampling
- ✅ Exchange abstraction
- ✅ Metadata management
- ✅ VectorBT integration

Just call `fetch_data()` with your requirements, and the module takes care of the rest! 