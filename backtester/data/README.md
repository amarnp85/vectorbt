# Data Module

The data module provides a robust, efficient, and VectorBT-compatible data fetching and caching system for cryptocurrency backtesting. It handles multiple exchanges, timeframes, and provides seamless integration with VectorBT PRO's data structures.

## Overview

This module is designed to:
- Fetch historical OHLCV data from multiple exchanges (Binance, Bybit, Hyperliquid)
- Cache data locally for efficient reuse
- Provide intelligent resampling to minimize API calls
- Seamlessly integrate with VectorBT PRO's data structures
- Handle both spot and futures/swap markets
- Provide health checking and data quality validation

## Architecture

```
backtester/data/
├── fetching/           # Data fetching logic
│   ├── core/          # Core components (modular design)
│   │   ├── cache_handler.py      # Cache management
│   │   ├── data_fetcher.py       # Main fetching logic
│   │   ├── data_merger.py        # Merge multiple data sources
│   │   ├── exchange_fetcher.py   # Exchange-specific fetching
│   │   ├── fetch_logger.py       # Logging utilities
│   │   ├── freshness_checker.py  # Data freshness validation
│   │   ├── resampler.py          # Timeframe resampling
│   │   ├── symbol_resolver.py    # Symbol filtering/resolution
│   │   └── vbt_data_handler.py   # VBT data object creation
│   └── data_fetcher_new.py       # Public API interface
├── storage/            # Data storage layer
│   └── data_storage.py           # HDF5/Parquet storage
├── cache_system/       # Metadata caching
│   ├── cache_manager.py          # Cache operations
│   └── metadata_fetcher.py      # Exchange metadata fetching
├── health_check/       # Data validation
│   └── data_healthcheck.py      # Quality checks
└── exchange_config.py  # Exchange configurations
```

## Quick Start

### Basic Usage

```python
from backtester.data.fetching.data_fetcher_new import fetch_data, quick_fetch

# Quick fetch for single symbol
data = quick_fetch('BTC/USDT', days=365)
print(data.close)  # Access close prices

# Fetch multiple symbols
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
data = fetch_data(symbols, timeframe='1d', start_date='2024-01-01')

# Access VBT data features
returns = data.returns
rsi = data.run('talib:RSI', 14)
```

### VectorBT Integration

The module provides native VectorBT Data objects with full feature support:

```python
# Magnet features
open_prices = data.open
close_prices = data.close
high_prices = data.high
low_prices = data.low
volume = data.volume

# Calculations
hlc3 = data.hlc3  # (High + Low + Close) / 3
ohlc4 = data.ohlc4  # (Open + High + Low + Close) / 4

# Indicators via run()
sma = data.run('talib:SMA', 20)
ema = data.run('talib:EMA', 20)
bbands = data.run('talib:BBANDS', 20)

# Multi-symbol handling
btc_data = data.select('BTC/USDT')
eth_data = data.select('ETH/USDT')
```

## Core Features

### 1. Intelligent Data Fetching

The system uses a three-tier approach to minimize API calls:

1. **Cache Check**: First checks local storage for existing data
2. **Resampling**: Attempts to resample from lower timeframes if available
3. **Exchange Fetch**: Only fetches from exchange when necessary

```python
# This will use cache if available, resample if possible, or fetch from exchange
data = fetch_data(['BTC/USDT'], timeframe='4h', use_cache=True, prefer_resampling=True)
```

### 2. Multi-Exchange Support

Supports multiple exchanges with consistent interface:

```python
# Binance spot
binance_data = fetch_data(['BTC/USDT'], exchange_id='binance', market_type='spot')

# Bybit futures
bybit_data = fetch_data(['BTCUSDT'], exchange_id='bybit', market_type='swap')

# Hyperliquid perpetuals
hyper_data = fetch_data(['BTC-USD-PERP'], exchange_id='hyperliquid', market_type='swap')
```

### 3. Top Symbols Discovery

Fetch top symbols by volume:

```python
# Get top 20 USDT pairs by volume
top_data = fetch_top_symbols(limit=20, quote_currency='USDT')
print(f"Top symbols: {top_data.symbols}")
```

### 4. Data Updates

Keep cached data fresh:

```python
# Update all cached daily data
success = update_data('binance', '1d')

# Update specific symbols
success = update_data('binance', '1h', symbols=['BTC/USDT', 'ETH/USDT'])
```

### 5. Multi-Timeframe Analysis

Seamless resampling for multi-timeframe strategies:

```python
# Fetch hourly data
h1_data = fetch_data(['BTC/USDT'], timeframe='1h')

# Resample to higher timeframes
h4_data = h1_data.resample('4h')
d1_data = h1_data.resample('1d')
```

## Storage System

### File Structure

Data is stored in an organized hierarchy:

```
vbt_data/
├── binance/
│   ├── spot/
│   │   ├── 1h/
│   │   │   ├── BTC_USDT.h5
│   │   │   └── ETH_USDT.h5
│   │   └── 1d/
│   │       └── ...
│   └── swap/
│       └── ...
└── cache/
    ├── binance/
    │   ├── volume.json
    │   └── timestamps.json
    └── blacklist.json
```

### Storage Formats

- **HDF5**: Default format for OHLCV data (efficient, compressed)
- **Parquet**: Alternative format (better for cloud storage)
- **JSON**: Metadata caching (volumes, timestamps, blacklists)

## Health Checking

The module includes comprehensive health checking:

```python
from backtester.data.health_check.data_healthcheck import HealthChecker

# Run health check
checker = HealthChecker()
report = checker.generate_report()

# Check specific aspects
checker.check_data_quality(data)
checker.check_missing_data(data)
checker.check_price_anomalies(data)
```

## Configuration

### Environment Variables

```bash
# Cache settings
CACHE_DIR=./backtester/data/cache_system/cache
VBT_DATA_DIR=./vbt_data

# Exchange-specific (optional)
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
```

### Exchange Configuration

See `exchange_config.py` for exchange-specific settings:

```python
EXCHANGE_CONFIG = {
    'binance': {
        'ccxt_id': 'binance',
        'rate_limit': 1200,
        'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
        'markets': ['spot', 'swap']
    },
    # ... other exchanges
}
```

## Advanced Usage

### Custom Data Enhancement

Add technical indicators on fetch:

```python
from backtester.data.fetching.data_fetcher_new import create_enhanced_data

# Fetch base data
data = quick_fetch('BTC/USDT')

# Enhance with technicals
enhanced = create_enhanced_data(data, add_technicals=True)
# Now has: enhanced._sma_20, enhanced._sma_50, enhanced._rsi_14
```

### Integration Testing

Test VBT compatibility:

```python
from backtester.data.fetching.data_fetcher_new import test_vbt_integration

data = fetch_data(['BTC/USDT'])
results = test_vbt_integration(data)
print(results)  # Shows which VBT features are working
```

### Performance Tips

1. **Use Caching**: Always use `use_cache=True` unless you need fresh data
2. **Prefer Resampling**: Set `prefer_resampling=True` to minimize API calls
3. **Batch Requests**: Fetch multiple symbols in one call
4. **Update Incrementally**: Use `update_data()` instead of re-fetching everything

## Troubleshooting

### Common Issues

1. **Missing Data**
   ```python
   # Check what's in cache
   from backtester.data.fetching.data_fetcher_new import get_storage_info
   print(get_storage_info())
   ```

2. **Symbol Not Found**
   ```python
   # Check available symbols
   from backtester.data.cache_system.metadata_fetcher import fetch_metadata
   metadata = fetch_metadata('binance')
   print([s for s in metadata['symbols'] if 'BTC' in s][:10])
   ```

3. **VBT Integration Issues**
   ```python
   # Run integration test
   python backtester/data/test_vbt_integration.py
   ```

## API Reference

See the module docstrings for detailed API documentation. Key functions:

- `fetch_data()`: Main data fetching function
- `quick_fetch()`: Simplified single-symbol fetching
- `fetch_top_symbols()`: Get top symbols by volume
- `update_data()`: Update cached data
- `load_latest()`: Load from cache without fetching
- `test_vbt_integration()`: Test VBT compatibility

## Contributing

When adding new features:

1. Maintain VBT Data compatibility
2. Add appropriate logging
3. Update cache metadata appropriately
4. Include error handling
5. Add tests in `test_vbt_integration.py`

## Future Enhancements

- [ ] Real-time data streaming
- [ ] More exchange integrations
- [ ] Advanced data quality metrics
- [ ] Automated data validation
- [ ] Cloud storage backends 