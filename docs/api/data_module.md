# Data Module API Documentation

## Overview

The data module provides a simplified interface for fetching, caching, and managing cryptocurrency OHLCV data. It automatically handles caching, resampling, and metadata management.

## Primary Interface

### `fetch_data()`

The main entry point for fetching OHLCV data with automatic caching and resampling.

```python
from backtester.data import fetch_data

data = fetch_data(
    symbols: List[str],
    exchange: str = 'binance',
    timeframe: str = '1h',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Optional[vbt.Data]
```

**Parameters:**
- `symbols` (List[str]): List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
- `exchange` (str): Exchange ID (default: 'binance')
- `timeframe` (str): Candle timeframe (e.g., '1m', '5m', '15m', '1h', '4h', '1d')
- `start_date` (str, optional): Start date in 'YYYY-MM-DD' format
- `end_date` (str, optional): End date in 'YYYY-MM-DD' format
- `cache_dir` (str, optional): Custom cache directory path

**Returns:**
- `vbt.Data`: VectorBT Data object containing OHLCV data
- `None`: If data fetching fails

**Features:**
- Automatic caching to reduce API calls
- Smart resampling from lower to higher timeframes
- Metadata updates (symbol lists, volumes)
- Multi-symbol support

**Example:**
```python
# Single symbol
btc_data = fetch_data(
    symbols=['BTC/USDT'],
    timeframe='1h',
    start_date='2024-01-01'
)

# Multiple symbols
portfolio_data = fetch_data(
    symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT'],
    timeframe='4h',
    start_date='2024-01-01',
    end_date='2024-12-31'
)
```

### `fetch_top_symbols()`

Fetch the top symbols by 24-hour trading volume.

```python
from backtester.data import fetch_top_symbols

top_symbols = fetch_top_symbols(
    exchange: str = 'binance',
    quote_currency: str = 'USDT',
    top_n: int = 10,
    exclude_symbols: Optional[List[str]] = None
) -> List[str]
```

**Parameters:**
- `exchange` (str): Exchange ID (default: 'binance')
- `quote_currency` (str): Quote currency to filter by (default: 'USDT')
- `top_n` (int): Number of top symbols to return (default: 10)
- `exclude_symbols` (List[str], optional): Symbols to exclude from results

**Returns:**
- `List[str]`: List of symbol names sorted by volume

**Example:**
```python
# Get top 20 USDT pairs
top_20 = fetch_top_symbols(top_n=20)

# Get top BTC pairs excluding stablecoins
top_btc = fetch_top_symbols(
    quote_currency='BTC',
    exclude_symbols=['USDT/BTC', 'USDC/BTC']
)
```

### `update_data()`

Update cached data to the latest available.

```python
from backtester.data import update_data

updated = update_data(
    symbols: Optional[List[str]] = None,
    exchange: str = 'binance',
    timeframes: Optional[List[str]] = None,
    cache_dir: Optional[str] = None
) -> bool
```

**Parameters:**
- `symbols` (List[str], optional): Specific symbols to update (None = all cached)
- `exchange` (str): Exchange ID (default: 'binance')
- `timeframes` (List[str], optional): Specific timeframes to update
- `cache_dir` (str, optional): Custom cache directory

**Returns:**
- `bool`: True if update successful, False otherwise

**Example:**
```python
# Update all cached data
update_data()

# Update specific symbols
update_data(symbols=['BTC/USDT', 'ETH/USDT'])

# Update specific timeframes
update_data(timeframes=['1h', '4h'])
```

### `quick_fetch()`

Simplified single-symbol data fetching for prototyping.

```python
from backtester.data import quick_fetch

data = quick_fetch(
    symbol: str,
    days: int = 30,
    timeframe: str = '1h',
    exchange: str = 'binance'
) -> Optional[pd.DataFrame]
```

**Parameters:**
- `symbol` (str): Trading pair (e.g., 'BTC/USDT')
- `days` (int): Number of days of historical data (default: 30)
- `timeframe` (str): Candle timeframe (default: '1h')
- `exchange` (str): Exchange ID (default: 'binance')

**Returns:**
- `pd.DataFrame`: OHLCV DataFrame with columns: open, high, low, close, volume
- `None`: If fetching fails

**Example:**
```python
# Quick 30-day BTC data
btc = quick_fetch('BTC/USDT')

# 90 days of daily ETH data
eth_daily = quick_fetch('ETH/USDT', days=90, timeframe='1d')
```

### `load_cached()`

Load previously cached data without fetching.

```python
from backtester.data import load_cached

data = load_cached(
    symbols: List[str],
    exchange: str = 'binance',
    timeframe: str = '1h',
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> Optional[vbt.Data]
```

**Parameters:**
- Same as `fetch_data()`

**Returns:**
- `vbt.Data`: Cached data if available
- `None`: If no cached data found

**Example:**
```python
# Load cached data for offline analysis
cached_data = load_cached(
    symbols=['BTC/USDT', 'ETH/USDT'],
    timeframe='1h',
    start_date='2024-01-01'
)

if cached_data is None:
    print("No cached data found, fetching fresh data...")
    cached_data = fetch_data(...)
```

## Exchange Utilities

### `list_available_exchanges()`

Get list of supported exchanges.

```python
from backtester.data import list_available_exchanges

exchanges = list_available_exchanges() -> List[str]
```

**Returns:**
- `List[str]`: List of exchange IDs (e.g., ['binance', 'coinbase', 'kraken'])

### `get_exchange_info()`

Get detailed information about an exchange.

```python
from backtester.data import get_exchange_info

info = get_exchange_info(exchange_id: str) -> Dict[str, Any]
```

**Parameters:**
- `exchange_id` (str): Exchange identifier

**Returns:**
- `Dict`: Exchange information including:
  - `name`: Display name
  - `has`: Feature support flags
  - `timeframes`: Available timeframes
  - `symbols`: Number of trading pairs

### `get_exchange_timeframes()`

Get available timeframes for an exchange.

```python
from backtester.data import get_exchange_timeframes

timeframes = get_exchange_timeframes(exchange_id: str) -> List[str]
```

**Parameters:**
- `exchange_id` (str): Exchange identifier

**Returns:**
- `List[str]`: Available timeframes (e.g., ['1m', '5m', '15m', '1h', '4h', '1d'])

## Advanced Features

### Caching System

The data module automatically manages a sophisticated caching system:

- **Location**: `backtester/data/cache_system/cache/`
- **Structure**: Organized by exchange → symbol → timeframe
- **Validation**: Automatic freshness checking
- **Updates**: Incremental updates to minimize API calls

### Resampling Hierarchy

Data is intelligently resampled following this hierarchy:
```
1m → 5m → 15m → 1h → 4h → 1d → 1w → 1M
```

Higher timeframes are automatically created from lower ones when available.

### Metadata Management

The module maintains metadata for each exchange:
- Symbol lists with trading status
- 24-hour volumes for ranking
- Blacklists for invalid symbols
- Last update timestamps

## Error Handling

All functions include comprehensive error handling:

```python
data = fetch_data(symbols=['INVALID/PAIR'])
if data is None:
    # Handle error - check logs for details
    logger.error("Failed to fetch data")
```

## Performance Tips

1. **Use Caching**: Always use the default caching to minimize API calls
2. **Batch Requests**: Fetch multiple symbols in one call
3. **Appropriate Timeframes**: Use the highest timeframe suitable for your strategy
4. **Update Wisely**: Use `update_data()` during off-peak hours

## Migration from Legacy Code

If you have code using old imports:

```python
# Old way (deprecated)
from backtester.data.fetching.ccxt_data_fetcher import CCXTDataFetcher
fetcher = CCXTDataFetcher()
data = fetcher.fetch_data(...)

# New way (recommended)
from backtester.data import fetch_data
data = fetch_data(symbols=[...], ...)
```

## Thread Safety

The data module is thread-safe for reading but not for writing. Use appropriate locking if updating cache from multiple threads.

## Dependencies

- `vectorbtpro`: For data structures and resampling
- `ccxt`: For exchange connectivity
- `pandas`: For data manipulation
- `numpy`: For numerical operations 