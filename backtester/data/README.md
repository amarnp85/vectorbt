# VBT-Native Data Management System

## Overview

This system provides a comprehensive VBT-native approach to cryptocurrency data fetching, caching, and storage. It leverages VectorBT Pro's built-in capabilities for optimal performance and functionality preservation.

## 🎯 Core Philosophy

**VBT-First Approach**: Everything is designed around VectorBT Pro's native capabilities:
- Uses VBT's built-in `update()` for incremental data fetching
- Uses VBT's `merge()` for intelligent data combination
- Uses VBT's pickle persistence for complete metadata preservation
- Uses VBT's datetime handling instead of manual pandas operations
- Preserves all VBT functionality (indicators, portfolios, etc.)

## 📁 Architecture

```
backtester/data/
├── fetching/
│   └── data_fetcher.py      # Main VBT-native data fetching logic
├── storage/
│   └── data_storage.py      # VBT pickle storage management
├── cache_system/
│   ├── cache_manager.py     # Volume/timestamp cache management
│   └── metadata_fetcher.py  # Exchange metadata utilities
├── scripts/
│   ├── fetch_data_cli.py    # Command-line interface
│   └── fetch                # CLI alias script
└── vbt_data/               # Storage directory (created automatically)
    ├── binance_spot_1h.pickle.blosc
    ├── binance_spot_4h.pickle.blosc
    └── binance_spot_1d.pickle.blosc
```

## 🚀 Core Components

### 1. Data Fetcher (`fetching/data_fetcher.py`)

The main interface for data operations using VBT-native methods:

```python
from backtester.data.fetching.data_fetcher import fetch_data, fetch_top_symbols

# Basic data fetching
data = fetch_data(
    symbols=['BTC/USDT', 'ETH/USDT'],
    exchange_id='binance',
    timeframe='1h',
    start_date='7 days ago',
    use_cache=True,
    market_type='spot'
)

# Volume-based symbol selection
data = fetch_top_symbols(
    exchange_id='binance',
    quote_currency='USDT',
    limit=10,
    timeframe='1h',
    market_type='spot'
)
```

#### Key Features:
- **Smart Incremental Caching**: Only fetches missing symbols or data
- **VBT Native Merging**: Uses `vbt.Data.merge()` for combining cached and fresh data
- **Automatic Updates**: Uses VBT's `data.update()` for seamless data appending
- **Market Type Support**: Distinguishes between spot, swap, and future markets

### 2. Storage System (`storage/data_storage.py`)

VBT-native pickle storage with compression:

```python
from backtester.data.storage.data_storage import data_storage

# Save VBT data (preserves all metadata)
success = data_storage.save_data(vbt_data, 'binance', '1h', 'spot')

# Load VBT data (full functionality preserved)
data = data_storage.load_data('binance', '1h', market_type='spot')

# Get storage summary
summary = data_storage.get_storage_summary()
```

#### Storage Format:
- **Filename**: `{exchange}_{market_type}_{timeframe}.pickle.blosc`
- **Compression**: Uses blosc for optimal space/speed balance
- **Location**: `vbt_data/` directory (no subdirectories)
- **Metadata**: Complete VBT wrapper and fetch parameters preserved

### 3. Cache System (`cache_system/`)

High-performance caching for volume data and metadata:

```python
from backtester.data.cache_system import cache_manager, data_fetcher

# Volume-based operations
volumes = cache_manager.get_all_volumes('binance')
cache_manager.save_all_volumes('binance', volume_data)

# Market data with volume
market_data = data_fetcher.get_market_data(
    exchange_id='binance',
    quote_currency='USDT',
    limit=20,
    top_by_volume=True
)
```

## 🔄 How Smart Caching Works

### 1. Cache Hit Flow
```
User Request: ['BTC/USDT', 'ETH/USDT'] 
    ↓
Check Storage: binance_spot_1h.pickle.blosc exists
    ↓
Load VBT Data: 8 symbols found in cache
    ↓
Symbol Check: All requested symbols present ✅
    ↓
Return: data.select(['BTC/USDT', 'ETH/USDT'])
```

### 2. Incremental Update Flow
```
User Request: ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    ↓
Check Storage: binance_spot_1h.pickle.blosc exists
    ↓
Load VBT Data: ['BTC/USDT', 'ETH/USDT'] found
    ↓
Missing Symbols: ['ADA/USDT'] identified 🔍
    ↓
Fetch Missing: vbt.CCXTData.pull(['ADA/USDT'])
    ↓
VBT Merge: vbt.Data.merge(cached_data, fresh_data)
    ↓
Save Combined: data_storage.save_data(merged_data)
    ↓
Return: merged_data.select(requested_symbols)
```

### 3. Date Range Extension
```
User Request: end_date beyond cached data
    ↓
Load Cached Data: existing VBT data object
    ↓
VBT Update: cached_data.update(end=new_end_date)
    ↓
Auto-fetch: VBT fetches gap automatically
    ↓
Save Updated: data_storage.save_data(updated_data)
```

## 🎮 Command Line Interface

### Basic Usage
```bash
# Fetch top 5 symbols by volume
python backtester/scripts/fetch_data_cli.py --top 5 --timeframe 1h

# Fetch specific symbols
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT,ETH/USDT --timeframe 4h

# Fetch from inception (maximum history)
python backtester/scripts/fetch_data_cli.py --symbols SOL/USDT --timeframe 1d --inception

# Fetch with custom date range
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --start "30 days ago" --end "now"

# Show storage summary
python backtester/scripts/fetch_data_cli.py --storage-summary

# Use alias script (if available)
./backtester/scripts/fetch --top 10 --timeframe 1d --verbose
```

### CLI Options
- `--top N`: Fetch top N symbols by volume
- `--symbols LIST`: Comma-separated symbol list
- `--timeframe TF`: Data timeframe (1m, 5m, 15m, 1h, 4h, 1d, etc.)
- `--exchange EX`: Exchange ID (default: binance)
- `--market-type TYPE`: spot/swap (default: spot)
- `--inception`: Fetch maximum available history
- `--start DATE`: Start date (VBT-compatible strings)
- `--end DATE`: End date (VBT-compatible strings)
- `--no-cache`: Disable caching
- `--verbose`: Detailed logging
- `--quiet`: Minimal output
- `--storage-summary`: Show current storage status

## 💡 Key Benefits

### 1. **VBT Functionality Preserved**
- All `data.get('field')` operations work seamlessly
- VBT indicators (`vbt.talib()`) work directly
- Portfolio analysis (`vbt.PF`) ready out of the box
- Native plotting capabilities maintained

### 2. **Intelligent Caching**
- Only fetches what's missing (symbols or date ranges)
- Uses VBT's native merge for data combination
- Handles timezone and index alignment automatically
- Preserves all VBT metadata during caching

### 3. **Performance Optimized**
- Blosc compression for fast I/O
- Smart cache lookups (0.002 second cache hits)
- Parallel fetching for multiple symbols
- VBT's optimized data structures

### 4. **Development Friendly**
- Comprehensive logging with emojis
- Detailed error handling
- Incremental development workflow
- Full CLI interface for testing

## 🧪 Usage Examples

### Basic Data Operations
```python
import vectorbtpro as vbt
from backtester.data.fetching.data_fetcher import fetch_data

# Fetch and use data
data = fetch_data(['BTC/USDT', 'ETH/USDT'], timeframe='1h')

# All VBT functionality works
close = data.get('close')
rsi = vbt.talib('RSI').run(close['BTC/USDT'])
portfolio = vbt.PF.from_signals(data, entries=True, exits=False)

print(f"Latest RSI: {rsi.real.iloc[-1]:.2f}")
print(f"Portfolio value: ${portfolio.value.iloc[-1].sum():.2f}")
```

### Volume-Based Selection
```python
from backtester.data.fetching.data_fetcher import fetch_top_symbols

# Get top symbols with volume metadata
data = fetch_top_symbols(
    exchange_id='binance',
    quote_currency='USDT',
    limit=10,
    timeframe='1h'
)

# Access volume rankings
if hasattr(data.wrapper, '_metadata'):
    volumes = data.wrapper._metadata['volume_data']
    for symbol in data.symbols:
        volume = volumes.get(symbol, 0)
        print(f"{symbol}: ${volume/1000000:.1f}M")
```

### Storage Management
```python
from backtester.data.storage.data_storage import data_storage

# Get comprehensive storage info
summary = data_storage.get_storage_summary()
print(f"Total files: {summary['pickle_files']}")
print(f"Total size: {summary['total_size_mb']:.2f} MB")

for filename, info in summary['files'].items():
    print(f"{filename}: {info['symbol_count']} symbols, {info['size_mb']} MB")
```

## 🔧 Technical Details

### Storage Format
- **Files**: `{exchange}_{market}_{timeframe}.pickle.blosc`
- **Compression**: Blosc (fast compression/decompression)
- **Content**: Complete VBT Data objects with metadata
- **Location**: `vbt_data/` directory (flat structure)

### Caching Strategy
1. **Symbol-level caching**: Missing symbols trigger incremental fetch
2. **Date-range extension**: VBT's `update()` handles gaps automatically
3. **Metadata preservation**: All VBT wrapper info maintained
4. **Smart merging**: VBT handles index alignment and missing data

### Error Handling
- Graceful degradation when cache fails
- Fallback to fresh fetch on merge errors
- Comprehensive logging for debugging
- Failed symbol tracking to avoid repeated attempts

## 📈 Performance Characteristics

### Cache Performance
- **Cache Hit**: ~0.002 seconds (near-instantaneous)
- **Incremental Fetch**: Only missing symbols fetched
- **Storage I/O**: Optimized with blosc compression

### Fetch Performance
- **Parallel Fetching**: ThreadPoolExecutor for multiple symbols
- **Exchange Optimization**: VBT handles rate limiting automatically
- **Data Alignment**: VBT merge handles mismatched indexes

## 🚧 Migration Notes

If migrating from the old HDF5 system:
1. Old files can coexist (different directory structure)
2. New system uses `vbt_data/` directory
3. All new fetches use VBT-native approach
4. CLI tools work with new system only
5. Existing scripts can be updated to use new functions

## 🔮 Future Enhancements

- **Real-time updates**: Streaming data integration
- **Multi-exchange merging**: Cross-exchange arbitrage data
- **Advanced caching**: Time-based cache invalidation
- **Performance metrics**: Detailed caching statistics
