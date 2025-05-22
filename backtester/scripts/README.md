# CLI Tools for Data Management

This directory contains the command-line tool for fetching and updating cryptocurrency data using the VBT-native approach with intelligent caching.

## Tools

### 🚀 `fetch_data_cli.py` - Intelligent Data Fetching & Updating Tool

A comprehensive CLI tool that handles both fetching and updating cryptocurrency data with:
- **Intelligent Caching**: Automatically detects existing data and uses VBT's native update()
- **Market Types**: `spot` and `swap` markets
- **Top Symbol Selection**: Fetch top N symbols by volume
- **Specific Symbols**: Fetch user-defined symbol lists
- **Historical Data**: Fetch from inception (maximum available history)
- **Custom Date Ranges**: Flexible start/end date specification
- **Automatic Updates**: Uses VBT's incremental update when cached data exists
- **Storage Management**: View stored data summaries

### 📁 Shell Alias

- **`fetch`** - Quick alias for `fetch_data_cli.py`

## Key Feature: Intelligent Update Behavior

The fetch tool is smart - it automatically:
1. **Checks for existing cached data**
2. **Uses VBT's native `update()` method** to get latest data points
3. **Only fetches fresh** if no cache exists or new symbols are requested
4. **Merges data intelligently** using VBT's native capabilities

**This means you only need one command for both fetching and updating!**

## Usage Examples

### Initial Data Fetching

```bash
# Fetch top 10 spot symbols, 1h timeframe, from inception
python backtester/scripts/fetch_data_cli.py --exchange binance --market spot --timeframe 1h --top 10 --inception

# Fetch specific symbols with date range
python backtester/scripts/fetch_data_cli.py --exchange binance --symbols "BTC/USDT,ETH/USDT" --start "30 days ago"
```

### Automatic Updates (Same Command!)

```bash
# Run the same command again - it will automatically update existing data!
python fetch_data_cli.py --exchange binance --market spot --timeframe 1h --top 10

# The tool automatically:
# 1. Detects existing cached data
# 2. Uses VBT's update() to get latest points
# 3. Saves updated cache
```

### Data Management

```bash
# View stored data summary
python fetch_data_cli.py --storage-summary

# Force fresh fetch (ignore cache)
python fetch_data_cli.py --exchange binance --top 5 --no-cache
```

### Using the Alias

```bash
# Same functionality with shorter command
./fetch --exchange binance --market spot --timeframe 1h --top 10 --inception
./fetch --storage-summary

# Updates work the same way
./fetch --exchange binance --timeframe 1h --top 10  # Auto-updates if data exists
```

## Options Reference

### Data Selection
- `--symbols SYMBOLS`: Comma-separated list of symbols (e.g., `BTC/USDT,ETH/USDT`)
- `--top N`: Number of top symbols by volume to fetch (default: 10)

### Exchange & Market
- `--exchange EXCHANGE`: Exchange identifier (default: `binance`)
- `--market {spot,swap}`: Market type (default: `spot`)
- `--quote CURRENCY`: Quote currency for top symbols filtering (default: `USDT`)

### Time Settings
- `--timeframe TF`: Timeframe (e.g., `1m`, `5m`, `1h`, `4h`, `1d`) (default: `1d`)
- `--start DATE`: Start date (e.g., `"7 days ago"`, `"2024-01-01"`)
- `--end DATE`: End date (e.g., `"1 day ago"`, `"2024-12-31"`)
- `--inception`: Fetch from inception (maximum available history)

### Cache & Output
- `--no-cache`: Disable caching (fetch fresh data)
- `--quiet`, `-q`: Quiet mode (minimal output)
- `--verbose`, `-v`: Verbose mode (detailed output)
- `--storage-summary`: Show storage summary and exit

## Storage Structure

Data is stored in VBT-native format with the following naming convention:

```
vbt_data/
├── {exchange}_{market}_{timeframe}.pickle.blosc
└── ...

Examples:
├── binance_spot_1h.pickle.blosc      # Binance spot 1h data
├── binance_spot_1d.pickle.blosc      # Binance spot 1d data  
├── binance_swap_4h.pickle.blosc      # Binance swap 4h data
└── bybit_spot_1d.pickle.blosc        # Bybit spot 1d data
```

## Features

### ✅ **VBT-Native Integration**
- Complete VectorBT Pro metadata preservation
- Seamless VBT functionality (indicators, backtesting, etc.)
- Efficient pickle storage with blosc compression

### ✅ **Intelligent Caching & Auto-Updates**
- Automatic cache detection and usage
- **VBT's native incremental update capabilities**
- **Single command handles both fetch and update scenarios**
- Support for both fresh fetching and intelligent caching

### ✅ **Volume-Based Selection**
- Automatically fetch top symbols by trading volume
- Market-specific volume ranking
- Quote currency filtering

### ✅ **Flexible Operations**
- Natural language dates (`"7 days ago"`)
- Inception fetching for maximum historical data
- Custom date ranges and symbol filtering

### ✅ **Market Type Support**
- Spot and swap market differentiation
- Clear filename separation
- Market-specific symbol filtering

## Automation & Scheduling

### Cron Job Example

To automatically update data daily using the same fetch command:

```bash
# Add to crontab (crontab -e)
# Update all data at 1 AM daily (same command, auto-updates existing data)
0 1 * * * cd /path/to/backtester && ./backtester/scripts/fetch --exchange binance --top 20 --timeframe 1d --quiet

# Update specific high-frequency data every hour
0 * * * * cd /path/to/backtester && ./backtester/scripts/fetch --exchange binance --timeframe 1h --top 10 --quiet
```

### Systemd Timer Example

```ini
# /etc/systemd/system/backtester-update.service
[Unit]
Description=Update Backtester Data

[Service]
Type=oneshot
WorkingDirectory=/path/to/backtester
ExecStart=/path/to/backtester/backtester/scripts/fetch --exchange binance --top 20 --timeframe 1d --quiet
User=yourusername

# /etc/systemd/system/backtester-update.timer
[Unit]
Description=Run Backtester Update Daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

## How the Intelligence Works

### First Run (Fresh Fetch)
```bash
$ ./fetch --exchange binance --timeframe 1h --top 5

🚀 Fetching Data
💾 No cached data found
📡 Fetching 5 symbols from binance
✅ Fresh fetch successful: (1000, 5)
💾 Saved fresh data to cache
```

### Second Run (Automatic Update)
```bash
$ ./fetch --exchange binance --timeframe 1h --top 5

🚀 Fetching Data  
📦 Found cached data with 5 symbols
✅ All symbols found in cache
📈 Updated cache with latest data  # ← VBT's update() called automatically!
✅ Cache hit: returning 5 symbols from cache
```

### Third Run (New Symbol Added)
```bash
$ ./fetch --exchange binance --timeframe 1h --top 6

🚀 Fetching Data
📦 Found cached data with 5 symbols
🔍 Cache missing symbols: ['SOL/USDT']  # ← Detects new symbol
📡 Fetching 1 symbols from binance    # ← Fetches only missing data
🔄 Merging cached and fresh data using VBT.merge()
✅ VBT merge successful: 6 total symbols
```

## Example Output

```bash
$ python fetch_data_cli.py --exchange binance --market spot --timeframe 1d --top 3 --inception --verbose

🚀 Fetching Data
===================================================
Exchange: BINANCE
Market: SPOT  
Timeframe: 1d
Top symbols: 3 (quote: USDT)
Date range: FROM INCEPTION
Cache: enabled
==================================================

📦 Found cached data with 3 symbols
✅ All symbols found in cache
📈 Updated cache with latest data
✅ Cache hit: returning 3 symbols from cache

✅ Data fetched successfully!
📊 Exchange: BINANCE
📊 Market: SPOT
📊 Timeframe: 1d
📊 Symbols: ['BTC/USDT', 'ETH/USDT', 'USDC/USDT']
📊 Shape: (2837, 5)  # ← New data point added automatically!
📊 Date range: 2017-08-17 00:00:00+00:00 to 2025-05-23 00:00:00+00:00

💾 Data saved to VBT storage
```

**One tool, intelligent behavior - it handles everything!** 🎯 