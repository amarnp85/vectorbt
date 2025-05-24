# VectorBT Pro Backtesting System

A high-performance cryptocurrency data management system built on VectorBT Pro, featuring intelligent caching, storage resampling, and comprehensive health monitoring.

## 🎯 What This System Does

- **Fetches crypto data** from major exchanges (Binance, Bybit, OKX, etc.) using VectorBT Pro
- **Intelligent caching** with 97% storage reduction and 226x performance improvement  
- **True inception fetching** using 3,256+ cached inception dates across exchanges
- **Storage resampling** - automatically creates higher timeframes from cached lower timeframes
- **Health monitoring** - comprehensive data quality checks and auto-fix capabilities
- **Perfect VBT compatibility** - all VectorBT functionality preserved

## 🚀 Quick Start

### Installation

```bash
git clone <repository>
cd backtester
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```

### Your First Data Fetch

```bash
# Fetch top 5 symbols by volume for 1 hour timeframe
python backtester/scripts/fetch_data_cli.py --exchange binance --market spot --timeframe 1h --top 5

# Check data health
python -m backtester.data.health_check.data_healthcheck --exchange binance --timeframe 1h
```

## 📊 Data Fetching CLI - Complete Guide

### Basic Usage Patterns

#### 1. **Top Symbols by Volume**
```bash
# Top 10 symbols from Binance spot market, 1-day timeframe
python backtester/scripts/fetch_data_cli.py --exchange binance --market spot --timeframe 1d --top 10

# Top 20 symbols from Binance swap (futures), 4-hour timeframe  
python backtester/scripts/fetch_data_cli.py --exchange binance --market swap --timeframe 4h --top 20

# Filter by quote currency (default is USDT)
python backtester/scripts/fetch_data_cli.py --exchange binance --timeframe 1h --top 15 --quote USDT
```

#### 2. **Specific Symbols**
```bash
# Single symbol
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --exchange binance

# Multiple symbols (comma-separated, no spaces)
python backtester/scripts/fetch_data_cli.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT" --timeframe 4h

# Different exchanges
python backtester/scripts/fetch_data_cli.py --symbols "BTC/USDT,ETH/USDT" --exchange bybit --timeframe 1d
```

#### 3. **Date Ranges**

**Inception Fetching (Maximum History)**
```bash
# Fetch from true inception using cached inception dates
python backtester/scripts/fetch_data_cli.py --symbols "BTC/USDT,ETH/USDT" --timeframe 1d --inception

# True inception for Binance: BTC/ETH start from 2017-08-17
# True inception for Bybit: Different dates per symbol
# 3,256+ cached inception dates across exchanges
```

**Custom Date Ranges**
```bash
# Last 30 days
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --start "30 days ago"

# Specific date range
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 4h --start "2024-01-01" --end "2024-06-01"

# Last week to now
python backtester/scripts/fetch_data_cli.py --symbols ETH/USDT --timeframe 1d --start "7 days ago" --end "now"
```

### Advanced Options

#### **Market Types**
```bash
# Spot markets (default)
python backtester/scripts/fetch_data_cli.py --market spot --symbols BTC/USDT --timeframe 1h

# Futures/Swap markets  
python backtester/scripts/fetch_data_cli.py --market swap --symbols BTC/USDT --timeframe 1h

# Different exchanges support different market types
```

#### **Caching Control**
```bash
# Disable caching (force fresh fetch)
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --no-cache

# Default: caching enabled (recommended for performance)
```

#### **Output Control**
```bash
# Verbose output (detailed logging)
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --verbose

# Quiet output (minimal)
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --quiet

# Show storage summary
python backtester/scripts/fetch_data_cli.py --storage-summary
```

### Complete Command Reference

```bash
python backtester/scripts/fetch_data_cli.py [OPTIONS]

Symbol Selection (choose one):
  --symbols SYMBOLS     Comma-separated symbol list (e.g., "BTC/USDT,ETH/USDT")
  --top TOP            Fetch top N symbols by volume

Exchange Options:
  --exchange EXCHANGE   Exchange ID (binance, bybit, okx, kucoin, coinbase)
  --market {spot,swap}  Market type (default: spot)
  --quote QUOTE        Quote currency filter (default: USDT)

Timeframe:
  --timeframe TF       Data timeframe (1m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w)

Date Range:
  --start START        Start date (VBT-compatible: "7 days ago", "2024-01-01", etc.)
  --end END            End date (VBT-compatible: "now", "2024-06-01", etc.)
  --inception          Fetch from true inception (uses cached inception dates)

Control:
  --no-cache           Disable caching (force fresh fetch)
  --verbose            Detailed output with logging
  --quiet              Minimal output
  --storage-summary    Show current data storage status
  --help               Show help message
```

## 🔍 Health Check System - Complete Guide

The health check system monitors data quality, identifies issues, and can automatically fix problems.

### Basic Health Checks

#### **Check Specific Exchange/Timeframe**
```bash
# Check Binance spot 1-hour data
python -m backtester.data.health_check.data_healthcheck --exchange binance --timeframe 1h

# Check Binance spot 1-day data
python -m backtester.data.health_check.data_healthcheck --exchange binance --timeframe 1d

# Check Bybit swap 4-hour data
python -m backtester.data.health_check.data_healthcheck --exchange bybit --timeframe 4h --market swap
```

#### **Comprehensive Analysis**
```bash
# All available data files
python -m backtester.data.health_check.data_healthcheck

# Detailed analysis with expanded reporting
python -m backtester.data.health_check.data_healthcheck --exchange binance --timeframe 1h --detailed

# Auto-fix critical issues
python -m backtester.data.health_check.data_healthcheck --exchange binance --timeframe 1h --auto-fix
```

### Understanding Health Check Results

#### **Issue Severity Levels**

🔴 **Critical Issues** - Require immediate attention:
- Missing inception data (symbols missing 500+ days from expected start)
- Data structure corruption (wrong symbols, malformed data)
- Loading errors (corrupted files)
- Invalid OHLCV relationships (High < Low, etc.)

🟡 **Warning Issues** - Should be addressed:
- Minor data gaps (< 500 days missing)
- Stale data (not updated recently)  
- Zero volume periods
- Inconsistent symbol coverage

ℹ️ **Info Issues** - Informational:
- Optimization suggestions
- Storage efficiency notes
- Coverage statistics

#### **Sample Health Check Output**

```
🔍 DATA HEALTH CHECK REPORT
================================================================================
Generated: 2025-05-23 10:57:04
Files analyzed: 3
Auto-fix enabled: Yes

📋 SUMMARY
----------------------------------------
🔴 Critical issues: 2
🟡 Warnings: 3
ℹ️  Info: 1
📊 Total issues: 6

🚨 ISSUES FOUND
----------------------------------------

CRITICAL ISSUES (2):
  • binance_spot_1d.pickle.blosc: BTC/USDT missing 1836 days from inception
    expected_inception: 2017-08-17 00:00:00+00:00
    actual_start: 2022-08-27 00:00:00+00:00
    missing_days: 1836
  
  • binance_spot_1h.pickle.blosc: Data structure corruption detected
    issue: VBT data symbols show ['Open', 'High', 'Low', 'Close', 'Volume'] instead of trading pairs

WARNING ISSUES (3):
  • binance_spot_4h.pickle.blosc: Data is 12.5 hours behind current time
  • binance_spot_1h.pickle.blosc: Significant gaps in ETH/USDT data (15 missing days)
  • binance_spot_1d.pickle.blosc: Zero volume detected for SOL/USDT on 2024-03-15

💡 RECOMMENDATIONS
----------------------------------------
🔴 CRITICAL ACTIONS NEEDED:
  1. Run with --auto-fix to automatically resolve critical issues
  2. Address critical data gaps - consider re-fetching affected periods
  3. Fix invalid OHLCV data - these can affect calculations

🔧 AUTO-FIX AVAILABLE:
  Run this script with --auto-fix to automatically resolve many issues.
```

### Auto-Fix Capabilities

The health check can automatically resolve many issues:

```bash
# Auto-fix critical issues (recommended)
python -m backtester.data.health_check.data_healthcheck --exchange binance --timeframe 1h --auto-fix

# What auto-fix does:
# ✅ Re-fetches corrupted data files
# ✅ Fills critical inception gaps  
# ✅ Updates stale data
# ✅ Removes invalid files
# ✅ Executes fetch commands automatically
```

#### **Manual Fix Commands**

When auto-fix isn't available, the health check provides exact commands:

```bash
# Example commands generated by health check:
python backtester/scripts/fetch_data_cli.py --exchange binance --market spot --timeframe 1d --symbols "BTC/USDT" --inception

python backtester/scripts/fetch_data_cli.py --exchange binance --market spot --timeframe 1h --symbols "ETH/USDT" --start "2024-01-01" --end "now"
```

### Health Check Command Reference

```bash
python -m backtester.data.health_check.data_healthcheck [OPTIONS]

Filtering:
  --exchange EXCHANGE   Check specific exchange (binance, bybit, etc.)
  --timeframe TF       Check specific timeframe (1h, 4h, 1d, etc.)
  --market TYPE        Check specific market type (spot, swap)

Analysis:
  --detailed           Show detailed analysis with expanded reporting
  --auto-fix           Automatically execute fix commands for critical issues

Output:
  --help              Show help message
```

## ⚡ How The System Works

### 1. **Intelligent Data Flow**

```
User Request → Cache Check → Storage Resampling → API Fetch → Health Monitoring
     ↓              ↓              ↓                ↓             ↓
   0.002s      97% storage    226x faster     Last resort   Auto-validation
```

**Example Flow:**
1. **Request**: `--symbols BTC/USDT --timeframe 1d`
2. **Cache Check**: Look for `binance_spot_1d.pickle.blosc`
3. **Storage Resampling**: If not found, try resampling from `binance_spot_4h.pickle.blosc`
4. **API Fetch**: Only if resampling fails
5. **Health Check**: Automatic validation of data quality

### 2. **Storage Resampling Magic**

The system automatically creates higher timeframes from lower ones:

```
1h data (67,939 points) → 4h data (17,001 points) → 1d data (2,837 points)
                         ↑ 75% reduction        ↑ 83% reduction
                         ✅ Perfect OHLCV       ✅ Perfect OHLCV
```

**OHLCV Aggregation Rules:**
- **Open**: First value in period
- **High**: Maximum value in period  
- **Low**: Minimum value in period
- **Close**: Last value in period
- **Volume**: Sum of all values in period

### 3. **Cached Inception Dates**

The system maintains 3,256+ cached inception dates:
- **Binance**: 1,994 symbols (e.g., BTC/USDT: 2017-08-17)
- **Bybit**: 946 symbols (various dates per symbol)
- **Hyperliquid**: 316 symbols
- **Auto-updated**: When new symbols are discovered

### 4. **Storage Architecture**

```
vbt_data/
├── binance_spot_1h.pickle.blosc    # Hourly data (VBT native format)
├── binance_spot_4h.pickle.blosc    # 4-hour data (resampled or fetched)
├── binance_spot_1d.pickle.blosc    # Daily data (resampled or fetched)
├── binance_swap_1h.pickle.blosc    # Futures hourly data
└── bybit_spot_1d.pickle.blosc      # Other exchanges
```

**File Format**: `{exchange}_{market}_{timeframe}.pickle.blosc`
- **Compression**: Blosc (fast compression/decompression)
- **Format**: VectorBT Pro native pickle format
- **Metadata**: Complete VBT functionality preserved

## 🛠️ Troubleshooting

### Common Issues & Solutions

#### **"No data returned" Error**
```bash
# Check if exchange/market/symbol combination is valid
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --exchange binance --market spot --verbose

# Try a different timeframe
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1d
```

#### **"Storage resampling failed" Warning**  
```bash
# Run health check to identify corrupted data
python -m backtester.data.health_check.data_healthcheck --auto-fix

# Force fresh fetch if needed
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --no-cache
```

#### **Missing Historical Data**
```bash
# Use inception fetching for maximum history
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1d --inception

# Check cached inception dates
python -c "from backtester.data.cache_system import cache_manager; print(cache_manager.get_all_timestamps('binance')['BTC/USDT'])"
```

#### **Health Check Shows Corruption**
```bash
# Auto-fix critical issues
python -m backtester.data.health_check.data_healthcheck --auto-fix

# Manual fix: delete corrupted file and re-fetch
rm backtester/vbt_data/binance_spot_1h.pickle.blosc
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --inception
```

### Performance Tips

#### **Maximize Cache Efficiency**
```bash
# Fetch lower timeframes first (enables resampling)
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --inception
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 4h  # Uses resampling!
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1d  # Uses resampling!
```

#### **Batch Symbol Fetching**
```bash
# More efficient than individual fetches
python backtester/scripts/fetch_data_cli.py --top 20 --timeframe 1h  # Single API call

# Less efficient  
# Multiple individual commands for each symbol
```

### Health Check Maintenance

#### **Regular Health Monitoring**
```bash
# Daily health check
python -m backtester.data.health_check.data_healthcheck

# Weekly comprehensive check with auto-fix
python -m backtester.data.health_check.data_healthcheck --auto-fix

# Check specific timeframes you use most
python -m backtester.data.health_check.data_healthcheck --timeframe 1h --detailed
```

## 📈 Advanced Usage

### Using in Python Code

```python
# Import the data fetching functions
from backtester.data.fetching.data_fetcher import fetch_data, fetch_top_symbols
import vectorbtpro as vbt

# Fetch data with intelligent caching
data = fetch_data(
    symbols=['BTC/USDT', 'ETH/USDT'],
    exchange_id='binance',
    timeframe='1h',
    start_date='30 days ago',
    prefer_resampling=True  # Enable storage resampling
)

# All VectorBT functionality works perfectly
close_prices = data.get('close')
portfolio = vbt.PF.from_signals(data, entries=True, exits=False)

# Volume-based symbol selection
top_data = fetch_top_symbols(
    exchange_id='binance',
    quote_currency='USDT',
    limit=10,
    timeframe='1h'
)
```

### Storage Management

```bash
# Check current storage usage
python backtester/scripts/fetch_data_cli.py --storage-summary

# Manual cleanup (if needed)
rm backtester/vbt_data/old_files*.pickle.blosc

# Health check will identify any issues
python -m backtester.data.health_check.data_healthcheck
```

---

**🎯 Built for Performance**: Intelligent caching, storage resampling, and VectorBT Pro integration deliver enterprise-grade cryptocurrency data management with minimal resource usage. 