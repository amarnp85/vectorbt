# Backtester CLI Commands Reference

This document provides a comprehensive reference for all CLI tools available in the backtester project for managing cryptocurrency data, cache inspection, health checks, and data fetching operations.

## Table of Contents

1. [Data Fetching](#data-fetching)
2. [Cache Inspection](#cache-inspection)
3. [Health Checks](#health-checks)
4. [Data Management](#data-management)
5. [Exchange Information](#exchange-information)
6. [Utility Scripts](#utility-scripts)

---

## Data Fetching

### Primary Data Fetcher (`fetch_data_cli.py`)

The main tool for fetching cryptocurrency data from various exchanges.

#### Basic Usage

```bash
# Navigate to project root
cd /path/to/backtester

# Fetch top 10 symbols by volume (default)
python backtester/scripts/fetch_data_cli.py --exchange binance --timeframe 1d

# Fetch specific symbols
python backtester/scripts/fetch_data_cli.py --symbols "BTC/USDT,ETH/USDT,SOL/USDT" --timeframe 4h

# Fetch from inception (maximum history)
python backtester/scripts/fetch_data_cli.py --symbols "BTC/USDT,ETH/USDT" --timeframe 1h --inception

# Fetch with custom date range
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --start "30 days ago" --end "now"
```

#### Advanced Options

```bash
# Different exchanges and markets
python backtester/scripts/fetch_data_cli.py --exchange bybit --market swap --symbols BTC/USDT --timeframe 4h

# Control caching and output
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --no-cache --verbose

# Show storage summary
python backtester/scripts/fetch_data_cli.py --storage-summary
```

#### Complete Command Reference

```bash
python backtester/scripts/fetch_data_cli.py [OPTIONS]

Symbol Selection (choose one):
  --symbols SYMBOLS     Comma-separated symbol list (e.g., "BTC/USDT,ETH/USDT")
  --top TOP            Fetch top N symbols by volume (default: 10)

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
```

### Shortcut Script (`fetch`)

A bash alias for the main data fetcher:

```bash
# Make executable (if needed)
chmod +x backtester/scripts/fetch

# Use as shortcut
./backtester/scripts/fetch --symbols BTC/USDT --timeframe 1d
```

---

## Cache Inspection

### Cache Inspector (`inspect_cache_cli.py`)

Comprehensive tool for examining cached data, validation, and analysis.

#### Basic Commands

```bash
# List all cached data files
python backtester/scripts/inspect_cache_cli.py list

# Comprehensive analysis of all symbols (default behavior)
python backtester/scripts/inspect_cache_cli.py inspect --exchange binance --market spot --timeframe 1h

# Show recent candle data for specific symbols only
python backtester/scripts/inspect_cache_cli.py inspect --exchange binance --market spot --timeframe 1d --symbols BTC/USDT ETH/USDT --tail 10

# Validate cache integrity
python backtester/scripts/inspect_cache_cli.py validate --exchange binance --market spot --timeframe 1h
```

#### Command Reference

```bash
python backtester/scripts/inspect_cache_cli.py COMMAND [OPTIONS]

Commands:
  list        List all cached data files
  inspect     Inspect cache contents with comprehensive analysis (default) or specific symbol data
  validate    Validate cache integrity

Options for inspect:
  --exchange EXCHANGE   Exchange ID (required)
  --market MARKET      Market type (default: spot)
  --timeframe TF       Timeframe (required)
  --symbols SYMBOLS    Show recent candle data for specific symbols (optional - without this, shows comprehensive analysis for all symbols)
  --tail N            Number of recent candles to show when using --symbols (default: 5)

Options for validate:
  --exchange EXCHANGE   Exchange ID (required)
  --market MARKET      Market type (default: spot)
  --timeframe TF       Timeframe (required)
```

**Note:** The `inspect` command behavior depends on whether you specify symbols:
- **Without `--symbols`**: Shows comprehensive analysis for all symbols with inception checks, gap analysis, completeness metrics, and health scores
- **With `--symbols`**: Shows recent candle data (OHLCV tails) for the specified symbols only

---

## Health Checks

### Main Health Check Tool (`data_healthcheck.py`)

Streamlined data quality analysis with automatic fixing capabilities.

#### Basic Usage

```bash
# Basic health check for all data
python -m backtester.data.health_check.data_healthcheck

# Check specific exchange/timeframe
python -m backtester.data.health_check.data_healthcheck --exchange binance --timeframe 1h

# Auto-fix critical issues
python -m backtester.data.health_check.data_healthcheck --auto-fix

# Detailed analysis with interpolation
python -m backtester.data.health_check.data_healthcheck --detailed --interpolate --auto-fix
```

#### Advanced Options

```bash
# Critical issues only
python -m backtester.data.health_check.data_healthcheck --critical-only

# Enable interpolation with specific strategy
python -m backtester.data.health_check.data_healthcheck --interpolate --interpolation-strategy linear --auto-fix

# Save report to file
python -m backtester.data.health_check.data_healthcheck --save-report --detailed
```

#### Command Reference

```bash
python -m backtester.data.health_check.data_healthcheck [OPTIONS]

Filtering:
  --exchange EXCHANGE           Check specific exchange only
  --timeframe TF               Check specific timeframe only
  --market {spot,swap}         Check specific market type

Analysis Mode:
  --detailed                   Show detailed analysis
  --critical-only              Show only critical issues
  --save-report                Save report to file

Auto-fixing:
  --auto-fix                   Automatically fix critical issues
  --interpolate                Enable interpolation to fill minor data gaps
  --interpolation-strategy STR Strategy: financial_forward_fill, linear, time_aware
```

### Health Check Wrapper (`health_check.py`)

Simple wrapper script in the scripts directory:

```bash
# Quick health check
python backtester/scripts/health_check.py

# Auto-fix with arguments passed through
python backtester/scripts/health_check.py --auto-fix --exchange binance --detailed
```

---

## Data Management

### Data Interpolation (`interpolate_data.py`)

Fill missing data points in cached cryptocurrency data.

#### Basic Usage

```bash
# Interpolate Binance 5m data with financial strategy
python backtester/scripts/interpolate_data.py --exchange binance --timeframe 5m

# Use linear interpolation for specific market
python backtester/scripts/interpolate_data.py --exchange binance --market spot --strategy linear

# Dry run to see what would be interpolated
python backtester/scripts/interpolate_data.py --dry-run --exchange binance --timeframe 1h
```

#### Command Reference

```bash
python backtester/scripts/interpolate_data.py [OPTIONS]

Required:
  --exchange EXCHANGE          Exchange to interpolate (required)

Optional:
  --market {spot,swap}         Market type (default: all available)
  --timeframe TF               Timeframe to interpolate (default: all available)
  --strategy STRATEGY          Interpolation strategy (default: financial_forward_fill)
  --max-gap N                  Maximum gap size to interpolate (default: 10000)
  --backup                     Create backup before interpolation
  --dry-run                    Show what would be interpolated without changes
  --force                      Interpolate even if no significant gaps found

Strategies:
  financial_forward_fill       OHLC uses last known close, Volume=0 (recommended)
  linear                      Linear interpolation between known points
  time_aware                  Time-weighted interpolation considering gap duration
```

### OHLCV Data Refresh (`refresh_ohlcv_data.py`)

Refresh cached data that has incomplete OHLCV structure.

```bash
# Refresh incomplete OHLCV data
python backtester/scripts/refresh_ohlcv_data.py
```

This script automatically:
- Checks existing 4h data for OHLCV completeness
- Creates backup of existing data
- Fetches fresh OHLCV data
- Tests storage resampling functionality

---

## Exchange Information

### Exchange Info Utility (`exchange_info.py`)

Get information about supported exchanges and their capabilities.

#### Basic Usage

```bash
# List all available exchanges
python backtester/utilities/exchange_info.py --list

# Get information about specific exchange
python backtester/utilities/exchange_info.py --info binance

# List timeframes for an exchange
python backtester/utilities/exchange_info.py --timeframes bybit

# Show data fetching example
python backtester/utilities/exchange_info.py --example binance

# Test exchange connectivity
python backtester/utilities/exchange_info.py --test binance
```

### Exchange Config (`exchange_config.py`)

Direct access to exchange configuration:

```bash
# List exchanges
python backtester/data/exchange_config.py --list

# Exchange info
python backtester/data/exchange_config.py --info binance --timeframes binance
```

---

## Utility Scripts

### Test Script (`test_stale_update.py`)

Test script for stale update functionality:

```bash
python test_stale_update.py
```

---

## Common Usage Patterns

### Getting Started

```bash
# 1. Check available exchanges
python backtester/utilities/exchange_info.py --list

# 2. Fetch some initial data
python backtester/scripts/fetch_data_cli.py --symbols "BTC/USDT,ETH/USDT" --timeframe 1d --inception

# 3. Inspect what was cached
python backtester/scripts/inspect_cache_cli.py list

# 4. Run health check
python backtester/scripts/health_check.py
```

### Daily Data Update

```bash
# Update top symbols
python backtester/scripts/fetch_data_cli.py --top 20 --timeframe 1d

# Health check after update
python backtester/scripts/health_check.py --auto-fix

# Inspect updated data
python backtester/scripts/inspect_cache_cli.py analyze --exchange binance --timeframe 1d
```

### Data Quality Maintenance

```bash
# Comprehensive health check
python -m backtester.data.health_check.data_healthcheck --detailed --auto-fix --interpolate

# Validate specific exchange data
python backtester/scripts/inspect_cache_cli.py validate --exchange binance --timeframe 1h

# Fix data gaps with interpolation
python backtester/scripts/interpolate_data.py --exchange binance --timeframe 5m --backup
```

### Troubleshooting

```bash
# Check storage status
python backtester/scripts/fetch_data_cli.py --storage-summary

# Force fresh data fetch
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --no-cache

# Detailed error analysis
python backtester/scripts/fetch_data_cli.py --symbols BTC/USDT --timeframe 1h --verbose

# Refresh OHLCV structure
python backtester/scripts/refresh_ohlcv_data.py
```

---

## Environment Setup

All scripts should be run from the project root directory:

```bash
cd /path/to/backtester
python backtester/scripts/[script_name].py [options]
```

Make sure the virtual environment is activated:

```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

---

## Notes

- **VectorBT Pro Integration**: All scripts use vectorbtpro as the primary library
- **No API Keys Required**: Data fetching works without API keys for public data
- **Caching**: Most operations use intelligent caching to improve performance
- **Error Handling**: Scripts include comprehensive error handling and user feedback
- **Rich Output**: Many scripts use rich console formatting for better readability

For detailed information about specific commands, use the `--help` flag with any script:

```bash
python backtester/scripts/[script_name].py --help
``` 