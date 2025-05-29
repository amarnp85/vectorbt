# Backtester CLI Commands Reference

This document provides a comprehensive reference for all CLI tools available in the backtester project for managing cryptocurrency data, cache inspection, health checks, and data fetching operations.

## Table of Contents

1. [Data Operations](#data-operations)
2. [Signal Analysis](#signal-analysis)
3. [Testing & Examples](#testing--examples)
4. [Common Usage Patterns](#common-usage-patterns)

---

## Data Operations

All data-related CLI tools are now organized under the `backtester.data.cli` module.

### Data Fetching (`backtester.data.cli.fetch`)

The main tool for fetching cryptocurrency data from various exchanges.

#### Basic Usage

```bash
# Navigate to project root
cd /path/to/backtester

# Fetch top 10 symbols by volume (default)
python -m backtester.data.cli.fetch --exchange binance --timeframe 1d

# Fetch specific symbols
python -m backtester.data.cli.fetch --symbols "BTC/USDT,ETH/USDT,SOL/USDT" --timeframe 4h

# Fetch from inception (maximum history)
python -m backtester.data.cli.fetch --symbols "BTC/USDT,ETH/USDT" --timeframe 1h --inception

# Fetch with custom date range
python -m backtester.data.cli.fetch --symbols BTC/USDT --timeframe 1h --start "30 days ago" --end "now"
```

#### Advanced Options

```bash
# Different exchanges and markets
python -m backtester.data.cli.fetch --exchange bybit --market swap --symbols BTC/USDT --timeframe 4h

# Control caching and output
python -m backtester.data.cli.fetch --symbols BTC/USDT --timeframe 1h --no-cache --verbose

# Show storage summary
python -m backtester.data.cli.fetch --storage-summary
```

#### Complete Command Reference

```bash
python -m backtester.data.cli.fetch [OPTIONS]

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

### Cache Inspection (`backtester.data.cli.inspect`)

Comprehensive tool for examining cached data, validation, and analysis.

#### Basic Commands

```bash
# List all cached data files
python -m backtester.data.cli.inspect list

# Comprehensive analysis of all symbols (default behavior)
python -m backtester.data.cli.inspect inspect --exchange binance --market spot --timeframe 1h

# Show recent candle data for specific symbols only
python -m backtester.data.cli.inspect inspect --exchange binance --market spot --timeframe 1d --symbols BTC/USDT ETH/USDT --tail 10

# Validate cache integrity
python -m backtester.data.cli.inspect validate --exchange binance --market spot --timeframe 1h
```

#### Command Reference

```bash
python -m backtester.data.cli.inspect COMMAND [OPTIONS]

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

### Data Health Checks (`backtester.data.health_check.data_healthcheck`)

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

### Data Interpolation (`backtester.data.cli.interpolate`)

Fill missing data points in cached cryptocurrency data.

#### Basic Usage

```bash
# Interpolate Binance 5m data with financial strategy
python -m backtester.data.cli.interpolate --exchange binance --timeframe 5m

# Use linear interpolation for specific market
python -m backtester.data.cli.interpolate --exchange binance --market spot --strategy linear

# Dry run to see what would be interpolated
python -m backtester.data.cli.interpolate --dry-run --exchange binance --timeframe 1h
```

#### Command Reference

```bash
python -m backtester.data.cli.interpolate [OPTIONS]

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

### OHLCV Data Refresh (`backtester.data.cli.refresh`)

Refresh cached data that has incomplete OHLCV structure.

```bash
# Refresh incomplete OHLCV data (default: binance/4h/spot)
python -m backtester.data.cli.refresh

# Refresh specific exchange/timeframe/market
python -m backtester.data.cli.refresh --exchange binance --timeframe 4h --market spot

# Test resampling without refreshing
python -m backtester.data.cli.refresh --test-only
```

This script automatically:
- Checks existing data for OHLCV completeness
- Creates backup of existing data
- Fetches fresh OHLCV data
- Tests storage resampling functionality

---

## Signal Analysis

Signal-related CLI tools are organized under the `backtester.signals.cli` module.

### Signal Diagnostics (`backtester.signals.cli.diagnose`)

Comprehensive diagnostics for signal generation issues.

#### Basic Usage

```bash
# Diagnose signals for BTC/USDT on 1h timeframe
python -m backtester.signals.cli.diagnose --symbols "BTC/USDT" --timeframe "1h" --start "2023-01-01" --end "2023-02-01"

# Diagnose with custom config
python -m backtester.signals.cli.diagnose --symbols "ETH/USDT" --timeframe "4h" --config custom_params.json

# Diagnose multiple symbols
python -m backtester.signals.cli.diagnose --symbols "BTC/USDT,ETH/USDT" --timeframe "1d" --output custom_diagnostics
```

#### Command Reference

```bash
python -m backtester.signals.cli.diagnose [OPTIONS]

Required:
  --symbols SYMBOLS    Trading symbol(s), comma-separated for multiple

Optional:
  --timeframe TF       Data timeframe (default: 1h)
  --start DATE         Start date (YYYY-MM-DD)
  --end DATE           End date (YYYY-MM-DD)
  --config PATH        Path to strategy configuration file
  --output DIR         Output directory for diagnostic results (default: diagnostics_output)
```

This tool:
- Tests multiple parameter configurations
- Provides comparative analysis
- Generates specific recommendations
- Saves detailed results and recommendations

---

## Testing & Examples

### Comprehensive Tests (`tests/test_main_runner.py`)

Test script for validating BacktestRunner functionality.

```bash
# Run all tests
python tests/test_main_runner.py

# Run from project root
cd /path/to/backtester && python tests/test_main_runner.py
```

Tests include:
- Basic functionality
- Single backtests
- Parameter optimization
- Configuration-based runs
- Multi-symbol backtests
- Command-line interface
- Error handling

### Usage Examples (`examples/main_runner_example.py`)

Demonstration script showing various BacktestRunner usage patterns.

```bash
# Run all examples
python examples/main_runner_example.py

# Run from project root
cd /path/to/backtester && python examples/main_runner_example.py
```

Examples include:
- Basic single backtest
- Parameter optimization
- Configuration-based backtest
- Multi-symbol backtest

---

## Common Usage Patterns

### Getting Started

```bash
# 1. Check available exchanges
python -m backtester.utilities.exchange_info --list

# 2. Fetch some initial data
python -m backtester.data.cli.fetch --symbols "BTC/USDT,ETH/USDT" --timeframe 1d --inception

# 3. Inspect what was cached
python -m backtester.data.cli.inspect list

# 4. Run health check
python -m backtester.data.health_check.data_healthcheck
```

### Daily Data Update

```bash
# Update top symbols
python -m backtester.data.cli.fetch --top 20 --timeframe 1d

# Health check after update
python -m backtester.data.health_check.data_healthcheck --auto-fix

# Inspect updated data
python -m backtester.data.cli.inspect inspect --exchange binance --timeframe 1d
```

### Data Quality Maintenance

```bash
# Comprehensive health check
python -m backtester.data.health_check.data_healthcheck --detailed --auto-fix --interpolate

# Validate specific exchange data
python -m backtester.data.cli.inspect validate --exchange binance --timeframe 1h

# Fix data gaps with interpolation
python -m backtester.data.cli.interpolate --exchange binance --timeframe 5m --backup
```

### Troubleshooting

```bash
# Check storage status
python -m backtester.data.cli.fetch --storage-summary

# Force fresh data fetch
python -m backtester.data.cli.fetch --symbols BTC/USDT --timeframe 1h --no-cache

# Detailed error analysis
python -m backtester.data.cli.fetch --symbols BTC/USDT --timeframe 1h --verbose

# Refresh OHLCV structure
python -m backtester.data.cli.refresh

# Diagnose signal generation issues
python -m backtester.signals.cli.diagnose --symbols "BTC/USDT" --timeframe "1h" --start "2024-01-01" --end "2024-02-01"
```

---

## Environment Setup

All scripts should be run from the project root directory:

```bash
cd /path/to/backtester
python -m backtester.module.cli.script [options]
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
- **Module Structure**: CLI tools are organized by functionality (data, signals, etc.)

For detailed information about specific commands, use the `--help` flag with any script:

```bash
python -m backtester.module.cli.script --help
``` 