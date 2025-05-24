# VBT Data Health Check System

## Overview

The streamlined health check system provides focused data quality analysis for VBT cached cryptocurrency data with realistic gap resolution strategies. This system integrates with the refactored codebase and leverages the new CLI tools for automated fixes.

## Key Features

### 🔍 **Critical Issue Detection**
- **Data Gaps**: Identifies missing periods with >1000 missing data points as critical
- **Stale Data**: Flags data older than 2x the expected timeframe threshold as critical
- **Load Errors**: Detects corrupted or inaccessible cache files
- **Cache Integrity**: Validates cache system health and consistency

### 🔧 **Realistic Gap Resolution**
Unlike theoretical gap filling, our approach focuses on practical, actionable solutions:

1. **Data Fetching**: Uses existing CLI tools to fetch missing data from exchanges
2. **Cache Updates**: Refreshes stale volume caches automatically
3. **Smart Interpolation**: Fills minor gaps with financial data-appropriate strategies
4. **Prioritized Fixes**: Focuses on critical issues first, with configurable thresholds

### 📊 **Smart Interpolation System**
New interpolation capabilities for filling minor data gaps:

- **Financial Forward Fill** (Recommended): OHLC uses last known close, Volume=0
- **Linear Interpolation**: Straight-line interpolation between known points  
- **Time-Aware Interpolation**: Considers time gaps for weighted interpolation
- **Safety Limits**: Maximum gap size of 10,000 periods (~7 days of 1m data)
- **OHLCV Consistency**: Maintains proper relationships between price/volume data

### 🎯 **Dynamic Thresholds**
Gap severity is determined by data context:
- **Multi-year data**: More tolerant of natural market gaps
- **Shorter timeframes**: Stricter gap detection for recent data
- **Market-aware**: Accounts for normal crypto market trading patterns

### 🚀 **Auto-Fix Capabilities**
- **Critical Gaps**: Automatically re-fetch data from inception for severely incomplete symbols
- **Stale Data**: Update recent data to current time for time-sensitive timeframes
- **Cache Refresh**: Trigger volume cache updates when cache is stale
- **Command Validation**: Verify fix commands complete successfully before marking as resolved

## Usage Examples

### Basic Health Check
```bash
# Quick overview of all data health
python -m backtester.data.health_check.data_healthcheck

# Focus on critical issues only
python -m backtester.data.health_check.data_healthcheck --critical-only

# Check specific exchange with details
python -m backtester.data.health_check.data_healthcheck --exchange binance --detailed
```

### Auto-Fix with Standard Resolution
```bash
# Automatically fix critical issues (fetch missing data, update caches)
python -m backtester.data.health_check.data_healthcheck --auto-fix

# Fix issues for specific exchange/timeframe
python -m backtester.data.health_check.data_healthcheck --exchange binance --timeframe 5m --auto-fix
```

### Smart Interpolation for Gap Filling
```bash
# Enable interpolation for minor gaps (financial strategy)
python -m backtester.data.health_check.data_healthcheck --interpolate --auto-fix

# Use linear interpolation instead
python -m backtester.data.health_check.data_healthcheck --interpolate --interpolation-strategy linear --auto-fix

# Interpolate specific exchange data
python -m backtester.data.health_check.data_healthcheck --exchange binance --interpolate --auto-fix
```

### Manual Interpolation (Standalone)
```bash
# Dry run to see what would be interpolated
python backtester/scripts/interpolate_data.py --exchange binance --dry-run

# Interpolate Binance 5m data with backup
python backtester/scripts/interpolate_data.py --exchange binance --timeframe 5m --backup

# Force interpolation of even minor gaps
python backtester/scripts/interpolate_data.py --exchange binance --force --strategy linear
```

## Gap Resolution Strategies

### 1. **Critical Gap Classification** 
- **Critical**: >1000 missing periods (>16 hours for 1m data, >41 days for 1h data)
- **Minor**: 100-1000 missing periods (manageable gaps)
- **Acceptable**: <100 missing periods (normal market behavior)

### 2. **Fix Command Generation**
```bash
# Example: Fill critical gaps from inception
python backtester/scripts/fetch_data_cli.py --exchange binance --market spot --timeframe 1h --symbols BTC/USDT,ETH/USDT --inception

# Example: Update stale data to current time
python backtester/scripts/fetch_data_cli.py --exchange binance --market spot --timeframe 1h --symbols BTC/USDT,ETH/USDT --end now
```

### 3. **Staleness Thresholds**
- **1m-30m**: 2-24 hours (high-frequency trading data)
- **1h-12h**: 6-72 hours (medium-frequency analysis)
- **1d-1w**: 48-168 hours (long-term analysis)

## Integration with CLI Tools

### Cache Inspection
```bash
# View comprehensive cache analysis (complements health check)
python backtester/scripts/inspect_cache_cli.py
```

### Data Fetching
```bash
# Fetch top symbols by volume
python backtester/scripts/fetch_data_cli.py --exchange binance --top 10

# Fetch specific symbols from inception
python backtester/scripts/fetch_data_cli.py --exchange binance --symbols BTC/USDT,ETH/USDT --inception
```

## Architecture Changes

### Removed Redundancies
- **Eliminated duplicate cache inspection** (now handled by `inspect_cache_cli.py`)
- **Removed complex inception date logic** (delegated to cache system)
- **Simplified OHLCV validation** (focuses on critical data integrity only)
- **Streamlined reporting** (actionable insights over comprehensive details)

### Enhanced Focus Areas
- **Gap criticality assessment** using realistic thresholds
- **Fix command generation** using available CLI tools
- **Batch processing limits** to prevent API abuse
- **Command success validation** with proper error handling

## Output and Reporting

### Console Output
- **Real-time progress** with emoji indicators
- **Fix application status** with success/failure tracking
- **Actionable recommendations** with copy-paste commands
- **Exit codes** for automation integration (0=good, 1=warnings, 2=critical)

### Report Generation
- **Timestamped reports** in `reports/` directory
- **Summary statistics** with health scores
- **Quick command reference** for common fixes
- **Issue categorization** with severity levels

## Best Practices

### 1. **Regular Health Monitoring**
```bash
# Daily quick check
python -m backtester.data.health_check.data_healthcheck --critical-only

# Weekly comprehensive analysis
python -m backtester.data.health_check.data_healthcheck --detailed --save-report
```

### 2. **Gradual Gap Resolution**
- Start with critical gaps on high-priority symbols (BTC/USDT, ETH/USDT)
- Process 5-10 symbols at a time to avoid API rate limits
- Use inception fetching for symbols with >50% missing data
- Use targeted date ranges for smaller gaps

### 3. **Automated Workflows**
```bash
# Morning data health check with auto-fix
python -m backtester.data.health_check.data_healthcheck --auto-fix --critical-only

# Evening comprehensive update
python backtester/scripts/fetch_data_cli.py --exchange binance --top 20
```

## Error Handling

- **Graceful degradation** when data files are corrupted
- **Command validation** before applying fixes
- **Progress tracking** for multi-command fixes
- **Rollback guidance** when fixes fail

## Performance Considerations

- **Efficient gap detection** using vectorized operations
- **Lazy data loading** to minimize memory usage
- **Parallel-friendly design** for future enhancement
- **Resource limits** to prevent system overload

This streamlined approach ensures that health checks are actionable, realistic, and integrate seamlessly with the existing data management infrastructure. 