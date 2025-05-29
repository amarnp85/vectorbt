# Configuration System Guide

This comprehensive guide consolidates all documentation for the backtesting framework's configuration system, including strategy parameters, configuration loading, validation, optimization support, and system status.

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Strategy Parameters vs Database](#strategy-parameters-vs-database)
4. [Directory Structure](#directory-structure)
5. [Configuration Loading & Management](#configuration-loading--management)
6. [Database-Based Parameter Storage](#database-based-parameter-storage)
7. [Strategy Configuration](#strategy-configuration)
8. [VectorBTPro Integration](#vectorbtpro-integration)
9. [Multi-Symbol & MTF Support](#multi-symbol--mtf-support)
10. [Usage Examples](#usage-examples)
11. [Best Practices](#best-practices)
12. [System Status](#system-status)
13. [Migration History](#migration-history)
14. [Troubleshooting](#troubleshooting)

## Overview

The configuration system provides a comprehensive framework for managing strategy parameters, optimization settings, and portfolio configurations. It features:

- **Dual Storage System**: JSON-based templates + SQLite database for optimal parameters
- **Generic Strategy Support**: Works with any strategy type (trend following, mean reversion, momentum, etc.)
- **VectorBTPro Integration**: Native support for VBT parameter optimization
- **Multi-Symbol/MTF Support**: Configuration for complex multi-asset strategies
- **Database Parameter Storage**: SQLite-based storage for optimization results
- **Environment Variable Support**: Secure API key management
- **Validation Framework**: Comprehensive parameter and data validation

## System Architecture

### Core Components

1. **StrategyConfigLoader**: Base configuration loader with validation
2. **EnhancedConfigLoader**: Extended loader with VBT support and environment variables
3. **ConfigManager**: High-level interface for configuration management
4. **OptimalParametersDB**: Database-based storage and retrieval of optimization results

### Configuration Flow

```python
# 1. Load base strategy configuration (JSON template)
base_config = load_config("production/dma_atr_trend_params.json")

# 2. Query database for optimal parameters
optimal_params = db.get_optimal_params("BTC/USDT", "4h", "DMAATRTrendStrategy")

# 3. Merge optimal parameters with base config
if optimal_params:
    final_config = merge_with_optimal(base_config, optimal_params)
else:
    final_config = base_config  # Use defaults

# 4. Run strategy with final configuration
strategy = DMAATRTrendStrategy(data, final_config)
```

## Strategy Parameters vs Database

### Purpose of Strategy Parameters (`strategy_params/`)

The JSON configuration files serve multiple critical purposes:

1. **Base Templates**: Define strategy structure, parameter ranges, validation rules
2. **Default Values**: Provide fallback when no optimal parameters exist in database
3. **Manual Override**: Allow developers to manually edit parameters for testing
4. **Documentation**: Serve as living documentation of strategy requirements
5. **Development Environments**: Different configs for production, development, testing

### Database Integration

The SQLite database stores **optimization results** and **optimal parameters**:

- **Scalable Storage**: Handles thousands of optimization results
- **Query Capabilities**: Filter by symbol, timeframe, strategy, performance
- **History Tracking**: Complete optimization history with timestamps
- **Type Safety**: Proper handling of parameter types

### How They Work Together

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ JSON Templates  │    │ Optimal Params   │    │ Final Config    │
│ (strategy_params)│ +  │ (Database)       │ =  │ (Merged)        │
│                 │    │                  │    │                 │
│ • Structure     │    │ • Optimized vals │    │ • Best of both  │
│ • Defaults      │    │ • Performance    │    │ • Ready to use  │
│ • Validation    │    │ • History        │    │ • Type-safe     │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

**Benefits of Dual System:**
- **Reliability**: Always has working defaults
- **Flexibility**: Can override any parameter manually
- **Performance**: Database provides optimized parameters when available
- **Development**: Easy testing with different parameter sets

## Directory Structure

```
backtester/config/
├── __init__.py                    # Module exports
├── config_loader.py               # Base and enhanced configuration loaders
├── config_manager.py              # High-level configuration management
├── optimal_parameters_db.py       # Database-based parameter storage
├── CONFIGURATION_GUIDE.md         # This comprehensive guide
├── optimal_parameters/            # Database and backup storage
│   ├── optimal_parameters.db      # SQLite database for parameters
│   └── backups/                   # JSON backups and exports
└── strategy_params/               # Strategy configuration files
    ├── production/                # Production-ready configurations
    │   ├── dma_atr_trend_params.json
    │   ├── mean_reversion_params.json
    │   ├── momentum_params.json
    │   ├── pairs_trading_params.json
    │   └── mtf_dma_atr_params.json
    ├── development/               # Development/testing configurations
    │   ├── dma_atr_trend_fast.json
    │   └── dma_atr_trend_minimal.json
    ├── templates/                 # Configuration templates
    │   └── strategy_template.json
    ├── examples/                  # Example configurations
    │   ├── rsi_mean_reversion_params.json
    │   └── multi_symbol_momentum_params.json
    └── archived/                  # Deprecated configurations
```

## Configuration Loading & Management

### Basic Loading

```python
from backtester.config import load_strategy_config

# Load configuration
config = load_strategy_config('production/mean_reversion_params.json')

# Access parameters
bb_period = config['technical_parameters']['bb_period']
stop_loss = config['risk_management']['sl_atr_multiplier']
```

### Using ConfigManager with Database Integration

```python
from backtester.config import ConfigManager

# Initialize manager
config_manager = ConfigManager()

# Load configuration with optimal parameters from database
config = config_manager.load_with_optimal_params(
    "production/dma_atr_trend_params.json",
    symbol="BTC/USDT",
    timeframe="4h"
)

# Extract different parameter sections
strategy_params = config_manager.get_strategy_params()
portfolio_params = config_manager.get_portfolio_params()
optimization_params = config_manager.get_optimization_params()

# Validate configuration
errors = config_manager.validate_config(MomentumStrategy)
```

### Environment Variable Support

The system supports environment variable substitution:

```json
{
  "portfolio_parameters": {
    "init_cash": "${INIT_CASH:100000}",
    "fees": "${TRADING_FEE:0.001}"
  }
}
```

## Database-Based Parameter Storage

### OptimalParametersDB Features

The database system provides:

- **Scalable Storage**: SQLite database handles thousands of optimization results
- **Query Capabilities**: Easy filtering and searching of results
- **History Tracking**: Complete optimization history with timestamps
- **Performance Metrics**: Comprehensive performance tracking
- **Type Safety**: Proper handling of parameter types

### Storing Optimization Results

```python
from backtester.config import OptimalParametersDB

param_db = OptimalParametersDB()

# Store optimization results
param_db.store_optimization_result(
    symbol="BTC/USDT",
    timeframe="4h",
    strategy_name="DMAATRTrendStrategy",
    best_params={'fast_window': 15, 'slow_window': 50},
    performance_metrics={'sharpe_ratio': 1.5, 'total_return': 0.25},
    parameter_ranges={'fast_window': [10, 15, 20], 'slow_window': [40, 50, 60]},
    optimization_stats={'total_combinations': 9, 'valid_combinations': 8}
)
```

### Loading Optimal Parameters

```python
# Get optimal parameters
optimal_params = param_db.get_optimal_params("BTC/USDT", "4h", "DMAATRTrendStrategy")

# Get performance metrics
performance = param_db.get_performance_metrics("BTC/USDT", "4h", "DMAATRTrendStrategy")

# Get complete summary
summary = param_db.get_optimization_summary("BTC/USDT", "4h", "DMAATRTrendStrategy")

# List all optimized combinations
results_df = param_db.list_optimized_combinations()

# Get top performers
top_performers = param_db.get_best_performers(limit=10)
```

### Database Management

- **Location**: `backtester/config/optimal_parameters/optimal_parameters.db`
- **Backups**: Use `param_db.export_to_csv()` for backup and analysis
- **Export**: Use `param_db.export_to_csv()` for backup

## Strategy Configuration

### Configuration Structure

Each strategy configuration follows a standardized structure:

```json
{
  "strategy_name": "Human-readable strategy name",
  "strategy_class": "PythonClassName",
  "description": "Brief description of the strategy",
  "version": "1.0.0",
  "author": "Strategy author",
  "category": "trend_following|mean_reversion|momentum|arbitrage",
  
  "technical_parameters": {
    // Strategy-specific technical indicators
  },
  
  "risk_management": {
    // Risk and position sizing parameters
  },
  
  "entry_conditions": {
    // Conditions for trade entry
  },
  
  "exit_conditions": {
    // Conditions for trade exit
  },
  
  "signal_processing": {
    // Signal generation and cleaning
  },
  
  "portfolio_parameters": {
    // Portfolio simulation settings
  },
  
  "optimization_ranges": {
    // Parameter ranges for optimization
  },
  
  "vbt_optimization": {
    // VectorBTPro-specific optimization format
  },
  
  "data_parameters": {
    // Data requirements and specifications
  },
  
  "performance_targets": {
    // Minimum performance thresholds
  },
  
  "validation_rules": {
    // Parameter and data validation rules
  },
  
  "metadata": {
    // Additional metadata
  }
}
```

### Available Strategy Configurations

#### Production Configurations

1. **DMA ATR Trend** (`dma_atr_trend_params.json`)
   - Dual moving average crossover with ATR-based stops
   - Conservative parameters (20/50/200 MAs)
   - Suitable for trending markets

2. **Mean Reversion** (`mean_reversion_params.json`)
   - Bollinger Bands and RSI based
   - Regime filters to avoid trending markets
   - ATR-based dynamic stops

3. **Momentum** (`momentum_params.json`)
   - Multiple momentum indicators (ROC, RSI, MACD)
   - ADX trend strength filter
   - Composite momentum scoring

4. **Pairs Trading** (`pairs_trading_params.json`)
   - Statistical arbitrage for cointegrated pairs
   - Z-score based signals
   - Automatic pair selection

5. **Multi-Timeframe DMA ATR** (`mtf_dma_atr_params.json`)
   - Multi-timeframe trend confirmation
   - Timeframe alignment scoring
   - Higher timeframe trend validation

#### Development Configurations

- **Fast Testing**: Shorter MA windows for more signals
- **Minimal**: Basic configuration for quick tests

## VectorBTPro Integration

### Standard Optimization Ranges

```json
{
  "optimization_ranges": {
    "bb_period": [15, 20, 25, 30],
    "bb_std": [1.5, 2.0, 2.5, 3.0],
    "rsi_period": [10, 14, 18, 21]
  }
}
```

### VBT-Specific Format

```json
{
  "vbt_optimization": {
    "bb_period": {"type": "range", "start": 15, "stop": 31, "step": 5},
    "rsi_period": {"type": "list", "values": [10, 14, 18, 21]},
    "bb_std": {"type": "linspace", "start": 1.5, "stop": 3.0, "num": 4}
  }
}
```

### Using VBT Parameters

```python
# Get VBT-formatted parameters
vbt_params = config_manager.get_vbt_params()

# Use with strategy
strategy = MomentumStrategy(data, vbt_params)
result = strategy.run_signal_generation()
```

## Multi-Symbol & MTF Support

### Multi-Symbol Configuration

```python
# Create multi-symbol config
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
symbol_overrides = {
    "BTC/USDT": {"momentum_threshold": 0.015},
    "ETH/USDT": {"momentum_threshold": 0.020}
}

multi_config = config_manager.create_multi_symbol_config(
    symbols, symbol_overrides
)
```

### Multi-Timeframe Configuration

```python
# Create MTF config
timeframes = ["1h", "4h", "1d"]
timeframe_weights = {"1h": 1.0, "4h": 1.5, "1d": 2.0}

mtf_config = config_manager.create_mtf_config(
    timeframes, 
    base_timeframe="1h",
    timeframe_weights=timeframe_weights
)
```

## Usage Examples

### Example 1: Basic Strategy with Database Parameters

```python
import vectorbtpro as vbt
from backtester.strategies import MeanReversionStrategy
from backtester.config import OptimalParametersDB

# Load data
data = vbt.YFData.fetch("BTC-USD", start="2023-01-01", end="2023-12-31")

# Get optimal parameters from database
param_db = OptimalParametersDB()
optimal_params = param_db.get_optimal_params("BTC/USDT", "1d", "MeanReversionStrategy")

if optimal_params:
    strategy = MeanReversionStrategy(data, optimal_params)
else:
    # Fallback to configuration file
    strategy = MeanReversionStrategy(data, "production/mean_reversion_params.json")

result = strategy.run_signal_generation()
```

### Example 2: Parameter Optimization with Database Storage

```python
# Load configuration with VBT parameters
config_manager = ConfigManager("production/momentum_params.json")
vbt_params = config_manager.get_vbt_params()

# Create strategy with optimization parameters
strategy = MomentumStrategy(data, vbt_params)
result = strategy.run_signal_generation()

# Store best results in database
param_db = OptimalParametersDB()
param_db.store_optimization_result(
    symbol="BTC/USDT",
    timeframe="1d",
    strategy_name="MomentumStrategy",
    best_params=best_params,
    performance_metrics=performance_metrics
)
```

### Example 3: Multi-Symbol Strategy with Database

```python
# Load multi-symbol data
symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
data = vbt.YFData.fetch(symbols, start="2023-01-01", end="2023-12-31")

# Get optimal parameters for each symbol
param_db = OptimalParametersDB()
symbol_params = {}

for symbol in symbols:
    optimal = param_db.get_optimal_params(symbol, "1d", "MomentumStrategy")
    if optimal:
        symbol_params[symbol] = optimal
    else:
        # Use default parameters
        config_manager = ConfigManager("examples/multi_symbol_momentum_params.json")
        symbol_params[symbol] = config_manager.get_strategy_params()

# Use with strategy
strategy = MultiSymbolMomentumStrategy(data, symbol_params)
```

## Best Practices

### 1. Configuration Organization

- **Production**: Thoroughly tested, conservative parameters
- **Development**: Fast iteration configs for testing
- **Templates**: Starting points for new strategies
- **Examples**: Reference implementations

### 2. Database Usage

- **Regular Optimization**: Keep parameters updated with recent market data
- **Backup Strategy**: Export database regularly to CSV
- **Performance Monitoring**: Track strategy performance over time
- **Version Control**: Use metadata to track parameter evolution

### 3. Parameter Selection

- Match MA windows to timeframe (shorter for intraday, longer for daily)
- Use 2:1 or 3:1 take-profit to stop-loss ratios
- Test parameters on multiple symbols
- Consider market conditions (trending vs ranging)

### 4. Validation Rules

Always define:
- Parameter constraints (e.g., `fast_window < slow_window`)
- Data quality checks
- Required indicators
- Performance targets

### 5. Version Control

- Track configuration changes in git
- Document parameter modifications
- Keep backups of working configurations
- Use meaningful commit messages

## System Status

### ✅ Completed Features

#### 1. **Core Configuration System**
- ✅ `StrategyConfigLoader` - Base configuration loader with validation
- ✅ `EnhancedConfigLoader` - Extended loader with VBT support
- ✅ `ConfigManager` - High-level configuration management interface
- ✅ `OptimalParametersDB` - Database-based storage and retrieval of optimization results

#### 2. **Strategy Configurations**
- ✅ **Production Configs** (5 complete strategies):
  - `dma_atr_trend_params.json` - Dual MA with ATR stops
  - `mean_reversion_params.json` - Bollinger Bands + RSI
  - `momentum_params.json` - Multi-indicator momentum
  - `pairs_trading_params.json` - Statistical arbitrage
  - `mtf_dma_atr_params.json` - Multi-timeframe DMA

- ✅ **Development Configs**:
  - `dma_atr_trend_fast.json` - Fast testing config
  - `dma_atr_trend_minimal.json` - Minimal test config

- ✅ **Examples**:
  - `rsi_mean_reversion_params.json` - Example mean reversion
  - `multi_symbol_momentum_params.json` - Multi-symbol example

- ✅ **Templates**:
  - `strategy_template.json` - Comprehensive template

#### 3. **Database-Based Parameter Storage**
- ✅ SQLite database for scalable parameter storage (`optimal_parameters.db`)
- ✅ Complete optimization history tracking with timestamps
- ✅ Performance metrics storage and retrieval
- ✅ Type-safe parameter handling with automatic conversion
- ✅ Bulk operations for multiple symbol/timeframe combinations
- ✅ Export capabilities to CSV for backup and analysis
- ✅ Migration utility from old JSON-based system

#### 4. **Advanced Features**
- ✅ VectorBTPro parameter conversion (`to_vbt_params()`)
- ✅ Environment variable substitution (`${VAR:default}` syntax)
- ✅ Multi-symbol configuration builder
- ✅ Multi-timeframe configuration builder
- ✅ Strategy-specific validation
- ✅ Integration with BaseStrategy (accepts config files directly)
- ✅ Database parameter merging with configuration files
- ✅ Comprehensive validation framework

#### 5. **Integration with Examples**
- ✅ All examples updated to use database-based parameter storage
- ✅ Structured logging integration for clean output
- ✅ Automatic fallback to configuration files when no optimal parameters exist
- ✅ Consolidated runners (`backtest_runner.py`, `optimization_runner.py`)
- ✅ Parallel processing support for optimization and backtesting

### Current Database Status

```
Migration Status: COMPLETED
Total Parameters in Database: 3

Strategies:
  DMAATRTrendStrategy: 3 parameter sets

Symbols:
  BTC/USDT: 1 parameter sets
  ETH/USDT: 1 parameter sets  
  SOL/USDT: 1 parameter sets

Timeframes:
  1h: 1 parameter sets
  4h: 2 parameter sets
```

### Usage Patterns
```python
# Direct database usage
param_db = OptimalParametersDB()
optimal_params = param_db.get_optimal_params("BTC/USDT", "4h", "DMAATRTrendStrategy")

# Via ConfigManager with database integration
config_manager = ConfigManager()
config = config_manager.load_with_optimal_params(
    "production/dma_atr_trend_params.json",
    symbol="BTC/USDT",
    timeframe="4h"
)

# Storing optimization results
param_db.store_optimization_result(
    symbol="BTC/USDT",
    timeframe="4h",
    strategy_name="DMAATRTrendStrategy",
    best_params=best_params,
    performance_metrics=performance_metrics
)
```

## Migration History

### 🔄 Migration Completed (2025-05-28)

#### What Was Changed

1. **Removed Legacy JSON-Based System**
   - ✅ Deleted `backtester/config/optimal_parameters.py` (OptimalParameterManager)
   - ✅ Deleted `backtester/config/optimal_parameters/symbol_optimal_parameters.json`
   - ✅ Deleted `backtester/config/optimal_parameters/performance_history.json`
   - ✅ Updated all imports and references to use the new database system

2. **Enhanced Database System**
   - ✅ `OptimalParametersDB` now handles all parameter storage and retrieval
   - ✅ Added missing methods: `get_all_optimal_params()`, `list_optimized_combinations()`
   - ✅ Fixed table name references (using `optimization_results` table)
   - ✅ Improved type conversion for parameter values to prevent string vs int errors

3. **Updated Configuration Loading**
   - ✅ `ConfigManager` now uses `OptimalParametersDB` exclusively
   - ✅ `EnhancedConfigLoader.merge_with_optimal()` updated to use database
   - ✅ Improved type conversion in `_convert_value_type()` method
   - ✅ Better error handling and logging

4. **Fixed Critical Bugs**
   - ✅ **String vs Int Comparison Error**: Fixed parameter type conversion in database retrieval
   - ✅ **Cache Directory Creation**: Improved error handling in cache manager
   - ✅ **Parameter Merging**: Proper type conversion when merging optimal parameters

#### Key Improvements

1. **Type Safety**
   - All parameter values are properly converted to correct types
   - Prevents "string vs int comparison" errors
   - Robust handling of JSON-loaded numeric values

2. **Database Integration**
   - Single source of truth for optimal parameters
   - Atomic operations and data consistency
   - Better performance and scalability

3. **Error Handling**
   - Graceful fallback when optimal parameters not found
   - Improved logging and debugging information
   - Better error messages for troubleshooting

4. **Maintainability**
   - Cleaner codebase with removed legacy code
   - Consistent API across all configuration components
   - Better separation of concerns

## Troubleshooting

### Common Issues

1. **No Optimal Parameters Found**
   - Run optimization first using optimization_runner.py
   - Check database for existing results
   - Verify symbol/timeframe/strategy combination

2. **Database Connection Issues**
   - Check database file permissions
   - Verify disk space availability
   - Ensure database file isn't corrupted

3. **Configuration Not Found**
   - Check file path relative to config directory
   - Ensure .json extension
   - Verify file exists in correct subdirectory

4. **Validation Errors**
   - Review parameter constraints
   - Check required fields
   - Validate data format

5. **Optimization Issues**
   - Ensure VBT format is correct
   - Check parameter ranges are valid
   - Verify sufficient data for backtesting

### Debugging

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check database status
param_db = OptimalParametersDB()
print(f"Database location: {param_db.db_path}")
results_df = param_db.list_optimized_combinations()
print(f"Total optimized combinations: {len(results_df)}")

# Validate configuration
config_manager = ConfigManager("production/dma_atr_trend_params.json")
errors = config_manager.validate_config(StrategyClass)
if errors:
    for error in errors:
        print(f"Error: {error}")

# Check loaded parameters
print("Strategy params:", config_manager.get_strategy_params())
print("Portfolio params:", config_manager.get_portfolio_params())
```

### Database Management

- **Location**: `backtester/config/optimal_parameters/optimal_parameters.db`
- **Backups**: Use `param_db.export_to_csv()` for backup and analysis
- **Export**: Use `param_db.export_to_csv()` for backup

## Summary

The configuration module is **complete and production-ready** with modern database-based parameter storage. It provides a robust, flexible system for managing strategy parameters with excellent VectorBTPro integration and scalable optimization result storage.

**Key Benefits:**
- **Dual System**: JSON templates provide structure and defaults, database provides optimized parameters
- **Flexibility**: Can manually override any parameter via JSON files
- **Reliability**: Always has fallback defaults even if database is empty
- **Performance**: Fast parameter retrieval and storage operations
- **Scalability**: Handles thousands of optimization results efficiently
- **Type Safety**: Proper handling of parameter types prevents runtime errors

The system successfully balances flexibility for development with performance for production use, making it suitable for both manual parameter tuning and automated optimization workflows.

---

**Last Updated**: 2025-05-28  
**Status**: ✅ PRODUCTION READY  
**Migration**: ✅ COMPLETED 