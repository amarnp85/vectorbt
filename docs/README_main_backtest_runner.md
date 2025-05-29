# Main Backtest Runner - Comprehensive Guide

## Table of Contents
1. [Overview](#overview)
2. [Core Functions](#core-functions)
3. [Command Line Interface](#command-line-interface)
4. [Strategy Configuration](#strategy-configuration)
5. [Parameter Monitoring & Optimization](#parameter-monitoring--optimization)
6. [Available Strategies](#available-strategies)
7. [Performance Analysis](#performance-analysis)
8. [Advanced Usage](#advanced-usage)
9. [Troubleshooting](#troubleshooting)

## Overview

The `main_backtest_runner.py` is the central orchestrator for the backtesting framework. It integrates all components including data fetching, strategy execution, portfolio simulation, performance analysis, and visualization while maximizing VectorBTPro's built-in functionality.

### Key Features
- **Unified Interface**: Single entry point for all backtesting operations
- **VectorBTPro Integration**: Leverages VBT's optimized data structures and calculations
- **Intelligent Caching**: UTC timestamp-based caching for lightning-fast iterations
- **Enhanced Logging**: Rich console output with performance tracking
- **Comprehensive Analysis**: Automated generation of metrics, plots, and reports
- **Parameter Optimization**: Grid search and random search optimization methods
- **Multi-Symbol Support**: Efficient handling of multiple trading pairs

## Core Functions

### BacktestRunner Class

The main class that orchestrates all backtesting operations:

```python
class BacktestRunner:
    def __init__(self, exchange_id: str = "binance", market_type: str = "spot")
```

#### Primary Methods

##### 1. `load_data()`
```python
def load_data(
    self,
    symbols: Union[str, List[str]],
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    prefer_resampling: bool = True,
) -> Optional[vbt.Data]
```

**Purpose**: Load and prepare market data for backtesting
**Returns**: VectorBTPro Data object with OHLCV data
**Features**:
- Automatic symbol validation and filtering
- Intelligent caching system (UTC-based daily refresh)
- Date range filtering with timezone handling
- Resampling support for different timeframes

##### 2. `run_single_backtest()`
```python
def run_single_backtest(
    self,
    data: vbt.Data,
    strategy_class: type,
    strategy_params: Dict[str, Any],
    portfolio_params: Optional[Dict[str, Any]] = None,
    output_dir: str = "results",
) -> Dict[str, Any]
```

**Purpose**: Execute a complete backtest with a single parameter set
**Returns**: Comprehensive results dictionary
**Process**:
1. Strategy initialization and indicator calculation
2. Signal generation using strategy logic
3. Portfolio simulation with realistic trading costs
4. Performance analysis and metric calculation
5. Plot generation and file export

**Generated Outputs**:
- `performance_metrics.csv` - Key performance indicators
- `overview.html` - Interactive portfolio dashboard
- `equity.html` - Equity curve analysis
- `drawdowns.html` - Drawdown visualization
- `trades.html` - Trade analysis plots
- `indicators.html` - Strategy indicator plots
- `monthly_returns.html` - Monthly returns heatmap

##### 3. `run_optimization()`
```python
def run_optimization(
    self,
    data: vbt.Data,
    strategy_class: type,
    base_params: Dict[str, Any],
    optimization_params: Dict[str, Any],
    metric: str = "total_return",
    method: str = "grid",
    output_dir: str = "results",
) -> Dict[str, Any]
```

**Purpose**: Perform parameter optimization using grid or random search
**Returns**: Optimization results with best parameters and backtest
**Features**:
- Grid search: Tests all parameter combinations
- Random search: Samples parameter space efficiently
- Parallel processing support
- Automatic best parameter selection
- Comprehensive optimization plots

**Generated Outputs**:
- `optimization_results.csv` - All parameter combinations tested
- `best_params.json` - Optimal parameter set
- `optimization_distribution.html` - Parameter distribution plots
- `optimization_heatmap.html` - 2D parameter heatmaps
- `optimization_convergence.html` - Convergence analysis
- `best_backtest/` - Full backtest with optimal parameters

##### 4. `run_from_config()`
```python
def run_from_config(
    self,
    config_path: str,
    symbols: Union[str, List[str]],
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    optimize: bool = False,
    metric: str = "total_return",
    output_dir: str = "results",
) -> Dict[str, Any]
```

**Purpose**: Run backtest or optimization from JSON configuration file
**Features**:
- Automatic parameter mapping from config to strategy
- Support for nested parameter structures
- Validation of parameter constraints
- Flexible configuration format

## Command Line Interface

### Basic Usage

```bash
python backtester/main_backtest_runner.py [OPTIONS]
```

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--symbols` | Trading symbols (comma-separated) | `"BTC/USDT,ETH/USDT"` |

### Data Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--timeframe` | `"1d"` | Data timeframe (1m, 5m, 15m, 1h, 4h, 1d) |
| `--start` | `None` | Start date (YYYY-MM-DD) |
| `--end` | `None` | End date (YYYY-MM-DD) |
| `--exchange` | `"binance"` | Exchange to use |
| `--market` | `"spot"` | Market type (spot or swap) |

### Strategy Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `"dma_atr_trend_params.json"` | Strategy configuration file |
| `--strategy` | `"dma_atr_trend"` | Strategy type |

### Optimization Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimize` | `False` | Enable parameter optimization |
| `--metric` | `"total_return"` | Optimization target metric |
| `--method` | `"grid"` | Optimization method (grid/random) |

### Output Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output` | `"results"` | Output directory |
| `--no-cache` | `False` | Disable data caching |

### Debug Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--debug` | `False` | Enable debug logging |
| `--no-rich` | `False` | Disable rich formatting |

### Example Commands

#### Basic Backtest
```bash
python backtester/main_backtest_runner.py \
    --symbols "BTC/USDT" \
    --timeframe "1h" \
    --start "2023-01-01" \
    --end "2024-01-01" \
    --output "results/btc_test"
```

#### Multi-Symbol Backtest
```bash
python backtester/main_backtest_runner.py \
    --symbols "BTC/USDT,ETH/USDT,ADA/USDT" \
    --timeframe "4h" \
    --start "2023-01-01" \
    --end "2024-01-01" \
    --output "results/multi_symbol"
```

#### Parameter Optimization
```bash
python backtester/main_backtest_runner.py \
    --symbols "BTC/USDT" \
    --timeframe "1h" \
    --start "2023-01-01" \
    --end "2024-01-01" \
    --optimize \
    --metric "sharpe_ratio" \
    --method "grid" \
    --output "results/optimization"
```

## Strategy Configuration

### Configuration File Structure

Strategy parameters are defined in JSON files located in `backtester/config/strategy_params/`. The configuration follows a hierarchical structure:

```json
{
  "strategy_name": "Strategy Name",
  "strategy_class": "StrategyClassName",
  "description": "Strategy description",
  "version": "1.0.0",
  
  "technical_parameters": {
    "short_ma_window": 20,
    "long_ma_window": 50,
    "atr_period": 14
  },
  
  "risk_management": {
    "sl_atr_multiplier": 2.0,
    "tp_atr_multiplier": 4.0,
    "max_position_size": 1.0
  },
  
  "portfolio_parameters": {
    "init_cash": 100000,
    "fees": 0.001,
    "slippage": 0.0005
  },
  
  "optimization_ranges": {
    "short_ma_window": [10, 15, 20, 25, 30],
    "long_ma_window": [40, 50, 60, 70, 80]
  }
}
```

### Parameter Categories

#### Technical Parameters
Controls the strategy's technical indicators and signal generation:
- `short_ma_window`: Short moving average period
- `long_ma_window`: Long moving average period
- `trend_ma_window`: Trend filter moving average period
- `atr_period`: Average True Range calculation period
- `ma_type`: Moving average type ("SMA", "EMA", "WMA")

#### Risk Management
Defines risk control and position sizing:
- `sl_atr_multiplier`: Stop-loss distance as ATR multiple
- `tp_atr_multiplier`: Take-profit distance as ATR multiple
- `max_position_size`: Maximum position size (0.0-1.0)
- `risk_per_trade`: Risk per trade as portfolio fraction

#### Trend Confirmation
Optional trend filtering parameters:
- `use_adx_filter`: Enable ADX trend strength filter
- `adx_period`: ADX calculation period
- `adx_threshold`: Minimum ADX value for trend confirmation

#### Signal Processing
Controls signal generation and filtering:
- `clean_signals`: Remove conflicting signals
- `min_signal_gap`: Minimum bars between signals
- `signal_confirmation_bars`: Bars to wait for signal confirmation

#### Portfolio Parameters
Simulation settings:
- `init_cash`: Initial portfolio value
- `fees`: Trading fees (as decimal, e.g., 0.001 = 0.1%)
- `slippage`: Market impact (as decimal)
- `freq`: Data frequency for calculations

## Parameter Monitoring & Optimization

### Performance Metrics

The system tracks comprehensive performance metrics:

#### Returns Metrics
- `total_return`: Total portfolio return
- `annualized_return`: Annualized return
- `volatility`: Return volatility (annualized)
- `sharpe_ratio`: Risk-adjusted return measure
- `sortino_ratio`: Downside risk-adjusted return
- `calmar_ratio`: Return to max drawdown ratio

#### Trade Metrics
- `total_trades`: Number of completed trades
- `win_rate`: Percentage of profitable trades
- `profit_factor`: Gross profit / Gross loss
- `avg_trade_return`: Average return per trade
- `avg_trade_duration`: Average trade duration
- `expectancy`: Expected value per trade

#### Risk Metrics
- `max_drawdown`: Maximum portfolio decline
- `avg_drawdown`: Average drawdown
- `max_drawdown_duration`: Longest drawdown period
- `var_95`: Value at Risk (95% confidence)
- `cvar_95`: Conditional Value at Risk

### Optimization Metrics

Available optimization targets:

| Metric | Description | Optimize For |
|--------|-------------|--------------|
| `total_return` | Total portfolio return | Higher |
| `sharpe_ratio` | Risk-adjusted return | Higher |
| `sortino_ratio` | Downside risk-adjusted return | Higher |
| `calmar_ratio` | Return to max drawdown | Higher |
| `max_drawdown` | Maximum portfolio decline | Lower |
| `win_rate` | Percentage of winning trades | Higher |
| `profit_factor` | Gross profit / Gross loss | Higher |
| `expectancy` | Expected value per trade | Higher |

### Parameter Monitoring

#### Real-time Performance Tracking
The enhanced logging system provides real-time monitoring:

```
🚀 Starting Single Backtest
✅ Strategy initialized (0.12s)
✅ Indicators calculated (0.08s)
✅ Signals generated (0.15s)
✅ Portfolio simulation completed (0.25s)
✅ Performance analysis completed (0.10s)
✅ Plots generated (0.30s)
✅ Backtest completed in 1.00s

📊 Performance Summary:
   Total Return: 15.23%
   Sharpe Ratio: 1.85
   Max Drawdown: 8.56%
   Win Rate: 62.34%
   Total Trades: 45
```

#### Parameter Sensitivity Analysis
The optimization module provides parameter sensitivity insights:

```
🔍 Starting Grid Optimization
📈 Testing 125 parameter combinations
✅ Optimization completed (45.2s)

🏆 Best Parameters:
   short_ma_window: 15
   long_ma_window: 45
   sl_atr_multiplier: 2.5
   
📊 Best Performance:
   Sharpe Ratio: 2.14 (target metric)
   Total Return: 18.67%
   Max Drawdown: 6.23%
```

## Available Strategies

### DMA-ATR-Trend Strategy

**File**: `backtester/strategies/dma_atr_trend_strategy.py`
**Config**: `backtester/config/strategy_params/dma_atr_trend_params.json`

#### Strategy Logic
1. **Signal Generation**: Short MA crossing above/below Long MA
2. **Trend Filter**: Price must be above/below Trend MA for long/short signals
3. **Risk Management**: ATR-based stop-loss and take-profit levels
4. **Optional ADX Filter**: Trend strength confirmation

#### Key Parameters
```json
{
  "short_ma_window": 20,      // Short MA period
  "long_ma_window": 50,       // Long MA period  
  "trend_ma_window": 200,     // Trend filter period
  "atr_period": 14,           // ATR calculation period
  "sl_atr_multiplier": 2.0,   // Stop-loss ATR multiple
  "tp_atr_multiplier": 4.0,   // Take-profit ATR multiple
  "ma_type": "SMA",           // MA type (SMA/EMA/WMA)
  "use_adx": false,           // Enable ADX filter
  "adx_threshold": 25.0       // ADX threshold
}
```

#### Optimization Ranges
```json
{
  "short_ma_window": [10, 15, 20, 25, 30],
  "long_ma_window": [40, 50, 60, 70, 80],
  "atr_period": [10, 14, 20],
  "sl_atr_multiplier": [1.5, 2.0, 2.5, 3.0],
  "tp_atr_multiplier": [3.0, 4.0, 5.0, 6.0]
}
```

### Adding New Strategies

#### 1. Create Strategy Class
```python
# backtester/strategies/my_strategy.py
from .base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def init_indicators(self):
        # Calculate indicators
        pass
    
    def generate_signals(self):
        # Generate trading signals
        pass
```

#### 2. Create Configuration File
```json
// backtester/config/strategy_params/my_strategy_params.json
{
  "strategy_name": "My Strategy",
  "strategy_class": "MyStrategy",
  "technical_parameters": {
    // Strategy-specific parameters
  }
}
```

#### 3. Register Strategy
```python
# In main_backtest_runner.py
strategy_map = {
    "dma_atr_trend": DMAATRTrendStrategy,
    "my_strategy": MyStrategy,  // Add new strategy
}
```

## Performance Analysis

### Generated Reports

#### 1. Performance Metrics CSV
```csv
metric,value,description
total_return,0.1523,Total portfolio return (15.23%)
sharpe_ratio,1.85,Risk-adjusted return measure
max_drawdown,0.0856,Maximum portfolio decline (8.56%)
win_rate,0.6234,Percentage of winning trades (62.34%)
profit_factor,1.67,Gross profit / Gross loss
total_trades,45,Number of completed trades
avg_trade_duration,2.3,Average trade duration (days)
```

#### 2. Trade Details CSV
```csv
entry_time,exit_time,symbol,side,entry_price,exit_price,pnl,pnl_pct,duration
2023-01-15 10:00,2023-01-16 14:00,BTC/USDT,long,21500,22100,600,2.79%,1d 4h
2023-01-20 08:00,2023-01-21 12:00,BTC/USDT,short,22800,22200,600,2.63%,1d 4h
```

#### 3. Monthly Returns CSV
```csv
year,month,return,trades,win_rate,avg_return
2023,1,0.0234,8,0.625,0.0029
2023,2,0.0156,6,0.667,0.0026
2023,3,-0.0089,4,0.250,-0.0022
```

### Interactive Visualizations

#### 1. Portfolio Overview (`overview.html`)
- Equity curve with drawdown overlay
- Monthly returns heatmap
- Key metrics dashboard
- Trade distribution charts

#### 2. Equity Analysis (`equity.html`)
- Cumulative returns vs benchmark
- Rolling Sharpe ratio
- Underwater curve (drawdowns)
- Return distribution histogram

#### 3. Trade Analysis (`trades.html`)
- Trade P&L distribution
- Trade duration analysis
- Win/loss streak analysis
- Entry/exit timing patterns

#### 4. Indicator Analysis (`indicators.html`)
- Strategy indicators overlay on price
- Signal generation visualization
- Indicator correlation analysis

#### 5. Monthly Returns (`monthly_returns.html`)
- Monthly returns heatmap
- Seasonal performance patterns
- Year-over-year comparison

### Performance Evaluation Guidelines

#### Excellent Performance ✅
- Sharpe Ratio > 2.0
- Max Drawdown < 10%
- Win Rate > 60%
- Profit Factor > 1.5
- Consistent monthly returns

#### Acceptable Performance ⚠️
- Sharpe Ratio > 1.0
- Max Drawdown < 20%
- Win Rate > 50%
- Profit Factor > 1.2

#### Poor Performance ❌
- Sharpe Ratio < 0.5
- Max Drawdown > 30%
- Win Rate < 45%
- Profit Factor < 1.0

## Advanced Usage

### Custom Portfolio Parameters

```python
portfolio_params = {
    "init_cash": 100000,        # Initial capital
    "fees": 0.001,              # Trading fees (0.1%)
    "slippage": 0.0005,         # Market impact (0.05%)
    "freq": "1H",               # Data frequency
}

results = runner.run_single_backtest(
    data=data,
    strategy_class=DMAATRTrendStrategy,
    strategy_params=strategy_params,
    portfolio_params=portfolio_params,
    output_dir="results/custom_portfolio"
)
```

### Multi-Timeframe Analysis

```bash
# Test same strategy on different timeframes
for tf in 1h 4h 1d; do
    python backtester/main_backtest_runner.py \
        --symbols "BTC/USDT" \
        --timeframe "$tf" \
        --start "2023-01-01" \
        --end "2024-01-01" \
        --output "results/timeframe_${tf}"
done
```

### Batch Symbol Testing

```bash
# Test multiple symbol groups
symbols_groups=(
    "BTC/USDT,ETH/USDT"
    "ADA/USDT,SOL/USDT,MATIC/USDT"
    "LINK/USDT,DOT/USDT,AVAX/USDT"
)

for symbols in "${symbols_groups[@]}"; do
    python backtester/main_backtest_runner.py \
        --symbols "$symbols" \
        --timeframe "1h" \
        --start "2023-01-01" \
        --end "2024-01-01" \
        --output "results/batch_$(echo $symbols | tr '/' '_' | tr ',' '_')"
done
```

## Troubleshooting

### Common Issues

#### 1. No Trades Generated
**Symptoms**: `total_trades: 0` in results
**Causes**:
- Strategy parameters too restrictive
- Insufficient data for indicator calculation
- Date range too short

**Solutions**:
```bash
# Check with debug mode
python backtester/main_backtest_runner.py \
    --symbols "BTC/USDT" \
    --timeframe "1h" \
    --debug \
    --output "results/debug"

# Try more relaxed parameters
# Reduce MA windows, ATR multipliers
```

#### 2. Data Loading Errors
**Symptoms**: "Failed to load data" error
**Causes**:
- Invalid symbol names
- Network connectivity issues
- Exchange API limits

**Solutions**:
```bash
# Test with known good symbols
python backtester/main_backtest_runner.py \
    --symbols "BTC/USDT" \
    --timeframe "1d" \
    --no-cache

# Check symbol availability
python -c "
from backtester.data import fetch_top_symbols
# Get top 10 symbols by volume
top_symbols = fetch_top_symbols(
    exchange='binance',
    quote_currency='USDT',
    top_n=10
)
print(top_symbols)
"
```

#### 3. Performance Issues
**Symptoms**: Slow execution times
**Causes**:
- Cache not working
- Large parameter optimization
- Memory constraints

**Solutions**:
```bash
# Check cache status
ls -la backtester/data/cache_system/cache/

# Reduce optimization space
# Use random search instead of grid search
python backtester/main_backtest_runner.py \
    --optimize \
    --method "random" \
    --output "results/random_opt"
```

#### 4. Memory Issues
**Symptoms**: Out of memory errors
**Causes**:
- Too many symbols
- Long time periods
- Large optimization grids

**Solutions**:
```bash
# Reduce data size
python backtester/main_backtest_runner.py \
    --symbols "BTC/USDT" \
    --start "2023-06-01" \
    --end "2023-12-31" \
    --output "results/reduced"

# Use smaller optimization ranges
```

### Debug Mode

Enable comprehensive debugging:

```bash
python backtester/main_backtest_runner.py \
    --symbols "BTC/USDT" \
    --timeframe "1h" \
    --debug \
    --output "results/debug"
```

Debug mode provides:
- Detailed execution timing
- Parameter validation logs
- Signal generation statistics
- Memory usage tracking
- Error stack traces

### Log Analysis

Check log files for detailed information:
```bash
# View recent logs
tail -f backtester.log

# Search for errors
grep -i error backtester.log

# Check performance metrics
grep -i "performance" backtester.log
```

## Best Practices

### 1. Development Workflow
1. **Start Small**: Test with single symbol and short period
2. **Validate Logic**: Check signal generation with debug mode
3. **Scale Gradually**: Add more symbols and longer periods
4. **Optimize Carefully**: Use reasonable parameter ranges
5. **Validate Results**: Cross-check metrics and visualizations

### 2. Parameter Selection
- **MA Windows**: Ensure short < long < trend
- **ATR Multipliers**: Start with 2.0 SL, 4.0 TP
- **Optimization Ranges**: Use 3-7 values per parameter
- **Time Periods**: Use at least 1 year for reliable results

### 3. Performance Monitoring
- **Track Key Metrics**: Focus on Sharpe ratio and max drawdown
- **Monitor Trade Count**: Ensure sufficient trades for statistical significance
- **Check Consistency**: Look for stable performance across periods
- **Validate Assumptions**: Test on out-of-sample data

### 4. Resource Management
- **Use Caching**: Let the system cache data for faster iterations
- **Batch Operations**: Group similar tests together
- **Monitor Memory**: Watch memory usage with large datasets
- **Parallel Processing**: Use for large optimization tasks

---

## Quick Reference

### Essential Commands
```bash
# Basic backtest
python backtester/main_backtest_runner.py --symbols "BTC/USDT" --timeframe "1h"

# Multi-symbol test
python backtester/main_backtest_runner.py --symbols "BTC/USDT,ETH/USDT" --timeframe "4h"

# Parameter optimization
python backtester/main_backtest_runner.py --symbols "BTC/USDT" --optimize --metric "sharpe_ratio"

# Debug mode
python backtester/main_backtest_runner.py --symbols "BTC/USDT" --debug
```

### Key Files
- **Main Script**: `backtester/main_backtest_runner.py`
- **Strategy Config**: `backtester/config/strategy_params/dma_atr_trend_params.json`
- **Strategy Implementation**: `backtester/strategies/dma_atr_trend_strategy.py`
- **Results**: `results/` directory with CSV and HTML files

### Performance Targets
- **Sharpe Ratio**: > 1.5 (excellent > 2.0)
- **Max Drawdown**: < 15% (excellent < 10%)
- **Win Rate**: > 55% (excellent > 60%)
- **Profit Factor**: > 1.3 (excellent > 1.5) 