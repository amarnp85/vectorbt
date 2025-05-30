# 🚀 Multi-Symbol Portfolio Backtest - Quick Start Guide

## Overview

This guide shows you how to run and understand the multi-symbol portfolio backtest in the VectorBT Pro backtesting framework.

## Quick Start Commands

### 1. Basic Run (Recommended)

```bash
# Navigate to the project
cd /Users/amarpatel/python/backtester

# Run the simple working example
python examples/03_multi_symbol_portfolio_simple.py

# Run without plots (faster)
python examples/03_multi_symbol_portfolio_simple.py --no-plots
```

### 2. Detailed Analysis Run

```bash
# Run with comprehensive analysis
python examples/03_multi_symbol_portfolio_analysis.py
```

## What the Backtest Does

### Strategy Overview
- **Type**: Simple Moving Average Crossover
- **Fast MA**: 20-day moving average
- **Slow MA**: 50-day moving average
- **Signal**: Buy when fast MA crosses above slow MA, sell when it crosses below
- **Symbols**: BTC/USDT, ETH/USDT, SOL/USDT, ADA/USDT
- **Capital**: $100,000 shared across all symbols

### Key Concepts

1. **Multi-Symbol Data Structure**
   - Single `vbt.Data` object contains all symbols
   - Each symbol is a column in the DataFrame
   - All calculations happen simultaneously across symbols

2. **Signal Generation**
   - Signals are DataFrames with one column per symbol
   - VectorBT broadcasts operations across all symbols at once
   - Example: `long_entries` DataFrame has 4 columns (one per symbol)

3. **Portfolio Simulation**
   - Uses `MultiAssetPortfolioSimulator` class
   - `cash_sharing=True` means all symbols share the $100k capital
   - Position size: 25% of equity per position (max 4 positions)

4. **Risk Management**
   - Stop Loss: 2x ATR (Average True Range)
   - Take Profit: 3x ATR
   - Commission: 0.1% per trade
   - Slippage: 0.05%

## Understanding the Output

### Performance Metrics

```
📈 Backtest Results
╭──────────────┬─────────┬────────╮
│ Total Return │ 42.62%  │ 🟢     │  <- Overall portfolio return
│ Sharpe Ratio │ 0.74    │ 🟡     │  <- Risk-adjusted return (>1 is good)
│ Max Drawdown │ -11.19% │ 🟡     │  <- Largest peak-to-trough loss
│ Win Rate     │ 55.56%  │ 🟢     │  <- Percentage of winning trades
│ Total Trades │ 36      │ 🟢     │  <- Number of completed trades
╰──────────────┴─────────┴────────╯
```

### Symbol Analysis

```
📊 Symbol Analysis
Symbol-wise trade statistics:
ADA/USDT: 3 trades, PnL: -1616.33, Avg Return: -2.80%
BTC/USDT: 3 trades, PnL: 180.18, Avg Return: 0.21%
ETH/USDT: 27 trades, PnL: 38661.28, Avg Return: 5.12%  <- Main contributor
SOL/USDT: 3 trades, PnL: 5396.09, Avg Return: 7.52%
```

## Key Files Explained

### 1. `examples/03_multi_symbol_portfolio_simple.py`
- **Purpose**: Working example of multi-symbol portfolio
- **Status**: ✅ WORKING - Use this one
- **Features**: Basic MA crossover across 4 crypto symbols

### 2. `examples/03_multi_symbol_portfolio_analysis.py`
- **Purpose**: Detailed analysis version with extensive metrics
- **Status**: ✅ WORKING
- **Features**: Correlation analysis, diversification metrics, position sizing impact

### 3. `examples/03_multi_symbol_portfolio.py`
- **Purpose**: Original example (has issues)
- **Status**: ❌ BROKEN - Don't use

### 4. `backtester/strategies/multi_symbol_dma_atr_strategy.py`
- **Purpose**: Advanced multi-symbol strategy with cross-symbol filters
- **Status**: 🔧 In progress - not fully working yet

## Visualizations

When you run without `--no-plots`, the system generates:

1. **portfolio_performance.html** - Interactive price chart with:
   - Portfolio equity curve
   - Entry/exit signals for all symbols
   - Stop loss and take profit levels
   - Volume indicators

2. **strategy_analysis.html** - Performance analytics with:
   - Monthly returns heatmap
   - Trade distribution
   - Drawdown analysis
   - Returns histogram

Files are saved to: `results/example_03_multi_symbol_simple/`

## Common Issues & Solutions

### Issue 1: "Can only compare identically-labeled DataFrame objects"
**Cause**: VBT indicators return MultiIndex columns
**Solution**: Already fixed in the simple example by extracting values and resetting columns

### Issue 2: Zero returns or no trades
**Cause**: Signals not properly structured as DataFrames
**Solution**: Ensure signals have columns matching data.symbols

### Issue 3: Missing data warnings
**Cause**: Some symbols may not have data for the full date range
**Solution**: Use date ranges where all symbols have data (e.g., 2023-01-01 onwards)

## Customization Options

### Change Symbols
```python
symbols = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "MATIC/USDT"]
```

### Adjust Strategy Parameters
```python
signals = create_simple_ma_crossover_signals(data, fast_period=10, slow_period=30)
```

### Modify Position Sizing
```python
sim_config = SimulationConfig(
    position_size_value=0.20,  # 20% per position instead of 25%
)
```

### Change Date Range
```python
start_date = "2024-01-01"
end_date = "2024-12-31"
```

## Performance Insights

From the example run:

1. **Diversification Works**: Portfolio volatility (11% drawdown) is lower than individual symbol volatility
2. **Unequal Contribution**: ETH dominates with 27 trades and $38k profit
3. **Risk-Adjusted Returns**: Sharpe of 0.74 is decent but not exceptional
4. **Capital Efficiency**: With 25% position sizing, maximum 4 concurrent positions

## Next Steps

1. **Optimize Parameters**: Try different MA periods for better performance
2. **Add Filters**: Implement volume or trend strength filters
3. **Risk Management**: Experiment with different stop loss/take profit ratios
4. **More Symbols**: Add uncorrelated assets for better diversification
5. **Advanced Strategies**: Implement the MultiSymbolDMAATRStrategy when ready

## Summary

The multi-symbol portfolio backtest demonstrates:
- ✅ Proper use of VectorBT Pro's multi-symbol capabilities
- ✅ Portfolio-level capital management
- ✅ Simultaneous signal generation across symbols
- ✅ Realistic transaction costs and slippage
- ✅ Professional performance reporting

This forms the foundation for more sophisticated multi-asset strategies in the VectorBT Pro framework.