# Results Directory Structure

This document outlines the simplified directory structure for storing backtesting and optimization results.

## Overview

Results are organized into two main categories:
- **Symbol Level**: Individual ticker analysis and optimization
- **Portfolio Level**: Multi-symbol portfolio analysis and optimization

## Directory Structure

```
results/
├── symbols/                           # Individual symbol analysis
│   ├── BTC_USDT/                     # Bitcoin analysis (all timeframes)
│   │   ├── backtest_results.csv      # All backtest results
│   │   ├── optimization/             # Symbol-specific optimization
│   │   │   ├── 4h/                   # 4-hour timeframe optimization
│   │   │   │   ├── optimization_summary.json
│   │   │   │   ├── parameter_analysis.csv
│   │   │   │   ├── top_performers.csv
│   │   │   │   └── visualizations/
│   │   │   │       ├── parameter_heatmap.html
│   │   │   │       ├── performance_distribution.html
│   │   │   │       └── top_performers_parallel.html
│   │   │   ├── 1d/                   # Daily timeframe optimization
│   │   │   │   ├── optimization_summary.json
│   │   │   │   ├── parameter_analysis.csv
│   │   │   │   ├── top_performers.csv
│   │   │   │   └── visualizations/
│   │   │   └── 1h/                   # Hourly timeframe optimization
│   │   │       └── ...
│   │   └── plots/                    # All charts and visualizations
│   ├── ETH_USDT/                     # Ethereum analysis (all timeframes)
│   │   ├── backtest_results.csv
│   │   ├── optimization/
│   │   │   └── 4h/                   # Organized by timeframe
│   │   └── plots/
│   ├── SOL_USDT/                     # Solana analysis (all timeframes)
│   │   └── ...
│   └── [SYMBOL]/                     # Any other symbols
│       └── ...
├── portfolios/                       # Multi-symbol portfolio analysis
│   ├── crypto_majors/                # BTC, ETH, SOL portfolio
│   ├── defi_tokens/                  # DeFi token portfolio
│   ├── custom_portfolio_1/           # User-defined portfolios
│   └── optimization/                 # Portfolio-level optimization
└── general/                          # General testing and comparisons
    ├── strategy_comparisons/         # Compare strategies across symbols
    ├── timeframe_analysis/           # Cross-timeframe analysis
    └── testing/                      # General testing results
```

## File Organization

### Symbol Level Results
Each symbol folder contains all timeframes together:
```
BTC_USDT/
├── backtest_results.csv              # All backtest results (includes timeframe column)
├── performance_metrics.json          # Performance summary
├── optimization/
│   ├── 4h/                           # 4-hour timeframe optimization
│   │   ├── optimization_summary.json # Key results and metadata
│   │   ├── parameter_analysis.csv    # Parameter sensitivity analysis
│   │   ├── top_performers.csv        # Best 20 parameter combinations
│   │   └── visualizations/
│   │       ├── parameter_heatmap.html # Parameter sensitivity visualization
│   │       ├── performance_distribution.html # Optimization quality analysis
│   │       └── top_performers_parallel.html  # Parameter relationships
│   ├── 1d/                           # Daily timeframe optimization
│   │   ├── optimization_summary.json
│   │   ├── parameter_analysis.csv
│   │   ├── top_performers.csv
│   │   └── visualizations/
│   └── 1h/                           # Hourly timeframe optimization
│       └── ...
└── plots/
    ├── BTC_USDT_4h_main_chart.html
    ├── BTC_USDT_1d_main_chart.html
    ├── BTC_USDT_4h_strategy_analysis.html
    └── BTC_USDT_1d_strategy_analysis.html
```

### Portfolio Level Results
```
portfolios/custom_portfolio_name/
├── portfolio_backtest.csv
├── portfolio_metrics.json
├── allocation_analysis.csv
├── correlation_matrix.csv
└── plots/
    ├── portfolio_performance.html
    ├── allocation_pie_chart.html
    └── correlation_heatmap.html
```

## Usage Examples

### Symbol Level Testing
```bash
# Results go to: results/symbols/BTC_USDT/
python strategy_tester.py --symbols BTC/USDT --timeframes 4h

# Optimization results go to: results/symbols/BTC_USDT/optimization/
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --optimize

# Charts go to: results/symbols/BTC_USDT/plots/
```

### Portfolio Level Testing
```bash
# Results go to: results/portfolios/my_crypto_portfolio/
python strategy_tester.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --portfolio-name my_crypto_portfolio

# Portfolio optimization results go to: results/portfolios/optimization/
python 04_portfolio_optimization.py
```

### General Testing
```bash
# Results go to: results/general/testing/
python strategy_tester.py --symbols BTC/USDT,ETH/USDT --compare-strategies
```

## Benefits

1. **Flattened Structure**: No timeframe subdirectories - all results for a symbol in one place
2. **Symbol Organization**: Each ticker has its own dedicated folder
3. **Flexible Portfolios**: Bespoke portfolio names and organization
4. **Scalable**: Easy to add new symbols or portfolios
5. **Intuitive**: Natural organization that matches trading workflow

## Migration

The system will automatically create this structure and migrate existing results to the appropriate locations based on the content and naming patterns. 