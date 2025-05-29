# Examples - Strategy Testing Framework

This directory contains comprehensive examples demonstrating how to test trading strategies from start to finish using the backtesting framework.

## 🎯 Quick Start

### **Simple Testing (No Configuration Required)**
```bash
# Test single symbol with automatic optimization
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --optimize

# Test multiple symbols and timeframes
python strategy_tester.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframes 4h,1d --optimize

# Quick test without plots
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --optimize --no-plots
```

### **Learning Examples**
```bash
# Start with basic concepts
python 01_single_symbol_backtest.py

# Learn optimization process
python 02_parameter_optimization.py
```

## 📁 **Simplified Results Structure**

Results are now organized by **symbol level** vs **portfolio level**:

```
results/
├── symbols/                          # Individual symbol analysis
│   ├── BTC_USDT/                    # Bitcoin analysis (all timeframes)
│   │   ├── optimization/            # Symbol-specific optimization
│   │   └── plots/                   # All charts and visualizations
│   │       ├── BTC_USDT_4h_main_chart.html
│   │       ├── BTC_USDT_1d_main_chart.html
│   │       └── BTC_USDT_4h_strategy_analysis.html
│   ├── ETH_USDT/                    # Ethereum analysis (all timeframes)
│   │   ├── optimization/
│   │   └── plots/
│   └── SOL_USDT/                    # Solana analysis (all timeframes)
│       ├── optimization/
│       └── plots/
├── portfolios/                       # Multi-symbol portfolio analysis
│   ├── my_crypto_portfolio/         # Bespoke portfolio names
│   ├── defi_experiment/             # Custom portfolio analysis
│   ├── momentum_strategy/           # Strategy-specific portfolios
│   └── optimization/                # Portfolio-level optimization
└── general/                         # General testing and comparisons
    ├── strategy_comparisons/        # Compare strategies across symbols
    ├── timeframe_analysis/          # Cross-timeframe analysis
    └── testing/                     # General testing results
```

## 📅 **Fixed Date Ranges**

The system uses **fixed date ranges** for consistent, reproducible results:

- **`recent_data`** (default): 2020-01-01 to 2023-12-30 - Comprehensive historical data
- **`full_data`**: 2020-01-01 to 2023-12-31 - Full historical analysis
- **`optimization_period`**: 2022-01-01 to 2023-12-31 - Optimization period
- **`validation_period`**: 2024-01-01 to 2024-06-30 - Validation period

```bash
# Use different periods
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --period full_data --optimize
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --period validation_period
```

## 📊 **What You Get**

### **Symbol-Level Results**
When testing individual symbols, results go to `results/symbols/{SYMBOL}/`:

```
BTC_USDT/
├── optimization/
│   ├── 4h/                             # 4-hour timeframe optimization
│   │   ├── optimization_summary.json   # Quick overview & best parameters
│   │   ├── parameter_analysis.csv      # Statistical analysis per parameter
│   │   ├── top_performers.csv          # Best 20 combinations ranked
│   │   └── visualizations/
│   │       ├── parameter_heatmap.html
│   │       ├── performance_distribution.html
│   │       └── top_performers_parallel.html
│   ├── 1d/                             # Daily timeframe optimization
│   │   ├── optimization_summary.json
│   │   ├── parameter_analysis.csv
│   │   ├── top_performers.csv
│   │   └── visualizations/
│   └── 1h/                             # Hourly timeframe optimization
│       └── ...
└── plots/                              # All charts and visualizations
    ├── BTC_USDT_4h_main_chart.html    # Timeframe included in filename
    ├── BTC_USDT_1d_main_chart.html
    ├── BTC_USDT_4h_strategy_analysis.html
    └── BTC_USDT_1d_strategy_analysis.html
```

### **General Testing Results**
Multi-symbol testing results go to `results/general/testing/`:

```
general/testing/
├── testing_results.csv                    # Complete testing results with parameters
├── BTC_USDT_4h_main_chart.html           # Individual symbol charts
├── ETH_USDT_4h_main_chart.html
└── SOL_USDT_4h_main_chart.html
```

### **Results with Full Parameter Information**
```csv
symbol,timeframe,param_source,param_summary,fast_window,slow_window,atr_window,total_return,sharpe_ratio,max_drawdown,win_rate,total_trades,start_date,end_date,period_description
BTC/USDT,4h,optimized,F10/S80/A10,10,80,10,43.22%,0.652,-8.67%,30.6%,330,2020-01-01,2023-12-30,Comprehensive historical data
```

## 📁 **Example Files**

### **Core Examples**
1. **`01_single_symbol_backtest.py`** - Basic backtesting introduction
   - Results: `results/symbols/{SYMBOL}/plots/`
2. **`02_parameter_optimization.py`** - Parameter optimization demonstration  
   - Results: `results/symbols/{SYMBOL}/optimization/`
3. **`strategy_tester.py`** - Production-ready consolidated testing
   - Results: `results/general/testing/` (multi-symbol) or `results/symbols/{SYMBOL}/` (single symbol)

### **Advanced Examples**
4. **`03_multi_symbol_portfolio.py`** - Portfolio backtesting
   - Results: `results/portfolios/{CUSTOM_NAME}/`
5. **`04_portfolio_optimization.py`** - Portfolio weight optimization
   - Results: `results/portfolios/optimization/`
6. **`05_walk_forward_analysis.py`** - Cross-validation and robustness testing
7. **`06_multi_timeframe_strategy.py`** - Multi-timeframe analysis
8. **`07_advanced_mtf_strategy.py`** - Advanced MTF infrastructure
9. **`08_advanced_risk_management.py`** - Risk management techniques
10. **`09_pairs_trading.py`** - Statistical arbitrage
11. **`10_momentum_strategy.py`** - Momentum strategy with regime filtering

## 🔧 **Common Use Cases**

### **Single Symbol Analysis**
```bash
# Optimize and test BTC on 4h timeframe
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --optimize

# Results will be in:
# - results/symbols/BTC_USDT/plots/ (charts and analysis)
# - results/symbols/BTC_USDT/optimization/ (optimization analysis)
```

### **Multi-Symbol Comparison**
```bash
# Compare multiple symbols
python strategy_tester.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --timeframes 4h --optimize

# Results will be in:
# - results/general/testing/testing_results.csv (comparison table)
# - results/symbols/{SYMBOL}/optimization/ (individual optimizations)
# - results/symbols/{SYMBOL}/plots/ (individual charts)
```

### **Bespoke Portfolio Analysis**
```bash
# Create custom portfolio with specific name
python strategy_tester.py --symbols BTC/USDT,ETH/USDT,SOL/USDT --portfolio-name my_crypto_experiment

# Results will be in:
# - results/portfolios/my_crypto_experiment/

# Different portfolio strategies
python strategy_tester.py --symbols BTC/USDT,ETH/USDT --portfolio-name momentum_strategy
python strategy_tester.py --symbols UNI/USDT,AAVE/USDT --portfolio-name defi_experiment
```

### **Fix Parameter Issues**
```bash
# Reset corrupted parameters (fixes type errors)
python strategy_tester.py --symbols ETH/USDT --timeframes 4h --reset-params --optimize
```

### **Force Re-optimization**
```bash
# Re-optimize even if parameters exist
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --optimize --force
```

### **Cross-Period Analysis**
```bash
# Test robustness across different periods
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --period optimization_period
python strategy_tester.py --symbols BTC/USDT --timeframes 4h --period validation_period
```

## 🎯 **Workflow**

### **1. Learning Phase**
```bash
# Understand the basics
python 01_single_symbol_backtest.py
# Results: results/symbols/ETH_USDT/plots/

# Learn optimization
python 02_parameter_optimization.py  
# Results: results/symbols/BTC_USDT/optimization/
```

### **2. Testing Phase**
```bash
# Test your symbols with optimization
python strategy_tester.py --symbols BTC/USDT,ETH/USDT --timeframes 4h,1d --optimize
# Results: results/general/testing/ + individual symbol folders
```

### **3. Analysis Phase**
- Review `results/general/testing/testing_results.csv` for performance comparison
- Examine `results/symbols/{SYMBOL}/optimization/` for parameter insights
- Open HTML visualizations in `results/symbols/{SYMBOL}/plots/` for interactive analysis

### **4. Validation Phase**
```bash
# Test on different periods for robustness
python strategy_tester.py --symbols BTC/USDT,ETH/USDT --timeframes 4h,1d --period validation_period
```

## 📈 **Understanding Results**

### **Parameter Sources**
- **optimized**: Parameters found through optimization
- **database**: Previously optimized parameters from database
- **default**: Fallback parameters from config files

### **Key Metrics**
- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
- **Total Return**: Overall profit/loss percentage
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Total Trades**: Number of trades executed

### **Optimization Analysis**
- **Parameter Sensitivity**: Which parameters matter most
- **Performance Distribution**: How results vary across parameter space
- **Top Performers**: Best parameter combinations found
- **Robustness**: Consistency across parameter ranges

## 🗄️ **Database Integration**

The framework uses SQLite database for persistent storage:
- **Location**: `backtester/config/optimal_parameters/optimal_parameters.db`
- **Automatic**: Parameters stored and retrieved automatically
- **Persistent**: Results survive between runs
- **Date Tracking**: Date ranges stored for walk-forward analysis

## ⚙️ **Configuration**

### **Strategy Parameters**
Edit `backtester/config/strategy_params/production/dma_atr_trend_params.json`:
```json
{
  "default_parameters": {
    "fast_window": 20,
    "slow_window": 50,
    "atr_window": 14
  },
  "optimization_ranges": {
    "short_ma_window": [10, 15, 20, 25, 30],
    "long_ma_window": [40, 50, 60, 70, 80]
  }
}
```

## 🚀 **Advanced Features**

### **Optimization Analysis**
- **Parameter sensitivity analysis** - identify which parameters matter most
- **Performance distribution analysis** - understand optimization quality
- **Interactive visualizations** - 7 HTML charts for comprehensive analysis
- **Statistical insights** - robustness and consistency metrics

### **Walk-Forward Analysis Ready**
- Fixed date ranges ensure consistent periods
- Date information stored in database for reference
- Parameter performance validated on unseen data
- Ready for rolling optimization implementation

## 💡 **Tips**

### **Best Practices**
1. **Start Simple**: Begin with single symbol testing
2. **Optimize First**: Run optimization before extensive testing
3. **Use Fixed Periods**: Leverage predefined date ranges for consistency
4. **Review Analysis**: Check optimization analysis files for insights
5. **Validate Robustness**: Test across different time periods

### **Performance Tips**
1. **Use `--no-plots`**: Skip visualizations for faster testing
2. **Batch Testing**: Test multiple symbols/timeframes together
3. **Database Leverage**: Let system use stored optimal parameters
4. **Period Selection**: Choose appropriate date ranges for your analysis

### **Troubleshooting**
1. **Parameter Errors**: Use `--reset-params` to fix corrupted parameters
2. **Database Issues**: Check database permissions and disk space
3. **Memory Errors**: Reduce parameter ranges or test fewer combinations
4. **Data Issues**: Verify market data availability for symbols/timeframes

## 🎉 **Key Benefits**

1. **✅ Zero Configuration** - Smart defaults work automatically
2. **✅ Fixed Date Ranges** - Consistent, reproducible results
3. **✅ Comprehensive Analysis** - Detailed optimization insights
4. **✅ Interactive Visualizations** - Professional-grade charts
5. **✅ Database Integration** - Persistent parameter storage
6. **✅ Walk-Forward Ready** - Prepared for advanced analysis
7. **✅ Production Ready** - Scalable testing workflow
8. **✅ Flattened Structure** - No timeframe subdirectories, all results in one place per symbol
9. **✅ Bespoke Portfolios** - Custom portfolio names and flexible organization

## 📚 **Next Steps**

1. **Start Testing**: Use the quick start commands above
2. **Explore Examples**: Run through examples 1-10 to understand capabilities
3. **Analyze Results**: Review CSV files and HTML visualizations in symbol plots folders
4. **Test Robustness**: Compare performance across different periods
5. **Scale Up**: Test more symbols and timeframes as needed
6. **Create Portfolios**: Use bespoke portfolio names for custom analysis
7. **Implement Walk-Forward**: Use stored date ranges for rolling analysis

The framework provides everything needed for professional-grade strategy testing with a clean, flattened directory structure! 