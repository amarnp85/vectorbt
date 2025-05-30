# Analysis Module

The Analysis module provides comprehensive performance analysis and visualization for backtesting results, featuring a modular architecture with specialized components for different analysis aspects.

## 🏗️ Architecture Overview

The module has been refactored into a clean, modular architecture:

```
analysis/
├── chart_components/     # Modular chart building blocks
│   ├── base.py          # Interfaces and base classes
│   ├── processors.py    # Data processing components
│   ├── builders.py      # Chart structure builders
│   ├── renderers.py     # Visual element renderers
│   └── managers.py      # State and configuration managers
├── signal_components/    # Signal processing components
│   ├── timing.py        # Signal timing calculations
│   ├── extractors.py    # Signal extraction from various sources
│   └── validators.py    # Signal validation and quality checks
├── trading_charts.py    # Main orchestrator (488 lines, -59% from original)
├── trading_signals.py   # Signal orchestrator (439 lines, -70% from original)
└── performance_analyzer.py  # Performance metrics engine
```

## 📊 Key Components

### TradingChartsEngine
The main orchestrator that coordinates all chart components to create professional trading visualizations.

**Features:**
- **Main Trading Chart**: Price action with indicators, signals, and stop levels
- **Strategy Analysis Chart**: Comprehensive performance dashboard with 6 subplots
- **Smart Component System**: Each component has a single responsibility
- **Enhanced Interactivity**: Improved zoom, pan, resize with Plotly
- **Stop Level Visualization**: Shows exact SL/TP levels for all trades

**Usage:**
```python
from backtester.analysis import TradingChartsEngine

# Create engine
charts = TradingChartsEngine(
    portfolio=portfolio,
    data=data,
    indicators=indicators,
    signals=signals,
    signal_config=SignalConfig(signal_timing_mode="execution")
)

# Generate main trading chart
main_fig = charts.create_main_chart(
    title="BTC/USDT Trading Analysis",
    show_volume=True,
    show_signals=True,
    show_equity=True
)

# Generate strategy analysis dashboard
analysis_fig = charts.create_strategy_analysis_chart(
    title="Strategy Performance Analysis"
)
```

### Chart Components

#### Processors
- **DataProcessor**: Extracts and validates OHLCV data from VBT objects
- **IndicatorProcessor**: Categorizes indicators (price overlay, volume overlay, subplot)

#### Builders
- **ChartBuilder**: Creates subplot structure with optimal layout
- **Smart range calculation**: Prevents chart scaling issues

#### Renderers
- **CandlestickRenderer**: Main price chart
- **IndicatorRenderer**: All indicator types with proper placement
- **SignalRenderer**: Trading signals with SL/TP levels
- **VolumeRenderer**: Volume bars with color coding
- **EquityRenderer**: Portfolio equity curve

#### Managers
- **LegendManager**: Prevents legend duplication
- **LayoutManager**: Calculates optimal subplot heights
- **ThemeManager**: Consistent styling across charts

### SignalProcessor
Orchestrates signal extraction, validation, and timing adjustments.

**Features:**
- **Unified Signal Extraction**: From portfolio trades and strategy signals
- **Timing Modes**: Signal time vs execution time to prevent lookahead bias
- **Comprehensive Validation**: Ensures signal quality and consistency
- **Stop Level Handling**: Preserves SL/TP levels through the pipeline

**Usage:**
```python
from backtester.analysis.trading_signals import SignalProcessor, SignalConfig

# Configure signal processing
signal_config = SignalConfig(
    signal_timing_mode="execution",  # Realistic execution timing
    show_signals=True,
    show_stop_levels=True
)

# Process signals
processor = SignalProcessor(portfolio, data_processor, strategy_signals, signal_config)
signals = processor.extract_signals()
validation_report = processor.validate_signals()
```

### PerformanceAnalyzer
Comprehensive performance metrics calculation engine.

**Metrics Categories:**
- **Returns Metrics**: Total, annualized, volatility, Sharpe, Sortino
- **Trade Metrics**: Win rate, profit factor, expectancy, average trade
- **Drawdown Metrics**: Max drawdown, duration, recovery
- **Risk Metrics**: VaR, CVaR, downside deviation
- **Benchmark Comparison**: Relative performance, alpha, beta

**Usage:**
```python
from backtester.analysis import PerformanceAnalyzer

analyzer = PerformanceAnalyzer(portfolio, signals=signals)

# Get all metrics
metrics = analyzer.get_all_metrics()

# Export results
analyzer.export_results("performance_report.csv")
```

### Strategy Analysis Chart
The new analysis chart provides 6 key visualizations:

1. **Portfolio Equity Curve**: Shows portfolio value over time
2. **Trade Distribution & Win Rate**: Cumulative trades and rolling win rate
3. **Drawdown Analysis**: Visualizes drawdowns with max DD highlighted
4. **Trade Returns Distribution**: Histogram showing win/loss distribution
5. **Trade Duration Distribution**: How long trades typically last
6. **Monthly Returns Heatmap**: Performance consistency over time

## 🚀 Quick Start

### Basic Trading Analysis
```python
from backtester.analysis import TradingChartsEngine, PerformanceAnalyzer

# Analyze performance
analyzer = PerformanceAnalyzer(portfolio)
metrics = analyzer.get_all_metrics()

# Create visualizations
charts = TradingChartsEngine(portfolio, data, indicators, signals)

# Main trading chart
trading_fig = charts.create_main_chart()
charts.save_chart(trading_fig, "trading_analysis.html")

# Performance dashboard
analysis_fig = charts.create_strategy_analysis_chart()
charts.save_chart(analysis_fig, "performance_analysis.html")
```

### Multi-Timeframe Analysis
```python
from backtester.analysis import MTFPlottingEngine

mtf_engine = MTFPlottingEngine(data_dict, indicators_dict)
fig = mtf_engine.create_mtf_chart(
    symbol="BTC/USDT",
    show_volume=True,
    height_per_timeframe=400
)
```

## 📈 Chart Types

### 1. Main Trading Chart
- Price action with candlesticks
- Technical indicators (overlays and subplots)
- Entry/exit signals with arrows
- Stop loss and take profit levels (horizontal dashes)
- Volume analysis
- Portfolio equity curve

### 2. Strategy Analysis Dashboard
- 6 synchronized subplots for comprehensive analysis
- Performance metrics visualization
- Risk analysis
- Trade quality assessment
- Time-based performance breakdown

## 🔧 Advanced Features

### Signal Timing Modes
Prevent lookahead bias with proper signal timing:
- **Signal Mode**: Shows signals at generation time (for analysis)
- **Execution Mode**: Shows signals at execution time (realistic)

### Stop Level Visualization
- Automatic detection of SL/TP levels from strategy
- Red dashes for stop losses
- Green dashes for take profits
- Works for both long and short positions

### Smart Indicator Placement
- Automatically categorizes indicators
- Price overlays (MA, Bollinger Bands)
- Volume overlays (OBV)
- Separate subplots (RSI, MACD)

## 📝 Best Practices

1. **Use Modular Components**: Leverage the component system for custom visualizations
2. **Configure Signal Timing**: Always use "execution" mode for realistic results
3. **Export High Quality**: Use PNG export with scale=2 for publications
4. **Validate Signals**: Always check the validation report for signal quality
5. **Benchmark Comparison**: Use the analyzer's benchmark features for relative performance

## 🔗 Related Documentation

- [VBT_REFERENCE.md](VBT_REFERENCE.md) - VectorBT Pro patterns and techniques
- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) - Future enhancements roadmap
- [Module Architecture](../README.md) - Overall framework architecture

## 📦 Dependencies

- `vectorbtpro`: Core backtesting engine
- `plotly`: Interactive visualizations
- `pandas`: Data manipulation
- `numpy`: Numerical computations

The Analysis module represents a modern, modular approach to backtesting analysis, providing both powerful visualizations and comprehensive performance metrics in a maintainable architecture.