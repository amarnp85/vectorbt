# Analysis Module

The Analysis module provides comprehensive performance analysis and modern interactive visualization capabilities for backtesting results.

## Overview

This module has been streamlined and consolidated around a powerful new `TradingChartsEngine` that provides professional-grade interactive charts with enhanced zoom, pan, and resize functionality.

## Key Components

### 📊 TradingChartsEngine
The core plotting engine that provides:
- **Enhanced Interactivity**: Improved zoom, pan, and resize with responsive design
- **Smart Indicator Categorization**: Automatically places indicators in appropriate chart locations
- **Multiple Chart Types**: Main trading charts, strategy analysis, and performance metrics
- **Multiple Backends**: Plotly (primary), mplfinance (static), bokeh (interactive)
- **Professional Export**: High-quality HTML, PNG, PDF exports with drawing tools

### 📈 PerformanceAnalyzer
Comprehensive performance metrics calculation:
- Portfolio-level metrics (returns, Sharpe, Sortino, Calmar ratios)
- Trade-level analysis (win rate, profit factor, expectancy)
- Risk-adjusted returns and drawdown analysis
- Benchmark comparison and relative performance

### 🔧 VBTCompatibilityLayer
Handles differences between vectorbtpro versions and provides safe access to portfolio methods.

### 📊 BenchmarkAnalyzer
Advanced benchmark comparison and sizing calibration using vectorbtpro's built-in capabilities.

### ⏰ MTFPlottingEngine
Specialized plotting for multi-timeframe analysis with aligned indicators and trend comparisons.

## Quick Start

### Basic Usage

```python
from backtester.analysis import TradingChartsEngine, PerformanceAnalyzer

# Create charts engine
charts = TradingChartsEngine(portfolio, data, indicators)

# Generate main trading chart
main_chart = charts.create_main_chart(
    title="My Strategy Analysis",
    show_volume=True,
    show_signals=True,
    show_equity=True
)

# Save with enhanced interactivity
charts.save_chart(main_chart, "strategy_analysis.html")

# Generate performance analysis
strategy_chart = charts.create_strategy_analysis_chart(
    title="Strategy Performance Metrics"
)
charts.save_chart(strategy_chart, "performance_metrics.html")
```

### Performance Analysis

```python
# Analyze performance
analyzer = PerformanceAnalyzer(portfolio)

# Get comprehensive metrics
summary_stats = analyzer.get_summary_stats()
trade_metrics = analyzer.get_trade_metrics()
risk_metrics = analyzer.get_risk_metrics()

# Export detailed report
analyzer.export_results("performance_report.csv", include_trades=True)

# Generate formatted report
report = analyzer.generate_report(detailed=True)
print(report)
```

## Chart Features

### Enhanced Interactivity
- **Responsive Design**: Charts automatically resize to fit container
- **Advanced Zoom**: Box zoom, wheel zoom, and pan with reset functionality
- **Drawing Tools**: Add lines, shapes, and annotations directly on charts
- **Spike Lines**: Precise data reading with crosshair functionality
- **Unified Hover**: Better hover experience across all subplots

### Smart Indicator Placement
The engine automatically categorizes indicators:
- **Price Overlays**: Moving averages, Bollinger Bands on main chart
- **Volume Indicators**: Volume SMA, OBV in volume subplot
- **Oscillators**: RSI, MACD in separate subplots with reference lines

### Professional Export
- **High-Quality HTML**: Responsive design with full interactivity
- **Static Images**: PNG, PDF, SVG at publication quality (1920x1080, 2x scale)
- **Drawing Tools**: Export includes user annotations and drawings

## Chart Types

### 1. Main Trading Chart
Comprehensive analysis with:
- Candlestick price data
- Technical indicators (smart placement)
- Trade entry/exit signals
- Volume analysis
- Portfolio equity curve
- Drawdown visualization

### 2. Strategy Analysis Chart
Advanced performance metrics with:
- Monthly returns heatmap
- Rolling Sharpe ratio
- Trade distribution analysis
- Drawdown analysis with annotations

## Migration from Old System

The module has been consolidated from multiple plotting engines into a single, powerful `TradingChartsEngine`. 

### What Changed
- ✅ **Removed**: `PlottingEngine`, `EnhancedPlottingEngine`, `OptimizationPlotter`
- ✅ **Added**: Enhanced `TradingChartsEngine` with all functionality
- ✅ **Improved**: Better interactivity, responsive design, professional export

### Migration Guide
```python
# Old way (deprecated)
from backtester.analysis import PlottingEngine
plotter = PlottingEngine(portfolio, data, indicators)
fig = plotter.plot_portfolio_overview()

# New way (recommended)
from backtester.analysis import TradingChartsEngine
charts = TradingChartsEngine(portfolio, data, indicators)
fig = charts.create_main_chart()
```

## Configuration

### Chart Appearance
```python
# Configure backend
charts = TradingChartsEngine(portfolio, data, indicators, backend="plotly")

# Customize chart
fig = charts.create_main_chart(
    title="Custom Strategy Analysis",
    show_volume=True,
    show_signals=True,
    show_equity=True,
    height=1200
)
```

### Export Options
```python
# Enhanced HTML export
charts.save_chart(fig, "analysis.html", format="html")

# High-quality image export
charts.save_chart(fig, "analysis.png", format="png")
```

## Performance Tips

1. **Large Datasets**: Use date range filtering for better performance
2. **Multiple Charts**: Reuse the same TradingChartsEngine instance
3. **Export Quality**: Use PNG/PDF for presentations, HTML for interactive analysis
4. **Memory Usage**: Clear browser cache if working with many large charts

## Future Enhancements

See [FUTURE_IMPROVEMENTS_PLAN.md](./FUTURE_IMPROVEMENTS_PLAN.md) for detailed roadmap including:
- **Phase 1**: Streamlit dashboard integration
- **Phase 2**: Advanced 3D visualization and real-time features
- **Phase 3**: Machine learning integration and predictive analytics
- **Phase 4**: Enterprise features and cloud deployment

## Examples

### Complete Analysis Workflow
```python
from backtester.analysis import TradingChartsEngine, PerformanceAnalyzer

# Initialize
charts = TradingChartsEngine(portfolio, data, indicators)
analyzer = PerformanceAnalyzer(portfolio)

# Generate main analysis
main_chart = charts.create_main_chart()
charts.save_chart(main_chart, "results/main_chart.html")

# Generate performance analysis  
perf_chart = charts.create_strategy_analysis_chart(analyzer=analyzer)
charts.save_chart(perf_chart, "results/strategy_analysis.html")

# Export metrics
analyzer.export_results("results/performance_metrics.csv")

print("Analysis complete! Check the results/ directory.")
```

### Multi-Timeframe Analysis
```python
from backtester.analysis import MTFPlottingEngine

# Multi-timeframe plotting
mtf_plotter = MTFPlottingEngine()
mtf_chart = mtf_plotter.plot_mtf_price_overview(mtf_data)
```

## Dependencies

- **vectorbtpro**: Core backtesting and data handling
- **plotly**: Interactive plotting (primary backend)
- **pandas**: Data manipulation
- **numpy**: Numerical computations
- **mplfinance**: Static financial charts (optional)
- **bokeh**: Alternative interactive backend (optional)

## Support

For issues, feature requests, or questions about the analysis module, please refer to the main project documentation or create an issue in the project repository. 