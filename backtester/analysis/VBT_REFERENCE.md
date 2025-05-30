# VectorBT Pro Reference Guide

This document consolidates important VectorBT Pro patterns and techniques used throughout the analysis module, based on the official documentation.

## Core Plotting Patterns

### Basic Plotting via Accessors
VectorBT Pro provides powerful plotting capabilities through pandas accessors:

```python
# Basic line plot
fig = data.vbt.plot()

# Specific plot types
fig = series.vbt.lineplot()          # Line plots
fig = series.vbt.scatterplot()       # Scatter plots
fig = series.vbt.barplot()           # Bar charts
fig = series.vbt.histplot()          # Histograms
fig = series.vbt.boxplot()           # Box plots
fig = series.vbt.areaplot()          # Area charts
fig = series.vbt.heatmap()           # Heatmaps
fig = series.vbt.ts_heatmap()        # Time series heatmaps
fig = series.vbt.qqplot()            # QQ plots for distribution analysis
```

### Portfolio Plotting
```python
# Portfolio value over time
fig = portfolio.value.vbt.lineplot()

# Portfolio metrics visualization
fig = portfolio.sharpe_ratio.vbt.barplot()
fig = portfolio.allocations.vbt.areaplot(line_shape="hv")

# Returns analysis
fig = portfolio.returns.vbt.histplot(trace_kwargs=dict(nbinsx=100))
fig = portfolio.returns.vbt.qqplot()

# Monthly returns heatmap
monthly_returns = portfolio.returns_acc.resample("M").get()
fig = monthly_returns.vbt.heatmap()
fig = monthly_returns.vbt.ts_heatmap()
```

### Multi-dimensional Heatmaps
```python
# Parameter optimization heatmaps
fig = portfolio.sharpe_ratio.vbt.heatmap(
    x_level="fast_window",
    y_level="slow_window",
    symmetric=True,
    trace_kwargs=dict(
        colorbar=dict(title="Sharpe Ratio"),
        hovertemplate="%{z:.4f}<extra></extra>"
    )
)
```

## Signal Processing

### Signal Cleaning
VectorBT provides signal cleaning to ensure proper entry/exit sequences:

```python
# Basic signal cleaning
clean_entries, clean_exits = entries.vbt.signals.clean(exits)

# Custom signal cleaning with Numba
@njit
def custom_clean_nb(long_en, long_ex, short_en, short_ex):
    new_long_en = np.full_like(long_en, False)
    new_long_ex = np.full_like(long_ex, False)
    new_short_en = np.full_like(short_en, False)
    new_short_ex = np.full_like(short_ex, False)
    
    for col in range(long_en.shape[1]):
        position = 0  # 0: flat, 1: long, -1: short
        for i in range(long_en.shape[0]):
            if long_en[i, col] and position != 1:
                new_long_en[i, col] = True
                position = 1
            elif short_en[i, col] and position != -1:
                new_short_en[i, col] = True
                position = -1
            elif long_ex[i, col] and position == 1:
                new_long_ex[i, col] = True
                position = 0
            elif short_ex[i, col] and position == -1:
                new_short_ex[i, col] = True
                position = 0
                
    return new_long_en, new_long_ex, new_short_en, new_short_ex
```

### Signal Generation Helpers
```python
# Generate signals from crossovers
long_entries = fast_ma.crossed_above(slow_ma)
long_exits = fast_ma.crossed_below(slow_ma)

# Generate signals from thresholds
oversold = rsi < 30
overbought = rsi > 70
```

## Portfolio Analysis

### Key Portfolio Attributes
```python
# Portfolio value and equity
portfolio.value  # Total portfolio value
portfolio.cash   # Available cash
portfolio.asset_value  # Value of positions

# Returns
portfolio.returns  # Simple returns
portfolio.log_returns  # Log returns
portfolio.cumulative_returns  # Cumulative returns
portfolio.returns_acc  # Returns accumulator for resampling

# Performance metrics
portfolio.sharpe_ratio  # Sharpe ratio
portfolio.sortino_ratio  # Sortino ratio
portfolio.calmar_ratio  # Calmar ratio
portfolio.max_drawdown  # Maximum drawdown

# Trade analysis
portfolio.trades.records  # Trade records
portfolio.trades.pnl  # Trade PnL
portfolio.trades.returns  # Trade returns
```

### Advanced Portfolio Methods
```python
# Get portfolio statistics
stats = portfolio.stats()

# Compare multiple portfolios
compared = vbt.Portfolio.from_signals(
    close, entries, exits, 
    param_product=True
).analyze()

# Resample portfolio data
daily_value = portfolio.value
monthly_value = daily_value.resample('M').last()
```

## Plotting Configuration

### Enhanced Plotly Configuration
```python
# Recommended configuration for interactive charts
plotly_config = {
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
    'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
    'toImageButtonOptions': {
        'format': 'png',
        'width': 1920,
        'height': 1080,
        'scale': 2
    }
}

# Layout updates for better interactivity
layout_updates = dict(
    hovermode='x unified',
    spikedistance=-1,
    xaxis=dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showline=True,
        showgrid=True
    ),
    yaxis=dict(
        showspikes=True,
        spikemode='across',
        spikesnap='cursor',
        showline=True,
        showgrid=True
    )
)
```

## Common Patterns

### Safe Attribute Access
```python
# Always check for attribute existence with VBT objects
if hasattr(portfolio, 'trades') and len(portfolio.trades.records) > 0:
    trades = portfolio.trades.records_readable
```

### Efficient Data Processing
```python
# Use VBT's built-in methods for data manipulation
# Good - uses VBT's optimized methods
data_resampled = data.resample('4H')

# Avoid - manual pandas operations
# df_resampled = df.resample('4H').agg({...})
```

### Multi-column Operations
```python
# VBT handles multi-column data efficiently
portfolio = vbt.Portfolio.from_signals(
    close,  # Can be multi-column
    entries,  # Can be multi-column
    exits,  # Can be multi-column
    direction='both',
    freq='H'
)
```

## Performance Tips

1. **Use VBT's built-in plotting**: It's optimized for financial data
2. **Leverage accessors**: `.vbt` accessor provides optimized methods
3. **Batch operations**: Process multiple assets/parameters together
4. **Use caching**: VBT caches many calculations automatically
5. **Prefer vectorized operations**: Avoid loops when possible

## References

- [VectorBT Pro Documentation](https://vectorbt.pro/docs/)
- Signal Development Tutorial
- Portfolio Tutorial
- Plotting Tutorial

This reference is based on the cookbook examples and patterns found throughout the VectorBT Pro documentation and should be used as a quick reference when working with the analysis module.