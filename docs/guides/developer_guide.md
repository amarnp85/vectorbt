# Developer Guide - Extending the Backtester Framework

## Overview

This guide provides comprehensive instructions for developers who want to extend the backtester framework with new strategies, indicators, risk management techniques, or analysis tools.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Creating Custom Strategies](#creating-custom-strategies)
3. [Developing New Indicators](#developing-new-indicators)
4. [Adding Risk Management Components](#adding-risk-management-components)
5. [Extending the Analysis Module](#extending-the-analysis-module)
6. [Working with the Data Module](#working-with-the-data-module)
7. [CLI Integration](#cli-integration)
8. [Testing Guidelines](#testing-guidelines)
9. [Performance Optimization](#performance-optimization)
10. [Best Practices](#best-practices)

## Architecture Overview

The backtester follows a modular architecture:

```
backtester/
├── strategies/      # Trading strategy implementations
├── indicators/      # Technical indicators
├── signals/         # Signal generation engine
├── portfolio/       # Portfolio management
├── optimization/    # Parameter optimization
├── analysis/        # Performance analysis
├── data/           # Data fetching and storage
├── risk_management/ # Risk management tools
├── config/         # Configuration management
└── cli.py          # Command-line interface
```

### Core Design Principles

1. **Modularity**: Each component is independent and reusable
2. **Vectorization**: Use pandas/numpy for performance
3. **VectorBT Integration**: Leverage vectorbtpro capabilities
4. **Type Safety**: Use type hints for clarity
5. **Error Handling**: Graceful failure with informative messages

## Creating Custom Strategies

### Step 1: Inherit from BaseStrategy

```python
# backtester/strategies/my_custom_strategy.py
from backtester.strategies.base_strategy import BaseStrategy
import pandas as pd
import numpy as np
from typing import Dict, Optional

class MyCustomStrategy(BaseStrategy):
    """
    Custom strategy implementation.
    
    This strategy does X based on Y indicators.
    """
    
    def __init__(
        self,
        param1: int = 20,
        param2: float = 2.0,
        param3: bool = True
    ):
        """
        Initialize strategy with parameters.
        
        Args:
            param1: Description of param1
            param2: Description of param2
            param3: Description of param3
        """
        super().__init__()
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3
        
        # Validate parameters
        self._validate_parameters()
        
    def _validate_parameters(self) -> None:
        """Validate strategy parameters."""
        if self.param1 <= 0:
            raise ValueError("param1 must be positive")
        if self.param2 < 0:
            raise ValueError("param2 must be non-negative")
```

### Step 2: Implement Signal Generation

```python
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on strategy logic.
        
        Args:
            data: DataFrame with OHLCV columns
            
        Returns:
            Series with values: 1 (long), -1 (short), 0 (neutral)
        """
        # Extract price data
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Calculate indicators
        indicator1 = self._calculate_indicator1(close)
        indicator2 = self._calculate_indicator2(high, low, close)
        
        # Generate signals
        signals = pd.Series(0, index=data.index)
        
        # Long conditions
        long_condition = (
            (indicator1 > indicator1.shift(1)) &
            (indicator2 > self.param2) &
            (volume > volume.rolling(20).mean())
        )
        
        # Short conditions
        short_condition = (
            (indicator1 < indicator1.shift(1)) &
            (indicator2 < -self.param2) &
            (volume > volume.rolling(20).mean())
        )
        
        # Apply signals
        signals[long_condition] = 1
        signals[short_condition] = -1
        
        # Optional: Apply filters
        if self.param3:
            signals = self._apply_trend_filter(signals, close)
            
        return signals
```

### Step 3: Add Helper Methods

```python
    def _calculate_indicator1(self, close: pd.Series) -> pd.Series:
        """Calculate custom indicator 1."""
        return close.rolling(self.param1).mean()
        
    def _calculate_indicator2(
        self, 
        high: pd.Series, 
        low: pd.Series, 
        close: pd.Series
    ) -> pd.Series:
        """Calculate custom indicator 2."""
        typical_price = (high + low + close) / 3
        return (close - typical_price) / typical_price * 100
        
    def _apply_trend_filter(
        self, 
        signals: pd.Series, 
        close: pd.Series
    ) -> pd.Series:
        """Apply trend filter to signals."""
        trend_ma = close.rolling(200).mean()
        
        # Only long in uptrend
        signals[(close < trend_ma) & (signals == 1)] = 0
        
        # Only short in downtrend
        signals[(close > trend_ma) & (signals == -1)] = 0
        
        return signals
```

### Step 4: Add Strategy Metadata

```python
    def get_parameters(self) -> Dict:
        """Return strategy parameters."""
        return {
            'param1': self.param1,
            'param2': self.param2,
            'param3': self.param3
        }
        
    def get_required_columns(self) -> list:
        """Return required data columns."""
        return ['open', 'high', 'low', 'close', 'volume']
        
    def get_min_periods(self) -> int:
        """Return minimum periods needed for calculation."""
        return max(self.param1, 200 if self.param3 else 0)
```

### Step 5: Register the Strategy

Add to `backtester/strategies/__init__.py`:

```python
from .my_custom_strategy import MyCustomStrategy

__all__ = [
    # ... existing strategies
    'MyCustomStrategy'
]
```

## Developing New Indicators

### Basic Indicator Template

```python
# backtester/indicators/my_indicator.py
import pandas as pd
import numpy as np
from typing import Union, Tuple

def my_indicator(
    data: pd.Series,
    period: int = 14,
    multiplier: float = 2.0,
    method: str = 'ema'
) -> Union[pd.Series, Tuple[pd.Series, ...]]:
    """
    Calculate My Custom Indicator.
    
    Args:
        data: Input price series
        period: Lookback period
        multiplier: Scaling factor
        method: Calculation method ('sma', 'ema')
        
    Returns:
        Indicator values or tuple of values
        
    Example:
        >>> upper, middle, lower = my_indicator(close, period=20)
    """
    # Validate inputs
    if period <= 0:
        raise ValueError("Period must be positive")
    if method not in ['sma', 'ema']:
        raise ValueError(f"Unknown method: {method}")
        
    # Calculate base line
    if method == 'sma':
        middle = data.rolling(period).mean()
    else:
        middle = data.ewm(span=period, adjust=False).mean()
        
    # Calculate bands
    std = data.rolling(period).std()
    upper = middle + (multiplier * std)
    lower = middle - (multiplier * std)
    
    return upper, middle, lower
```

### Advanced Indicator with State

```python
# backtester/indicators/stateful_indicator.py
class AdaptiveIndicator:
    """Indicator that adapts based on market conditions."""
    
    def __init__(self, base_period: int = 20):
        self.base_period = base_period
        self._cache = {}
        
    def calculate(
        self, 
        data: pd.DataFrame,
        volatility_threshold: float = 0.02
    ) -> pd.Series:
        """
        Calculate adaptive indicator.
        
        The period adjusts based on volatility.
        """
        close = data['close']
        
        # Calculate volatility
        returns = close.pct_change()
        volatility = returns.rolling(self.base_period).std()
        
        # Adaptive period
        adaptive_period = pd.Series(self.base_period, index=close.index)
        high_vol = volatility > volatility_threshold
        adaptive_period[high_vol] = self.base_period * 2
        
        # Calculate indicator with adaptive period
        result = pd.Series(np.nan, index=close.index)
        
        for i in range(len(close)):
            if i < self.base_period:
                continue
                
            period = int(adaptive_period.iloc[i])
            if i >= period:
                result.iloc[i] = close.iloc[i-period:i].mean()
                
        return result
```

## Adding Risk Management Components

### Position Sizer Template

```python
# backtester/risk_management/my_position_sizer.py
from backtester.risk_management.base import BasePositionSizer
import pandas as pd
import numpy as np

class MyPositionSizer(BasePositionSizer):
    """Custom position sizing based on X."""
    
    def __init__(
        self,
        base_size: float = 0.1,
        scale_factor: float = 2.0,
        max_size: float = 0.5
    ):
        super().__init__()
        self.base_size = base_size
        self.scale_factor = scale_factor
        self.max_size = max_size
        
    def calculate_position_size(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        portfolio_value: float,
        current_positions: dict = None
    ) -> pd.Series:
        """
        Calculate position sizes.
        
        Args:
            data: OHLCV data
            signals: Trading signals
            portfolio_value: Current portfolio value
            current_positions: Current open positions
            
        Returns:
            Position sizes as fraction of portfolio
        """
        # Start with base size
        sizes = pd.Series(self.base_size, index=signals.index)
        
        # Scale based on some metric (e.g., momentum)
        close = data['close']
        momentum = close.pct_change(20)
        
        # Increase size for strong momentum
        strong_momentum = momentum.abs() > 0.1
        sizes[strong_momentum] *= self.scale_factor
        
        # Cap at maximum
        sizes = sizes.clip(upper=self.max_size)
        
        # Only size where we have signals
        sizes[signals == 0] = 0
        
        return sizes
```

### Stop Loss Manager

```python
# backtester/risk_management/my_stop_loss.py
class MyStopLoss:
    """Custom stop loss implementation."""
    
    def __init__(self, method: str = 'trailing', param: float = 0.02):
        self.method = method
        self.param = param
        
    def calculate_stop_levels(
        self,
        entry_prices: pd.Series,
        current_prices: pd.Series,
        high_prices: pd.Series = None
    ) -> pd.Series:
        """Calculate stop loss levels."""
        if self.method == 'fixed':
            # Fixed percentage below entry
            return entry_prices * (1 - self.param)
            
        elif self.method == 'trailing':
            # Trailing stop
            if high_prices is None:
                raise ValueError("High prices required for trailing stop")
                
            # Trail below highest price since entry
            trailing_high = high_prices.expanding().max()
            return trailing_high * (1 - self.param)
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
```

## Extending the Analysis Module

### Custom Performance Metrics

```python
# backtester/analysis/my_metrics.py
import pandas as pd
import numpy as np
from typing import Dict

def calculate_custom_metrics(
    returns: pd.Series,
    positions: pd.Series,
    prices: pd.Series
) -> Dict[str, float]:
    """
    Calculate custom performance metrics.
    
    Args:
        returns: Portfolio returns
        positions: Position sizes over time
        prices: Price series
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Custom Sharpe with different risk-free rate
    risk_free = 0.03 / 252  # 3% annual
    excess_returns = returns - risk_free
    metrics['custom_sharpe'] = (
        excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    )
    
    # Win rate by magnitude
    winning_returns = returns[returns > 0]
    losing_returns = returns[returns < 0]
    
    big_wins = winning_returns[winning_returns > 0.02]
    big_losses = losing_returns[losing_returns < -0.02]
    
    metrics['big_win_rate'] = len(big_wins) / len(returns) * 100
    metrics['big_loss_rate'] = len(big_losses) / len(returns) * 100
    
    # Average holding period
    position_changes = positions.diff().fillna(0)
    entries = position_changes != 0
    
    holding_periods = []
    current_period = 0
    
    for i, has_trade in enumerate(entries):
        if has_trade and current_period > 0:
            holding_periods.append(current_period)
            current_period = 0
        else:
            current_period += 1
            
    metrics['avg_holding_period'] = np.mean(holding_periods) if holding_periods else 0
    
    return metrics
```

### Custom Visualization

```python
# backtester/analysis/my_plots.py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def create_custom_dashboard(
    data: pd.DataFrame,
    portfolio_values: pd.Series,
    signals: pd.Series,
    custom_indicator: pd.Series
) -> go.Figure:
    """Create custom analysis dashboard."""
    
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.4, 0.2, 0.2, 0.2],
        subplot_titles=[
            'Price & Signals',
            'Custom Indicator',
            'Portfolio Value',
            'Drawdown'
        ]
    )
    
    # Price and signals
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Add buy/sell markers
    buys = data[signals == 1]
    sells = data[signals == -1]
    
    fig.add_trace(
        go.Scatter(
            x=buys.index,
            y=buys['low'] * 0.99,
            mode='markers',
            marker=dict(symbol='triangle-up', size=10, color='green'),
            name='Buy'
        ),
        row=1, col=1
    )
    
    # Custom indicator
    fig.add_trace(
        go.Scatter(
            x=custom_indicator.index,
            y=custom_indicator.values,
            name='Custom Indicator',
            line=dict(color='purple')
        ),
        row=2, col=1
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_values.index,
            y=portfolio_values.values,
            name='Portfolio Value',
            fill='tozeroy'
        ),
        row=3, col=1
    )
    
    # Drawdown
    rolling_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - rolling_max) / rolling_max * 100
    
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            name='Drawdown %',
            fill='tozeroy',
            line=dict(color='red')
        ),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='Custom Strategy Dashboard',
        xaxis_title='Date',
        height=1000,
        showlegend=True
    )
    
    return fig
```

## Working with the Data Module

### Custom Data Source

```python
# backtester/data/sources/my_data_source.py
from typing import Optional, List
import pandas as pd

class MyDataSource:
    """Custom data source implementation."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self._validate_connection()
        
    def fetch_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch data from custom source.
        
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        # Implementation here
        pass
        
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols."""
        # Implementation here
        pass
```

### Data Transformer

```python
# backtester/data/transformers/my_transformer.py
import pandas as pd

class MyDataTransformer:
    """Transform data for special use cases."""
    
    @staticmethod
    def add_custom_features(data: pd.DataFrame) -> pd.DataFrame:
        """Add custom features to OHLCV data."""
        # Add dollar volume
        data['dollar_volume'] = data['close'] * data['volume']
        
        # Add price ranges
        data['range'] = data['high'] - data['low']
        data['range_pct'] = data['range'] / data['close'] * 100
        
        # Add gaps
        data['gap'] = data['open'] - data['close'].shift(1)
        data['gap_pct'] = data['gap'] / data['close'].shift(1) * 100
        
        return data
```

## CLI Integration

### Adding New Commands

```python
# Add to backtester/cli.py

@click.command()
@click.option('--symbol', '-s', required=True, help='Trading symbol')
@click.option('--param1', '-p1', default=20, help='Parameter 1')
@click.option('--param2', '-p2', default=2.0, help='Parameter 2')
def mycommand(symbol: str, param1: int, param2: float):
    """Run my custom analysis."""
    click.echo(f"Running custom analysis for {symbol}")
    
    # Implementation
    from backtester.data import fetch_data
    from backtester.strategies import MyCustomStrategy
    
    # Fetch data
    data = fetch_data(symbols=[symbol])
    
    # Run strategy
    strategy = MyCustomStrategy(param1=param1, param2=param2)
    signals = strategy.generate_signals(data)
    
    # Analyze results
    # ...
    
# Register command
cli.add_command(mycommand)
```

## Testing Guidelines

### Unit Tests

```python
# tests/strategies/test_my_custom_strategy.py
import pytest
import pandas as pd
import numpy as np
from backtester.strategies import MyCustomStrategy

class TestMyCustomStrategy:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        dates = pd.date_range('2023-01-01', periods=100, freq='1h')
        
        # Generate realistic price data
        close = 100 + np.cumsum(np.random.randn(100) * 0.5)
        
        return pd.DataFrame({
            'open': close + np.random.randn(100) * 0.1,
            'high': close + abs(np.random.randn(100) * 0.2),
            'low': close - abs(np.random.randn(100) * 0.2),
            'close': close,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
    def test_initialization(self):
        """Test strategy initialization."""
        strategy = MyCustomStrategy(param1=30, param2=3.0)
        assert strategy.param1 == 30
        assert strategy.param2 == 3.0
        
    def test_invalid_parameters(self):
        """Test parameter validation."""
        with pytest.raises(ValueError):
            MyCustomStrategy(param1=-1)
            
    def test_signal_generation(self, sample_data):
        """Test signal generation."""
        strategy = MyCustomStrategy()
        signals = strategy.generate_signals(sample_data)
        
        # Check signal properties
        assert isinstance(signals, pd.Series)
        assert len(signals) == len(sample_data)
        assert all(s in [-1, 0, 1] for s in signals.dropna())
        
    def test_edge_cases(self):
        """Test edge cases."""
        strategy = MyCustomStrategy()
        
        # Empty data
        empty_data = pd.DataFrame()
        with pytest.raises(KeyError):
            strategy.generate_signals(empty_data)
            
        # Single row
        single_row = pd.DataFrame({
            'close': [100],
            'high': [101],
            'low': [99],
            'volume': [1000]
        })
        signals = strategy.generate_signals(single_row)
        assert len(signals) == 1
```

### Integration Tests

```python
# tests/integration/test_strategy_integration.py
import pytest
from backtester.data import fetch_data
from backtester.strategies import MyCustomStrategy
from backtester.portfolio import PortfolioManager

def test_full_backtest_flow():
    """Test complete backtest workflow."""
    # Fetch data
    data = fetch_data(
        symbols=['BTC/USDT'],
        timeframe='1h',
        start_date='2023-01-01',
        end_date='2023-01-31'
    )
    
    # Initialize strategy
    strategy = MyCustomStrategy(param1=20, param2=2.0)
    
    # Run backtest
    portfolio = PortfolioManager(initial_cash=100000)
    results = portfolio.run_backtest(
        data=data,
        strategy=strategy,
        commission=0.001
    )
    
    # Verify results
    assert 'total_return' in results
    assert 'sharpe_ratio' in results
    assert results['num_trades'] > 0
```

## Performance Optimization

### Vectorization Best Practices

```python
# Good - Vectorized
def calculate_signals_vectorized(data: pd.DataFrame) -> pd.Series:
    close = data['close']
    sma_fast = close.rolling(10).mean()
    sma_slow = close.rolling(30).mean()
    
    # Vectorized comparison
    signals = pd.Series(0, index=data.index)
    signals[sma_fast > sma_slow] = 1
    signals[sma_fast < sma_slow] = -1
    
    return signals

# Bad - Loop-based
def calculate_signals_loop(data: pd.DataFrame) -> pd.Series:
    signals = []
    
    for i in range(len(data)):
        if i < 30:
            signals.append(0)
            continue
            
        sma_fast = data['close'].iloc[i-10:i].mean()
        sma_slow = data['close'].iloc[i-30:i].mean()
        
        if sma_fast > sma_slow:
            signals.append(1)
        elif sma_fast < sma_slow:
            signals.append(-1)
        else:
            signals.append(0)
            
    return pd.Series(signals, index=data.index)
```

### Memory Optimization

```python
# Use views instead of copies
def process_data_efficient(data: pd.DataFrame) -> pd.DataFrame:
    # Good - modifies in place
    data['returns'] = data['close'].pct_change()
    
    # Good - uses view
    subset = data.loc['2023-01-01':'2023-12-31']
    
    # Bad - creates copy
    # subset = data[data.index >= '2023-01-01'].copy()
    
    return data

# Use generators for large datasets
def process_chunks(data: pd.DataFrame, chunk_size: int = 10000):
    """Process data in chunks to save memory."""
    for start in range(0, len(data), chunk_size):
        end = min(start + chunk_size, len(data))
        yield data.iloc[start:end]
```

## Best Practices

### 1. Code Organization

```python
# Good structure
backtester/
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py
│   ├── my_custom_strategy.py
│   └── tests/
│       └── test_my_custom_strategy.py
```

### 2. Documentation

```python
def my_function(param1: int, param2: float = 1.0) -> pd.Series:
    """
    Brief description of function.
    
    Longer description explaining the purpose,
    algorithm, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2 (default: 1.0)
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When param1 is negative
        
    Example:
        >>> result = my_function(10, 2.5)
        >>> print(result.head())
    """
    pass
```

### 3. Error Handling

```python
def robust_calculation(data: pd.Series) -> pd.Series:
    """Example of robust error handling."""
    try:
        # Validate input
        if data.empty:
            raise ValueError("Input data is empty")
            
        if not isinstance(data, pd.Series):
            raise TypeError("Input must be pandas Series")
            
        # Perform calculation
        result = data.rolling(20).mean()
        
        # Check output
        if result.isna().all():
            logger.warning("All values are NaN")
            
        return result
        
    except Exception as e:
        logger.error(f"Calculation failed: {e}")
        # Return empty series with same index
        return pd.Series(index=data.index)
```

### 4. Configuration

```python
# Use configuration files
{
    "strategy": {
        "name": "MyCustomStrategy",
        "parameters": {
            "param1": 20,
            "param2": 2.0,
            "param3": true
        }
    },
    "risk_management": {
        "position_size": 0.1,
        "stop_loss": 0.02,
        "take_profit": 0.04
    }
}
```

### 5. Logging

```python
import logging
from backtester.utilities.logging import setup_logger

logger = setup_logger(__name__)

class MyStrategy(BaseStrategy):
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        logger.info(f"Generating signals for {len(data)} periods")
        
        try:
            signals = self._calculate_signals(data)
            logger.debug(f"Generated {signals.sum()} non-zero signals")
            return signals
            
        except Exception as e:
            logger.error(f"Signal generation failed: {e}", exc_info=True)
            raise
```

## Conclusion

This guide provides the foundation for extending the backtester framework. Remember to:

1. Follow the established patterns and conventions
2. Write comprehensive tests for new components
3. Document your code thoroughly
4. Optimize for performance using vectorization
5. Handle errors gracefully
6. Integrate with existing modules where possible

For specific questions or advanced use cases, refer to the existing implementations in the codebase or create an issue for discussion. 