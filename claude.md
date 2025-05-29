# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# VectorBT Pro Backtesting Framework

A comprehensive Python backtesting framework built on VectorBT Pro for cryptocurrency trading strategy development and optimization.

## Common Development Commands

### Running Backtests
```bash
# Run single symbol backtest
python -m backtester backtest --symbol BTC/USDT --timeframe 1h

# Run parameter optimization
python -m backtester optimize --symbol BTC/USDT --timeframe 4h --metric sharpe_ratio

# Run portfolio backtest
python -m backtester portfolio --symbols BTC/USDT,ETH/USDT --timeframe 4h

# Use comprehensive strategy tester
python examples/strategy_tester.py --symbols BTC/USDT --timeframes 4h --optimize
```

### Example Scripts (Numbered Learning Path)
```bash
# Progressive learning examples
python examples/01_single_symbol_backtest.py
python examples/02_parameter_optimization.py
python examples/03_multi_symbol_portfolio.py
python examples/04_portfolio_optimization.py
python examples/05_walk_forward_analysis.py
python examples/06_multi_timeframe_strategy.py
python examples/07_advanced_mtf_strategy.py
python examples/08_advanced_risk_management.py
python examples/09_pairs_trading.py
python examples/10_momentum_strategy.py
```

### Data Management Commands
```bash
# Fetch data for specific symbols
python -m backtester.data.cli.fetch --exchange binance --market spot --symbols BTC/USDT,ETH/USDT --timeframe 1h

# Fetch top symbols by volume
python -m backtester.data.cli.fetch --exchange binance --market spot --top 10 --timeframe 4h --inception

# Check data health
python -m backtester.data.cli.inspect --symbols BTC/USDT --timeframe 1h

# Refresh cached data
python -m backtester.data.cli.refresh --symbols BTC/USDT --timeframe 1h

# Interpolate missing data
python -m backtester.data.cli.interpolate --symbols BTC/USDT --timeframe 1h
```

### Package Installation
```bash
# Install the framework in development mode
pip install -e .
```

### Testing Approach
- Create temporary `test_*.py` files for feature verification
- Use `examples/strategy_tester.py` for comprehensive testing
- Delete test files after verification (part of the framework's philosophy)
- Focus on VBT's built-in testing utilities

### Code Quality Commands
```bash
# Run linting (if configured)
# Check package.json or setup.cfg for specific lint commands
python -m pylint backtester
python -m flake8 backtester

# Run type checking (if configured)
python -m mypy backtester

# Run tests (if test suite exists)
python -m pytest tests/
```

## Architecture Overview

### Core Philosophy
- **VBT-First**: Always use VectorBT Pro's built-in functionality before custom implementations
- **Module Boundaries**: Strict separation of concerns between data, strategies, signals, portfolio, and analysis
- **Test and Delete**: Create temporary test files during development, delete after verification
- **Configuration-Driven**: External parameter storage with database persistence

### Module Structure

#### Data Module (`backtester/data/`)
- **Primary Interface**: Use `fetch_data()` - returns `vbt.Data` objects
- **Caching System**: Automatic caching to minimize API calls via `cache_system/`
- **CLI Tools**: Data fetching, inspection, refresh, and interpolation utilities in `cli/`
- **Multi-Exchange**: Supports Binance, Bybit, Hyperliquid through unified interface
- **Storage**: Persistent storage in `vbt_data/` with .pickle.blosc format
- **MTF Support**: Multi-timeframe data handling in `mtf/` subdirectory

#### Strategy Module (`backtester/strategies/`)
- **Base Class**: All strategies inherit from `BaseStrategy`
- **Pure Functions**: Strategies ONLY calculate indicators and generate signals
- **No Portfolio Logic**: Portfolio simulation handled separately
- **Signal Structure**: Standardized format with entry/exit signals and risk levels

#### Indicators Module (`backtester/indicators/`)
- **VBT-Based**: All indicators built on VectorBT Pro's indicator framework
- **Custom Indicators**: Extended indicators not available in standard VBT
- **Caching**: Indicator results cached for performance

#### Signals Module (`backtester/signals/`)
- **Signal Processing**: Validation, cleaning, and preparation
- **Risk Management**: Stop-loss and take-profit level calculation
- **Quality Metrics**: Signal strength and reliability analysis
- **Signal Engine**: Centralized signal generation and management

#### Portfolio Module (`backtester/portfolio/`)
- **Simulation Engine**: Uses `vbt.Portfolio.from_signals()` exclusively
- **Position Sizing**: Dynamic position sizing with `PositionSizeMode` enum
- **Risk Management**: Integration with stop-loss and risk management systems
- **Portfolio Config**: `SimulationConfig` dataclass for standardized settings

#### Optimization Module (`backtester/optimization/`)
- **Parameter Optimization**: Grid search and advanced optimization methods
- **Walk-Forward**: Time series validation and out-of-sample testing
- **Result Storage**: Automatic storage in SQLite database

#### Risk Management Module (`backtester/risk_management/`)
- **Position Control**: Maximum position sizes and exposure limits
- **Stop Loss**: Dynamic and fixed stop-loss strategies
- **Risk Metrics**: Real-time risk assessment and monitoring

#### Analysis Module (`backtester/analysis/`)
- **TradingChartsEngine**: Unified interactive plotting with enhanced zoom/pan
- **PerformanceAnalyzer**: Comprehensive performance metrics calculation
- **BenchmarkAnalyzer**: Strategy comparison and relative performance
- **MTFPlottingEngine**: Multi-timeframe visualization

### Configuration System (`backtester/config/`)
- **Strategy Parameters**: JSON files organized in subdirectories:
  - `development/` - Parameters under active development
  - `production/` - Validated production parameters
  - `archived/` - Historical parameter sets
  - `examples/` - Example parameter configurations
  - `templates/` - Parameter templates for new strategies
- **Optimal Parameters Database**: SQLite storage in `optimal_parameters/optimal_parameters.db`
- **Configuration Classes**: `ConfigManager` and `OptimalParametersDB` for management

### Critical Module Boundaries

**ALWAYS use module interfaces - NEVER bypass them:**

```python
# ✅ CORRECT - Use data module interface
from backtester.data import fetch_data
data = fetch_data(['BTC/USDT'], timeframe='1h')

# ❌ WRONG - Bypassing data module
import vectorbtpro as vbt
data = vbt.CCXTData.fetch(...)  # Don't do this
```

**Module Communication Flow:**
1. Data Module → Provides `vbt.Data` objects
2. Strategy Module → Calculates indicators, generates signal dictionaries  
3. Signals Module → Validates and prepares signals for simulation
4. Portfolio Module → Simulates portfolio using prepared signals
5. Analysis Module → Analyzes portfolio results

### Results Organization

The framework organizes results by analysis type:

```
results/
├── symbols/              # Individual symbol analysis
│   ├── BTC_USDT/
│   │   ├── optimization/ # Parameter optimization results
│   │   └── plots/        # Interactive visualizations
├── portfolios/           # Multi-symbol portfolio analysis
│   └── custom_names/     # User-defined portfolio names
└── general/              # Cross-strategy comparisons
    └── testing/          # General testing results
```

### Database Integration

- **Location**: `backtester/config/optimal_parameters/optimal_parameters.db`
- **Purpose**: Persistent storage of optimized parameters
- **Automatic**: Parameters stored and retrieved automatically via `OptimalParametersDB`
- **Walk-Forward Ready**: Date range tracking for time-series validation

## VectorBT Pro Framework Rules

## **Core Philosophy**
- **Less is More**: Write minimal, efficient code that maximizes VectorBT Pro's built-in functionality
- **VBT-First Approach**: Always check if VectorBT Pro has a method before implementing custom solutions
- **Documentation-Driven**: Consult `vectorbtpro_docs/` frequently before and during implementation
- **Modular Design**: Keep components separate and reusable for easy strategy creation
- **Test and Delete**: Create test files during development, delete them once functionality is verified

## **Architecture Principles**

### **Module Structure**
- **Data Module** (`backtester/data/`): Handles all data fetching, caching, and storage
  - Use `fetch_data()` as the primary interface - it returns `vbt.Data` objects
  - Leverage VBT's native caching and persistence
  - Never create custom data structures when VBT provides them

- **Strategy Module** (`backtester/strategies/`): Contains trading strategy implementations
  - All strategies inherit from `BaseStrategy`
  - Strategies ONLY calculate indicators and generate signals
  - Portfolio simulation is handled separately by the portfolio module

- **Portfolio Module** (`backtester/portfolio/`): Manages portfolio simulation and optimization
  - Uses `vbt.Portfolio.from_signals()` for all simulations
  - Supports advanced features like dynamic stops, multi-asset portfolios
  - Integrates with portfolio optimization libraries

- **Analysis Module** (`backtester/analysis/`): Provides performance analysis and visualization
  - Leverages VBT's built-in metrics and plotting
  - Extends with custom analysis when needed

## **Module Boundaries and Interfaces**

### **CRITICAL: Use Module Interfaces, Not Direct VBT Calls**

Each module has specific entry points that MUST be used. Direct VBT calls should only happen within the appropriate module.

### **Data Access Rules**
```python
# ✅ ALWAYS: Use the data module interface
from backtester.data import fetch_data, quick_fetch, update_data, load_cached

# Fetch data through the module
data = fetch_data(['BTC/USDT'], timeframe='1h', start_date='2023-01-01')

# ❌ NEVER: Bypass the data module
import vectorbtpro as vbt
data = vbt.YFData.pull(['BTC-USD'])  # Don't do this
data = vbt.CCXTData.fetch(...)  # Don't do this either
```

**Why**: The data module handles:
- Automatic caching to minimize API calls
- Exchange-specific symbol formatting
- Metadata updates (volumes, timestamps)
- Consistent data structure across all exchanges

### **Signal Generation Rules**
```python
# ✅ ALWAYS: Use the signals module for signal generation
from backtester.signals import SignalEngine, generate_ma_crossover_signals
from backtester.signals.signal_utils import prepare_signals, validate_signals

# Generate signals through the module
signal_result = generate_ma_crossover_signals(short_ma, long_ma)
signals = prepare_signals(signal_dict, data.index)

# ❌ NEVER: Generate signals directly in strategies without the module
entries = close > sma  # Don't do this inline
exits = close < sma    # Use signal module functions
```

**Why**: The signals module provides:
- Standardized signal validation
- Signal cleaning (opposing signals, entry/exit pairs)
- Risk level calculation (SL/TP)
- Signal quality metrics

### **Portfolio Simulation Rules**
```python
# ✅ ALWAYS: Use the portfolio module for simulations
from backtester.portfolio import PortfolioSimulator, SimulationConfig

# Create simulator and run simulation
config = SimulationConfig(init_cash=10000, fees=0.001)
simulator = PortfolioSimulator(data, config)
portfolio = simulator.simulate_from_signals(signals)

# ❌ NEVER: Call vbt.Portfolio directly in strategies or analysis
portfolio = vbt.Portfolio.from_signals(...)  # Don't do this outside portfolio module
```

**Why**: The portfolio module handles:
- Position sizing strategies
- Risk management integration
- Multi-asset portfolio coordination
- Performance tracking and analysis

### **Module Communication Flow**

```
1. Data Module → Provides vbt.Data objects
2. Strategy Module → Uses data to calculate indicators and generate signal dictionaries
3. Signals Module → Validates and prepares signals for portfolio simulation
4. Portfolio Module → Simulates portfolio using prepared signals
5. Analysis Module → Analyzes portfolio results
```

### **Example: Correct Module Usage**
```python
# 1. Fetch data through data module
from backtester.data import fetch_data
data = fetch_data(['BTC/USDT', 'ETH/USDT'], timeframe='4h')

# 2. Create strategy that uses the data
from backtester.strategies import DMAATRStrategy
strategy = DMAATRStrategy(fast_period=20, slow_period=50)
indicators = strategy.calculate_indicators(data)
raw_signals = strategy.generate_signals(data, indicators)

# 3. Prepare signals through signals module
from backtester.signals.signal_utils import prepare_signals
signals = prepare_signals(raw_signals, data.index)

# 4. Simulate through portfolio module
from backtester.portfolio import PortfolioSimulator, SimulationConfig
config = SimulationConfig(init_cash=10000, fees=0.001)
simulator = PortfolioSimulator(data, config)
portfolio = simulator.simulate_from_signals(signals)

# 5. Analyze through analysis module
from backtester.analysis import TradingChartsEngine, PerformanceAnalyzer
charts = TradingChartsEngine(portfolio, data, indicators)
analyzer = PerformanceAnalyzer(portfolio)
report = analyzer.generate_report()
```

### **Module-Specific VBT Usage**

**Data Module** - Can use:
- `vbt.CCXTData`, `vbt.YFData` for fetching
- `vbt.Data` for data structure
- VBT's caching decorators

**Strategy Module** - Can use:
- VBT indicators (`vbt.MA`, `vbt.RSI`, etc.)
- Signal methods (`crossed_above`, `crossed_below`)
- NO portfolio methods

**Portfolio Module** - Can use:
- `vbt.Portfolio.from_signals()`
- Portfolio optimization methods
- Position sizing utilities

**Analysis Module** - Can use:
- Portfolio metrics and stats
- VBT plotting functions
- Custom analysis on portfolio results

## **VectorBT Pro Best Practices**

### **Data Handling**
```python
# ✅ DO: Use vbt.Data for all data operations
data = vbt.CCXTData.fetch(
    symbols=['BTC/USDT'],
    exchange='binance',
    timeframe='1h',
    start='2023-01-01',
    end='2023-12-31'
)

# ❌ DON'T: Use pandas DataFrames directly for data fetching
df = pd.DataFrame(...)  # Avoid this
```

### **Signal Generation**
```python
# ✅ DO: Use VBT's signal methods
long_entries = sma.close_crossed_above(data.close)
long_exits = sma.close_crossed_below(data.close)

# ✅ DO: Use VBT's signal cleaning
clean_entries, clean_exits = long_entries.vbt.signals.clean(long_exits)
```

### **Portfolio Simulation**
```python
# ✅ DO: Use SimulationConfig for configuration
config = SimulationConfig(
    size=0.95,  # 95% of available cash
    size_type='percent',
    fees=0.001,
    slippage=0.001,
    init_cash=10000
)

portfolio = vbt.Portfolio.from_signals(
    data.close,
    entries=entries,
    exits=exits,
    **config.to_dict()
)
```

### **Indicator Calculation**
```python
# ✅ DO: Use VBT's indicator classes
sma = vbt.MA.run(data.close, window=20)
rsi = vbt.RSI.run(data.close, window=14)
bbands = vbt.BBANDS.run(data.close, window=20, alpha=2)

# ❌ DON'T: Implement indicators from scratch
def calculate_sma(prices, window):  # Avoid this
    return prices.rolling(window).mean()
```

## **Code Organization Rules**

### **Strategy Development**
1. **Inherit from BaseStrategy**: All strategies must extend the base class
2. **Implement Required Methods**:
   - `calculate_indicators()`: Calculate all indicators
   - `generate_signals()`: Create entry/exit signals
   - `get_parameters()`: Return strategy parameters
   - `validate_parameters()`: Validate parameter values

3. **Signal Structure**:
   ```python
   signals = {
       'long_entries': pd.Series,   # Boolean series
       'long_exits': pd.Series,     # Boolean series
       'short_entries': pd.Series,  # Boolean series
       'short_exits': pd.Series,    # Boolean series
       'sl_levels': pd.Series,      # Numeric series (optional)
       'tp_levels': pd.Series       # Numeric series (optional)
   }
   ```

### **File Management**
- **One README per module**: Maintain a single README.md in each module directory
- **One LESSONS_LEARNED.md**: Document insights and patterns discovered
- **No proliferation of markdown files**: Avoid creating multiple documentation files
- **Test files are temporary**: Create `test_*.py` files during development, delete after verification

### **Import Organization**
```python
# ✅ DO: Organize imports properly
import vectorbtpro as vbt
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from backtester.strategies.base_strategy import BaseStrategy
from backtester.data import fetch_data
from backtester.utilities.structured_logging import get_logger

# ❌ DON'T: Use wildcard imports or mix import styles
from backtester.strategies import *  # Avoid
```

## **Development Workflow**

### **Testing Approach**
1. Create focused test files: `test_feature.py`
2. Test using VBT's methods: `assert portfolio.total_return > 0`
3. Delete test files after verification
4. Document lessons learned in module's LESSONS_LEARNED.md

### **VBT Documentation Usage**
- **Primary Reference**: Always consult `vectorbtpro_docs/` first
- **Terminal Exploration**: Use `vbt.help()`, `dir(vbt.Portfolio)` to explore methods
- **Avoid Private Methods**: Never use methods starting with underscore
- **Check Examples**: Reference the `examples/` directory for patterns

### **Performance Optimization**
```python
# ✅ DO: Use VBT's vectorized operations
results = vbt.Portfolio.from_signals(
    close=data.close,
    entries=entries,
    exits=exits,
    param_product=True  # Test multiple parameters efficiently
)

# ❌ DON'T: Use loops for parameter optimization
for param in params:
    portfolio = simulate(param)  # Avoid loops
```

## **Error Handling and Logging**

### **Structured Logging**
```python
# ✅ DO: Use the structured logging system
from backtester.utilities.structured_logging import get_logger
logger = get_logger(__name__)

logger.info("Starting backtest")
with logger.operation("Loading data"):
    data = fetch_data(symbols)

# ❌ DON'T: Use print statements or basic logging
print("Starting backtest")  # Avoid
```

### **Error Handling**
```python
# ✅ DO: Handle VBT-specific exceptions
try:
    portfolio = vbt.Portfolio.from_signals(**params)
except vbt.portfolio.base.CashError:
    logger.error("Insufficient cash for position sizing")
    
# ✅ DO: Validate data before processing
if not isinstance(data, vbt.Data):
    raise TypeError("Strategy requires vbt.Data object")
```

## **Configuration Management**

### **Parameter Storage**
- Use JSON files in `backtester/config/` for strategy parameters
- Store optimal parameters in SQLite database
- Use `SimulationConfig` dataclass for portfolio settings

### **Exchange Configuration**
- CCXT is only for fetching exchange metadata
- All price data must go through VBT's data classes
- Cache exchange data to minimize API calls

## **Multi-Timeframe and Multi-Asset**

### **MTF Strategies**
```python
# ✅ DO: Use VBT's data alignment features
data_1h = fetch_data(symbol, timeframe='1h')
data_4h = data_1h.resample('4h')  # Use VBT's resampling

# Align signals properly
aligned_signals = signals_1h.vbt.align_to(data_4h.index)
```

### **Multi-Asset Portfolios**
```python
# ✅ DO: Use VBT's multi-asset capabilities
data = vbt.YFData.pull(['BTC-USD', 'ETH-USD', 'SOL-USD'])
portfolio = vbt.Portfolio.from_signals(
    data.close,
    entries,
    exits,
    group_by=True,  # Group into single portfolio
    cash_sharing=True  # Share cash across assets
)
```

## **Future Development Guidelines**

### **Adding New Strategies**
1. Create new file in `backtester/strategies/`
2. Inherit from appropriate base class
3. Implement only indicator calculation and signal generation
4. Let portfolio module handle simulation
5. Test with example script, then delete test file

### **Extending Functionality**
- **Prefer composition over inheritance**
- **Reuse VBT components**: Check if VBT already provides the functionality
- **Document VBT usage**: Add examples showing VBT methods used
- **Keep modules independent**: Avoid circular dependencies

### **Performance Considerations**
- **Vectorize everything**: Use VBT's vectorized operations
- **Cache expensive operations**: Use VBT's caching decorators
- **Minimize data copies**: Work with views when possible
- **Profile with VBT tools**: Use `vbt.profile()` for performance analysis

## **Common Pitfalls to Avoid**

1. **Don't bypass module interfaces**: Always use `fetch_data()` from data module, not direct VBT data fetching
2. **Don't generate signals outside signals module**: Use the signals module for all signal generation and validation
3. **Don't simulate portfolios in strategies**: Strategies generate signals only; portfolio module handles simulation
4. **Don't bypass VBT's data structure**: Always use `vbt.Data` objects from the data module
5. **Don't implement custom indicators**: Check VBT's extensive indicator library first
6. **Don't use pandas operations on signals**: Use VBT's signal methods
7. **Don't create unnecessary files**: Test inline or with temporary files
8. **Don't ignore VBT's portfolio features**: Use stops, sizing, and advanced features through portfolio module
9. **Don't hardcode parameters**: Use the config system
10. **Don't skip the cache**: Use the caching system through the data module for expensive operations
11. **Don't mix module responsibilities**: Each module has a specific purpose - respect the boundaries

## **Quick Reference**

### **Module Entry Points**
```python
# Data Module - ALWAYS use these for data access
from backtester.data import (
    fetch_data,          # Primary data fetching - returns vbt.Data objects
    fetch_top_symbols,   # Get top symbols by volume
    update_data,         # Update cached data
    quick_fetch,         # Simple single-symbol fetch
    load_cached          # Load from cache
)

# Signals Module - Use for signal generation
from backtester.signals import (
    SignalEngine,
    generate_ma_crossover_signals,
    generate_threshold_signals,
    generate_atr_risk_levels
)
from backtester.signals.signal_utils import (
    prepare_signals,
    validate_signals,
    clean_signals
)

# Portfolio Module - Use for simulation
from backtester.portfolio import (
    PortfolioSimulator,
    SimulationConfig,
    create_position_sizer,
    calculate_position_sizes
)

# Strategy Module - Base classes
from backtester.strategies import BaseStrategy

# Analysis Module  
from backtester.analysis import TradingChartsEngine, PerformanceAnalyzer
```

### **Essential VBT Methods (Use Within Appropriate Modules)**
- Data: `vbt.Data` objects from data module only
- Indicators: `data.run('talib:RSI')`, `close.vbt.ma.run()` in strategies
- Signals: `vbt.signals.clean()`, `crossed_above()`, `crossed_below()` in signals module
- Portfolio: `vbt.Portfolio.from_signals()` in portfolio module only
- Analysis: `portfolio.plot()`, `portfolio.trades.plot()` in analysis module

Remember: **Always use module interfaces, never bypass them with direct VBT calls!**

# Development Workflow

## Core Development Principles

- **Clean Code Philosophy**: Write less code that does more by leveraging VectorBT Pro's capabilities
- **Test-Driven Development**: Create test files to verify functionality, then delete them
- **Documentation-First**: Always check `vectorbtpro_docs/` before implementing features
- **Modular Architecture**: Keep components independent and reusable
- **Module Boundaries**: NEVER bypass module interfaces - always use the designated entry points

## Standard Development Process

### 1. Feature Planning
- Check if VectorBT Pro already provides the functionality
- Review existing modules to avoid duplication
- Plan integration with existing architecture

### 2. Implementation Workflow
```python
# Step 1: Create a test file (e.g., test_feature.py)
import vectorbtpro as vbt
from backtester.data import fetch_data

# Step 2: Test your implementation
data = fetch_data(['BTC/USDT'], '1h', limit=1000)
# ... test code ...

# Step 3: Verify results
print(results)

# Step 4: Integrate into appropriate module
# Step 5: Delete test file
```

### 3. Code Integration
- Add new functionality to existing modules when possible
- Create new files only when necessary for modularity
- Update examples in `examples/` directory
- Avoid creating documentation files

## Strategy Development Workflow

### 1. Create Strategy Class
```python
# backtester/strategies/my_strategy.py
from backtester.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, **params):
        super().__init__(**params)
        # Validate parameters
    
    def calculate_indicators(self, data):
        # Use VBT indicators
        pass
    
    def generate_signals(self, data, indicators):
        # Generate entry/exit signals
        pass
```

### 2. Test Strategy
```python
# Use examples/strategy_tester.py
from backtester.strategies.my_strategy import MyStrategy

# Test with fixed date range for consistency
strategy = MyStrategy(param1=value1, param2=value2)
results = test_strategy(strategy, symbols=['BTC/USDT'], timeframe='1h')
```

### 3. Optimize Parameters
- Use existing optimization framework
- Store optimal parameters in database
- Add to `config/strategy_params/`

## Data Management Workflow

### 1. Data Fetching
```python
# Always use the standard interface
from backtester.data import fetch_data

data = fetch_data(
    symbols=['BTC/USDT', 'ETH/USDT'],
    timeframe='1h',
    start='2023-01-01',
    end='2023-12-31'
)
```

### 2. Data Caching
- Automatic caching is handled by the framework
- Check cache before fetching
- Use appropriate cache keys

## Testing Best Practices

### 1. Unit Testing
- Test individual components in isolation
- Use VBT's testing utilities
- Focus on edge cases

### 2. Integration Testing
- Test complete strategies
- Verify portfolio calculations
- Check performance metrics

### 3. Performance Testing
- Use consistent date ranges
- Compare against benchmarks
- Monitor execution time

## Configuration Management

### 1. Strategy Parameters
```yaml
# config/strategy_params/production/strategy_name.yaml
BTC/USDT:
  param1: value1
  param2: value2
ETH/USDT:
  param1: value3
  param2: value4
```

### 2. Environment Configuration
- Use `.env` for sensitive data
- Keep exchange configs centralized
- Version control non-sensitive configs

## Performance Optimization

### 1. Use VBT's Optimization Features
- Leverage vectorized operations
- Enable parallel processing
- Use caching decorators

### 2. Memory Management
- Process data in chunks for large datasets
- Clear unnecessary variables
- Use VBT's memory-efficient methods

## Common Development Patterns

### 1. Adding New Indicators
```python
# Check if VBT has it first
indicator = vbt.IndicatorName.run(data.close, **params)

# If custom needed, extend VBT
class CustomIndicator(vbt.indicators.factory.IndicatorFactory):
    # Implementation
```

### 2. Extending Functionality
- Build on top of VBT classes
- Use composition over inheritance
- Keep extensions minimal

### 3. Error Handling
```python
from backtester.utilities.logging import get_logger

logger = get_logger(__name__)

try:
    # Operation
except vbt.portfolio.errors.CashError:
    logger.error("Insufficient cash")
    # Handle appropriately
```

## Code Review Checklist

- [ ] Uses VBT methods where available
- [ ] No duplicate functionality
- [ ] Proper error handling
- [ ] Appropriate logging
- [ ] Test files deleted
- [ ] Configuration externalized
- [ ] Examples updated if needed
- [ ] No unnecessary files created

## Development Workflow

### 1. Strategy Development
```python
# Inherit from BaseStrategy
from backtester.strategies.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def calculate_indicators(self, data):
        # Use VBT indicators only
        pass
    
    def generate_signals(self, data, indicators):
        # Return signal dictionary
        pass
```

### 2. Testing Process
```python
# Create temporary test file
# test_my_feature.py
from backtester.data import fetch_data
data = fetch_data(['BTC/USDT'], '1h', limit=1000)
# ... test implementation ...
# Delete file after verification
```

### 3. Configuration
- Add parameters to `config/strategy_params/`
- Use database for optimal parameters
- External configuration for all settings

### 4. Analysis
- Use module interfaces for all operations
- Leverage VBT's built-in analysis tools
- Generate interactive visualizations

## VectorBT Pro Integration

### Mandatory VBT Usage Patterns

```python
# Data handling
data = vbt.CCXTData.fetch(...)  # Only in data module
portfolio = vbt.Portfolio.from_signals(...)  # Only in portfolio module

# Indicators (in strategy module)
sma = vbt.MA.run(data.close, window=20)
rsi = vbt.RSI.run(data.close, window=14)

# Signals (in signals module)
entries = price.vbt.crossed_above(sma)
exits = price.vbt.crossed_below(sma)

# Analysis (in analysis module)  
portfolio.plot()
portfolio.trades.plot()
```

### Performance Optimization
- Use VBT's vectorized operations
- Enable parallel processing with `param_product=True`
- Leverage VBT's caching decorators
- Avoid Python loops for parameter optimization

## Key Development Principles

1. **Module Boundaries**: Never bypass module interfaces
2. **VBT-First**: Check VBT capabilities before custom implementation
3. **Test and Delete**: Temporary test files, permanent features
4. **Configuration External**: No hardcoded parameters
5. **Documentation-Driven**: Consult `vectorbtpro_docs/` frequently
6. **Clean Results**: Organized output structure for analysis

## Common Pitfalls to Avoid

1. **Don't bypass data module**: Always use `fetch_data()`, never direct VBT data calls
2. **Don't simulate in strategies**: Strategies generate signals only
3. **Don't ignore VBT features**: Use built-in portfolio, indicators, and analysis tools
4. **Don't create unnecessary files**: Follow test-and-delete philosophy
5. **Don't hardcode parameters**: Use configuration system
6. **Don't mix module responsibilities**: Respect architectural boundaries

## Quick Reference

### Essential Module Imports
```python
# Data access
from backtester.data import fetch_data, quick_fetch, load_cached

# Strategy development  
from backtester.strategies import BaseStrategy

# Signal processing
from backtester.signals.signal_utils import prepare_signals, validate_signals

# Portfolio simulation
from backtester.portfolio import PortfolioSimulator, SimulationConfig

# Analysis
from backtester.analysis import TradingChartsEngine, PerformanceAnalyzer
```

### Documentation Resources
- `vectorbtpro_docs/`: Primary VBT reference documentation
- `examples/`: Comprehensive strategy testing examples
- `docs/`: Framework-specific guides and API documentation

Remember: **The best code is the code you don't have to write - leverage VectorBT Pro's extensive capabilities!**