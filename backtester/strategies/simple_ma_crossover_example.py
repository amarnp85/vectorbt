"""
Simple Moving Average Crossover Strategy Example

This module demonstrates a minimal, clean implementation of a trading strategy
using VectorBTPro's native capabilities. It serves as a template for building
more complex strategies while maintaining proper separation of concerns.

Key Features:
- Direct use of VectorBTPro indicators
- Automatic multi-symbol support via broadcasting
- Proper signal cleaning using VBT utilities
- Clear separation from portfolio management
- Support for parameter optimization

Usage:
    # Single symbol
    data = vbt.YFData.fetch("AAPL", start="2023-01-01", end="2023-12-31")
    strategy = SimpleMAStrategy(data, {'fast_window': 10, 'slow_window': 30})
    result = strategy.run_signal_generation()
    
    # Multi-symbol (VBT handles broadcasting automatically)
    data = vbt.YFData.fetch(["AAPL", "MSFT", "GOOGL"], start="2023-01-01", end="2023-12-31")
    strategy = SimpleMAStrategy(data, {'fast_window': 10, 'slow_window': 30})
    result = strategy.run_signal_generation()
    
    # Parameter optimization
    params = {
        'fast_window': vbt.Param([5, 10, 15, 20]),
        'slow_window': vbt.Param([20, 30, 40, 50])
    }
    strategy = SimpleMAStrategy(data, params)
    result = strategy.run_signal_generation()
"""

import pandas as pd
import vectorbtpro as vbt
from typing import Dict, Any, Optional

from .base_strategy import BaseStrategy


class SimpleMAStrategy(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    This is a minimal example demonstrating:
    - Clean strategy structure
    - VectorBTPro's native indicator usage
    - Automatic multi-symbol support
    - Proper signal generation
    
    The strategy generates long signals when the fast MA crosses above the slow MA,
    and exits when the fast MA crosses below the slow MA. No short signals are
    generated in this simple example.
    """
    
    def __init__(self, data: vbt.Data, params: Dict[str, Any]):
        """
        Initialize the strategy.
        
        Args:
            data: VectorBTPro Data object with OHLCV data
            params: Strategy parameters
                - fast_window: Period for fast moving average (default: 10)
                - slow_window: Period for slow moving average (default: 30)
                - ma_type: Type of MA - 'SMA', 'EMA', 'WMA' (default: 'SMA')
        """
        # Set default parameters
        default_params = {
            'fast_window': 10,
            'slow_window': 30,
            'ma_type': 'SMA'
        }
        
        # Merge with provided parameters
        params = {**default_params, **params}
        
        # Validate parameters
        if params['fast_window'] >= params['slow_window']:
            raise ValueError("Fast window must be less than slow window")
        
        # Initialize base class
        super().__init__(data, params)
    
    def init_indicators(self) -> Dict[str, Any]:
        """
        Calculate moving averages using VectorBTPro's native functions.
        
        Returns:
            Dictionary of calculated indicators
        """
        # Get close prices - works with both single and multi-symbol data
        close = self.data.close
        
        # Select MA function based on type
        ma_type = self.params['ma_type'].upper()
        
        # Calculate moving averages using VBT's TA-Lib integration
        # VBT automatically handles multi-symbol broadcasting
        if ma_type == 'SMA':
            self.indicators['fast_ma'] = vbt.talib("SMA").run(
                close, timeperiod=self.params['fast_window']
            ).real
            
            self.indicators['slow_ma'] = vbt.talib("SMA").run(
                close, timeperiod=self.params['slow_window']
            ).real
            
        elif ma_type == 'EMA':
            self.indicators['fast_ma'] = vbt.talib("EMA").run(
                close, timeperiod=self.params['fast_window']
            ).real
            
            self.indicators['slow_ma'] = vbt.talib("EMA").run(
                close, timeperiod=self.params['slow_window']
            ).real
            
        elif ma_type == 'WMA':
            self.indicators['fast_ma'] = vbt.talib("WMA").run(
                close, timeperiod=self.params['fast_window']
            ).real
            
            self.indicators['slow_ma'] = vbt.talib("WMA").run(
                close, timeperiod=self.params['slow_window']
            ).real
            
        else:
            raise ValueError(f"Unsupported MA type: {ma_type}")
        
        # Store close price for reference
        self.indicators['close'] = close
        
        return self.indicators
    
    def generate_signals(self) -> Dict[str, Any]:
        """
        Generate trading signals from moving average crossovers.
        
        Returns:
            Dictionary with signal arrays
        """
        # Get indicators
        fast_ma = self.indicators['fast_ma']
        slow_ma = self.indicators['slow_ma']
        
        # For multi-symbol data with different date ranges, ensure alignment
        if isinstance(fast_ma, pd.DataFrame) and isinstance(slow_ma, pd.DataFrame):
            # Align the dataframes to ensure they have the same index and columns
            fast_ma, slow_ma = fast_ma.align(slow_ma, join='inner')
        
        # Generate crossover signals
        # VBT handles multi-symbol data automatically through broadcasting
        long_entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        long_exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
        
        # Clean signals using VectorBTPro's signal utilities
        # This ensures no conflicting signals on the same bar
        if hasattr(long_entries, 'vbt'):
            # Use VBT's signal cleaning for proper entry/exit sequences
            long_entries, long_exits = long_entries.vbt.signals.clean(long_exits)
        
        # For this simple strategy, we don't generate short signals
        # Create empty signals with proper shape
        if isinstance(long_entries, pd.DataFrame):
            # Multi-symbol case
            short_entries = pd.DataFrame(
                False, 
                index=long_entries.index, 
                columns=long_entries.columns
            )
            short_exits = pd.DataFrame(
                False, 
                index=long_entries.index, 
                columns=long_entries.columns
            )
        else:
            # Single symbol case
            short_entries = pd.Series(False, index=long_entries.index)
            short_exits = pd.Series(False, index=long_entries.index)
        
        # Store signals
        self.signals = {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'sl_levels': None,  # No stop-loss in this simple example
            'tp_levels': None   # No take-profit in this simple example
        }
        
        return self.signals
    
    def get_strategy_description(self) -> str:
        """
        Get a human-readable description of the strategy configuration.
        
        Returns:
            String description of the strategy
        """
        return f"""
Simple Moving Average Crossover Strategy

Configuration:
- Fast {self.params['ma_type']}: {self.params['fast_window']} periods
- Slow {self.params['ma_type']}: {self.params['slow_window']} periods
- Signal Type: Long only (no shorting)

Rules:
- Enter Long: Fast MA crosses above Slow MA
- Exit Long: Fast MA crosses below Slow MA

This strategy works automatically with:
- Single symbol data
- Multi-symbol data (via VBT broadcasting)
- Parameter optimization (using vbt.Param)
"""
    
    def __repr__(self) -> str:
        """String representation of the strategy."""
        return (
            f"SimpleMAStrategy("
            f"fast={self.params['fast_window']}, "
            f"slow={self.params['slow_window']}, "
            f"type={self.params['ma_type']})"
        )


# Example usage functions
def example_single_symbol():
    """Example: Running strategy on single symbol."""
    # Fetch data
    data = vbt.YFData.fetch("AAPL", start="2023-01-01", end="2023-12-31")
    
    # Create strategy
    strategy = SimpleMAStrategy(data, {
        'fast_window': 10,
        'slow_window': 30,
        'ma_type': 'EMA'
    })
    
    # Generate signals
    result = strategy.run_signal_generation()
    
    # Signals are ready for portfolio simulation
    return result


def example_multi_symbol():
    """Example: Running strategy on multiple symbols."""
    # Fetch multi-symbol data
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    data = vbt.YFData.fetch(symbols, start="2023-01-01", end="2023-12-31")
    
    # Same strategy code works for multiple symbols!
    strategy = SimpleMAStrategy(data, {
        'fast_window': 10,
        'slow_window': 30
    })
    
    # Generate signals for all symbols
    result = strategy.run_signal_generation()
    
    # result['signals'] contains DataFrames with columns for each symbol
    return result


def example_parameter_optimization():
    """Example: Parameter optimization using vbt.Param."""
    # Fetch data
    data = vbt.YFData.fetch("SPY", start="2022-01-01", end="2023-12-31")
    
    # Define parameter ranges
    params = {
        'fast_window': vbt.Param([5, 10, 15, 20]),
        'slow_window': vbt.Param([20, 30, 40, 50, 60]),
        'ma_type': 'SMA'  # Keep this fixed
    }
    
    # Create strategy with parameter ranges
    strategy = SimpleMAStrategy(data, params)
    
    # Generate signals for all parameter combinations
    result = strategy.run_signal_generation()
    
    # result contains signals for all parameter combinations
    # Ready for portfolio simulation and optimization
    return result


if __name__ == "__main__":
    # Run examples
    print("Running single symbol example...")
    single_result = example_single_symbol()
    print(f"Generated signals: {list(single_result['signals'].keys())}")
    
    print("\nRunning multi-symbol example...")
    multi_result = example_multi_symbol()
    print(f"Symbols processed: {multi_result['metadata']['symbols']}")
    
    print("\nRunning parameter optimization example...")
    opt_result = example_parameter_optimization()
    print("Parameter optimization ready for portfolio simulation") 