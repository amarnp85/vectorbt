"""
Strategies Module

This module provides trading strategy implementations for the backtesting system.
All strategies follow a consistent interface defined by BaseStrategy and focus
exclusively on indicator calculation and signal generation.

Key Features:
- Abstract base strategy class for consistent interface
- Multiple strategy implementations (trend following, mean reversion, etc.)
- Multi-timeframe and multi-symbol support
- Integration with VectorBTPro for efficient computation
- Clear separation of concerns (no portfolio simulation in strategies)

Strategy Categories:
1. Single Symbol Strategies:
   - DMAATRTrendStrategy: Dual moving average with ATR-based stops
   - MeanReversionStrategy: Bollinger Bands and RSI mean reversion
   - MomentumStrategy: Trend following with momentum indicators
   - PairsTradingStrategy: Statistical arbitrage between correlated pairs

2. Multi-Timeframe Strategies:
   - MTFStrategy: Base class for multi-timeframe analysis
   - MTF_DMA_ATR_Strategy: Multi-timeframe version of DMA ATR

3. Multi-Symbol Strategies:
   - MultiSymbolStrategy: Base class for portfolio-wide strategies

4. Examples:
   - SimpleMAStrategy: Clean example of VectorBTPro best practices

Note: Position sizing, risk management, and portfolio simulation are handled
by the portfolio module, not within strategies.
"""

# Base classes
from .base_strategy import BaseStrategy
from .mtf_strategy_base import MTFStrategy
from .multi_symbol_strategy_base import MultiSymbolStrategy

# Single symbol strategies
from .dma_atr_trend_strategy import DMAATRTrendStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .momentum_strategy import MomentumStrategy
from .pairs_trading_strategy import PairsTradingStrategy

# Multi-timeframe strategies
from .mtf_dma_atr_strategy import MTF_DMA_ATR_Strategy

# Example strategies
from .simple_ma_crossover_example import SimpleMAStrategy

__all__ = [
    # Base classes
    'BaseStrategy',
    'MTFStrategy',
    'MultiSymbolStrategy',
    
    # Single symbol strategies
    'DMAATRTrendStrategy',
    'MeanReversionStrategy',
    'MomentumStrategy',
    'PairsTradingStrategy',
    
    # Multi-timeframe strategies
    'MTF_DMA_ATR_Strategy',
    
    # Examples
    'SimpleMAStrategy'
]
