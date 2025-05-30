"""
Analysis Module

This module provides performance analysis and visualization capabilities for backtesting results.
It includes comprehensive performance metrics calculation and modern interactive plotting functionality.

Key Features:
- Portfolio performance analysis with comprehensive metrics
- Trade-level analysis and risk metrics calculation
- Modern interactive plotting with enhanced TradingChartsEngine
- Multi-timeframe analysis and plotting capabilities
- Benchmark comparison and analysis tools
- VectorBTPro compatibility layer for version differences

Usage:
    from backtester.analysis import PerformanceAnalyzer
    from backtester.analysis import TradingChartsEngine
    from backtester.analysis import BenchmarkAnalyzer
    from backtester.analysis import MTFPlottingEngine
    from backtester.analysis import VBTCompatibilityLayer
"""

from .performance_analyzer import PerformanceAnalyzer
from .trading_charts import TradingChartsEngine
from .vbt_compatibility import VBTCompatibilityLayer
from .benchmark_analyzer import BenchmarkAnalyzer
from .mtf_plotting import MTFPlottingEngine

__all__ = [
    "PerformanceAnalyzer",
    "TradingChartsEngine",
    "VBTCompatibilityLayer",
    "BenchmarkAnalyzer",
    "MTFPlottingEngine"
]
