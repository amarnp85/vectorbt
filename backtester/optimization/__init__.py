"""
Optimization Module

This module provides parameter optimization functionality for trading strategies.
It leverages vectorbtpro's parameter broadcasting capabilities for efficient optimization.

Key Features:
- Parameter space exploration
- Grid search optimization
- Walk-forward analysis
- Multi-objective optimization
- Performance-based parameter selection

Usage:
    from backtester.optimization.optimizer_engine import optimize_strategy_parameters
    results = optimize_strategy_parameters(strategy, param_ranges)
"""

from . import optimizer_engine

__all__ = ["optimizer_engine"]
