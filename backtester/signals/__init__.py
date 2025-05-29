"""
Signals Module

This module provides trading signal generation functionality for the backtesting system.
It processes technical indicator outputs and applies strategy logic to generate
entry and exit signals for trading strategies.

Key Features:
- Boolean signal generation from indicator combinations
- Signal cleaning and validation utilities
- Support for long and short signals
- Parameterized signal generation for optimization

Usage:
    from backtester.signals.signal_engine import generate_dma_atr_trend_signals
    long_entries, long_exits, short_entries, short_exits = generate_dma_atr_trend_signals(...)
"""

from . import signal_engine

# CLI tools (available as submodules)
try:
    from backtester.signals import cli
except ImportError:
    # CLI tools may not be available in all environments
    cli = None

__all__ = ["signal_engine", "cli"]
