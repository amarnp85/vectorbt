"""
Signal Components Module

This module provides modular components for processing, validating, and extracting
trading signals from various sources. The components work together to ensure
signal quality and proper timing in backtesting visualization.

Module Structure:
================
- timing.py: Timing calculations and corrections
- extractors.py: Signal extraction from portfolio and strategy sources
- validators.py: Signal validation and quality assessment

Key Features:
============
- Proper timing handling to prevent lookahead bias
- Multiple signal source integration (portfolio + strategy)
- Comprehensive signal validation and quality metrics
- Unified signal format for consistency
- Extensive documentation and debugging support

Quick Start:
===========
    from backtester.analysis.signal_components import (
        UnifiedSignalExtractor,
        ComprehensiveSignalValidator,
        TimingConfig
    )
    
    # Configure timing
    timing_config = TimingConfig(mode=TimingMode.EXECUTION)
    
    # Extract signals
    extractor = UnifiedSignalExtractor(portfolio, data_processor, strategy_signals, timing_config)
    signals = extractor.extract_all_signals()
    
    # Validate signals
    validator = ComprehensiveSignalValidator()
    report = validator.validate_signals(signals)

Integration Points:
==================
- Used by: ../trading_signals.py (main orchestrator)
- Uses: ../signals/signal_interface.py for unified format
- Related to: chart_components for signal rendering

For detailed usage, see the documentation in each module.
"""

# Timing components
from .timing import (
    TimingMode,
    TimingConfig,
    TimingCalculator,
    TimestampNormalizer,
    TimingValidator,
    get_timing_recommendations,
    explain_signal_timing
)

# Extraction components
from .extractors import (
    ISignalExtractor,
    PortfolioSignalExtractor,
    StrategySignalExtractor,
    UnifiedSignalExtractor
)

# Validation components
from .validators import (
    ValidationSeverity,
    ValidationIssue,
    ValidationReport,
    SignalConsistencyValidator,
    SignalQualityAnalyzer,
    SignalCleaner,
    ComprehensiveSignalValidator
)

__all__ = [
    # Timing
    'TimingMode',
    'TimingConfig',
    'TimingCalculator',
    'TimestampNormalizer',
    'TimingValidator',
    'get_timing_recommendations',
    'explain_signal_timing',
    # Extraction
    'ISignalExtractor',
    'PortfolioSignalExtractor',
    'StrategySignalExtractor',
    'UnifiedSignalExtractor',
    # Validation
    'ValidationSeverity',
    'ValidationIssue',
    'ValidationReport',
    'SignalConsistencyValidator',
    'SignalQualityAnalyzer',
    'SignalCleaner',
    'ComprehensiveSignalValidator'
]