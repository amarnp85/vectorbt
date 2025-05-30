"""
Simplified Trading Signals Processing Orchestrator

This module provides a simplified, modular interface for trading signal processing
that replaces the original monolithic trading_signals.py file. It orchestrates
the signal components to provide clean signal extraction, validation, and rendering.

Architecture:
============
The simplified orchestrator follows the same pattern as trading_charts.py:

1. SignalProcessor: Main orchestrator class
2. Component Integration: Uses signal_components for all processing
3. Clean Interfaces: Simple methods for common operations
4. Backward Compatibility: Maintains the same external API

Key Features:
============
- Unified signal extraction from portfolio and strategy sources
- Comprehensive signal validation and quality assessment  
- Configurable timing modes (signal vs execution timing)
- Stop loss and take profit level processing
- Detailed logging and debugging support
- Component-based architecture for easy testing and maintenance

Usage:
======
    from backtester.analysis.trading_signals_simplified import SignalProcessor
    
    # Create processor with configuration
    config = SignalConfig(signal_timing_mode="execution")
    processor = SignalProcessor(portfolio, data_processor, strategy_signals, config)
    
    # Extract and validate signals
    signals = processor.extract_signals()
    report = processor.validate_signals()
    
    # Get signals in different formats
    signals_dict = processor.get_signals_dict()
    signals_format = processor.get_signals_format()

Integration:
===========
This module works seamlessly with:
- chart_components: For signal rendering on charts
- signal_components: For all signal processing logic
- portfolio module: For portfolio simulation
- strategies module: For strategy signal generation

The module is designed to eventually replace the original trading_signals.py
while maintaining full backward compatibility.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

from backtester.analysis.signal_components import (
    UnifiedSignalExtractor,
    ComprehensiveSignalValidator,
    TimingConfig,
    TimingMode,
    ValidationReport
)
from backtester.signals.signal_interface import SignalFormat
from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class SignalConfig:
    """Configuration for signal processing and rendering."""
    # Timing configuration - CRITICAL for preventing lookahead bias
    signal_timing_mode: str = "execution"  # "signal", "execution", or "both"
    execution_delay: int = 1  # Bars between signal and execution
    show_timing_indicator: bool = True  # Show timing mode in chart
    
    # Signal processing options
    validate_signals: bool = True
    clean_signals: bool = True
    portfolio_signals_priority: bool = True  # Portfolio signals override strategy
    use_unified_signal_interface: bool = True  # Use new unified interface
    
    # Signal extraction options - CRITICAL FIX
    extract_from_trades_only: bool = True  # Only extract from trades, not orders
    separate_exit_types: bool = True  # Separate long_exits and short_exits
    
    # Rendering options
    show_signals: bool = True
    show_stop_levels: bool = True  # Show SL/TP symbols
    
    # Colors for different signal types
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'long_entry': 'lime',
                'short_entry': 'orangered', 
                'long_exit': 'purple',
                'short_exit': 'purple',
                'exit': 'purple',  # Unified exit color
                'stop_level': 'red',      # Red for SL
                'profit_level': 'green'   # Green for TP
            }


class SignalProcessor:
    """
    Simplified signal processor that orchestrates signal components.
    
    This class provides a clean, simplified interface for signal processing
    while delegating all the complex logic to specialized components.
    
    Features:
    ========
    - Unified signal extraction from multiple sources
    - Comprehensive validation and quality assessment
    - Configurable timing modes for different use cases
    - Clean interfaces for common operations
    - Detailed logging for debugging
    
    Design Philosophy:
    =================
    - Composition over inheritance
    - Single responsibility principle
    - Clear separation of concerns
    - Easy to test and maintain
    - Backward compatible API
    
    Usage:
    ======
        processor = SignalProcessor(portfolio, data_processor, strategy_signals)
        signals = processor.extract_signals()
        report = processor.validate_signals()
    """
    
    def __init__(
        self,
        portfolio: vbt.Portfolio,
        data_processor: Any,
        strategy_signals: Optional[Dict] = None,
        config: Optional[SignalConfig] = None
    ):
        """
        Initialize the simplified signal processor.
        
        Args:
            portfolio: VectorBT Portfolio object
            data_processor: Data processor with get_ohlcv_data() method
            strategy_signals: Optional strategy signals dictionary
            config: Optional signal configuration
        """
        self.portfolio = portfolio
        self.data_processor = data_processor
        self.strategy_signals = strategy_signals or {}
        self.config = config or SignalConfig()
        
        # Convert timing configuration
        timing_mode = TimingMode.EXECUTION if self.config.signal_timing_mode == "execution" else TimingMode.SIGNAL
        self.timing_config = TimingConfig(
            mode=timing_mode,
            execution_delay=self.config.execution_delay
        )
        
        # Initialize components
        self.extractor = UnifiedSignalExtractor(
            portfolio=self.portfolio,
            data_processor=self.data_processor,
            strategy_signals=self.strategy_signals,
            timing_config=self.timing_config
        )
        
        self.validator = ComprehensiveSignalValidator(
            strict_mode=False,
            auto_clean=self.config.clean_signals
        )
        
        # State
        self._extracted_signals = None
        self._validation_report = None
        
        logger.info(f"Initialized SignalProcessor with timing mode: {self.config.signal_timing_mode}")
    
    def extract_signals(self) -> SignalFormat:
        """
        Extract signals from portfolio and strategy sources.
        
        This is the main entry point for signal extraction. It orchestrates
        the extraction process and returns a unified signal format.
        
        Returns:
            SignalFormat object with all extracted signals
        """
        logger.info("=== STARTING SIGNAL EXTRACTION ===")
        
        try:
            # Extract signals using the unified extractor
            self._extracted_signals = self.extractor.extract_all_signals()
            
            # Log extraction summary
            summary = self._extracted_signals.get_summary()
            logger.info(f"Signal extraction complete: {summary}")
            
            return self._extracted_signals
            
        except Exception as e:
            logger.error(f"Signal extraction failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def validate_signals(self, signals: Optional[SignalFormat] = None) -> ValidationReport:
        """
        Validate extracted signals for quality and consistency.
        
        Args:
            signals: Optional signals to validate (uses extracted signals if None)
            
        Returns:
            ValidationReport with detailed validation results
        """
        if signals is None:
            if self._extracted_signals is None:
                self.extract_signals()
            signals = self._extracted_signals
        
        logger.info("=== STARTING SIGNAL VALIDATION ===")
        
        try:
            # Validate signals using the comprehensive validator
            self._validation_report = self.validator.validate_signals(signals)
            
            # Log validation summary
            summary = self._validation_report.get_summary()
            logger.info(f"Signal validation complete: {summary}")
            
            # Log any recommendations
            if self._validation_report.recommendations:
                logger.info("Validation recommendations:")
                for rec in self._validation_report.recommendations:
                    logger.info(f"  - {rec}")
            
            return self._validation_report
            
        except Exception as e:
            logger.error(f"Signal validation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def get_signals_dict(self) -> Dict[str, pd.Series]:
        """
        Get signals in dictionary format for backward compatibility.
        
        Returns:
            Dictionary of signal name -> pandas Series
        """
        if self._extracted_signals is None:
            self.extract_signals()
        
        return self._extracted_signals.to_dict()
    
    def get_signals_format(self) -> SignalFormat:
        """
        Get signals in the unified SignalFormat.
        
        Returns:
            SignalFormat object with all signals
        """
        if self._extracted_signals is None:
            self.extract_signals()
        
        return self._extracted_signals
    
    def get_validation_report(self) -> Optional[ValidationReport]:
        """
        Get the latest validation report.
        
        Returns:
            ValidationReport if validation has been run, None otherwise
        """
        return self._validation_report
    
    def process_signals(self) -> Tuple[Dict[str, pd.Series], ValidationReport]:
        """
        Complete signal processing: extract and validate.
        
        This is a convenience method that performs both extraction and validation
        in a single call.
        
        Returns:
            Tuple of (signals_dict, validation_report)
        """
        logger.info("=== STARTING COMPLETE SIGNAL PROCESSING ===")
        
        # Extract signals
        signals = self.extract_signals()
        
        # Validate signals
        report = self.validate_signals(signals)
        
        # Convert to dictionary format
        signals_dict = signals.to_dict()
        
        logger.info("=== SIGNAL PROCESSING COMPLETE ===")
        logger.info(f"Final signal counts: {signals.get_summary()}")
        
        return signals_dict, report
    
    def get_timing_info(self) -> Dict[str, Any]:
        """
        Get information about the current timing configuration.
        
        Returns:
            Dictionary with timing configuration details
        """
        return {
            "timing_mode": self.config.signal_timing_mode,
            "execution_delay": self.config.execution_delay,
            "timing_description": self._get_timing_description(),
            "timing_config": self.timing_config
        }
    
    def _get_timing_description(self) -> str:
        """Get a human-readable description of the timing configuration."""
        if self.config.signal_timing_mode == "execution":
            return (
                f"Signals shown at execution time (signal time + {self.config.execution_delay} bars). "
                "This prevents lookahead bias and shows realistic trading results."
            )
        else:
            return (
                "Signals shown at decision time. Useful for strategy analysis but may "
                "appear to have lookahead bias in charts."
            )
    
    def get_component_info(self) -> Dict[str, Any]:
        """
        Get information about the internal components.
        
        Returns:
            Dictionary with component status and configuration
        """
        return {
            "extractor": {
                "class": self.extractor.__class__.__name__,
                "timing_config": self.timing_config,
                "has_strategy_signals": bool(self.strategy_signals)
            },
            "validator": {
                "class": self.validator.__class__.__name__,
                "strict_mode": self.validator.strict_mode,
                "auto_clean": self.validator.auto_clean
            },
            "config": {
                "timing_mode": self.config.signal_timing_mode,
                "validate_signals": self.config.validate_signals,
                "clean_signals": self.config.clean_signals,
                "show_signals": self.config.show_signals,
                "show_stop_levels": self.config.show_stop_levels
            }
        }


# Convenience functions for common operations

def process_signals_simple(
    portfolio: vbt.Portfolio,
    data_processor: Any,
    strategy_signals: Optional[Dict] = None,
    timing_mode: str = "execution"
) -> Tuple[Dict[str, pd.Series], ValidationReport]:
    """
    Simple signal processing function for quick usage.
    
    Args:
        portfolio: VectorBT Portfolio object
        data_processor: Data processor with get_ohlcv_data() method
        strategy_signals: Optional strategy signals dictionary
        timing_mode: Timing mode ("signal" or "execution")
        
    Returns:
        Tuple of (signals_dict, validation_report)
    """
    config = SignalConfig(signal_timing_mode=timing_mode)
    processor = SignalProcessor(portfolio, data_processor, strategy_signals, config)
    return processor.process_signals()


def extract_signals_only(
    portfolio: vbt.Portfolio,
    data_processor: Any,
    strategy_signals: Optional[Dict] = None,
    timing_mode: str = "execution"
) -> Dict[str, pd.Series]:
    """
    Extract signals without validation for performance-critical scenarios.
    
    Args:
        portfolio: VectorBT Portfolio object
        data_processor: Data processor with get_ohlcv_data() method
        strategy_signals: Optional strategy signals dictionary
        timing_mode: Timing mode ("signal" or "execution")
        
    Returns:
        Dictionary of signal name -> pandas Series
    """
    config = SignalConfig(signal_timing_mode=timing_mode, validate_signals=False)
    processor = SignalProcessor(portfolio, data_processor, strategy_signals, config)
    signals = processor.extract_signals()
    return signals.to_dict()


def validate_signals_only(signals: SignalFormat) -> ValidationReport:
    """
    Validate signals without extraction for testing scenarios.
    
    Args:
        signals: SignalFormat object to validate
        
    Returns:
        ValidationReport with validation results
    """
    validator = ComprehensiveSignalValidator()
    return validator.validate_signals(signals)


# Module exports
__all__ = [
    'SignalConfig',
    'SignalProcessor',
    'process_signals_simple',
    'extract_signals_only',
    'validate_signals_only'
]