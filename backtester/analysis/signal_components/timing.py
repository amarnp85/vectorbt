"""
Signal Timing Logic and Corrections

This module handles all timing-related functionality for trading signals,
including the critical logic that prevents lookahead bias in backtesting.

Module Purpose:
==============
VectorBT's execution model separates signal generation from order execution:
- Signal Generation: Happens at bar close (time T) based on available data
- Order Execution: Happens at next bar open (time T+1) to prevent lookahead bias
- Chart Display: Configurable to show signals at generation time or execution time

This module provides the timing calculations and corrections needed to
properly display signals according to realistic trading constraints.

Key Concepts:
============
1. Signal Timing: When decisions are made (bar close T)
2. Execution Timing: When orders are filled (next bar open T+1)
3. Display Timing: Where signals appear on charts (configurable)

Integration Points:
==================
- Used by: SignalProcessor for timing corrections
- Related to: Portfolio execution model in VectorBT
- Chart Display: Affects how signals appear on trading charts

Related Modules:
===============
- extractors.py: Uses timing calculations during signal extraction
- ../trading_signals.py: Main orchestrator imports timing utilities
"""

import pandas as pd
import vectorbtpro as vbt
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


class TimingMode(Enum):
    """
    Signal timing modes for chart display.
    
    These modes control where signals appear on charts relative to
    when trading decisions were made vs when orders were executed.
    """
    SIGNAL = "signal"          # Show at decision time (T)
    EXECUTION = "execution"    # Show at execution time (T+1) - RECOMMENDED
    BOTH = "both"             # Show both (for analysis)


@dataclass
class TimingConfig:
    """
    Configuration for signal timing behavior.
    
    This configuration controls how timing corrections are applied
    and how signals are displayed on charts.
    """
    mode: TimingMode = TimingMode.EXECUTION
    execution_delay: int = 1           # Bars between signal and execution
    show_timing_indicator: bool = True  # Show timing mode on charts
    validate_timing: bool = True       # Validate timing consistency


class TimingCalculator:
    """
    Calculates timing corrections for signal display.
    
    This class handles the core timing calculations that ensure signals
    are displayed at the correct time on charts based on the configured
    timing mode.
    
    Usage:
        calculator = TimingCalculator(config)
        display_time = calculator.calculate_display_timestamp(
            signal_time, data_index
        )
    
    Integration:
    - Used by: SignalExtractor for timing corrections
    - Purpose: Prevent lookahead bias in chart visualization
    """
    
    def __init__(self, config: TimingConfig):
        """
        Initialize timing calculator.
        
        Args:
            config: Timing configuration
        """
        self.config = config
    
    def calculate_display_timestamp(
        self, 
        signal_timestamp: pd.Timestamp, 
        data_index: pd.Index
    ) -> pd.Timestamp:
        """
        Calculate where a signal should be displayed on the chart.
        
        This is the core timing correction logic that ensures signals
        are shown at the appropriate time based on the timing mode.
        
        Args:
            signal_timestamp: When the signal was generated
            data_index: Full time index from the data
            
        Returns:
            Timestamp where signal should be displayed
        """
        try:
            signal_pos = data_index.get_loc(signal_timestamp)
            
            if self.config.mode == TimingMode.EXECUTION:
                # Show at execution time (signal + delay)
                execution_pos = signal_pos + self.config.execution_delay
                
                if execution_pos < len(data_index):
                    return data_index[execution_pos]
                else:
                    # Signal at end of data - show at signal time
                    return signal_timestamp
            
            elif self.config.mode == TimingMode.SIGNAL:
                # Show at signal generation time
                return signal_timestamp
            
            else:  # BOTH mode would need special handling
                return signal_timestamp
                
        except (KeyError, IndexError):
            # Fallback to signal timestamp if calculation fails
            logger.debug(f"Could not calculate display timestamp for {signal_timestamp}")
            return signal_timestamp
    
    def get_execution_price_timestamp(
        self,
        signal_timestamp: pd.Timestamp,
        data_index: pd.Index
    ) -> Optional[pd.Timestamp]:
        """
        Get the timestamp for price data at execution time.
        
        This helps extract the correct price for executed orders.
        
        Args:
            signal_timestamp: When signal was generated
            data_index: Full time index
            
        Returns:
            Timestamp for execution price, or None if not available
        """
        try:
            signal_pos = data_index.get_loc(signal_timestamp)
            execution_pos = signal_pos + self.config.execution_delay
            
            if execution_pos < len(data_index):
                return data_index[execution_pos]
            
        except (KeyError, IndexError):
            pass
        
        return None


class TimestampNormalizer:
    """
    Normalizes timestamps for consistency across data sources.
    
    Different data sources (portfolio, strategy signals, market data)
    may have timestamps in different formats or timezones. This class
    ensures all timestamps are comparable.
    
    Usage:
        normalizer = TimestampNormalizer(reference_index)
        normalized_ts = normalizer.normalize(raw_timestamp)
    """
    
    def __init__(self, reference_index: pd.Index):
        """
        Initialize with reference index for timezone alignment.
        
        Args:
            reference_index: Index to use as timezone reference
        """
        self.reference_index = reference_index
        self.reference_tz = getattr(reference_index, 'tz', None)
    
    def normalize(self, timestamp: Any) -> Optional[pd.Timestamp]:
        """
        Normalize a timestamp for comparison with reference index.
        
        Args:
            timestamp: Raw timestamp in any format
            
        Returns:
            Normalized timestamp or None if invalid
        """
        if timestamp is None:
            return None
        
        # Convert to Timestamp if needed
        if not isinstance(timestamp, pd.Timestamp):
            try:
                timestamp = pd.Timestamp(timestamp)
            except (ValueError, TypeError):
                return None
        
        # Handle timezone consistency
        if self.reference_tz is None and timestamp.tz is not None:
            # Reference has no timezone, remove timezone from timestamp
            timestamp = timestamp.tz_localize(None)
        elif self.reference_tz is not None and timestamp.tz is None:
            # Reference has timezone, add it to timestamp
            timestamp = timestamp.tz_localize(self.reference_tz)
        elif self.reference_tz is not None and timestamp.tz is not None:
            # Both have timezones, convert to reference timezone
            timestamp = timestamp.tz_convert(self.reference_tz)
        
        return timestamp


class TimingValidator:
    """
    Validates timing consistency and provides recommendations.
    
    This class helps detect potential timing issues and provides
    guidance on optimal timing configurations for different scenarios.
    
    Usage:
        validator = TimingValidator()
        result = validator.validate_portfolio_timing(portfolio, data, config)
    """
    
    def validate_portfolio_timing(
        self,
        portfolio: vbt.Portfolio,
        data: vbt.Data,
        config: TimingConfig
    ) -> Dict[str, Any]:
        """
        Validate timing configuration against portfolio data.
        
        Args:
            portfolio: VectorBT Portfolio object
            data: VectorBT Data object
            config: Timing configuration to validate
            
        Returns:
            Validation results with recommendations
        """
        result = {
            "valid": True,
            "warnings": [],
            "recommendations": [],
            "timing_analysis": {}
        }
        
        try:
            trades = portfolio.trades.records_readable
            if len(trades) == 0:
                result["warnings"].append("No trades found in portfolio")
                return result
            
            # Check execution delay appropriateness
            if config.execution_delay > 5:
                result["warnings"].append(
                    f"Execution delay of {config.execution_delay} bars seems excessive"
                )
                result["recommendations"].append(
                    "Consider reducing execution_delay to 1-2 bars for more realistic timing"
                )
            
            # Analyze sample trades for timing patterns
            sample_size = min(10, len(trades))
            sample_trades = trades.head(sample_size)
            
            timing_issues = 0
            for idx, trade in sample_trades.iterrows():
                signal_ts = pd.Timestamp(trade.get('Entry Index'))
                # In a realistic scenario, execution should be delayed
                # This is a simplified check
                timing_issues += 0  # Placeholder for actual timing analysis
            
            result["timing_analysis"] = {
                "sample_trades_analyzed": sample_size,
                "timing_issues_found": timing_issues,
                "recommended_mode": config.mode.value
            }
            
            # Generate mode-specific recommendations
            if config.mode == TimingMode.SIGNAL:
                result["recommendations"].append(
                    "Signal timing mode shows decision points but not realistic execution. "
                    "Consider using execution mode for realistic backtesting visualization."
                )
            
        except Exception as e:
            result["valid"] = False
            result["warnings"].append(f"Validation failed: {str(e)}")
        
        return result


def get_timing_recommendations(
    timeframe: str,
    trading_style: str = "swing"
) -> Dict[str, Any]:
    """
    Get timing recommendations based on timeframe and trading style.
    
    This function provides recommended timing settings for different
    trading scenarios to help users configure appropriate timing.
    
    Args:
        timeframe: Trading timeframe (e.g., '1h', '4h', '1d')
        trading_style: Trading style ('scalping', 'day', 'swing', 'position')
        
    Returns:
        Dictionary with recommended timing settings
    """
    recommendations = {
        "execution_delay": 1,
        "timing_mode": TimingMode.EXECUTION,
        "explanation": ""
    }
    
    # Adjust based on timeframe
    timeframe_lower = timeframe.lower()
    if timeframe_lower in ['1m', '5m', '15m']:
        recommendations["execution_delay"] = 1
        recommendations["explanation"] = "Short timeframes: 1 bar delay simulates near-instant execution"
        
        if trading_style == "scalping":
            recommendations["timing_mode"] = TimingMode.SIGNAL
            recommendations["explanation"] += " (scalping may use signal timing for analysis)"
            
    elif timeframe_lower in ['30m', '1h', '2h']:
        recommendations["execution_delay"] = 1
        recommendations["explanation"] = "Medium timeframes: 1 bar delay represents realistic order processing"
        
    elif timeframe_lower in ['4h', '6h', '8h']:
        recommendations["execution_delay"] = 1
        recommendations["explanation"] = "Longer timeframes: 1 bar delay allows for analysis and order placement"
        
    else:  # Daily and above
        recommendations["execution_delay"] = 1
        recommendations["explanation"] = "Daily+ timeframes: 1 bar delay represents end-of-day processing"
    
    # Adjust based on trading style
    style_adjustments = {
        "scalping": {
            "timing_mode": TimingMode.SIGNAL,
            "note": "Scalping often requires signal timing analysis"
        },
        "day": {
            "timing_mode": TimingMode.EXECUTION,
            "note": "Day trading benefits from realistic execution timing"
        },
        "swing": {
            "timing_mode": TimingMode.EXECUTION,
            "note": "Swing trading should use execution timing for realism"
        },
        "position": {
            "timing_mode": TimingMode.EXECUTION,
            "note": "Position trading benefits from realistic execution delays"
        }
    }
    
    if trading_style in style_adjustments:
        recommendations.update(style_adjustments[trading_style])
        recommendations["explanation"] += f" | {style_adjustments[trading_style]['note']}"
    
    return recommendations


def explain_signal_timing() -> str:
    """
    Provide a comprehensive explanation of signal timing concepts.
    
    This function returns educational content about timing modes
    and their implications for backtesting and analysis.
    
    Returns:
        Detailed explanation of signal timing
    """
    return """
Signal Timing Modes in Trading Charts
====================================

To prevent lookahead bias, trading charts support different timing modes:

1. EXECUTION TIMING (DEFAULT - RECOMMENDED):
   - Shows signals at actual execution time (T+1)
   - Uses actual execution prices
   - Reflects realistic trading constraints
   - Prevents misleading visualization

2. SIGNAL TIMING (ANALYSIS MODE):
   - Shows signals at decision time (T)
   - Uses signal-generation prices
   - Useful for strategy analysis
   - May appear to have lookahead bias

3. VectorBT EXECUTION MODEL:
   - Signal Generation: Bar close (time T) based on available data
   - Order Execution: Next bar open (time T+1) 
   - Price Used: Next bar's open price
   - Chart Display: Configurable timing mode

EXAMPLE:
========
Decision made: 2023-01-15 close, based on data up to 2023-01-15
Order executed: 2023-01-16 open, at 2023-01-16 open price
Chart shows (execution mode): Entry at 2023-01-16 with 2023-01-16 price
Chart shows (signal mode): Entry at 2023-01-15 with 2023-01-15 price

RECOMMENDATION:
==============
Use execution timing mode for realistic backtesting visualization.
Use signal timing mode only for strategy development and analysis.
"""


# Module exports
__all__ = [
    'TimingMode',
    'TimingConfig',
    'TimingCalculator',
    'TimestampNormalizer',
    'TimingValidator',
    'get_timing_recommendations',
    'explain_signal_timing'
]