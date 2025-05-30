"""
Unified Signal Interface for Trading Systems

This module provides a standardized interface for trading signals that ensures
consistent communication between the trading_signals.py module and simulation_engine.py.
It defines canonical signal formats, validation rules, and cross-reference capabilities
to prevent data inconsistencies.

Key Features:
- Standardized signal format and naming conventions
- Signal validation and consistency checks
- Cross-reference validation between portfolio trades and chart signals
- Timezone and timestamp normalization
- Price integrity validation
- Trade count reconciliation

Classes:
- SignalFormat: Defines the canonical signal structure
- SignalValidator: Validates signal integrity and consistency
- TradeSignalCrossReference: Cross-references portfolio trades with chart signals
"""

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import warnings

from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


class SignalTypeEnum(Enum):
    """Standardized signal types."""
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    LONG_EXIT = "long_exit"
    SHORT_EXIT = "short_exit"
    UNIFIED_EXIT = "exit"  # For consolidated exits
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class SignalFormat:
    """
    Canonical signal format for consistent communication between modules.
    
    This defines the standard structure that both trading_signals.py and 
    simulation_engine.py must use for signal exchange.
    """
    # Entry/Exit Signals (Boolean Series)
    long_entries: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    short_entries: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    long_exits: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    short_exits: pd.Series = field(default_factory=lambda: pd.Series(dtype=bool))
    
    # Price Information (Numeric Series)
    entry_prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    exit_prices: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    
    # Risk Management Levels (Numeric Series)
    sl_levels: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    tp_levels: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    sl_price_levels: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    tp_price_levels: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    
    # Metadata
    index: pd.DatetimeIndex = field(default_factory=pd.DatetimeIndex)
    symbol: Optional[str] = None
    timeframe: Optional[str] = None
    
    def to_dict(self) -> Dict[str, pd.Series]:
        """Convert to dictionary format for compatibility."""
        # Create unified exits for backward compatibility with renderer
        unified_exits = self.long_exits | self.short_exits
        
        return {
            'long_entries': self.long_entries,
            'short_entries': self.short_entries,
            'long_exits': self.long_exits,
            'short_exits': self.short_exits,
            'exits': unified_exits,  # Unified exits for renderer compatibility
            'entry_prices': self.entry_prices,
            'exit_prices': self.exit_prices,
            'sl_levels': self.sl_levels,
            'tp_levels': self.tp_levels,
            'sl_price_levels': self.sl_price_levels,
            'tp_price_levels': self.tp_price_levels
        }
    
    @classmethod
    def from_dict(cls, signals_dict: Dict[str, pd.Series], index: pd.DatetimeIndex) -> 'SignalFormat':
        """Create SignalFormat from dictionary."""
        # Initialize with empty series if not provided
        def get_series(key: str, dtype=bool) -> pd.Series:
            if key in signals_dict and signals_dict[key] is not None:
                series = signals_dict[key]
                if hasattr(series, 'reindex'):
                    return series.reindex(index, fill_value=False if dtype == bool else np.nan)
                else:
                    # Convert to Series if needed
                    return pd.Series(series, index=index, dtype=dtype)
            else:
                fill_value = False if dtype == bool else np.nan
                return pd.Series(fill_value, index=index, dtype=dtype)
        
        return cls(
            long_entries=get_series('long_entries', bool),
            short_entries=get_series('short_entries', bool),
            long_exits=get_series('long_exits', bool),
            short_exits=get_series('short_exits', bool),
            entry_prices=get_series('entry_prices', float),
            exit_prices=get_series('exit_prices', float),
            sl_levels=get_series('sl_levels', float),
            tp_levels=get_series('tp_levels', float),
            sl_price_levels=get_series('sl_price_levels', float),
            tp_price_levels=get_series('tp_price_levels', float),
            index=index
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get signal summary statistics."""
        return {
            'long_entries_count': self.long_entries.sum() if hasattr(self.long_entries, 'sum') else 0,
            'short_entries_count': self.short_entries.sum() if hasattr(self.short_entries, 'sum') else 0,
            'long_exits_count': self.long_exits.sum() if hasattr(self.long_exits, 'sum') else 0,
            'short_exits_count': self.short_exits.sum() if hasattr(self.short_exits, 'sum') else 0,
            'total_entries': (self.long_entries.sum() if hasattr(self.long_entries, 'sum') else 0) + 
                           (self.short_entries.sum() if hasattr(self.short_entries, 'sum') else 0),
            'total_exits': (self.long_exits.sum() if hasattr(self.long_exits, 'sum') else 0) + 
                         (self.short_exits.sum() if hasattr(self.short_exits, 'sum') else 0),
            'entries_with_prices': (~self.entry_prices.isna()).sum() if hasattr(self.entry_prices, 'isna') else 0,
            'exits_with_prices': (~self.exit_prices.isna()).sum() if hasattr(self.exit_prices, 'isna') else 0,
            'sl_levels_count': (~self.sl_levels.isna()).sum() if hasattr(self.sl_levels, 'isna') else 0,
            'tp_levels_count': (~self.tp_levels.isna()).sum() if hasattr(self.tp_levels, 'isna') else 0
        }
    
    def copy(self) -> 'SignalFormat':
        """Create a deep copy of the SignalFormat."""
        return SignalFormat(
            long_entries=self.long_entries.copy() if hasattr(self.long_entries, 'copy') else self.long_entries,
            short_entries=self.short_entries.copy() if hasattr(self.short_entries, 'copy') else self.short_entries,
            long_exits=self.long_exits.copy() if hasattr(self.long_exits, 'copy') else self.long_exits,
            short_exits=self.short_exits.copy() if hasattr(self.short_exits, 'copy') else self.short_exits,
            entry_prices=self.entry_prices.copy() if hasattr(self.entry_prices, 'copy') else self.entry_prices,
            exit_prices=self.exit_prices.copy() if hasattr(self.exit_prices, 'copy') else self.exit_prices,
            sl_levels=self.sl_levels.copy() if hasattr(self.sl_levels, 'copy') else self.sl_levels,
            tp_levels=self.tp_levels.copy() if hasattr(self.tp_levels, 'copy') else self.tp_levels,
            sl_price_levels=self.sl_price_levels.copy() if hasattr(self.sl_price_levels, 'copy') else self.sl_price_levels,
            tp_price_levels=self.tp_price_levels.copy() if hasattr(self.tp_price_levels, 'copy') else self.tp_price_levels,
            index=self.index.copy() if hasattr(self.index, 'copy') else self.index
        )


@dataclass
class ValidationResult:
    """Result of signal validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
        
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)


class SignalValidator:
    """
    Validates signal integrity and consistency between modules.
    """
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
    
    def validate_signals(self, signals: Union[SignalFormat, Dict[str, pd.Series]]) -> ValidationResult:
        """
        Comprehensive signal validation.
        
        Args:
            signals: Signals in SignalFormat or dictionary format
            
        Returns:
            ValidationResult with validation status and details
        """
        result = ValidationResult(is_valid=True)
        
        # Convert to SignalFormat if needed
        if isinstance(signals, dict):
            if 'index' in signals:
                index = signals['index']
            else:
                # Try to infer index from first signal
                for signal_series in signals.values():
                    if hasattr(signal_series, 'index'):
                        index = signal_series.index
                        break
                else:
                    result.add_error("Cannot determine index for signal validation")
                    return result
            
            try:
                signals = SignalFormat.from_dict(signals, index)
            except Exception as e:
                result.add_error(f"Failed to convert signals to standard format: {e}")
                return result
        
        # Get summary for validation
        summary = signals.get_summary()
        result.summary = summary
        
        # 1. Check for conflicting entry signals
        conflicting_entries = signals.long_entries & signals.short_entries
        if conflicting_entries.any():
            conflict_count = conflicting_entries.sum()
            result.add_error(f"Found {conflict_count} conflicting entry signals (both long and short)")
        
        # 2. Check entry/exit balance
        total_entries = summary['total_entries']
        total_exits = summary['total_exits']
        
        if total_entries == 0:
            result.add_warning("No entry signals found")
        
        if total_exits == 0:
            result.add_warning("No exit signals found")
        
        # Allow some flexibility in entry/exit counts (±1 for open positions)
        entry_exit_diff = abs(total_entries - total_exits)
        if entry_exit_diff > 1:
            result.add_warning(f"Entry/exit imbalance: {total_entries} entries vs {total_exits} exits (diff: {entry_exit_diff})")
        
        # 3. Check for entries without prices
        entries_without_prices = (signals.long_entries | signals.short_entries) & signals.entry_prices.isna()
        if entries_without_prices.any():
            count = entries_without_prices.sum()
            if self.strict_mode:
                result.add_error(f"Found {count} entry signals without prices")
            else:
                result.add_warning(f"Found {count} entry signals without prices")
        
        # 4. Check for exits without prices
        exits_without_prices = (signals.long_exits | signals.short_exits) & signals.exit_prices.isna()
        if exits_without_prices.any():
            count = exits_without_prices.sum()
            if self.strict_mode:
                result.add_error(f"Found {count} exit signals without prices")
            else:
                result.add_warning(f"Found {count} exit signals without prices")
        
        # 5. Check for reasonable price ranges
        if not signals.entry_prices.isna().all():
            entry_prices = signals.entry_prices.dropna()
            if (entry_prices <= 0).any():
                result.add_error("Found non-positive entry prices")
        
        if not signals.exit_prices.isna().all():
            exit_prices = signals.exit_prices.dropna()
            if (exit_prices <= 0).any():
                result.add_error("Found non-positive exit prices")
        
        # 6. Check index consistency
        expected_index = signals.index
        for attr_name in ['long_entries', 'short_entries', 'long_exits', 'short_exits']:
            signal_series = getattr(signals, attr_name)
            if hasattr(signal_series, 'index') and not signal_series.index.equals(expected_index):
                result.add_error(f"{attr_name} has inconsistent index")
        
        # 7. Check timezone consistency
        if hasattr(expected_index, 'tz'):
            tz_info = expected_index.tz
            for attr_name in ['entry_prices', 'exit_prices']:
                price_series = getattr(signals, attr_name)
                if hasattr(price_series, 'index') and hasattr(price_series.index, 'tz'):
                    if price_series.index.tz != tz_info:
                        result.add_warning(f"{attr_name} has timezone mismatch")
        
        logger.info(f"Signal validation completed: {'PASSED' if result.is_valid else 'FAILED'}")
        logger.info(f"Summary: {summary}")
        
        for warning in result.warnings:
            logger.warning(f"Validation warning: {warning}")
        
        for error in result.errors:
            logger.error(f"Validation error: {error}")
        
        return result
    
    def validate_signal_consistency(self, 
                                  chart_signals: Union[SignalFormat, Dict[str, pd.Series]], 
                                  portfolio_signals: Union[SignalFormat, Dict[str, pd.Series]]) -> ValidationResult:
        """
        Validate consistency between chart signals and portfolio signals.
        
        Args:
            chart_signals: Signals used for chart rendering
            portfolio_signals: Signals from portfolio simulation
            
        Returns:
            ValidationResult with consistency analysis
        """
        result = ValidationResult(is_valid=True)
        
        # Convert both to SignalFormat
        if isinstance(chart_signals, dict):
            chart_index = chart_signals.get('index') or next(iter(chart_signals.values())).index
            chart_signals = SignalFormat.from_dict(chart_signals, chart_index)
        
        if isinstance(portfolio_signals, dict):
            portfolio_index = portfolio_signals.get('index') or next(iter(portfolio_signals.values())).index
            portfolio_signals = SignalFormat.from_dict(portfolio_signals, portfolio_index)
        
        # Compare summaries
        chart_summary = chart_signals.get_summary()
        portfolio_summary = portfolio_signals.get_summary()
        
        result.summary = {
            'chart_signals': chart_summary,
            'portfolio_signals': portfolio_summary,
            'differences': {}
        }
        
        # Check key differences
        for key in ['total_entries', 'total_exits', 'long_entries_count', 'short_entries_count']:
            chart_val = chart_summary.get(key, 0)
            portfolio_val = portfolio_summary.get(key, 0)
            diff = abs(chart_val - portfolio_val)
            
            result.summary['differences'][key] = {
                'chart': chart_val,
                'portfolio': portfolio_val,
                'difference': diff
            }
            
            if diff > 0:
                result.add_warning(f"{key} mismatch: chart={chart_val}, portfolio={portfolio_val}")
        
        return result


class TradeSignalCrossReference:
    """
    Cross-references portfolio trade records with signal data for consistency validation.
    """
    
    def __init__(self, portfolio: vbt.Portfolio):
        self.portfolio = portfolio
        
    def validate_against_csv(self, csv_file_path: str, signals: Union[SignalFormat, Dict[str, pd.Series]]) -> ValidationResult:
        """
        Validate signals against CSV trade records.
        
        Args:
            csv_file_path: Path to performance metrics CSV file
            signals: Signal data to validate
            
        Returns:
            ValidationResult with cross-reference analysis
        """
        result = ValidationResult(is_valid=True)
        
        try:
            # Load CSV trade records
            trades_df = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(trades_df)} trades from CSV: {csv_file_path}")
            
            # Convert signals to standard format
            if isinstance(signals, dict):
                signal_index = signals.get('index') or next(iter(signals.values())).index
                signals = SignalFormat.from_dict(signals, signal_index)
            
            # Extract trade information from CSV
            csv_summary = self._analyze_csv_trades(trades_df)
            signal_summary = signals.get_summary()
            
            result.summary = {
                'csv_trades': csv_summary,
                'signal_data': signal_summary,
                'cross_reference': {}
            }
            
            # Cross-reference key metrics
            csv_total_trades = csv_summary['total_trades']
            signal_total_entries = signal_summary['total_entries']
            
            # Check trade count consistency
            trade_diff = abs(csv_total_trades - signal_total_entries)
            if trade_diff > 1:  # Allow ±1 for open positions
                result.add_warning(f"Trade count mismatch: CSV has {csv_total_trades} trades, signals have {signal_total_entries} entries")
            
            # Detailed timestamp cross-reference
            timestamp_analysis = self._cross_reference_timestamps(trades_df, signals)
            result.summary['cross_reference']['timestamps'] = timestamp_analysis
            
            if timestamp_analysis['mismatched_entries'] > 0:
                result.add_warning(f"Found {timestamp_analysis['mismatched_entries']} entry timestamp mismatches")
            
            if timestamp_analysis['mismatched_exits'] > 0:
                result.add_warning(f"Found {timestamp_analysis['mismatched_exits']} exit timestamp mismatches")
            
            # Price consistency checks
            price_analysis = self._cross_reference_prices(trades_df, signals)
            result.summary['cross_reference']['prices'] = price_analysis
            
            if price_analysis['entry_price_mismatches'] > 0:
                result.add_warning(f"Found {price_analysis['entry_price_mismatches']} entry price mismatches")
            
            logger.info("Cross-reference validation completed")
            
        except Exception as e:
            result.add_error(f"Cross-reference validation failed: {e}")
            logger.error(f"Cross-reference validation error: {e}")
        
        return result
    
    def _analyze_csv_trades(self, trades_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trade information from CSV."""
        analysis = {
            'total_trades': len(trades_df),
            'long_trades': 0,
            'short_trades': 0,
            'entry_timestamps': [],
            'exit_timestamps': [],
            'entry_prices': [],
            'exit_prices': []
        }
        
        # Parse direction information
        if 'Direction' in trades_df.columns:
            direction_counts = trades_df['Direction'].value_counts()
            analysis['long_trades'] = direction_counts.get('Long', 0)
            analysis['short_trades'] = direction_counts.get('Short', 0)
        
        # Extract timestamps
        for timestamp_col in ['Entry Index', 'Entry Timestamp', 'Entry Time']:
            if timestamp_col in trades_df.columns:
                analysis['entry_timestamps'] = pd.to_datetime(trades_df[timestamp_col]).tolist()
                break
        
        for timestamp_col in ['Exit Index', 'Exit Timestamp', 'Exit Time']:
            if timestamp_col in trades_df.columns:
                analysis['exit_timestamps'] = pd.to_datetime(trades_df[timestamp_col]).tolist()
                break
        
        # Extract prices
        for price_col in ['Entry Price', 'Avg Entry Price', 'Entry Avg Price']:
            if price_col in trades_df.columns:
                analysis['entry_prices'] = trades_df[price_col].tolist()
                break
        
        for price_col in ['Exit Price', 'Avg Exit Price', 'Exit Avg Price']:
            if price_col in trades_df.columns:
                analysis['exit_prices'] = trades_df[price_col].tolist()
                break
        
        return analysis
    
    def _cross_reference_timestamps(self, trades_df: pd.DataFrame, signals: SignalFormat) -> Dict[str, Any]:
        """Cross-reference timestamps between CSV and signals."""
        analysis = {
            'matched_entries': 0,
            'mismatched_entries': 0,
            'matched_exits': 0,
            'mismatched_exits': 0,
            'entry_details': [],
            'exit_details': []
        }
        
        # Get signal timestamps
        signal_entry_times = signals.long_entries[signals.long_entries].index.tolist() + \
                           signals.short_entries[signals.short_entries].index.tolist()
        signal_exit_times = signals.long_exits[signals.long_exits].index.tolist() + \
                          signals.short_exits[signals.short_exits].index.tolist()
        
        # Cross-reference entry timestamps
        for _, trade in trades_df.iterrows():
            trade_entry_time = None
            for col in ['Entry Index', 'Entry Timestamp', 'Entry Time']:
                if col in trades_df.columns:
                    trade_entry_time = pd.to_datetime(trade[col])
                    break
            
            if trade_entry_time:
                # Check if this timestamp exists in signals (with some tolerance)
                matched = any(abs((t - trade_entry_time).total_seconds()) < 3600 for t in signal_entry_times)
                if matched:
                    analysis['matched_entries'] += 1
                else:
                    analysis['mismatched_entries'] += 1
                    analysis['entry_details'].append({
                        'csv_timestamp': trade_entry_time,
                        'closest_signal': min(signal_entry_times, key=lambda x: abs((x - trade_entry_time).total_seconds())) if signal_entry_times else None
                    })
        
        # Cross-reference exit timestamps
        for _, trade in trades_df.iterrows():
            trade_exit_time = None
            for col in ['Exit Index', 'Exit Timestamp', 'Exit Time']:
                if col in trades_df.columns:
                    trade_exit_time = pd.to_datetime(trade[col])
                    break
            
            if trade_exit_time:
                matched = any(abs((t - trade_exit_time).total_seconds()) < 3600 for t in signal_exit_times)
                if matched:
                    analysis['matched_exits'] += 1
                else:
                    analysis['mismatched_exits'] += 1
                    analysis['exit_details'].append({
                        'csv_timestamp': trade_exit_time,
                        'closest_signal': min(signal_exit_times, key=lambda x: abs((x - trade_exit_time).total_seconds())) if signal_exit_times else None
                    })
        
        return analysis
    
    def _cross_reference_prices(self, trades_df: pd.DataFrame, signals: SignalFormat) -> Dict[str, Any]:
        """Cross-reference prices between CSV and signals."""
        analysis = {
            'entry_price_matches': 0,
            'entry_price_mismatches': 0,
            'exit_price_matches': 0,
            'exit_price_mismatches': 0,
            'price_tolerance': 0.01  # 1% tolerance
        }
        
        tolerance = analysis['price_tolerance']
        
        # Cross-reference entry prices
        for _, trade in trades_df.iterrows():
            csv_entry_price = None
            for col in ['Entry Price', 'Avg Entry Price', 'Entry Avg Price']:
                if col in trades_df.columns:
                    csv_entry_price = trade[col]
                    break
            
            if csv_entry_price and not pd.isna(csv_entry_price):
                # Find corresponding signal price
                trade_time = None
                for col in ['Entry Index', 'Entry Timestamp', 'Entry Time']:
                    if col in trades_df.columns:
                        trade_time = pd.to_datetime(trade[col])
                        break
                
                if trade_time and trade_time in signals.entry_prices.index:
                    signal_price = signals.entry_prices.loc[trade_time]
                    if not pd.isna(signal_price):
                        price_diff = abs(csv_entry_price - signal_price) / csv_entry_price
                        if price_diff <= tolerance:
                            analysis['entry_price_matches'] += 1
                        else:
                            analysis['entry_price_mismatches'] += 1
        
        # Cross-reference exit prices
        for _, trade in trades_df.iterrows():
            csv_exit_price = None
            for col in ['Exit Price', 'Avg Exit Price', 'Exit Avg Price']:
                if col in trades_df.columns:
                    csv_exit_price = trade[col]
                    break
            
            if csv_exit_price and not pd.isna(csv_exit_price):
                trade_time = None
                for col in ['Exit Index', 'Exit Timestamp', 'Exit Time']:
                    if col in trades_df.columns:
                        trade_time = pd.to_datetime(trade[col])
                        break
                
                if trade_time and trade_time in signals.exit_prices.index:
                    signal_price = signals.exit_prices.loc[trade_time]
                    if not pd.isna(signal_price):
                        price_diff = abs(csv_exit_price - signal_price) / csv_exit_price
                        if price_diff <= tolerance:
                            analysis['exit_price_matches'] += 1
                        else:
                            analysis['exit_price_mismatches'] += 1
        
        return analysis


# Utility functions for signal interface
def convert_legacy_signals(legacy_signals: Dict[str, Any], index: pd.DatetimeIndex) -> SignalFormat:
    """
    Convert legacy signal format to standardized SignalFormat.
    
    This handles the conversion from the old unified 'exits' format to 
    the new separated 'long_exits' and 'short_exits' format.
    """
    # Handle unified exits by splitting them based on entry types
    if 'exits' in legacy_signals and 'exits' not in ['long_exits', 'short_exits']:
        unified_exits = legacy_signals['exits']
        long_entries = legacy_signals.get('long_entries', pd.Series(False, index=index))
        short_entries = legacy_signals.get('short_entries', pd.Series(False, index=index))
        
        # Create separated exits based on position tracking
        long_exits = pd.Series(False, index=index)
        short_exits = pd.Series(False, index=index)
        
        # Simple heuristic: assign exits to the most recent entry type
        current_position = None  # 'long', 'short', or None
        
        for timestamp in index:
            if long_entries.get(timestamp, False):
                current_position = 'long'
            elif short_entries.get(timestamp, False):
                current_position = 'short'
            elif unified_exits.get(timestamp, False):
                if current_position == 'long':
                    long_exits.loc[timestamp] = True
                elif current_position == 'short':
                    short_exits.loc[timestamp] = True
                current_position = None
        
        # Replace unified exits with separated exits
        legacy_signals = dict(legacy_signals)  # Make a copy
        legacy_signals['long_exits'] = long_exits
        legacy_signals['short_exits'] = short_exits
        del legacy_signals['exits']  # Remove unified exits
    
    return SignalFormat.from_dict(legacy_signals, index)


def normalize_timestamps(signals: SignalFormat, target_tz=None) -> SignalFormat:
    """Normalize all timestamps in signals to consistent timezone."""
    if target_tz is None:
        target_tz = signals.index.tz
    
    # Create a copy with normalized timestamps
    normalized_signals = SignalFormat()
    normalized_signals.index = signals.index
    
    if target_tz is not None:
        for attr_name in ['long_entries', 'short_entries', 'long_exits', 'short_exits', 
                         'entry_prices', 'exit_prices', 'sl_levels', 'tp_levels']:
            series = getattr(signals, attr_name)
            if hasattr(series, 'index') and hasattr(series.index, 'tz'):
                if series.index.tz != target_tz:
                    if series.index.tz is None:
                        series = series.copy()
                        series.index = series.index.tz_localize(target_tz)
                    else:
                        series = series.copy()
                        series.index = series.index.tz_convert(target_tz)
            setattr(normalized_signals, attr_name, series)
    
    return normalized_signals


def create_signal_summary_report(signals: Union[SignalFormat, Dict[str, pd.Series]], 
                                portfolio: Optional[vbt.Portfolio] = None,
                                csv_file_path: Optional[str] = None) -> str:
    """
    Create a comprehensive signal summary report.
    
    Args:
        signals: Signal data
        portfolio: Portfolio object for cross-reference (optional)
        csv_file_path: CSV file path for cross-reference (optional)
        
    Returns:
        Formatted summary report
    """
    # Convert to SignalFormat if needed
    if isinstance(signals, dict):
        index = signals.get('index') or next(iter(signals.values())).index
        signals = SignalFormat.from_dict(signals, index)
    
    summary = signals.get_summary()
    
    report = f"""
Signal Analysis Report
=====================

Basic Signal Counts:
- Long Entries: {summary['long_entries_count']}
- Short Entries: {summary['short_entries_count']}
- Long Exits: {summary['long_exits_count']}
- Short Exits: {summary['short_exits_count']}
- Total Entries: {summary['total_entries']}
- Total Exits: {summary['total_exits']}

Price Information:
- Entries with Prices: {summary['entries_with_prices']} / {summary['total_entries']}
- Exits with Prices: {summary['exits_with_prices']} / {summary['total_exits']}

Risk Management:
- Stop Loss Levels: {summary['sl_levels_count']}
- Take Profit Levels: {summary['tp_levels_count']}

Entry/Exit Balance: {'✓ BALANCED' if abs(summary['total_entries'] - summary['total_exits']) <= 1 else '⚠ IMBALANCED'}
Price Coverage: {'✓ COMPLETE' if summary['entries_with_prices'] == summary['total_entries'] and summary['exits_with_prices'] == summary['total_exits'] else '⚠ INCOMPLETE'}
"""
    
    # Add portfolio cross-reference if available
    if portfolio is not None:
        try:
            trades = portfolio.trades.records_readable
            portfolio_stats = {
                'total_trades': len(trades),
                'long_trades': len(trades[trades['Direction'] == 'Long']) if 'Direction' in trades.columns else 0,
                'short_trades': len(trades[trades['Direction'] == 'Short']) if 'Direction' in trades.columns else 0
            }
            
            report += f"""
Portfolio Cross-Reference:
- Portfolio Total Trades: {portfolio_stats['total_trades']}
- Portfolio Long Trades: {portfolio_stats['long_trades']}
- Portfolio Short Trades: {portfolio_stats['short_trades']}
- Match Status: {'✓ MATCH' if portfolio_stats['total_trades'] == summary['total_entries'] else '⚠ MISMATCH'}
"""
        except Exception as e:
            report += f"\nPortfolio Cross-Reference: ❌ ERROR - {e}"
    
    # Add CSV cross-reference if available
    if csv_file_path is not None:
        try:
            trades_df = pd.read_csv(csv_file_path)
            csv_trade_count = len(trades_df)
            report += f"""
CSV Cross-Reference:
- CSV Trade Count: {csv_trade_count}
- Signal Entry Count: {summary['total_entries']}
- Match Status: {'✓ MATCH' if csv_trade_count == summary['total_entries'] else '⚠ MISMATCH'}
"""
        except Exception as e:
            report += f"\nCSV Cross-Reference: ❌ ERROR - {e}"
    
    return report 