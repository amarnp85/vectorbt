#!/usr/bin/env python3
"""
Signal Validation Tool

This tool provides comprehensive validation and cross-reference capabilities for trading signals.
It helps identify inconsistencies between signal extraction and actual portfolio trades.

Features:
- Cross-reference signals with CSV trade records
- Validate signal integrity and consistency
- Generate detailed validation reports
- Identify timing issues and missing data
- Provide recommendations for fixes

Usage:
    python tools/signal_validation_tool.py --csv results/symbols/SOL_USDT/performance_metrics.trades.csv --validate-signals
"""

import sys
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from backtester.signals.signal_interface import (
        SignalFormat, SignalValidator, ValidationResult,
        TradeSignalCrossReference, create_signal_summary_report
    )
    from backtester.utilities.structured_logging import get_logger
    UNIFIED_INTERFACE_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Unified signal interface not available: {e}")
    UNIFIED_INTERFACE_AVAILABLE = False

logger = get_logger("signal_validator")


class SignalValidationTool:
    """
    Comprehensive signal validation and cross-reference tool.
    """
    
    def __init__(self, csv_file_path: Optional[str] = None):
        self.csv_file_path = csv_file_path
        self.trades_df = None
        self.validator = None
        
        if UNIFIED_INTERFACE_AVAILABLE:
            self.validator = SignalValidator(strict_mode=False)
        
        if csv_file_path and os.path.exists(csv_file_path):
            self.load_csv_trades()
    
    def load_csv_trades(self):
        """Load trade data from CSV file."""
        try:
            self.trades_df = pd.read_csv(self.csv_file_path)
            logger.info(f"Loaded {len(self.trades_df)} trades from {self.csv_file_path}")
            logger.debug(f"CSV columns: {list(self.trades_df.columns)}")
        except Exception as e:
            logger.error(f"Failed to load CSV trades: {e}")
            self.trades_df = None
    
    def analyze_csv_trades(self) -> Dict[str, Any]:
        """Analyze trade data from CSV."""
        if self.trades_df is None:
            return {"error": "No CSV trade data loaded"}
        
        analysis = {
            "total_trades": len(self.trades_df),
            "columns": list(self.trades_df.columns),
            "date_range": {},
            "direction_analysis": {},
            "price_analysis": {},
            "pnl_analysis": {}
        }
        
        # Date range analysis
        for date_col in ['Entry Index', 'Exit Index']:
            if date_col in self.trades_df.columns:
                dates = pd.to_datetime(self.trades_df[date_col])
                analysis["date_range"][date_col] = {
                    "start": dates.min(),
                    "end": dates.max(),
                    "count": dates.notna().sum()
                }
        
        # Direction analysis
        if 'Direction' in self.trades_df.columns:
            direction_counts = self.trades_df['Direction'].value_counts()
            analysis["direction_analysis"] = {
                "long_trades": direction_counts.get('Long', 0),
                "short_trades": direction_counts.get('Short', 0),
                "unknown": len(self.trades_df) - direction_counts.sum()
            }
        
        # Price analysis
        for price_col in ['Entry Price', 'Exit Price', 'Avg Entry Price', 'Avg Exit Price']:
            if price_col in self.trades_df.columns:
                prices = self.trades_df[price_col].dropna()
                if len(prices) > 0:
                    analysis["price_analysis"][price_col] = {
                        "count": len(prices),
                        "min": prices.min(),
                        "max": prices.max(),
                        "mean": prices.mean(),
                        "missing": self.trades_df[price_col].isna().sum()
                    }
        
        # PnL analysis
        if 'PnL' in self.trades_df.columns:
            pnl = self.trades_df['PnL'].dropna()
            if len(pnl) > 0:
                analysis["pnl_analysis"] = {
                    "total_pnl": pnl.sum(),
                    "winning_trades": (pnl > 0).sum(),
                    "losing_trades": (pnl < 0).sum(),
                    "avg_win": pnl[pnl > 0].mean() if (pnl > 0).any() else 0,
                    "avg_loss": pnl[pnl < 0].mean() if (pnl < 0).any() else 0
                }
        
        return analysis
    
    def validate_signals_format(self, signals: Any) -> ValidationResult:
        """Validate signal format and integrity."""
        if not UNIFIED_INTERFACE_AVAILABLE:
            result = ValidationResult(is_valid=False)
            result.add_error("Unified signal interface not available")
            return result
        
        return self.validator.validate_signals(signals)
    
    def cross_reference_with_csv(self, signals: Any) -> Dict[str, Any]:
        """
        Cross-reference signals with CSV trade data.
        """
        if self.trades_df is None:
            return {"error": "No CSV trade data available for cross-reference"}
        
        if not UNIFIED_INTERFACE_AVAILABLE:
            return {"error": "Unified signal interface not available"}
        
        try:
            # Convert signals to unified format if needed
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
                        return {"error": "Cannot determine signal index"}
                
                from backtester.signals.signal_interface import convert_legacy_signals
                unified_signals = convert_legacy_signals(signals, index)
            else:
                unified_signals = signals
            
            # Perform detailed cross-reference
            cross_ref = {
                "csv_analysis": self.analyze_csv_trades(),
                "signal_analysis": unified_signals.get_summary() if hasattr(unified_signals, 'get_summary') else {},
                "cross_reference": self._detailed_cross_reference(unified_signals),
                "recommendations": []
            }
            
            # Generate recommendations
            cross_ref["recommendations"] = self._generate_recommendations(cross_ref)
            
            return cross_ref
            
        except Exception as e:
            logger.error(f"Cross-reference failed: {e}")
            return {"error": f"Cross-reference failed: {e}"}
    
    def _detailed_cross_reference(self, signals: Any) -> Dict[str, Any]:
        """Perform detailed cross-reference analysis."""
        analysis = {
            "entry_analysis": {},
            "exit_analysis": {},
            "timing_analysis": {},
            "price_analysis": {},
            "missing_data": {}
        }
        
        # Entry analysis
        signal_entries = signals.long_entries.sum() + signals.short_entries.sum()
        csv_entries = len(self.trades_df)
        
        analysis["entry_analysis"] = {
            "signal_entries": signal_entries,
            "csv_trades": csv_entries,
            "difference": signal_entries - csv_entries,
            "match_ratio": signal_entries / csv_entries if csv_entries > 0 else 0
        }
        
        # Exit analysis
        signal_exits = signals.long_exits.sum() + signals.short_exits.sum()
        csv_exits = len(self.trades_df)  # Each trade has an exit
        
        analysis["exit_analysis"] = {
            "signal_exits": signal_exits,
            "csv_exits": csv_exits,
            "difference": signal_exits - csv_exits,
            "match_ratio": signal_exits / csv_exits if csv_exits > 0 else 0
        }
        
        # Timing analysis
        analysis["timing_analysis"] = self._analyze_timing_matches(signals)
        
        # Price analysis
        analysis["price_analysis"] = self._analyze_price_matches(signals)
        
        # Missing data analysis
        analysis["missing_data"] = self._analyze_missing_data(signals)
        
        return analysis
    
    def _analyze_timing_matches(self, signals: Any) -> Dict[str, Any]:
        """Analyze timing matches between signals and CSV."""
        timing_analysis = {
            "entry_timing_matches": 0,
            "entry_timing_mismatches": 0,
            "exit_timing_matches": 0,
            "exit_timing_mismatches": 0,
            "timing_tolerance_hours": 1
        }
        
        try:
            # Get signal timestamps
            signal_entry_times = (
                signals.long_entries[signals.long_entries].index.tolist() +
                signals.short_entries[signals.short_entries].index.tolist()
            )
            signal_exit_times = (
                signals.long_exits[signals.long_exits].index.tolist() +
                signals.short_exits[signals.short_exits].index.tolist()
            )
            
            # Check entry timing matches
            for _, trade in self.trades_df.iterrows():
                # Entry timing
                for entry_col in ['Entry Index', 'Entry Timestamp']:
                    if entry_col in self.trades_df.columns:
                        csv_entry_time = pd.to_datetime(trade[entry_col])
                        # Check if any signal entry is within tolerance
                        matched = any(
                            abs((signal_time - csv_entry_time).total_seconds()) < 3600
                            for signal_time in signal_entry_times
                        )
                        if matched:
                            timing_analysis["entry_timing_matches"] += 1
                        else:
                            timing_analysis["entry_timing_mismatches"] += 1
                        break
                
                # Exit timing
                for exit_col in ['Exit Index', 'Exit Timestamp']:
                    if exit_col in self.trades_df.columns:
                        csv_exit_time = pd.to_datetime(trade[exit_col])
                        matched = any(
                            abs((signal_time - csv_exit_time).total_seconds()) < 3600
                            for signal_time in signal_exit_times
                        )
                        if matched:
                            timing_analysis["exit_timing_matches"] += 1
                        else:
                            timing_analysis["exit_timing_mismatches"] += 1
                        break
            
        except Exception as e:
            timing_analysis["error"] = str(e)
        
        return timing_analysis
    
    def _analyze_price_matches(self, signals: Any) -> Dict[str, Any]:
        """Analyze price matches between signals and CSV."""
        price_analysis = {
            "entry_price_matches": 0,
            "entry_price_mismatches": 0,
            "exit_price_matches": 0,
            "exit_price_mismatches": 0,
            "price_tolerance_percent": 1.0
        }
        
        try:
            tolerance = price_analysis["price_tolerance_percent"] / 100.0
            
            # Check entry price matches
            for _, trade in self.trades_df.iterrows():
                # Entry prices
                csv_entry_price = None
                for price_col in ['Entry Price', 'Avg Entry Price']:
                    if price_col in self.trades_df.columns:
                        csv_entry_price = trade[price_col]
                        break
                
                if csv_entry_price and not pd.isna(csv_entry_price):
                    # Find corresponding signal entry time
                    for entry_col in ['Entry Index', 'Entry Timestamp']:
                        if entry_col in self.trades_df.columns:
                            csv_entry_time = pd.to_datetime(trade[entry_col])
                            if csv_entry_time in signals.entry_prices.index:
                                signal_price = signals.entry_prices.loc[csv_entry_time]
                                if not pd.isna(signal_price):
                                    price_diff = abs(csv_entry_price - signal_price) / csv_entry_price
                                    if price_diff <= tolerance:
                                        price_analysis["entry_price_matches"] += 1
                                    else:
                                        price_analysis["entry_price_mismatches"] += 1
                            break
                
                # Exit prices
                csv_exit_price = None
                for price_col in ['Exit Price', 'Avg Exit Price']:
                    if price_col in self.trades_df.columns:
                        csv_exit_price = trade[price_col]
                        break
                
                if csv_exit_price and not pd.isna(csv_exit_price):
                    for exit_col in ['Exit Index', 'Exit Timestamp']:
                        if exit_col in self.trades_df.columns:
                            csv_exit_time = pd.to_datetime(trade[exit_col])
                            if csv_exit_time in signals.exit_prices.index:
                                signal_price = signals.exit_prices.loc[csv_exit_time]
                                if not pd.isna(signal_price):
                                    price_diff = abs(csv_exit_price - signal_price) / csv_exit_price
                                    if price_diff <= tolerance:
                                        price_analysis["exit_price_matches"] += 1
                                    else:
                                        price_analysis["exit_price_mismatches"] += 1
                            break
            
        except Exception as e:
            price_analysis["error"] = str(e)
        
        return price_analysis
    
    def _analyze_missing_data(self, signals: Any) -> Dict[str, Any]:
        """Analyze missing data in signals."""
        missing_analysis = {
            "entries_without_prices": 0,
            "exits_without_prices": 0,
            "entries_without_exits": 0,
            "orphaned_exits": 0
        }
        
        try:
            # Count entries without prices
            all_entries = signals.long_entries | signals.short_entries
            entries_without_prices = all_entries & signals.entry_prices.isna()
            missing_analysis["entries_without_prices"] = entries_without_prices.sum()
            
            # Count exits without prices
            all_exits = signals.long_exits | signals.short_exits
            exits_without_prices = all_exits & signals.exit_prices.isna()
            missing_analysis["exits_without_prices"] = exits_without_prices.sum()
            
            # More sophisticated position tracking would be needed for 
            # entries_without_exits and orphaned_exits analysis
            
        except Exception as e:
            missing_analysis["error"] = str(e)
        
        return missing_analysis
    
    def _generate_recommendations(self, cross_ref: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on cross-reference analysis."""
        recommendations = []
        
        try:
            # Check entry/exit balance
            if "cross_reference" in cross_ref and "entry_analysis" in cross_ref["cross_reference"]:
                entry_diff = cross_ref["cross_reference"]["entry_analysis"]["difference"]
                if abs(entry_diff) > 1:
                    if entry_diff > 0:
                        recommendations.append(f"Signal extraction is creating {entry_diff} extra entry signals. Check for duplicate signal extraction from trades and orders.")
                    else:
                        recommendations.append(f"Signal extraction is missing {abs(entry_diff)} entry signals. Check if all trades are being processed.")
            
            # Check timing issues
            if "timing_analysis" in cross_ref["cross_reference"]:
                timing = cross_ref["cross_reference"]["timing_analysis"]
                if timing.get("entry_timing_mismatches", 0) > timing.get("entry_timing_matches", 0):
                    recommendations.append("High number of entry timing mismatches. Check signal timing configuration and execution delay settings.")
                
                if timing.get("exit_timing_mismatches", 0) > timing.get("exit_timing_matches", 0):
                    recommendations.append("High number of exit timing mismatches. Verify that exits are being extracted correctly and not double-counted.")
            
            # Check missing data
            if "missing_data" in cross_ref["cross_reference"]:
                missing = cross_ref["cross_reference"]["missing_data"]
                if missing.get("entries_without_prices", 0) > 0:
                    recommendations.append(f"{missing['entries_without_prices']} entries are missing prices. Check price extraction logic in _extract_price method.")
                
                if missing.get("exits_without_prices", 0) > 0:
                    recommendations.append(f"{missing['exits_without_prices']} exits are missing prices. Verify exit price extraction from trade records.")
            
            # Check if using both trades and orders
            exit_analysis = cross_ref["cross_reference"].get("exit_analysis", {})
            if exit_analysis.get("difference", 0) > 10:  # Significant excess exits
                recommendations.append("Large number of excess exit signals suggests double-counting. Set extract_from_trades_only=True in SignalConfig.")
            
        except Exception as e:
            recommendations.append(f"Failed to generate recommendations: {e}")
        
        return recommendations
    
    def generate_detailed_report(self, signals: Any = None) -> str:
        """Generate a comprehensive validation report."""
        report_lines = [
            "="*80,
            "SIGNAL VALIDATION REPORT",
            "="*80,
            ""
        ]
        
        # CSV Analysis
        if self.trades_df is not None:
            csv_analysis = self.analyze_csv_trades()
            report_lines.extend([
                "CSV TRADE ANALYSIS:",
                f"  Total Trades: {csv_analysis['total_trades']}",
                f"  Columns: {', '.join(csv_analysis['columns'])}",
                ""
            ])
            
            if "direction_analysis" in csv_analysis:
                dir_analysis = csv_analysis["direction_analysis"]
                report_lines.extend([
                    "  Direction Analysis:",
                    f"    Long Trades: {dir_analysis['long_trades']}",
                    f"    Short Trades: {dir_analysis['short_trades']}",
                    f"    Unknown Direction: {dir_analysis['unknown']}",
                    ""
                ])
            
            if "pnl_analysis" in csv_analysis:
                pnl_analysis = csv_analysis["pnl_analysis"]
                report_lines.extend([
                    "  PnL Analysis:",
                    f"    Total PnL: {pnl_analysis['total_pnl']:.2f}",
                    f"    Winning Trades: {pnl_analysis['winning_trades']}",
                    f"    Losing Trades: {pnl_analysis['losing_trades']}",
                    f"    Win Rate: {pnl_analysis['winning_trades'] / (pnl_analysis['winning_trades'] + pnl_analysis['losing_trades']) * 100:.1f}%",
                    ""
                ])
        
        # Signal Analysis
        if signals is not None:
            if UNIFIED_INTERFACE_AVAILABLE:
                validation_result = self.validate_signals_format(signals)
                
                report_lines.extend([
                    "SIGNAL VALIDATION:",
                    f"  Valid: {'✓' if validation_result.is_valid else '✗'}",
                    ""
                ])
                
                if validation_result.errors:
                    report_lines.extend([
                        "  Errors:",
                        *[f"    - {error}" for error in validation_result.errors],
                        ""
                    ])
                
                if validation_result.warnings:
                    report_lines.extend([
                        "  Warnings:",
                        *[f"    - {warning}" for warning in validation_result.warnings],
                        ""
                    ])
                
                # Signal summary
                if hasattr(signals, 'get_summary'):
                    summary = signals.get_summary()
                    report_lines.extend([
                        "SIGNAL SUMMARY:",
                        f"  Long Entries: {summary['long_entries_count']}",
                        f"  Short Entries: {summary['short_entries_count']}",
                        f"  Long Exits: {summary['long_exits_count']}",
                        f"  Short Exits: {summary['short_exits_count']}",
                        f"  Total Entries: {summary['total_entries']}",
                        f"  Total Exits: {summary['total_exits']}",
                        f"  Entries with Prices: {summary['entries_with_prices']}",
                        f"  Exits with Prices: {summary['exits_with_prices']}",
                        ""
                    ])
                
                # Cross-reference analysis
                if self.trades_df is not None:
                    cross_ref = self.cross_reference_with_csv(signals)
                    if "cross_reference" in cross_ref:
                        cr = cross_ref["cross_reference"]
                        
                        report_lines.extend([
                            "CROSS-REFERENCE ANALYSIS:",
                            ""
                        ])
                        
                        if "entry_analysis" in cr:
                            ea = cr["entry_analysis"]
                            report_lines.extend([
                                "  Entry Analysis:",
                                f"    Signal Entries: {ea['signal_entries']}",
                                f"    CSV Trades: {ea['csv_trades']}",
                                f"    Difference: {ea['difference']}",
                                f"    Match Ratio: {ea['match_ratio']:.2f}",
                                ""
                            ])
                        
                        if "exit_analysis" in cr:
                            ea = cr["exit_analysis"]
                            report_lines.extend([
                                "  Exit Analysis:",
                                f"    Signal Exits: {ea['signal_exits']}",
                                f"    CSV Exits: {ea['csv_exits']}",
                                f"    Difference: {ea['difference']}",
                                f"    Match Ratio: {ea['match_ratio']:.2f}",
                                ""
                            ])
                        
                        if "timing_analysis" in cr:
                            ta = cr["timing_analysis"]
                            report_lines.extend([
                                "  Timing Analysis:",
                                f"    Entry Matches: {ta['entry_timing_matches']}",
                                f"    Entry Mismatches: {ta['entry_timing_mismatches']}",
                                f"    Exit Matches: {ta['exit_timing_matches']}",
                                f"    Exit Mismatches: {ta['exit_timing_mismatches']}",
                                ""
                            ])
                        
                        if "missing_data" in cr:
                            md = cr["missing_data"]
                            report_lines.extend([
                                "  Missing Data:",
                                f"    Entries without Prices: {md['entries_without_prices']}",
                                f"    Exits without Prices: {md['exits_without_prices']}",
                                ""
                            ])
                    
                    # Recommendations
                    if "recommendations" in cross_ref and cross_ref["recommendations"]:
                        report_lines.extend([
                            "RECOMMENDATIONS:",
                            *[f"  - {rec}" for rec in cross_ref["recommendations"]],
                            ""
                        ])
        
        report_lines.extend([
            "="*80,
            "END OF REPORT",
            "="*80
        ])
        
        return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(description="Signal Validation Tool")
    parser.add_argument("--csv", type=str, help="Path to CSV trade file")
    parser.add_argument("--validate-signals", action="store_true", help="Run signal validation")
    parser.add_argument("--output", type=str, help="Output file for report")
    
    args = parser.parse_args()
    
    # Initialize validation tool
    validator = SignalValidationTool(csv_file_path=args.csv)
    
    if args.csv:
        print(f"Analyzing CSV file: {args.csv}")
        csv_analysis = validator.analyze_csv_trades()
        print(f"Found {csv_analysis['total_trades']} trades")
        
        # Generate report
        report = validator.generate_detailed_report()
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Report saved to {args.output}")
        else:
            print(report)
    
    if args.validate_signals:
        print("Signal validation functionality available when used as module")


if __name__ == "__main__":
    main() 