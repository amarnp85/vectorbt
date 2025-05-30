"""
Signal Validation and Quality Assessment

This module provides comprehensive validation and quality assessment for
trading signals to ensure they are consistent, realistic, and ready for
chart rendering and analysis.

Module Purpose:
==============
Signal validation catches common issues that can affect backtesting accuracy
and chart visualization:
- Overlapping or conflicting signals
- Missing price information
- Invalid signal sequences
- Timing inconsistencies
- Data quality issues

The validators provide both automated fixing and detailed reporting to help
users understand and resolve signal quality issues.

Key Features:
============
- Signal consistency validation
- Price data validation
- Timing validation
- Signal cleaning and conflict resolution
- Quality metrics and reporting
- Automated fixes with user notification

Integration Points:
==================
- Used by: SignalProcessor for signal quality assurance
- Input: SignalFormat objects from extractors
- Output: Validated signals and quality reports
- Logging: Detailed validation reports for debugging

Related Modules:
===============
- extractors.py: Provides signals for validation
- timing.py: Validates timing consistency
- ../signals/signal_interface.py: Uses SignalFormat and ValidationResult
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from backtester.signals.signal_interface import SignalFormat, ValidationResult
from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    severity: ValidationSeverity
    message: str
    count: int = 1
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    fix_applied: bool = False
    fix_description: str = ""


@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    quality_score: float = 1.0  # 0-1 scale
    signal_counts: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    def add_issue(self, issue: ValidationIssue):
        """Add an issue to the report."""
        self.issues.append(issue)
        
        # Update validity status
        if issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.is_valid = False
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get all issues of a specific severity."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the validation report."""
        severity_counts = {}
        for severity in ValidationSeverity:
            severity_counts[severity.value] = len(self.get_issues_by_severity(severity))
        
        return {
            "is_valid": self.is_valid,
            "quality_score": self.quality_score,
            "total_issues": len(self.issues),
            "severity_counts": severity_counts,
            "signal_counts": self.signal_counts,
            "has_recommendations": len(self.recommendations) > 0
        }


class SignalConsistencyValidator:
    """
    Validates signal consistency and identifies conflicts.
    
    This validator checks for logical inconsistencies in signals such as:
    - Overlapping entry signals (long and short at same time)
    - Missing exit signals for entries
    - Signal sequences that don't make trading sense
    - Price inconsistencies
    
    Usage:
        validator = SignalConsistencyValidator()
        issues = validator.validate(signals)
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize consistency validator.
        
        Args:
            strict_mode: If True, applies stricter validation rules
        """
        self.strict_mode = strict_mode
    
    def validate(self, signals: SignalFormat) -> List[ValidationIssue]:
        """
        Validate signal consistency.
        
        Args:
            signals: SignalFormat to validate
            
        Returns:
            List of validation issues found
        """
        issues = []
        
        # Check for overlapping entry signals
        issues.extend(self._check_overlapping_entries(signals))
        
        # Check signal sequences
        issues.extend(self._check_signal_sequences(signals))
        
        # Check price consistency
        issues.extend(self._check_price_consistency(signals))
        
        # Check risk level consistency
        issues.extend(self._check_risk_level_consistency(signals))
        
        return issues
    
    def _check_overlapping_entries(self, signals: SignalFormat) -> List[ValidationIssue]:
        """Check for overlapping long and short entry signals."""
        issues = []
        
        # Find overlapping entries
        overlaps = signals.long_entries & signals.short_entries
        if overlaps.any():
            overlap_times = signals.index[overlaps].tolist()
            
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Found overlapping long and short entry signals",
                count=len(overlap_times),
                timestamps=overlap_times
            )
            issues.append(issue)
        
        return issues
    
    def _check_signal_sequences(self, signals: SignalFormat) -> List[ValidationIssue]:
        """Check for logical signal sequences."""
        issues = []
        
        # Check for entries without corresponding exits
        total_long_entries = signals.long_entries.sum()
        total_long_exits = signals.long_exits.sum()
        
        if total_long_entries > 0 and total_long_exits == 0:
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Found {total_long_entries} long entries but no long exits",
                count=total_long_entries
            )
            issues.append(issue)
        
        total_short_entries = signals.short_entries.sum()
        total_short_exits = signals.short_exits.sum()
        
        if total_short_entries > 0 and total_short_exits == 0:
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Found {total_short_entries} short entries but no short exits",
                count=total_short_entries
            )
            issues.append(issue)
        
        # Check for significant imbalances
        if total_long_entries > 0 and total_long_exits > 0:
            entry_exit_ratio = total_long_entries / total_long_exits
            if entry_exit_ratio > 2 or entry_exit_ratio < 0.5:
                issue = ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Long entry/exit ratio seems unbalanced: {entry_exit_ratio:.2f}",
                    count=1
                )
                issues.append(issue)
        
        return issues
    
    def _check_price_consistency(self, signals: SignalFormat) -> List[ValidationIssue]:
        """Check for price data consistency."""
        issues = []
        
        # Check for entries without prices
        entries_mask = signals.long_entries | signals.short_entries
        entries_without_prices = entries_mask & signals.entry_prices.isna()
        
        if entries_without_prices.any():
            count = entries_without_prices.sum()
            timestamps = signals.index[entries_without_prices].tolist()
            
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Found {count} entry signals without prices",
                count=count,
                timestamps=timestamps[:10]  # Limit to first 10
            )
            issues.append(issue)
        
        # Check for exits without prices
        exits_mask = signals.long_exits | signals.short_exits
        exits_without_prices = exits_mask & signals.exit_prices.isna()
        
        if exits_without_prices.any():
            count = exits_without_prices.sum()
            timestamps = signals.index[exits_without_prices].tolist()
            
            issue = ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Found {count} exit signals without prices",
                count=count,
                timestamps=timestamps[:10]
            )
            issues.append(issue)
        
        # Check for invalid prices (negative or zero)
        invalid_entry_prices = (signals.entry_prices <= 0) & ~signals.entry_prices.isna()
        if invalid_entry_prices.any():
            issue = ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message="Found invalid entry prices (negative or zero)",
                count=invalid_entry_prices.sum()
            )
            issues.append(issue)
        
        return issues
    
    def _check_risk_level_consistency(self, signals: SignalFormat) -> List[ValidationIssue]:
        """Check risk level data consistency."""
        issues = []
        
        # Check for entries with stop levels but no corresponding signals
        has_sl_prices = ~signals.sl_price_levels.isna()
        has_tp_prices = ~signals.tp_price_levels.isna()
        has_entries = signals.long_entries | signals.short_entries
        
        # SL/TP levels without entries might indicate timing issues
        sl_without_entries = has_sl_prices & ~has_entries
        if sl_without_entries.any():
            issue = ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Found stop loss levels without corresponding entry signals",
                count=sl_without_entries.sum()
            )
            issues.append(issue)
        
        tp_without_entries = has_tp_prices & ~has_entries
        if tp_without_entries.any():
            issue = ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Found take profit levels without corresponding entry signals",
                count=tp_without_entries.sum()
            )
            issues.append(issue)
        
        return issues


class SignalQualityAnalyzer:
    """
    Analyzes signal quality and provides quality metrics.
    
    This analyzer provides quantitative measures of signal quality including:
    - Signal density and distribution
    - Price coverage
    - Risk level coverage
    - Timing consistency
    
    Usage:
        analyzer = SignalQualityAnalyzer()
        metrics = analyzer.analyze(signals)
    """
    
    def analyze(self, signals: SignalFormat) -> Dict[str, Any]:
        """
        Analyze signal quality and return metrics.
        
        Args:
            signals: SignalFormat to analyze
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Basic signal counts
        metrics["signal_counts"] = self._calculate_signal_counts(signals)
        
        # Signal density (signals per time period)
        metrics["signal_density"] = self._calculate_signal_density(signals)
        
        # Price coverage
        metrics["price_coverage"] = self._calculate_price_coverage(signals)
        
        # Risk level coverage
        metrics["risk_coverage"] = self._calculate_risk_coverage(signals)
        
        # Signal distribution
        metrics["signal_distribution"] = self._calculate_signal_distribution(signals)
        
        # Overall quality score
        metrics["quality_score"] = self._calculate_quality_score(metrics)
        
        return metrics
    
    def _calculate_signal_counts(self, signals: SignalFormat) -> Dict[str, int]:
        """Calculate basic signal counts."""
        return {
            "long_entries": signals.long_entries.sum(),
            "short_entries": signals.short_entries.sum(),
            "long_exits": signals.long_exits.sum(),
            "short_exits": signals.short_exits.sum(),
            "total_entries": (signals.long_entries | signals.short_entries).sum(),
            "total_exits": (signals.long_exits | signals.short_exits).sum(),
            "entries_with_prices": ((signals.long_entries | signals.short_entries) & ~signals.entry_prices.isna()).sum(),
            "exits_with_prices": ((signals.long_exits | signals.short_exits) & ~signals.exit_prices.isna()).sum(),
            "sl_levels": (~signals.sl_price_levels.isna()).sum(),
            "tp_levels": (~signals.tp_price_levels.isna()).sum()
        }
    
    def _calculate_signal_density(self, signals: SignalFormat) -> Dict[str, float]:
        """Calculate signal density metrics."""
        total_periods = len(signals.index)
        total_signals = (signals.long_entries | signals.short_entries | 
                        signals.long_exits | signals.short_exits).sum()
        
        return {
            "signals_per_period": total_signals / total_periods if total_periods > 0 else 0,
            "entry_density": (signals.long_entries | signals.short_entries).sum() / total_periods if total_periods > 0 else 0,
            "exit_density": (signals.long_exits | signals.short_exits).sum() / total_periods if total_periods > 0 else 0
        }
    
    def _calculate_price_coverage(self, signals: SignalFormat) -> Dict[str, float]:
        """Calculate price data coverage."""
        entry_signals = signals.long_entries | signals.short_entries
        exit_signals = signals.long_exits | signals.short_exits
        
        entry_price_coverage = 0
        if entry_signals.any():
            entry_price_coverage = (~signals.entry_prices[entry_signals].isna()).sum() / entry_signals.sum()
        
        exit_price_coverage = 0
        if exit_signals.any():
            exit_price_coverage = (~signals.exit_prices[exit_signals].isna()).sum() / exit_signals.sum()
        
        return {
            "entry_price_coverage": entry_price_coverage,
            "exit_price_coverage": exit_price_coverage,
            "overall_price_coverage": (entry_price_coverage + exit_price_coverage) / 2
        }
    
    def _calculate_risk_coverage(self, signals: SignalFormat) -> Dict[str, float]:
        """Calculate risk level coverage."""
        entry_signals = signals.long_entries | signals.short_entries
        
        sl_coverage = 0
        tp_coverage = 0
        
        if entry_signals.any():
            sl_coverage = (~signals.sl_price_levels[entry_signals].isna()).sum() / entry_signals.sum()
            tp_coverage = (~signals.tp_price_levels[entry_signals].isna()).sum() / entry_signals.sum()
        
        return {
            "sl_coverage": sl_coverage,
            "tp_coverage": tp_coverage,
            "risk_level_coverage": (sl_coverage + tp_coverage) / 2
        }
    
    def _calculate_signal_distribution(self, signals: SignalFormat) -> Dict[str, Any]:
        """Calculate signal distribution across time."""
        # This could be enhanced with more sophisticated temporal analysis
        return {
            "long_short_ratio": self._calculate_long_short_ratio(signals),
            "entry_exit_balance": self._calculate_entry_exit_balance(signals)
        }
    
    def _calculate_long_short_ratio(self, signals: SignalFormat) -> float:
        """Calculate ratio of long to short signals."""
        long_count = signals.long_entries.sum()
        short_count = signals.short_entries.sum()
        
        if short_count == 0:
            return float('inf') if long_count > 0 else 0
        
        return long_count / short_count
    
    def _calculate_entry_exit_balance(self, signals: SignalFormat) -> float:
        """Calculate balance between entries and exits."""
        total_entries = (signals.long_entries | signals.short_entries).sum()
        total_exits = (signals.long_exits | signals.short_exits).sum()
        
        if total_exits == 0:
            return float('inf') if total_entries > 0 else 0
        
        return total_entries / total_exits
    
    def _calculate_quality_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-1)."""
        # This is a simplified quality score calculation
        # Can be enhanced with more sophisticated scoring
        
        price_coverage = metrics["price_coverage"]["overall_price_coverage"]
        risk_coverage = metrics["risk_coverage"]["risk_level_coverage"]
        
        # Balance between entries and exits
        entry_exit_balance = metrics["signal_distribution"]["entry_exit_balance"]
        balance_score = 1.0 / (1.0 + abs(entry_exit_balance - 1.0))
        
        # Weighted average
        quality_score = (
            price_coverage * 0.4 +
            risk_coverage * 0.3 +
            balance_score * 0.3
        )
        
        return min(1.0, max(0.0, quality_score))


class SignalCleaner:
    """
    Cleans and fixes common signal issues.
    
    This class provides automated fixes for common signal problems:
    - Removing overlapping signals
    - Filling missing prices
    - Balancing entry/exit signals
    - Cleaning invalid data
    
    Usage:
        cleaner = SignalCleaner()
        cleaned_signals, fixes = cleaner.clean(signals)
    """
    
    def __init__(self, aggressive_cleaning: bool = False):
        """
        Initialize signal cleaner.
        
        Args:
            aggressive_cleaning: If True, applies more aggressive cleaning
        """
        self.aggressive_cleaning = aggressive_cleaning
    
    def clean(self, signals: SignalFormat) -> Tuple[SignalFormat, List[ValidationIssue]]:
        """
        Clean signals and return cleaned version with applied fixes.
        
        Args:
            signals: SignalFormat to clean
            
        Returns:
            Tuple of (cleaned_signals, list_of_fixes_applied)
        """
        cleaned_signals = signals.copy()
        fixes_applied = []
        
        # Fix overlapping entry signals
        fixes_applied.extend(self._fix_overlapping_entries(cleaned_signals))
        
        # Clean invalid prices
        fixes_applied.extend(self._clean_invalid_prices(cleaned_signals))
        
        # Balance entry/exit signals if requested
        if self.aggressive_cleaning:
            fixes_applied.extend(self._balance_entry_exit_signals(cleaned_signals))
        
        return cleaned_signals, fixes_applied
    
    def _fix_overlapping_entries(self, signals: SignalFormat) -> List[ValidationIssue]:
        """Fix overlapping entry signals by prioritizing long entries."""
        fixes = []
        
        overlaps = signals.long_entries & signals.short_entries
        if overlaps.any():
            # Prioritize long entries, remove short entries at overlap points
            signals.short_entries.loc[overlaps] = False
            
            fix = ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Fixed overlapping entry signals by prioritizing long entries",
                count=overlaps.sum(),
                fix_applied=True,
                fix_description="Removed short entry signals where they overlapped with long entries"
            )
            fixes.append(fix)
        
        return fixes
    
    def _clean_invalid_prices(self, signals: SignalFormat) -> List[ValidationIssue]:
        """Clean invalid price data."""
        fixes = []
        
        # Remove negative or zero prices
        invalid_entry_mask = (signals.entry_prices <= 0) & ~signals.entry_prices.isna()
        if invalid_entry_mask.any():
            signals.entry_prices.loc[invalid_entry_mask] = np.nan
            
            fix = ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Cleaned invalid entry prices (set negative/zero prices to NaN)",
                count=invalid_entry_mask.sum(),
                fix_applied=True,
                fix_description="Set invalid entry prices to NaN"
            )
            fixes.append(fix)
        
        invalid_exit_mask = (signals.exit_prices <= 0) & ~signals.exit_prices.isna()
        if invalid_exit_mask.any():
            signals.exit_prices.loc[invalid_exit_mask] = np.nan
            
            fix = ValidationIssue(
                severity=ValidationSeverity.INFO,
                message="Cleaned invalid exit prices (set negative/zero prices to NaN)",
                count=invalid_exit_mask.sum(),
                fix_applied=True,
                fix_description="Set invalid exit prices to NaN"
            )
            fixes.append(fix)
        
        return fixes
    
    def _balance_entry_exit_signals(self, signals: SignalFormat) -> List[ValidationIssue]:
        """Balance entry and exit signals (aggressive cleaning)."""
        fixes = []
        
        # This is a placeholder for more sophisticated signal balancing
        # In practice, this would require careful analysis of the strategy logic
        
        return fixes


class ComprehensiveSignalValidator:
    """
    Main signal validator that coordinates all validation components.
    
    This is the primary interface for signal validation, combining
    consistency checking, quality analysis, and optional cleaning.
    
    Usage:
        validator = ComprehensiveSignalValidator()
        report = validator.validate_signals(signals)
    """
    
    def __init__(self, strict_mode: bool = False, auto_clean: bool = False):
        """
        Initialize comprehensive validator.
        
        Args:
            strict_mode: Apply stricter validation rules
            auto_clean: Automatically apply fixes for common issues
        """
        self.strict_mode = strict_mode
        self.auto_clean = auto_clean
        
        # Initialize components
        self.consistency_validator = SignalConsistencyValidator(strict_mode)
        self.quality_analyzer = SignalQualityAnalyzer()
        self.cleaner = SignalCleaner(aggressive_cleaning=strict_mode)
    
    def validate_signals(self, signals: SignalFormat) -> ValidationReport:
        """
        Perform comprehensive signal validation.
        
        Args:
            signals: SignalFormat to validate
            
        Returns:
            Comprehensive validation report
        """
        logger.debug("Starting comprehensive signal validation")
        
        # Create validation report
        report = ValidationReport(is_valid=True)
        
        # Run consistency validation
        consistency_issues = self.consistency_validator.validate(signals)
        for issue in consistency_issues:
            report.add_issue(issue)
        
        # Run quality analysis
        quality_metrics = self.quality_analyzer.analyze(signals)
        report.signal_counts = quality_metrics["signal_counts"]
        report.quality_score = quality_metrics["quality_score"]
        
        # Apply cleaning if requested
        if self.auto_clean:
            cleaned_signals, fixes_applied = self.cleaner.clean(signals)
            for fix in fixes_applied:
                report.add_issue(fix)
            
            # Re-analyze quality after cleaning
            post_clean_metrics = self.quality_analyzer.analyze(cleaned_signals)
            report.quality_score = post_clean_metrics["quality_score"]
        
        # Generate recommendations
        report.recommendations = self._generate_recommendations(report, quality_metrics)
        
        logger.info(f"Signal validation complete: {report.get_summary()}")
        return report
    
    def _generate_recommendations(
        self,
        report: ValidationReport,
        quality_metrics: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Price coverage recommendations
        price_coverage = quality_metrics["price_coverage"]["overall_price_coverage"]
        if price_coverage < 0.8:
            recommendations.append(
                f"Price coverage is low ({price_coverage:.1%}). Consider improving "
                "price data extraction or using fallback pricing methods."
            )
        
        # Risk level recommendations
        risk_coverage = quality_metrics["risk_coverage"]["risk_level_coverage"]
        if risk_coverage < 0.5:
            recommendations.append(
                f"Risk level coverage is low ({risk_coverage:.1%}). Consider adding "
                "stop loss and take profit levels to your strategy signals."
            )
        
        # Signal balance recommendations
        entry_exit_balance = quality_metrics["signal_distribution"]["entry_exit_balance"]
        if entry_exit_balance > 2 or entry_exit_balance < 0.5:
            recommendations.append(
                f"Entry/exit signal balance seems off (ratio: {entry_exit_balance:.2f}). "
                "Check if exits are being generated properly."
            )
        
        # Error-specific recommendations
        error_issues = report.get_issues_by_severity(ValidationSeverity.ERROR)
        if error_issues:
            recommendations.append(
                "Critical signal errors found. Review signal generation logic "
                "and consider enabling auto-cleaning."
            )
        
        return recommendations


# Module exports
__all__ = [
    'ValidationSeverity',
    'ValidationIssue',
    'ValidationReport',
    'SignalConsistencyValidator',
    'SignalQualityAnalyzer',
    'SignalCleaner',
    'ComprehensiveSignalValidator'
]