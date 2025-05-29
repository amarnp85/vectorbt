"""
Signal Utilities Module

Provides centralized signal validation, preparation, and processing utilities
to eliminate duplication across the backtesting system. This module serves as
the single source of truth for signal handling operations.

Key Features:
- Unified signal validation and preparation
- Signal name mapping for backward compatibility
- Comprehensive signal quality checks
- VectorBTPro-optimized signal processing
- Type validation and error handling
- Performance-optimized operations using vectorized functions

Usage:
    from backtester.signals.signal_utils import SignalPreparator

    preparator = SignalPreparator()
    prepared_signals = preparator.prepare_signals(raw_signals, data_index)
    validation_result = preparator.validate_signals(prepared_signals)
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Enumeration of signal types."""

    LONG_ENTRIES = "long_entries"
    LONG_EXITS = "long_exits"
    SHORT_ENTRIES = "short_entries"
    SHORT_EXITS = "short_exits"
    SL_LEVELS = "sl_levels"
    TP_LEVELS = "tp_levels"


@dataclass
class SignalValidationResult:
    """Result of signal validation with detailed diagnostics."""

    is_valid: bool
    warnings: List[str]
    errors: List[str]
    recommendations: List[str]
    stats: Dict[str, Any]

    def has_issues(self) -> bool:
        """Check if there are any warnings or errors."""
        return len(self.warnings) > 0 or len(self.errors) > 0


class SignalPreparator:
    """
    Centralized signal preparation and validation utility.

    This class provides a unified interface for preparing, validating, and
    processing trading signals across the backtesting system.
    """

    # Standard signal name mapping for backward compatibility
    SIGNAL_MAPPING = {
        "entries": "long_entries",
        "exits": "long_exits",
        "sl_stop": "sl_levels",
        "tp_stop": "tp_levels",
        # Add more mappings as needed
    }

    # Required signal names for VectorBTPro
    REQUIRED_SIGNALS = ["long_entries", "long_exits", "short_entries", "short_exits"]

    # Optional signal names
    OPTIONAL_SIGNALS = ["sl_levels", "tp_levels"]

    def __init__(self, strict_validation: bool = False):
        """
        Initialize the signal preparator.

        Args:
            strict_validation: Whether to use strict validation (raises errors)
        """
        self.strict_validation = strict_validation

    def prepare_signals(
        self, signals: Dict[str, Any], data_index: pd.Index, fill_missing: bool = True
    ) -> Dict[str, Any]:
        """
        Prepare and validate signals for VectorBTPro simulation.

        Args:
            signals: Dictionary containing raw trading signals
            data_index: Index to use for creating missing signals
            fill_missing: Whether to create empty signals for missing required signals

        Returns:
            Dictionary of prepared and validated signals

        Raises:
            ValueError: If strict_validation is True and validation fails
        """
        logger.debug("Preparing signals for portfolio simulation")

        # Step 1: Apply signal name mapping
        mapped_signals = self._apply_signal_mapping(signals)

        # Step 2: Validate and convert signal types
        validated_signals = self._validate_signal_types(mapped_signals, data_index)

        # Step 3: Fill missing required signals if requested
        if fill_missing:
            validated_signals = self._fill_missing_signals(
                validated_signals, data_index
            )

        # Step 4: Validate signal alignment and consistency
        validation_result = self.validate_signals(validated_signals)

        if not validation_result.is_valid and self.strict_validation:
            error_msg = f"Signal validation failed: {validation_result.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log warnings if any
        for warning in validation_result.warnings:
            logger.warning(f"Signal preparation warning: {warning}")

        logger.debug("Signal preparation completed successfully")
        return validated_signals

    def validate_signals(self, signals: Dict[str, Any]) -> SignalValidationResult:
        """
        Comprehensive signal validation with detailed diagnostics.

        Args:
            signals: Dictionary of signals to validate

        Returns:
            SignalValidationResult with validation details
        """
        warnings = []
        errors = []
        recommendations = []

        # Check for required signals
        missing_required = []
        for signal_name in self.REQUIRED_SIGNALS:
            if signal_name not in signals:
                missing_required.append(signal_name)

        if missing_required:
            errors.append(f"Missing required signals: {missing_required}")

        # Validate signal types and data
        for signal_name, signal_data in signals.items():
            if signal_data is None:
                warnings.append(f"Signal '{signal_name}' is None")
                continue

            # Check if it's a pandas Series
            if not isinstance(signal_data, pd.Series):
                errors.append(
                    f"Signal '{signal_name}' must be a pandas Series, got {type(signal_data)}"
                )
                continue

            # For entry/exit signals, check they are boolean
            if signal_name in [
                "long_entries",
                "long_exits",
                "short_entries",
                "short_exits",
            ]:
                if signal_data.dtype != bool:
                    warnings.append(
                        f"Signal '{signal_name}' is not boolean type, should be converted"
                    )

                # Check for reasonable signal frequency
                signal_rate = signal_data.sum() / len(signal_data)
                if signal_rate > 0.2:  # More than 20% of bars have signals
                    warnings.append(
                        f"High signal frequency for '{signal_name}': {signal_rate:.2%}"
                    )
                    recommendations.append(
                        f"Consider stricter conditions for '{signal_name}'"
                    )
                elif signal_rate < 0.001:  # Less than 0.1% of bars have signals
                    warnings.append(
                        f"Very low signal frequency for '{signal_name}': {signal_rate:.2%}"
                    )
                    recommendations.append(
                        f"Consider relaxing conditions for '{signal_name}'"
                    )

            # For stop levels, check they are numeric
            elif signal_name in ["sl_levels", "tp_levels"]:
                if not pd.api.types.is_numeric_dtype(signal_data):
                    errors.append(f"Stop level '{signal_name}' must be numeric")

                # Check for reasonable number of non-NaN values
                non_nan_count = signal_data.notna().sum()
                if non_nan_count == 0:
                    warnings.append(f"Stop level '{signal_name}' has no valid values")

        # Check signal alignment (all signals should have same index)
        if len(signals) > 1:
            signal_items = list(signals.items())
            base_signal_name, base_signal = signal_items[0]

            if isinstance(base_signal, pd.Series):
                base_index = base_signal.index

                for signal_name, signal_data in signal_items[1:]:
                    if isinstance(signal_data, pd.Series):
                        if not signal_data.index.equals(base_index):
                            errors.append(
                                f"Signal '{signal_name}' has misaligned index with '{base_signal_name}'"
                            )

        # Calculate signal statistics
        stats = self._calculate_signal_stats(signals)

        # Check for signal balance
        if "long_entries_count" in stats and "short_entries_count" in stats:
            total_entries = stats["long_entries_count"] + stats["short_entries_count"]
            if total_entries > 0:
                long_ratio = stats["long_entries_count"] / total_entries
                if long_ratio > 0.9:
                    warnings.append("Heavily biased towards long signals")
                    recommendations.append("Consider adding short signal conditions")
                elif long_ratio < 0.1:
                    warnings.append("Heavily biased towards short signals")
                    recommendations.append("Consider adding long signal conditions")

        # Determine overall validity
        is_valid = len(errors) == 0

        return SignalValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            recommendations=recommendations,
            stats=stats,
        )

    def clean_signals(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean signals using VectorBTPro utilities to avoid conflicts.

        Args:
            signals: Dictionary of signals to clean

        Returns:
            Dictionary of cleaned signals
        """
        logger.debug("Cleaning signals to avoid conflicts")

        cleaned_signals = signals.copy()

        try:
            # Clean opposing entry signals
            if "long_entries" in signals and "short_entries" in signals:
                long_entries = signals["long_entries"]
                short_entries = signals["short_entries"]

                if isinstance(long_entries, pd.Series) and isinstance(
                    short_entries, pd.Series
                ):
                    cleaned_long, cleaned_short = self._clean_opposing_signals(
                        long_entries, short_entries
                    )
                    cleaned_signals["long_entries"] = cleaned_long
                    cleaned_signals["short_entries"] = cleaned_short

            # Clean entry/exit pairs
            signal_pairs = [
                ("long_entries", "long_exits"),
                ("short_entries", "short_exits"),
            ]

            for entry_key, exit_key in signal_pairs:
                if entry_key in signals and exit_key in signals:
                    entries = signals[entry_key]
                    exits = signals[exit_key]

                    if isinstance(entries, pd.Series) and isinstance(exits, pd.Series):
                        cleaned_entries, cleaned_exits = self._clean_entry_exit_pairs(
                            entries, exits
                        )
                        cleaned_signals[entry_key] = cleaned_entries
                        cleaned_signals[exit_key] = cleaned_exits

            logger.debug("Signal cleaning completed successfully")

        except Exception as e:
            logger.warning(f"Signal cleaning failed: {e}, using original signals")

        return cleaned_signals

    def convert_signal_result_to_dict(self, signal_result) -> Dict[str, Any]:
        """
        Convert a SignalResult object to a dictionary format.

        Args:
            signal_result: SignalResult object from signal_engine

        Returns:
            Dictionary of signals compatible with portfolio simulation
        """
        if not hasattr(signal_result, "long_entries"):
            raise ValueError("Invalid signal result object")

        signals = {
            "long_entries": signal_result.long_entries,
            "long_exits": signal_result.long_exits,
            "short_entries": signal_result.short_entries,
            "short_exits": signal_result.short_exits,
        }

        # Add optional signals if they exist
        if hasattr(signal_result, "sl_levels") and signal_result.sl_levels is not None:
            signals["sl_levels"] = signal_result.sl_levels

        if hasattr(signal_result, "tp_levels") and signal_result.tp_levels is not None:
            signals["tp_levels"] = signal_result.tp_levels

        return signals

    def _apply_signal_mapping(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Apply signal name mapping for backward compatibility."""
        mapped_signals = {}

        for key, value in signals.items():
            mapped_key = self.SIGNAL_MAPPING.get(key, key)
            mapped_signals[mapped_key] = value

        return mapped_signals

    def _validate_signal_types(
        self, signals: Dict[str, Any], data_index: pd.Index
    ) -> Dict[str, Any]:
        """Validate and convert signal types."""
        validated_signals = {}

        for signal_name, signal_data in signals.items():
            if signal_data is None:
                validated_signals[signal_name] = None
                continue

            # Convert to pandas Series if needed
            if not isinstance(signal_data, pd.Series):
                if isinstance(signal_data, (list, np.ndarray)):
                    signal_data = pd.Series(signal_data, index=data_index)
                else:
                    logger.warning(
                        f"Cannot convert signal '{signal_name}' of type {type(signal_data)} to Series"
                    )
                    validated_signals[signal_name] = signal_data
                    continue

            # For entry/exit signals, ensure boolean type
            if signal_name in [
                "long_entries",
                "long_exits",
                "short_entries",
                "short_exits",
            ]:
                if signal_data.dtype != bool:
                    try:
                        signal_data = signal_data.astype(bool)
                        logger.debug(
                            f"Converted signal '{signal_name}' to boolean type"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to convert signal '{signal_name}' to boolean: {e}"
                        )

            # For stop levels, ensure numeric type
            elif signal_name in ["sl_levels", "tp_levels"]:
                if not pd.api.types.is_numeric_dtype(signal_data):
                    try:
                        signal_data = pd.to_numeric(signal_data, errors="coerce")
                        logger.debug(
                            f"Converted signal '{signal_name}' to numeric type"
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to convert signal '{signal_name}' to numeric: {e}"
                        )

            validated_signals[signal_name] = signal_data

        return validated_signals

    def _fill_missing_signals(
        self, signals: Dict[str, Any], data_index: pd.Index
    ) -> Dict[str, Any]:
        """Fill missing required signals with empty signals."""
        filled_signals = signals.copy()

        for signal_name in self.REQUIRED_SIGNALS:
            if signal_name not in filled_signals or filled_signals[signal_name] is None:
                logger.warning(f"Missing signal: {signal_name}, creating empty signal")
                filled_signals[signal_name] = pd.Series(False, index=data_index)

        return filled_signals

    def _calculate_signal_stats(self, signals: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive signal statistics."""
        stats = {}

        for signal_name, signal_data in signals.items():
            if signal_data is None or not isinstance(signal_data, pd.Series):
                continue

            if signal_name in [
                "long_entries",
                "long_exits",
                "short_entries",
                "short_exits",
            ]:
                # Boolean signal statistics
                signal_count = signal_data.sum() if signal_data.dtype == bool else 0
                signal_rate = (
                    signal_count / len(signal_data) if len(signal_data) > 0 else 0
                )

                stats[f"{signal_name}_count"] = signal_count
                stats[f"{signal_name}_rate"] = signal_rate

            elif signal_name in ["sl_levels", "tp_levels"]:
                # Numeric signal statistics
                valid_count = signal_data.notna().sum()
                stats[f"{signal_name}_valid_count"] = valid_count

                if valid_count > 0:
                    stats[f"{signal_name}_mean"] = signal_data.mean()
                    stats[f"{signal_name}_std"] = signal_data.std()

        # Calculate derived statistics
        if "long_entries_count" in stats and "short_entries_count" in stats:
            total_entries = stats["long_entries_count"] + stats["short_entries_count"]
            stats["total_entry_signals"] = total_entries

            if total_entries > 0:
                stats["long_bias"] = stats["long_entries_count"] / total_entries
                stats["short_bias"] = stats["short_entries_count"] / total_entries

        return stats

    def _clean_opposing_signals(
        self, long_signals: pd.Series, short_signals: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Clean opposing signals using VectorBTPro utilities."""
        try:
            # Use VBT's signal cleaning for opposing signals
            cleaned_long, cleaned_short = long_signals.vbt.signals.clean(short_signals)
            return cleaned_long, cleaned_short
        except Exception as e:
            logger.warning(f"VBT signal cleaning failed: {e}, using original signals")
            return long_signals, short_signals

    def _clean_entry_exit_pairs(
        self, entries: pd.Series, exits: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """Clean entry/exit pairs using VectorBTPro utilities."""
        try:
            # Use VBT's signal cleaning for entry/exit pairs
            cleaned_entries, cleaned_exits = entries.vbt.signals.clean(exits)
            return cleaned_entries, cleaned_exits
        except Exception as e:
            logger.warning(
                f"VBT entry/exit cleaning failed: {e}, using original signals"
            )
            return entries, exits


# Convenience functions for easy access
def prepare_signals(
    signals: Dict[str, Any],
    data_index: pd.Index,
    strict_validation: bool = False,
    fill_missing: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to prepare signals using default SignalPreparator.

    Args:
        signals: Dictionary containing raw trading signals
        data_index: Index to use for creating missing signals
        strict_validation: Whether to use strict validation
        fill_missing: Whether to create empty signals for missing required signals

    Returns:
        Dictionary of prepared and validated signals
    """
    preparator = SignalPreparator(strict_validation=strict_validation)
    return preparator.prepare_signals(signals, data_index, fill_missing)


def validate_signals(signals: Dict[str, Any]) -> SignalValidationResult:
    """
    Convenience function to validate signals using default SignalPreparator.

    Args:
        signals: Dictionary of signals to validate

    Returns:
        SignalValidationResult with validation details
    """
    preparator = SignalPreparator()
    return preparator.validate_signals(signals)


def clean_signals(signals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to clean signals using default SignalPreparator.

    Args:
        signals: Dictionary of signals to clean

    Returns:
        Dictionary of cleaned signals
    """
    preparator = SignalPreparator()
    return preparator.clean_signals(signals)
