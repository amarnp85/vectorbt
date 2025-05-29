"""
Signal Engine Module

This module provides the core signal generation logic for trading strategies.
It serves as an intermediary layer between indicator outputs and strategy implementations,
focusing purely on signal logic without strategy-specific orchestration.

ARCHITECTURAL ROLE:
- Signal Engine: Pure, stateless signal generation functions (THIS MODULE)
- Strategy Module: Orchestrates indicators, signals, and portfolio simulation
- Indicator Module: Provides technical indicator calculations

Key Distinctions from Strategy Module:
- Signal Engine: Pure signal generation logic, stateless functions
- Strategy Module: Orchestrates indicators, signals, and portfolio simulation

Key Features:
- VectorBTPro-native signal generation using .vbt.crossed_above/below
- Proper signal cleaning using vbt.signals utilities
- Support for complex multi-condition signals
- ATR-based dynamic risk management levels
- Comprehensive signal validation and preprocessing
- Modular design for easy testing and reuse

Usage:
    from backtester.signals.signal_engine import generate_dma_atr_trend_signals

    signals = generate_dma_atr_trend_signals(
        close=close_prices,
        short_ma=short_ma_values,
        long_ma=long_ma_values,
        trend_filter=trend_confirmation,
        atr=atr_values,
        sl_multiplier=2.0,
        tp_multiplier=4.0
    )
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SignalResult:
    """Container for signal generation results with metadata."""

    def __init__(
        self,
        long_entries: pd.Series,
        long_exits: pd.Series,
        short_entries: pd.Series,
        short_exits: pd.Series,
        sl_levels: Optional[pd.Series] = None,
        tp_levels: Optional[pd.Series] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize signal result container.

        Args:
            long_entries: Boolean series for long entry signals
            long_exits: Boolean series for long exit signals
            short_entries: Boolean series for short entry signals
            short_exits: Boolean series for short exit signals
            sl_levels: Stop-loss levels (price distance from entry)
            tp_levels: Take-profit levels (price distance from entry)
            metadata: Additional signal generation metadata
        """
        self.long_entries = long_entries
        self.long_exits = long_exits
        self.short_entries = short_entries
        self.short_exits = short_exits
        self.sl_levels = sl_levels
        self.tp_levels = tp_levels
        self.metadata = metadata or {}

    def get_signal_stats(self) -> Dict[str, Any]:
        """Get basic statistics about generated signals."""
        return {
            "long_entries_count": self.long_entries.sum(),
            "long_exits_count": self.long_exits.sum(),
            "short_entries_count": self.short_entries.sum(),
            "short_exits_count": self.short_exits.sum(),
            "total_signals": (
                self.long_entries.sum()
                + self.long_exits.sum()
                + self.short_entries.sum()
                + self.short_exits.sum()
            ),
            "signal_rate": (
                (self.long_entries.sum() + self.short_entries.sum())
                / len(self.long_entries)
            ),
            "has_risk_levels": self.sl_levels is not None
            and self.tp_levels is not None,
        }


def generate_ma_crossover_signals(
    short_ma: pd.Series,
    long_ma: pd.Series,
    clean_signals: bool = True,
    wait_confirmation: int = 1,
) -> SignalResult:
    """
    Generate pure moving average crossover signals.
    
    This is a fundamental, reusable signal pattern that can be used
    by any strategy that needs MA crossover detection.

    Args:
        short_ma: Short-term moving average
        long_ma: Long-term moving average
        clean_signals: Whether to clean conflicting signals
        wait_confirmation: Bars to wait for crossover confirmation

    Returns:
        SignalResult object containing basic crossover signals
    """
    logger.debug("Generating MA crossover signals")

    # Validate inputs
    _validate_signal_inputs(short_ma, long_ma)

    # Generate crossover signals using VBT
    long_entries = short_ma.vbt.crossed_above(long_ma, wait=wait_confirmation)
    short_entries = short_ma.vbt.crossed_below(long_ma, wait=wait_confirmation)

    # Exit signals (reverse crossover)
    long_exits = short_ma.vbt.crossed_below(long_ma, wait=wait_confirmation)
    short_exits = short_ma.vbt.crossed_above(long_ma, wait=wait_confirmation)

    if clean_signals:
        # Clean signals using VBT utilities
        long_entries, short_entries = _clean_opposing_signals(
            long_entries, short_entries
        )
        long_entries, long_exits = _clean_entry_exit_pairs(long_entries, long_exits)
        short_entries, short_exits = _clean_entry_exit_pairs(short_entries, short_exits)

    metadata = {
        "strategy_type": "MA_Crossover",
        "wait_confirmation": wait_confirmation,
        "signals_cleaned": clean_signals,
    }

    return SignalResult(
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        metadata=metadata,
    )


def generate_atr_risk_levels(
    close: pd.Series,
    atr: pd.Series,
    entry_signals: pd.Series,
    sl_multiplier: float = 2.0,
    tp_multiplier: float = 4.0,
    signal_direction: str = "long",
) -> Tuple[pd.Series, pd.Series]:
    """
    Generate ATR-based stop-loss and take-profit levels.
    
    This is a reusable risk management function that can be used
    by any strategy that needs ATR-based risk levels.

    Args:
        close: Close price series
        atr: Average True Range values
        entry_signals: Boolean series indicating entry points
        sl_multiplier: Stop-loss multiplier for ATR
        tp_multiplier: Take-profit multiplier for ATR
        signal_direction: 'long' or 'short' for direction-specific calculations

    Returns:
        Tuple of (stop_loss_levels, take_profit_levels) as price levels
    """
    logger.debug(f"Generating ATR risk levels for {signal_direction} signals")

    # Initialize with NaN (no stops by default)
    sl_levels = pd.Series(np.nan, index=close.index)
    tp_levels = pd.Series(np.nan, index=close.index)

    # Get entry indices
    entry_indices = entry_signals[entry_signals].index
    
    if len(entry_indices) > 0:
        for idx in entry_indices:
            entry_price = close.loc[idx]
            atr_value = atr.loc[idx]
            
            if signal_direction == "long":
                sl_levels.loc[idx] = entry_price - (atr_value * sl_multiplier)
                tp_levels.loc[idx] = entry_price + (atr_value * tp_multiplier)
            elif signal_direction == "short":
                sl_levels.loc[idx] = entry_price + (atr_value * sl_multiplier)
                tp_levels.loc[idx] = entry_price - (atr_value * tp_multiplier)
            else:
                raise ValueError("signal_direction must be 'long' or 'short'")

    return sl_levels, tp_levels


def generate_trend_filter_signals(
    price: pd.Series,
    trend_indicator: pd.Series,
    filter_type: str = "above_below",
) -> pd.Series:
    """
    Generate trend confirmation filter signals.
    
    This is a reusable trend filter that can be used by any strategy
    that needs trend direction confirmation.

    Args:
        price: Price series (usually close)
        trend_indicator: Trend indicator (usually long-term MA)
        filter_type: 'above_below' for price vs indicator comparison

    Returns:
        Boolean series where True indicates uptrend (allows longs)
    """
    logger.debug(f"Generating trend filter signals using {filter_type}")

    if filter_type == "above_below":
        # Price above trend indicator = uptrend
        return price > trend_indicator
    else:
        raise ValueError("filter_type must be 'above_below'")


def generate_adx_filter_signals(
    adx: pd.Series,
    threshold: float = 25.0,
) -> pd.Series:
    """
    Generate ADX-based trend strength filter signals.
    
    This is a reusable trend strength filter that can be used by any
    strategy that needs trend strength confirmation.

    Args:
        adx: ADX indicator values
        threshold: Minimum ADX value for strong trend

    Returns:
        Boolean series where True indicates strong trend
    """
    logger.debug(f"Generating ADX filter signals with threshold {threshold}")
    
    return adx >= threshold


# Note: generate_simple_ma_crossover_signals has been replaced by the more generic generate_ma_crossover_signals


def generate_threshold_signals(
    close: pd.Series,
    indicator: pd.Series,
    upper_threshold: float,
    lower_threshold: float,
    signal_type: str = "mean_reversion",
    clean_signals: bool = True,
) -> SignalResult:
    """
    Generate signals based on indicator threshold crossovers.

    Args:
        close: Close price series
        indicator: Indicator values (e.g., RSI, Stochastic)
        upper_threshold: Upper threshold for signals
        lower_threshold: Lower threshold for signals
        signal_type: 'mean_reversion' or 'trend_following'
        clean_signals: Whether to clean conflicting signals

    Returns:
        SignalResult object containing threshold-based signals
    """
    logger.debug(f"Generating {signal_type} threshold signals")

    # Validate inputs
    if len(close) != len(indicator):
        raise ValueError("Close and indicator series must have same length")

    if signal_type == "mean_reversion":
        # Mean reversion: buy oversold, sell overbought
        long_entries = indicator.vbt.crossed_below(lower_threshold)
        long_exits = indicator.vbt.crossed_above(upper_threshold)
        short_entries = indicator.vbt.crossed_above(upper_threshold)
        short_exits = indicator.vbt.crossed_below(lower_threshold)
    elif signal_type == "trend_following":
        # Trend following: buy strength, sell weakness
        long_entries = indicator.vbt.crossed_above(upper_threshold)
        long_exits = indicator.vbt.crossed_below(lower_threshold)
        short_entries = indicator.vbt.crossed_below(lower_threshold)
        short_exits = indicator.vbt.crossed_above(upper_threshold)
    else:
        raise ValueError("signal_type must be 'mean_reversion' or 'trend_following'")

    if clean_signals:
        long_entries, short_entries = _clean_opposing_signals(
            long_entries, short_entries
        )
        long_entries, long_exits = _clean_entry_exit_pairs(long_entries, long_exits)
        short_entries, short_exits = _clean_entry_exit_pairs(short_entries, short_exits)

    metadata = {
        "strategy_type": f"Threshold_{signal_type}",
        "upper_threshold": upper_threshold,
        "lower_threshold": lower_threshold,
        "signals_cleaned": clean_signals,
    }

    return SignalResult(
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        metadata=metadata,
    )


def combine_signal_conditions(
    *conditions: pd.Series, operator: str = "and"
) -> pd.Series:
    """
    Combine multiple signal conditions using logical operators.

    Args:
        *conditions: Variable number of boolean Series to combine
        operator: 'and', 'or', or 'xor'

    Returns:
        Combined boolean Series
    """
    if not conditions:
        raise ValueError("At least one condition must be provided")

    result = conditions[0].copy()

    for condition in conditions[1:]:
        if operator == "and":
            result = result & condition
        elif operator == "or":
            result = result | condition
        elif operator == "xor":
            result = result ^ condition
        else:
            raise ValueError("operator must be 'and', 'or', or 'xor'")

    return result


def apply_signal_filters(
    signals: pd.Series,
    price: pd.Series,
    min_price_change: Optional[float] = None,
    max_signals_per_period: Optional[int] = None,
    period: str = "1D",
) -> pd.Series:
    """
    Apply additional filters to generated signals.

    Args:
        signals: Boolean signal series
        price: Price series for validation
        min_price_change: Minimum price change required for signal
        max_signals_per_period: Maximum signals allowed per period
        period: Time period for signal limiting

    Returns:
        Filtered signal series
    """
    filtered_signals = signals.copy()

    # Filter by minimum price change
    if min_price_change is not None:
        price_change = price.pct_change().abs()
        price_filter = price_change >= min_price_change
        filtered_signals = filtered_signals & price_filter

    # Limit signals per period
    if max_signals_per_period is not None:
        # Use VBT's signal utilities to limit signals per period
        signal_counts = filtered_signals.resample(period).sum()
        for period_start, count in signal_counts.items():
            if count > max_signals_per_period:
                period_mask = (filtered_signals.index >= period_start) & (
                    filtered_signals.index < period_start + pd.Timedelta(period)
                )
                period_signals = filtered_signals[period_mask]
                # Keep only first N signals in period
                signal_indices = period_signals[period_signals].index[
                    :max_signals_per_period
                ]
                filtered_signals[period_mask] = False
                filtered_signals[signal_indices] = True

    return filtered_signals


# --- Helper Functions ---


def _validate_signal_inputs(*series: pd.Series) -> None:
    """Validate that all input series have compatible indexes."""
    if not series:
        return

    base_index = series[0].index
    for i, s in enumerate(series[1:], 1):
        if not base_index.equals(s.index):
            raise ValueError(f"Series {i} has incompatible index with base series")


def _clean_opposing_signals(
    long_signals: pd.Series, short_signals: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """Clean opposing signals using VectorBTPro utilities."""
    try:
        # Use VBT's signal cleaning for opposing signals
        # For opposing signals, we want to ensure they don't happen simultaneously
        cleaned_long, cleaned_short = long_signals.vbt.signals.clean(short_signals)
        return cleaned_long, cleaned_short
    except Exception as e:
        logger.warning(f"Signal cleaning failed, using original signals: {e}")
        return long_signals, short_signals


def _clean_entry_exit_pairs(
    entries: pd.Series, exits: pd.Series
) -> Tuple[pd.Series, pd.Series]:
    """Clean entry/exit pairs using VectorBTPro utilities."""
    try:
        # Use VBT's signal cleaning for entry/exit pairs
        # This ensures proper entry->exit->entry sequence
        cleaned_entries, cleaned_exits = entries.vbt.signals.clean(exits)
        return cleaned_entries, cleaned_exits
    except Exception as e:
        logger.warning(f"Entry/exit cleaning failed, using original signals: {e}")
        return entries, exits


def validate_signal_quality(signal_result: SignalResult) -> Dict[str, Any]:
    """
    Validate signal quality and return diagnostic information.

    Args:
        signal_result: SignalResult object to validate

    Returns:
        Dictionary with validation results and recommendations
    """
    stats = signal_result.get_signal_stats()

    validation = {
        "is_valid": True,
        "warnings": [],
        "recommendations": [],
        "stats": stats,
    }

    # Check for reasonable signal frequency
    if stats["signal_rate"] > 0.1:  # More than 10% of bars have signals
        validation["warnings"].append("High signal frequency may lead to overtrading")
        validation["recommendations"].append("Consider stricter entry conditions")

    if stats["signal_rate"] < 0.01:  # Less than 1% of bars have signals
        validation["warnings"].append("Very low signal frequency")
        validation["recommendations"].append("Consider relaxing entry conditions")

    # Check for signal balance
    long_ratio = stats["long_entries_count"] / max(stats["total_signals"], 1)
    if long_ratio > 0.8 or long_ratio < 0.2:
        validation["warnings"].append("Imbalanced long/short signal distribution")
        validation["recommendations"].append("Review trend filters and conditions")

    # Check for exit coverage
    long_exit_coverage = stats["long_exits_count"] / max(stats["long_entries_count"], 1)
    short_exit_coverage = stats["short_exits_count"] / max(
        stats["short_entries_count"], 1
    )

    if long_exit_coverage < 0.5 or short_exit_coverage < 0.5:
        validation["warnings"].append("Low exit signal coverage")
        validation["recommendations"].append("Ensure adequate exit conditions")

    if validation["warnings"]:
        validation["is_valid"] = False

    return validation
