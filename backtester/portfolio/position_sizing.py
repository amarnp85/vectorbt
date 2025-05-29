"""
Position Sizing Module

This module provides various position sizing strategies for portfolio management.
It integrates with the Portfolio Simulation Engine to determine optimal position sizes
based on different risk management approaches.

Key Features:
- Fixed position sizing (cash amount, percentage)
- Volatility-based position sizing
- Kelly Criterion optimization
- Risk parity approaches
- ATR-based position sizing
- Maximum drawdown-based sizing

Usage:
    sizer = VolatilityPositionSizer(target_volatility=0.15)
    position_sizes = sizer.calculate_sizes(data, signals)
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional, Union, List
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PositionSizingMethod(Enum):
    """Available position sizing methods."""

    FIXED_AMOUNT = "fixed_amount"
    FIXED_PERCENT = "fixed_percent"
    VOLATILITY_TARGET = "volatility_target"
    ATR_BASED = "atr_based"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    MAX_DRAWDOWN = "max_drawdown"
    EQUAL_RISK = "equal_risk"


@dataclass
class PositionSizingConfig:
    """Configuration for position sizing strategies."""

    method: PositionSizingMethod = PositionSizingMethod.FIXED_PERCENT
    base_size: float = 0.1  # Base position size (10% for percent, $10k for amount)
    max_position_size: float = 0.25  # Maximum position size (25%)
    min_position_size: float = 0.01  # Minimum position size (1%)

    # Volatility-based parameters
    target_volatility: float = 0.15  # 15% annual volatility target
    volatility_lookback: int = 20  # Days for volatility calculation

    # ATR-based parameters
    atr_period: int = 14  # ATR calculation period
    atr_multiplier: float = 2.0  # Risk per trade as multiple of ATR

    # Kelly Criterion parameters
    kelly_lookback: int = 252  # Days for Kelly calculation
    kelly_fraction: float = 0.25  # Fraction of Kelly to use (conservative)

    # Risk parity parameters
    risk_lookback: int = 60  # Days for risk calculation

    # Additional constraints
    max_total_exposure: float = 1.0  # Maximum total portfolio exposure
    rebalance_threshold: float = 0.05  # Rebalance when allocation drifts by 5%


class BasePositionSizer(ABC):
    """Base class for position sizing strategies."""

    def __init__(self, config: Optional[PositionSizingConfig] = None):
        """
        Initialize position sizer.

        Args:
            config: Position sizing configuration
        """
        self.config = config or PositionSizingConfig()
        logger.info(f"Initialized {self.__class__.__name__} with config: {self.config}")

    @abstractmethod
    def calculate_sizes(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        signals: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[float, pd.Series, pd.DataFrame]:
        """
        Calculate position sizes.

        Args:
            data: Price data
            signals: Trading signals (optional)
            **kwargs: Additional parameters

        Returns:
            Position sizes (scalar, Series, or DataFrame)
        """

    def _validate_sizes(
        self, sizes: Union[float, pd.Series, pd.DataFrame]
    ) -> Union[float, pd.Series, pd.DataFrame]:
        """Apply size constraints and validation."""
        if isinstance(sizes, (pd.Series, pd.DataFrame)):
            # Apply min/max constraints
            sizes = sizes.clip(
                lower=self.config.min_position_size, upper=self.config.max_position_size
            )

            # Handle NaN values
            sizes = sizes.fillna(self.config.base_size)

        elif isinstance(sizes, (int, float)):
            # Apply constraints to scalar
            sizes = max(
                self.config.min_position_size, min(self.config.max_position_size, sizes)
            )

        return sizes


class FixedPositionSizer(BasePositionSizer):
    """Fixed position sizing strategy."""

    def calculate_sizes(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        signals: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> float:
        """
        Calculate fixed position sizes.

        Returns:
            Fixed position size
        """
        return self.config.base_size


class VolatilityPositionSizer(BasePositionSizer):
    """Volatility-based position sizing strategy."""

    def calculate_sizes(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        signals: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Calculate volatility-based position sizes.

        Args:
            data: Price data
            signals: Trading signals (optional)
            **kwargs: Additional parameters

        Returns:
            Position sizes based on volatility targeting
        """
        # Get price data
        if hasattr(data, "close"):
            prices = data.close
        else:
            prices = data

        # Calculate returns
        returns = prices.pct_change()

        # Calculate rolling volatility (annualized)
        volatility = returns.rolling(
            window=self.config.volatility_lookback
        ).std() * np.sqrt(252)

        # Calculate position sizes to target volatility
        target_vol = self.config.target_volatility
        position_sizes = target_vol / volatility

        # Apply base size scaling
        position_sizes = position_sizes * self.config.base_size

        # Validate and constrain sizes
        position_sizes = self._validate_sizes(position_sizes)

        logger.info(f"Calculated volatility-based position sizes")
        logger.info(f"  Target volatility: {target_vol*100:.1f}%")
        logger.info(f"  Average position size: {position_sizes.mean():.3f}")

        return position_sizes


class ATRPositionSizer(BasePositionSizer):
    """ATR-based position sizing strategy."""

    def calculate_sizes(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        signals: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Calculate ATR-based position sizes.

        Args:
            data: OHLC price data
            signals: Trading signals (optional)
            **kwargs: Additional parameters

        Returns:
            Position sizes based on ATR risk
        """
        # Calculate ATR
        if hasattr(data, "run"):
            # Use vbt.Data object
            atr = data.run("atr", window=self.config.atr_period, hide_params=True)
        else:
            # Calculate ATR manually (requires OHLC data)
            if not all(col in data.columns for col in ["high", "low", "close"]):
                raise ValueError(
                    "ATR position sizing requires OHLC data (high, low, close)"
                )

            high = data["high"]
            low = data["low"]
            close = data["close"]

            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))

            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=self.config.atr_period).mean()

        # Get current prices for position size calculation
        if hasattr(data, "close"):
            prices = data.close
        else:
            prices = data["close"] if "close" in data.columns else data

        # Calculate position sizes based on ATR risk
        # Risk per trade = ATR * multiplier
        # Position size = (Risk budget) / (ATR * multiplier)
        risk_per_trade = atr * self.config.atr_multiplier
        risk_budget = self.config.base_size  # Use base_size as risk budget

        position_sizes = risk_budget / (risk_per_trade / prices)

        # Validate and constrain sizes
        position_sizes = self._validate_sizes(position_sizes)

        logger.info(f"Calculated ATR-based position sizes")
        logger.info(f"  ATR period: {self.config.atr_period}")
        logger.info(f"  ATR multiplier: {self.config.atr_multiplier}")
        logger.info(f"  Average position size: {position_sizes.mean():.3f}")

        return position_sizes


class KellyPositionSizer(BasePositionSizer):
    """Kelly Criterion position sizing strategy."""

    def calculate_sizes(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        signals: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Calculate Kelly Criterion position sizes.

        Args:
            data: Price data
            signals: Trading signals (required for Kelly calculation)
            **kwargs: Additional parameters

        Returns:
            Position sizes based on Kelly Criterion
        """
        if signals is None:
            raise ValueError("Kelly position sizing requires trading signals")

        # Get price data
        if hasattr(data, "close"):
            prices = data.close
        else:
            prices = data

        # Get entry signals
        entries = signals.get("long_entries", pd.Series(False, index=prices.index))

        # Calculate returns for Kelly estimation
        returns = prices.pct_change()

        # Calculate rolling Kelly fraction
        kelly_fractions = []

        for i in range(len(prices)):
            if i < self.config.kelly_lookback:
                kelly_fractions.append(self.config.base_size)
                continue

            # Get historical returns for this lookback period
            hist_returns = returns.iloc[i - self.config.kelly_lookback : i]

            # Calculate win rate and average win/loss
            positive_returns = hist_returns[hist_returns > 0]
            negative_returns = hist_returns[hist_returns < 0]

            if len(positive_returns) == 0 or len(negative_returns) == 0:
                kelly_fractions.append(self.config.base_size)
                continue

            win_rate = len(positive_returns) / len(hist_returns[hist_returns != 0])
            avg_win = positive_returns.mean()
            avg_loss = abs(negative_returns.mean())

            # Kelly formula: f = (bp - q) / b
            # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
            if avg_loss > 0:
                b = avg_win / avg_loss
                kelly_fraction = (b * win_rate - (1 - win_rate)) / b

                # Apply Kelly fraction (conservative approach)
                kelly_fraction = max(0, kelly_fraction) * self.config.kelly_fraction
            else:
                kelly_fraction = self.config.base_size

            kelly_fractions.append(kelly_fraction)

        position_sizes = pd.Series(kelly_fractions, index=prices.index)

        # Validate and constrain sizes
        position_sizes = self._validate_sizes(position_sizes)

        logger.info(f"Calculated Kelly Criterion position sizes")
        logger.info(f"  Kelly lookback: {self.config.kelly_lookback}")
        logger.info(f"  Kelly fraction: {self.config.kelly_fraction}")
        logger.info(f"  Average position size: {position_sizes.mean():.3f}")

        return position_sizes


class RiskParityPositionSizer(BasePositionSizer):
    """Risk parity position sizing strategy."""

    def calculate_sizes(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        signals: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate risk parity position sizes.

        Args:
            data: Price data (multi-asset for true risk parity)
            signals: Trading signals (optional)
            **kwargs: Additional parameters

        Returns:
            Position sizes based on risk parity
        """
        # Handle single asset case
        if isinstance(data, pd.Series) or (
            hasattr(data, "close") and isinstance(data.close, pd.Series)
        ):
            # For single asset, use volatility-based sizing
            return VolatilityPositionSizer(self.config).calculate_sizes(
                data, signals, **kwargs
            )

        # Multi-asset risk parity
        if hasattr(data, "close"):
            prices = data.close
        else:
            prices = data

        # Calculate returns
        returns = prices.pct_change()

        # Calculate rolling volatilities
        volatilities = returns.rolling(
            window=self.config.risk_lookback
        ).std() * np.sqrt(252)

        # Calculate inverse volatility weights (risk parity)
        inv_vol = 1 / volatilities
        weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)

        # Scale by base size
        position_sizes = weights * self.config.base_size

        # Validate and constrain sizes
        if isinstance(position_sizes, pd.DataFrame):
            for col in position_sizes.columns:
                position_sizes[col] = self._validate_sizes(position_sizes[col])
        else:
            position_sizes = self._validate_sizes(position_sizes)

        logger.info(f"Calculated risk parity position sizes")
        logger.info(f"  Risk lookback: {self.config.risk_lookback}")
        if isinstance(position_sizes, pd.DataFrame):
            logger.info(f"  Average position sizes: {position_sizes.mean().to_dict()}")

        return position_sizes


class AdaptivePositionSizer(BasePositionSizer):
    """Adaptive position sizing that combines multiple strategies."""

    def __init__(
        self,
        config: Optional[PositionSizingConfig] = None,
        strategies: Optional[Dict[str, BasePositionSizer]] = None,
    ):
        """
        Initialize adaptive position sizer.

        Args:
            config: Position sizing configuration
            strategies: Dictionary of strategy name -> sizer instance
        """
        super().__init__(config)

        self.strategies = strategies or {
            "volatility": VolatilityPositionSizer(config),
            "atr": ATRPositionSizer(config),
            "fixed": FixedPositionSizer(config),
        }

        logger.info(
            f"Initialized adaptive sizer with strategies: {list(self.strategies.keys())}"
        )

    def calculate_sizes(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        signals: Optional[Dict[str, Any]] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
        **kwargs,
    ) -> pd.Series:
        """
        Calculate adaptive position sizes by combining multiple strategies.

        Args:
            data: Price data
            signals: Trading signals
            strategy_weights: Weights for combining strategies
            **kwargs: Additional parameters

        Returns:
            Combined position sizes
        """
        if strategy_weights is None:
            strategy_weights = {
                name: 1.0 / len(self.strategies) for name in self.strategies
            }

        # Normalize weights
        total_weight = sum(strategy_weights.values())
        strategy_weights = {k: v / total_weight for k, v in strategy_weights.items()}

        # Calculate sizes from each strategy
        strategy_sizes = {}
        for name, sizer in self.strategies.items():
            try:
                sizes = sizer.calculate_sizes(data, signals, **kwargs)
                strategy_sizes[name] = sizes
                logger.info(f"Calculated sizes for strategy '{name}'")
            except Exception as e:
                logger.warning(f"Strategy '{name}' failed: {e}, skipping")
                continue

        if not strategy_sizes:
            raise ValueError("All position sizing strategies failed")

        # Combine strategies
        if hasattr(data, "close"):
            index = data.close.index
        else:
            index = data.index

        combined_sizes = pd.Series(0.0, index=index)

        for name, sizes in strategy_sizes.items():
            weight = strategy_weights.get(name, 0.0)
            if isinstance(sizes, (int, float)):
                # Convert scalar to series
                sizes = pd.Series(sizes, index=index)

            combined_sizes += sizes * weight

        # Validate and constrain final sizes
        combined_sizes = self._validate_sizes(combined_sizes)

        logger.info(f"Calculated adaptive position sizes")
        logger.info(f"  Strategy weights: {strategy_weights}")
        logger.info(f"  Average position size: {combined_sizes.mean():.3f}")

        return combined_sizes


# Factory function for creating position sizers
def create_position_sizer(
    method: Union[str, PositionSizingMethod],
    config: Optional[PositionSizingConfig] = None,
) -> BasePositionSizer:
    """
    Factory function to create position sizers.

    Args:
        method: Position sizing method
        config: Position sizing configuration

    Returns:
        Position sizer instance
    """
    if isinstance(method, str):
        method = PositionSizingMethod(method)

    sizer_map = {
        PositionSizingMethod.FIXED_AMOUNT: FixedPositionSizer,
        PositionSizingMethod.FIXED_PERCENT: FixedPositionSizer,
        PositionSizingMethod.VOLATILITY_TARGET: VolatilityPositionSizer,
        PositionSizingMethod.ATR_BASED: ATRPositionSizer,
        PositionSizingMethod.KELLY_CRITERION: KellyPositionSizer,
        PositionSizingMethod.RISK_PARITY: RiskParityPositionSizer,
    }

    if method not in sizer_map:
        raise ValueError(f"Unknown position sizing method: {method}")

    return sizer_map[method](config)


# Convenience functions
def calculate_position_sizes(
    data: Union[vbt.Data, pd.DataFrame, pd.Series],
    method: str = "fixed_percent",
    signals: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> Union[float, pd.Series, pd.DataFrame]:
    """
    Convenience function to calculate position sizes.

    Args:
        data: Price data
        method: Position sizing method
        signals: Trading signals
        **kwargs: Additional configuration parameters

    Returns:
        Position sizes
    """
    config = PositionSizingConfig(**kwargs)
    sizer = create_position_sizer(method, config)
    return sizer.calculate_sizes(data, signals)


def optimize_position_sizes(
    data: Union[vbt.Data, pd.DataFrame, pd.Series],
    signals: Dict[str, Any],
    methods: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Compare different position sizing methods and return results.

    Args:
        data: Price data
        signals: Trading signals
        methods: List of methods to compare
        **kwargs: Additional parameters

    Returns:
        Dictionary with results from each method
    """
    if methods is None:
        methods = ["fixed_percent", "volatility_target", "atr_based"]

    results = {}

    for method in methods:
        try:
            sizes = calculate_position_sizes(data, method, signals, **kwargs)
            results[method] = {
                "sizes": sizes,
                "mean_size": sizes.mean() if hasattr(sizes, "mean") else sizes,
                "std_size": sizes.std() if hasattr(sizes, "std") else 0,
            }
            logger.info(f"Calculated position sizes for method: {method}")
        except Exception as e:
            logger.warning(f"Method {method} failed: {e}")
            continue

    return results
