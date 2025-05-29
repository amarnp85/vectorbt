"""
Portfolio Simulation Engine

This module provides comprehensive portfolio simulation functionality using vectorbtpro's
Portfolio capabilities. It supports advanced features like dynamic stops, multi-asset portfolios,
portfolio optimization integration, and flexible position sizing strategies.

NEW: Updated to use unified signal interface for consistent communication with trading_signals.py.

Key Features:
- Signal-based portfolio simulation with vbt.Portfolio.from_signals()
- Dynamic stop-loss and take-profit handling
- Multi-asset portfolio support with cash sharing
- Portfolio optimization integration (Riskfolio-Lib, PyPortfolioOpt)
- Advanced position sizing strategies
- Transaction costs and slippage modeling
- Leverage and margin trading support
- Performance analysis and reporting
- Unified signal interface compatibility
- Signal validation and cross-reference capabilities

Usage:
    # Basic simulation
    simulator = PortfolioSimulator(data)
    portfolio = simulator.simulate_from_signals(signals)

    # Advanced multi-asset simulation
    simulator = MultiAssetPortfolioSimulator(data)
    portfolio = simulator.simulate_with_optimization(signals, optimization_method='riskfolio')
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional, Union, Tuple, Callable
from ..utilities.structured_logging import get_logger
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from .position_sizing import (
    PositionSizingConfig, 
    PositionSizingMethod,
    create_position_sizer
)

# Import unified signal interface
try:
    from backtester.signals.signal_interface import (
        SignalFormat, SignalValidator, ValidationResult,
        convert_legacy_signals, create_signal_summary_report
    )
    UNIFIED_INTERFACE_AVAILABLE = True
except ImportError:
    UNIFIED_INTERFACE_AVAILABLE = False

logger = get_logger("portfolio")


class PositionSizeMode(Enum):
    """Position sizing modes for portfolio simulation."""

    FIXED_CASH = "fixed_cash"
    FIXED_SHARES = "fixed_shares"
    PERCENT_EQUITY = "percent_equity"
    VOLATILITY_TARGET = "volatility_target"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""

    EQUAL_WEIGHT = "equal_weight"
    RISKFOLIO = "riskfolio"
    PYPFOPT = "pypfopt"
    CUSTOM = "custom"


@dataclass
class SimulationConfig:
    """Configuration for portfolio simulation."""

    init_cash: float = 100000.0
    fees: float = 0.001  # 0.1%
    slippage: float = 0.0005  # 0.05%
    fixed_fees: float = 0.0
    freq: Optional[str] = None

    # Position sizing
    position_size_mode: PositionSizeMode = PositionSizeMode.PERCENT_EQUITY
    position_size_value: float = 0.1  # 10% of equity per position
    max_position_size: Optional[float] = None
    
    # Position sizing configuration (new)
    position_sizing_config: Optional[PositionSizingConfig] = None

    # Volatility targeting parameters
    volatility_lookback: int = 20  # Lookback period for volatility calculation
    target_volatility: Optional[float] = (
        None  # Target volatility for volatility-based sizing
    )

    # Risk management
    max_leverage: float = 1.0
    cash_sharing: bool = True

    # Stop management
    use_dynamic_stops: bool = True
    sl_method: str = "percent"  # "percent", "atr", "fixed"
    tp_method: str = "percent"  # "percent", "atr", "fixed"

    # Advanced features
    allow_partial_fills: bool = True
    use_limit_orders: bool = False
    limit_delta: float = 0.001

    # Portfolio optimization
    optimization_method: OptimizationMethod = OptimizationMethod.EQUAL_WEIGHT
    rebalance_freq: Optional[str] = None  # "M", "Q", "Y" for monthly, quarterly, yearly

    # Unified signal interface options
    use_unified_signals: bool = True  # Use new unified signal interface
    validate_signals: bool = True  # Validate signals before simulation
    signal_cross_reference: bool = False  # Enable cross-reference validation

    # Additional parameters
    additional_params: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_config_dict(cls, config: Dict[str, Any]) -> 'SimulationConfig':
        """Create SimulationConfig from configuration dictionary."""
        portfolio_params = config.get('portfolio_parameters', {})
        risk_params = config.get('risk_management', {})
        position_sizing_params = risk_params.get('position_sizing', {})
        
        # Create position sizing config if parameters exist
        position_sizing_config = None
        if position_sizing_params:
            # Map method string to enum
            method_str = position_sizing_params.get('method', 'fixed_percent')
            method_map = {
                'fixed_percent': PositionSizingMethod.FIXED_PERCENT,
                'fixed_amount': PositionSizingMethod.FIXED_AMOUNT,
                'volatility_target': PositionSizingMethod.VOLATILITY_TARGET,
                'atr_based': PositionSizingMethod.ATR_BASED,
                'kelly_criterion': PositionSizingMethod.KELLY_CRITERION,
                'risk_parity': PositionSizingMethod.RISK_PARITY
            }
            method = method_map.get(method_str, PositionSizingMethod.FIXED_PERCENT)
            
            position_sizing_config = PositionSizingConfig(
                method=method,
                base_size=position_sizing_params.get('base_size', 0.1),
                max_position_size=position_sizing_params.get('max_position_size', 1.0),
                min_position_size=position_sizing_params.get('min_position_size', 0.01),
                target_volatility=position_sizing_params.get('volatility_target', 0.15),
                volatility_lookback=position_sizing_params.get('volatility_lookback', 20),
                atr_period=position_sizing_params.get('atr_period', 14),
                atr_multiplier=position_sizing_params.get('atr_multiplier', 2.0),
                kelly_lookback=position_sizing_params.get('kelly_lookback', 252),
                kelly_fraction=position_sizing_params.get('kelly_fraction', 0.25)
            )
        
        # Map position size mode from config
        mode_str = portfolio_params.get('position_size_mode', 'fixed_percent')
        mode_map = {
            'fixed_percent': PositionSizeMode.PERCENT_EQUITY,
            'percent_equity': PositionSizeMode.PERCENT_EQUITY,
            'fixed_cash': PositionSizeMode.FIXED_CASH,
            'fixed_shares': PositionSizeMode.FIXED_SHARES,
            'volatility_target': PositionSizeMode.VOLATILITY_TARGET,
            'kelly_criterion': PositionSizeMode.KELLY_CRITERION,
            'risk_parity': PositionSizeMode.RISK_PARITY
        }
        position_size_mode = mode_map.get(mode_str, PositionSizeMode.PERCENT_EQUITY)
        
        return cls(
            init_cash=portfolio_params.get('init_cash', 100000.0),
            fees=portfolio_params.get('fees', 0.001),
            slippage=portfolio_params.get('slippage', 0.0005),
            fixed_fees=portfolio_params.get('fixed_fees', 0.0),
            freq=portfolio_params.get('freq'),
            position_size_mode=position_size_mode,
            position_size_value=portfolio_params.get('position_size_value', 0.1),
            position_sizing_config=position_sizing_config,
            max_leverage=portfolio_params.get('max_leverage', 1.0),
            cash_sharing=portfolio_params.get('cash_sharing', True)
        )


class BasePortfolioSimulator(ABC):
    """Base class for portfolio simulators."""

    def __init__(
        self,
        data: Union[vbt.Data, pd.DataFrame, pd.Series],
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize the portfolio simulator.

        Args:
            data: Price data (vbt.Data, DataFrame, or Series)
            config: Simulation configuration
        """
        self.data = data
        self.config = config or SimulationConfig()
        self.portfolio = None
        self.prep_result = None

        # Initialize signal validator if unified interface is available
        if UNIFIED_INTERFACE_AVAILABLE and self.config.use_unified_signals:
            self.signal_validator = SignalValidator(strict_mode=False)
        else:
            self.signal_validator = None

        # Ensure data has proper frequency
        if hasattr(data, "wrapper") and hasattr(data.wrapper, "freq"):
            if self.config.freq is None:
                self.config.freq = data.wrapper.freq

        # Initialization logged at debug level only
        logger.debug(f"Initialized {self.__class__.__name__}")

    @abstractmethod
    def simulate_from_signals(self, signals: Dict[str, Any], **kwargs) -> vbt.Portfolio:
        """Simulate portfolio from trading signals."""

    def _prepare_signals(self, signals: Union[SignalFormat, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Prepare and validate signals for VectorBTPro simulation.
        
        NEW: Supports both unified SignalFormat and legacy dictionary format.
        """
        # Get data index for signal preparation
        if hasattr(self.data, "wrapper"):
            data_index = self.data.wrapper.index
        elif hasattr(self.data, "index"):
            data_index = self.data.index
        else:
            raise ValueError("Cannot determine data index for signal preparation")

        # Handle unified signal format
        if UNIFIED_INTERFACE_AVAILABLE and isinstance(signals, SignalFormat):
            logger.debug("Processing unified SignalFormat")
            
            # Validate signals if enabled
            if self.config.validate_signals and self.signal_validator:
                validation_result = self.signal_validator.validate_signals(signals)
                if not validation_result.is_valid:
                    logger.error(f"Signal validation failed: {validation_result.errors}")
                    for error in validation_result.errors:
                        logger.error(f"Validation error: {error}")
                for warning in validation_result.warnings:
                    logger.warning(f"Signal validation: {warning}")
            
            # Convert to dictionary format for VectorBT
            prepared_signals = signals.to_dict()
            
            # Log signal summary
            summary = signals.get_summary()
            logger.info(f"Prepared unified signals: {summary}")
            
        elif isinstance(signals, dict):
            logger.debug("Processing legacy dictionary format")
            
            # Convert legacy format to unified format for validation
            if UNIFIED_INTERFACE_AVAILABLE and self.config.use_unified_signals:
                try:
                    unified_signals = convert_legacy_signals(signals, data_index)
                    
                    # Validate converted signals
                    if self.config.validate_signals and self.signal_validator:
                        validation_result = self.signal_validator.validate_signals(unified_signals)
                        if not validation_result.is_valid:
                            logger.error(f"Legacy signal validation failed: {validation_result.errors}")
                        for warning in validation_result.warnings:
                            logger.warning(f"Legacy signal validation: {warning}")
                    
                    # Use converted signals
                    prepared_signals = unified_signals.to_dict()
                    
                    # Log conversion summary
                    summary = unified_signals.get_summary()
                    logger.info(f"Converted legacy signals: {summary}")
                    
                except Exception as e:
                    logger.warning(f"Failed to convert legacy signals to unified format: {e}")
                    prepared_signals = self._prepare_signals_legacy(signals, data_index)
            else:
                prepared_signals = self._prepare_signals_legacy(signals, data_index)
        else:
            raise ValueError(f"Unsupported signal format: {type(signals)}")

        return prepared_signals

    def _prepare_signals_legacy(self, signals: Dict[str, Any], data_index: pd.Index) -> Dict[str, Any]:
        """Legacy signal preparation method."""
        from backtester.signals.signal_utils import SignalPreparator

        # Use centralized signal preparator
        preparator = SignalPreparator(strict_validation=False)
        prepared_signals = preparator.prepare_signals(
            signals, data_index, fill_missing=True
        )

        return prepared_signals

    def _calculate_position_sizes(
        self, signals: Dict[str, Any]
    ) -> Union[float, pd.Series]:
        """Calculate position sizes based on configuration."""
        
        # If position sizing config is provided, use the position sizer
        if self.config.position_sizing_config is not None:
            position_sizer = create_position_sizer(
                self.config.position_sizing_config.method,  # Pass method, not config
                self.config.position_sizing_config  # Pass config as second argument
            )
            
            # Get close prices for position sizing
            if hasattr(self.data, "close"):
                close_data = self.data.close
            else:
                close_data = self.data
                
            # Calculate sizes using the position sizer
            sizes = position_sizer.calculate_sizes(
                close_data,
                signals
            )
            
            # Apply constraints
            if self.config.max_position_size:
                sizes = sizes.clip(upper=self.config.max_position_size)
                
            return sizes
        
        # Otherwise use legacy position size calculation
        if self.config.position_size_mode == PositionSizeMode.PERCENT_EQUITY:
            # For percent equity, size is a fixed percentage
            return self.config.position_size_value

        elif self.config.position_size_mode == PositionSizeMode.FIXED_CASH:
            # For fixed cash, size is a fixed amount
            return self.config.position_size_value

        elif self.config.position_size_mode == PositionSizeMode.VOLATILITY_TARGET:
            # Calculate volatility-based position sizing
            if hasattr(self.data, "returns"):
                returns = self.data.returns
            else:
                returns = self.data.pct_change()

            lookback = self.config.volatility_lookback
            volatility = returns.rolling(window=lookback).std() * np.sqrt(
                252
            )  # Annualized volatility

            # Use target_volatility if specified, otherwise use position_size_value
            target_vol = (
                self.config.target_volatility or self.config.position_size_value
            )
            position_sizes = target_vol / volatility

            # Cap position sizes
            if self.config.max_position_size:
                position_sizes = position_sizes.clip(
                    upper=self.config.max_position_size
                )

            return position_sizes.fillna(0.1)  # Default to 10% if volatility is NaN

        else:
            return self.config.position_size_value

    def _prepare_stops(
        self, signals: Dict[str, Any]
    ) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
        """Prepare stop-loss and take-profit levels."""
        sl_stop = None
        tp_stop = None

        if "sl_levels" in signals and signals["sl_levels"] is not None:
            sl_stop = signals["sl_levels"]

        if "tp_levels" in signals and signals["tp_levels"] is not None:
            tp_stop = signals["tp_levels"]

        return sl_stop, tp_stop

    def get_performance_stats(self) -> pd.Series:
        """Get comprehensive performance statistics."""
        if self.portfolio is None:
            raise ValueError("No portfolio found. Run simulation first.")

        return self.portfolio.stats()

    def plot_portfolio(self, **kwargs) -> Any:
        """Plot portfolio performance."""
        if self.portfolio is None:
            raise ValueError("No portfolio found. Run simulation first.")

        return self.portfolio.plot(**kwargs)

    def get_trades_analysis(self) -> pd.DataFrame:
        """Get detailed trades analysis."""
        if self.portfolio is None:
            raise ValueError("No portfolio found. Run simulation first.")

        return self.portfolio.trades.readable

    def validate_signals_against_portfolio(self, signals: Union[SignalFormat, Dict[str, Any]]) -> Optional[ValidationResult]:
        """
        Validate signals against the portfolio results for consistency.
        
        Args:
            signals: Signal data to validate
            
        Returns:
            ValidationResult if validation was performed, None otherwise
        """
        if not UNIFIED_INTERFACE_AVAILABLE or not self.config.signal_cross_reference:
            return None
            
        if self.portfolio is None:
            logger.warning("No portfolio available for cross-reference validation")
            return None
        
        try:
            from backtester.signals.signal_interface import TradeSignalCrossReference
            
            # Convert signals to unified format if needed
            if isinstance(signals, dict):
                data_index = self.data.wrapper.index if hasattr(self.data, "wrapper") else self.data.index
                unified_signals = convert_legacy_signals(signals, data_index)
            else:
                unified_signals = signals
            
            # Perform cross-reference validation
            cross_ref = TradeSignalCrossReference(self.portfolio)
            # Note: CSV path would need to be provided separately for full validation
            
            # For now, validate signal consistency
            if self.signal_validator:
                portfolio_summary = {
                    'total_trades': len(self.portfolio.trades.records_readable),
                    'portfolio_return': self.portfolio.total_return
                }
                signal_summary = unified_signals.get_summary()
                
                logger.info(f"Portfolio vs Signal Cross-Reference:")
                logger.info(f"  Portfolio trades: {portfolio_summary['total_trades']}")
                logger.info(f"  Signal entries: {signal_summary['total_entries']}")
                logger.info(f"  Portfolio return: {portfolio_summary['portfolio_return']:.2%}")
                
                # Basic consistency check
                trade_diff = abs(portfolio_summary['total_trades'] - signal_summary['total_entries'])
                if trade_diff > 1:
                    logger.warning(f"Trade count mismatch: Portfolio={portfolio_summary['total_trades']}, Signals={signal_summary['total_entries']}")
                
        except Exception as e:
            logger.error(f"Signal validation against portfolio failed: {e}")
            return None
    
    def generate_signal_report(self, signals: Union[SignalFormat, Dict[str, Any]]) -> str:
        """
        Generate a comprehensive signal report.
        
        Args:
            signals: Signal data to analyze
            
        Returns:
            Formatted signal report
        """
        if not UNIFIED_INTERFACE_AVAILABLE:
            return "Signal reporting requires unified interface module"
        
        try:
            return create_signal_summary_report(
                signals=signals,
                portfolio=self.portfolio
            )
        except Exception as e:
            return f"Signal report generation failed: {e}"


class PortfolioSimulator(BasePortfolioSimulator):
    """Single-asset portfolio simulator with advanced features."""

    def simulate_from_signals(self, signals: Union[SignalFormat, Dict[str, Any]], **kwargs) -> vbt.Portfolio:
        """
        Run portfolio simulation from trading signals using VectorBTPro.

        Args:
            signals: Dictionary containing trading signals and risk levels (now supports SignalFormat)
            **kwargs: Additional parameters for Portfolio.from_signals()

        Returns:
            vbt.Portfolio: Portfolio simulation result
        """
        # Prepare signals with proper mapping and validation
        signals = self._prepare_signals(signals)

        # Calculate position sizes
        position_sizes = self._calculate_position_sizes(signals)

        # Extract stop levels - these are now actual price levels, not deltas
        sl_stop = None
        tp_stop = None
        
        # Check for stop levels in signals
        if 'sl_levels' in signals and signals['sl_levels'] is not None:
            sl_levels = signals['sl_levels']
            # Only use non-NaN values
            if not sl_levels.isna().all():
                sl_stop = sl_levels
                
        if 'tp_levels' in signals and signals['tp_levels'] is not None:
            tp_levels = signals['tp_levels']
            # Only use non-NaN values
            if not tp_levels.isna().all():
                tp_stop = tp_levels

        # Use VectorBTPro's native data handling
        if hasattr(self.data, "close"):
            close_data = self.data.close
        else:
            close_data = self.data

        # Prepare portfolio parameters for VectorBTPro
        portfolio_params = {
            "close": close_data,
            "entries": signals["long_entries"],
            "exits": signals["long_exits"],
            "short_entries": signals["short_entries"],
            "short_exits": signals["short_exits"],
            "init_cash": self.config.init_cash,
            "fees": self.config.fees,
            "slippage": self.config.slippage,
            "fixed_fees": self.config.fixed_fees,
            "freq": self.config.freq,
            "cash_sharing": self.config.cash_sharing,
            **kwargs,  # Allow override of any parameter
        }

        # Add position sizing configuration
        if self.config.position_size_mode == PositionSizeMode.PERCENT_EQUITY:
            portfolio_params["size"] = position_sizes
            portfolio_params["size_type"] = "percent"
        elif self.config.position_size_mode == PositionSizeMode.FIXED_CASH:
            portfolio_params["size"] = position_sizes
            portfolio_params["size_type"] = "amount"
        else:
            portfolio_params["size"] = position_sizes

        # Add stops as VectorBT parameters
        # VectorBT expects stops as percentage values (0.05 = 5% stop)
        if sl_stop is not None:
            try:
                # The strategy already provides percentage-based stops
                portfolio_params["sl_stop"] = sl_stop
                logger.debug(f"Added stop loss levels: {sl_stop.notna().sum()} levels")
            except Exception as e:
                logger.warning(f"Could not add stop loss levels: {e}")
            
        if tp_stop is not None:
            try:
                # The strategy already provides percentage-based stops
                portfolio_params["tp_stop"] = tp_stop
                logger.debug(f"Added take profit levels: {tp_stop.notna().sum()} levels")
            except Exception as e:
                logger.warning(f"Could not add take profit levels: {e}")

        # Add advanced features
        if self.config.max_leverage != 1.0:
            portfolio_params["leverage"] = self.config.max_leverage

        if self.config.use_limit_orders:
            portfolio_params["order_type"] = "limit"
            portfolio_params["limit_delta"] = self.config.limit_delta

        # Merge additional config parameters
        portfolio_params.update(self.config.additional_params)

        # Run VectorBTPro simulation
        try:
            # Log portfolio creation for debugging
            logger.debug(f"Creating portfolio with {signals['long_entries'].sum()} long entries, "
                        f"{signals['long_exits'].sum()} long exits, "
                        f"{signals['short_entries'].sum()} short entries, "
                        f"{signals['short_exits'].sum()} short exits")
            
            self.portfolio = vbt.Portfolio.from_signals(**portfolio_params)
            
            # Log portfolio statistics
            if hasattr(self.portfolio, 'total_return'):
                logger.debug(f"Portfolio created successfully. Total return: {self.portfolio.total_return:.2%}")
            
            # Perform cross-reference validation if enabled
            if self.config.signal_cross_reference:
                self.validate_signals_against_portfolio(signals)
            
            return self.portfolio

        except Exception as e:
            logger.error(f"Portfolio simulation failed: {str(e)}")
            # Log portfolio params for debugging
            logger.error(f"Portfolio params keys: {list(portfolio_params.keys())}")
            
            # Enhanced error logging with signal information
            signal_summary = {}
            for key in ['long_entries', 'long_exits', 'short_entries', 'short_exits']:
                if key in signals:
                    signal_summary[key] = f"shape={signals[key].shape}, sum={signals[key].sum()}"
            logger.error(f"Signal summary: {signal_summary}")
            raise

    def simulate_with_prep_result(self, prep_result: Any) -> vbt.Portfolio:
        """Simulate portfolio from a preparation result."""
        try:
            self.prep_result = prep_result
            self.portfolio = vbt.Portfolio.from_signals(prep_result)

            logger.info("Simulation from prep result completed successfully")
            return self.portfolio

        except Exception as e:
            logger.error(f"Simulation from prep result failed: {str(e)}")
            raise


class MultiAssetPortfolioSimulator(BasePortfolioSimulator):
    """Multi-asset portfolio simulator with optimization capabilities."""

    def __init__(self, data: vbt.Data, config: Optional[SimulationConfig] = None):
        """
        Initialize multi-asset portfolio simulator.

        Args:
            data: Multi-asset price data (vbt.Data object)
            config: Simulation configuration
        """
        super().__init__(data, config)

        if not hasattr(data, "symbols"):
            raise ValueError(
                "Multi-asset simulator requires vbt.Data object with multiple symbols"
            )

        self.symbols = data.symbols
        # Multi-asset initialization logged at debug level
        logger.debug(f"Initialized multi-asset simulator with {len(self.symbols)} symbols")

    def simulate_from_signals(self, signals: Union[SignalFormat, Dict[str, Any]], **kwargs) -> vbt.Portfolio:
        """
        Run multi-asset portfolio simulation from trading signals.

        Args:
            signals: Dictionary containing trading signals for each asset (now supports SignalFormat)
            **kwargs: Additional parameters for Portfolio.from_signals()

        Returns:
            vbt.Portfolio: Multi-asset portfolio simulation result
        """
        # Prepare signals for all assets
        signals = self._prepare_signals(signals)

        # Calculate position sizes
        position_sizes = self._calculate_position_sizes(signals)

        # Prepare stops
        sl_stop, tp_stop = self._prepare_stops(signals)

        # Prepare portfolio parameters
        portfolio_params = {
            "data": self.data,
            "entries": signals["long_entries"],
            "exits": signals["long_exits"],
            "short_entries": signals["short_entries"],
            "short_exits": signals["short_exits"],
            "init_cash": self.config.init_cash,
            "fees": self.config.fees,
            "slippage": self.config.slippage,
            "fixed_fees": self.config.fixed_fees,
            "freq": self.config.freq,
            "cash_sharing": self.config.cash_sharing,
            "group_by": True,  # Group all assets into one portfolio
            **self.config.additional_params,
            **kwargs,
        }

        # Add position sizing
        if isinstance(position_sizes, (int, float)):
            portfolio_params["size"] = position_sizes
            portfolio_params["size_type"] = (
                "percent"
                if self.config.position_size_mode == PositionSizeMode.PERCENT_EQUITY
                else "amount"
            )
        else:
            portfolio_params["size"] = position_sizes

        # Add stops
        if sl_stop is not None:
            portfolio_params["sl_stop"] = sl_stop
        if tp_stop is not None:
            portfolio_params["tp_stop"] = tp_stop

        # Add leverage if specified
        if self.config.max_leverage != 1.0:
            portfolio_params["leverage"] = self.config.max_leverage

        # Run simulation
        try:
            self.portfolio = vbt.Portfolio.from_signals(**portfolio_params)
            
            # Perform cross-reference validation if enabled
            if self.config.signal_cross_reference:
                self.validate_signals_against_portfolio(signals)
                
            return self.portfolio

        except Exception as e:
            logger.error(f"Multi-asset portfolio simulation failed: {str(e)}")
            raise

    def simulate_with_optimization(
        self,
        signals: Union[SignalFormat, Dict[str, Any]],
        optimization_method: str = "equal_weight",
        rebalance_freq: str = "M",
        **kwargs,
    ) -> vbt.Portfolio:
        """
        Run portfolio simulation with optimization-based position sizing.

        Args:
            signals: Trading signals for each asset (now supports SignalFormat)
            optimization_method: Optimization method ("equal_weight", "riskfolio", "pypfopt")
            rebalance_freq: Rebalancing frequency ("M", "Q", "Y")
            **kwargs: Additional parameters

        Returns:
            vbt.Portfolio: Optimized portfolio simulation result
        """
        try:
            # First run basic simulation to get returns
            basic_portfolio = self.simulate_from_signals(signals, **kwargs)

            if optimization_method == "equal_weight":
                # Equal weight allocation
                weights = {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}
                optimized_portfolio = basic_portfolio.apply_weights(
                    weights, rescale=True
                )

            elif optimization_method == "riskfolio":
                # Use Riskfolio-Lib for optimization
                try:
                    prices = basic_portfolio.get_value(group_by=False)
                    weights = vbt.pypfopt_optimize(prices=prices)
                    optimized_portfolio = basic_portfolio.apply_weights(
                        weights, rescale=True
                    )

                    logger.info(f"Riskfolio optimization weights: {weights}")

                except Exception as e:
                    logger.warning(
                        f"Riskfolio optimization failed: {e}, falling back to equal weight"
                    )
                    weights = {
                        symbol: 1.0 / len(self.symbols) for symbol in self.symbols
                    }
                    optimized_portfolio = basic_portfolio.apply_weights(
                        weights, rescale=True
                    )

            elif optimization_method == "pypfopt":
                # Use PyPortfolioOpt for optimization
                try:
                    prices = basic_portfolio.get_value(group_by=False)
                    weights = vbt.pypfopt_optimize(prices=prices)
                    optimized_portfolio = basic_portfolio.apply_weights(
                        weights, rescale=True
                    )

                    logger.info(f"PyPortfolioOpt optimization weights: {weights}")

                except Exception as e:
                    logger.warning(
                        f"PyPortfolioOpt optimization failed: {e}, falling back to equal weight"
                    )
                    weights = {
                        symbol: 1.0 / len(self.symbols) for symbol in self.symbols
                    }
                    optimized_portfolio = basic_portfolio.apply_weights(
                        weights, rescale=True
                    )

            else:
                logger.warning(
                    f"Unknown optimization method: {optimization_method}, using equal weight"
                )
                weights = {symbol: 1.0 / len(self.symbols) for symbol in self.symbols}
                optimized_portfolio = basic_portfolio.apply_weights(
                    weights, rescale=True
                )

            self.portfolio = optimized_portfolio

            # Log optimization results
            opt_return = optimized_portfolio.total_return
            opt_sharpe = optimized_portfolio.sharpe_ratio

            logger.info(f"Optimized portfolio results:")
            logger.info(f"  Optimized Return: {opt_return*100:.2f}%")
            logger.info(f"  Optimized Sharpe: {opt_sharpe:.3f}")

            return optimized_portfolio

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {str(e)}")
            raise

    def get_asset_allocation(self) -> pd.Series:
        """Get current asset allocation weights."""
        if self.portfolio is None:
            raise ValueError("No portfolio found. Run simulation first.")

        return self.portfolio.weights


class AdvancedPortfolioSimulator(MultiAssetPortfolioSimulator):
    """Advanced portfolio simulator with additional features."""

    def simulate_with_dynamic_rebalancing(
        self,
        signals: Union[SignalFormat, Dict[str, Any]],
        rebalance_func: Callable,
        rebalance_freq: str = "M",
        **kwargs,
    ) -> vbt.Portfolio:
        """
        Run simulation with dynamic rebalancing based on custom function.

        Args:
            signals: Trading signals (now supports SignalFormat)
            rebalance_func: Function that returns new weights
            rebalance_freq: Rebalancing frequency
            **kwargs: Additional parameters

        Returns:
            vbt.Portfolio: Portfolio with dynamic rebalancing
        """
        # This would require more advanced implementation
        # For now, delegate to basic optimization
        return self.simulate_with_optimization(signals, **kwargs)

    def simulate_with_risk_budgeting(
        self, signals: Union[SignalFormat, Dict[str, Any]], risk_budget: Dict[str, float], **kwargs
    ) -> vbt.Portfolio:
        """
        Run simulation with risk budgeting approach.

        Args:
            signals: Trading signals (now supports SignalFormat)
            risk_budget: Risk budget allocation per asset
            **kwargs: Additional parameters

        Returns:
            vbt.Portfolio: Risk-budgeted portfolio
        """
        # Implement risk budgeting logic
        # For now, use risk budget as weights
        basic_portfolio = self.simulate_from_signals(signals, **kwargs)
        optimized_portfolio = basic_portfolio.apply_weights(risk_budget, rescale=True)

        self.portfolio = optimized_portfolio
        return optimized_portfolio


# Convenience functions for backward compatibility and ease of use
def run_portfolio_simulation(
    signals: Union[SignalFormat, Dict[str, Any]],
    data: Union[vbt.Data, pd.DataFrame, pd.Series],
    config: Optional[SimulationConfig] = None,
    **kwargs,
) -> vbt.Portfolio:
    """
    Convenience function to run portfolio simulation.

    Args:
        signals: Trading signals dictionary or SignalFormat
        data: Price data
        config: Simulation configuration
        **kwargs: Additional parameters

    Returns:
        vbt.Portfolio: Portfolio simulation result
    """
    if config is None:
        config = SimulationConfig()

    # Update config with kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Choose appropriate simulator
    if hasattr(data, "symbols") and len(data.symbols) > 1:
        simulator = MultiAssetPortfolioSimulator(data, config)
    else:
        simulator = PortfolioSimulator(data, config)

    return simulator.simulate_from_signals(signals)


def create_simulation_config(**kwargs) -> SimulationConfig:
    """
    Create a simulation configuration with custom parameters.

    Args:
        **kwargs: Configuration parameters

    Returns:
        SimulationConfig: Configuration object
    """
    return SimulationConfig(**kwargs)


def run_custom_simulation(
    close_data: pd.Series,
    entry_signals: pd.Series,
    exit_signals: pd.Series,
    size_func: callable,
    init_cash: float = 100000,
    **kwargs,
) -> vbt.Portfolio:
    """
    Run custom portfolio simulation with dynamic sizing function.

    This function provides more control over position sizing and execution logic
    using vectorbtpro's from_order_func constructor.

    Args:
        close_data (pd.Series): Close price data
        entry_signals (pd.Series): Entry signals
        exit_signals (pd.Series): Exit signals
        size_func (callable): Function to calculate position sizes
        init_cash (float): Initial cash
        **kwargs: Additional portfolio parameters

    Returns:
        vbt.Portfolio: Portfolio simulation result
    """
    logger.info("Running custom portfolio simulation with dynamic sizing")

    # This is a placeholder for more advanced simulation logic
    # In practice, you would implement custom order generation logic here
    # using vectorbtpro's order generation capabilities

    raise NotImplementedError(
        "Custom simulation with dynamic sizing not yet implemented"
    )


def validate_signals(signals: Union[SignalFormat, Dict[str, pd.Series]]) -> bool:
    """
    Validate signal data for portfolio simulation.

    Args:
        signals: Signal dictionary or SignalFormat

    Returns:
        bool: True if signals are valid

    Raises:
        ValueError: If signals are invalid
    """
    if UNIFIED_INTERFACE_AVAILABLE:
        validator = SignalValidator(strict_mode=True)
        
        if isinstance(signals, SignalFormat):
            validation_result = validator.validate_signals(signals)
        else:
            # Convert dictionary to unified format for validation
            index = signals.get('index') or next(iter(signals.values())).index
            unified_signals = convert_legacy_signals(signals, index)
            validation_result = validator.validate_signals(unified_signals)
        
        if not validation_result.is_valid:
            error_msg = f"Signal validation failed: {validation_result.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log warnings if any
        for warning in validation_result.warnings:
            logger.warning(f"Signal validation warning: {warning}")

        logger.info("Signal validation passed")
        return True
    else:
        # Fallback to legacy validation
        from backtester.signals.signal_utils import SignalPreparator

        # Use centralized signal validation
        preparator = SignalPreparator(strict_validation=True)
        validation_result = preparator.validate_signals(signals)

        if not validation_result.is_valid:
            error_msg = f"Signal validation failed: {validation_result.errors}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Log warnings if any
        for warning in validation_result.warnings:
            logger.warning(f"Signal validation warning: {warning}")

        logger.info("Signal validation passed")
        return True


def calculate_portfolio_metrics(portfolio: vbt.Portfolio) -> Dict[str, float]:
    """
    Calculate comprehensive portfolio performance metrics.

    Args:
        portfolio (vbt.Portfolio): Portfolio simulation result

    Returns:
        Dict[str, float]: Dictionary of performance metrics
    """

    metrics = {}

    try:
        # Basic return metrics
        metrics["total_return"] = portfolio.total_return
        metrics["total_return_pct"] = portfolio.total_return * 100
        metrics["annualized_return"] = portfolio.annualized_return
        metrics["annualized_return_pct"] = portfolio.annualized_return * 100

        # Risk metrics
        metrics["max_drawdown"] = portfolio.max_drawdown
        metrics["max_drawdown_pct"] = portfolio.max_drawdown * 100
        metrics["volatility"] = portfolio.annualized_volatility
        metrics["downside_risk"] = portfolio.downside_risk()

        # Risk-adjusted returns
        metrics["sharpe_ratio"] = portfolio.sharpe_ratio
        metrics["sortino_ratio"] = portfolio.sortino_ratio()
        metrics["calmar_ratio"] = portfolio.calmar_ratio()

        # Trade metrics
        metrics["win_rate"] = portfolio.trades.win_rate()
        metrics["win_rate_pct"] = portfolio.trades.win_rate() * 100
        metrics["avg_win"] = portfolio.trades.avg_win()
        metrics["avg_loss"] = portfolio.trades.avg_loss()
        metrics["profit_factor"] = portfolio.trades.profit_factor()

        # Count metrics
        metrics["total_trades"] = len(portfolio.trades.records_arr)
        metrics["winning_trades"] = portfolio.trades.winning.count()
        metrics["losing_trades"] = portfolio.trades.losing.count()

        logger.info("Portfolio metrics calculated successfully")

    except Exception as e:
        logger.error(f"Error calculating portfolio metrics: {str(e)}")
        metrics["error"] = str(e)

    return metrics


def create_portfolio_summary(portfolio: vbt.Portfolio) -> str:
    """
    Create a formatted summary of portfolio performance.

    Args:
        portfolio (vbt.Portfolio): Portfolio simulation result

    Returns:
        str: Formatted performance summary
    """
    metrics = calculate_portfolio_metrics(portfolio)

    summary = f"""
Portfolio Performance Summary
============================

Return Metrics:
- Total Return: {metrics.get('total_return_pct', 'N/A'):.2f}%
- Annualized Return: {metrics.get('annualized_return_pct', 'N/A'):.2f}%

Risk Metrics:
- Max Drawdown: {metrics.get('max_drawdown_pct', 'N/A'):.2f}%
- Volatility: {metrics.get('volatility', 'N/A'):.2f}%
- Downside Risk: {metrics.get('downside_risk', 'N/A'):.2f}%

Risk-Adjusted Returns:
- Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.3f}
- Sortino Ratio: {metrics.get('sortino_ratio', 'N/A'):.3f}
- Calmar Ratio: {metrics.get('calmar_ratio', 'N/A'):.3f}

Trade Statistics:
- Total Trades: {metrics.get('total_trades', 'N/A')}
- Win Rate: {metrics.get('win_rate_pct', 'N/A'):.2f}%
- Profit Factor: {metrics.get('profit_factor', 'N/A'):.3f}
- Avg Win: {metrics.get('avg_win', 'N/A'):.2f}
- Avg Loss: {metrics.get('avg_loss', 'N/A'):.2f}
"""

    return summary
