"""
Portfolio Module

This module provides comprehensive portfolio simulation and management functionality for the backtesting system.
It leverages vectorbtpro's Portfolio capabilities for realistic backtesting with proper risk management.

Key Features:
- Portfolio simulation using vectorbtpro.Portfolio.from_signals()
- Advanced position sizing strategies (fixed, volatility-based, ATR, Kelly, risk parity)
- Multi-asset portfolio support with optimization
- Dynamic stop-loss and take-profit handling
- Portfolio optimization integration (Riskfolio-Lib, PyPortfolioOpt)
- Performance analysis and reporting

Main Components:
- simulation_engine: Core portfolio simulation functionality
- position_sizing: Various position sizing strategies

Usage:
    # Basic simulation
    from backtester.portfolio import PortfolioSimulator, SimulationConfig
    config = SimulationConfig(init_cash=100000, fees=0.001)
    simulator = PortfolioSimulator(data, config)
    portfolio = simulator.simulate_from_signals(signals)

    # Advanced multi-asset simulation with optimization
    from backtester.portfolio import MultiAssetPortfolioSimulator
    simulator = MultiAssetPortfolioSimulator(data)
    portfolio = simulator.simulate_with_optimization(signals, optimization_method='riskfolio')

    # Position sizing
    from backtester.portfolio import VolatilityPositionSizer, PositionSizingConfig
    config = PositionSizingConfig(target_volatility=0.15)
    sizer = VolatilityPositionSizer(config)
    position_sizes = sizer.calculate_sizes(data, signals)
"""

# Import simulation engine components
from .simulation_engine import (
    # Main simulator classes
    BasePortfolioSimulator,
    PortfolioSimulator,
    MultiAssetPortfolioSimulator,
    AdvancedPortfolioSimulator,
    # Configuration and enums
    SimulationConfig,
    PositionSizeMode,
    OptimizationMethod,
    # Convenience functions
    run_portfolio_simulation,
    create_simulation_config,
)

# Import position sizing components
from .position_sizing import (
    # Position sizer classes
    BasePositionSizer,
    FixedPositionSizer,
    VolatilityPositionSizer,
    ATRPositionSizer,
    KellyPositionSizer,
    RiskParityPositionSizer,
    AdaptivePositionSizer,
    # Configuration and enums
    PositionSizingConfig,
    PositionSizingMethod,
    # Factory and convenience functions
    create_position_sizer,
    calculate_position_sizes,
    optimize_position_sizes,
)

# Legacy imports for backward compatibility
from . import simulation_engine
from . import position_sizing

__all__ = [
    # Simulation engine exports
    "BasePortfolioSimulator",
    "PortfolioSimulator",
    "MultiAssetPortfolioSimulator",
    "AdvancedPortfolioSimulator",
    "SimulationConfig",
    "PositionSizeMode",
    "OptimizationMethod",
    "run_portfolio_simulation",
    "create_simulation_config",
    # Position sizing exports
    "BasePositionSizer",
    "FixedPositionSizer",
    "VolatilityPositionSizer",
    "ATRPositionSizer",
    "KellyPositionSizer",
    "RiskParityPositionSizer",
    "AdaptivePositionSizer",
    "PositionSizingConfig",
    "PositionSizingMethod",
    "create_position_sizer",
    "calculate_position_sizes",
    "optimize_position_sizes",
    # Module exports for legacy compatibility
    "simulation_engine",
    "position_sizing",
]
