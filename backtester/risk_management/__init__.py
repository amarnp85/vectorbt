"""
Risk Management Module

Provides advanced position sizing and risk management functionality
including Kelly Criterion, volatility-based sizing, and portfolio heat monitoring.
"""

from .position_sizing import (
    VolatilityPositionSizer,
    KellyCriterionSizer,
    FixedRiskSizer,
    calculate_portfolio_heat,
    RiskParityAllocator
)

from .stop_loss import (
    DynamicStopLoss,
    ATRStopLoss,
    TrailingStopLoss
)

from .regime_detection import (
    MarketRegimeDetector,
    VolatilityRegime,
    TrendRegime
)

__all__ = [
    'VolatilityPositionSizer',
    'KellyCriterionSizer',
    'FixedRiskSizer',
    'calculate_portfolio_heat',
    'RiskParityAllocator',
    'DynamicStopLoss',
    'ATRStopLoss',
    'TrailingStopLoss',
    'MarketRegimeDetector',
    'VolatilityRegime',
    'TrendRegime'
] 