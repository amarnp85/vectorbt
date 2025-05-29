"""
Indicators Module - Simplified Direct VectorBTPro Usage

This module has been drastically simplified to provide direct access to
VectorBTPro's highly optimized indicator implementations.

Migration Guide:
1. For new code: Use simple_indicators directly
2. For existing code: Use compatibility_layer temporarily
3. Gradually migrate to direct VectorBTPro usage

Example:
    # New way (recommended)
    from backtester.indicators.simple_indicators import sma, rsi, atr
    
    # Old way (compatibility mode)
    from backtester.indicators.compatibility_layer import get_indicator_manager
"""

# Import simple indicators for direct usage (recommended)
from .simple_indicators import (
    # Moving averages
    sma, ema, wma, hma, dema, tema, kama,
    
    # Momentum indicators
    rsi, stochastic, williams_r, roc, momentum,
    
    # Volatility indicators
    atr, bollinger_bands, keltner_channels, donchian_channels,
    
    # Trend indicators
    macd, adx, aroon, psar, supertrend,
    
    # Volume indicators
    obv, cmf, mfi, vwap, volume_profile,
    
    # Utility functions
    calculate_multiple, optimize_indicator, calculate_standard_indicators,
    create_custom_indicator
)

# Import compatibility layer for backward compatibility
from .compatibility_layer import (
    get_indicator_manager,
    reset_indicator_manager,
    CompatibilityManager,
    CompatibilityResult,
    # Legacy functions
    calculate_sma,
    calculate_rsi,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_macd
)

# Version info
__version__ = "2.0.0"
__all__ = [
    # Simple indicators (recommended)
    'sma', 'ema', 'wma', 'hma', 'dema', 'tema', 'kama',
    'rsi', 'stochastic', 'williams_r', 'roc', 'momentum',
    'atr', 'bollinger_bands', 'keltner_channels', 'donchian_channels',
    'macd', 'adx', 'aroon', 'psar', 'supertrend',
    'obv', 'cmf', 'mfi', 'vwap', 'volume_profile',
    'calculate_multiple', 'optimize_indicator', 'calculate_standard_indicators',
    'create_custom_indicator',
    
    # Compatibility layer (temporary)
    'get_indicator_manager', 'reset_indicator_manager',
    'CompatibilityManager', 'CompatibilityResult',
    'calculate_sma', 'calculate_rsi', 'calculate_atr',
    'calculate_bollinger_bands', 'calculate_macd'
]

# Deprecation notice for old imports
def __getattr__(name):
    """Handle deprecated imports with warnings."""
    deprecated_modules = {
        'IndicatorManager': 'Use CompatibilityManager or simple_indicators directly',
        'CacheManager': 'Caching is handled internally by VectorBTPro',
        'IndicatorStorage': 'Use VectorBTPro data structures directly',
        'LazyIndicator': 'Use VectorBTPro lazy evaluation features',
        'DaskIntegration': 'Use VectorBTPro parallel processing',
        'indicator_manager': 'Use simple_indicators module',
        'data_storage': 'Use VectorBTPro data structures',
        'dask_integration': 'Use VectorBTPro parallel features'
    }
    
    if name in deprecated_modules:
        import warnings
        warnings.warn(
            f"{name} is deprecated. {deprecated_modules[name]}",
            DeprecationWarning,
            stacklevel=2
        )
        # Return a dummy object to prevent immediate errors
        return type(name, (), {})
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
