"""
Data Module - Simplified Interface for VectorBT Pro Data Management

This module provides a clean, simple interface for all data operations in the backtester.
It automatically handles caching, resampling, and exchange fetching.

MAIN ENTRY POINTS:
    - fetch_data(): Get historical OHLCV data (primary interface)
    - fetch_top_symbols(): Get top symbols by volume
    - update_data(): Update cached data to latest
    - quick_fetch(): Simple single-symbol fetch
    - load_cached(): Load previously cached data

Example Usage:
    >>> from backtester.data import fetch_data, quick_fetch
    >>> 
    >>> # Quick single symbol fetch
    >>> btc_data = quick_fetch('BTC/USDT', days=30)
    >>> 
    >>> # Multi-symbol fetch with options
    >>> data = fetch_data(
    ...     symbols=['BTC/USDT', 'ETH/USDT'],
    ...     timeframe='1h',
    ...     start_date='2024-01-01'
    ... )
    >>> 
    >>> # Access VBT features
    >>> close_prices = data.close
    >>> returns = data.returns
    >>> rsi = data.run('talib:RSI', 14)

The module uses VectorBT Pro's native data persistence and caching capabilities,
preserving complete VBT metadata and functionality for optimal performance.
"""

# =============================================================================
# PRIMARY INTERFACE - Use these functions for all data operations
# =============================================================================
from backtester.data.simple_interface import (
    fetch_data,          # Main data fetching function
    fetch_top_symbols,   # Get top symbols by volume
    update_data,         # Update cached data
    quick_fetch,         # Simple single-symbol fetch
    load_cached,         # Load from cache without fetching
    get_cache_info,      # Get cache statistics
    # Aliases
    get_data,           # Alias for fetch_data
    get_top_symbols,    # Alias for fetch_top_symbols
)

# =============================================================================
# EXCHANGE UTILITIES - For exchange information and configuration
# =============================================================================
from backtester.data.exchange_config import (
    list_available_exchanges,
    get_exchange_info,
    get_exchange_timeframes,
)

# =============================================================================
# ADVANCED FEATURES - For specialized use cases
# =============================================================================
# Direct access to storage system
from backtester.data.storage.data_storage import data_storage, DataStorage

# Cache management utilities
from backtester.data.cache_system import cache_manager, data_fetcher

# Low-level fetching functions (use simple_interface functions instead)
from backtester.data.fetching.data_fetcher_new import (
    get_storage_info,
    fetch_ohlcv,  # Alias for fetch_data
)

# =============================================================================
# LEGACY COMPATIBILITY - Deprecated, use primary interface instead
# =============================================================================
def fetch_multi_exchange(symbols, exchange_ids, **kwargs):
    """
    DEPRECATED: Use fetch_data() for each exchange instead.
    
    Legacy compatibility function for multi-exchange fetching.
    """
    import warnings
    warnings.warn(
        "fetch_multi_exchange is deprecated. Use fetch_data() for each exchange.",
        DeprecationWarning,
        stacklevel=2
    )
    results = {}
    for exchange_id in exchange_ids:
        data = fetch_data(symbols, exchange_id, **kwargs)
        if data is not None:
            results[exchange_id] = data
    return results


def fetch_by_market_type(
    exchange_id, market_type="spot", quote_currency="USDT", limit=20, **kwargs
):
    """
    DEPRECATED: Use fetch_top_symbols() instead.
    
    Legacy compatibility function for market type filtering.
    """
    import warnings
    warnings.warn(
        "fetch_by_market_type is deprecated. Use fetch_top_symbols() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return fetch_top_symbols(
        exchange=exchange_id,
        quote_currency=quote_currency,
        market_type=market_type,
        limit=limit,
    )


def fetch_top_symbols_with_auto_cache_update(exchange_id, **kwargs):
    """
    DEPRECATED: Use fetch_top_symbols() instead.
    
    Legacy compatibility function.
    """
    import warnings
    warnings.warn(
        "fetch_top_symbols_with_auto_cache_update is deprecated. Use fetch_top_symbols().",
        DeprecationWarning,
        stacklevel=2
    )
    return fetch_top_symbols(exchange=exchange_id, **kwargs)


# Legacy storage compatibility removed - use DataStorage directly

# =============================================================================
# CLI TOOLS - Command-line utilities
# =============================================================================
try:
    from backtester.data import cli
except ImportError:
    cli = None

# =============================================================================
# PUBLIC API
# =============================================================================
__all__ = [
    # Primary Interface (USE THESE)
    "fetch_data",
    "fetch_top_symbols", 
    "update_data",
    "quick_fetch",
    "load_cached",
    "get_cache_info",
    "get_data",  # Alias
    "get_top_symbols",  # Alias
    
    # Exchange utilities
    "list_available_exchanges",
    "get_exchange_info",
    "get_exchange_timeframes",
    
    # Advanced features
    "data_storage",
    "DataStorage",
    "cache_manager",
    "data_fetcher",
    
    # CLI tools
    "cli",
    
    # Legacy (avoid using)
    "fetch_multi_exchange",
    "fetch_by_market_type",
    "fetch_top_symbols_with_auto_cache_update",
    "get_storage_info",
    "fetch_ohlcv",
]
