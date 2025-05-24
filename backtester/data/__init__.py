"""
Data-related modules for the backtester package.

This package contains organized modules for handling exchange data, market data fetching,
cache management, and data storage using VectorBT Pro's native capabilities.

VBT DATA APPROACH:
This package uses VectorBT Pro's built-in data persistence and caching capabilities:
- Preserves complete VBT metadata and functionality
- Uses VBT's native pickle storage for optimal performance
- Leverages VBT's built-in caching and optimization
- Provides seamless OHLCV access via data.get() methods
- Supports VBT's native update() capabilities

All data fetching follows the project guidelines of using VectorBT Pro's native
functionality. API keys are NOT required for OHLCV data fetching operations.
"""

# Exchange configuration functions (informational utilities)
from backtester.data.exchange_config import (
    list_available_exchanges,
    get_exchange_info, 
    get_exchange_timeframes
)

# Exchange settings - no longer needed, VBT handles this natively

# Cache system (organized module for metadata and volume data)
from backtester.data.cache_system import (
    cache_manager,
    data_fetcher
)

# Data fetching and storage
from backtester.data.fetching.data_fetcher_new import (
    fetch_data,
    fetch_top_symbols,
    update_data,
    get_storage_info,
    fetch_ohlcv
)

from backtester.data.storage.data_storage import (
    data_storage,
    DataStorage
)

# BACKWARD COMPATIBILITY ALIASES
# These redirect old function calls to current implementations
def fetch_multi_exchange(symbols, exchange_ids, **kwargs):
    """
    Legacy compatibility function - simplified multi-exchange fetch.
    
    DEPRECATED: Use fetch_data() for each exchange instead.
    """
    results = {}
    for exchange_id in exchange_ids:
        data = fetch_data(symbols, exchange_id, **kwargs)
        if data is not None:
            results[exchange_id] = data
    return results

def fetch_by_market_type(exchange_id, market_type='spot', quote_currency='USDT', limit=20, **kwargs):
    """
    Legacy compatibility function - redirects to volume-based selection.
    
    DEPRECATED: Use fetch_top_symbols() instead.
    """
    return fetch_top_symbols(
        exchange_id=exchange_id,
        quote_currency=quote_currency,
        market_type=market_type,
        limit=limit,
        **kwargs
    )

def fetch_top_symbols_with_auto_cache_update(exchange_id, **kwargs):
    """
    Legacy compatibility function - redirects to current implementation.
    
    DEPRECATED: Use fetch_top_symbols() instead.
    """
    return fetch_top_symbols(exchange_id=exchange_id, **kwargs)

# Legacy storage compatibility
class OptimizedHDFStorage:
    """Legacy compatibility class - redirects to VBT data storage."""
    
    def __init__(self):
        import warnings
        warnings.warn(
            "OptimizedHDFStorage is deprecated. Use DataStorage instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._storage = data_storage
    
    def __getattr__(self, name):
        # Redirect all calls to VBT data storage
        return getattr(self._storage, name)

# Create legacy compatibility instance
optimized_hdf_storage = OptimizedHDFStorage()

# Expose key functionality for easy imports
__all__ = [
    # Exchange configuration
    'list_available_exchanges',
    'get_exchange_info',
    'get_exchange_timeframes',
    
    # Cache system
    'cache_manager',
    'data_fetcher',
    
    # Data functionality
    'fetch_data',
    'fetch_top_symbols', 
    'update_data',
    'get_storage_info',
    'fetch_ohlcv',
    'data_storage',
    'DataStorage',
    
    # Legacy compatibility (DEPRECATED)
    'fetch_multi_exchange', 
    'fetch_by_market_type',
    'fetch_top_symbols_with_auto_cache_update',
    'optimized_hdf_storage',
    'OptimizedHDFStorage',
]
