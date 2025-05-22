"""
Data Fetching Module

This module contains data fetching functionality using VectorBT Pro's
native data sources, pickle persistence, and intelligent resampling.

Functions:
- fetch_data(): Main data fetching function using VBT with resampling fallback
- fetch_top_symbols(): Volume-based symbol selection with VBT data
- fetch_ohlcv(): Convenience function matching existing API
- update_data(): Update stored data using VBT's native capabilities
- get_resampling_info(): Get resampling metrics and capabilities

Resampling Functions:
- can_resample_from_to(): Check if resampling is possible between timeframes
- resample_data(): Resample VBT data using native capabilities
- validate_resampled_data(): Validate resampled data integrity

For more information, see README_VBT_NATIVE.md
"""

from .data_fetcher import (
    fetch_data,
    fetch_top_symbols,
    fetch_ohlcv,
    update_data,
    get_storage_info,
    get_resampling_info
)

from .storage_resampling import (
    can_resample_from_to,
    resample_ohlcv_for_storage,
    validate_storage_resampled_data,
    TIMEFRAME_HIERARCHY
)

__all__ = [
    'fetch_data',
    'fetch_top_symbols',
    'fetch_ohlcv',
    'update_data',
    'get_storage_info',
    'get_resampling_info',
    # Storage resampling functions
    'can_resample_from_to',
    'resample_ohlcv_for_storage',
    'validate_storage_resampled_data',
    'TIMEFRAME_HIERARCHY'
] 