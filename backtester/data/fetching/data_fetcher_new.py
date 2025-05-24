#!/usr/bin/env python3
"""Data Fetcher Module - Simplified interface using refactored core.

This module provides the public API for data fetching, delegating to
the refactored core components for clean separation of concerns.
"""

import logging
from typing import List, Optional, Union, Dict, Any
import vectorbtpro as vbt
from .core import DataFetcher as CoreDataFetcher

logger = logging.getLogger(__name__)

# Module-level instance for backward compatibility
_default_fetcher = None


def get_default_fetcher(exchange_id: str = 'binance', market_type: str = 'spot') -> CoreDataFetcher:
    """Get or create default fetcher instance."""
    global _default_fetcher
    if _default_fetcher is None or _default_fetcher.exchange_id != exchange_id:
        _default_fetcher = CoreDataFetcher(exchange_id, market_type)
    return _default_fetcher


# Main public functions
def fetch_data(
    symbols: List[str],
    exchange_id: str = 'binance',
    timeframe: str = '1d',
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    use_cache: bool = True,
    market_type: str = 'spot',
    prefer_resampling: bool = True
) -> Optional[vbt.Data]:
    """
    Fetch OHLCV data with intelligent source selection.
    
    This function provides a clean interface for fetching financial data,
    automatically handling caching, resampling, and exchange fetching.
    
    Args:
        symbols: List of trading symbols (e.g., ['BTC/USDT', 'ETH/USDT'])
        exchange_id: Exchange identifier (default: 'binance')
        timeframe: Timeframe string (e.g., '1d', '1h', '5m')
        start_date: Start date (None for inception)
        end_date: End date (None for latest)
        use_cache: Whether to use caching (default: True)
        market_type: Market type ('spot' or 'swap')
        prefer_resampling: Whether to try resampling before exchange fetch
        
    Returns:
        VBT Data object containing OHLCV data or None if fetch failed
        
    Examples:
        >>> # Fetch daily data for BTC and ETH
        >>> data = fetch_data(['BTC/USDT', 'ETH/USDT'], timeframe='1d')
        
        >>> # Fetch from specific date without cache
        >>> data = fetch_data(
        ...     ['BTC/USDT'],
        ...     start_date='2024-01-01',
        ...     use_cache=False
        ... )
    """
    fetcher = get_default_fetcher(exchange_id, market_type)
    return fetcher.fetch_data(
        symbols=symbols,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        prefer_resampling=prefer_resampling
    )


def fetch_top_symbols(
    exchange_id: str = 'binance',
    quote_currency: str = 'USDT',
    market_type: str = 'spot',
    limit: int = 10,
    timeframe: str = '1d',
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    use_cache: bool = True
) -> Optional[vbt.Data]:
    """
    Fetch data for top symbols by volume.
    
    Args:
        exchange_id: Exchange identifier
        quote_currency: Quote currency filter (e.g., 'USDT')
        market_type: Market type ('spot' or 'swap')
        limit: Number of top symbols to fetch
        timeframe: Timeframe string
        start_date: Start date
        end_date: End date
        use_cache: Whether to use caching
        
    Returns:
        VBT Data object with top symbols or None
        
    Examples:
        >>> # Get top 10 USDT pairs by volume
        >>> data = fetch_top_symbols(limit=10)
        
        >>> # Get top 5 BTC pairs
        >>> data = fetch_top_symbols(quote_currency='BTC', limit=5)
    """
    fetcher = get_default_fetcher(exchange_id, market_type)
    return fetcher.fetch_top_symbols(
        quote_currency=quote_currency,
        limit=limit,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache
    )


def update_data(
    exchange_id: str,
    timeframe: str,
    symbols: Optional[List[str]] = None,
    market_type: str = 'spot'
) -> bool:
    """
    Update cached data to latest available.
    
    Args:
        exchange_id: Exchange identifier
        timeframe: Timeframe to update
        symbols: Specific symbols to update (None for all cached)
        market_type: Market type
        
    Returns:
        True if update successful
        
    Examples:
        >>> # Update all cached daily data
        >>> success = update_data('binance', '1d')
        
        >>> # Update specific symbols
        >>> success = update_data('binance', '1h', ['BTC/USDT', 'ETH/USDT'])
    """
    fetcher = get_default_fetcher(exchange_id, market_type)
    return fetcher.update_data(timeframe, symbols)


# Backward compatibility aliases
def fetch_ohlcv(
    symbols: List[str],
    exchange_id: str = 'binance',
    timeframe: str = '1d',
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    use_cache: bool = True,
    market_type: str = 'spot',
    prefer_resampling: bool = True
) -> Optional[vbt.Data]:
    """Backward compatibility alias for fetch_data."""
    return fetch_data(
        symbols=symbols,
        exchange_id=exchange_id,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        market_type=market_type,
        prefer_resampling=prefer_resampling
    )


# Storage info functions
def get_storage_info() -> Dict[str, Any]:
    """Get information about data storage."""
    from ..storage.data_storage import data_storage
    return data_storage.get_storage_summary()


def get_resampling_info() -> Dict[str, Any]:
    """Get information about resampling capabilities."""
    from .core.resampler import TIMEFRAME_HIERARCHY
    from ..storage.data_storage import data_storage
    
    return {
        'supported_timeframes': TIMEFRAME_HIERARCHY,
        'resampling_type': 'pandas_based',
        'storage_summary': data_storage.get_storage_summary()
    } 