#!/usr/bin/env python3
"""
Simplified Data Module Interface

This module provides a clear, simple interface to the data fetching functionality.
It serves as the primary entry point for all data operations in the backtester.

Main Functions:
    - fetch_data(): Get historical OHLCV data
    - fetch_top_symbols(): Get top symbols by volume
    - update_data(): Update cached data to latest
    - quick_fetch(): Simple single-symbol fetch
    - load_cached(): Load previously cached data

All functions automatically handle:
    - Caching (stores data locally for reuse)
    - Resampling (creates higher timeframes from lower ones)
    - Exchange fetching (only when necessary)
    - Metadata updates (symbol lists, volumes, etc.)
"""

from typing import List, Optional, Union, Dict, Any
import vectorbtpro as vbt
from .fetching.data_fetcher_new import (
    fetch_data as _fetch_data,
    fetch_top_symbols as _fetch_top_symbols,
    update_data as _update_data,
    quick_fetch as _quick_fetch,
    load_latest as _load_latest,
    get_storage_info as _get_storage_info,
)


def fetch_data(
    symbols: List[str],
    exchange: str = "binance",
    timeframe: str = "1d",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    market_type: str = "spot",
) -> Optional[vbt.Data]:
    """
    Fetch historical OHLCV data for given symbols.
    
    This is the MAIN ENTRY POINT for getting data. It automatically:
    1. Checks local cache first
    2. Tries to resample from lower timeframes if available
    3. Fetches from exchange only when necessary
    4. Updates metadata (symbol lists, volumes) as needed
    
    Args:
        symbols: List of trading pairs (e.g., ['BTC/USDT', 'ETH/USDT'])
        exchange: Exchange name ('binance', 'bybit', 'hyperliquid')
        timeframe: Candle period ('1m', '5m', '15m', '1h', '4h', '1d')
        start_date: Start date string (e.g., '2024-01-01')
        end_date: End date string (defaults to now)
        market_type: 'spot' or 'swap' (futures/perpetuals)
    
    Returns:
        VectorBT Data object with OHLCV data, or None if fetch failed
    
    Examples:
        >>> # Get daily BTC and ETH data for the last year
        >>> data = fetch_data(['BTC/USDT', 'ETH/USDT'])
        >>> 
        >>> # Get hourly data from a specific date
        >>> data = fetch_data(
        ...     ['BTC/USDT'], 
        ...     timeframe='1h',
        ...     start_date='2024-01-01'
        ... )
        >>> 
        >>> # Access the data
        >>> close_prices = data.close
        >>> returns = data.returns
        >>> rsi = data.run('talib:RSI', 14)
    """
    return _fetch_data(
        symbols=symbols,
        exchange_id=exchange,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
        market_type=market_type,
        prefer_resampling=True,
    )


def fetch_top_symbols(
    exchange: str = "binance",
    quote_currency: str = "USDT",
    limit: int = 10,
    timeframe: str = "1d",
    market_type: str = "spot",
) -> Optional[vbt.Data]:
    """
    Fetch data for the top symbols by 24h volume.
    
    Automatically discovers and fetches the most traded symbols.
    
    Args:
        exchange: Exchange name
        quote_currency: Quote currency filter (e.g., 'USDT', 'BTC')
        limit: Number of top symbols to fetch
        timeframe: Candle period
        market_type: 'spot' or 'swap'
    
    Returns:
        VectorBT Data object with top symbols
    
    Example:
        >>> # Get top 20 USDT pairs
        >>> data = fetch_top_symbols(limit=20)
        >>> print(f"Fetched symbols: {data.symbols}")
    """
    return _fetch_top_symbols(
        exchange_id=exchange,
        quote_currency=quote_currency,
        market_type=market_type,
        limit=limit,
        timeframe=timeframe,
        use_cache=True,
    )


def update_data(
    exchange: str = "binance",
    timeframe: str = "1d",
    symbols: Optional[List[str]] = None,
    market_type: str = "spot",
) -> bool:
    """
    Update cached data to the latest available.
    
    Args:
        exchange: Exchange name
        timeframe: Timeframe to update
        symbols: Specific symbols to update (None = all cached)
        market_type: 'spot' or 'swap'
    
    Returns:
        True if update successful
    
    Example:
        >>> # Update all cached daily data
        >>> success = update_data()
        >>> 
        >>> # Update specific symbols
        >>> success = update_data(
        ...     timeframe='1h',
        ...     symbols=['BTC/USDT', 'ETH/USDT']
        ... )
    """
    return _update_data(
        exchange_id=exchange,
        timeframe=timeframe,
        symbols=symbols,
        market_type=market_type,
    )


def quick_fetch(
    symbol: str,
    days: int = 365,
    timeframe: str = "1d",
    exchange: str = "binance",
) -> Optional[vbt.Data]:
    """
    Quick single-symbol fetch for prototyping.
    
    Simplified interface for getting data for one symbol.
    
    Args:
        symbol: Trading pair (e.g., 'BTC/USDT')
        days: Number of days to look back
        timeframe: Candle period
        exchange: Exchange name
    
    Returns:
        VectorBT Data object
    
    Example:
        >>> # Get last 30 days of BTC hourly data
        >>> data = quick_fetch('BTC/USDT', days=30, timeframe='1h')
        >>> close = data.close
    """
    return _quick_fetch(
        symbol=symbol,
        days=days,
        timeframe=timeframe,
        exchange_id=exchange,
    )


def load_cached(
    exchange: str = "binance",
    timeframe: str = "1d",
    symbols: Optional[List[str]] = None,
    market_type: str = "spot",
) -> Optional[vbt.Data]:
    """
    Load previously cached data without fetching.
    
    Useful for offline analysis or when you know data is already cached.
    
    Args:
        exchange: Exchange name
        timeframe: Timeframe to load
        symbols: Specific symbols (None = all cached)
        market_type: 'spot' or 'swap'
    
    Returns:
        VectorBT Data object from cache
    
    Example:
        >>> # Load all cached daily data
        >>> data = load_cached()
        >>> 
        >>> # Load specific symbols
        >>> data = load_cached(symbols=['BTC/USDT', 'ETH/USDT'])
    """
    return _load_latest(
        exchange_id=exchange,
        timeframe=timeframe,
        symbols=symbols,
        market_type=market_type,
    )


def get_cache_info() -> Dict[str, Any]:
    """
    Get information about cached data.
    
    Returns:
        Dictionary with cache statistics
    
    Example:
        >>> info = get_cache_info()
        >>> print(f"Cached exchanges: {info['exchanges']}")
        >>> print(f"Total symbols: {info['total_symbols']}")
    """
    return _get_storage_info()


# Convenience aliases for common operations
get_data = fetch_data  # Alias for fetch_data
get_top_symbols = fetch_top_symbols  # Alias for fetch_top_symbols


if __name__ == "__main__":
    # Example usage
    print("Data Module Simple Interface")
    print("=" * 50)
    
    # Show cache info
    info = get_cache_info()
    print(f"\nCache Info: {info}")
    
    # Example fetch
    print("\nExample: Fetching BTC/USDT daily data...")
    data = quick_fetch('BTC/USDT', days=30)
    if data:
        print(f"Fetched {len(data)} candles")
        print(f"Date range: {data.index[0]} to {data.index[-1]}")
        print(f"Latest close: ${data.close.iloc[-1]:,.2f}") 