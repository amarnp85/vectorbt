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


def get_default_fetcher(
    exchange_id: str = "binance", market_type: str = "spot"
) -> CoreDataFetcher:
    """Get or create default fetcher instance."""
    global _default_fetcher
    if _default_fetcher is None or _default_fetcher.exchange_id != exchange_id:
        _default_fetcher = CoreDataFetcher(exchange_id, market_type)
    return _default_fetcher


# Main public functions
def fetch_data(
    symbols: List[str],
    exchange_id: str = "binance",
    timeframe: str = "1d",
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    use_cache: bool = True,
    market_type: str = "spot",
    prefer_resampling: bool = True,
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
        prefer_resampling=prefer_resampling,
    )


def fetch_top_symbols(
    exchange_id: str = "binance",
    quote_currency: str = "USDT",
    market_type: str = "spot",
    limit: int = 10,
    timeframe: str = "1d",
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    use_cache: bool = True,
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
        use_cache=use_cache,
    )


def update_data(
    exchange_id: str,
    timeframe: str,
    symbols: Optional[List[str]] = None,
    market_type: str = "spot",
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
    exchange_id: str = "binance",
    timeframe: str = "1d",
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    use_cache: bool = True,
    market_type: str = "spot",
    prefer_resampling: bool = True,
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
        prefer_resampling=prefer_resampling,
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
        "supported_timeframes": TIMEFRAME_HIERARCHY,
        "resampling_type": "pandas_based",
        "storage_summary": data_storage.get_storage_summary(),
    }


# Enhanced VBT Data Access Helpers
def test_vbt_integration(data: vbt.Data) -> Dict[str, bool]:
    """
    Test VBT data integration points.

    Args:
        data: VBT Data object to test

    Returns:
        Dict with test results for each integration point
    """
    results = {}

    # Test 1: Magnet features (data.open, data.close, etc.)
    try:
        results["magnet_features"] = all(
            [
                hasattr(data, "open") and data.open is not None,
                hasattr(data, "high") and data.high is not None,
                hasattr(data, "low") and data.low is not None,
                hasattr(data, "close") and data.close is not None,
                hasattr(data, "volume") and data.volume is not None,
            ]
        )
    except Exception:
        results["magnet_features"] = False

    # Test 2: get() method for specific features
    try:
        open_data = data.get("Open")
        close_data = data.get("Close")
        results["get_method"] = (
            open_data is not None
            and close_data is not None
            and len(open_data) > 0
            and len(close_data) > 0
        )
    except Exception:
        results["get_method"] = False

    # Test 3: Returns calculation
    try:
        returns = data.returns
        results["returns_calc"] = returns is not None and len(returns) > 0
    except Exception:
        results["returns_calc"] = False

    # Test 4: HLC/OHLC calculations
    try:
        results["hlc_calc"] = hasattr(data, "hlc3") and data.hlc3 is not None
        results["ohlc_calc"] = hasattr(data, "ohlc4") and data.ohlc4 is not None
    except Exception:
        results["hlc_calc"] = False
        results["ohlc_calc"] = False

    # Test 5: run() method for indicators
    try:
        # Test with a simple SMA
        sma_result = data.run("talib:SMA", 14)
        results["run_method"] = sma_result is not None
    except Exception:
        results["run_method"] = False

    # Test 6: Resampling capability
    try:
        # Only test if we have intraday data
        if hasattr(data, "wrapper") and data.wrapper.freq:
            freq_str = str(data.wrapper.freq)
            if any(x in freq_str for x in ["H", "T", "min"]):
                resampled = data.resample("1D")
                results["resample_method"] = resampled is not None
            else:
                results["resample_method"] = True  # Daily or lower freq
        else:
            results["resample_method"] = True  # Can't determine freq
    except Exception:
        results["resample_method"] = False

    # Test 7: Symbol selection (for multi-symbol data)
    try:
        if len(data.symbols) > 1:
            first_symbol = data.symbols[0]
            selected = data.select(first_symbol)
            results["symbol_selection"] = selected is not None
        else:
            results["symbol_selection"] = True  # Single symbol, no selection needed
    except Exception:
        results["symbol_selection"] = False

    return results


def create_enhanced_data(data: vbt.Data, add_technicals: bool = False) -> vbt.Data:
    """
    Create an enhanced VBT Data object with additional features.

    Args:
        data: Base VBT Data object
        add_technicals: Whether to add technical indicators

    Returns:
        Enhanced VBT Data object
    """
    if add_technicals and hasattr(data, "close"):
        # Add some basic technical indicators
        try:
            # These would be accessible as data attributes
            data._sma_20 = data.run("talib:SMA", 20)
            data._sma_50 = data.run("talib:SMA", 50)
            data._rsi_14 = data.run("talib:RSI", 14)
        except Exception as e:
            logger.debug(f"Could not add technical indicators: {e}")

    return data


def quick_fetch(
    symbol: str, days: int = 365, timeframe: str = "1d", exchange_id: str = "binance"
) -> Optional[vbt.Data]:
    """
    Quick single-symbol fetch for prototyping and strategy development.

    Args:
        symbol: Trading symbol (e.g., 'BTC/USDT')
        days: Number of days to look back (default: 365)
        timeframe: Timeframe string (default: '1d')
        exchange_id: Exchange identifier (default: 'binance')

    Returns:
        VBT Data object or None

    Examples:
        >>> # Quick fetch last year of BTC data
        >>> data = quick_fetch('BTC/USDT')
        >>> close_price = data.close
        >>> rsi = data.run('talib:RSI', 14)
    """
    import pandas as pd

    # Calculate start date
    end_date = pd.Timestamp.now(tz="UTC")
    start_date = end_date - pd.Timedelta(days=days)

    return fetch_data(
        symbols=[symbol],
        exchange_id=exchange_id,
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,
    )


def load_latest(
    exchange_id: str = "binance",
    timeframe: str = "1d",
    symbols: Optional[List[str]] = None,
    market_type: str = "spot",
) -> Optional[vbt.Data]:
    """
    Load latest cached data for quick analysis.

    Args:
        exchange_id: Exchange identifier
        timeframe: Timeframe to load
        symbols: Specific symbols to load (None for all)
        market_type: Market type

    Returns:
        VBT Data object with cached data or None

    Examples:
        >>> # Load all cached daily data
        >>> data = load_latest()
        >>>
        >>> # Load specific symbols
        >>> data = load_latest(symbols=['BTC/USDT', 'ETH/USDT'])
    """
    from ..storage.data_storage import data_storage

    # Load from storage
    data = data_storage.load_data(exchange_id, timeframe, symbols, market_type)

    if data is not None:
        logger.info(
            f"Loaded {len(data.symbols)} symbols from cache: "
            f"{exchange_id} {timeframe} {market_type}"
        )

        # Run integration tests in debug mode
        if logger.isEnabledFor(logging.DEBUG):
            test_results = test_vbt_integration(data)
            logger.debug(f"VBT integration test results: {test_results}")

    return data
