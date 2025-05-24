#!/usr/bin/env python3
"""Exchange Fetcher - Handle direct exchange API data fetching.

This module manages fetching data directly from exchanges using VBT's
native exchange integration.
"""

import logging
import time
from typing import Dict, List, Optional, Union, TYPE_CHECKING
import vectorbtpro as vbt
import pandas as pd

if TYPE_CHECKING:
    from .fetch_logger import FetchLogger

logger = logging.getLogger(__name__)


class ExchangeFetcher:
    """Fetch data directly from exchanges."""
    
    # Supported exchanges
    SUPPORTED_EXCHANGES = ['binance', 'bybit', 'okx', 'kucoin', 'coinbase']
    
    def __init__(self, exchange_id: str):
        """Initialize with exchange ID."""
        self.exchange_id = exchange_id.lower()
        
        if self.exchange_id not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"Exchange {exchange_id} not supported")
    
    def fetch_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        start_dates: Dict[str, str],
        end_date: Optional[Union[str, vbt.timestamp]] = None,
        parallel: bool = True,
        fetch_logger: Optional['FetchLogger'] = None
    ) -> Optional[vbt.Data]:
        """
        Fetch OHLCV data for symbols from exchange.
        
        Args:
            symbols: List of symbols to fetch
            timeframe: Timeframe string
            start_dates: Dict mapping symbols to their start dates
            end_date: End date for fetching
            parallel: Whether to use parallel fetching
            fetch_logger: Optional logger for detailed progress tracking
            
        Returns:
            VBT Data object or None if failed
        """
        fetch_start_time = time.time()
        
        try:
            if not symbols:
                logger.error("No symbols to fetch")
                if fetch_logger:
                    fetch_logger.log_exchange_result(False)
                return None
            
            # Log exchange fetch start
            if fetch_logger:
                fetch_logger.log_exchange_start(symbols)
            
            # Determine common start date
            start_date = self._get_common_start_date(symbols, start_dates)
            
            # Calculate safe end date to avoid incomplete candles
            safe_end_date = self._get_safe_end_date(end_date, timeframe)
            
            logger.info(
                f"Fetching {len(symbols)} symbols from {self.exchange_id.upper()}: "
                f"{start_date or 'inception'} to {safe_end_date or 'latest'} ({timeframe})"
            )
            
            # Log symbol details
            if len(symbols) <= 10:
                logger.info(f"Symbols: {', '.join(symbols)}")
            else:
                logger.info(f"Symbols: {', '.join(symbols[:5])}... (+{len(symbols)-5} more)")
            
            # Prepare fetch kwargs
            fetch_kwargs = {
                'show_progress': False  # We'll handle progress ourselves
            }
            
            # Enable parallel fetching for multiple symbols
            if parallel and len(symbols) > 1:
                fetch_kwargs['execute_kwargs'] = {'engine': 'threadpool'}
                logger.debug("Using parallel fetching")
            else:
                logger.debug("Using sequential fetching")
            
            # Log progress start
            if fetch_logger:
                fetch_logger.log_symbol_progress(0, len(symbols))
            
            # Fetch using VBT's CCXTData
            logger.debug(f"Starting VBT fetch for {len(symbols)} symbols...")
            data = vbt.CCXTData.pull(
                symbols,
                exchange=self.exchange_id,
                timeframe=timeframe,
                start=start_date,
                end=safe_end_date,
                **fetch_kwargs
            )
            
            fetch_time = time.time() - fetch_start_time
            
            if data is not None:
                # Analyze results per symbol
                successful_symbols = list(data.symbols) if hasattr(data, 'symbols') else []
                failed_symbols = set(symbols) - set(successful_symbols)
                
                logger.info(
                    f"Exchange fetch completed: {len(successful_symbols)}/{len(symbols)} symbols "
                    f"in {fetch_time:.1f}s"
                )
                
                # Log detailed per-symbol results
                if fetch_logger:
                    self._log_symbol_results(
                        fetch_logger, data, symbols, successful_symbols, failed_symbols
                    )
                    fetch_logger.log_exchange_result(
                        True, len(successful_symbols), fetch_time
                    )
                
                # Log any failures
                if failed_symbols:
                    logger.warning(f"Failed to fetch {len(failed_symbols)} symbols: {list(failed_symbols)}")
                    
                return data
            else:
                logger.error("Exchange fetch returned None")
                if fetch_logger:
                    fetch_logger.log_exchange_result(False, fetch_time=fetch_time)
                return None
                
        except Exception as e:
            fetch_time = time.time() - fetch_start_time
            logger.error(f"Exchange fetch failed after {fetch_time:.1f}s: {e}")
            if fetch_logger:
                fetch_logger.log_exchange_result(False, fetch_time=fetch_time)
            return None
    
    def _log_symbol_results(
        self,
        fetch_logger: 'FetchLogger',
        data: vbt.Data,
        requested_symbols: List[str],
        successful_symbols: List[str],
        failed_symbols: set
    ):
        """Log detailed results for each symbol."""
        try:
            # Log successful symbols with data point counts
            for symbol in successful_symbols:
                data_points = None
                
                # Try to get data point count
                try:
                    from .vbt_data_handler import VBTDataHandler
                    symbol_data = VBTDataHandler.get_symbol_data(data, symbol, 'close')
                    if symbol_data is not None:
                        data_points = len(symbol_data)
                except Exception:
                    pass
                
                fetch_logger.log_symbol_success(symbol, data_points)
            
            # Log failed symbols
            for symbol in failed_symbols:
                fetch_logger.log_symbol_failure(symbol, "No data returned")
                
            # Final progress update
            fetch_logger.log_symbol_progress(len(successful_symbols), len(requested_symbols))
            
        except Exception as e:
            logger.debug(f"Error logging symbol results: {e}")
    
    def _get_common_start_date(
        self,
        symbols: List[str],
        start_dates: Dict[str, str]
    ) -> Optional[str]:
        """Get the common start date for fetching."""
        # Filter to only symbols being fetched
        symbol_dates = [
            start_dates[symbol] 
            for symbol in symbols 
            if symbol in start_dates and start_dates[symbol] is not None
        ]
        
        if not symbol_dates:
            return None
            
        # Use earliest date to ensure we get all data
        earliest_date = min(symbol_dates)
        logger.debug(f"Using common start date: {earliest_date}")
        return earliest_date
    
    def _get_safe_end_date(
        self,
        end_date: Optional[Union[str, vbt.timestamp]],
        timeframe: str
    ) -> Optional[vbt.timestamp]:
        """Get a safe end date that avoids incomplete candles."""
        try:
            from .freshness_checker import FreshnessChecker
            
            # If no end date specified, use current time logic
            if end_date is None or (isinstance(end_date, str) and end_date.lower() == "now"):
                current_time = vbt.utc_timestamp()
                # Convert to pandas timestamp for calculations
                current_pd = pd.Timestamp(current_time)
                
                # Get safe end time (end of latest complete candle)
                safe_end_pd = FreshnessChecker.get_safe_end_time_for_fetch(current_pd, timeframe)
                
                # Convert back to vbt timestamp
                return vbt.timestamp(safe_end_pd)
            else:
                # Use provided end date as-is
                if isinstance(end_date, str):
                    return vbt.timestamp(end_date, tz='UTC')
                else:
                    return end_date
                    
        except Exception as e:
            logger.debug(f"Error calculating safe end date: {e}")
            # Fallback to original end_date
            return end_date
    
    def fetch_missing_period(
        self,
        symbols: List[str],
        start_date: str,
        end_date: Optional[Union[str, vbt.timestamp]] = None
    ) -> Optional[vbt.Data]:
        """
        Fetch data for a specific period (used for updates).
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date string
            end_date: End date
            
        Returns:
            VBT Data object or None
        """
        try:
            logger.info(
                f"Fetching missing period for {len(symbols)} symbols: "
                f"{start_date} to {end_date or 'latest'}"
            )
            
            # Create start_dates dict with same date for all symbols
            start_dates = {symbol: start_date for symbol in symbols}
            
            return self.fetch_symbols(
                symbols=symbols,
                timeframe='1d',  # Default, will be overridden by caller
                start_dates=start_dates,
                end_date=end_date,
                parallel=True
            )
            
        except Exception as e:
            logger.error(f"Error fetching missing period: {e}")
            return None 