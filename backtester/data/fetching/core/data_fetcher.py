#!/usr/bin/env python3
"""Data Fetcher - Main orchestrator for data fetching operations.

This module provides the main interface for fetching financial data,
coordinating between cache, resampling, and exchange sources.
"""

import logging
from typing import List, Optional, Union
import vectorbtpro as vbt
from .cache_handler import CacheHandler
from .symbol_resolver import SymbolResolver
from .exchange_fetcher import ExchangeFetcher
from .data_merger import DataMerger
from .fetch_logger import FetchLogger
import pandas as pd

logger = logging.getLogger(__name__)


class DataFetcher:
    """Main data fetching orchestrator."""

    def __init__(self, exchange_id: str = "binance", market_type: str = "spot"):
        """Initialize data fetcher."""
        self.exchange_id = exchange_id
        self.market_type = market_type
        self.symbol_resolver = SymbolResolver(exchange_id)
        self.exchange_fetcher = ExchangeFetcher(exchange_id)
        self.merger = DataMerger()

    def fetch_data(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        start_date: Optional[Union[str, vbt.timestamp]] = None,
        end_date: Optional[Union[str, vbt.timestamp]] = None,
        use_cache: bool = True,
        prefer_resampling: bool = True,
    ) -> Optional[vbt.Data]:
        """
        Fetch data with intelligent source selection.

        Strategy:
        1. Filter symbols (blacklist, availability)
        2. Try cache (if enabled)
        3. Try resampling from lower timeframes (if preferred)
        4. Fetch from exchange (final fallback)
        5. Save to cache (if enabled)

        Args:
            symbols: List of trading symbols
            timeframe: Timeframe string (e.g., '1d', '1h')
            start_date: Start date (None for inception)
            end_date: End date (None for latest)
            use_cache: Whether to use caching
            prefer_resampling: Whether to try resampling first

        Returns:
            VBT Data object or None
        """
        # Initialize logger
        fetch_logger = FetchLogger(symbols, self.exchange_id, timeframe)
        fetch_logger.log_start(start_date, end_date)

        try:
            # Step 1: Filter symbols
            symbols = self.symbol_resolver.filter_symbols(symbols)
            fetch_logger.log_filter_result(symbols)

            if not symbols:
                logger.warning("No symbols remaining after filtering")
                fetch_logger.log_final_result(None)
                return None

            # Step 2: Get optimal start dates
            symbol_start_dates = self.symbol_resolver.get_optimal_start_dates(
                symbols, start_date
            )

            # Initialize cache handler for this operation
            cache_handler = CacheHandler(self.exchange_id, timeframe, self.market_type)

            # Step 3: Try cache
            if use_cache:
                cached_data, cached_symbols, missing_symbols, stale_symbols = (
                    cache_handler.try_load_cached(symbols, end_date, require_fresh=True)
                )

                if cached_data and not missing_symbols and not stale_symbols:
                    # All symbols found and fresh - this is a successful completion
                    fetch_logger.log_cache_result("hit", {"symbols": len(symbols)})
                    result_data = cached_data.select_symbols(symbols)
                    fetch_logger.log_final_result(result_data)
                    logger.info(
                        f"All requested symbols found in fresh cache - no fetch needed"
                    )
                    return result_data

                elif cached_data and (missing_symbols or stale_symbols):
                    # Partial cache hit
                    fetch_logger.log_cache_result(
                        "partial",
                        {
                            "available": len(cached_symbols),
                            "missing": len(missing_symbols),
                            "stale": len(stale_symbols),
                            "requested": len(symbols),
                            "cached_symbols": cached_symbols,
                            "missing_symbols": missing_symbols,
                            "stale_symbols": stale_symbols,
                        },
                    )

                    # Handle stale symbols first - try to update them
                    updated_stale_data = None
                    if stale_symbols:
                        updated_stale_data = self._update_stale_symbols(
                            cached_data,
                            list(stale_symbols),
                            timeframe,
                            end_date,
                            cache_handler,
                            prefer_resampling,
                            fetch_logger,
                        )

                    # Fetch missing symbols
                    missing_data = None
                    if missing_symbols:
                        missing_data = self._fetch_missing_symbols(
                            list(missing_symbols),
                            timeframe,
                            symbol_start_dates,
                            end_date,
                            cache_handler,
                            prefer_resampling,
                            fetch_logger,
                        )

                    # Merge all data
                    if updated_stale_data or missing_data:
                        # Start with fresh cached data
                        final_symbols = cached_symbols.copy()
                        merged = (
                            cached_data.select_symbols(list(cached_symbols))
                            if cached_symbols
                            else None
                        )

                        # Add updated stale data
                        if updated_stale_data:
                            final_symbols.update(stale_symbols)
                            if merged:
                                merged = self.merger.merge_symbol_sets(
                                    merged,
                                    cached_symbols,
                                    updated_stale_data,
                                    stale_symbols,
                                )
                            else:
                                merged = updated_stale_data

                        # Add new missing data
                        if missing_data:
                            final_symbols.update(missing_symbols)
                            if merged:
                                merged = self.merger.merge_symbol_sets(
                                    merged,
                                    final_symbols - missing_symbols,
                                    missing_data,
                                    missing_symbols,
                                )
                            else:
                                merged = missing_data

                        if merged and use_cache:
                            cache_handler.save_data(merged)

                        fetch_logger.log_final_result(merged, saved=use_cache)
                        return merged
                    else:
                        # Failed to fetch missing/update stale - return cached only
                        result = cached_data.select_symbols(list(cached_symbols))
                        fetch_logger.log_final_result(result)
                        return result
                else:
                    # Cache miss
                    fetch_logger.log_cache_result("miss", {})

            # Step 4: Try resampling (if no cache or cache miss)
            if prefer_resampling:
                resampled = cache_handler.try_load_from_lower_timeframe(
                    symbols, end_date, require_fresh=True
                )

                if resampled:
                    fetch_logger.log_resample_result(True)

                    if use_cache:
                        cache_handler.save_data(resampled)

                    fetch_logger.log_final_result(resampled, saved=use_cache)
                    return resampled
                else:
                    fetch_logger.log_resample_result(False)

            # Step 5: Fetch from exchange
            exchange_data = self._fetch_from_exchange(
                symbols, timeframe, symbol_start_dates, end_date, fetch_logger
            )

            if exchange_data:
                fetch_logger.log_exchange_result(True, len(exchange_data.symbols))

                if use_cache:
                    cache_handler.save_data(exchange_data)

                fetch_logger.log_final_result(exchange_data, saved=use_cache)
                return exchange_data
            else:
                fetch_logger.log_exchange_result(False)
                fetch_logger.log_final_result(None)
                return None

        except Exception as e:
            logger.error(f"Error in fetch_data: {e}")
            fetch_logger.log_final_result(None)
            return None

    def _update_stale_symbols(
        self,
        cached_data: vbt.Data,
        stale_symbols: List[str],
        timeframe: str,
        end_date: Optional[Union[str, vbt.timestamp]],
        cache_handler: CacheHandler,
        prefer_resampling: bool,
        fetch_logger: FetchLogger,
    ) -> Optional[vbt.Data]:
        """Update stale symbols by fetching only recent missing data."""
        try:
            # Get the latest timestamp from cached data for these symbols
            from .vbt_data_handler import VBTDataHandler
            from datetime import timedelta

            # Find the earliest latest timestamp among stale symbols
            latest_timestamps = {}
            for symbol in stale_symbols:
                try:
                    symbol_data = VBTDataHandler.get_symbol_data(
                        cached_data, symbol, "close"
                    )
                    if symbol_data is not None and len(symbol_data) > 0:
                        latest_timestamps[symbol] = symbol_data.index[-1]
                except Exception:
                    pass

            if not latest_timestamps:
                # Can't determine timestamps, fall back to full fetch
                return self._fetch_missing_symbols(
                    stale_symbols,
                    timeframe,
                    {},
                    end_date,
                    cache_handler,
                    prefer_resampling,
                    fetch_logger,
                )

            # Calculate update start date (next period after latest data)
            # Add one period to avoid duplicate data
            tf_minutes = cache_handler.freshness_checker.parse_timeframe_minutes(
                timeframe
            )
            period_delta = timedelta(minutes=tf_minutes)

            update_start_dates = {}
            for symbol in stale_symbols:
                if symbol in latest_timestamps:
                    next_period = latest_timestamps[symbol] + period_delta
                    update_start_dates[symbol] = next_period.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

            logger.debug(
                f"Updating {len(stale_symbols)} stale symbols from recent timestamps"
            )

            # First try resampling if preferred
            if prefer_resampling:
                resampled = cache_handler.try_load_from_lower_timeframe(
                    stale_symbols, end_date, require_fresh=True
                )
                if resampled:
                    # Extract only the new data we need
                    new_data_dict = {}
                    for symbol in stale_symbols:
                        if symbol in latest_timestamps:
                            symbol_data = VBTDataHandler.get_symbol_data(
                                resampled, symbol, "close"
                            )
                            if symbol_data is not None:
                                # Get only data after the cached latest
                                new_data = symbol_data[
                                    symbol_data.index > latest_timestamps[symbol]
                                ]
                                if len(new_data) > 0:
                                    # Build OHLCV dict for this symbol
                                    ohlcv = VBTDataHandler.extract_ohlcv(resampled)
                                    symbol_df = pd.DataFrame()
                                    for feature, df in ohlcv.items():
                                        if df is not None and symbol in df.columns:
                                            feature_data = df[symbol][
                                                df.index > latest_timestamps[symbol]
                                            ]
                                            symbol_df[feature.capitalize()] = (
                                                feature_data
                                            )
                                    if not symbol_df.empty:
                                        new_data_dict[symbol] = symbol_df

                    if new_data_dict:
                        # Merge with cached data for these symbols
                        cached_symbol_dict = self.merger._extract_symbols(
                            cached_data, set(stale_symbols)
                        )

                        # Append new data to cached data
                        for symbol, new_df in new_data_dict.items():
                            if symbol in cached_symbol_dict:
                                cached_symbol_dict[symbol] = pd.concat(
                                    [cached_symbol_dict[symbol], new_df]
                                ).sort_index()

                        return VBTDataHandler.create_from_dict(cached_symbol_dict)

            # Fall back to exchange fetch for update period only
            update_data = self.exchange_fetcher.fetch_symbols(
                stale_symbols,
                timeframe,
                update_start_dates,
                end_date,
                parallel=True,
                fetch_logger=fetch_logger,
            )

            if update_data:
                # Merge update with cached data
                cached_symbol_dict = self.merger._extract_symbols(
                    cached_data, set(stale_symbols)
                )
                update_symbol_dict = self.merger._extract_symbols(
                    update_data, set(stale_symbols)
                )

                # Append new data
                for symbol in stale_symbols:
                    if symbol in cached_symbol_dict and symbol in update_symbol_dict:
                        cached_symbol_dict[symbol] = (
                            pd.concat(
                                [cached_symbol_dict[symbol], update_symbol_dict[symbol]]
                            )
                            .sort_index()
                            .drop_duplicates()
                        )

                return VBTDataHandler.create_from_dict(cached_symbol_dict)

            return None

        except Exception as e:
            logger.error(f"Error updating stale symbols: {e}")
            # Fall back to full fetch
            return self._fetch_missing_symbols(
                stale_symbols,
                timeframe,
                {},
                end_date,
                cache_handler,
                prefer_resampling,
                fetch_logger,
            )

    def _fetch_missing_symbols(
        self,
        symbols: List[str],
        timeframe: str,
        symbol_start_dates: dict,
        end_date: Optional[Union[str, vbt.timestamp]],
        cache_handler: CacheHandler,
        prefer_resampling: bool,
        fetch_logger: FetchLogger,
    ) -> Optional[vbt.Data]:
        """Fetch missing symbols using resampling or exchange."""
        # Try resampling first if preferred
        if prefer_resampling:
            resampled = cache_handler.try_load_from_lower_timeframe(
                symbols, end_date, require_fresh=True
            )
            if resampled:
                return resampled

        # Fall back to exchange
        return self._fetch_from_exchange(
            symbols, timeframe, symbol_start_dates, end_date, fetch_logger
        )

    def _fetch_from_exchange(
        self,
        symbols: List[str],
        timeframe: str,
        symbol_start_dates: dict,
        end_date: Optional[Union[str, vbt.timestamp]],
        fetch_logger: FetchLogger,
    ) -> Optional[vbt.Data]:
        """Fetch data from exchange."""
        # Get start dates for symbols being fetched
        fetch_start_dates = {
            symbol: symbol_start_dates.get(symbol)
            for symbol in symbols
            if symbol in symbol_start_dates
        }

        return self.exchange_fetcher.fetch_symbols(
            symbols, timeframe, fetch_start_dates, end_date, fetch_logger=fetch_logger
        )

    def fetch_top_symbols(
        self,
        quote_currency: str = "USDT",
        limit: int = 10,
        timeframe: str = "1d",
        start_date: Optional[Union[str, vbt.timestamp]] = None,
        end_date: Optional[Union[str, vbt.timestamp]] = None,
        use_cache: bool = True,
    ) -> Optional[vbt.Data]:
        """
        Fetch top symbols by volume.

        Args:
            quote_currency: Quote currency filter
            limit: Number of top symbols
            timeframe: Timeframe string
            start_date: Start date
            end_date: End date
            use_cache: Whether to use caching

        Returns:
            VBT Data object or None
        """
        # Get top symbols
        top_symbols = self.symbol_resolver.get_top_symbols(quote_currency, limit)

        if not top_symbols:
            logger.error("No top symbols found")
            return None

        # Fetch data for top symbols
        return self.fetch_data(
            symbols=top_symbols,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
        )

    def update_data(self, timeframe: str, symbols: Optional[List[str]] = None) -> bool:
        """
        Update cached data to latest.

        Args:
            timeframe: Timeframe to update
            symbols: Specific symbols (None for all cached)

        Returns:
            True if successful
        """
        try:
            cache_handler = CacheHandler(self.exchange_id, timeframe, self.market_type)

            # Load existing data without freshness check
            existing_data, _, _, _ = cache_handler.try_load_cached(
                symbols or [], require_fresh=False
            )

            if not existing_data:
                logger.error("No existing data to update")
                return False

            # Now check which symbols are stale
            symbols_to_check = symbols if symbols else list(existing_data.symbols)
            _, fresh_symbols, _, stale_symbols = cache_handler.try_load_cached(
                symbols_to_check, end_date="now", require_fresh=True
            )

            if not stale_symbols:
                logger.info("All symbols are up to date")
                return True

            # Update stale symbols
            logger.info(f"Updating {len(stale_symbols)} stale symbols")

            # Try native VBT update first
            updated = cache_handler.update_cached_data(existing_data)
            if updated:
                return True

            # Fall back to our update method
            fetch_logger = FetchLogger(list(stale_symbols), self.exchange_id, timeframe)
            updated_data = self._update_stale_symbols(
                existing_data,
                list(stale_symbols),
                timeframe,
                "now",
                cache_handler,
                prefer_resampling=True,
                fetch_logger=fetch_logger,
            )

            if updated_data:
                # Merge with fresh symbols
                if fresh_symbols:
                    merged = self.merger.merge_symbol_sets(
                        existing_data, fresh_symbols, updated_data, stale_symbols
                    )
                    if merged:
                        return cache_handler.save_data(merged)
                else:
                    return cache_handler.save_data(updated_data)

            return False

        except Exception as e:
            logger.error(f"Update error: {e}")
            return False
