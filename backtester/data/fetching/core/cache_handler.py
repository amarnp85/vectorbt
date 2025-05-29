#!/usr/bin/env python3
"""Cache Handler - Manage data cache operations.

This module handles all cache-related operations including loading,
saving, and checking cached data availability.
"""

import logging
from typing import List, Optional, Set, Tuple
import vectorbtpro as vbt
from ...storage.data_storage import data_storage
from .freshness_checker import FreshnessChecker
from .symbol_resolver import SymbolResolver
from .resampler import DataResampler

logger = logging.getLogger(__name__)


class CacheHandler:
    """Handle all cache operations for data fetching."""

    def __init__(self, exchange_id: str, timeframe: str, market_type: str = "spot"):
        """Initialize cache handler."""
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.market_type = market_type
        self.symbol_resolver = SymbolResolver(exchange_id)
        self.freshness_checker = FreshnessChecker()
        self.resampler = DataResampler()

    def try_load_cached(
        self,
        symbols: List[str],
        end_date: Optional[str] = None,
        require_fresh: bool = True,
    ) -> Tuple[Optional[vbt.Data], Set[str], Set[str], Set[str]]:
        """
        Try to load data from cache.

        Returns:
            Tuple of (data, available_symbols, missing_symbols, stale_symbols)
            data is None if cache miss or all data is stale
        """
        try:
            # Load cached data
            cached_data = data_storage.load_data(
                self.exchange_id, self.timeframe, [], self.market_type
            )

            if cached_data is None:
                logger.debug("No cached data found")
                return None, set(), set(symbols), set()

            # Check symbol availability
            cached_symbols = set(cached_data.symbols)
            requested_symbols = set(symbols)
            available_symbols = requested_symbols & cached_symbols
            missing_symbols = requested_symbols - cached_symbols
            stale_symbols = set()

            if not available_symbols:
                logger.debug("No requested symbols in cache")
                return None, set(), requested_symbols, set()

            # Check freshness if required
            if require_fresh and end_date:
                stale_list, fresh_list = self.freshness_checker.identify_stale_symbols(
                    cached_data, list(available_symbols), end_date, self.timeframe
                )

                if stale_list:
                    logger.debug(f"Found {len(stale_list)} stale symbols")
                    # Keep stale symbols separate - don't move to missing
                    stale_symbols = set(stale_list)
                    available_symbols = set(fresh_list)

                    if not available_symbols and not stale_symbols:
                        # All cached symbols are missing
                        return None, set(), requested_symbols, set()

            # Return cached data with symbol sets
            return cached_data, available_symbols, missing_symbols, stale_symbols

        except Exception as e:
            logger.error(f"Error loading cached data: {e}")
            return None, set(), set(symbols), set()

    def try_load_from_lower_timeframe(
        self,
        symbols: List[str],
        end_date: Optional[str] = None,
        require_fresh: bool = True,
    ) -> Optional[vbt.Data]:
        """
        Try to load and resample data from lower timeframes.

        Returns:
            Resampled data or None if not available
        """
        try:
            # Get available lower timeframes
            available_sources = self.resampler.get_available_sources(self.timeframe)

            # Try from highest to lowest (best quality first)
            for source_tf in reversed(available_sources):
                logger.debug(f"Checking {source_tf} for resampling")

                # Check if data exists
                if not data_storage.data_exists(
                    self.exchange_id, source_tf, self.market_type
                ):
                    continue

                # Load source data
                source_data = data_storage.load_data(
                    self.exchange_id, source_tf, [], self.market_type
                )

                if source_data is None:
                    continue

                # Check if all symbols are available
                source_symbols = set(source_data.symbols)
                if not set(symbols).issubset(source_symbols):
                    missing = set(symbols) - source_symbols
                    logger.debug(f"Missing symbols in {source_tf}: {missing}")
                    continue

                # Check freshness if required
                if require_fresh and end_date:
                    if not self.freshness_checker.is_data_fresh(
                        source_data.select_symbols(symbols), end_date, source_tf
                    ):
                        logger.debug(f"Data in {source_tf} is stale")
                        continue

                # Try to resample
                logger.debug(f"Resampling from {source_tf} to {self.timeframe}")
                filtered_data = source_data.select_symbols(symbols)
                resampled = self.resampler.resample_data(filtered_data, self.timeframe)

                if resampled is not None:
                    return resampled

            logger.debug("No suitable lower timeframe found for resampling")
            return None

        except Exception as e:
            logger.error(f"Error in lower timeframe loading: {e}")
            return None

    def save_data(self, data: vbt.Data) -> bool:
        """Save data to cache."""
        try:
            return data_storage.save_data(
                data, self.exchange_id, self.timeframe, self.market_type
            )
        except Exception as e:
            logger.error(f"Error saving data: {e}")
            return False

    def update_cached_data(
        self, cached_data: vbt.Data, end_date: Optional[str] = None
    ) -> Optional[vbt.Data]:
        """
        Try to update cached data to current time.

        Uses VBT's native update method if available.
        """
        try:
            if end_date is None:
                target_end = vbt.utc_timestamp()
            elif isinstance(end_date, str) and end_date.lower() == "now":
                target_end = vbt.utc_timestamp()
            else:
                target_end = vbt.timestamp(end_date, tz="UTC")

            # Try VBT's native update
            logger.debug("Attempting VBT native update")
            updated = cached_data.update(
                end=target_end, show_progress=False, silence_warnings=True
            )

            if updated is not None:
                # Save updated data
                self.save_data(updated)
                return updated
            else:
                logger.debug("VBT update returned None")
                return None

        except NotImplementedError:
            logger.debug("VBT update not implemented for this data type")
            return None
        except Exception as e:
            logger.debug(f"Update failed: {e}")
            return None
