#!/usr/bin/env python3
"""Symbol Resolver - Handle symbol filtering, metadata, and inception dates.

This module manages symbol-related operations including blacklist filtering,
inception date resolution, and metadata retrieval.
"""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime
import vectorbtpro as vbt
from ...cache_system import cache_manager

logger = logging.getLogger(__name__)

# Class-level cache to prevent redundant metadata updates
_metadata_update_cache = {}


class SymbolResolver:
    """Resolve and manage symbol metadata and filtering."""

    def __init__(self, exchange_id: str):
        """Initialize with exchange ID."""
        self.exchange_id = exchange_id
        self._metadata_checked = False

    def _ensure_metadata_current(self) -> None:
        """Ensure volume and symbol metadata are up to date (lazy initialization)."""
        # Skip if already checked for this instance
        if self._metadata_checked:
            return

        # Skip if recently updated by another instance (within last 5 minutes)
        cache_key = f"{self.exchange_id}_metadata_update"
        current_time = datetime.now().timestamp()

        if cache_key in _metadata_update_cache:
            last_update = _metadata_update_cache[cache_key]
            if current_time - last_update < 300:  # 5 minutes
                logger.debug(
                    f"Skipping metadata update for {self.exchange_id} - recently updated"
                )
                self._metadata_checked = True
                return

        try:
            logger.debug(f"Ensuring metadata current for {self.exchange_id}")

            # Check if we have any cached volume data at all
            cached_volumes = cache_manager.get_all_volumes(self.exchange_id)
            
            # Only fetch fresh metadata if:
            # 1. No cached volume data exists, OR
            # 2. Cache is stale AND we have very few symbols (< 100)
            should_update = False
            
            if not cached_volumes:
                logger.debug("No cached volume data found, will fetch fresh metadata")
                should_update = True
            elif not cache_manager.is_volume_cache_fresh(self.exchange_id):
                # Cache is stale, but only update if we have very few symbols
                # This prevents unnecessary fetches for well-populated caches
                if len(cached_volumes) < 100:
                    logger.debug(f"Cache is stale with only {len(cached_volumes)} symbols, will refresh")
                    should_update = True
                else:
                    logger.debug(f"Cache is stale but has {len(cached_volumes)} symbols, skipping refresh to avoid unnecessary API calls")
            
            if should_update:
                self._update_volume_data()
                # Mark as updated in cache
                _metadata_update_cache[cache_key] = current_time

            # Apply blacklist filtering
            self._apply_blacklist_filtering()

            self._metadata_checked = True

        except Exception as e:
            logger.debug(f"Error ensuring metadata current: {e}")
            self._metadata_checked = True  # Mark as checked to avoid retry loops

    def _is_volume_cache_current(self, target_timestamp: float) -> bool:
        """DEPRECATED: Use cache_manager.is_volume_cache_fresh() instead."""
        # This method is now deprecated in favor of the simpler file-based check
        return cache_manager.is_volume_cache_fresh(self.exchange_id)

    def _update_volume_data(self) -> None:
        """Update volume data from exchange."""
        try:
            from ...cache_system.metadata_fetcher import data_fetcher

            market_data = data_fetcher.get_market_data(
                exchange_id=self.exchange_id, use_cache=False, force_refresh=True
            )

            if market_data:
                logger.debug(f"Updated volume data: {len(market_data)} symbols")

                # Check for new symbols and update timestamps
                existing_timestamps = cache_manager.get_all_timestamps(self.exchange_id)

                new_symbols = [
                    symbol
                    for symbol, data in market_data.items()
                    if symbol not in existing_timestamps
                    and isinstance(data, dict)
                    and data.get("volume", 0) > 0
                ]

                if new_symbols:
                    # Limit to reasonable number
                    MAX_NEW_TIMESTAMPS = 100
                    if len(new_symbols) > MAX_NEW_TIMESTAMPS:
                        # Sort by volume and take top ones
                        new_symbols.sort(
                            key=lambda s: market_data[s].get("volume", 0), reverse=True
                        )
                        new_symbols = new_symbols[:MAX_NEW_TIMESTAMPS]

                    logger.debug(
                        f"Fetching timestamps for {len(new_symbols)} new symbols"
                    )
                    data_fetcher.get_inception_timestamps(
                        exchange_id=self.exchange_id,
                        symbols=new_symbols,
                        use_cache=True,
                    )

        except Exception as e:
            logger.debug(f"Error updating volume data: {e}")

    def _apply_blacklist_filtering(self) -> None:
        """Apply blacklist filtering to cached data."""
        try:
            blacklist_data = cache_manager.get_blacklist()

            # Combine global and exchange-specific blacklists
            blacklisted = set(blacklist_data.get("global", []))
            blacklisted.update(blacklist_data.get(self.exchange_id.lower(), []))

            if not blacklisted:
                return

            # Remove from volume cache
            volume_data = cache_manager.get_all_volumes(self.exchange_id)
            if volume_data:
                to_remove = set(volume_data.keys()) & blacklisted
                if to_remove:
                    logger.debug(
                        f"Removing {len(to_remove)} blacklisted from volume cache"
                    )
                    for symbol in to_remove:
                        cache_manager._volume_cache[self.exchange_id].pop(symbol, None)
                    cache_manager._save_to_disk(
                        self.exchange_id,
                        "volume",
                        cache_manager._volume_cache[self.exchange_id],
                    )

            # Remove from timestamp cache
            timestamp_data = cache_manager.get_all_timestamps(self.exchange_id)
            if timestamp_data:
                to_remove = set(timestamp_data.keys()) & blacklisted
                if to_remove:
                    logger.debug(
                        f"Removing {len(to_remove)} blacklisted from timestamp cache"
                    )
                    for symbol in to_remove:
                        cache_manager._timestamp_cache[self.exchange_id].pop(
                            symbol, None
                        )
                    cache_manager._save_to_disk(
                        self.exchange_id,
                        "timestamps",
                        cache_manager._timestamp_cache[self.exchange_id],
                    )

            # Mark as failed to prevent future fetching
            for symbol in blacklisted:
                cache_manager.add_failed_symbol(self.exchange_id, symbol)

        except Exception as e:
            logger.debug(f"Error applying blacklist: {e}")

    def filter_symbols(self, symbols: List[str]) -> List[str]:
        """Filter symbols by blacklist and availability."""
        # Ensure metadata is current before filtering
        self._ensure_metadata_current()
        return cache_manager.filter_blacklisted_symbols(symbols, self.exchange_id)

    def get_optimal_start_dates(
        self, symbols: List[str], requested_start: Optional[Union[str, vbt.timestamp]]
    ) -> Dict[str, str]:
        """Get optimal start dates for symbols based on inception data."""
        # Ensure metadata is current before getting start dates
        self._ensure_metadata_current()

        symbol_start_dates = {}

        # Check if this is an inception request
        if self._is_inception_request(requested_start):
            logger.debug("Using inception dates from cache")
            cached_timestamps = cache_manager.get_all_timestamps(self.exchange_id)

            if cached_timestamps:
                for symbol in symbols:
                    if symbol in cached_timestamps:
                        try:
                            timestamp_ms = cached_timestamps[symbol]
                            dt = datetime.fromtimestamp(timestamp_ms / 1000)
                            symbol_start_dates[symbol] = dt.strftime("%Y-%m-%d")
                        except Exception:
                            symbol_start_dates[symbol] = requested_start
                    else:
                        symbol_start_dates[symbol] = requested_start
            else:
                # No cached inception dates
                for symbol in symbols:
                    symbol_start_dates[symbol] = requested_start
        else:
            # Use provided start date for all symbols
            for symbol in symbols:
                symbol_start_dates[symbol] = requested_start

        return symbol_start_dates

    def _is_inception_request(
        self, start_date: Optional[Union[str, vbt.timestamp]]
    ) -> bool:
        """Check if this is an inception fetch request."""
        if start_date is None:
            return True
        if isinstance(start_date, str):
            try:
                start_year = int(start_date[:4]) if len(start_date) >= 4 else 9999
                return start_year < 2010  # Reasonable cutoff for crypto
            except (ValueError, TypeError):
                return False
        return False

    def check_inception_completeness(
        self, data: vbt.Data, symbols: List[str], symbol_start_dates: Dict[str, str]
    ) -> bool:
        """Check if data contains complete inception data for all symbols."""
        try:
            for symbol in symbols:
                if symbol not in data.symbols:
                    return False

                # Get expected start date
                expected_start = symbol_start_dates.get(symbol)
                if not expected_start:
                    continue

                # Get actual start date from data
                from .vbt_data_handler import VBTDataHandler

                symbol_data = VBTDataHandler.get_symbol_data(data, symbol, "close")

                if symbol_data is None or len(symbol_data) == 0:
                    return False

                cached_start = symbol_data.index[0].date()
                expected_start_dt = datetime.strptime(expected_start, "%Y-%m-%d").date()

                # Allow 1 day tolerance
                days_diff = abs((cached_start - expected_start_dt).days)
                if days_diff > 1:
                    logger.debug(
                        f"Inception incomplete for {symbol}: "
                        f"cached starts {cached_start}, expected {expected_start_dt}"
                    )
                    return False

            return True

        except Exception as e:
            logger.debug(f"Error checking inception completeness: {e}")
            return False

    def get_top_symbols(
        self, quote_currency: Optional[str] = None, limit: int = 10
    ) -> List[str]:
        """Get top symbols by volume."""
        # Ensure metadata is current before getting top symbols
        self._ensure_metadata_current()

        try:
            volume_data = cache_manager.get_all_volumes(self.exchange_id)

            # Filter by quote currency if specified
            if quote_currency and volume_data:
                filtered_volume_data = {
                    symbol: volume
                    for symbol, volume in volume_data.items()
                    if "/" in symbol and symbol.split("/")[1] == quote_currency
                }
                volume_data = filtered_volume_data

            if not volume_data:
                return []

            # Sort by volume and get top symbols
            sorted_symbols = sorted(
                volume_data.items(), key=lambda x: x[1], reverse=True
            )

            # Get extra in case some are blacklisted
            top_symbols = [symbol for symbol, _ in sorted_symbols[: limit * 2]]

            # Apply blacklist filtering
            top_symbols = self.filter_symbols(top_symbols)

            # Return requested limit
            return top_symbols[:limit]

        except Exception as e:
            logger.error(f"Error getting top symbols: {e}")
            return []
