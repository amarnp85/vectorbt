#!/usr/bin/env python3
"""Data Merger - Combine data from multiple sources.

This module handles merging data from different sources like cached data,
resampled data, and freshly fetched data.
"""

import logging
from typing import Dict, List, Optional, Set
import pandas as pd
import vectorbtpro as vbt
from .vbt_data_handler import VBTDataHandler

logger = logging.getLogger(__name__)


class DataMerger:
    """Merge data from multiple sources."""

    @staticmethod
    def merge_symbol_sets(
        cached_data: Optional[vbt.Data],
        cached_symbols: Set[str],
        new_data: Optional[vbt.Data],
        new_symbols: Set[str],
    ) -> Optional[vbt.Data]:
        """
        Merge data from cache and newly fetched symbols.

        Args:
            cached_data: VBT Data with cached symbols
            cached_symbols: Set of symbols to use from cache
            new_data: VBT Data with newly fetched symbols
            new_symbols: Set of symbols to use from new data

        Returns:
            Merged VBT Data or None
        """
        try:
            # Handle edge cases
            if not cached_symbols and not new_symbols:
                return None

            if not cached_symbols:
                return new_data

            if not new_symbols:
                return cached_data.select_symbols(list(cached_symbols))

            # Both have symbols - need to merge
            logger.debug(
                f"Merging {len(cached_symbols)} cached symbols with "
                f"{len(new_symbols)} new symbols"
            )

            # Build symbol dictionary
            symbol_dict = {}

            # Add cached symbols
            if cached_data and cached_symbols:
                symbol_dict.update(
                    DataMerger._extract_symbols(cached_data, cached_symbols)
                )

            # Add new symbols
            if new_data and new_symbols:
                symbol_dict.update(DataMerger._extract_symbols(new_data, new_symbols))

            if not symbol_dict:
                logger.error("No valid symbol data after merge attempt")
                return None

            # Create merged VBT Data
            merged = VBTDataHandler.create_from_dict(symbol_dict)

            if merged is not None:
                logger.debug(f"Successfully merged {len(merged.symbols)} symbols")

            return merged

        except Exception as e:
            logger.error(f"Error merging symbol sets: {e}")
            return None

    @staticmethod
    def _extract_symbols(data: vbt.Data, symbols: Set[str]) -> Dict[str, pd.DataFrame]:
        """Extract specific symbols from VBT Data as dictionary."""
        symbol_dict = {}

        try:
            ohlcv_data = VBTDataHandler.extract_ohlcv(data)
            if not ohlcv_data:
                logger.error("Failed to extract OHLCV for merging")
                return symbol_dict

            for symbol in symbols:
                if symbol not in data.symbols:
                    continue

                # Build DataFrame for this symbol
                symbol_df = pd.DataFrame()

                for feature, feature_data in ohlcv_data.items():
                    if feature_data is None:
                        continue

                    col_name = feature.capitalize()

                    # Handle both Series and DataFrame cases
                    if isinstance(feature_data, pd.Series):
                        # Single symbol case - use the series directly
                        if len(data.symbols) == 1 and data.symbols[0] == symbol:
                            symbol_df[col_name] = feature_data
                    elif isinstance(feature_data, pd.DataFrame):
                        # Multi-symbol case - extract the specific symbol column
                        if symbol in feature_data.columns:
                            symbol_df[col_name] = feature_data[symbol]
                    else:
                        logger.debug(
                            f"Unexpected data type for {feature}: {type(feature_data)}"
                        )

                # Only add if we have at least OHLC
                if len(symbol_df.columns) >= 4:
                    symbol_dict[symbol] = symbol_df.dropna()

        except Exception as e:
            logger.error(f"Error extracting symbols: {e}")

        return symbol_dict

    @staticmethod
    def append_time_period(
        base_data: vbt.Data, new_period_data: vbt.Data
    ) -> Optional[vbt.Data]:
        """
        Append new time period data to existing data.

        Used for updating cached data with recent candles.

        Args:
            base_data: Existing data
            new_period_data: New data to append

        Returns:
            Combined data or None
        """
        try:
            # Use VBT's concat with drop_duplicates
            combined = base_data.concat(new_period_data, drop_duplicates=True)

            if combined is not None:
                logger.debug("Successfully appended new time period")
                return combined
            else:
                logger.debug("VBT concat returned None")
                return None

        except Exception as e:
            logger.error(f"Error appending time period: {e}")
            return None

    @staticmethod
    def validate_merged_data(
        merged_data: vbt.Data, expected_symbols: List[str]
    ) -> bool:
        """
        Validate that merged data contains all expected symbols.

        Args:
            merged_data: Merged VBT Data
            expected_symbols: List of symbols that should be present

        Returns:
            True if valid, False otherwise
        """
        try:
            if merged_data is None:
                return False

            if not hasattr(merged_data, "symbols"):
                return False

            actual_symbols = set(merged_data.symbols)
            expected_set = set(expected_symbols)

            if actual_symbols != expected_set:
                missing = expected_set - actual_symbols
                extra = actual_symbols - expected_set

                if missing:
                    logger.error(f"Missing symbols after merge: {missing}")
                if extra:
                    logger.warning(f"Extra symbols after merge: {extra}")

                return False

            return True

        except Exception as e:
            logger.error(f"Error validating merged data: {e}")
            return False
