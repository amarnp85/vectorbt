#!/usr/bin/env python3
"""
Simplified Cache Manager

A streamlined cache manager focused on the core requirements:
1. Volume data cache (refreshed daily)
2. Inception timestamp cache (persistent)
3. Basic failed symbol tracking
4. Blacklist - manually specified symbols to exclude from all operations

This replaces the complex multi-cache system with a focused, efficient implementation.
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Set

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


class SimpleCacheManager:
    """
    Simplified cache manager for cryptocurrency exchange data.

    Focuses on two main data types:
    1. Volume data - cached daily, automatically expires after 24 hours
    2. Timestamp data - persistent inception timestamps for symbols
    3. Failed symbols - basic tracking to avoid retrying problematic symbols
    4. Blacklist - manually specified symbols to exclude from all operations

    Each exchange has its own directory with three files:
    - volume.json (expires after 24 hours)
    - timestamps.json (persistent)
    - failed_symbols.json (list of problematic symbols)

    Global blacklist.json file in cache root excludes symbols across all exchanges.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the simplified cache manager."""
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # In-memory caches
        self._volume_cache = {}  # exchange -> symbol -> {volume, timestamp}
        self._timestamp_cache = {}  # exchange -> symbol -> timestamp
        self._failed_symbols = {}  # exchange -> set of failed symbols

        # Initialize default blacklist if needed
        self._ensure_default_blacklist()

        # Load existing caches
        self._load_caches()

    def _ensure_default_blacklist(self) -> None:
        """
        Ensure a default blacklist.json exists with common problematic symbols.

        This is called during cache manager initialization to guarantee that
        any exchange will have a sensible default blacklist applied, particularly
        for stable-stable pairs that provide minimal trading value.
        """
        blacklist_file = os.path.join(self.cache_dir, "blacklist.json")

        if not os.path.exists(blacklist_file):
            logger.info(f"📝 Creating default blacklist.json at {blacklist_file}")

            # Default blacklist with common stable-stable pairs and problematic symbols
            default_blacklist = {
                "global": [
                    # Major stable-stable pairs (minimal price movement)
                    "USDC/USDT",
                    "BUSD/USDT",
                    "TUSD/USDT",
                    "DAI/USDT",
                    "USDP/USDT",
                    "USDC/BUSD",
                    "USDT/BUSD",
                    "DAI/USDC",
                    "TUSD/USDC",
                    "USDP/USDC",
                ],
                "binance": [
                    # Binance-specific stable pairs
                    "BUSD/FDUSD",
                    "FDUSD/USDT",
                ],
                "bybit": [
                    # Bybit futures stable pairs
                    "USDC/USDT:USDT",
                    "BUSD/USDT:USDT",
                ],
                "okx": [
                    # OKX-specific exclusions
                ],
                "kucoin": [
                    # KuCoin-specific exclusions
                ],
                "coinbase": [
                    # Coinbase-specific exclusions
                ],
                "_comment": "Global blacklist applies to all exchanges. Exchange-specific lists apply only to that exchange. Stable-stable pairs are blacklisted as they have minimal price movement and low trading value.",
                "_usage": "Add symbols here to exclude them from all fetch operations. Use 'global' for all exchanges or specific exchange names for targeted exclusions.",
                "_last_updated": datetime.now().isoformat(),
            }

            try:
                with open(blacklist_file, "w") as f:
                    json.dump(default_blacklist, f, indent=2)
                logger.info(
                    f"✅ Created default blacklist with {len(default_blacklist['global'])} global exclusions"
                )
                logger.info(
                    f"🚫 Default global blacklist: {default_blacklist['global'][:5]}{'...' if len(default_blacklist['global']) > 5 else ''}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Could not create default blacklist.json: {e}")
        else:
            # Validate existing blacklist structure and update if needed
            try:
                with open(blacklist_file, "r") as f:
                    existing_blacklist = json.load(f)

                # Ensure USDC/USDT is in global blacklist (user requirement)
                needs_update = False
                if isinstance(existing_blacklist, dict):
                    global_list = existing_blacklist.get("global", [])
                    if "USDC/USDT" not in global_list:
                        global_list.append("USDC/USDT")
                        existing_blacklist["global"] = global_list
                        needs_update = True
                        logger.info("🔄 Adding USDC/USDT to existing global blacklist")

                # Update last_updated timestamp if we made changes
                if needs_update:
                    existing_blacklist["_last_updated"] = datetime.now().isoformat()
                    with open(blacklist_file, "w") as f:
                        json.dump(existing_blacklist, f, indent=2)
                    logger.info("✅ Updated existing blacklist.json")

            except Exception as e:
                logger.warning(f"⚠️ Could not validate existing blacklist.json: {e}")

    def get_blacklist(self) -> Dict[str, List[str]]:
        """
        Load and return the current blacklist configuration.

        Returns:
            Dictionary with global and exchange-specific blacklists
        """
        blacklist_file = os.path.join(self.cache_dir, "blacklist.json")

        try:
            if os.path.exists(blacklist_file):
                with open(blacklist_file, "r") as f:
                    return json.load(f)
            else:
                # Should not happen due to _ensure_default_blacklist(), but handle gracefully
                return {"global": ["USDC/USDT"]}
        except Exception as e:
            logger.warning(f"⚠️ Error loading blacklist: {e}")
            return {"global": ["USDC/USDT"]}  # Fallback to minimum default

    def is_symbol_blacklisted(self, symbol: str, exchange_id: str = None) -> bool:
        """
        Check if a symbol is blacklisted.

        Args:
            symbol: Symbol to check
            exchange_id: Optional exchange ID for exchange-specific checks

        Returns:
            True if symbol is blacklisted
        """
        blacklist_data = self.get_blacklist()

        # Check global blacklist
        global_blacklist = blacklist_data.get("global", [])
        if symbol in global_blacklist:
            return True

        # Check exchange-specific blacklist
        if exchange_id:
            exchange_blacklist = blacklist_data.get(exchange_id.lower(), [])
            if symbol in exchange_blacklist:
                return True

        return False

    def filter_blacklisted_symbols(
        self, symbols: List[str], exchange_id: str = None
    ) -> List[str]:
        """
        Filter out blacklisted symbols from a list.

        Args:
            symbols: List of symbols to filter
            exchange_id: Optional exchange ID for exchange-specific filtering

        Returns:
            Filtered list of symbols
        """
        blacklist_data = self.get_blacklist()

        # Combine global and exchange-specific blacklists
        blacklisted = set(blacklist_data.get("global", []))
        if exchange_id:
            exchange_blacklist = blacklist_data.get(exchange_id.lower(), [])
            blacklisted.update(exchange_blacklist)

        if not blacklisted:
            return symbols

        # Filter out blacklisted symbols
        filtered = [s for s in symbols if s not in blacklisted]

        removed_count = len(symbols) - len(filtered)
        if removed_count > 0:
            logger.info(f"🚫 Filtered out {removed_count} blacklisted symbols")
            removed_symbols = [s for s in symbols if s in blacklisted]
            logger.debug(f"🚫 Removed symbols: {removed_symbols}")

        return filtered

    def _get_exchange_dir(self, exchange_id: str) -> str:
        """Get the directory path for an exchange."""
        exchange_dir = os.path.join(self.cache_dir, exchange_id.lower())
        os.makedirs(exchange_dir, exist_ok=True)
        return exchange_dir

    def _migrate_legacy_volume_cache(self, exchange_id: str, file_path: str) -> bool:
        """
        Migrate legacy volume cache format to new format with _metadata section.
        
        Args:
            exchange_id: Exchange identifier
            file_path: Path to the volume.json file
            
        Returns:
            True if migration was successful or not needed, False on error
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            # Check if already in new format
            if isinstance(data, dict) and "_metadata" in data:
                return True  # Already migrated
            
            # Check if it's a valid legacy format (symbol -> volume data)
            if not isinstance(data, dict):
                logger.warning(f"Invalid volume cache format for {exchange_id}")
                return False
            
            # Validate legacy format - should have symbols with volume data
            sample_keys = list(data.keys())[:5]  # Check first 5 entries
            is_legacy_format = True
            
            for key in sample_keys:
                if key.startswith("_"):  # Skip metadata keys
                    continue
                    
                value = data[key]
                if isinstance(value, dict):
                    # New format: {"volume": 123, "updated": "..."}
                    if "volume" not in value:
                        is_legacy_format = False
                        break
                elif isinstance(value, (int, float)):
                    # Very old format: just volume numbers
                    is_legacy_format = True
                    break
                else:
                    is_legacy_format = False
                    break
            
            if not is_legacy_format:
                logger.warning(f"Unrecognized volume cache format for {exchange_id}")
                return False
            
            # Get file modification time for migration timestamp
            file_mtime = os.path.getmtime(file_path)
            migration_timestamp = datetime.fromtimestamp(file_mtime, tz=timezone.utc).isoformat()
            
            # Normalize legacy data to new format
            normalized_data = {}
            for symbol, volume_data in data.items():
                if symbol.startswith("_"):  # Skip any existing metadata
                    continue
                    
                try:
                    if isinstance(volume_data, dict) and "volume" in volume_data:
                        # Already in intermediate format
                        normalized_data[symbol] = volume_data
                    elif isinstance(volume_data, (int, float)):
                        # Very old format - just volume numbers
                        volume = float(volume_data)
                        if volume > 0:
                            normalized_data[symbol] = {
                                "volume": volume,
                                "updated": migration_timestamp,
                            }
                    else:
                        logger.debug(f"Skipping invalid volume data for {symbol}: {volume_data}")
                        
                except (ValueError, TypeError):
                    logger.debug(f"Skipping invalid volume data for {symbol}: {volume_data}")
            
            # Create new format with metadata
            migrated_data = {
                "_metadata": {
                    "last_updated_utc": migration_timestamp,
                    "exchange_id": exchange_id,
                    "total_symbols": len(normalized_data),
                    "cache_version": "2.0",
                    "migrated_from": "legacy",
                    "migration_timestamp": datetime.now(timezone.utc).isoformat(),
                },
                **normalized_data,
            }
            
            # Atomic write of migrated data
            temp_path = f"{file_path}.migration.tmp"
            with open(temp_path, "w") as f:
                json.dump(migrated_data, f, indent=2)
            
            # Replace original file
            os.replace(temp_path, file_path)
            
            logger.info(f"✅ Migrated legacy volume cache for {exchange_id}: {len(normalized_data)} symbols")
            logger.debug(f"Migration preserved timestamp: {migration_timestamp}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error migrating legacy volume cache for {exchange_id}: {e}")
            return False

    def _load_caches(self) -> None:
        """Load all caches from disk."""
        if not os.path.exists(self.cache_dir):
            return

        for item in os.listdir(self.cache_dir):
            exchange_dir = os.path.join(self.cache_dir, item)
            if not os.path.isdir(exchange_dir):
                continue

            exchange_id = item

            # Load volume cache
            volume_file = os.path.join(exchange_dir, "volume.json")
            if os.path.exists(volume_file):
                try:
                    # First, attempt to migrate legacy format if needed
                    migration_success = self._migrate_legacy_volume_cache(exchange_id, volume_file)
                    
                    if not migration_success:
                        logger.warning(f"Failed to migrate volume cache for {exchange_id}, skipping")
                        continue
                    
                    # Now load the (potentially migrated) cache
                    with open(volume_file, "r") as f:
                        data = json.load(f)
                        
                        # Always load cache into memory, regardless of freshness
                        # The SymbolResolver will decide whether to refresh based on its own logic
                        if isinstance(data, dict) and "_metadata" in data:
                            # Extract only symbol data, excluding metadata
                            symbol_data = {
                                k: v
                                for k, v in data.items()
                                if not k.startswith("_")
                            }
                            self._volume_cache[exchange_id] = symbol_data
                            cache_version = data['_metadata'].get('cache_version', '1.0')
                            migrated_note = " (migrated)" if data['_metadata'].get('migrated_from') == 'legacy' else ""
                            
                            # Check freshness for logging purposes
                            is_fresh = self._is_volume_cache_valid(volume_file)
                            freshness_note = " (fresh)" if is_fresh else " (stale)"
                            
                            logger.debug(
                                f"Loaded volume cache for {exchange_id}: {len(symbol_data)} symbols (v{cache_version}){migrated_note}{freshness_note}"
                            )
                        else:
                            # This shouldn't happen after migration, but handle gracefully
                            self._volume_cache[exchange_id] = data
                            logger.debug(
                                f"Loaded legacy volume cache for {exchange_id}: {len(data)} symbols (migration may have failed)"
                            )
                except Exception as e:
                    logger.warning(f"Error loading volume cache for {exchange_id}: {e}")

            # Load timestamp cache
            timestamp_file = os.path.join(exchange_dir, "timestamps.json")
            if os.path.exists(timestamp_file):
                try:
                    with open(timestamp_file, "r") as f:
                        self._timestamp_cache[exchange_id] = json.load(f)
                        logger.debug(
                            f"Loaded timestamps for {exchange_id}: {len(self._timestamp_cache[exchange_id])} symbols"
                        )
                except Exception as e:
                    logger.warning(f"Error loading timestamps for {exchange_id}: {e}")

            # Load failed symbols
            failed_file = os.path.join(exchange_dir, "failed_symbols.json")
            if os.path.exists(failed_file):
                try:
                    with open(failed_file, "r") as f:
                        failed_list = json.load(f)
                        self._failed_symbols[exchange_id] = set(failed_list)
                        logger.debug(
                            f"Loaded failed symbols for {exchange_id}: {len(failed_list)} symbols"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error loading failed symbols for {exchange_id}: {e}"
                    )

    def _is_volume_cache_valid(self, file_path: str) -> bool:
        """Check if volume cache file is still valid (updated today after UTC midnight)."""
        try:
            # Load the volume file to check internal timestamp
            with open(file_path, "r") as f:
                data = json.load(f)

            # Check if metadata exists with last_updated timestamp
            if isinstance(data, dict) and "_metadata" in data:
                last_updated_str = data["_metadata"].get("last_updated_utc")
                if last_updated_str:
                    try:
                        last_updated = datetime.fromisoformat(
                            last_updated_str.replace("Z", "+00:00")
                        )
                        # Convert to UTC if not already
                        if last_updated.tzinfo is None:
                            last_updated = last_updated.replace(tzinfo=timezone.utc)

                        # Get current UTC time
                        now_utc = datetime.now(timezone.utc)

                        # Check if last update was today (after UTC midnight)
                        today_utc_start = now_utc.replace(
                            hour=0, minute=0, second=0, microsecond=0
                        )

                        is_fresh = last_updated >= today_utc_start

                        if is_fresh:
                            logger.debug(
                                f"Volume cache is fresh - updated {last_updated_str}"
                            )
                        else:
                            logger.debug(
                                f"Volume cache is stale - last updated {last_updated_str}, need update after {today_utc_start}"
                            )

                        return is_fresh

                    except (ValueError, TypeError) as e:
                        logger.debug(f"Error parsing timestamp in volume cache: {e}")

            # Fallback to file modification time if no internal timestamp
            file_age = time.time() - os.path.getmtime(file_path)
            is_valid = file_age < 86400  # 24 hours in seconds

            if not is_valid:
                logger.debug(
                    f"Volume cache fallback check: file is {file_age/3600:.1f} hours old"
                )

            return is_valid

        except (OSError, json.JSONDecodeError) as e:
            logger.debug(f"Error checking volume cache validity: {e}")
            return False

    def _save_to_disk(self, exchange_id: str, cache_type: str, data: Any) -> bool:
        """Save cache data to disk atomically."""
        try:
            exchange_dir = self._get_exchange_dir(exchange_id)
            
            # Ensure the exchange directory exists
            if not os.path.exists(exchange_dir):
                try:
                    os.makedirs(exchange_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"Failed to create exchange directory {exchange_dir}: {e}")
                    return False
            
            file_path = os.path.join(exchange_dir, f"{cache_type}.json")
            temp_path = f"{file_path}.tmp"

            # Convert sets to lists for JSON serialization
            if cache_type == "failed_symbols" and isinstance(data, set):
                data = list(data)

            # Atomic write with better error handling
            try:
                with open(temp_path, "w") as f:
                    json.dump(data, f, indent=2)
                
                # Only replace if write was successful
                os.replace(temp_path, file_path)
                logger.debug(f"Saved {cache_type} for {exchange_id}")
                return True
                
            except OSError as e:
                logger.error(f"Failed to write {cache_type} file for {exchange_id}: {e}")
                # Clean up temp file if it exists
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass  # Ignore cleanup errors
                return False

        except Exception as e:
            logger.error(f"Error saving {cache_type} for {exchange_id}: {e}")
            return False

    # Volume cache methods
    def get_volume(self, exchange_id: str, symbol: str) -> Optional[float]:
        """Get volume for a symbol if cached and valid."""
        if (
            exchange_id in self._volume_cache
            and symbol in self._volume_cache[exchange_id]
        ):
            volume_data = self._volume_cache[exchange_id][symbol]
            if isinstance(volume_data, dict) and "volume" in volume_data:
                return volume_data["volume"]
            elif isinstance(volume_data, (int, float)):
                return float(volume_data)
        return None

    def get_all_volumes(self, exchange_id: str) -> Dict[str, float]:
        """Get all volume data for an exchange."""
        if exchange_id not in self._volume_cache:
            return {}

        result = {}
        for symbol, data in self._volume_cache[exchange_id].items():
            if isinstance(data, dict) and "volume" in data:
                result[symbol] = data["volume"]
            elif isinstance(data, (int, float)):
                result[symbol] = float(data)
        return result

    def save_volume(self, exchange_id: str, symbol: str, volume: float) -> bool:
        """Save volume data for a symbol."""
        try:
            volume = float(volume)
            if volume <= 0:
                return False

            if exchange_id not in self._volume_cache:
                self._volume_cache[exchange_id] = {}

            self._volume_cache[exchange_id][symbol] = {
                "volume": volume,
                "updated": datetime.now(timezone.utc).isoformat(),
            }

            # When saving individual volume, we need to preserve existing data and metadata
            # Load existing data first
            exchange_dir = self._get_exchange_dir(exchange_id)
            volume_file = os.path.join(exchange_dir, "volume.json")

            existing_data = {}
            if os.path.exists(volume_file):
                try:
                    with open(volume_file, "r") as f:
                        file_data = json.load(f)
                        if isinstance(file_data, dict) and "_metadata" in file_data:
                            existing_data = {
                                k: v
                                for k, v in file_data.items()
                                if not k.startswith("_")
                            }
                        else:
                            existing_data = file_data
                except Exception:
                    pass

            # Merge with existing data
            existing_data.update(self._volume_cache[exchange_id])

            # Create updated cache data with metadata
            volume_cache_data = {
                "_metadata": {
                    "last_updated_utc": datetime.now(timezone.utc).isoformat(),
                    "exchange_id": exchange_id,
                    "total_symbols": len(existing_data),
                    "cache_version": "2.0",
                },
                **existing_data,
            }

            return self._save_to_disk(exchange_id, "volume", volume_cache_data)

        except (ValueError, TypeError):
            logger.warning(f"Invalid volume data for {exchange_id}:{symbol}: {volume}")
            return False

    def save_all_volumes(self, exchange_id: str, volumes: Dict[str, float]) -> bool:
        """Save all volume data for an exchange with UTC timestamp metadata."""
        normalized_data = {}
        current_time_utc = datetime.now(timezone.utc).isoformat()

        for symbol, volume in volumes.items():
            try:
                volume = float(volume)
                if volume > 0:
                    normalized_data[symbol] = {
                        "volume": volume,
                        "updated": current_time_utc,
                    }
            except (ValueError, TypeError):
                logger.warning(f"Skipping invalid volume for {symbol}: {volume}")

        # Add metadata with UTC timestamp for freshness checking
        volume_cache_data = {
            "_metadata": {
                "last_updated_utc": current_time_utc,
                "exchange_id": exchange_id,
                "total_symbols": len(normalized_data),
                "cache_version": "2.0",
            },
            **normalized_data,
        }

        self._volume_cache[exchange_id] = normalized_data  # Keep in-memory cache clean
        return self._save_to_disk(exchange_id, "volume", volume_cache_data)

    def is_volume_cache_fresh(self, exchange_id: str) -> bool:
        """Check if volume cache exists and is fresh (less than 24 hours old)."""
        exchange_dir = self._get_exchange_dir(exchange_id)
        volume_file = os.path.join(exchange_dir, "volume.json")
        return os.path.exists(volume_file) and self._is_volume_cache_valid(volume_file)

    # Timestamp cache methods
    def get_timestamp(self, exchange_id: str, symbol: str) -> Optional[int]:
        """Get inception timestamp for a symbol."""
        if (
            exchange_id in self._timestamp_cache
            and symbol in self._timestamp_cache[exchange_id]
        ):
            return self._timestamp_cache[exchange_id][symbol]
        return None

    def get_all_timestamps(self, exchange_id: str) -> Dict[str, int]:
        """Get all timestamps for an exchange."""
        return self._timestamp_cache.get(exchange_id, {})

    def save_timestamp(self, exchange_id: str, symbol: str, timestamp: int) -> bool:
        """Save inception timestamp for a symbol."""
        try:
            timestamp = int(timestamp)
            if exchange_id not in self._timestamp_cache:
                self._timestamp_cache[exchange_id] = {}

            self._timestamp_cache[exchange_id][symbol] = timestamp
            return self._save_to_disk(
                exchange_id, "timestamps", self._timestamp_cache[exchange_id]
            )

        except (ValueError, TypeError):
            logger.warning(f"Invalid timestamp for {exchange_id}:{symbol}: {timestamp}")
            return False

    def save_all_timestamps(self, exchange_id: str, timestamps: Dict[str, int]) -> bool:
        """Save all timestamps for an exchange."""
        normalized_data = {}
        for symbol, timestamp in timestamps.items():
            try:
                normalized_data[symbol] = int(timestamp)
            except (ValueError, TypeError):
                logger.warning(f"Skipping invalid timestamp for {symbol}: {timestamp}")

        self._timestamp_cache[exchange_id] = normalized_data
        return self._save_to_disk(exchange_id, "timestamps", normalized_data)

    # Failed symbols methods
    def is_failed_symbol(self, exchange_id: str, symbol: str) -> bool:
        """Check if a symbol is marked as failed."""
        return (
            exchange_id in self._failed_symbols
            and symbol in self._failed_symbols[exchange_id]
        )

    def add_failed_symbol(self, exchange_id: str, symbol: str) -> bool:
        """Mark a symbol as failed."""
        if exchange_id not in self._failed_symbols:
            self._failed_symbols[exchange_id] = set()

        self._failed_symbols[exchange_id].add(symbol)
        return self._save_to_disk(
            exchange_id, "failed_symbols", self._failed_symbols[exchange_id]
        )

    def get_failed_symbols(self, exchange_id: str) -> Set[str]:
        """Get all failed symbols for an exchange."""
        return self._failed_symbols.get(exchange_id, set())

    def filter_failed_symbols(self, exchange_id: str, symbols: List[str]) -> List[str]:
        """Filter out failed symbols from a list."""
        failed = self.get_failed_symbols(exchange_id)
        filtered = [s for s in symbols if s not in failed]

        if len(filtered) < len(symbols):
            logger.info(
                f"Filtered out {len(symbols) - len(filtered)} failed symbols for {exchange_id}"
            )

        return filtered

    # General cache methods
    def clear_cache(
        self, exchange_id: Optional[str] = None, cache_type: Optional[str] = None
    ) -> bool:
        """Clear cache data."""
        success = True

        if exchange_id:
            exchanges = [exchange_id]
        else:
            exchanges = list(
                set(
                    list(self._volume_cache.keys())
                    + list(self._timestamp_cache.keys())
                    + list(self._failed_symbols.keys())
                )
            )

        for exchange in exchanges:
            if not cache_type or cache_type == "volume":
                if exchange in self._volume_cache:
                    del self._volume_cache[exchange]
                if not self._save_to_disk(exchange, "volume", {}):
                    success = False

            if not cache_type or cache_type == "timestamps":
                if exchange in self._timestamp_cache:
                    del self._timestamp_cache[exchange]
                if not self._save_to_disk(exchange, "timestamps", {}):
                    success = False

            if not cache_type or cache_type == "failed_symbols":
                if exchange in self._failed_symbols:
                    del self._failed_symbols[exchange]
                if not self._save_to_disk(exchange, "failed_symbols", set()):
                    success = False

        cache_desc = f"{cache_type} cache" if cache_type else "all caches"
        exchange_desc = f"for {exchange_id}" if exchange_id else "for all exchanges"
        logger.info(f"Cleared {cache_desc} {exchange_desc}")

        return success

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        all_exchanges = set(
            list(self._volume_cache.keys())
            + list(self._timestamp_cache.keys())
            + list(self._failed_symbols.keys())
        )

        stats = {
            "exchanges": list(all_exchanges),
            "volume_symbols": sum(len(data) for data in self._volume_cache.values()),
            "timestamp_symbols": sum(
                len(data) for data in self._timestamp_cache.values()
            ),
            "failed_symbols": sum(len(data) for data in self._failed_symbols.values()),
            "by_exchange": {},
        }

        for exchange in all_exchanges:
            stats["by_exchange"][exchange] = {
                "volume_symbols": len(self._volume_cache.get(exchange, {})),
                "timestamp_symbols": len(self._timestamp_cache.get(exchange, {})),
                "failed_symbols": len(self._failed_symbols.get(exchange, set())),
                "volume_cache_fresh": self.is_volume_cache_fresh(exchange),
            }

        return stats

    def list_exchanges(self) -> List[str]:
        """List all exchanges with cached data."""
        return list(
            set(
                list(self._volume_cache.keys())
                + list(self._timestamp_cache.keys())
                + list(self._failed_symbols.keys())
            )
        )


# Global instance
cache_manager = SimpleCacheManager()
