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
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# Default cache directory
DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache')

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
        self._volume_cache = {}      # exchange -> symbol -> {volume, timestamp}
        self._timestamp_cache = {}   # exchange -> symbol -> timestamp
        self._failed_symbols = {}    # exchange -> set of failed symbols
        
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
        blacklist_file = os.path.join(self.cache_dir, 'blacklist.json')
        
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
                    "USDP/USDC"
                ],
                "binance": [
                    # Binance-specific stable pairs
                    "BUSD/FDUSD",
                    "FDUSD/USDT"
                ],
                "bybit": [
                    # Bybit futures stable pairs
                    "USDC/USDT:USDT",
                    "BUSD/USDT:USDT"
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
                "_last_updated": datetime.now().isoformat()
            }
            
            try:
                with open(blacklist_file, 'w') as f:
                    json.dump(default_blacklist, f, indent=2)
                logger.info(f"✅ Created default blacklist with {len(default_blacklist['global'])} global exclusions")
                logger.info(f"🚫 Default global blacklist: {default_blacklist['global'][:5]}{'...' if len(default_blacklist['global']) > 5 else ''}")
            except Exception as e:
                logger.warning(f"⚠️ Could not create default blacklist.json: {e}")
        else:
            # Validate existing blacklist structure and update if needed
            try:
                with open(blacklist_file, 'r') as f:
                    existing_blacklist = json.load(f)
                
                # Ensure USDC/USDT is in global blacklist (user requirement)
                needs_update = False
                if isinstance(existing_blacklist, dict):
                    global_list = existing_blacklist.get('global', [])
                    if 'USDC/USDT' not in global_list:
                        global_list.append('USDC/USDT')
                        existing_blacklist['global'] = global_list
                        needs_update = True
                        logger.info("🔄 Adding USDC/USDT to existing global blacklist")
                
                # Update last_updated timestamp if we made changes
                if needs_update:
                    existing_blacklist['_last_updated'] = datetime.now().isoformat()
                    with open(blacklist_file, 'w') as f:
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
        blacklist_file = os.path.join(self.cache_dir, 'blacklist.json')
        
        try:
            if os.path.exists(blacklist_file):
                with open(blacklist_file, 'r') as f:
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
        global_blacklist = blacklist_data.get('global', [])
        if symbol in global_blacklist:
            return True
        
        # Check exchange-specific blacklist
        if exchange_id:
            exchange_blacklist = blacklist_data.get(exchange_id.lower(), [])
            if symbol in exchange_blacklist:
                return True
        
        return False
    
    def filter_blacklisted_symbols(self, symbols: List[str], exchange_id: str = None) -> List[str]:
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
        blacklisted = set(blacklist_data.get('global', []))
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
            volume_file = os.path.join(exchange_dir, 'volume.json')
            if os.path.exists(volume_file):
                try:
                    with open(volume_file, 'r') as f:
                        data = json.load(f)
                        # Check if cache is still valid (less than 24 hours old)
                        if self._is_volume_cache_valid(volume_file):
                            self._volume_cache[exchange_id] = data
                            logger.debug(f"Loaded volume cache for {exchange_id}: {len(data)} symbols")
                        else:
                            logger.debug(f"Volume cache for {exchange_id} expired, will refresh")
                except Exception as e:
                    logger.warning(f"Error loading volume cache for {exchange_id}: {e}")
            
            # Load timestamp cache
            timestamp_file = os.path.join(exchange_dir, 'timestamps.json')
            if os.path.exists(timestamp_file):
                try:
                    with open(timestamp_file, 'r') as f:
                        self._timestamp_cache[exchange_id] = json.load(f)
                        logger.debug(f"Loaded timestamps for {exchange_id}: {len(self._timestamp_cache[exchange_id])} symbols")
                except Exception as e:
                    logger.warning(f"Error loading timestamps for {exchange_id}: {e}")
            
            # Load failed symbols
            failed_file = os.path.join(exchange_dir, 'failed_symbols.json')
            if os.path.exists(failed_file):
                try:
                    with open(failed_file, 'r') as f:
                        failed_list = json.load(f)
                        self._failed_symbols[exchange_id] = set(failed_list)
                        logger.debug(f"Loaded failed symbols for {exchange_id}: {len(failed_list)} symbols")
                except Exception as e:
                    logger.warning(f"Error loading failed symbols for {exchange_id}: {e}")
    
    def _is_volume_cache_valid(self, file_path: str) -> bool:
        """Check if volume cache file is still valid (less than 24 hours old)."""
        try:
            file_age = time.time() - os.path.getmtime(file_path)
            return file_age < 86400  # 24 hours in seconds
        except OSError:
            return False
    
    def _save_to_disk(self, exchange_id: str, cache_type: str, data: Any) -> bool:
        """Save cache data to disk atomically."""
        try:
            exchange_dir = self._get_exchange_dir(exchange_id)
            file_path = os.path.join(exchange_dir, f'{cache_type}.json')
            temp_path = f'{file_path}.tmp'
            
            # Convert sets to lists for JSON serialization
            if cache_type == 'failed_symbols' and isinstance(data, set):
                data = list(data)
            
            # Atomic write
            with open(temp_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            os.replace(temp_path, file_path)
            logger.debug(f"Saved {cache_type} for {exchange_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving {cache_type} for {exchange_id}: {e}")
            return False
    
    # Volume cache methods
    def get_volume(self, exchange_id: str, symbol: str) -> Optional[float]:
        """Get volume for a symbol if cached and valid."""
        if exchange_id in self._volume_cache and symbol in self._volume_cache[exchange_id]:
            volume_data = self._volume_cache[exchange_id][symbol]
            if isinstance(volume_data, dict) and 'volume' in volume_data:
                return volume_data['volume']
            elif isinstance(volume_data, (int, float)):
                return float(volume_data)
        return None
    
    def get_all_volumes(self, exchange_id: str) -> Dict[str, float]:
        """Get all volume data for an exchange."""
        if exchange_id not in self._volume_cache:
            return {}
            
        result = {}
        for symbol, data in self._volume_cache[exchange_id].items():
            if isinstance(data, dict) and 'volume' in data:
                result[symbol] = data['volume']
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
                'volume': volume,
                'updated': datetime.now().isoformat()
            }
            
            return self._save_to_disk(exchange_id, 'volume', self._volume_cache[exchange_id])
            
        except (ValueError, TypeError):
            logger.warning(f"Invalid volume data for {exchange_id}:{symbol}: {volume}")
            return False
    
    def save_all_volumes(self, exchange_id: str, volumes: Dict[str, float]) -> bool:
        """Save all volume data for an exchange."""
        normalized_data = {}
        current_time = datetime.now().isoformat()
        
        for symbol, volume in volumes.items():
            try:
                volume = float(volume)
                if volume > 0:
                    normalized_data[symbol] = {
                        'volume': volume,
                        'updated': current_time
                    }
            except (ValueError, TypeError):
                logger.warning(f"Skipping invalid volume for {symbol}: {volume}")
        
        self._volume_cache[exchange_id] = normalized_data
        return self._save_to_disk(exchange_id, 'volume', normalized_data)
    
    def is_volume_cache_fresh(self, exchange_id: str) -> bool:
        """Check if volume cache exists and is fresh (less than 24 hours old)."""
        exchange_dir = self._get_exchange_dir(exchange_id)
        volume_file = os.path.join(exchange_dir, 'volume.json')
        return os.path.exists(volume_file) and self._is_volume_cache_valid(volume_file)
    
    # Timestamp cache methods
    def get_timestamp(self, exchange_id: str, symbol: str) -> Optional[int]:
        """Get inception timestamp for a symbol."""
        if exchange_id in self._timestamp_cache and symbol in self._timestamp_cache[exchange_id]:
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
            return self._save_to_disk(exchange_id, 'timestamps', self._timestamp_cache[exchange_id])
            
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
        return self._save_to_disk(exchange_id, 'timestamps', normalized_data)
    
    # Failed symbols methods
    def is_failed_symbol(self, exchange_id: str, symbol: str) -> bool:
        """Check if a symbol is marked as failed."""
        return (exchange_id in self._failed_symbols and 
                symbol in self._failed_symbols[exchange_id])
    
    def add_failed_symbol(self, exchange_id: str, symbol: str) -> bool:
        """Mark a symbol as failed."""
        if exchange_id not in self._failed_symbols:
            self._failed_symbols[exchange_id] = set()
        
        self._failed_symbols[exchange_id].add(symbol)
        return self._save_to_disk(exchange_id, 'failed_symbols', self._failed_symbols[exchange_id])
    
    def get_failed_symbols(self, exchange_id: str) -> Set[str]:
        """Get all failed symbols for an exchange."""
        return self._failed_symbols.get(exchange_id, set())
    
    def filter_failed_symbols(self, exchange_id: str, symbols: List[str]) -> List[str]:
        """Filter out failed symbols from a list."""
        failed = self.get_failed_symbols(exchange_id)
        filtered = [s for s in symbols if s not in failed]
        
        if len(filtered) < len(symbols):
            logger.info(f"Filtered out {len(symbols) - len(filtered)} failed symbols for {exchange_id}")
        
        return filtered
    
    # General cache methods
    def clear_cache(self, exchange_id: Optional[str] = None, cache_type: Optional[str] = None) -> bool:
        """Clear cache data."""
        success = True
        
        if exchange_id:
            exchanges = [exchange_id]
        else:
            exchanges = list(set(list(self._volume_cache.keys()) + 
                               list(self._timestamp_cache.keys()) + 
                               list(self._failed_symbols.keys())))
        
        for exchange in exchanges:
            if not cache_type or cache_type == 'volume':
                if exchange in self._volume_cache:
                    del self._volume_cache[exchange]
                if not self._save_to_disk(exchange, 'volume', {}):
                    success = False
            
            if not cache_type or cache_type == 'timestamps':
                if exchange in self._timestamp_cache:
                    del self._timestamp_cache[exchange]
                if not self._save_to_disk(exchange, 'timestamps', {}):
                    success = False
            
            if not cache_type or cache_type == 'failed_symbols':
                if exchange in self._failed_symbols:
                    del self._failed_symbols[exchange]
                if not self._save_to_disk(exchange, 'failed_symbols', set()):
                    success = False
        
        cache_desc = f"{cache_type} cache" if cache_type else "all caches"
        exchange_desc = f"for {exchange_id}" if exchange_id else "for all exchanges"
        logger.info(f"Cleared {cache_desc} {exchange_desc}")
        
        return success
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        all_exchanges = set(list(self._volume_cache.keys()) + 
                           list(self._timestamp_cache.keys()) + 
                           list(self._failed_symbols.keys()))
        
        stats = {
            'exchanges': list(all_exchanges),
            'volume_symbols': sum(len(data) for data in self._volume_cache.values()),
            'timestamp_symbols': sum(len(data) for data in self._timestamp_cache.values()),
            'failed_symbols': sum(len(data) for data in self._failed_symbols.values()),
            'by_exchange': {}
        }
        
        for exchange in all_exchanges:
            stats['by_exchange'][exchange] = {
                'volume_symbols': len(self._volume_cache.get(exchange, {})),
                'timestamp_symbols': len(self._timestamp_cache.get(exchange, {})),
                'failed_symbols': len(self._failed_symbols.get(exchange, set())),
                'volume_cache_fresh': self.is_volume_cache_fresh(exchange)
            }
        
        return stats
    
    def list_exchanges(self) -> List[str]:
        """List all exchanges with cached data."""
        return list(set(list(self._volume_cache.keys()) + 
                       list(self._timestamp_cache.keys()) + 
                       list(self._failed_symbols.keys())))


# Global instance
cache_manager = SimpleCacheManager() 