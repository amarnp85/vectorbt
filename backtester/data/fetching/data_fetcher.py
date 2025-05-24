#!/usr/bin/env python3
"""Data Fetcher Module

Simplified data fetching with clear fallback chain:
1. Check cache for existing data
2. Try resampling from lower timeframes if symbols missing
3. Fetch from exchange as final fallback
4. Save results to cache

Uses VectorBT Pro's native data fetching and persistence capabilities.
Leverages VBT's built-in caching, symbol alignment, and data management.

=== CRITICAL REFACTORING NOTES ===

This module underwent extensive refactoring to address several complex issues.
Future developers should be aware of these challenges and their solutions:

## 1. TIMEZONE WARNINGS ELIMINATION
ISSUE: "Cannot subtract tz-naive and tz-aware datetime" warnings
CAUSE: Mixing timezone-naive and timezone-aware datetime objects in _is_data_fresh()
SOLUTION: All datetime operations now use VBT's timezone utilities with proper UTC conversion
LOCATION: _is_data_fresh() function
KEY FIX: str(latest_dt.tz) comparison instead of .zone attribute

## 2. MIXED SYMBOL AVAILABILITY LOGIC 
ISSUE: Cache said "missing 10 symbols" but then "fetched 20 symbols"
CAUSE: Original logic fell back to fetching ALL symbols when ANY were missing
SOLUTION: New _fetch_missing_symbols_and_merge() function that fetches ONLY missing symbols
LOCATION: _try_cache_fetch() function
KEY FIX: Smart merge of cached + newly fetched data instead of re-fetching everything

## 3. START DATE CALCULATION ERRORS
ISSUE: Used earliest start date from ALL symbols instead of just symbols being fetched
CAUSE: symbol_start_dates contained dates for all requested symbols, not just those being fetched
SOLUTION: Filter start dates to only symbols actually being fetched in each function
LOCATIONS: _try_resampling_fetch() and _fetch_from_exchange()
KEY FIX: fetch_symbol_start_dates = {symbol: dates[symbol] for symbol in symbols_to_fetch}

## 4. VBT SYMBOL NAME PRESERVATION
ISSUE: Merged data showed generic 'symbol' instead of actual symbol names like 'BTC/USDT'
CAUSE: VBT's Data.from_data() doesn't preserve symbol names when using pandas concat
SOLUTION: Use VBT's symbol dictionary approach for proper symbol recognition
LOCATION: _fetch_missing_symbols_and_merge() function
KEY FIX: Create symbol_dict with individual DataFrames per symbol, not concatenated DataFrame

## 5. VBT DATA STRUCTURE HANDLING
ISSUE: VBT's .get() method returns tuples instead of DataFrames in some cases
CAUSE: VBT sometimes returns (dataframe, metadata) tuples
SOLUTION: extract_dataframe() helper function to handle tuple extraction
LOCATION: _fetch_missing_symbols_and_merge() function
KEY FIX: Iterate through tuple elements to find valid DataFrame objects

## 6. UPDATE METHOD ROBUSTNESS
ISSUE: VBT's update() method not implemented for all data types, causing NotImplementedError
CAUSE: Some VBT data types don't support native update() functionality
SOLUTION: Multiple fallback strategies: VBT update() → manual fetch + merge → graceful error handling
LOCATION: update_data() function
KEY FIX: Try VBT native update, fall back to manual date-based fetching and concatenation

## 7. LOGGING CONSISTENCY
ISSUE: Inconsistent symbol counts and success messages between cache hits and mixed fetches
CAUSE: Success logging didn't distinguish between different fetch types
SOLUTION: Proper success message differentiation and accurate metrics calculation
LOCATION: fetch_data() main function and success logging
KEY FIX: Check result symbols vs requested symbols to determine operation type

=== VBT-SPECIFIC GOTCHAS ===

- VBT expects symbol dictionaries for multi-symbol data: {symbol: dataframe_with_features}
- Each symbol's DataFrame should have feature names as columns (open, high, low, close, volume)
- VBT's concat() method has strict parameter requirements (no 'keys' parameter)
- VBT's .get() method may return tuples requiring careful extraction
- VBT's Data.from_data() loses symbol names unless properly structured
- VBT's update() method is not implemented for all data types

=== PERFORMANCE OPTIMIZATIONS ===

- Only fetch missing symbols instead of re-fetching cached ones
- Use parallel processing for multiple symbol fetches
- Proper timezone handling eliminates conversion overhead
- Smart caching prevents unnecessary API calls
- Graceful fallbacks ensure robustness without performance degradation

=== TESTING VALIDATION ===

The following test scenarios validate all fixes:
1. Mixed availability: some cached, some missing symbols
2. All cached: proper cache hit detection
3. All missing: proper exchange fetch
4. Non-existent symbols: graceful error handling
5. Timezone edge cases: no warnings in datetime operations
6. Update scenarios: multiple fallback strategies working

Future modifications should test all these scenarios to prevent regressions.
"""

import logging
import pandas as pd
from typing import List, Optional, Union, Dict, Any, Tuple
import vectorbtpro as vbt
from ..storage.data_storage import data_storage
from ..cache_system import cache_manager
from .storage_resampling import fetch_with_storage_resampling_fallback
from datetime import datetime
import os
import json

# Rich imports for logging
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text
from rich.columns import Columns

logger = logging.getLogger(__name__)
console = Console()

# Logging state tracker
class FetchLogger:
    """Track and display fetch operation state in a structured way."""
    
    def __init__(self, symbols: List[str], exchange_id: str, timeframe: str):
        self.requested_symbols = symbols
        self.exchange_id = exchange_id
        self.timeframe = timeframe
        self.filtered_symbols = []
        self.operation_path = []
        self.cache_state = {}
        self.final_result = None
        self.cache_was_stale = False  # Track if cache staleness was detected
        # Track per-symbol information with more accurate tracking
        self.symbol_info = {symbol: {
            'inception_timestamp': None,
            'earliest_timestamp': None,
            'latest_timestamp': None,
            'total_candles': 0,
            'new_candles': 0,
            'source': 'unknown',
            'is_stale': False,
            'was_cached': False,  # Track if symbol came from cache
            'was_fetched': False,  # Track if symbol was fetched
            'cache_end_time': None,  # Track cache boundary for new candles calculation
        } for symbol in symbols}
        # Track which symbols came from which sources
        self.symbol_sources = {}
        
    def log_start(self, start_date: Optional[str], end_date: Optional[str]):
        """Log the start of a fetch operation."""
        # Store target end date for staleness checking
        self._target_end_date = end_date
        
        # Create a compact summary panel
        date_range = f"{start_date or 'inception'} → {end_date or 'now'}"
        
        console.print(
            Panel(
                f"[bold cyan]{self.exchange_id.upper()}[/bold cyan] | "
                f"[cyan]{self.timeframe}[/cyan] | "
                f"[cyan]{len(self.requested_symbols)} symbols[/cyan] | "
                f"[dim]{date_range}[/dim]",
                title="📊 Data Fetch",
                expand=False,
                style="cyan"
            )
        )
        
    def log_filter_result(self, filtered_symbols: List[str]):
        """Log blacklist filtering results."""
        self.filtered_symbols = filtered_symbols
        if len(filtered_symbols) < len(self.requested_symbols):
            removed = len(self.requested_symbols) - len(filtered_symbols)
            console.print(f"   🚫 Filtered: {removed} blacklisted → {len(filtered_symbols)} symbols")
            
    def log_cache_result(self, status: str, details: Dict[str, Any]):
        """Log cache check results."""
        self.cache_state = details
        self.operation_path.append(('cache', status))
        
        icon = "✅" if status == "hit" else "🔍" if status == "partial" else "❌"
        
        if status == "hit":
            console.print(f"   {icon} Cache: [green]Complete[/green]")
        elif status == "partial":
            console.print(
                f"   {icon} Cache: [yellow]Partial[/yellow] "
                f"({details['available']}/{details['requested']} symbols)"
            )
        elif status == "stale":
            self.cache_was_stale = True  # Track that cache was stale
            if details.get("selective_update"):
                # Selective update case
                stale_count = details.get("stale_count", 0)
                fresh_count = details.get("fresh_count", 0)
                console.print(f"   {icon} Cache: [yellow]Selective Update[/yellow] ({stale_count} stale, {fresh_count} fresh)")
            else:
                # Complete update case
                reason = details.get("reason", "updating...")
                console.print(f"   {icon} Cache: [yellow]Stale[/yellow] ({reason})")
        else:
            console.print(f"   {icon} Cache: [red]Miss[/red]")
            
    def log_resample_result(self, success: bool, source_tf: Optional[str] = None):
        """Log resampling attempt results."""
        self.operation_path.append(('resample', 'success' if success else 'fail'))
        
        if success and source_tf:
            console.print(f"   ✅ Resample: [green]Success[/green] (from {source_tf})")
        elif success:
            console.print(f"   ✅ Resample: [green]Success[/green]")
        else:
            console.print(f"   ⏭️  Resample: [dim]Not available[/dim]")
            
    def log_exchange_result(self, success: bool, symbol_count: Optional[int] = None):
        """Log exchange fetch results."""
        self.operation_path.append(('exchange', 'success' if success else 'fail'))
        
        if success:
            msg = f"   ✅ Exchange: [green]Success[/green]"
            if symbol_count:
                msg += f" ({symbol_count} symbols)"
            console.print(msg)
        else:
            console.print(f"   ❌ Exchange: [red]Failed[/red]")
            
    def update_symbol_info(self, symbol: str, info: Dict[str, Any]):
        """Update information for a specific symbol."""
        if symbol in self.symbol_info:
            self.symbol_info[symbol].update(info)
            
    def set_symbol_source(self, symbol: str, source: str, was_cached: bool = False, was_fetched: bool = False, cache_end_time=None):
        """Set the source for a specific symbol with detailed tracking."""
        self.symbol_sources[symbol] = source
        if symbol in self.symbol_info:
            self.symbol_info[symbol]['source'] = source
            self.symbol_info[symbol]['was_cached'] = was_cached
            self.symbol_info[symbol]['was_fetched'] = was_fetched
            if cache_end_time is not None:
                self.symbol_info[symbol]['cache_end_time'] = cache_end_time
            
    def log_final_result(self, data: Optional[vbt.Data], saved: bool = False):
        """Log the final result of the fetch operation."""
        self.final_result = data
        
        if data is None:
            console.print(
                Panel(
                    "[red]Fetch failed[/red] - No data retrieved",
                    style="red",
                    expand=False
                )
            )
            return
            
        # Update symbol information from final data
        self._update_symbol_info_from_data(data)
        
        # CRITICAL FIX: Handle cases where merge failed and some symbols missing from final data
        self._fill_missing_symbol_info_from_cache(data)
        
        # Determine the fetch path taken
        path_desc = self._get_path_description()
        
        # Build result summary
        symbol_count = len(data.symbols) if hasattr(data, 'symbols') else 0
        
        result_text = f"[green]✓ Complete[/green] | {symbol_count} symbols | {path_desc}"
        if saved:
            result_text += " | 💾 Cached"
            
        console.print(
            Panel(
                result_text,
                expand=False,
                style="green"
            )
        )
        
        # Display detailed symbol table
        self._display_symbol_table()
        
    def _update_symbol_info_from_data(self, data: vbt.Data):
        """Extract symbol information from the final data."""
        if not hasattr(data, 'symbols'):
            return
            
        # Get inception timestamps and volume data from cache
        from ..cache_system import cache_manager
        inception_timestamps = cache_manager.get_all_timestamps(self.exchange_id)
        volume_data = cache_manager.get_all_volumes(self.exchange_id)
        
        # Determine target end time for staleness check
        target_end_time = None
        if hasattr(self, '_target_end_date'):
            try:
                if isinstance(self._target_end_date, str):
                    if self._target_end_date.lower() == "now":
                        target_end_time = vbt.utc_timestamp()
                    else:
                        target_end_time = vbt.timestamp(self._target_end_date, tz='UTC')
                else:
                    target_end_time = vbt.timestamp(self._target_end_date, tz='UTC') if self._target_end_date else None
            except Exception as e:
                pass
        
        for symbol in data.symbols:
            if symbol not in self.symbol_info:
                self.symbol_info[symbol] = {
                    'inception_timestamp': None,
                    'earliest_timestamp': None,
                    'latest_timestamp': None,
                    'total_candles': 0,
                    'new_candles': 0,
                    'volume_usd': 0,
                    'source': 'unknown',
                    'is_stale': False
                }
            
            # Get inception timestamp from cache
            if inception_timestamps and symbol in inception_timestamps:
                self.symbol_info[symbol]['inception_timestamp'] = inception_timestamps[symbol]
            
            # Get volume data from cache
            if volume_data and symbol in volume_data:
                self.symbol_info[symbol]['volume_usd'] = volume_data[symbol]
            
            # Get data range for this symbol
            try:
                if len(data.symbols) > 1:
                    symbol_data = data.close[symbol].dropna()
                else:
                    symbol_data = data.close.dropna()
                
                if len(symbol_data) > 0:
                    self.symbol_info[symbol]['earliest_timestamp'] = symbol_data.index[0]
                    self.symbol_info[symbol]['latest_timestamp'] = symbol_data.index[-1]
                    self.symbol_info[symbol]['total_candles'] = len(symbol_data)
                    
                    # Check per-symbol staleness using exact candle boundary awareness
                    if target_end_time is not None:
                        try:
                            latest_ts = symbol_data.index[-1]
                            
                            # Ensure timezone-aware comparison
                            if latest_ts.tz is None:
                                latest_ts = latest_ts.tz_localize('UTC')
                            elif str(latest_ts.tz) != 'UTC':
                                latest_ts = latest_ts.tz_convert('UTC')
                            
                            # Use exact candle boundary calculation
                            expected_latest_candle = _calculate_latest_complete_candle_time(target_end_time, self.timeframe)
                            
                            # Special handling for daily timeframes
                            if self.timeframe.endswith('d') or 'day' in self.timeframe.lower():
                                # For daily: stale if different day
                                is_stale = target_end_time.date() > latest_ts.date()
                            else:
                                # For intraday: stale if missing expected latest candle
                                is_stale = latest_ts < expected_latest_candle
                            
                            self.symbol_info[symbol]['is_stale'] = is_stale
                            
                        except Exception as e:
                            logger.debug(f"Error in staleness check for {symbol}: {e}")
                            self.symbol_info[symbol]['is_stale'] = False
                    
                    # Calculate new candles based on per-symbol tracking
                    new_candles = 0
                    if self.symbol_info[symbol]['was_cached'] and self.symbol_info[symbol]['cache_end_time']:
                        # Symbol came from cache - count only candles after cache end time
                        try:
                            cache_end = self.symbol_info[symbol]['cache_end_time']
                            if hasattr(cache_end, 'tz') and cache_end.tz is not None:
                                # Already timezone aware
                                pass
                            else:
                                cache_end = cache_end.tz_localize('UTC') if cache_end.tz is None else cache_end
                            
                            # Count candles after cache end time
                            new_data_mask = symbol_data.index > cache_end
                            new_candles = new_data_mask.sum()
                        except Exception as e:
                            logger.debug(f"Error calculating new candles for cached {symbol}: {e}")
                            new_candles = 0
                    elif self.symbol_info[symbol]['was_fetched']:
                        # Symbol was fetched - all candles are new
                        new_candles = len(symbol_data)
                    else:
                        # Unknown case - assume no new candles for safety
                        new_candles = 0
                        
                    self.symbol_info[symbol]['new_candles'] = new_candles
            except Exception as e:
                logger.debug(f"Error extracting data for {symbol}: {e}")
    
    def _fill_missing_symbol_info_from_cache(self, final_data: vbt.Data):
        """
        Fill missing symbol information from cache when merge failed.
        
        This handles cases where cached symbols don't appear in final_data due to merge failures,
        but we still want to display their informational metadata (volume, inception, etc.).
        Only "New" and "Source" should be operation-specific - the rest should come from cache.
        """
        try:
            from ..cache_system import cache_manager
            from ..storage.data_storage import data_storage
            
            # Check which requested symbols are missing from final data
            final_symbols = set(final_data.symbols) if final_data and hasattr(final_data, 'symbols') else set()
            missing_symbols = set(self.requested_symbols) - final_symbols
            
            if not missing_symbols:
                return  # All symbols present in final data
            
            logger.debug(f"Filling cache info for {len(missing_symbols)} missing symbols: {missing_symbols}")
            
            # Get cached metadata
            inception_timestamps = cache_manager.get_all_timestamps(self.exchange_id)
            volume_data = cache_manager.get_all_volumes(self.exchange_id)
            
            # Try to load cached data for the missing symbols
            try:
                cached_data = data_storage.load_data(self.exchange_id, self.timeframe, [], 'spot')
                if cached_data and hasattr(cached_data, 'symbols'):
                    cached_symbols = set(cached_data.symbols)
                else:
                    cached_symbols = set()
            except Exception as e:
                logger.debug(f"Could not load cached data for missing symbols: {e}")
                cached_symbols = set()
            
            # Fill information for each missing symbol
            for symbol in missing_symbols:
                if symbol not in self.symbol_info:
                    continue  # Skip if we don't have tracking info
                
                info = self.symbol_info[symbol]
                
                # Fill volume data from cache
                if volume_data and symbol in volume_data:
                    info['volume_usd'] = volume_data[symbol]
                
                # Fill inception timestamp from cache
                if inception_timestamps and symbol in inception_timestamps:
                    info['inception_timestamp'] = inception_timestamps[symbol]
                
                # Fill actual data range from cached data
                if symbol in cached_symbols:
                    try:
                        if len(cached_data.symbols) > 1:
                            symbol_data = cached_data.close[symbol].dropna()
                        else:
                            symbol_data = cached_data.close.dropna()
                        
                        if len(symbol_data) > 0:
                            info['earliest_timestamp'] = symbol_data.index[0]
                            info['latest_timestamp'] = symbol_data.index[-1]
                            info['total_candles'] = len(symbol_data)
                            
                            # For cached symbols, new_candles should be 0 (no new data)
                            info['new_candles'] = 0
                            
                            logger.debug(f"Filled cache info for {symbol}: {len(symbol_data)} candles, {symbol_data.index[0]} to {symbol_data.index[-1]}")
                    except Exception as e:
                        logger.debug(f"Error extracting cached data for {symbol}: {e}")
                
        except Exception as e:
            logger.debug(f"Error filling missing symbol info from cache: {e}")
                
    def _display_symbol_table(self):
        """Display a detailed table of symbol information sorted by volume."""
        if not self.symbol_info:
            return
            
        # Create table
        table = Table(
            title="📊 Symbol Details (Sorted by Volume)",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan"
        )
        
        # Add columns
        table.add_column("Symbol", style="cyan", no_wrap=True)
        table.add_column("Volume (M)", justify="right", style="bright_green")
        table.add_column("Inception", style="yellow")
        table.add_column("Earliest", style="green")
        table.add_column("Latest", justify="center")  # No default style - will be colored based on freshness
        table.add_column("Candles", justify="right", style="magenta")
        table.add_column("New", justify="right", style="bright_magenta")
        table.add_column("Source", style="blue")
        
        # Format timestamp helper
        def format_timestamp(ts):
            if ts is None:
                return "-"
            if isinstance(ts, (int, float)):
                # Convert milliseconds to datetime
                from datetime import datetime
                dt = datetime.fromtimestamp(ts / 1000)
                return dt.strftime("%Y-%m-%d")
            elif hasattr(ts, 'strftime'):
                return ts.strftime("%Y-%m-%d")
            return str(ts)
        
        # Format latest timestamp with time and color coding based on freshness
        def format_latest_timestamp(ts, is_stale):
            if ts is None:
                return "[dim]-[/dim]"
            
            try:
                if isinstance(ts, (int, float)):
                    # Convert milliseconds to datetime
                    from datetime import datetime
                    dt = datetime.fromtimestamp(ts / 1000)
                    formatted = dt.strftime("%m-%d %H:%M")
                elif hasattr(ts, 'strftime'):
                    formatted = ts.strftime("%m-%d %H:%M")
                else:
                    formatted = str(ts)
                
                # Apply color coding: green for fresh, red for stale
                if is_stale:
                    return f"[red]{formatted}[/red]"
                else:
                    return f"[green]{formatted}[/green]"
                    
            except Exception:
                return "[dim]-[/dim]"
        
        # Format volume helper
        def format_volume(volume_usd):
            if volume_usd is None or volume_usd == 0:
                return "-"
            # Convert to millions and round to 1 decimal place
            volume_millions = volume_usd / 1_000_000
            if volume_millions >= 1000:
                return f"{volume_millions/1000:.1f}B"  # Show as billions for very large volumes
            elif volume_millions >= 1:
                return f"{volume_millions:.1f}M"
            else:
                return f"{volume_millions:.2f}M"  # Show more precision for smaller volumes
        
        # Sort symbols by volume (descending order, highest volume first)
        sorted_symbols = sorted(
            self.symbol_info.keys(), 
            key=lambda symbol: self.symbol_info[symbol].get('volume_usd', 0), 
            reverse=True
        )
        
        # Add rows for each symbol (sorted by volume)
        for symbol in sorted_symbols:
            info = self.symbol_info[symbol]
            
            # Get source from tracked symbol sources, fallback to old method
            source = self.symbol_sources.get(symbol, self._determine_symbol_source(symbol))
            
            # Format new candles column
            new_candles = info.get('new_candles', 0)
            new_candles_str = str(new_candles) if new_candles > 0 else "-"
            
            # Determine if symbol is stale for color coding
            is_stale = info.get('is_stale', False) or (self.cache_was_stale and source.startswith('cache'))
            
            table.add_row(
                symbol,
                format_volume(info.get('volume_usd', 0)),
                format_timestamp(info.get('inception_timestamp')),
                format_timestamp(info.get('earliest_timestamp')),
                format_latest_timestamp(info.get('latest_timestamp'), is_stale),
                str(info.get('total_candles', 0)),
                new_candles_str,
                source
            )
        
        console.print(table)
        
    def _determine_symbol_source(self, symbol: str) -> str:
        """Determine the source of data for a specific symbol."""
        # Check cache status first
        cache_status = next((status for source, status in self.operation_path if source == 'cache'), None)
        
        if cache_status == 'hit':
            return "cache"
        elif cache_status == 'stale':
            # For selective updates, some symbols may have been from cache, others from exchange
            if 'available_symbols' in self.cache_state and symbol in self.cache_state['available_symbols']:
                return "cache (fresh)"
            elif 'missing_symbols' in self.cache_state and symbol in self.cache_state['missing_symbols']:
                return "exchange"
            else:
                return "cache (updated)"
        
        # Check if symbol was in cache (for partial hits)
        if 'available_symbols' in self.cache_state:
            if symbol in self.cache_state['available_symbols']:
                # Symbol was in cache
                if cache_status == 'partial':
                    # Check if resampling was used
                    resample_status = next((status for source, status in self.operation_path if source == 'resample'), None)
                    if resample_status == 'success':
                        return "cache"
                    return "cache"
        
        # Check if symbol was fetched via resampling
        resample_status = next((status for source, status in self.operation_path if source == 'resample'), None)
        if resample_status == 'success':
            # Could be resampled or mixed
            if 'missing_symbols' in self.cache_state:
                if symbol in self.cache_state['missing_symbols']:
                    return "exchange"
            return "resampled"
        
        # Check if symbol was fetched from exchange
        exchange_status = next((status for source, status in self.operation_path if source == 'exchange'), None)
        if exchange_status == 'success':
            return "exchange"
        
        return "unknown"
        
    def _get_path_description(self) -> str:
        """Get a description of the fetch path taken."""
        if not self.operation_path:
            return "unknown path"
            
        # Check cache state
        cache_status = next((status for source, status in self.operation_path if source == 'cache'), None)
        
        if cache_status == 'hit':
            return "from cache"
        elif cache_status == 'partial':
            # Mixed fetch
            if any(source == 'resample' and status == 'success' for source, status in self.operation_path):
                return "cache + resampled"
            elif any(source == 'exchange' and status == 'success' for source, status in self.operation_path):
                return "cache + exchange"
            return "mixed sources"
        elif cache_status == 'stale':
            # Check if it was a selective update or complete update
            if 'available_symbols' in self.cache_state and 'missing_symbols' in self.cache_state:
                # Selective update happened
                fresh_count = len(self.cache_state.get('available_symbols', []))
                stale_count = len(self.cache_state.get('missing_symbols', []))
                return f"selective update ({fresh_count} cached, {stale_count} fetched)"
            else:
                return "from exchange"
        else:
            # Full fetch
            if any(source == 'resample' and status == 'success' for source, status in self.operation_path):
                return "resampled"
            elif any(source == 'exchange' and status == 'success' for source, status in self.operation_path):
                return "from exchange"
            return "fetch failed"

# Core data fetching function
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
    Simplified data fetching with clear fallback chain.
    
    Args:
        symbols: List of trading symbols to fetch
        exchange_id: Exchange identifier (e.g., 'binance')
        timeframe: Timeframe string (e.g., '1d', '1h')
        start_date: Start date (None for inception)
        end_date: End date (None for latest)
        use_cache: Whether to use cached data
        market_type: Market type ('spot' or 'swap')
        prefer_resampling: Whether to try resampling before exchange fetch
        
    Returns:
        VBT Data object or None if fetch failed
    """
    
    # Initialize structured logger
    fetch_logger = FetchLogger(symbols, exchange_id, timeframe)
    
    try:
        # Log start
        fetch_logger.log_start(start_date, end_date)
        
        # Step 1: Ensure volume data is current (for symbol selection and metadata)
        _ensure_volume_data_current(exchange_id)
        
        # Step 2: Apply blacklist filtering to input symbols
        symbols = _filter_blacklisted_symbols(symbols, exchange_id)
        fetch_logger.log_filter_result(symbols)
        
        if not symbols:
            logger.warning("No symbols remaining after blacklist filtering")
            fetch_logger.log_final_result(None)
            return None
        
        # Step 3: Determine optimal start dates using inception data
        symbol_start_dates = _get_optimal_start_dates(symbols, exchange_id, start_date)
        
        # Track original inception intent (before start_date conversion)
        is_original_inception_request = _is_inception_request(start_date)
        
        # Step 4: Try cache first
        if use_cache:
            cache_result, operation_type, cache_is_stale = _try_cache_fetch(
                symbols, exchange_id, timeframe, symbol_start_dates, 
                end_date, market_type, start_date, fetch_logger
            )
            if cache_result is not None:
                fetch_logger.log_final_result(cache_result, saved=False)
                return cache_result
            
            # CRITICAL FIX: If cache was stale, skip resampling to avoid using stale lower timeframes
            if cache_is_stale:
                logger.debug("Cache was stale, skipping resampling to ensure fresh data from exchange")
                prefer_resampling = False
        
        # Step 5: Try resampling from lower timeframes (only if cache wasn't stale)
        if prefer_resampling:
            resample_result = _try_resampling_fetch(
                symbols, exchange_id, timeframe, symbol_start_dates, 
                end_date, market_type, fetch_logger, is_original_inception_request
            )
            if resample_result is not None:
                if use_cache:
                    data_storage.save_data(resample_result, exchange_id, timeframe, market_type)
                fetch_logger.log_final_result(resample_result, saved=use_cache)
                return resample_result
        
        # Step 6: Fetch from exchange (final fallback)
        exchange_result = _fetch_from_exchange(
            symbols, exchange_id, timeframe, symbol_start_dates, 
            end_date, market_type, fetch_logger
        )
        if exchange_result is not None:
            if use_cache:
                data_storage.save_data(exchange_result, exchange_id, timeframe, market_type)
            fetch_logger.log_final_result(exchange_result, saved=use_cache)
            return exchange_result
        
        # All methods failed
        fetch_logger.log_final_result(None)
        return None
        
    except Exception as e:
        logger.error(f"Error in fetch_data: {e}")
        fetch_logger.log_final_result(None)
        return None

# Cache operations
def _calculate_latest_complete_candle_time(current_time: vbt.timestamp, timeframe: str) -> vbt.timestamp:
    """
    Calculate the exact START time of the latest complete candle for a given timeframe.
    
    This provides precise awareness of when candles should be available,
    eliminating the need for imprecise buffer-based approaches.
    
    Args:
        current_time: Current UTC timestamp
        timeframe: Timeframe string (e.g., '5m', '1h', '1d')
        
    Returns:
        UTC timestamp of when the latest complete candle starts (its timestamp)
    """
    import pandas as pd
    
    # Convert to pandas timestamp for easier manipulation
    # Handle timezone-aware timestamps properly
    if hasattr(current_time, 'tz') and current_time.tz is not None:
        if str(current_time.tz) != 'UTC':
            current_pd = pd.Timestamp(current_time).tz_convert('UTC')
        else:
            current_pd = pd.Timestamp(current_time)
    else:
        current_pd = pd.Timestamp(current_time, tz='UTC')
    
    tf = timeframe.lower().strip()
    
    try:
        if tf.endswith('s'):
            # Seconds
            seconds = int(tf[:-1])
            # Round down to nearest interval, then subtract one interval for latest complete
            total_seconds = current_pd.hour * 3600 + current_pd.minute * 60 + current_pd.second
            current_candle_start = (total_seconds // seconds) * seconds
            latest_complete_start = current_candle_start - seconds
            
            if latest_complete_start < 0:
                # Handle day boundary
                latest_complete_start += 24 * 3600
                latest_complete = (current_pd - pd.Timedelta(days=1)).replace(
                    hour=latest_complete_start // 3600,
                    minute=(latest_complete_start % 3600) // 60,
                    second=latest_complete_start % 60,
                    microsecond=0
                )
            else:
                latest_complete = current_pd.replace(
                    hour=latest_complete_start // 3600,
                    minute=(latest_complete_start % 3600) // 60,
                    second=latest_complete_start % 60,
                    microsecond=0
                )
            
        elif tf.endswith('m') or tf.endswith('min'):
            # Minutes
            minutes = int(tf[:-3] if tf.endswith('min') else tf[:-1])
            # Round down to nearest interval, then subtract one interval for latest complete
            total_minutes = current_pd.hour * 60 + current_pd.minute
            current_candle_start = (total_minutes // minutes) * minutes
            latest_complete_start = current_candle_start - minutes
            
            if latest_complete_start < 0:
                # Handle day boundary
                latest_complete_start += 24 * 60
                latest_complete = (current_pd - pd.Timedelta(days=1)).replace(
                    hour=latest_complete_start // 60,
                    minute=latest_complete_start % 60,
                    second=0,
                    microsecond=0
                )
            else:
                latest_complete = current_pd.replace(
                    hour=latest_complete_start // 60,
                    minute=latest_complete_start % 60,
                    second=0,
                    microsecond=0
                )
            
        elif tf.endswith('h') or tf.endswith('hour'):
            # Hours
            hours = int(tf[:-4] if tf.endswith('hour') else tf[:-1])
            # Round down to nearest interval, then subtract one interval for latest complete
            current_candle_start = (current_pd.hour // hours) * hours
            latest_complete_start = current_candle_start - hours
            
            if latest_complete_start < 0:
                # Handle day boundary
                latest_complete_start += 24
                latest_complete = (current_pd - pd.Timedelta(days=1)).replace(
                    hour=latest_complete_start,
                    minute=0,
                    second=0,
                    microsecond=0
                )
            else:
                latest_complete = current_pd.replace(
                    hour=latest_complete_start,
                    minute=0,
                    second=0,
                    microsecond=0
                )
            
        elif tf.endswith('d') or tf.endswith('day'):
            # Daily - latest complete is previous day
            latest_complete = current_pd.replace(
                hour=0,
                minute=0, 
                second=0,
                microsecond=0
            ) - pd.Timedelta(days=1)
                
        elif tf.endswith('w') or tf.endswith('week'):
            # Weekly - latest complete is previous week start (Monday)
            days_since_monday = current_pd.weekday()
            # Go to current week start, then subtract one week
            current_week_start = current_pd.replace(
                hour=0,
                minute=0,
                second=0,
                microsecond=0
            ) - pd.Timedelta(days=days_since_monday)
            latest_complete = current_week_start - pd.Timedelta(weeks=1)
            
        else:
            # Unknown format - default to previous hour
            latest_complete = current_pd.replace(
                minute=0,
                second=0,
                microsecond=0
            ) - pd.Timedelta(hours=1)
        
        # Convert back to vbt timestamp
        # Ensure it's in UTC - if it already has timezone info, convert it
        if latest_complete.tz is not None:
            if str(latest_complete.tz) != 'UTC':
                latest_complete = latest_complete.tz_convert('UTC')
            return vbt.timestamp(latest_complete)
        else:
            return vbt.timestamp(latest_complete, tz='UTC')
        
    except (ValueError, TypeError) as e:
        logger.debug(f"Error calculating latest complete candle time: {e}")
        # Fallback: assume 1 hour ago
        return current_time - pd.Timedelta(hours=1)

def _identify_stale_symbols(
    data: vbt.Data,
    symbols: List[str],
    target_end: Union[str, vbt.timestamp],
    timeframe: str
) -> Tuple[List[str], List[str]]:
    """
    Identify which specific symbols are stale vs fresh using exact candle boundary awareness.
    
    Returns:
        Tuple of (stale_symbols, fresh_symbols)
    """
    try:
        # Convert target_end to UTC timestamp
        if isinstance(target_end, str):
            if target_end.lower() == "now":
                current_time = vbt.utc_timestamp()
            else:
                current_time = vbt.timestamp(target_end, tz='UTC')
        else:
            current_time = vbt.timestamp(target_end, tz='UTC')
        
        # Calculate exactly when the latest complete candle should end
        expected_latest_candle = _calculate_latest_complete_candle_time(current_time, timeframe)
        
        logger.debug(f"Exact candle boundary analysis for {timeframe}:")
        logger.debug(f"   Current time: {current_time}")
        logger.debug(f"   Expected latest complete candle: {expected_latest_candle}")
        
        stale_symbols = []
        fresh_symbols = []
        
        for symbol in symbols:
            try:
                # Get symbol-specific data
                if len(data.symbols) > 1:
                    symbol_data = data.close[symbol].dropna()
                else:
                    symbol_data = data.close.dropna()
                
                if len(symbol_data) == 0:
                    stale_symbols.append(symbol)
                    continue
                
                # Get latest timestamp for this symbol
                actual_latest = symbol_data.index[-1]
                
                # Ensure timezone-aware comparison
                if actual_latest.tz is None:
                    actual_latest = actual_latest.tz_localize('UTC')
                elif str(actual_latest.tz) != 'UTC':
                    actual_latest = actual_latest.tz_convert('UTC')
                
                # Exact comparison: is our data missing the expected latest candle?
                is_stale = actual_latest < expected_latest_candle
                
                logger.debug(f"   {symbol}: actual={actual_latest}, expected={expected_latest_candle}, stale={is_stale}")
                
                if is_stale:
                    stale_symbols.append(symbol)
                else:
                    fresh_symbols.append(symbol)
                    
            except Exception as e:
                logger.debug(f"Error checking staleness for {symbol}: {e}")
                stale_symbols.append(symbol)  # Assume stale if can't determine
        
        return stale_symbols, fresh_symbols
        
    except Exception as e:
        logger.debug(f"Error in per-symbol staleness detection: {e}")
        return symbols, []  # Assume all stale if error

def _get_volume_content_timestamp(volume_file_path: str) -> Optional[float]:
    """
    Extract the actual content timestamp from volume.json file.
    
    Args:
        volume_file_path: Path to the volume.json file
        
    Returns:
        Latest timestamp from file content as Unix timestamp, or None if parsing fails
    """
    try:
        import json
        with open(volume_file_path, 'r') as f:
            volume_data = json.load(f)
        
        if not volume_data:
            return None
        
        # Find the latest 'updated' timestamp in the content
        latest_timestamp = None
        for symbol_data in volume_data.values():
            if isinstance(symbol_data, dict) and 'updated' in symbol_data:
                try:
                    # Parse ISO format timestamp to Unix timestamp
                    timestamp_str = symbol_data['updated']
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    unix_timestamp = dt.timestamp()
                    
                    if latest_timestamp is None or unix_timestamp > latest_timestamp:
                        latest_timestamp = unix_timestamp
                except Exception:
                    continue
        
        return latest_timestamp
        
    except Exception as e:
        logger.debug(f"Error parsing volume content timestamp: {e}")
        return None

def _ensure_volume_data_current(exchange_id: str) -> None:
    """
    Ensure volume data and symbol metadata are current and up-to-date.
    
    This function:
    1. Checks if volume.json is updated to the latest midnight UTC time
    2. If not, automatically triggers a volume update
    3. Re-checks for new symbols that might have been added to the exchange
    4. Updates timestamps for any new symbols discovered (with safety limits)
    5. Applies blacklist filtering to exclude manually blacklisted symbols
    
    CRITICAL: This is called at the start of every fetch operation to ensure
    we have the most current exchange metadata before attempting data fetches.
    """
    try:
        from datetime import datetime, timedelta
        from ..cache_system.metadata_fetcher import data_fetcher
        
        logger.debug(f"Ensuring volume data is current for {exchange_id}")
        
        # Calculate the latest midnight UTC time
        now_utc = datetime.utcnow()
        latest_midnight_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        
        # If it's still early in the day (before 1 AM UTC), use previous day's midnight
        # This accounts for exchanges that may take time to update their data
        if now_utc.hour < 1:
            latest_midnight_utc = latest_midnight_utc - timedelta(days=1)
        
        latest_midnight_timestamp = latest_midnight_utc.timestamp()
        
        # Check if volume cache exists and is from the latest midnight or later
        volume_cache_fresh = cache_manager.is_volume_cache_fresh(exchange_id)
        volume_file_path = os.path.join(
            cache_manager._get_exchange_dir(exchange_id), 
            'volume.json'
        )
        
        needs_volume_update = True
        
        if volume_cache_fresh and os.path.exists(volume_file_path):
            # Check if the volume cache content is from the latest midnight or later
            # Use content timestamps instead of file modification time for accuracy
            try:
                content_timestamp = _get_volume_content_timestamp(volume_file_path)
                
                if content_timestamp and content_timestamp >= latest_midnight_timestamp:
                    logger.debug(f"Volume cache for {exchange_id} is current (content timestamp check)")
                    needs_volume_update = False
                elif content_timestamp:
                    content_dt = datetime.fromtimestamp(content_timestamp)
                    logger.debug(f"Volume cache for {exchange_id} is stale (content from {content_dt})")
                else:
                    # Fall back to file modification time if content parsing fails
                    logger.debug(f"Content timestamp unavailable, falling back to file modification time")
                    volume_file_mtime = os.path.getmtime(volume_file_path)
                    if volume_file_mtime >= latest_midnight_timestamp:
                        logger.debug(f"Volume cache for {exchange_id} is current (file timestamp check)")
                        needs_volume_update = False
                    else:
                        logger.debug(f"Volume cache for {exchange_id} is stale (file timestamp check)")
                        
            except OSError as e:
                logger.debug(f"Could not check volume file timestamp: {e}")
        else:
            logger.debug(f"No fresh volume cache found for {exchange_id}")
        
        # Update volume data if needed
        if needs_volume_update:
            logger.debug(f"Updating volume data for {exchange_id}...")
            
            try:
                # Fetch fresh market data with volume information
                market_data = data_fetcher.get_market_data(
                    exchange_id=exchange_id,
                    use_cache=False,
                    force_refresh=True
                )
                
                if market_data:
                    logger.debug(f"Updated volume data for {exchange_id}: {len(market_data)} symbols")
                    
                    # Check for new symbols that weren't in our timestamp cache
                    existing_timestamps = cache_manager.get_all_timestamps(exchange_id)
                    
                    # Only consider symbols with actual volume > 0 as "new"
                    active_new_symbols = []
                    for symbol, data in market_data.items():
                        if symbol not in existing_timestamps:
                            # Only include if it has meaningful volume (> 0)
                            volume = data.get('volume', 0) if isinstance(data, dict) else 0
                            if volume > 0:
                                active_new_symbols.append(symbol)
                    
                    if active_new_symbols:
                        # SAFETY LIMIT: Cap the number of new timestamps to fetch at once
                        MAX_NEW_TIMESTAMPS = 100  # Reasonable limit for incremental updates
                        
                        if len(active_new_symbols) > MAX_NEW_TIMESTAMPS:
                            logger.debug(f"Found {len(active_new_symbols)} new symbols, limiting to {MAX_NEW_TIMESTAMPS}")
                            # Sort by volume and take the top ones
                            active_new_symbols_with_volume = [
                                (symbol, market_data[symbol].get('volume', 0) if isinstance(market_data[symbol], dict) else 0)
                                for symbol in active_new_symbols
                            ]
                            active_new_symbols_with_volume.sort(key=lambda x: x[1], reverse=True)
                            active_new_symbols = [symbol for symbol, _ in active_new_symbols_with_volume[:MAX_NEW_TIMESTAMPS]]
                        
                        logger.debug(f"Fetching timestamps for {len(active_new_symbols)} new active symbols")
                        
                        try:
                            new_timestamps = data_fetcher.get_inception_timestamps(
                                exchange_id=exchange_id,
                                symbols=active_new_symbols,
                                use_cache=True
                            )
                            
                            if new_timestamps:
                                logger.debug(f"Updated timestamps for {len(new_timestamps)} new symbols")
                                
                        except Exception as timestamp_error:
                            logger.debug(f"Error fetching timestamps for new symbols: {timestamp_error}")
                    else:
                        logger.debug("No new active symbols found")
                else:
                    logger.debug(f"Failed to fetch market data for {exchange_id}")
                    
            except Exception as e:
                logger.debug(f"Error updating volume data for {exchange_id}: {e}")
        
        # Apply blacklist filtering
        _apply_blacklist_filtering(exchange_id)
        
    except Exception as e:
        logger.debug(f"Could not ensure volume data current for {exchange_id}: {e}")

def _apply_blacklist_filtering(exchange_id: str) -> None:
    """
    Apply blacklist filtering to remove symbols from volume and timestamp caches.
    
    Uses the cache_manager's built-in blacklist functionality which automatically
    ensures USDC/USDT and other stable-stable pairs are blacklisted by default.
    """
    try:
        # Get blacklist data from cache manager (automatically initialized)
        blacklist_data = cache_manager.get_blacklist()
        
        # Combine global and exchange-specific blacklists
        blacklisted_symbols = set(blacklist_data.get('global', []))
        if exchange_id:
            exchange_blacklist = blacklist_data.get(exchange_id.lower(), [])
            blacklisted_symbols.update(exchange_blacklist)
        
        if not blacklisted_symbols:
            return  # No blacklist to apply
        
        logger.debug(f"Applying blacklist filter: {len(blacklisted_symbols)} symbols")
        
        # Remove blacklisted symbols from volume cache
        exchange_volumes = cache_manager.get_all_volumes(exchange_id)
        if exchange_volumes:
            blacklisted_found = set(exchange_volumes.keys()) & blacklisted_symbols
            if blacklisted_found:
                logger.debug(f"Removing {len(blacklisted_found)} blacklisted symbols from volume cache")
                
                # Remove from in-memory cache
                if exchange_id in cache_manager._volume_cache:
                    for symbol in blacklisted_found:
                        cache_manager._volume_cache[exchange_id].pop(symbol, None)
                    
                    # Save updated cache to disk
                    cache_manager._save_to_disk(exchange_id, 'volume', cache_manager._volume_cache[exchange_id])
        
        # Remove blacklisted symbols from timestamp cache
        exchange_timestamps = cache_manager.get_all_timestamps(exchange_id)
        if exchange_timestamps:
            blacklisted_found = set(exchange_timestamps.keys()) & blacklisted_symbols
            if blacklisted_found:
                logger.debug(f"Removing {len(blacklisted_found)} blacklisted symbols from timestamp cache")
                
                # Remove from in-memory cache
                if exchange_id in cache_manager._timestamp_cache:
                    for symbol in blacklisted_found:
                        cache_manager._timestamp_cache[exchange_id].pop(symbol, None)
                    
                    # Save updated cache to disk
                    cache_manager._save_to_disk(exchange_id, 'timestamps', cache_manager._timestamp_cache[exchange_id])
        
        # Add blacklisted symbols to failed symbols to prevent future fetching
        for symbol in blacklisted_symbols:
            cache_manager.add_failed_symbol(exchange_id, symbol)
            
    except Exception as e:
        logger.debug(f"Error applying blacklist filtering for {exchange_id}: {e}")

def _filter_blacklisted_symbols(symbols: List[str], exchange_id: str = None) -> List[str]:
    """
    Filter out blacklisted symbols from a list using cache_manager's built-in functionality.
    
    Args:
        symbols: List of symbols to filter
        exchange_id: Optional exchange ID for exchange-specific filtering
        
    Returns:
        Filtered list of symbols
    """
    try:
        # Use cache_manager's built-in filtering (which includes automatic default blacklist)
        return cache_manager.filter_blacklisted_symbols(symbols, exchange_id)
        
    except Exception as e:
        logger.debug(f"Error filtering blacklisted symbols: {e}")
        return symbols

def _get_optimal_start_dates(
    symbols: List[str], 
    exchange_id: str, 
    start_date: Optional[Union[str, vbt.timestamp]]
) -> Dict[str, str]:
    """Get optimal start dates for symbols using cached inception data."""
    
    symbol_start_dates = {}
    
    # If start_date is None or indicates inception fetching, use cached inception dates
    if _is_inception_request(start_date):
        logger.debug("Using inception dates from cache")
        cached_timestamps = cache_manager.get_all_timestamps(exchange_id)
        
        if cached_timestamps:
            for symbol in symbols:
                if symbol in cached_timestamps:
                    try:
                        timestamp_ms = cached_timestamps[symbol]
                        dt = datetime.fromtimestamp(timestamp_ms / 1000)
                        symbol_start_dates[symbol] = dt.strftime('%Y-%m-%d')
                    except Exception:
                        symbol_start_dates[symbol] = start_date
                else:
                    symbol_start_dates[symbol] = start_date
        else:
            # No cached inception dates
            for symbol in symbols:
                symbol_start_dates[symbol] = start_date
    else:
        # Use provided start_date for all symbols
        for symbol in symbols:
            symbol_start_dates[symbol] = start_date
    
    return symbol_start_dates

def _is_inception_request(start_date: Optional[Union[str, vbt.timestamp]]) -> bool:
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

def _check_inception_completeness(
    cached_data: vbt.Data,
    symbols: List[str], 
    exchange_id: str,
    symbol_start_dates: Dict[str, str]
) -> bool:
    """Check if cached data contains complete inception data for all symbols."""
    
    try:
        for symbol in symbols:
            if symbol not in cached_data.symbols:
                return False
            
            # Check if symbol data starts from its true inception date
            expected_start = symbol_start_dates.get(symbol)
            if expected_start:
                # Get first date in cached data for this symbol
                if len(cached_data.symbols) > 1:
                    symbol_data = cached_data.close[symbol].dropna()
                else:
                    symbol_data = cached_data.close.dropna()
                
                if len(symbol_data) == 0:
                    return False
                
                cached_start = symbol_data.index[0].date()
                expected_start_dt = datetime.strptime(expected_start, '%Y-%m-%d').date()
                
                # Allow 1 day tolerance
                days_diff = abs((cached_start - expected_start_dt).days)
                if days_diff > 1:
                    logger.debug(f"Inception completeness failed for {symbol}: cached starts {cached_start}, expected {expected_start_dt}")
                    return False
        
        return True
        
    except Exception as e:
        logger.debug(f"Error checking inception completeness: {e}")
        return False

def _is_data_fresh(
    data: vbt.Data, 
    target_end: Union[str, vbt.timestamp], 
    timeframe: str
) -> bool:
    """
    Check if cached data is fresh enough for the target end date.
    
    CRITICAL FIX: For multi-symbol data, checks the EARLIEST latest timestamp
    across all symbols to ensure ALL symbols are fresh, not just the latest one.
    """
    
    try:
        # Convert target_end to UTC timestamp
        if isinstance(target_end, str):
            if target_end.lower() == "now":
                target_dt = vbt.utc_timestamp()
            else:
                target_dt = vbt.timestamp(target_end, tz='UTC')
        else:
            target_dt = vbt.timestamp(target_end, tz='UTC')
        
        # CRITICAL FIX: For multi-symbol data, find the earliest "latest timestamp"
        # This ensures ALL symbols are fresh, not just the overall latest
        if hasattr(data, 'symbols') and len(data.symbols) > 1:
            symbol_latest_times = []
            for symbol in data.symbols:
                try:
                    symbol_data = data.close[symbol].dropna()
                    if len(symbol_data) > 0:
                        symbol_latest_times.append(symbol_data.index[-1])
                except Exception:
                    continue
            
            if symbol_latest_times:
                # Use the EARLIEST latest timestamp - ensures ALL symbols are fresh
                latest_dt = min(symbol_latest_times)
                logger.debug(f"Multi-symbol freshness check: earliest latest timestamp = {latest_dt}")
            else:
                # Fallback to overall latest
                latest_dt = data.wrapper.index[-1]
        else:
            # Single symbol or fallback
            latest_dt = data.wrapper.index[-1]
        
        # Ensure both timestamps are timezone-aware in UTC
        if latest_dt.tz is None:
            latest_dt = latest_dt.tz_localize('UTC')
        elif str(latest_dt.tz) != 'UTC':
            latest_dt = latest_dt.tz_convert('UTC')
        
        if target_dt.tz is None:
            target_dt = target_dt.tz_localize('UTC')
        elif str(target_dt.tz) != 'UTC':
            target_dt = target_dt.tz_convert('UTC')
        
        # Use exact candle boundary awareness instead of buffer-based approach
        expected_latest_candle = _calculate_latest_complete_candle_time(target_dt, timeframe)
        
        # Special handling for daily+ timeframes
        if timeframe.endswith('d') or 'day' in timeframe.lower():
            # For daily: fresh if same day or target is in past
            is_fresh = target_dt.date() <= latest_dt.date()
        else:
            # For intraday: fresh if we have the expected latest complete candle
            is_fresh = latest_dt >= expected_latest_candle
        
        logger.debug(f"Exact freshness check: latest={latest_dt}, expected_candle={expected_latest_candle}, fresh={is_fresh}")
        return is_fresh
            
    except Exception as e:
        logger.debug(f"Error checking data freshness: {e}")
        return True  # Assume fresh if can't determine

def _try_cache_fetch(
    symbols: List[str], 
    exchange_id: str, 
    timeframe: str, 
    symbol_start_dates: Dict[str, str],
    end_date: Optional[Union[str, vbt.timestamp]],
    market_type: str,
    original_start_date: Optional[Union[str, vbt.timestamp]],
    fetch_logger: FetchLogger
) -> Tuple[Optional[vbt.Data], str, bool]:
    """
    Try to fetch data from cache with exact candle boundary awareness.
    
    Returns:
        Tuple of (data, operation_type, cache_is_stale)
        cache_is_stale is True if the cache was detected as stale
    """
    
    cached_data = data_storage.load_data(exchange_id, timeframe, market_type=market_type)
    
    if cached_data is None:
        fetch_logger.log_cache_result("miss", {})
        return None, "Cache Operation", False
    
    # Check symbol availability
    cached_symbols = set(cached_data.symbols)
    requested_symbols = set(symbols)
    missing_symbols = requested_symbols - cached_symbols
    available_symbols = requested_symbols & cached_symbols
    
    if missing_symbols:
        # Partial cache hit
        fetch_logger.log_cache_result("partial", {
            "available": len(available_symbols),
            "requested": len(requested_symbols),
            "missing": len(missing_symbols),
            "available_symbols": available_symbols,
            "missing_symbols": missing_symbols
        })
        
        # Check freshness of available symbols before mixed fetch
        updated_cached_data = cached_data
        cache_was_stale = False
        
        if end_date and available_symbols:
            # Check freshness only for the symbols we're using, not the entire cache
            available_cached_data = cached_data.select(list(available_symbols))
            
            if not _is_data_fresh(available_cached_data, end_date, timeframe):
                logger.debug("Available cached symbols are outdated, updating before mixed fetch...")
                cache_was_stale = True
                
                try:
                    updated_cached_data = cached_data.update(
                        end=vbt.timestamp(end_date, tz='UTC') if isinstance(end_date, str) and end_date.lower() != "now" else vbt.utc_timestamp(),
                        show_progress=False
                    )
                    
                    if updated_cached_data is not None:
                        data_storage.save_data(updated_cached_data, exchange_id, timeframe, market_type)
                        logger.debug("Successfully updated cached data before mixed fetch")
                    else:
                        updated_cached_data = cached_data
                        logger.debug("VBT update returned None, using original cached data")
                        
                except Exception as update_error:
                    logger.debug(f"Failed to update cached data: {update_error}")
                    updated_cached_data = cached_data
            else:
                logger.debug("Available cached symbols are fresh, no update needed")
        
        # Fetch missing symbols and merge with cached data
        return (_fetch_missing_symbols_and_merge(
            all_symbols=symbols,
            missing_symbols=list(missing_symbols),
            available_symbols=list(available_symbols),
            cached_data=updated_cached_data,
            exchange_id=exchange_id,
            timeframe=timeframe,
            symbol_start_dates=symbol_start_dates,
            end_date=end_date,
            market_type=market_type,
            original_start_date=original_start_date,
            fetch_logger=fetch_logger
        ), "Mixed Fetch", cache_was_stale)
    
    # All symbols are cached - check completeness and freshness
    
    # Check inception completeness (only for explicit inception requests)
    is_inception_request = _is_inception_request(original_start_date)
    logger.debug(f"Is inception request: {is_inception_request}")
    
    # IMPORTANT: Don't fail cache hit just because cached data doesn't go back to true inception
    # This allows users to re-run the same command and get cache hits with partial data
    # Only check inception if user explicitly provided a very early start date
    if is_inception_request and original_start_date is not None:
        # Only check inception completeness for explicit early start dates
        inception_complete = _check_inception_completeness(
            cached_data, symbols, exchange_id, symbol_start_dates
        )
        logger.debug(f"Inception complete: {inception_complete}")
        if not inception_complete:
            fetch_logger.log_cache_result("incomplete", {"reason": "missing inception data"})
            return None, "Cache Operation", False
    else:
        logger.debug("Skipping inception completeness check for implicit inception request")
    
    # Check data freshness using exact candle boundary awareness
    if end_date:
        logger.debug(f"Checking freshness with end_date: {end_date}")
        requested_cached_data = cached_data.select(symbols)
        
        # Identify which specific symbols are stale vs fresh
        stale_symbols, fresh_symbols = _identify_stale_symbols(
            requested_cached_data, symbols, end_date, timeframe
        )
        
        logger.debug(f"Per-symbol staleness analysis:")
        logger.debug(f"   Stale symbols ({len(stale_symbols)}): {stale_symbols}")
        logger.debug(f"   Fresh symbols ({len(fresh_symbols)}): {fresh_symbols}")
    else:
        logger.debug("No end_date specified, skipping freshness check")
        # Initialize variables for case when no freshness check is needed
        stale_symbols = []
        fresh_symbols = symbols
        
        if stale_symbols and fresh_symbols:
            # MIXED STALENESS: Some symbols are stale, some are fresh
            # Use selective update instead of complete cache update
            logger.debug(f"Mixed staleness detected: performing selective update for {len(stale_symbols)} stale symbols")
            
            fetch_logger.log_cache_result("stale", {
                "selective_update": True,
                "stale_count": len(stale_symbols),
                "fresh_count": len(fresh_symbols)
            })
            
            # Update cache state for logging
            fetch_logger.cache_state.update({
                "available_symbols": fresh_symbols,
                "missing_symbols": stale_symbols,
                "selective_update": True
            })
            
            # Fetch only stale symbols and merge with fresh cached symbols
            return (_fetch_missing_symbols_and_merge(
                all_symbols=symbols,
                missing_symbols=stale_symbols,
                available_symbols=fresh_symbols,
                cached_data=cached_data,
                exchange_id=exchange_id,
                timeframe=timeframe,
                symbol_start_dates=symbol_start_dates,
                end_date=end_date,
                market_type=market_type,
                original_start_date=original_start_date,
                fetch_logger=fetch_logger
            ), "Selective Update", True)
            
        elif stale_symbols and not fresh_symbols:
            # ALL SYMBOLS ARE STALE: Use VBT's complete update
            logger.debug(f"All symbols are stale: attempting complete cache update")
            fetch_logger.log_cache_result("stale", {"reason": "all symbols stale"})
            
            # Use VBT's native update() method for complete update
            try:
                if isinstance(end_date, str):
                    if end_date.lower() == "now":
                        update_end = vbt.utc_timestamp()
                    else:
                        update_end = vbt.timestamp(end_date, tz='UTC')
                else:
                    update_end = vbt.timestamp(end_date, tz='UTC')
                
                updated_data = cached_data.update(
                    end=update_end,
                    show_progress=False
                )
                
                if updated_data is not None:
                    data_storage.save_data(updated_data, exchange_id, timeframe, market_type)
                    logger.debug("Successfully updated complete cached data (all symbols were stale)")
                    return updated_data.select(symbols), "Cache Operation", True
                else:
                    logger.debug("VBT update returned None for complete update, falling back to exchange fetch")
                    
            except Exception as e:
                logger.debug(f"VBT complete update failed: {e}, falling back to exchange fetch")
            
            # If VBT update fails, fall back to complete exchange fetch
            logger.debug("Complete cache update failed, falling back to complete exchange fetch")
            return None, "Cache Operation", True
            
        else:
            # ALL SYMBOLS ARE FRESH: No update needed
            logger.debug("All requested symbols are fresh, no update needed")
    
    # Cache hit - return requested symbols
    logger.debug("Cache validation passed, returning cache hit")
    fetch_logger.log_cache_result("hit", {"symbols": len(symbols)})
    
    # Track that all symbols came from cache
    for symbol in symbols:
        if symbol in cached_data.symbols:
            # Get the last timestamp from cached data for new candles calculation
            try:
                if len(cached_data.symbols) > 1:
                    symbol_data = cached_data.close[symbol].dropna()
                else:
                    symbol_data = cached_data.close.dropna()
                
                cache_end_time = symbol_data.index[-1] if len(symbol_data) > 0 else None
                fetch_logger.set_symbol_source(symbol, "cache", was_cached=True, cache_end_time=cache_end_time)
            except Exception as e:
                logger.debug(f"Error tracking cache info for {symbol}: {e}")
                fetch_logger.set_symbol_source(symbol, "cache", was_cached=True)
    
    return cached_data.select(symbols), "Cache Hit", False

def _fetch_missing_symbols_and_merge(
    all_symbols: List[str],
    missing_symbols: List[str],
    available_symbols: List[str],
    cached_data: vbt.Data,
    exchange_id: str,
    timeframe: str,
    symbol_start_dates: Dict[str, str],
    end_date: Optional[Union[str, vbt.timestamp]],
    market_type: str,
    original_start_date: Optional[Union[str, vbt.timestamp]],
    fetch_logger: FetchLogger
) -> Optional[vbt.Data]:
    """Fetch missing symbols and merge with cached data."""
    
    logger.debug(f"Fetching {len(missing_symbols)} missing symbols, merging with {len(available_symbols)} cached")
    
    # Track cached symbols first
    for symbol in available_symbols:
        if symbol in cached_data.symbols:
            try:
                if len(cached_data.symbols) > 1:
                    symbol_data = cached_data.close[symbol].dropna()
                else:
                    symbol_data = cached_data.close.dropna()
                
                cache_end_time = symbol_data.index[-1] if len(symbol_data) > 0 else None
                fetch_logger.set_symbol_source(symbol, "cache", was_cached=True, cache_end_time=cache_end_time)
            except Exception as e:
                logger.debug(f"Error tracking cache info for {symbol}: {e}")
                fetch_logger.set_symbol_source(symbol, "cache", was_cached=True)
    
    # Get start dates only for missing symbols
    missing_symbol_start_dates = {
        symbol: symbol_start_dates.get(symbol, original_start_date)
        for symbol in missing_symbols
    }
    
    # Try resampling first for missing symbols
    missing_data = _try_resampling_fetch(
        symbols=missing_symbols,
        exchange_id=exchange_id,
        timeframe=timeframe,
        symbol_start_dates=missing_symbol_start_dates,
        end_date=end_date,
        market_type=market_type,
        fetch_logger=fetch_logger
    )
    
    # Track which symbols came from resampling
    if missing_data is not None:
        for symbol in missing_symbols:
            if symbol in missing_data.symbols:
                fetch_logger.set_symbol_source(symbol, "resampled", was_fetched=True)
    
    # If resampling didn't work, fetch from exchange
    if missing_data is None:
        missing_data = _fetch_from_exchange(
            symbols=missing_symbols,
            exchange_id=exchange_id,
            timeframe=timeframe,
            symbol_start_dates=missing_symbol_start_dates,
            end_date=end_date,
            market_type=market_type,
            fetch_logger=fetch_logger
        )
        
        # Track which symbols came from exchange
        if missing_data is not None:
            for symbol in missing_symbols:
                if symbol in missing_data.symbols:
                    fetch_logger.set_symbol_source(symbol, "exchange", was_fetched=True)
    
    if missing_data is None:
        logger.error("Failed to fetch missing symbols")
        return None
    
    # Multi-strategy merge approach for maximum reliability
    try:
        logger.debug("Merging cached and newly fetched data")
        
        # Strategy 1: Try VBT's native concat with selected cached data
        try:
            available_cached_data = cached_data.select_symbols(available_symbols)
            combined_data = available_cached_data.concat(missing_data)
            data_storage.save_data(combined_data, exchange_id, timeframe, market_type)
            logger.debug("Strategy 1 (VBT concat) successful")
            return combined_data
        except Exception as concat_error:
            logger.debug(f"Strategy 1 (VBT concat) failed: {concat_error}")
        
        # Strategy 2: Try VBT's merge method
        try:
            available_cached_data = cached_data.select_symbols(available_symbols)
            combined_data = vbt.Data.concat([available_cached_data, missing_data])
            data_storage.save_data(combined_data, exchange_id, timeframe, market_type)
            logger.debug("Strategy 2 (VBT Data.concat) successful")
            return combined_data
        except Exception as merge_error:
            logger.debug(f"Strategy 2 (VBT Data.concat) failed: {merge_error}")
        
        # Strategy 3: Manual symbol-by-symbol reconstruction (most reliable)
        try:
            logger.debug("Using Strategy 3 (manual reconstruction)")
            
            # Get all symbols in the order we want them
            all_symbols = list(available_symbols) + list(missing_data.symbols)
            
            # Create symbol dictionary approach
            symbol_dict = {}
            
            # Add cached symbols
            available_cached_data = cached_data.select_symbols(available_symbols)
            for symbol in available_symbols:
                try:
                    if len(available_cached_data.symbols) > 1:
                        symbol_data = {
                            'Open': available_cached_data.open[symbol],
                            'High': available_cached_data.high[symbol],
                            'Low': available_cached_data.low[symbol], 
                            'Close': available_cached_data.close[symbol],
                            'Volume': available_cached_data.volume[symbol] if hasattr(available_cached_data, 'volume') else None
                        }
                    else:
                        symbol_data = {
                            'Open': available_cached_data.open,
                            'High': available_cached_data.high,
                            'Low': available_cached_data.low,
                            'Close': available_cached_data.close,
                            'Volume': available_cached_data.volume if hasattr(available_cached_data, 'volume') else None
                        }
                    
                    # Remove None values and create DataFrame
                    symbol_data = {k: v for k, v in symbol_data.items() if v is not None}
                    if symbol_data:
                        symbol_dict[symbol] = pd.DataFrame(symbol_data)
                        
                except Exception as symbol_error:
                    logger.debug(f"Error adding cached symbol {symbol}: {symbol_error}")
            
            # Add newly fetched symbols
            for symbol in missing_data.symbols:
                try:
                    if len(missing_data.symbols) > 1:
                        symbol_data = {
                            'Open': missing_data.open[symbol],
                            'High': missing_data.high[symbol],
                            'Low': missing_data.low[symbol],
                            'Close': missing_data.close[symbol], 
                            'Volume': missing_data.volume[symbol] if hasattr(missing_data, 'volume') else None
                        }
                    else:
                        symbol_data = {
                            'Open': missing_data.open,
                            'High': missing_data.high,
                            'Low': missing_data.low,
                            'Close': missing_data.close,
                            'Volume': missing_data.volume if hasattr(missing_data, 'volume') else None
                        }
                    
                    # Remove None values and create DataFrame
                    symbol_data = {k: v for k, v in symbol_data.items() if v is not None}
                    if symbol_data:
                        symbol_dict[symbol] = pd.DataFrame(symbol_data)
                        
                except Exception as symbol_error:
                    logger.debug(f"Error adding new symbol {symbol}: {symbol_error}")
            
            if symbol_dict:
                combined_data = vbt.Data.from_data(symbol_dict)
                if combined_data is not None:
                    data_storage.save_data(combined_data, exchange_id, timeframe, market_type)
                    logger.debug(f"Strategy 3 successful: reconstructed {len(symbol_dict)} symbols")
                    return combined_data
                else:
                    logger.debug("Strategy 3 failed: VBT Data.from_data returned None")
            else:
                logger.debug("Strategy 3 failed: no valid symbol data")
                
        except Exception as manual_error:
            logger.debug(f"Strategy 3 (manual reconstruction) failed: {manual_error}")
        
        # All strategies failed
        logger.error("All merge strategies failed, returning newly fetched data only")
        return missing_data
        
    except Exception as e:
        logger.error(f"Unexpected error in merge logic: {e}")
        return missing_data

def _try_resampling_fetch(
    symbols: List[str],
    exchange_id: str, 
    timeframe: str,
    symbol_start_dates: Dict[str, str],
    end_date: Optional[Union[str, vbt.timestamp]],
    market_type: str,
    fetch_logger: FetchLogger,
    is_original_inception_request: bool = False
) -> Optional[vbt.Data]:
    """Try to fetch data through resampling from lower timeframes."""
    
    try:
        # Get start dates only for symbols we're actually fetching
        fetch_symbol_start_dates = {
            symbol: symbol_start_dates.get(symbol)
            for symbol in symbols
            if symbol in symbol_start_dates
        }
        
        # Use the earliest start date from symbols we're fetching
        earliest_start = None
        if fetch_symbol_start_dates:
            earliest_start = min(fetch_symbol_start_dates.values())
            logger.debug(f"Using earliest start date for {len(symbols)} symbols: {earliest_start}")
        
        resampled_data = fetch_with_storage_resampling_fallback(
            symbols=symbols,
            exchange_id=exchange_id,
            timeframe=timeframe,
            start_date=earliest_start,
            end_date=end_date,
            market_type=market_type,
            prefer_resampling=True,
            is_inception_request=is_original_inception_request,  # Use original intent, not computed start date
            require_fresh_data=True,  # Ensure we only use fresh data for resampling
            fetch_logger=fetch_logger  # Pass the real logger for proper source tracking
        )
        
        if resampled_data is not None:
            fetch_logger.log_resample_result(True)
            # Note: Symbol sources are now tracked inside the resampling functions, no need to override here
            return resampled_data
        else:
            fetch_logger.log_resample_result(False)
            return None
            
    except Exception as e:
        logger.debug(f"Resampling failed: {e}")
        fetch_logger.log_resample_result(False)
        return None

def _fetch_from_exchange(
    symbols: List[str],
    exchange_id: str,
    timeframe: str, 
    symbol_start_dates: Dict[str, str],
    end_date: Optional[Union[str, vbt.timestamp]],
    market_type: str,
    fetch_logger: FetchLogger
) -> Optional[vbt.Data]:
    """Fetch data directly from exchange using VBT's native methods."""
    
    try:
        # Get start dates only for symbols we're actually fetching
        fetch_symbol_start_dates = {
            symbol: symbol_start_dates.get(symbol)
            for symbol in symbols
            if symbol in symbol_start_dates and symbol_start_dates.get(symbol) is not None
        }
        
        # Calculate start date based on symbols we're fetching
        start_date = None
        if fetch_symbol_start_dates:
            if len(set(fetch_symbol_start_dates.values())) > 1:
                start_date = min(fetch_symbol_start_dates.values())
                logger.debug(f"Using earliest start date: {start_date}")
            else:
                start_date = list(fetch_symbol_start_dates.values())[0]
                logger.debug(f"Using start date: {start_date}")
        
        # Use VBT's native parallel fetching
        fetch_kwargs = {
            'show_progress': False  # Disable progress bar for cleaner logs
        }
        
        # Enable parallel processing for multiple symbols
        if len(symbols) > 1:
            fetch_kwargs['execute_kwargs'] = {'engine': 'threadpool'}
            logger.debug(f"Using parallel fetch for {len(symbols)} symbols")
        
        # Fetch using VBT's CCXTData
        if exchange_id.lower() in ['binance', 'bybit', 'okx', 'kucoin', 'coinbase']:
            data = vbt.CCXTData.pull(
                symbols,
                exchange=exchange_id.lower(),
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                **fetch_kwargs
            )
            
            if data is not None:
                fetch_logger.log_exchange_result(True, len(data.symbols))
                # Track all symbols as from exchange
                for symbol in symbols:
                    if symbol in data.symbols:
                        fetch_logger.set_symbol_source(symbol, "exchange", was_fetched=True)
                return data
            else:
                fetch_logger.log_exchange_result(False)
                return None
        else:
            logger.error(f"Exchange {exchange_id} not supported")
            fetch_logger.log_exchange_result(False)
            return None
            
    except Exception as e:
        logger.error(f"Exchange fetch failed: {e}")
        fetch_logger.log_exchange_result(False)
        return None

# Logging functions
def _log_fetch_start(
    symbols: List[str], 
    exchange_id: str, 
    timeframe: str,
    start_date: Optional[Union[str, vbt.timestamp]],
    end_date: Optional[Union[str, vbt.timestamp]]
) -> None:
    """Log fetch operation start."""
    
    panel = Panel(
        f"[bold cyan]Symbols:[/bold cyan] {len(symbols)}\n"
        f"[bold cyan]Exchange:[/bold cyan] {exchange_id.upper()}\n"
        f"[bold cyan]Timeframe:[/bold cyan] {timeframe}\n"
        f"[bold cyan]Date Range:[/bold cyan] {start_date or 'inception'} → {end_date or 'latest'}",
        title="🚀 Data Fetch Request",
        box=box.ROUNDED,
        style="blue"
    )
    console.print(panel)

def _log_success(method: str, symbol_count: int) -> None:
    """Log successful fetch operation."""
    
    panel = Panel(
        f"[bold green]✅ {method} successful![/bold green]\n"
        f"[cyan]Symbols:[/cyan] {symbol_count}\n"
        f"[cyan]Status:[/cyan] Ready for use",
        title="📊 Fetch Complete",
        box=box.ROUNDED,
        style="green"
    )
    console.print(panel)

# Top symbols convenience function
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
    Fetch top symbols by volume.
    
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
        VBT Data object with volume metadata or None
    """
    
    try:
        logger.debug(f"Selecting top {limit} symbols by volume from {exchange_id}")
        
        # Get volume data from cache
        volume_data = cache_manager.get_all_volumes(exchange_id)
        
        # Filter by quote currency if specified
        if quote_currency and volume_data:
            filtered_volume_data = {
                symbol: volume for symbol, volume in volume_data.items()
                if '/' in symbol and symbol.split('/')[1] == quote_currency
            }
            volume_data = filtered_volume_data
        
        if not volume_data:
            logger.error("No volume data available for symbol selection")
            return None
        
        # Select top symbols by volume
        sorted_symbols = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_symbols[:limit*2]]  # Get extra in case some are blacklisted
        
        # Apply blacklist filtering
        top_symbols = _filter_blacklisted_symbols(top_symbols, exchange_id)
        
        # Take only the requested limit after filtering
        top_symbols = top_symbols[:limit]
        
        if not top_symbols:
            logger.error("No symbols remaining after blacklist filtering")
            return None
        
        logger.debug(f"Selected symbols: {top_symbols}")
        
        # Fetch data for top symbols
        return fetch_data(
            symbols=top_symbols,
            exchange_id=exchange_id,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            market_type=market_type
        )
        
    except Exception as e:
        logger.error(f"Error fetching top symbols: {e}")
        return None

# Update function using VBT's native capabilities
def update_data(
    exchange_id: str,
    timeframe: str,
    symbols: Optional[List[str]] = None,
    market_type: str = 'spot'
) -> bool:
    """
    Update stored data using a combination of VBT's update() method and manual fetching.
    
    Args:
        exchange_id: Exchange identifier
        timeframe: Timeframe identifier  
        symbols: Specific symbols to update (None for all cached symbols)
        market_type: Market type ('spot' or 'swap')
        
    Returns:
        True if successful
    """
    
    try:
        # Initialize a simple update logger
        console.print(
            Panel(
                f"[bold cyan]{exchange_id.upper()}[/bold cyan] | "
                f"[cyan]{timeframe}[/cyan] | "
                f"[cyan]{market_type}[/cyan]",
                title="🔄 Data Update",
                expand=False,
                style="cyan"
            )
        )
        
        # Load existing data
        existing_data = data_storage.load_data(exchange_id, timeframe, symbols, market_type)
        
        if existing_data is None:
            console.print("   ❌ No existing data to update")
            return False
        
        # If symbols specified, check if they exist in cached data
        if symbols:
            available_symbols = set(existing_data.symbols)
            requested_symbols = set(symbols)
            missing_symbols = requested_symbols - available_symbols
            
            if missing_symbols:
                console.print(f"   ⚠️  Missing symbols: {list(missing_symbols)}")
                
            # Filter to only symbols that exist
            symbols_to_update = list(requested_symbols & available_symbols)
            if not symbols_to_update:
                console.print("   ❌ None of the requested symbols found")
                return False
                
            console.print(f"   📊 Updating {len(symbols_to_update)} symbols")
        else:
            symbols_to_update = list(existing_data.symbols)
            console.print(f"   📊 Updating all {len(symbols_to_update)} symbols")
        
        # Get the latest date in existing data
        latest_date = existing_data.wrapper.index[-1]
        
        # Try VBT's native update first
        logger.debug("Attempting VBT's native update() method")
        
        try:
            updated_data = existing_data.update(
                show_progress=False,
                silence_warnings=False
            )
            
            if updated_data is not None:
                # Check if any data was actually updated
                if hasattr(updated_data, 'wrapper') and hasattr(existing_data, 'wrapper'):
                    old_end = existing_data.wrapper.index[-1]
                    new_end = updated_data.wrapper.index[-1]
                    
                    if new_end > old_end:
                        console.print(f"   ✅ Updated: {old_end.date()} → {new_end.date()}")
                    else:
                        console.print(f"   ✅ Already up to date")
                
                # Save updated data
                success = data_storage.save_data(updated_data, exchange_id, timeframe, market_type)
                if success:
                    console.print(
                        Panel(
                            "[green]✓ Update complete[/green] | 💾 Cached",
                            expand=False,
                            style="green"
                        )
                    )
                    return True
                else:
                    console.print("   ❌ Failed to save updated data")
                    return False
            else:
                logger.debug("VBT update returned None - trying manual update")
                
        except NotImplementedError:
            logger.debug("VBT update() not implemented for this data type")
            
        except Exception as update_error:
            logger.debug(f"VBT update() failed: {str(update_error)}")
        
        # Manual update fallback
        console.print("   🔧 Using manual update...")
        
        # Calculate start date for new data (day after latest cached date)
        from datetime import timedelta
        next_date = latest_date + timedelta(days=1)
        start_date_str = next_date.strftime('%Y-%m-%d')
        
        # Fetch new data for the symbols
        new_data = fetch_data(
            symbols=symbols_to_update,
            exchange_id=exchange_id,
            timeframe=timeframe,
            start_date=start_date_str,
            end_date=None,
            use_cache=False,
            market_type=market_type,
            prefer_resampling=False
        )
        
        if new_data is not None and len(new_data.wrapper.index) > 0:
            # Check if we got new data beyond what we already have
            new_data_start = new_data.wrapper.index[0]
            new_data_end = new_data.wrapper.index[-1]
            
            if new_data_end > latest_date:
                console.print(f"   ✅ New data: {new_data_start.date()} → {new_data_end.date()}")
                
                # Combine old and new data
                try:
                    combined_data = existing_data.concat(new_data, drop_duplicates=True)
                    
                    # Save combined data
                    success = data_storage.save_data(combined_data, exchange_id, timeframe, market_type)
                    if success:
                        console.print(
                            Panel(
                                "[green]✓ Manual update complete[/green] | 💾 Cached",
                                expand=False,
                                style="green"
                            )
                        )
                        return True
                    else:
                        console.print("   ❌ Failed to save updated data")
                        return False
                        
                except Exception as concat_error:
                    console.print(f"   ❌ Failed to combine data: {concat_error}")
                    return False
            else:
                console.print("   ✅ Already up to date")
                return True
        else:
            console.print("   ✅ Already up to date")
            return True
            
    except Exception as e:
        console.print(f"   ❌ Update error: {e}")
        return False

# Storage info functions
def get_storage_info() -> Dict[str, Any]:
    """Get information about data storage."""
    return data_storage.get_storage_summary()

def get_resampling_info() -> Dict[str, Any]:
    """Get information about storage resampling capabilities."""
    from .storage_resampling import TIMEFRAME_HIERARCHY
    return {
        'supported_timeframes': TIMEFRAME_HIERARCHY,
        'resampling_type': 'storage_optimized_pandas',
        'storage_summary': data_storage.get_storage_summary()
    }

# Convenience functions for API compatibility
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
    """Convenience function matching existing API."""
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