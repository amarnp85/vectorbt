#!/usr/bin/env python3
"""Data Fetcher Module

Uses VectorBT Pro's native data fetching and persistence capabilities.
Leverages VBT's built-in caching, symbol alignment, and data management.
Enhanced with intelligent resampling fallback to optimize data retrieval.
"""

import logging
import pandas as pd
from typing import List, Optional, Union, Dict, Any
import vectorbtpro as vbt
from ..storage.data_storage import data_storage
from ..cache_system import cache_manager
from .storage_resampling import fetch_with_storage_resampling_fallback, TIMEFRAME_HIERARCHY

logger = logging.getLogger(__name__)

def _log_fetch_summary(data: vbt.Data, operation: str = "fetch") -> None:
    """
    Log detailed summary statistics of fetched data including per-symbol information.
    
    Args:
        data: VBT Data object
        operation: Type of operation performed (fetch, update, merge)
    """
    if data is None:
        return
    
    logger.info(f"📊 {operation.title()} Summary:")
    logger.info(f"   Total symbols: {len(data.symbols)}")
    logger.info(f"   Data shape: {data.wrapper.shape}")
    
    # Get the data as DataFrame for analysis
    try:
        df = data.get()
        
        # Per-symbol summary statistics
        logger.info(f"📈 Per-Symbol Details:")
        
        # Handle different data structures
        if hasattr(data, 'close') and data.close is not None:
            # OHLCV data structure
            close_data = data.close
            
            for symbol in data.symbols:
                try:
                    # Get symbol's close data
                    if len(data.symbols) > 1:
                        symbol_close = close_data[symbol]
                    else:
                        symbol_close = close_data
                    
                    # Get non-NaN data points
                    valid_data = symbol_close.dropna()
                    
                    if len(valid_data) > 0:
                        inception_ts = valid_data.index[0]
                        latest_ts = valid_data.index[-1]
                        total_points = len(valid_data)
                        
                        # Format timestamps nicely
                        inception_str = inception_ts.strftime('%Y-%m-%d %H:%M:%S UTC')
                        latest_str = latest_ts.strftime('%Y-%m-%d %H:%M:%S UTC')
                        
                        # Calculate data coverage
                        time_span = latest_ts - inception_ts
                        days_span = max(time_span.days, 1)
                        
                        logger.info(f"   📍 {symbol}:")
                        logger.info(f"      Inception: {inception_str}")
                        logger.info(f"      Latest:    {latest_str}")
                        logger.info(f"      Duration:  {days_span} days ({total_points:,} points)")
                        
                        # Calculate coverage based on timeframe
                        expected_points_per_day = 24  # Default hourly
                        freq_str = str(data.wrapper.freq) if hasattr(data.wrapper, 'freq') else '1h'
                        
                        try:
                            # Parse VBT frequency strings more robustly
                            if 'days' in freq_str and 'hours' in freq_str:
                                # Handle compound frequencies like "0 days 01:00:00"
                                import re
                                hours_match = re.search(r'(\d+):(\d+):(\d+)', freq_str)
                                if hours_match:
                                    hours = int(hours_match.group(1))
                                    minutes = int(hours_match.group(2))
                                    total_hours = hours + minutes / 60.0
                                    expected_points_per_day = 24 / total_hours if total_hours > 0 else 24
                                else:
                                    expected_points_per_day = 24  # Default to hourly
                            elif 'h' in freq_str.lower() or 'hour' in freq_str.lower():
                                # Extract hours from strings like "1h", "4H", "1 hour"
                                import re
                                hours_match = re.search(r'(\d+)', freq_str)
                                if hours_match:
                                    hours = int(hours_match.group(1))
                                    expected_points_per_day = 24 / hours
                            elif 'd' in freq_str.lower() or 'day' in freq_str.lower():
                                expected_points_per_day = 1
                            elif 'm' in freq_str.lower() or 'min' in freq_str.lower():
                                # Extract minutes from strings like "15m", "30min"
                                import re
                                minutes_match = re.search(r'(\d+)', freq_str)
                                if minutes_match:
                                    minutes = int(minutes_match.group(1))
                                    expected_points_per_day = 24 * 60 / minutes
                        except Exception as e:
                            logger.debug(f"Error parsing frequency {freq_str}: {e}")
                            expected_points_per_day = 24  # Default to hourly
                        
                        expected_total = days_span * expected_points_per_day
                        coverage = min((total_points / expected_total) * 100, 100) if expected_total > 0 else 100
                        
                        logger.info(f"      Coverage:  {coverage:.1f}% ({expected_points_per_day:.1f} expected/day)")
                        
                        # Get last close price
                        last_close = valid_data.iloc[-1]
                        logger.info(f"      Last Price: ${last_close:,.2f}")
                        
                        # Get first close price for comparison
                        first_close = valid_data.iloc[0]
                        change_pct = ((last_close - first_close) / first_close) * 100
                        change_emoji = "📈" if change_pct >= 0 else "📉"
                        logger.info(f"      Period Change: {change_emoji} {change_pct:+.2f}%")
                        
                    else:
                        logger.info(f"   📍 {symbol}: No valid data points")
                        
                except Exception as e:
                    logger.debug(f"   📍 {symbol}: Error analyzing data - {e}")
        
        else:
            # Generic data structure - use the first column or overall DataFrame
            logger.info(f"   📍 Generic data structure detected")
            
            if len(df.index) > 0:
                inception_ts = df.index[0]
                latest_ts = df.index[-1]
                total_points = len(df.index)
                
                # Format timestamps
                inception_str = inception_ts.strftime('%Y-%m-%d %H:%M:%S UTC')
                latest_str = latest_ts.strftime('%Y-%m-%d %H:%M:%S UTC')
                
                # Time span
                time_span = latest_ts - inception_ts
                days_span = max(time_span.days, 1)
                
                logger.info(f"      Inception: {inception_str}")
                logger.info(f"      Latest:    {latest_str}")
                logger.info(f"      Duration:  {days_span} days ({total_points:,} points)")
        
        # Overall data quality metrics
        if hasattr(df, 'notna'):
            total_possible_points = df.size
            actual_points = df.notna().sum().sum()
            data_completeness = (actual_points / total_possible_points) * 100 if total_possible_points > 0 else 0
            
            logger.info(f"📊 Data Quality:")
            logger.info(f"   Completeness: {data_completeness:.1f}% ({actual_points:,}/{total_possible_points:,} data points)")
            logger.info(f"   Missing data: {total_possible_points - actual_points:,} points")
        
    except Exception as e:
        logger.debug(f"Error in fetch summary analysis: {e}")
        # Fallback to basic info
        logger.info(f"📊 Basic Summary:")
        logger.info(f"   Symbols: {list(data.symbols)}")
        logger.info(f"   Data available: ✅")

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
    Fetch cryptocurrency data using VBT's native data sources with smart incremental caching
    and intelligent resampling fallback.
    
    This approach leverages VBT's native capabilities:
    1. Uses VBT's built-in update() for incremental data fetching
    2. Uses VBT's merge() for combining data sources
    3. Uses VBT's datetime handling instead of pandas
    4. Preserves all VBT functionality and metadata
    5. Intelligent resampling from lower timeframes before API calls
    
    Args:
        symbols: List of trading symbols
        exchange_id: Exchange identifier
        timeframe: Timeframe string
        start_date: Start date for data fetch (VBT timestamp or string)
        end_date: End date for data fetch (VBT timestamp or string)
        use_cache: Whether to use VBT storage cache
        market_type: Market type ('spot' or 'swap')
        prefer_resampling: Whether to try resampling before API fetch
        
    Returns:
        VBT Data object or None
    """
    
    try:
        logger.info(f"🚀 Initiating data fetch: {len(symbols)} symbols, {timeframe}")
        
        # Try storage resampling fallback first if enabled
        if prefer_resampling:
            logger.info(f"🔄 Attempting storage resampling fallback for {timeframe}")
            resampled_data = fetch_with_storage_resampling_fallback(
                symbols=symbols,
                exchange_id=exchange_id,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                market_type=market_type,
                prefer_resampling=True
            )
            
            if resampled_data is not None:
                logger.info(f"✅ Storage resampling successful for {timeframe}")
                _log_fetch_summary(resampled_data, "resampling")
                return resampled_data
            
            logger.debug(f"🔄 Storage resampling unavailable, proceeding with exchange fetch")
        
        cached_data = None
        missing_symbols = symbols.copy()
        
        # Try to load from storage cache first
        if use_cache:
            logger.info(f"📦 Checking cache for {exchange_id}_{market_type}_{timeframe}")
            cached_data = data_storage.load_data(exchange_id, timeframe, market_type=market_type)
            
            if cached_data is not None:
                logger.info(f"📦 Found cached data with {len(cached_data.symbols)} symbols")
                
                # Check which symbols are missing from cache
                cached_symbols = set(cached_data.symbols)
                requested_symbols = set(symbols)
                missing_symbols = list(requested_symbols - cached_symbols)
                
                if missing_symbols:
                    logger.info(f"🔍 Cache missing symbols: {missing_symbols}")
                else:
                    # All symbols in cache - check if we need date range updates
                    logger.info("✅ All requested symbols found in cache")
                    
                    # For date range checking, VBT can handle this natively via update()
                    # If end_date is specified and it's beyond cached data, VBT update() will fetch the gap
                    if end_date:
                        try:
                            logger.info(f"📈 Attempting cache update to {end_date}")
                            # Use VBT's native update with end date
                            updated_data = cached_data.update(end=end_date)
                            if updated_data is not None:
                                logger.info("📈 Successfully updated cache with latest data")
                                # Save updated cache
                                data_storage.save_data(updated_data, exchange_id, timeframe, market_type)
                                cached_data = updated_data
                        except Exception as e:
                            logger.debug(f"Cache update attempt failed: {e}")
                    
                    # Return filtered data for requested symbols
                    filtered_data = cached_data.select(symbols)
                    logger.info(f"✅ Cache hit: returning {len(symbols)} symbols from cache")
                    _log_fetch_summary(filtered_data, "cache")
                    return filtered_data
            else:
                logger.info("💾 No cached data found")
        
        # Fetch missing data using VBT's native methods with enhanced logging
        if missing_symbols:
            logger.info(f"📡 Exchange Fetch Details:")
            logger.info(f"   Exchange: {exchange_id.upper()}")
            logger.info(f"   Market: {market_type.upper()}")
            logger.info(f"   Timeframe: {timeframe}")
            logger.info(f"   Symbols: {missing_symbols}")
            logger.info(f"   Date range: {start_date or 'inception'} → {end_date or 'latest'}")
            
            # Use VBT's CCXTData for crypto exchanges with native progress tracking
            if exchange_id.lower() in ['binance', 'bybit', 'okx', 'kucoin', 'coinbase']:
                # Convert string dates to VBT timestamps if needed
                vbt_start = start_date
                vbt_end = end_date
                if isinstance(start_date, str):
                    vbt_start = start_date  # VBT handles string dates natively
                if isinstance(end_date, str):
                    vbt_end = end_date  # VBT handles string dates natively
                
                logger.info(f"🔗 Initiating CCXT fetch with VBT native progress tracking...")
                
                # Enable VBT's native progress tracking and optimizations
                fetch_kwargs = {
                    'show_progress': True,  # Enable VBT progress bars
                    'silence_warnings': False,  # Show VBT warnings for transparency
                    'pbar_kwargs': {
                        'desc': f'Fetching {exchange_id.upper()} {timeframe}',
                        'unit': 'symbol'
                    }
                }
                
                # Use parallel fetching for multiple symbols
                if len(missing_symbols) > 1:
                    fetch_kwargs['execute_kwargs'] = dict(engine="threadpool")
                    logger.info(f"⚡ Using parallel fetch for {len(missing_symbols)} symbols")
                
                fresh_data = vbt.CCXTData.pull(
                    missing_symbols,
                    exchange=exchange_id.lower(),
                    timeframe=timeframe,
                    start=vbt_start,
                    end=vbt_end,
                    **fetch_kwargs
                )
            else:
                logger.error(f"❌ Exchange {exchange_id} not supported in CCXTData")
                return None
            
            if fresh_data is None:
                logger.error("❌ Exchange fetch returned None - no data available")
                return None
            
            logger.info(f"✅ Exchange fetch completed successfully")
            _log_fetch_summary(fresh_data, "exchange_fetch")
            
            # Handle data combination using VBT's native merge
            if cached_data is not None and use_cache:
                logger.info("🔄 Merging cached and fresh data using VBT.merge()")
                
                try:
                    # Use VBT's native merge functionality
                    combined_data = vbt.Data.merge(
                        cached_data, 
                        fresh_data, 
                        missing_columns="fill"  # Handle missing symbols gracefully
                    )
                    
                    if combined_data is not None:
                        logger.info(f"✅ VBT merge successful: {len(combined_data.symbols)} total symbols")
                        _log_fetch_summary(combined_data, "merge")
                        
                        # Save combined data to cache
                        if use_cache:
                            success = data_storage.save_data(combined_data, exchange_id, timeframe, market_type)
                            if success:
                                logger.info(f"💾 Saved merged data to cache")
                        
                        # Return requested symbols from combined data
                        available_symbols = [s for s in symbols if s in combined_data.symbols]
                        if available_symbols:
                            result = combined_data.select(available_symbols)
                            logger.info(f"🎯 Returning {len(available_symbols)} symbols from merged data")
                            return result
                    
                except Exception as e:
                    logger.warning(f"⚠️ VBT merge failed: {e}, falling back to fresh data")
                
                # Fallback: just use fresh data and save it
                final_data = fresh_data
            else:
                # No cached data, just use fresh data
                final_data = fresh_data
            
            # Save fresh data to cache
            if use_cache:
                success = data_storage.save_data(final_data, exchange_id, timeframe, market_type)
                if success:
                    logger.info(f"💾 Saved fresh data to cache")
                else:
                    logger.warning(f"⚠️ Failed to save to cache")
            
            # Return requested symbols
            available_symbols = [s for s in symbols if s in final_data.symbols]
            if available_symbols:
                result = final_data.select(available_symbols)
                logger.info(f"✅ Returning {len(available_symbols)} requested symbols")
                return result
        
        return None
        
    except Exception as e:
        logger.error(f"❌ Error in data fetch: {e}")
        return None

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
    Fetch top symbols by volume using VBT's native data handling.
    
    Args:
        exchange_id: Exchange identifier
        quote_currency: Quote currency filter
        market_type: Market type filter
        limit: Number of top symbols to fetch
        timeframe: Timeframe string
        start_date: Start date (VBT timestamp or string)
        end_date: End date (VBT timestamp or string)
        use_cache: Whether to use caching
        
    Returns:
        VBT Data object with volume metadata or None
    """
    
    try:
        # Get top symbols by volume using existing cache system
        logger.info(f"📊 Selecting top {limit} symbols by volume from {exchange_id}")
        volume_data = cache_manager.get_all_volumes(exchange_id)
        
        # Filter by quote currency if specified
        if quote_currency and volume_data:
            filtered_volume_data = {}
            for symbol, volume in volume_data.items():
                if '/' in symbol and symbol.split('/')[1] == quote_currency:
                    filtered_volume_data[symbol] = volume
            volume_data = filtered_volume_data
        
        if not volume_data:
            logger.error("❌ No volume data available for symbol selection")
            return None
        
        # Select top symbols
        sorted_symbols = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
        top_symbols = [symbol for symbol, _ in sorted_symbols[:limit]]
        
        logger.info(f"📈 Selected top {len(top_symbols)} symbols by volume:")
        for i, (symbol, volume) in enumerate(sorted_symbols[:limit], 1):
            volume_str = f"{volume/1000000:.1f}M" if volume > 1000000 else f"{volume/1000:.1f}K"
            logger.info(f"   {i:2d}. {symbol:12s} - ${volume_str:>8s}")
        
        # Fetch data using main data fetcher
        data = fetch_data(
            symbols=top_symbols,
            exchange_id=exchange_id,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_cache,
            market_type=market_type
        )
        
        if data is not None:
            # Add volume metadata to VBT wrapper (preserve VBT functionality)
            if hasattr(data.wrapper, '_metadata'):
                data.wrapper._metadata['volume_data'] = {
                    symbol: volume_data.get(symbol, 0) for symbol in data.symbols
                }
            else:
                # Create metadata if it doesn't exist
                data.wrapper._metadata = {
                    'volume_data': {symbol: volume_data.get(symbol, 0) for symbol in data.symbols}
                }
            
            logger.info(f"✅ Top symbols data fetch complete: {len(data.symbols)} symbols")
        
        return data
        
    except Exception as e:
        logger.error(f"❌ Error in top symbols fetch: {e}")
        return None

def update_data(
    exchange_id: str,
    timeframe: str,
    symbols: Optional[List[str]] = None,
    market_type: str = 'spot'
) -> bool:
    """
    Update stored VBT data using VBT's native update() method.
    
    VBT Data objects support native updating - this automatically:
    - Scans data for latest timestamp
    - Uses it as start for fetching new data
    - Appends new data seamlessly
    
    Args:
        exchange_id: Exchange identifier
        timeframe: Timeframe identifier
        symbols: Specific symbols to update (None for all)
        market_type: Market type ('spot' or 'swap')
        
    Returns:
        True if successful
    """
    
    try:
        # Load existing data
        existing_data = data_storage.load_data(exchange_id, timeframe, symbols, market_type)
        
        if existing_data is None:
            logger.warning("⚠️ No existing data found for update")
            return False
        
        logger.info("🔄 Using VBT's native update() method with progress tracking")
        
        # VBT's native update automatically handles incremental fetching
        updated_data = existing_data.update(
            show_progress=True,
            pbar_kwargs={
                'desc': f'Updating {exchange_id.upper()} {timeframe}',
                'unit': 'symbol'
            }
        )
        
        if updated_data is not None:
            logger.info(f"✅ VBT native update completed")
            _log_fetch_summary(updated_data, "update")
            
            # Save updated data using VBT persistence
            success = data_storage.save_data(updated_data, exchange_id, timeframe, market_type)
            
            if success:
                logger.info(f"💾 Updated data saved to cache")
                return True
            else:
                logger.error(f"❌ Failed to save updated VBT data")
                return False
        else:
            logger.info("ℹ️ VBT update: no new data available")
            return True  # No new data is not an error
            
    except Exception as e:
        logger.error(f"❌ Error updating data: {e}")
        return False

def get_storage_info() -> Dict[str, Any]:
    """Get information about data storage."""
    return data_storage.get_storage_summary()

def get_resampling_info() -> Dict[str, Any]:
    """Get information about storage resampling capabilities."""
    return {
        'supported_timeframes': TIMEFRAME_HIERARCHY,
        'resampling_type': 'storage_optimized_pandas',
        'storage_summary': data_storage.get_storage_summary()
    }

# Convenience functions that match our existing API
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
    """Convenience function matching existing API with resampling support."""
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