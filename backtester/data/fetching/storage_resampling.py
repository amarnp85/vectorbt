#!/usr/bin/env python3
"""Storage-Focused Resampling Module

Implements efficient OHLCV resampling for data storage and caching purposes.
Uses pandas resampling for lightweight, storage-optimized timeframe conversion.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple, Any
import pandas as pd
import vectorbtpro as vbt
from ..storage.data_storage import data_storage
from ..cache_system import cache_manager
from datetime import datetime

logger = logging.getLogger(__name__)

# Timeframe hierarchy for downsampling (lower to higher timeframes)
TIMEFRAME_HIERARCHY = [
    '1m', '2m', '3m', '5m', '15m', '30m',  # Minutes
    '1h', '2h', '4h', '6h', '8h', '12h',   # Hours  
    '1d', '3d', '1w', '2w',                # Days and weeks
    '1M', '3M', '6M', '1y'                 # Months and years
]

def get_timeframe_hierarchy_position(timeframe: str) -> int:
    """Get position of timeframe in hierarchy (-1 if not found)."""
    try:
        return TIMEFRAME_HIERARCHY.index(timeframe)
    except ValueError:
        return -1

def can_resample_from_to(source_tf: str, target_tf: str) -> bool:
    """Check if we can resample from source to target timeframe."""
    source_pos = get_timeframe_hierarchy_position(source_tf)
    target_pos = get_timeframe_hierarchy_position(target_tf)
    
    # Can only resample from lower to higher timeframes
    return source_pos != -1 and target_pos != -1 and source_pos < target_pos

def resample_ohlcv_for_storage(
    data: vbt.Data,
    target_timeframe: str
) -> Optional[vbt.Data]:
    """
    Resample OHLCV data for storage purposes using VBT's actual data structure.
    
    ROBUST VBT Data Structure Handling:
    - Detects and handles multiple VBT attribute naming conventions
    - Supports lowercase (open, high, low, close, volume) and capitalized (Open, High, Low, Close, Volume)
    - Fallback to VBT's .get() method if direct attributes aren't available
    - Comprehensive error handling and logging for debugging
    
    Args:
        data: VBT Data object to resample
        target_timeframe: Target timeframe string
        
    Returns:
        Resampled VBT Data object optimized for storage
    """
    try:
        logger.debug(f"Storage resampling to {target_timeframe}")
        
        # First, inspect the data structure to understand what we're working with
        logger.debug(f"Inspecting VBT data structure...")
        logger.debug(f"   Data type: {type(data)}")
        
        # Get symbols first
        symbols = list(data.symbols) if hasattr(data, 'symbols') else []
        if not symbols:
            logger.error("No symbols found in VBT data")
            return None
        
        logger.debug(f"Found {len(symbols)} symbols: {symbols}")
        
        # Log all available attributes for debugging
        available_attrs = [attr for attr in dir(data) if not attr.startswith('_')]
        logger.debug(f"Available VBT data attributes: {available_attrs}")
        
        # Try multiple strategies to extract OHLCV data
        ohlcv_data = None
        extraction_method = None
        
        # Strategy 1: Try lowercase OHLCV attributes (data.open, data.high, etc.)
        lowercase_attrs = ['open', 'high', 'low', 'close', 'volume']
        if all(hasattr(data, attr) and getattr(data, attr) is not None for attr in lowercase_attrs):
            logger.debug("Using lowercase OHLCV attributes (open, high, low, close, volume)")
            ohlcv_data = {
                'open': data.open,
                'high': data.high,
                'low': data.low,
                'close': data.close,
                'volume': data.volume
            }
            extraction_method = "lowercase_attributes"
        
        # Strategy 2: Try capitalized OHLCV attributes (data.Open, data.High, etc.)
        elif all(hasattr(data, attr.capitalize()) and getattr(data, attr.capitalize()) is not None for attr in lowercase_attrs):
            logger.debug("Using capitalized OHLCV attributes (Open, High, Low, Close, Volume)")
            ohlcv_data = {
                'open': data.Open,
                'high': data.High,
                'low': data.Low,
                'close': data.Close,
                'volume': data.Volume
            }
            extraction_method = "capitalized_attributes"
        
        # Strategy 3: Try mixed case or other common variations
        elif hasattr(data, 'close') and data.close is not None:
            logger.debug("Using available attributes (mixed detection)")
            ohlcv_data = {}
            
            # Build OHLCV dict with whatever attributes we can find
            attr_mapping = {
                'open': ['open', 'Open', 'OPEN'],
                'high': ['high', 'High', 'HIGH'],
                'low': ['low', 'Low', 'LOW'],
                'close': ['close', 'Close', 'CLOSE'],
                'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'Vol']
            }
            
            for ohlcv_key, possible_attrs in attr_mapping.items():
                found = False
                for attr_name in possible_attrs:
                    if hasattr(data, attr_name) and getattr(data, attr_name) is not None:
                        ohlcv_data[ohlcv_key] = getattr(data, attr_name)
                        logger.debug(f"   Found {ohlcv_key} as {attr_name}")
                        found = True
                        break
                
                if not found:
                    logger.debug(f"Could not find {ohlcv_key} attribute")
                    # For non-critical attributes, we can continue without them
                    if ohlcv_key not in ['open', 'high', 'low', 'close']:
                        logger.debug(f"   Continuing without {ohlcv_key} (non-critical)")
                        continue
                    else:
                        logger.error(f"Missing critical OHLCV attribute: {ohlcv_key}")
                        ohlcv_data = None
                        break
            
            if ohlcv_data and len(ohlcv_data) >= 4:  # At least OHLC
                extraction_method = "mixed_attributes"
            else:
                ohlcv_data = None
        
        # Strategy 4: Try using VBT's .get() method to extract underlying data
        if ohlcv_data is None:
            logger.debug("Trying VBT .get() method extraction...")
            try:
                underlying_data = data.get()
                logger.debug(f"   Underlying data type: {type(underlying_data)}")
                logger.debug(f"   Underlying data shape: {underlying_data.shape if hasattr(underlying_data, 'shape') else 'unknown'}")
                
                if hasattr(underlying_data, 'columns'):
                    available_columns = list(underlying_data.columns)
                    logger.debug(f"   Available columns: {available_columns}")
                    
                    # Check if we have MultiIndex columns (symbol, feature) structure
                    if isinstance(underlying_data.columns, pd.MultiIndex):
                        logger.debug("Detected MultiIndex structure, extracting OHLCV by feature")
                        
                        # For MultiIndex, we need to extract by level
                        ohlcv_data = {}
                        
                        # Try to find OHLCV features in MultiIndex
                        try:
                            if underlying_data.columns.nlevels >= 2:
                                # Assume structure is (symbol, feature) or (feature, symbol)
                                level_0_values = set(underlying_data.columns.get_level_values(0))
                                level_1_values = set(underlying_data.columns.get_level_values(1))
                                
                                # Determine which level contains features (OHLCV)
                                feature_names = ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume']
                                features_in_level_0 = any(name in level_0_values for name in feature_names)
                                features_in_level_1 = any(name in level_1_values for name in feature_names)
                                
                                if features_in_level_1:
                                    # Features are in level 1, symbols in level 0
                                    logger.debug("   Features in level 1, symbols in level 0")
                                    for feature in ['open', 'high', 'low', 'close', 'volume']:
                                        # Try both cases
                                        for feature_case in [feature, feature.capitalize()]:
                                            if feature_case in level_1_values:
                                                # Extract all symbols for this feature
                                                feature_df = underlying_data.xs(feature_case, axis=1, level=1)
                                                ohlcv_data[feature] = feature_df
                                                logger.debug(f"   Extracted {feature} from MultiIndex level 1")
                                                break
                                elif features_in_level_0:
                                    # Features are in level 0, symbols in level 1
                                    logger.debug("   Features in level 0, symbols in level 1")
                                    for feature in ['open', 'high', 'low', 'close', 'volume']:
                                        # Try both cases
                                        for feature_case in [feature, feature.capitalize()]:
                                            if feature_case in level_0_values:
                                                # Extract all symbols for this feature
                                                feature_df = underlying_data.xs(feature_case, axis=1, level=0)
                                                ohlcv_data[feature] = feature_df
                                                logger.debug(f"   Extracted {feature} from MultiIndex level 0")
                                                break
                                                
                            if len(ohlcv_data) >= 4:  # At least OHLC
                                extraction_method = "multiindex_get"
                                logger.debug(f"MultiIndex extraction successful: {list(ohlcv_data.keys())}")
                            else:
                                ohlcv_data = None
                                
                        except Exception as multiindex_error:
                            logger.debug(f"MultiIndex extraction failed: {multiindex_error}")
                            ohlcv_data = None
                    
                    else:
                        # Simple column structure - might be single symbol or multiple symbols as columns
                        logger.debug("Detected simple column structure")
                        
                        # Check if columns look like OHLCV features
                        feature_columns = [col for col in available_columns 
                                         if any(feature in str(col).lower() for feature in ['open', 'high', 'low', 'close', 'volume'])]
                        
                        if feature_columns:
                            logger.debug(f"   Found OHLCV-like columns: {feature_columns}")
                            # This might be a single-symbol case with OHLCV as columns
                            
                            # Try to map columns to OHLCV
                            column_mapping = {}
                            for col in available_columns:
                                col_lower = str(col).lower()
                                if 'open' in col_lower:
                                    column_mapping['open'] = col
                                elif 'high' in col_lower:
                                    column_mapping['high'] = col
                                elif 'low' in col_lower:
                                    column_mapping['low'] = col
                                elif 'close' in col_lower:
                                    column_mapping['close'] = col
                                elif 'volume' in col_lower or 'vol' in col_lower:
                                    column_mapping['volume'] = col
                            
                            if len(column_mapping) >= 4:  # At least OHLC
                                logger.debug(f"   Mapped columns: {column_mapping}")
                                
                                # Create OHLCV data with symbol expansion
                                ohlcv_data = {}
                                for feature, col_name in column_mapping.items():
                                    # Create DataFrame with symbols as columns
                                    if len(symbols) == 1:
                                        # Single symbol case
                                        feature_df = pd.DataFrame({
                                            symbols[0]: underlying_data[col_name]
                                        })
                                    else:
                                        # Multiple symbols - assume data is already in correct format
                                        feature_df = underlying_data[[col_name]]
                                        feature_df.columns = symbols  # Rename to symbol names
                                    
                                    ohlcv_data[feature] = feature_df
                                
                                extraction_method = "simple_columns_get"
                                logger.debug(f"Simple column extraction successful: {list(ohlcv_data.keys())}")
                            else:
                                ohlcv_data = None
                        else:
                            # Columns might be symbol names - this is the symbols-as-columns case
                            logger.debug(f"   Columns appear to be symbols: {available_columns}")
                            
                            # In this case, we might need to use VBT's feature access methods
                            # This is a fallback case - we can't resample without OHLCV separation
                            logger.debug("Cannot determine OHLCV structure from simple symbol columns")
                            ohlcv_data = None
                else:
                    logger.debug("Underlying data has no columns attribute")
                    ohlcv_data = None
                    
            except Exception as get_error:
                logger.debug(f"VBT .get() method failed: {get_error}")
                ohlcv_data = None
        
        # Final check - did we successfully extract OHLCV data?
        if ohlcv_data is None or len(ohlcv_data) < 4:
            logger.error("Failed to extract OHLCV data using any method")
            logger.error(f"   Available methods tried: lowercase_attrs, capitalized_attrs, mixed_attrs, get_method")
            logger.error(f"   Data structure inspection:")
            logger.error(f"     - Type: {type(data)}")
            logger.error(f"     - Available attributes: {[attr for attr in dir(data) if not attr.startswith('_')]}")
            if hasattr(data, 'wrapper'):
                logger.error(f"     - Wrapper shape: {data.wrapper.shape}")
                logger.error(f"     - Wrapper columns: {data.wrapper.columns}")
            
            # Add helpful suggestion for incomplete OHLCV data
            if hasattr(data, 'wrapper') and len(data.wrapper.columns) == 1 and 'close' in str(data.wrapper.columns).lower():
                logger.error("💡 SOLUTION: The cached data only contains 'close' prices, not full OHLCV data.")
                logger.error("   This likely means the data was cached before OHLCV fetching was properly implemented.")
                logger.error("   To fix this issue:")
                logger.error("   1. Delete the incomplete cache file")
                logger.error("   2. Re-fetch the data to get complete OHLCV structure")
                logger.error("   3. The storage resampling will then work properly")
                
            return None
        
        logger.debug(f"OHLCV extraction successful using {extraction_method}")
        logger.debug(f"   Extracted features: {list(ohlcv_data.keys())}")
        
        # Convert timeframe to pandas-compatible format (avoiding deprecated notation)
        pandas_timeframe = target_timeframe
        # For pandas 2.0+: 'h' for hours (not 'H'), 'D' for days, 'W' for weeks, 'min' for minutes
        if target_timeframe.endswith('h'):
            # Hours: keep 'h' (new standard), avoid deprecated 'H'
            pandas_timeframe = target_timeframe  # Already correct
        elif target_timeframe.endswith('d'):
            # Days: use 'D' 
            pandas_timeframe = target_timeframe.replace('d', 'D')
        elif target_timeframe.endswith('w'):
            # Weeks: use 'W'
            pandas_timeframe = target_timeframe.replace('w', 'W')
        elif target_timeframe.endswith('m') and not target_timeframe.endswith('M'):
            # Minutes: use 'min' to distinguish from months ('M')
            pandas_timeframe = target_timeframe.replace('m', 'min')
        logger.debug(f"Using pandas timeframe: {pandas_timeframe}")
        
        # CRITICAL FIX: Perform per-symbol resampling to preserve individual inception dates
        logger.debug(f"Performing per-symbol resampling to {target_timeframe}")
        
        # Essential OHLCV with proper aggregation methods
        aggregation_methods = {
            'open': 'first',   # First value in the period
            'high': 'max',     # Maximum value in the period
            'low': 'min',      # Minimum value in the period
            'close': 'last',   # Last value in the period
            'volume': 'sum'    # Sum of volumes in the period
        }
        
        # Structure: {symbol: DataFrame with OHLCV columns}
        symbol_dict = {}
        
        try:
            for symbol in symbols:
                logger.debug(f"   Resampling symbol: {symbol}")
                
                # Extract OHLCV data for this specific symbol
                symbol_ohlcv = {}
                
                for feature in ['open', 'high', 'low', 'close', 'volume']:
                    if feature in ohlcv_data:
                        df = ohlcv_data[feature]
                        if symbol in df.columns:
                            # Extract only this symbol's data - preserves its unique date range
                            symbol_series = df[symbol].dropna()
                            if len(symbol_series) > 0:
                                symbol_ohlcv[feature] = symbol_series
                                logger.debug(f"     {symbol} {feature}: {len(symbol_series)} data points")
                            else:
                                logger.debug(f"     {symbol} {feature}: no valid data")
                        else:
                            logger.debug(f"     {symbol} not found in {feature} data")
                
                # Skip symbol if we don't have enough OHLCV data
                critical_features = ['open', 'high', 'low', 'close']
                available_critical = [f for f in critical_features if f in symbol_ohlcv]
                
                if len(available_critical) < 4:
                    logger.warning(f"   ⚠️ Skipping {symbol}: insufficient OHLCV data ({available_critical})")
                    continue
                
                # Create DataFrame for this symbol's OHLCV data
                symbol_df = pd.DataFrame(symbol_ohlcv)
                
                if symbol_df.empty:
                    logger.warning(f"   ⚠️ Skipping {symbol}: empty DataFrame after extraction")
                    continue
                
                # Log symbol's original date range
                symbol_start = symbol_df.index.min()
                symbol_end = symbol_df.index.max()
                logger.debug(f"     {symbol} original range: {symbol_start} to {symbol_end} ({len(symbol_df)} rows)")
                
                # Resample this symbol's data individually - preserves its unique date range
                resampled_symbol_ohlcv = {}
                
                for feature, method in aggregation_methods.items():
                    if feature in symbol_df.columns:
                        feature_series = symbol_df[feature]
                        
                        if method == 'first':
                            resampled_series = feature_series.resample(pandas_timeframe).first()
                        elif method == 'max':
                            resampled_series = feature_series.resample(pandas_timeframe).max()
                        elif method == 'min':
                            resampled_series = feature_series.resample(pandas_timeframe).min()
                        elif method == 'last':
                            resampled_series = feature_series.resample(pandas_timeframe).last()
                        elif method == 'sum':
                            resampled_series = feature_series.resample(pandas_timeframe).sum()
                        else:
                            # Fallback
                            resampled_series = feature_series.resample(pandas_timeframe).last()
                        
                        # Remove NaN values for this feature
                        resampled_series = resampled_series.dropna()
                        resampled_symbol_ohlcv[feature.capitalize()] = resampled_series
                        
                        logger.debug(f"       {feature}: {len(feature_series)} → {len(resampled_series)} rows")
                
                # Create resampled DataFrame for this symbol (with capitalized column names for VBT)
                if resampled_symbol_ohlcv:
                    resampled_symbol_df = pd.DataFrame(resampled_symbol_ohlcv)
                    
                    # Remove any rows where critical OHLC data is missing
                    critical_cols = ['Open', 'High', 'Low', 'Close']
                    available_critical_cols = [col for col in critical_cols if col in resampled_symbol_df.columns]
                    
                    if available_critical_cols:
                        # Only keep rows where all critical OHLC data is present
                        valid_mask = resampled_symbol_df[available_critical_cols].notna().all(axis=1)
                        resampled_symbol_df = resampled_symbol_df[valid_mask]
                    
                    if not resampled_symbol_df.empty:
                        # Log the final resampled date range for this symbol
                        resampled_start = resampled_symbol_df.index.min()
                        resampled_end = resampled_symbol_df.index.max()
                        logger.debug(f"     {symbol} resampled range: {resampled_start} to {resampled_end} ({len(resampled_symbol_df)} rows)")
                        
                        symbol_dict[symbol] = resampled_symbol_df
                    else:
                        logger.warning(f"   ⚠️ {symbol}: no valid data after resampling and cleanup")
                else:
                    logger.warning(f"   ⚠️ {symbol}: no resampled data generated")
            
            if not symbol_dict:
                logger.error("No valid symbol data after per-symbol resampling")
                return None
            
            logger.debug(f"Per-symbol resampling successful: {len(symbol_dict)}/{len(symbols)} symbols")
            
            # Create VBT Data object from per-symbol resampled data
            logger.debug(f"Creating VBT Data from {len(symbol_dict)} resampled symbols")
            
            # Create VBT Data object - each symbol preserves its unique date range
            resampled_vbt_data = vbt.Data.from_data(symbol_dict)
            
            if resampled_vbt_data is None:
                logger.error("Failed to create VBT Data from per-symbol resampled data")
                return None
            
            # Verify the result
            if hasattr(resampled_vbt_data, 'symbols'):
                final_symbols = list(resampled_vbt_data.symbols)
                logger.debug(f"VBT Data reconstruction successful: {len(final_symbols)} symbols")
                logger.debug(f"   Final symbols: {final_symbols}")
                logger.debug(f"   Final shape: {resampled_vbt_data.wrapper.shape}")
                
                # Log each symbol's final date range to verify inception dates are preserved
                for symbol in final_symbols:
                    try:
                        if len(final_symbols) > 1:
                            symbol_close_data = resampled_vbt_data.close[symbol].dropna()
                        else:
                            symbol_close_data = resampled_vbt_data.close.dropna()
                        
                        if len(symbol_close_data) > 0:
                            symbol_start = symbol_close_data.index.min()
                            symbol_end = symbol_close_data.index.max()
                            logger.debug(f"     {symbol} final range: {symbol_start} to {symbol_end}")
                        else:
                            logger.debug(f"     {symbol} final range: no data")
                    except Exception as e:
                        logger.debug(f"     {symbol} final range: error getting range - {e}")
            else:
                logger.debug("No symbols metadata in final VBT data")
            
            return resampled_vbt_data
            
        except Exception as resample_error:
            logger.error(f"Resampling operation failed: {resample_error}")
            import traceback
            logger.debug(f"Resampling traceback: {traceback.format_exc()}")
            return None
        
    except Exception as e:
        logger.error(f"Error in storage resampling: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None

def find_available_lower_timeframes(
    target_timeframe: str,
    exchange_id: str,
    market_type: str = 'spot'
) -> List[str]:
    """Find available lower timeframes that can be used for resampling."""
    available_timeframes = []
    target_pos = get_timeframe_hierarchy_position(target_timeframe)
    
    if target_pos == -1:
        return available_timeframes
    
    # Check all lower timeframes
    for i in range(target_pos):
        lower_tf = TIMEFRAME_HIERARCHY[i]
        if data_storage.data_exists(exchange_id, lower_tf, market_type):
            available_timeframes.append(lower_tf)
    
    return available_timeframes

def get_best_resampling_source(
    target_timeframe: str,
    symbols: List[str],
    exchange_id: str,
    market_type: str = 'spot',
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    is_inception_request: bool = False
) -> Optional[Tuple[str, vbt.Data]]:
    """
    Find the best available lower timeframe data for storage resampling.
    Now includes freshness check to prevent resampling from stale data.
    """
    try:
        available_timeframes = find_available_lower_timeframes(
            target_timeframe, exchange_id, market_type
        )
        
        if not available_timeframes:
            logger.debug(f"No lower timeframes available for {target_timeframe}")
            return None
        
        # Try timeframes from highest to lowest (closest to target first)
        for source_tf in reversed(available_timeframes):
            logger.debug(f"Checking {source_tf} for resampling to {target_timeframe}")
            logger.debug(f"   is_inception_request = {is_inception_request}")
            
            # Load data for this timeframe - load all available symbols first
            source_data = data_storage.load_data(
                exchange_id, source_tf, [], market_type  # Load all available symbols
            )
            
            if source_data is not None:
                # Check if all requested symbols are available
                available_symbols = set(source_data.symbols)
                required_symbols = set(symbols)
                
                if required_symbols.issubset(available_symbols):
                    # CRITICAL FIX: Check if source data is fresh before using for resampling
                    if end_date is not None:
                        logger.debug(f"Checking freshness of {source_tf} data before resampling")
                        
                        # Import the freshness check function
                        from .data_fetcher import _is_data_fresh
                        
                        # Check if the source data is fresh enough
                        filtered_source = source_data.select_symbols(symbols)
                        if not _is_data_fresh(filtered_source, end_date, source_tf):
                            logger.debug(f"Source data in {source_tf} is stale, skipping for resampling")
                            continue
                        else:
                            logger.debug(f"Source data in {source_tf} is fresh, suitable for resampling")
                    
                    # CRITICAL FIX: Check inception completeness for inception requests
                    if is_inception_request:
                        logger.debug(f"Checking inception completeness of {source_tf} data before resampling")
                        
                        filtered_source = source_data.select_symbols(symbols)
                        inception_complete = _check_inception_completeness_simple(
                            filtered_source, symbols, exchange_id, is_inception_request
                        )
                        
                        if not inception_complete:
                            logger.debug(f"Source data in {source_tf} incomplete for inception request, skipping for resampling")
                            continue
                        else:
                            logger.debug(f"Source data in {source_tf} has complete inception data, suitable for resampling")
                    
                    logger.debug(f"Found complete and fresh data in {source_tf} for storage resampling")
                    # Filter to requested symbols
                    filtered_data = source_data.select_symbols(symbols)
                    return source_tf, filtered_data
                else:
                    missing = required_symbols - available_symbols
                    logger.debug(f"Missing symbols in {source_tf}: {missing}")
        
        logger.debug(f"No suitable fresh lower timeframe found for resampling")
        return None
        
    except Exception as e:
        logger.error(f"Error finding resampling source: {e}")
        return None 

def _use_cached_data_if_fresh(
    exact_data: vbt.Data,
    symbols: List[str],
    end_date: Optional[Union[str, vbt.timestamp]],
    timeframe: str,
    require_fresh_data: bool
) -> Optional[vbt.Data]:
    """
    Use cached data if it passes freshness checks, otherwise return None.
    
    Args:
        exact_data: The cached VBT data
        symbols: List of symbols to extract
        end_date: Target end date for freshness check
        timeframe: Timeframe for freshness calculation
        require_fresh_data: Whether to enforce freshness
        
    Returns:
        Filtered cached data if fresh, None if stale
    """
    # Check freshness if required
    if end_date is not None and require_fresh_data:
        try:
            # Import the freshness check function
            from .data_fetcher import _is_data_fresh
            
            # Check if data is fresh
            requested_data = exact_data.select_symbols(symbols) if len(symbols) < len(exact_data.symbols) else exact_data
            
            if not _is_data_fresh(requested_data, end_date, timeframe):
                logger.debug(f"Cached {timeframe} data is stale, will not use for resampling")
                return None  # Let it fall through to resampling/exchange fetch
            else:
                logger.debug(f"Cached {timeframe} data is fresh")
        except Exception as e:
            logger.warning(f"Could not check data freshness: {e}, assuming stale")
            return None
    else:
        logger.debug(f"No end_date specified or freshness check disabled, using cached data")
    
    # Return cached data (filtered if needed)
    if len(symbols) == len(exact_data.symbols):
        return exact_data
    else:
        try:
            return exact_data.select_symbols(symbols)
        except:
            logger.warning(f"Could not select symbols {symbols}, returning all data")
            return exact_data

def fetch_with_storage_resampling_fallback(
    symbols: List[str],
    exchange_id: str,
    timeframe: str,
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    market_type: str = 'spot',
    prefer_resampling: bool = True,
    is_inception_request: bool = False,
    require_fresh_data: bool = True,
    fetch_logger: Optional[Any] = None
) -> Optional[vbt.Data]:
    """
    Fetch data with storage-optimized resampling fallback.
    
    Strategy:
    1. Check for exact timeframe data first
    2. If not available or stale, try efficient storage resampling from lower timeframes
    3. Cache the resampled result for future use
    4. Fall back to API fetch if resampling not possible
    
    Args:
        require_fresh_data: If True, will not return stale data (default: True)
    """
    try:
        logger.debug(f"Fetch with storage resampling fallback: {timeframe}")
        
        # First, try to load exact timeframe data
        exact_data = data_storage.load_data(exchange_id, timeframe, symbols, market_type)
        if exact_data is not None:
            # Check if all symbols are available
            available_symbols = set(exact_data.symbols)
            required_symbols = set(symbols)
            
            if required_symbols.issubset(available_symbols):
                logger.debug(f"Found exact timeframe data: {timeframe}")
                
                # Check inception completeness FIRST if this is an inception request
                if is_inception_request:
                    inception_complete = _check_inception_completeness_simple(
                        exact_data, symbols, exchange_id, is_inception_request
                    )
                    if not inception_complete:
                        logger.debug(f"Storage resampling: cached {timeframe} data incomplete for inception request, will resample from lower timeframes")
                        # Skip cached data and continue to resampling logic below
                        pass
                    else:
                        logger.debug(f"Storage resampling: cached {timeframe} inception data is complete")
                        # Cached data has complete inception - proceed with freshness check and use it
                        return _use_cached_data_if_fresh(exact_data, symbols, end_date, timeframe, require_fresh_data)
                else:
                    # For non-inception requests, cached data is fine - proceed with freshness check
                    return _use_cached_data_if_fresh(exact_data, symbols, end_date, timeframe, require_fresh_data)
        
        # If prefer_resampling and we can resample, try that
        if prefer_resampling:
            # First try partial resampling (new approach)
            partial_resampled_data = fetch_with_partial_resampling(
                symbols, exchange_id, timeframe, start_date, end_date, 
                market_type, is_inception_request, require_fresh_data, fetch_logger
            )
            
            if partial_resampled_data is not None:
                logger.debug(f"Partial resampling successful for {timeframe}")
                
                # CRITICAL: Check if resampled data is fresh for target timeframe
                if end_date is not None and require_fresh_data:
                    from .data_fetcher import _is_data_fresh
                    if not _is_data_fresh(partial_resampled_data, end_date, timeframe):
                        logger.debug(f"Partial resampled {timeframe} data is stale for target timeframe, trying to update")
                        
                        # Try to update the resampled data to current timestamp
                        updated_data = _try_update_resampled_data(
                            partial_resampled_data, symbols, exchange_id, timeframe, 
                            end_date, market_type, fetch_logger
                        )
                        
                        if updated_data is not None:
                            logger.debug(f"Successfully updated partial resampled data")
                            partial_resampled_data = updated_data
                        else:
                            logger.debug(f"Failed to update partial resampled data, falling back to exchange")
                            return None  # Fall back to exchange fetch
                
                # Cache the result for future use
                success = data_storage.save_data(
                    partial_resampled_data, exchange_id, timeframe, market_type
                )
                
                if success:
                    logger.debug(f"Cached partial-resampled {timeframe} data")
                
                # Return the data
                return partial_resampled_data
            
            # Fallback to original full resampling approach if partial didn't work
            resampling_result = get_best_resampling_source(
                timeframe, symbols, exchange_id, market_type, end_date, is_inception_request
            )
            
            if resampling_result is not None:
                source_tf, source_data = resampling_result
                
                logger.debug(f"Storage resampling from {source_tf} to {timeframe}")
                
                # Use storage-optimized resampling
                resampled_data = resample_ohlcv_for_storage(source_data, timeframe)
                
                if resampled_data is not None:
                    # CRITICAL: Check if resampled data is fresh for target timeframe
                    if end_date is not None and require_fresh_data:
                        from .data_fetcher import _is_data_fresh
                        if not _is_data_fresh(resampled_data, end_date, timeframe):
                            logger.debug(f"Standard resampled {timeframe} data is stale for target timeframe, trying to update")
                            
                            # Try to update the resampled data to current timestamp
                            updated_data = _try_update_resampled_data(
                                resampled_data, symbols, exchange_id, timeframe, 
                                end_date, market_type, fetch_logger
                            )
                            
                            if updated_data is not None:
                                logger.debug(f"Successfully updated standard resampled data")
                                resampled_data = updated_data
                            else:
                                logger.debug(f"Failed to update standard resampled data, falling back to exchange")
                                return None  # Fall back to exchange fetch
                    
                    # Cache the resampled data for future use
                    success = data_storage.save_data(
                        resampled_data, exchange_id, timeframe, market_type
                    )
                    
                    if success:
                        logger.debug(f"Cached storage-resampled {timeframe} data")
                    
                    # Return filtered data - check if select is needed
                    if hasattr(resampled_data, 'symbols') and set(symbols).issubset(set(resampled_data.symbols)):
                        if len(symbols) == len(resampled_data.symbols):
                            # All symbols match, return as-is
                            return resampled_data
                        else:
                            # Need to filter symbols
                            try:
                                return resampled_data.select_symbols(symbols)
                            except:
                                # Fallback: just return all data if select fails
                                logger.warning(f"Could not select symbols {symbols}, returning all data")
                                return resampled_data
                    else:
                        return resampled_data
        
        # No resampling available or preferred - return None to trigger API fetch
        logger.debug(f"No storage resampling available for {timeframe}, will use API fetch")
        return None
        
    except Exception as e:
        logger.error(f"Error in storage resampling fallback: {e}")
        return None

def validate_storage_resampled_data(
    original_data: vbt.Data,
    resampled_data: vbt.Data,
    target_timeframe: str
) -> bool:
    """
    Validate that storage-resampled data maintains proper OHLCV relationships.
    """
    try:
        logger.debug(f"Validating storage-resampled data for {target_timeframe}")
        
        # Basic checks
        if resampled_data is None:
            logger.error("Resampled data is None")
            return False
        
        original_df = original_data.get()
        resampled_df = resampled_data.get()
        
        # Check that resampled data has fewer or equal rows (key difference from VBT resampling)
        if len(resampled_df) > len(original_df):
            logger.error(f"Storage resampled data has more rows than original: {len(resampled_df)} > {len(original_df)}")
            return False
        
        # Validate OHLCV columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in resampled_df.columns:
                logger.error(f"Missing required column: {col}")
                return False
        
        # Basic OHLCV sanity checks
        if (resampled_df['High'] < resampled_df['Low']).any():
            logger.error("Invalid OHLCV: High < Low found")
            return False
            
        if (resampled_df['High'] < resampled_df['Open']).any():
            logger.error("Invalid OHLCV: High < Open found")
            return False
            
        if (resampled_df['High'] < resampled_df['Close']).any():
            logger.error("Invalid OHLCV: High < Close found")
            return False
        
        logger.debug(f"Storage resampled data validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating storage resampled data: {e}")
        return False

def _check_inception_completeness_simple(
    cached_data: vbt.Data,
    symbols: List[str],
    exchange_id: str,
    is_inception_request: bool
) -> bool:
    """
    Simple check if cached data starts from inception dates for storage resampling.
    
    Args:
        cached_data: Existing cached VBT data
        symbols: List of symbols to check  
        exchange_id: Exchange identifier
        is_inception_request: Whether this is an inception request
        
    Returns:
        True if all symbols have complete inception data, False otherwise
    """
    if not is_inception_request or cached_data is None:
        return True
    
    # Get cached inception dates
    cached_timestamps = cache_manager.get_all_timestamps(exchange_id)
    if not cached_timestamps:
        return False
    
    # Check each symbol's inception completeness
    for symbol in symbols:
        if symbol not in cached_data.symbols:
            return False
            
        if symbol not in cached_timestamps:
            return False
        
        try:
            # Get true inception date from cache
            true_inception_ms = cached_timestamps[symbol]
            true_inception_dt = datetime.fromtimestamp(true_inception_ms / 1000)
            
            # Get the first date in cached data for this symbol
            if len(cached_data.symbols) > 1:
                symbol_data = cached_data.close[symbol].dropna()
            else:
                symbol_data = cached_data.close.dropna()
            
            if len(symbol_data) == 0:
                return False
                
            cached_start_dt = symbol_data.index[0]
            
            # Remove timezone info for comparison if present
            if hasattr(cached_start_dt, 'tz') and cached_start_dt.tz is not None:
                cached_start_dt = cached_start_dt.tz_localize(None)
            if hasattr(true_inception_dt, 'tz') and true_inception_dt.tz is not None:
                true_inception_dt = true_inception_dt.tz_localize(None)
            
            # Check if cached data starts within 1 day of true inception
            time_diff = abs((cached_start_dt - true_inception_dt).days)
            if time_diff > 1:  # More than 1 day gap means incomplete
                logger.debug(f"    Storage resampling: {symbol} cached data starts {cached_start_dt.date()}, true inception {true_inception_dt.date()} (gap: {time_diff} days)")
                return False
            
        except Exception as e:
            logger.warning(f"    Storage resampling: {symbol} error checking inception completeness - {e}")
            return False
    
    return True

def get_best_resampling_source_with_partial(
    target_timeframe: str,
    symbols: List[str],
    exchange_id: str,
    market_type: str = 'spot',
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    is_inception_request: bool = False
) -> Optional[Tuple[str, vbt.Data, List[str], List[str]]]:
    """
    Find the best available lower timeframe data for storage resampling, supporting partial matches.
    Now includes freshness check to prevent resampling from stale data.
    
    Returns:
        Tuple of (source_timeframe, source_data, available_symbols, missing_symbols) or None
    """
    try:
        available_timeframes = find_available_lower_timeframes(
            target_timeframe, exchange_id, market_type
        )
        
        if not available_timeframes:
            logger.debug(f"No lower timeframes available for {target_timeframe}")
            return None
        
        # Try timeframes from highest to lowest (closest to target first)
        for source_tf in reversed(available_timeframes):
            logger.debug(f"Checking {source_tf} for partial resampling to {target_timeframe}")
            
            # Load data for this timeframe
            source_data = data_storage.load_data(
                exchange_id, source_tf, [], market_type  # Load all available symbols
            )
            
            if source_data is not None:
                # Check which symbols are available
                available_symbols_set = set(source_data.symbols)
                required_symbols_set = set(symbols)
                
                available_symbols = list(required_symbols_set.intersection(available_symbols_set))
                missing_symbols = list(required_symbols_set - available_symbols_set)
                
                if available_symbols:  # We have at least some symbols
                    # CRITICAL FIX: Check if source data is fresh before using for resampling
                    if end_date is not None and available_symbols:
                        logger.debug(f"Checking freshness of {source_tf} data before partial resampling")
                        
                        # Import the freshness check function
                        from .data_fetcher import _is_data_fresh
                        
                        # Check if the available symbols data is fresh
                        try:
                            available_source = source_data.select_symbols(available_symbols)
                            if not _is_data_fresh(available_source, end_date, source_tf):
                                logger.debug(f"Source data in {source_tf} is stale for available symbols, skipping")
                                continue
                            else:
                                logger.debug(f"Source data in {source_tf} is fresh for available symbols")
                        except Exception as e:
                            logger.warning(f"Error checking freshness for partial data: {e}, skipping {source_tf}")
                            continue
                    
                    # CRITICAL FIX: Check inception completeness for inception requests
                    if is_inception_request and available_symbols:
                        logger.debug(f"Checking inception completeness of {source_tf} data for partial resampling")
                        
                        try:
                            available_source = source_data.select_symbols(available_symbols)
                            inception_complete = _check_inception_completeness_simple(
                                available_source, available_symbols, exchange_id, is_inception_request
                            )
                            
                            if not inception_complete:
                                logger.debug(f"Source data in {source_tf} incomplete for inception request, skipping for partial resampling")
                                continue
                            else:
                                logger.debug(f"Source data in {source_tf} has complete inception data for available symbols")
                        except Exception as e:
                            logger.warning(f"Error checking inception completeness for partial data: {e}, skipping {source_tf}")
                            continue
                    
                    logger.debug(f"Found partial fresh data in {source_tf}: {len(available_symbols)}/{len(symbols)} symbols")
                    logger.debug(f"   Available: {available_symbols}")
                    if missing_symbols:
                        logger.debug(f"   Missing: {missing_symbols}")
                    
                    return source_tf, source_data, available_symbols, missing_symbols
                else:
                    logger.debug(f"No matching symbols in {source_tf}")
        
        logger.debug(f"No suitable fresh timeframe found for partial resampling")
        return None
        
    except Exception as e:
        logger.error(f"Error finding partial resampling source: {e}")
        return None

def fetch_with_partial_resampling(
    symbols: List[str],
    exchange_id: str,
    timeframe: str,
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    market_type: str = 'spot',
    is_inception_request: bool = False,
    require_fresh_data: bool = True,
    fetch_logger: Optional[Any] = None
) -> Optional[vbt.Data]:
    """
    Fetch data using partial resampling + exchange fetch for missing symbols.
    
    Strategy:
    1. Find which symbols are available in lower timeframe cache
    2. Resample those symbols (only if fresh)
    3. Fetch missing symbols from exchange
    4. Combine the results
    
    Args:
        require_fresh_data: If True, will only resample from fresh source data
    """
    try:
        logger.debug(f"Attempting partial resampling for {timeframe}")
        logger.debug(f"   is_inception_request = {is_inception_request}")
        
        # Get partial resampling data
        partial_result = get_best_resampling_source_with_partial(
            timeframe, symbols, exchange_id, market_type, end_date, is_inception_request
        )
        
        if partial_result is None:
            logger.debug(f"No partial resampling available, using full exchange fetch")
            return None
        
        source_tf, source_data, available_symbols, missing_symbols = partial_result
        
        # If we have all symbols, use regular resampling
        if not missing_symbols:
            logger.debug(f"All symbols available in {source_tf}, using standard resampling")
            # Filter source data to requested symbols
            filtered_source = source_data.select_symbols(available_symbols)
            return resample_ohlcv_for_storage(filtered_source, timeframe)
        
        # Partial resampling scenario
        logger.debug(f"Partial resampling: {len(available_symbols)} from cache, {len(missing_symbols)} from exchange")
        
        # Step 1: Resample available symbols
        resampled_data = None
        if available_symbols:
            logger.debug(f"Resampling {len(available_symbols)} symbols from {source_tf}")
            filtered_source = source_data.select_symbols(available_symbols)
            resampled_data = resample_ohlcv_for_storage(filtered_source, timeframe)
            
            if resampled_data is None:
                logger.error(f"Failed to resample available symbols")
                return None
            
            # Track resampled symbols if we have a fetch_logger
            if fetch_logger is not None:
                for symbol in available_symbols:
                    if symbol in resampled_data.symbols:
                        fetch_logger.set_symbol_source(symbol, "resampled", was_fetched=True)
        
        # Step 2: Fetch missing symbols from exchange if any
        exchange_data = None
        if missing_symbols:
            logger.debug(f"Fetching {len(missing_symbols)} missing symbols from exchange")
            
            # Import here to avoid circular imports
            from .data_fetcher import _fetch_from_exchange, _get_optimal_start_dates
            
            try:
                # Get optimal start dates for missing symbols
                symbol_start_dates = _get_optimal_start_dates(
                    missing_symbols, exchange_id, start_date
                )
                
                # Use provided fetch_logger or create a dummy one as fallback
                if fetch_logger is not None:
                    # Use the real fetch_logger for proper source tracking
                    exchange_logger = fetch_logger
                else:
                    # Fallback to dummy logger if none provided
                    class DummyFetchLogger:
                        def __init__(self):
                            self.symbol_sources = {}
                        
                        def log_exchange_result(self, success: bool, symbol_count: Optional[int] = None):
                            if success:
                                logger.debug(f"Exchange fetch successful: {symbol_count} symbols")
                            else:
                                logger.error(f"Exchange fetch failed")
                        
                        def set_symbol_source(self, symbol: str, source: str, was_cached: bool = False, was_fetched: bool = False, cache_end_time=None):
                            """Track symbol source for debugging purposes."""
                            self.symbol_sources[symbol] = {
                                'source': source,
                                'was_cached': was_cached,
                                'was_fetched': was_fetched,
                                'cache_end_time': cache_end_time
                            }
                            logger.debug(f"Tracked symbol {symbol} from {source}")
                    
                    exchange_logger = DummyFetchLogger()
                
                exchange_data = _fetch_from_exchange(
                    missing_symbols, exchange_id, timeframe, 
                    symbol_start_dates, end_date, market_type, exchange_logger
                )
                
                if exchange_data is None:
                    logger.error(f"Failed to fetch missing symbols from exchange")
                    # Return resampled data only if we have some
                    if resampled_data is not None:
                        logger.debug(f"Returning partial data (resampled only)")
                        return resampled_data
                    return None
                    
            except Exception as e:
                logger.error(f"Exchange fetch failed: {e}")
                if resampled_data is not None:
                    logger.debug(f"Returning partial data (resampled only)")
                    return resampled_data
                return None
        
        # Step 3: Combine resampled and exchange data
        if resampled_data is not None and exchange_data is not None:
            logger.debug(f"Combining resampled and exchange data")
            combined_data = _combine_vbt_data_sources(resampled_data, exchange_data)
            
            # CRITICAL: Check if combined data is fresh for target timeframe
            if combined_data is not None and end_date is not None and require_fresh_data:
                from .data_fetcher import _is_data_fresh
                if not _is_data_fresh(combined_data, end_date, timeframe):
                    logger.debug(f"Combined data is stale for target timeframe, trying to update")
                    
                    # Try to update the combined data to current timestamp
                    updated_data = _try_update_resampled_data(
                        combined_data, symbols, exchange_id, timeframe, 
                        end_date, market_type, fetch_logger
                    )
                    
                    if updated_data is not None:
                        logger.debug(f"Successfully updated combined data")
                        combined_data = updated_data
                    else:
                        logger.debug(f"Failed to update combined data, using as-is")
                        # Don't return None here since combined_data is better than nothing
            
            return combined_data
        elif resampled_data is not None:
            logger.debug(f"Returning resampled data only")
            
            # CRITICAL: Check if resampled-only data is fresh for target timeframe
            if end_date is not None and require_fresh_data:
                from .data_fetcher import _is_data_fresh
                if not _is_data_fresh(resampled_data, end_date, timeframe):
                    logger.debug(f"Resampled-only data is stale for target timeframe, trying to update")
                    
                    # Try to update the resampled data to current timestamp
                    updated_data = _try_update_resampled_data(
                        resampled_data, symbols, exchange_id, timeframe, 
                        end_date, market_type, fetch_logger
                    )
                    
                    if updated_data is not None:
                        logger.debug(f"Successfully updated resampled-only data")
                        resampled_data = updated_data
                    else:
                        logger.debug(f"Failed to update resampled-only data, using as-is")
                        # Don't return None here since resampled_data is better than nothing
            
            return resampled_data
        elif exchange_data is not None:
            logger.debug(f"Returning exchange data only")
            return exchange_data
        else:
            logger.error(f"No data available from either source")
            return None
            
    except Exception as e:
        logger.error(f"Error in partial resampling: {e}")
        import traceback
        logger.debug(f"Partial resampling traceback: {traceback.format_exc()}")
        return None 

def _try_update_resampled_data(
    resampled_data: vbt.Data,
    symbols: List[str],
    exchange_id: str,
    timeframe: str,
    target_end: Union[str, vbt.timestamp],
    market_type: str = 'spot',
    fetch_logger: Optional[Any] = None
) -> Optional[vbt.Data]:
    """
    Try to update resampled data to the target timestamp using multiple strategies.
    
    Strategy:
    1. Try VBT's native update() method (if supported)
    2. Fetch missing period from exchange and append
    3. Return None if all methods fail (triggers full exchange fallback)
    
    Args:
        resampled_data: The resampled data that needs updating
        symbols: List of symbols to update
        exchange_id: Exchange identifier
        timeframe: Target timeframe
        target_end: Target end timestamp
        market_type: Market type
        fetch_logger: Logger for tracking (optional)
        
    Returns:
        Updated VBT Data or None if update failed
    """
    try:
        logger.debug(f"Attempting to update resampled {timeframe} data to {target_end}")
        
        # Strategy 1: Try VBT's native update() method
        try:
            logger.debug("Trying VBT native update() method")
            
            # Convert target_end to proper timestamp
            if isinstance(target_end, str):
                if target_end.lower() == "now":
                    update_end = vbt.utc_timestamp()
                else:
                    update_end = vbt.timestamp(target_end, tz='UTC')
            else:
                update_end = vbt.timestamp(target_end, tz='UTC')
            
            updated_data = resampled_data.update(
                end=update_end,
                show_progress=False,
                silence_warnings=True
            )
            
            if updated_data is not None:
                logger.debug("VBT native update successful")
                return updated_data
            else:
                logger.debug("VBT native update returned None")
                
        except NotImplementedError:
            logger.debug("VBT native update not implemented for this data type")
        except Exception as update_error:
            logger.debug(f"VBT native update failed: {update_error}")
        
        # Strategy 2: Fetch missing period from exchange and append
        try:
            logger.debug("Trying manual update by fetching missing period")
            
            # Get the latest timestamp in resampled data
            latest_timestamp = resampled_data.wrapper.index[-1]
            
            # Calculate the start date for missing data (next period after latest)
            from datetime import timedelta
            if timeframe.endswith('d') or 'day' in timeframe.lower():
                start_for_missing = latest_timestamp + timedelta(days=1)
            elif timeframe.endswith('h') or 'hour' in timeframe.lower():
                hours = int(timeframe[:-1]) if timeframe[:-1].isdigit() else 1
                start_for_missing = latest_timestamp + timedelta(hours=hours)
            elif timeframe.endswith('m') or 'min' in timeframe.lower():
                minutes = int(timeframe[:-1]) if timeframe[:-1].isdigit() else 1
                start_for_missing = latest_timestamp + timedelta(minutes=minutes)
            else:
                # Default to 1 day
                start_for_missing = latest_timestamp + timedelta(days=1)
            
            start_date_str = start_for_missing.strftime('%Y-%m-%d')
            
            logger.debug(f"Fetching missing data from {start_date_str} to {target_end}")
            
            # Import exchange fetch function
            from .data_fetcher import _fetch_from_exchange, _get_optimal_start_dates
            
            # Get optimal start dates for the missing period
            symbol_start_dates = _get_optimal_start_dates(
                symbols, exchange_id, start_date_str
            )
            
            # Create logger for the exchange fetch
            if fetch_logger is not None:
                exchange_logger = fetch_logger
            else:
                # Create a minimal logger for this operation
                class UpdateLogger:
                    def log_exchange_result(self, success: bool, symbol_count: Optional[int] = None):
                        if success:
                            logger.debug(f"Missing period fetch successful: {symbol_count} symbols")
                        else:
                            logger.debug(f"Missing period fetch failed")
                    
                    def set_symbol_source(self, symbol: str, source: str, was_cached: bool = False, was_fetched: bool = False, cache_end_time=None):
                        logger.debug(f"Update: tracked symbol {symbol} from {source}")
                
                exchange_logger = UpdateLogger()
            
            # Fetch missing data from exchange
            missing_data = _fetch_from_exchange(
                symbols, exchange_id, timeframe, 
                symbol_start_dates, target_end, market_type, exchange_logger
            )
            
            if missing_data is not None and len(missing_data.wrapper.index) > 0:
                logger.debug(f"Successfully fetched missing period data: {missing_data.wrapper.shape}")
                
                # Combine original resampled data with missing period data
                try:
                    combined_data = resampled_data.concat(missing_data, drop_duplicates=True)
                    
                    if combined_data is not None:
                        logger.debug("Successfully combined resampled data with missing period")
                        return combined_data
                    else:
                        logger.debug("Failed to combine resampled data with missing period")
                        
                except Exception as combine_error:
                    logger.debug(f"Error combining resampled data with missing period: {combine_error}")
            else:
                logger.debug("No missing period data available or fetched")
                
        except Exception as manual_error:
            logger.debug(f"Manual update strategy failed: {manual_error}")
        
        # All strategies failed
        logger.debug("All update strategies failed, returning None to trigger exchange fallback")
        return None
        
    except Exception as e:
        logger.error(f"Error in _try_update_resampled_data: {e}")
        return None

def _combine_vbt_data_sources(data1: vbt.Data, data2: vbt.Data) -> Optional[vbt.Data]:
    """
    Combine two VBT Data objects with potentially different symbols and date ranges.
    Uses VBT's native data creation methods for proper structure handling.
    """
    try:
        logger.debug(f"Combining VBT data sources")
        logger.debug(f"   Data1: {len(data1.symbols)} symbols, shape {data1.wrapper.shape}")
        logger.debug(f"   Data2: {len(data2.symbols)} symbols, shape {data2.wrapper.shape}")
        
        # Extract all symbols
        all_symbols = list(data1.symbols) + list(data2.symbols)
        logger.debug(f"   Combined symbols: {all_symbols}")
        
        # Find common date range by using VBT data directly instead of get() method
        try:
            # Get date ranges from VBT data wrapper index
            data1_start = data1.wrapper.index.min()
            data1_end = data1.wrapper.index.max()
            data2_start = data2.wrapper.index.min()
            data2_end = data2.wrapper.index.max()
            
            common_start = max(data1_start, data2_start)
            common_end = min(data1_end, data2_end)
            
            logger.debug(f"   Data1 range: {data1_start} to {data1_end}")
            logger.debug(f"   Data2 range: {data2_start} to {data2_end}")
            logger.debug(f"   Common date range: {common_start} to {common_end}")
            
        except Exception as date_error:
            logger.error(f"Error calculating date ranges from wrapper: {date_error}")
            return None
        
        # Get the underlying data as DataFrames for symbol extraction
        try:
            df1 = data1.get()
            df2 = data2.get()
            logger.debug(f"   Data1.get() type: {type(df1)}")
            logger.debug(f"   Data2.get() type: {type(df2)}")
        except Exception as get_error:
            logger.error(f"Error calling get() method: {get_error}")
            # Fall back to direct attribute access
            df1 = None
            df2 = None
        
        # Create symbol dictionary approach for VBT
        symbol_dict = {}
        
        # Process data1 symbols
        for symbol in data1.symbols:
            try:
                # Extract OHLCV data for this symbol from data1
                if hasattr(data1, 'open') and data1.open is not None:
                    # Direct attribute access (preferred method)
                    symbol_data = {
                        'Open': data1.open[symbol].loc[common_start:common_end] if len(data1.symbols) > 1 else data1.open.loc[common_start:common_end],
                        'High': data1.high[symbol].loc[common_start:common_end] if len(data1.symbols) > 1 else data1.high.loc[common_start:common_end],
                        'Low': data1.low[symbol].loc[common_start:common_end] if len(data1.symbols) > 1 else data1.low.loc[common_start:common_end],
                        'Close': data1.close[symbol].loc[common_start:common_end] if len(data1.symbols) > 1 else data1.close.loc[common_start:common_end],
                        'Volume': data1.volume[symbol].loc[common_start:common_end] if len(data1.symbols) > 1 and hasattr(data1, 'volume') else data1.volume.loc[common_start:common_end] if hasattr(data1, 'volume') else None
                    }
                elif df1 is not None and hasattr(df1, 'index'):
                    # Fallback to DataFrame extraction (only if get() worked)
                    if len(data1.symbols) > 1:
                        # MultiIndex case - extract by symbol
                        symbol_df = df1.xs(symbol, axis=1, level=0).loc[common_start:common_end]
                    else:
                        # Single symbol case - use all columns
                        symbol_df = df1.loc[common_start:common_end]
                    
                    # Create OHLCV dict from DataFrame
                    symbol_data = {}
                    for col in symbol_df.columns:
                        col_name = str(col).title()  # Capitalize first letter
                        if col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            symbol_data[col_name] = symbol_df[col]
                else:
                    logger.warning(f"   ⚠️ Cannot extract {symbol} from data1: no valid access method")
                    continue
                
                # Remove None values and create DataFrame
                symbol_data = {k: v for k, v in symbol_data.items() if v is not None}
                if symbol_data:
                    symbol_dict[symbol] = pd.DataFrame(symbol_data)
                    logger.debug(f"   Added {symbol} from data1: {list(symbol_data.keys())}")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Error extracting {symbol} from data1: {e}")
        
        # Process data2 symbols
        for symbol in data2.symbols:
            try:
                # Extract OHLCV data for this symbol from data2
                if hasattr(data2, 'open') and data2.open is not None:
                    # Direct attribute access (preferred method)
                    symbol_data = {
                        'Open': data2.open[symbol].loc[common_start:common_end] if len(data2.symbols) > 1 else data2.open.loc[common_start:common_end],
                        'High': data2.high[symbol].loc[common_start:common_end] if len(data2.symbols) > 1 else data2.high.loc[common_start:common_end],
                        'Low': data2.low[symbol].loc[common_start:common_end] if len(data2.symbols) > 1 else data2.low.loc[common_start:common_end],
                        'Close': data2.close[symbol].loc[common_start:common_end] if len(data2.symbols) > 1 else data2.close.loc[common_start:common_end],
                        'Volume': data2.volume[symbol].loc[common_start:common_end] if len(data2.symbols) > 1 and hasattr(data2, 'volume') else data2.volume.loc[common_start:common_end] if hasattr(data2, 'volume') else None
                    }
                elif df2 is not None and hasattr(df2, 'index'):
                    # Fallback to DataFrame extraction (only if get() worked)
                    if len(data2.symbols) > 1:
                        # MultiIndex case - extract by symbol
                        symbol_df = df2.xs(symbol, axis=1, level=0).loc[common_start:common_end]
                    else:
                        # Single symbol case - use all columns
                        symbol_df = df2.loc[common_start:common_end]
                    
                    # Create OHLCV dict from DataFrame
                    symbol_data = {}
                    for col in symbol_df.columns:
                        col_name = str(col).title()  # Capitalize first letter
                        if col_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            symbol_data[col_name] = symbol_df[col]
                else:
                    logger.warning(f"   ⚠️ Cannot extract {symbol} from data2: no valid access method")
                    continue
                
                # Remove None values and create DataFrame
                symbol_data = {k: v for k, v in symbol_data.items() if v is not None}
                if symbol_data:
                    symbol_dict[symbol] = pd.DataFrame(symbol_data)
                    logger.debug(f"   Added {symbol} from data2: {list(symbol_data.keys())}")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Error extracting {symbol} from data2: {e}")
        
        if not symbol_dict:
            logger.error(f"No valid symbol data extracted for combination")
            return None
        
        # Create combined VBT Data object using symbol dictionary
        logger.debug(f"Creating combined VBT Data from {len(symbol_dict)} symbols")
        combined_data = vbt.Data.from_data(symbol_dict)
        
        if combined_data is None:
            logger.error(f"Failed to create combined VBT Data")
            return None
        
        logger.debug(f"Successfully combined data: {len(combined_data.symbols)} symbols")
        logger.debug(f"   Final shape: {combined_data.wrapper.shape}")
        
        return combined_data
        
    except Exception as e:
        logger.error(f"Error combining VBT data: {e}")
        import traceback
        logger.debug(f"Combine traceback: {traceback.format_exc()}")
        return None 