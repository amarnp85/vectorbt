#!/usr/bin/env python3
"""Storage-Focused Resampling Module

Implements efficient OHLCV resampling for data storage and caching purposes.
Uses pandas resampling for lightweight, storage-optimized timeframe conversion.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import vectorbtpro as vbt
from ..storage.data_storage import data_storage

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
    Resample OHLCV data for storage purposes using pandas resampling.
    
    This approach:
    - Uses pandas for efficient OHLCV aggregation
    - Creates lightweight data suitable for storage
    - Avoids VBT's MTF alignment overhead
    
    Args:
        data: VBT Data object to resample
        target_timeframe: Target timeframe string
        
    Returns:
        Resampled VBT Data object optimized for storage
    """
    try:
        logger.info(f"🔄 Storage resampling to {target_timeframe}")
        
        # Get the raw DataFrame from VBT Data - use direct access to get OHLCV data
        if hasattr(data, 'symbols') and data.symbols:
            # For multi-symbol data, concatenate all symbols
            symbol_dfs = []
            for symbol in data.symbols:
                if symbol in data.data:
                    symbol_df = data.data[symbol].copy()
                    # Add symbol as column level if needed
                    if len(data.symbols) > 1:
                        symbol_df.columns = pd.MultiIndex.from_product([[symbol], symbol_df.columns])
                    symbol_dfs.append(symbol_df)
            
            if symbol_dfs:
                df = pd.concat(symbol_dfs, axis=1)
            else:
                logger.error("No symbol data found in VBT Data object")
                return None
        else:
            logger.error("VBT Data object has no symbols")
            return None
        
        if df is None or df.empty:
            logger.error("No data to resample")
            return None
        
        # Handle MultiIndex columns for multi-symbol data
        if isinstance(df.columns, pd.MultiIndex):
            # For MultiIndex columns, we need to create aggregation rules per symbol
            final_agg_rules = {}
            for symbol, field in df.columns:
                if field == 'Open':
                    final_agg_rules[(symbol, field)] = 'first'
                elif field == 'High':
                    final_agg_rules[(symbol, field)] = 'max'
                elif field == 'Low':
                    final_agg_rules[(symbol, field)] = 'min'
                elif field == 'Close':
                    final_agg_rules[(symbol, field)] = 'last'
                elif field == 'Volume':
                    final_agg_rules[(symbol, field)] = 'sum'
                elif 'volume' in field.lower() or 'quote' in field.lower():
                    final_agg_rules[(symbol, field)] = 'sum'
                elif 'count' in field.lower() or 'trade' in field.lower():
                    final_agg_rules[(symbol, field)] = 'sum'
                else:
                    final_agg_rules[(symbol, field)] = 'last'  # Default
        else:
            # For simple columns (single symbol)
            agg_rules = {
                'Open': 'first',
                'High': 'max', 
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }
            
            # Handle additional columns that might exist
            extra_columns = {}
            for col in df.columns:
                if col not in agg_rules:
                    if 'volume' in col.lower() or 'quote' in col.lower():
                        extra_columns[col] = 'sum'
                    elif 'count' in col.lower() or 'trade' in col.lower():
                        extra_columns[col] = 'sum'
                    elif 'time' in col.lower():
                        extra_columns[col] = 'last'
                    else:
                        extra_columns[col] = 'last'  # Default
            
            # Combine all aggregation rules
            all_agg_rules = {**agg_rules, **extra_columns}
            
            # Filter to only include columns that exist in the data
            final_agg_rules = {col: rule for col, rule in all_agg_rules.items() if col in df.columns}
        
        # Perform pandas resampling
        resampled_df = df.resample(target_timeframe).agg(final_agg_rules)
        
        # Remove any NaN rows that might result from resampling
        resampled_df = resampled_df.dropna()
        
        if resampled_df.empty:
            logger.warning("Resampling resulted in empty DataFrame")
            return None
        
        logger.info(f"✅ Storage resampling successful: {len(resampled_df)} rows")
        logger.info(f"   Original: {len(df)} rows → Resampled: {len(resampled_df)} rows")
        
        # Convert back to VBT Data object
        # Reconstruct the original VBT Data structure
        symbols = data.symbols if hasattr(data, 'symbols') and data.symbols else ['Unknown']
        
        # Create data dictionary matching VBT's structure
        data_dict = {}
        
        if isinstance(resampled_df.columns, pd.MultiIndex):
            # Multi-symbol case: extract each symbol's data
            for symbol in symbols:
                if symbol in resampled_df.columns.get_level_values(0):
                    symbol_data = resampled_df[symbol]
                    data_dict[symbol] = symbol_data
        else:
            # Single symbol case: use the entire DataFrame
            symbol = symbols[0]
            data_dict[symbol] = resampled_df
        
        # Create VBT Data object using from_data method
        if len(data_dict) == 1:
            # Single symbol: use the DataFrame directly but preserve symbol name
            symbol = list(data_dict.keys())[0] 
            df_for_vbt = data_dict[symbol]
            resampled_data = vbt.Data.from_data({symbol: df_for_vbt}, columns_are_symbols=True)
        else:
            # Multi-symbol: use the full data_dict
            resampled_data = vbt.Data.from_data(data_dict, columns_are_symbols=True)
        
        return resampled_data
        
    except Exception as e:
        logger.error(f"❌ Error in storage resampling: {e}")
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
    market_type: str = 'spot'
) -> Optional[Tuple[str, vbt.Data]]:
    """
    Find the best available lower timeframe data for storage resampling.
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
            
            # Load data for this timeframe
            source_data = data_storage.load_data(
                exchange_id, source_tf, symbols, market_type
            )
            
            if source_data is not None:
                # Check if all requested symbols are available
                available_symbols = set(source_data.symbols)
                required_symbols = set(symbols)
                
                if required_symbols.issubset(available_symbols):
                    logger.info(f"✅ Found complete data in {source_tf} for storage resampling")
                    return source_tf, source_data
                else:
                    missing = required_symbols - available_symbols
                    logger.debug(f"Missing symbols in {source_tf}: {missing}")
        
        logger.debug(f"No suitable lower timeframe found for resampling")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error finding resampling source: {e}")
        return None

def fetch_with_storage_resampling_fallback(
    symbols: List[str],
    exchange_id: str,
    timeframe: str,
    start_date: Optional[Union[str, vbt.timestamp]] = None,
    end_date: Optional[Union[str, vbt.timestamp]] = None,
    market_type: str = 'spot',
    prefer_resampling: bool = True
) -> Optional[vbt.Data]:
    """
    Fetch data with storage-optimized resampling fallback.
    
    Strategy:
    1. Check for exact timeframe data first
    2. If not available, try efficient storage resampling from lower timeframes
    3. Cache the resampled result for future use
    4. Fall back to API fetch if resampling not possible
    """
    try:
        logger.info(f"🎯 Fetch with storage resampling fallback: {timeframe}")
        
        # First, try to load exact timeframe data
        exact_data = data_storage.load_data(exchange_id, timeframe, symbols, market_type)
        if exact_data is not None:
            # Check if all symbols are available
            available_symbols = set(exact_data.symbols)
            required_symbols = set(symbols)
            
            if required_symbols.issubset(available_symbols):
                logger.info(f"✅ Found exact timeframe data: {timeframe}")
                if len(symbols) == len(exact_data.symbols):
                    return exact_data
                else:
                    try:
                        return exact_data.select_symbols(symbols)
                    except:
                        logger.warning(f"Could not select symbols {symbols}, returning all data")
                        return exact_data
        
        # If prefer_resampling and we can resample, try that
        if prefer_resampling:
            resampling_result = get_best_resampling_source(
                timeframe, symbols, exchange_id, market_type
            )
            
            if resampling_result is not None:
                source_tf, source_data = resampling_result
                
                logger.info(f"🔄 Storage resampling from {source_tf} to {timeframe}")
                
                # Use storage-optimized resampling
                resampled_data = resample_ohlcv_for_storage(source_data, timeframe)
                
                if resampled_data is not None:
                    # Cache the resampled data for future use
                    success = data_storage.save_data(
                        resampled_data, exchange_id, timeframe, market_type
                    )
                    
                    if success:
                        logger.info(f"💾 Cached storage-resampled {timeframe} data")
                    
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
        logger.info(f"📡 No storage resampling available for {timeframe}, will use API fetch")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error in storage resampling fallback: {e}")
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
        logger.debug(f"🔍 Validating storage-resampled data for {target_timeframe}")
        
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
        
        logger.debug(f"✅ Storage resampled data validation passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error validating storage resampled data: {e}")
        return False 