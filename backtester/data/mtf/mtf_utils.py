"""
Multi-Timeframe Utility Functions

Provides utility functions for MTF operations including timeframe
validation, hierarchy management, and safe resampling.
"""

import logging
from typing import List, Dict, Optional, Union, Tuple
import pandas as pd
import vectorbtpro as vbt

logger = logging.getLogger(__name__)


def get_timeframe_hierarchy(timeframes: List[str]) -> List[str]:
    """
    Sort timeframes from lowest to highest.
    
    Args:
        timeframes: List of timeframe strings
        
    Returns:
        Sorted list of timeframes
    """
    # Define timeframe order
    tf_order = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '3d': 4320,
        '1w': 10080,
        '1M': 43200,  # Approximate
    }
    
    # Sort by minutes
    def get_minutes(tf: str) -> int:
        if tf in tf_order:
            return tf_order[tf]
        
        # Try to parse custom timeframes
        try:
            if tf.endswith('m'):
                return int(tf[:-1])
            elif tf.endswith('h'):
                return int(tf[:-1]) * 60
            elif tf.endswith('d'):
                return int(tf[:-1]) * 1440
            elif tf.endswith('w'):
                return int(tf[:-1]) * 10080
            else:
                logger.warning(f"Unknown timeframe format: {tf}")
                return 9999999  # Put unknown at end
        except:
            return 9999999
    
    return sorted(timeframes, key=get_minutes)


def validate_timeframe_compatibility(
    timeframes: List[str],
    base_timeframe: str
) -> bool:
    """
    Check if timeframes are compatible for alignment.
    
    Args:
        timeframes: List of timeframes to check
        base_timeframe: Base timeframe for alignment
        
    Returns:
        True if all timeframes are compatible
    """
    # Convert to minutes for comparison
    base_minutes = _timeframe_to_minutes(base_timeframe)
    
    for tf in timeframes:
        tf_minutes = _timeframe_to_minutes(tf)
        
        # Check if higher timeframe is divisible by base
        if tf_minutes > base_minutes:
            if tf_minutes % base_minutes != 0:
                logger.warning(
                    f"Timeframe {tf} ({tf_minutes}m) not evenly divisible "
                    f"by base {base_timeframe} ({base_minutes}m)"
                )
                return False
        
        # Check if base is divisible by lower timeframe
        elif tf_minutes < base_minutes:
            if base_minutes % tf_minutes != 0:
                logger.warning(
                    f"Base timeframe {base_timeframe} ({base_minutes}m) not evenly "
                    f"divisible by {tf} ({tf_minutes}m)"
                )
                return False
    
    return True


def _timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes."""
    mappings = {
        '1m': 1,
        '5m': 5,
        '15m': 15,
        '30m': 30,
        '1h': 60,
        '2h': 120,
        '4h': 240,
        '6h': 360,
        '8h': 480,
        '12h': 720,
        '1d': 1440,
        '3d': 4320,
        '1w': 10080,
    }
    
    if timeframe in mappings:
        return mappings[timeframe]
    
    # Try to parse
    try:
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 1440
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]) * 10080
    except:
        pass
    
    logger.warning(f"Unknown timeframe: {timeframe}, defaulting to 60m")
    return 60


def align_timeframes(
    data_dict: Dict[str, Union[pd.Series, pd.DataFrame]],
    target_timeframe: str,
    method: str = 'auto'
) -> Dict[str, Union[pd.Series, pd.DataFrame]]:
    """
    Align multiple timeframe data to a target timeframe.
    
    Args:
        data_dict: Dictionary of timeframe to data
        target_timeframe: Target timeframe to align to
        method: Alignment method ('auto', 'opening', 'closing')
        
    Returns:
        Dictionary of aligned data
    """
    if target_timeframe not in data_dict:
        raise ValueError(f"Target timeframe {target_timeframe} not in data")
    
    target_data = data_dict[target_timeframe]
    target_index = target_data.index if hasattr(target_data, 'index') else None
    
    if target_index is None:
        raise ValueError("Target data must have an index")
    
    aligned_data = {target_timeframe: target_data}
    
    for tf, data in data_dict.items():
        if tf == target_timeframe:
            continue
        
        # Determine alignment method
        if method == 'auto':
            # Use closing for most data types
            align_method = 'closing'
        else:
            align_method = method
        
        # Align the data
        try:
            if hasattr(data, 'vbt'):
                if align_method == 'opening':
                    aligned = data.vbt.realign_opening(target_index)
                else:
                    aligned = data.vbt.realign_closing(target_index)
            else:
                # Manual alignment
                aligned = data.reindex(target_index, method='ffill')
                if align_method == 'closing':
                    # Shift to avoid look-ahead
                    aligned = aligned.shift(1)
            
            aligned_data[tf] = aligned
            
        except Exception as e:
            logger.error(f"Failed to align {tf} to {target_timeframe}: {e}")
            # Keep original data if alignment fails
            aligned_data[tf] = data
    
    return aligned_data


def resample_safe(
    data: Union[pd.Series, pd.DataFrame, vbt.Data],
    source_freq: str,
    target_freq: str,
    agg_func: str = 'last'
) -> Union[pd.Series, pd.DataFrame, vbt.Data]:
    """
    Safely resample data from source to target frequency.
    
    Args:
        data: Data to resample
        source_freq: Source frequency
        target_freq: Target frequency
        agg_func: Aggregation function ('last', 'mean', 'sum', etc.)
        
    Returns:
        Resampled data
    """
    # Check if upsampling or downsampling
    source_minutes = _timeframe_to_minutes(source_freq)
    target_minutes = _timeframe_to_minutes(target_freq)
    
    if isinstance(data, vbt.Data):
        # Use VBT's resample method
        return data.resample(target_freq)
    
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        if target_minutes < source_minutes:
            # Upsampling - use forward fill
            resampled = data.resample(target_freq).ffill()
        else:
            # Downsampling - use specified aggregation
            resampler = data.resample(target_freq)
            
            if agg_func == 'last':
                resampled = resampler.last()
            elif agg_func == 'first':
                resampled = resampler.first()
            elif agg_func == 'mean':
                resampled = resampler.mean()
            elif agg_func == 'sum':
                resampled = resampler.sum()
            elif agg_func == 'ohlc':
                resampled = resampler.ohlc()
            else:
                resampled = resampler.last()
        
        return resampled
    
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def create_mtf_labels(
    timeframes: List[str],
    base_label: str = ""
) -> Dict[str, str]:
    """
    Create labels for multi-timeframe features.
    
    Args:
        timeframes: List of timeframes
        base_label: Base label to prepend
        
    Returns:
        Dictionary of timeframe to label
    """
    labels = {}
    
    for tf in timeframes:
        if base_label:
            labels[tf] = f"{base_label}_{tf}"
        else:
            labels[tf] = tf
    
    return labels


def get_common_index(
    data_dict: Dict[str, Union[pd.Series, pd.DataFrame]]
) -> pd.DatetimeIndex:
    """
    Get the common index across all data.
    
    Args:
        data_dict: Dictionary of data with indexes
        
    Returns:
        Common datetime index
    """
    indexes = []
    
    for data in data_dict.values():
        if hasattr(data, 'index'):
            indexes.append(set(data.index))
    
    if not indexes:
        raise ValueError("No data with index found")
    
    # Find intersection of all indexes
    common_idx = indexes[0]
    for idx in indexes[1:]:
        common_idx = common_idx.intersection(idx)
    
    # Convert back to sorted index
    common_idx = sorted(list(common_idx))
    
    return pd.DatetimeIndex(common_idx)


def calculate_timeframe_ratios(timeframes: List[str]) -> Dict[str, float]:
    """
    Calculate ratios between timeframes.
    
    Args:
        timeframes: List of timeframes
        
    Returns:
        Dictionary of timeframe pairs to ratios
    """
    ratios = {}
    sorted_tfs = get_timeframe_hierarchy(timeframes)
    
    for i, tf1 in enumerate(sorted_tfs):
        for tf2 in sorted_tfs[i+1:]:
            minutes1 = _timeframe_to_minutes(tf1)
            minutes2 = _timeframe_to_minutes(tf2)
            
            ratio = minutes2 / minutes1
            ratios[f"{tf1}_to_{tf2}"] = ratio
            ratios[f"{tf2}_to_{tf1}"] = 1 / ratio
    
    return ratios 