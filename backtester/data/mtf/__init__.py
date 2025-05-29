"""
Multi-Timeframe (MTF) Data Module

This module provides utilities for handling multi-timeframe data analysis,
including proper alignment to avoid look-ahead bias and efficient resampling.

Key Features:
- Safe data alignment using vectorbtpro's realign methods
- Multi-timeframe data synchronization
- Look-ahead bias prevention
- Efficient caching and resampling
"""

from .mtf_data_handler import MTFDataHandler
from .mtf_alignment import MTFAlignmentEngine
from .mtf_utils import (
    align_timeframes,
    resample_safe,
    get_timeframe_hierarchy,
    validate_timeframe_compatibility
)

__all__ = [
    'MTFDataHandler',
    'MTFAlignmentEngine',
    'align_timeframes',
    'resample_safe',
    'get_timeframe_hierarchy',
    'validate_timeframe_compatibility'
] 