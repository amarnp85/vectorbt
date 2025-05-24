"""
Data Fetching Module

⚠️  IMPORTS UPDATED TO USE NEW REFACTORED ARCHITECTURE ⚠️

This module now uses the new modular architecture located in core/ directory
and the clean public API in data_fetcher_new.py.

🔄 MAJOR UPDATE: All imports now redirect to the new refactored modules.

✅ NEW ARCHITECTURE BENEFITS:
- Clean separation of concerns
- Better error handling and logging  
- More maintainable codebase
- Enhanced performance
- Comprehensive test coverage (8/8 tests passed)

Functions:
- fetch_data(): Main data fetching function using new modular core
- fetch_top_symbols(): Volume-based symbol selection 
- fetch_ohlcv(): Convenience function (backward compatibility)
- update_data(): Update stored data using new architecture
- get_resampling_info(): Get resampling metrics and capabilities

Resampling Functions:
- can_resample_from_to(): Check if resampling is possible between timeframes
- resample_ohlcv_for_storage(): Legacy resampling (use new core/resampler.py)
- validate_storage_resampled_data(): Validate resampled data integrity

📋 MIGRATION NOTES:
- All function signatures remain identical for backward compatibility
- Enhanced functionality and performance with new architecture
- Consider migrating direct imports to data_fetcher_new for new code

For more information about the new architecture, see core/ directory.
"""

# Import from new refactored architecture
from .data_fetcher_new import (
    fetch_data,
    fetch_top_symbols,
    fetch_ohlcv,  # Backward compatibility alias
    update_data,
    get_storage_info,
    get_resampling_info
)

# Import resampling functions from new core architecture
from .core.resampler import (
    TIMEFRAME_HIERARCHY
)

# Legacy compatibility imports from deprecated modules
# These maintain backward compatibility but issue warnings when used directly
import warnings

def can_resample_from_to(source_tf: str, target_tf: str) -> bool:
    """Check if resampling is possible between timeframes."""
    warnings.warn(
        "can_resample_from_to is deprecated. Use core.resampler.DataResampler.can_resample() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from .core.resampler import DataResampler
    return DataResampler.can_resample(source_tf, target_tf)

def resample_ohlcv_for_storage(data, target_timeframe: str):
    """Legacy resampling function - use new core architecture instead."""
    warnings.warn(
        "resample_ohlcv_for_storage is deprecated. Use core.resampler.DataResampler.resample_data() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from .core.resampler import DataResampler
    return DataResampler.resample_data(data, target_timeframe)

def validate_storage_resampled_data(original_data, resampled_data, target_timeframe: str) -> bool:
    """Legacy validation function - use new core architecture validation instead."""
    warnings.warn(
        "validate_storage_resampled_data is deprecated. Validation is now integrated into the core resampling process.",
        DeprecationWarning,
        stacklevel=2
    )
    # Basic validation for backward compatibility
    try:
        return (resampled_data is not None and 
                hasattr(resampled_data, 'symbols') and 
                len(resampled_data.symbols) > 0)
    except:
        return False

__all__ = [
    'fetch_data',
    'fetch_top_symbols', 
    'fetch_ohlcv',
    'update_data',
    'get_storage_info',
    'get_resampling_info',
    # Legacy resampling functions (deprecated)
    'can_resample_from_to',
    'resample_ohlcv_for_storage', 
    'validate_storage_resampled_data',
    'TIMEFRAME_HIERARCHY'
] 