"""
Compatibility Layer for Migration to Simple Indicators

This module provides a temporary compatibility layer to help migrate from the
old complex indicator system to the new simple direct VectorBTPro usage.

This layer will be removed once all strategies have been migrated.
"""

import warnings
import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime

from .simple_indicators import (
    sma, ema, rsi, atr, bollinger_bands, macd, 
    stochastic, adx, obv, vwap, supertrend,
    calculate_multiple, optimize_indicator, calculate_standard_indicators
)


@dataclass
class CompatibilityResult:
    """Compatibility wrapper for indicator results."""
    data: Union[pd.Series, pd.DataFrame, Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None
    calculation_time: Optional[float] = None
    
    def __getitem__(self, key):
        """Support dictionary-style access."""
        if isinstance(self.data, dict):
            return self.data[key]
        return self.data
    
    def __getattr__(self, name):
        """Support attribute access for backward compatibility."""
        # Common legacy attributes
        if name in ['ma', 'sma', 'ema']:
            return self.data
        elif name == 'real':
            return self.data
        elif name in ['macd', 'macdsignal', 'macdhist']:
            if isinstance(self.data, dict) and name in self.data:
                return self.data[name]
        elif name in ['upperband', 'middleband', 'lowerband']:
            if isinstance(self.data, dict):
                mapping = {
                    'upperband': 'upper',
                    'middleband': 'middle', 
                    'lowerband': 'lower'
                }
                return self.data.get(mapping[name])
        elif name in ['slowk', 'slowd']:
            if isinstance(self.data, dict):
                mapping = {'slowk': 'k', 'slowd': 'd'}
                return self.data.get(mapping[name])
        
        # If not found, try to get from data
        if hasattr(self.data, name):
            return getattr(self.data, name)
        
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class CompatibilityManager:
    """
    Compatibility manager that mimics the old IndicatorManager interface
    but uses the new simple indicators internally.
    """
    
    def __init__(self, **kwargs):
        """Initialize compatibility manager."""
        warnings.warn(
            "Using compatibility layer for indicators. Please migrate to direct "
            "simple_indicators usage for better performance.",
            DeprecationWarning,
            stacklevel=2
        )
        self._cache = {}
        self._stats = {
            'calculations': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
    
    def calculate_indicator(
        self,
        indicator_name: str,
        data: Any,
        **params
    ) -> CompatibilityResult:
        """
        Calculate indicator using compatibility interface.
        
        Maps old indicator names and parameters to new simple indicators.
        """
        start_time = datetime.now()
        
        # Normalize indicator name
        indicator_name = indicator_name.lower()
        
        # Map old parameter names to new ones
        params = self._map_parameters(indicator_name, params)
        
        # Calculate using simple indicators
        try:
            if indicator_name in ['sma', 'ma']:
                result = sma(self._extract_close(data), **params)
            elif indicator_name == 'ema':
                result = ema(self._extract_close(data), **params)
            elif indicator_name == 'rsi':
                result = rsi(self._extract_close(data), **params)
            elif indicator_name == 'atr':
                result = atr(
                    self._extract_high(data),
                    self._extract_low(data),
                    self._extract_close(data),
                    **params
                )
            elif indicator_name in ['bbands', 'bollinger_bands']:
                upper, middle, lower = bollinger_bands(self._extract_close(data), **params)
                result = {
                    'upper': upper,
                    'middle': middle,
                    'lower': lower,
                    'upperband': upper,  # Legacy names
                    'middleband': middle,
                    'lowerband': lower
                }
            elif indicator_name == 'macd':
                macd_line, signal, hist = macd(self._extract_close(data), **params)
                result = {
                    'macd': macd_line,
                    'signal': signal,
                    'histogram': hist,
                    'macdsignal': signal  # Legacy name
                }
            elif indicator_name in ['stoch', 'stochastic']:
                k, d = stochastic(
                    self._extract_high(data),
                    self._extract_low(data),
                    self._extract_close(data),
                    **params
                )
                result = {
                    'k': k,
                    'd': d,
                    'slowk': k,  # Legacy names
                    'slowd': d
                }
            elif indicator_name == 'adx':
                result = adx(
                    self._extract_high(data),
                    self._extract_low(data),
                    self._extract_close(data),
                    **params
                )
            elif indicator_name == 'obv':
                result = obv(
                    self._extract_close(data),
                    self._extract_volume(data)
                )
            elif indicator_name == 'supertrend':
                result = supertrend(
                    self._extract_high(data),
                    self._extract_low(data),
                    self._extract_close(data),
                    **params
                )
            else:
                raise ValueError(f"Unknown indicator: {indicator_name}")
            
            calc_time = (datetime.now() - start_time).total_seconds()
            self._stats['calculations'] += 1
            
            return CompatibilityResult(
                data=result,
                metadata={'name': indicator_name, 'type': 'compatibility'},
                parameters=params,
                calculation_time=calc_time
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to calculate {indicator_name}: {str(e)}") from e
    
    def calculate_multiple_indicators(
        self,
        indicator_configs: List[Dict[str, Any]],
        data: Any,
        parallel: bool = True
    ) -> Dict[str, CompatibilityResult]:
        """Calculate multiple indicators."""
        results = {}
        
        for config in indicator_configs:
            name = config['name']
            params = config.get('params', {})
            
            # Generate unique key for result
            key = f"{name}_{hash(str(params))}"
            
            results[key] = self.calculate_indicator(name, data, **params)
        
        return results
    
    def _map_parameters(self, indicator_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Map old parameter names to new ones."""
        mapped = params.copy()
        
        # Common parameter mappings
        if 'timeperiod' in params:
            mapped['window'] = params['timeperiod']
            del mapped['timeperiod']
        
        if 'period' in params and 'window' not in mapped:
            mapped['window'] = params['period']
            del mapped['period']
        
        if 'nbdevup' in params or 'nbdevdn' in params:
            mapped['std_dev'] = params.get('nbdevup', params.get('nbdevdn', 2.0))
            for key in ['nbdevup', 'nbdevdn']:
                if key in mapped:
                    del mapped[key]
        
        if indicator_name == 'macd':
            if 'fastperiod' in params:
                mapped['fast_window'] = params['fastperiod']
                del mapped['fastperiod']
            if 'slowperiod' in params:
                mapped['slow_window'] = params['slowperiod']
                del mapped['slowperiod']
            if 'signalperiod' in params:
                mapped['signal_window'] = params['signalperiod']
                del mapped['signalperiod']
        
        if indicator_name in ['stoch', 'stochastic']:
            if 'fastk_period' in params:
                mapped['k_period'] = params['fastk_period']
                del mapped['fastk_period']
            if 'slowk_period' in params or 'slowd_period' in params:
                mapped['d_period'] = params.get('slowk_period', params.get('slowd_period', 3))
                for key in ['slowk_period', 'slowd_period']:
                    if key in mapped:
                        del mapped[key]
        
        return mapped
    
    def _extract_close(self, data):
        """Extract close prices from various data formats."""
        if hasattr(data, 'close'):
            return data.close
        elif isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            return data['Close']
        elif isinstance(data, pd.DataFrame) and 'close' in data.columns:
            return data['close']
        elif isinstance(data, pd.Series):
            return data
        else:
            raise ValueError("Cannot extract close prices from data")
    
    def _extract_high(self, data):
        """Extract high prices from various data formats."""
        if hasattr(data, 'high'):
            return data.high
        elif isinstance(data, pd.DataFrame) and 'High' in data.columns:
            return data['High']
        elif isinstance(data, pd.DataFrame) and 'high' in data.columns:
            return data['high']
        else:
            # Fallback to close prices
            return self._extract_close(data)
    
    def _extract_low(self, data):
        """Extract low prices from various data formats."""
        if hasattr(data, 'low'):
            return data.low
        elif isinstance(data, pd.DataFrame) and 'Low' in data.columns:
            return data['Low']
        elif isinstance(data, pd.DataFrame) and 'low' in data.columns:
            return data['low']
        else:
            # Fallback to close prices
            return self._extract_close(data)
    
    def _extract_volume(self, data):
        """Extract volume from various data formats."""
        if hasattr(data, 'volume'):
            return data.volume
        elif isinstance(data, pd.DataFrame) and 'Volume' in data.columns:
            return data['Volume']
        elif isinstance(data, pd.DataFrame) and 'volume' in data.columns:
            return data['volume']
        else:
            # Return dummy volume if not available
            close = self._extract_close(data)
            return pd.Series(1, index=close.index)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for compatibility."""
        total = self._stats['cache_hits'] + self._stats['cache_misses']
        hit_rate = self._stats['cache_hits'] / max(total, 1)
        
        return {
            'size': len(self._cache),
            'hits': self._stats['cache_hits'],
            'misses': self._stats['cache_misses'],
            'hit_rate': hit_rate,
            'calculations': self._stats['calculations']
        }
    
    def clear_cache(self):
        """Clear cache."""
        self._cache.clear()
        self._stats['cache_hits'] = 0
        self._stats['cache_misses'] = 0


# Global compatibility manager instance
_compat_manager = None


def get_indicator_manager(**kwargs):
    """
    Get compatibility manager instance.
    
    This function mimics the old get_indicator_manager but returns
    a compatibility wrapper that uses simple indicators.
    """
    global _compat_manager
    if _compat_manager is None:
        _compat_manager = CompatibilityManager(**kwargs)
    return _compat_manager


def reset_indicator_manager():
    """Reset the compatibility manager."""
    global _compat_manager
    _compat_manager = None


# Legacy function mappings for backward compatibility
def calculate_sma(data, window=20, **kwargs):
    """Legacy function for SMA calculation."""
    warnings.warn(
        "calculate_sma is deprecated. Use simple_indicators.sma directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return CompatibilityResult(data=sma(data, window, **kwargs))


def calculate_rsi(data, window=14, **kwargs):
    """Legacy function for RSI calculation."""
    warnings.warn(
        "calculate_rsi is deprecated. Use simple_indicators.rsi directly.",
        DeprecationWarning,
        stacklevel=2
    )
    return CompatibilityResult(data=rsi(data, window, **kwargs))


def calculate_atr(data, window=14, **kwargs):
    """Legacy function for ATR calculation."""
    warnings.warn(
        "calculate_atr is deprecated. Use simple_indicators.atr directly.",
        DeprecationWarning,
        stacklevel=2
    )
    if hasattr(data, 'high'):
        return CompatibilityResult(
            data=atr(data.high, data.low, data.close, window, **kwargs)
        )
    else:
        raise ValueError("ATR requires high, low, and close prices")


def calculate_bollinger_bands(data, window=20, std_dev=2.0, **kwargs):
    """Legacy function for Bollinger Bands calculation."""
    warnings.warn(
        "calculate_bollinger_bands is deprecated. Use simple_indicators.bollinger_bands directly.",
        DeprecationWarning,
        stacklevel=2
    )
    upper, middle, lower = bollinger_bands(data, window, std_dev, **kwargs)
    return CompatibilityResult(
        data={'upper': upper, 'middle': middle, 'lower': lower}
    )


def calculate_macd(data, fast_window=12, slow_window=26, signal_window=9, **kwargs):
    """Legacy function for MACD calculation."""
    warnings.warn(
        "calculate_macd is deprecated. Use simple_indicators.macd directly.",
        DeprecationWarning,
        stacklevel=2
    )
    macd_line, signal, hist = macd(data, fast_window, slow_window, signal_window, **kwargs)
    return CompatibilityResult(
        data={'macd': macd_line, 'signal': signal, 'histogram': hist}
    ) 