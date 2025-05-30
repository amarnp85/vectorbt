"""
Advanced Multi-Symbol Strategy with Dynamic Features

This strategy extends the basic multi-symbol approach with:
1. Symbol-specific parameter optimization
2. Market regime detection
3. Correlation-based position sizing
4. Volume and volatility filters
5. Cross-symbol momentum confirmation
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .multi_symbol_strategy_base import MultiSymbolStrategy
from ..indicators.simple_indicators import sma, atr, rsi
from ..utilities.structured_logging import get_logger

logger = get_logger(__name__)


@dataclass
class SymbolParameters:
    """Parameters optimized per symbol."""
    fast_period: int
    slow_period: int
    atr_multiplier_sl: float
    atr_multiplier_tp: float
    rsi_threshold_long: float
    rsi_threshold_short: float
    volume_factor: float
    min_trend_strength: float


class AdvancedMultiSymbolStrategy(MultiSymbolStrategy):
    """
    Advanced multi-symbol strategy with dynamic features.
    
    Key Features:
    1. Symbol-specific parameters for better performance
    2. Market regime detection to adapt to conditions
    3. Correlation-based position sizing to manage risk
    4. Volume and volatility filters for quality signals
    5. Cross-symbol momentum for market confirmation
    """
    
    # Default parameters for each symbol type
    CRYPTO_MAJOR_PARAMS = SymbolParameters(
        fast_period=20,
        slow_period=50,
        atr_multiplier_sl=2.0,
        atr_multiplier_tp=3.0,
        rsi_threshold_long=30,
        rsi_threshold_short=70,
        volume_factor=1.2,
        min_trend_strength=0.005  # Reduced from 0.02
    )
    
    CRYPTO_ALT_PARAMS = SymbolParameters(
        fast_period=15,
        slow_period=40,
        atr_multiplier_sl=2.5,
        atr_multiplier_tp=4.0,
        rsi_threshold_long=35,
        rsi_threshold_short=65,
        volume_factor=1.5,
        min_trend_strength=0.01  # Reduced from 0.03
    )
    
    def __init__(self, data: vbt.Data, params: Optional[Dict[str, Any]] = None):
        """Initialize with symbol-specific parameters."""
        default_params = {
            'use_market_regime': True,
            'use_correlation_sizing': True,
            'use_volume_filter': True,
            'use_volatility_filter': True,
            'max_correlation': 0.8,
            'correlation_lookback': 60,
            'market_regime_lookback': 50,
            'volatility_lookback': 20,
            'min_volatility_percentile': 20,
            'max_volatility_percentile': 80,
            'position_reduction_factor': 0.5,
            'use_symbol_ranking': False,  # Disable symbol ranking for now
            'max_active_symbols': None  # Trade all symbols
        }
        
        if params is None:
            params = {}
        
        merged_params = {**default_params, **params}
        super().__init__(data, merged_params)
        
        # Initialize symbol-specific parameters
        self._init_symbol_parameters()
        
    def _init_symbol_parameters(self):
        """Initialize parameters for each symbol."""
        self.symbol_params = {}
        
        for symbol in self.data.symbols:
            # Classify symbol type (simplified - in practice use more sophisticated classification)
            if symbol in ['BTC/USDT', 'ETH/USDT']:
                self.symbol_params[symbol] = self.CRYPTO_MAJOR_PARAMS
            else:
                self.symbol_params[symbol] = self.CRYPTO_ALT_PARAMS
                
            # Override with any symbol-specific params passed in
            if f"{symbol}_params" in self.params:
                symbol_specific = self.params[f"{symbol}_params"]
                for key, value in symbol_specific.items():
                    setattr(self.symbol_params[symbol], key, value)
    
    def _calculate_custom_indicators(self) -> Dict[str, Any]:
        """Calculate strategy-specific indicators."""
        indicators = {}
        close = self.data.close
        high = self.data.high
        low = self.data.low
        volume = self.data.volume if hasattr(self.data, 'volume') else None
        
        # Symbol-specific MAs with custom periods
        fast_ma_dict = {}
        slow_ma_dict = {}
        atr_dict = {}
        rsi_dict = {}
        volume_ma_dict = {}
        
        for symbol in self.data.symbols:
            params = self.symbol_params[symbol]
            
            # Moving averages with symbol-specific periods
            fast_ma_dict[symbol] = sma(close[symbol], window=params.fast_period)
            slow_ma_dict[symbol] = sma(close[symbol], window=params.slow_period)
            
            # ATR for stops
            atr_dict[symbol] = atr(high[symbol], low[symbol], close[symbol], window=14)
            
            # RSI for momentum confirmation
            rsi_dict[symbol] = rsi(close[symbol], window=14)
            
            # Volume MA
            if volume is not None and symbol in volume.columns:
                volume_ma_dict[symbol] = sma(volume[symbol], window=20)
        
        # Create DataFrames from dictionaries
        indicators['fast_ma'] = pd.DataFrame(fast_ma_dict)
        indicators['slow_ma'] = pd.DataFrame(slow_ma_dict)
        indicators['atr'] = pd.DataFrame(atr_dict)
        indicators['rsi'] = pd.DataFrame(rsi_dict)
        if volume_ma_dict:
            indicators['volume_ma'] = pd.DataFrame(volume_ma_dict)
        
        # Market regime detection
        if self.params['use_market_regime']:
            indicators['market_regime'] = self._detect_market_regime(close)
            
        # Volatility percentiles
        returns = close.pct_change(fill_method=None)
        volatility = returns.rolling(self.params['volatility_lookback']).std()
        indicators['volatility_percentile'] = volatility.rank(pct=True)
        
        return indicators
    
    def _detect_market_regime(self, close: pd.DataFrame) -> pd.Series:
        """
        Detect market regime (trending vs ranging).
        
        Returns:
            Series with values: 'bull', 'bear', 'ranging'
        """
        # Use average price across all symbols
        avg_close = close.mean(axis=1)
        
        # Calculate trend using multiple timeframes
        ma_short = avg_close.rolling(self.params['market_regime_lookback']).mean()
        ma_long = avg_close.rolling(self.params['market_regime_lookback'] * 2).mean()
        
        # Calculate ADX-like metric for trend strength
        high_low_range = (close.max(axis=1) - close.min(axis=1)).rolling(14).mean()
        close_range = close.std(axis=1).rolling(14).mean()
        trend_strength = close_range / high_low_range
        
        # Classify regime
        regime = pd.Series('ranging', index=close.index)
        
        # Bull market: price above both MAs and strong trend
        bull_condition = (avg_close > ma_short) & (ma_short > ma_long) & (trend_strength > 0.5)
        regime[bull_condition] = 'bull'
        
        # Bear market: price below both MAs and strong trend
        bear_condition = (avg_close < ma_short) & (ma_short < ma_long) & (trend_strength > 0.5)
        regime[bear_condition] = 'bear'
        
        return regime
    
    def _generate_base_signals(self) -> Dict[str, pd.DataFrame]:
        """Generate base signals for all symbols."""
        close = self.data.close
        volume = self.data.volume if hasattr(self.data, 'volume') else None
        
        # Get custom indicators
        custom_indicators = self.indicators
        
        # Initialize signal DataFrames
        long_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        long_exits = pd.DataFrame(False, index=close.index, columns=close.columns)
        short_entries = pd.DataFrame(False, index=close.index, columns=close.columns)
        short_exits = pd.DataFrame(False, index=close.index, columns=close.columns)
        
        # Position size adjustments
        position_size_adj = pd.DataFrame(1.0, index=close.index, columns=close.columns)
        
        # Generate signals for each symbol
        for symbol in self.data.symbols:
            params = self.symbol_params[symbol]
            
            # Basic MA crossover signals
            fast_ma = custom_indicators['fast_ma'][symbol]
            slow_ma = custom_indicators['slow_ma'][symbol]
            
            # Trend strength filter
            trend_strength = (fast_ma - slow_ma) / slow_ma
            strong_uptrend = trend_strength > params.min_trend_strength
            strong_downtrend = trend_strength < -params.min_trend_strength
            
            # RSI confirmation
            rsi_val = custom_indicators['rsi'][symbol]
            rsi_oversold = rsi_val < params.rsi_threshold_long
            rsi_overbought = rsi_val > params.rsi_threshold_short
            
            # Volume filter
            volume_confirmed = pd.Series(True, index=close.index)
            if self.params['use_volume_filter'] and 'volume_ma' in custom_indicators and volume is not None:
                vol_ma = custom_indicators['volume_ma'][symbol]
                volume_confirmed = volume[symbol] > (vol_ma * params.volume_factor)
            
            # Volatility filter
            vol_filter = pd.Series(True, index=close.index)
            if self.params['use_volatility_filter'] and 'volatility_percentile' in custom_indicators:
                vol_pct = custom_indicators['volatility_percentile'][symbol]
                vol_filter = (
                    (vol_pct > self.params['min_volatility_percentile'] / 100) &
                    (vol_pct < self.params['max_volatility_percentile'] / 100)
                )
            
            # Market regime filter
            regime_filter = pd.Series(True, index=close.index)
            if self.params['use_market_regime'] and 'market_regime' in custom_indicators:
                regime = custom_indicators['market_regime']
                # Only take long signals in bull/ranging, shorts in bear/ranging
                long_regime_filter = regime.isin(['bull', 'ranging'])
                short_regime_filter = regime.isin(['bear', 'ranging'])
            else:
                long_regime_filter = pd.Series(True, index=close.index)
                short_regime_filter = pd.Series(True, index=close.index)
            
            # Generate entry signals with all filters
            # Basic crossover condition
            basic_long_entry = (
                (fast_ma > slow_ma) & 
                (fast_ma.shift(1) <= slow_ma.shift(1)) &
                strong_uptrend
            )
            
            # Apply filters progressively (use OR for RSI instead of AND)
            long_entries[symbol] = (
                basic_long_entry &
                (rsi_oversold | (rsi_val < 50)) &  # More relaxed RSI condition
                volume_confirmed &
                vol_filter &
                long_regime_filter
            )
            
            # Basic crossover condition for shorts
            basic_short_entry = (
                (fast_ma < slow_ma) & 
                (fast_ma.shift(1) >= slow_ma.shift(1)) &
                strong_downtrend
            )
            
            # Apply filters progressively (use OR for RSI instead of AND)
            short_entries[symbol] = (
                basic_short_entry &
                (rsi_overbought | (rsi_val > 50)) &  # More relaxed RSI condition
                volume_confirmed &
                vol_filter &
                short_regime_filter
            )
            
            # Exit signals
            long_exits[symbol] = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))
            short_exits[symbol] = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
            
            # Adjust position size based on correlation
            if self.params['use_correlation_sizing'] and len(self.data.symbols) > 1:
                position_size_adj[symbol] = self._calculate_correlation_adjustment(
                    symbol, position_size_adj.index
                )
        
        # Calculate stop levels
        sl_levels = pd.DataFrame(index=close.index, columns=close.columns)
        tp_levels = pd.DataFrame(index=close.index, columns=close.columns)
        
        for symbol in self.data.symbols:
            params = self.symbol_params[symbol]
            atr_val = custom_indicators['atr'][symbol]
            
            # Percentage-based stops
            sl_levels[symbol] = (atr_val / close[symbol]) * params.atr_multiplier_sl
            tp_levels[symbol] = (atr_val / close[symbol]) * params.atr_multiplier_tp
        
        return {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'sl_levels': sl_levels,
            'tp_levels': tp_levels,
            'position_size_adj': position_size_adj
        }
    
    def _calculate_correlation_adjustment(
        self, symbol: str, index: pd.DatetimeIndex
    ) -> pd.Series:
        """
        Calculate position size adjustment based on correlation with other positions.
        
        High correlation with existing positions -> reduce position size
        """
        adjustment = pd.Series(1.0, index=index)
        
        if 'cross_symbol' not in self.indicators or 'correlation_matrix' not in self.indicators['cross_symbol']:
            return adjustment
        
        # Get average correlation for the symbol
        if 'avg_correlations' in self.indicators['cross_symbol']:
            avg_corr = self.indicators['cross_symbol']['avg_correlations'][symbol]
            
            # Reduce position size if correlation is high
            high_corr_mask = avg_corr > self.params['max_correlation']
            adjustment[high_corr_mask] = self.params['position_reduction_factor']
            
            # Gradual reduction for moderate correlation
            moderate_corr_mask = (avg_corr > self.params['max_correlation'] * 0.8) & ~high_corr_mask
            adjustment[moderate_corr_mask] = 1 - (avg_corr[moderate_corr_mask] - self.params['max_correlation'] * 0.8) / \
                                           (self.params['max_correlation'] * 0.2) * \
                                           (1 - self.params['position_reduction_factor'])
        
        return adjustment
    
    def get_signal_strength(self, signals: Dict[str, pd.DataFrame], indicators: Dict[str, Any]) -> pd.DataFrame:
        """
        Calculate signal strength for each symbol.
        
        Returns:
            DataFrame with signal strength scores (0-1) for each symbol
        """
        close = self.data.close
        strength = pd.DataFrame(0.0, index=close.index, columns=close.columns)
        
        custom_indicators = self.indicators
        
        for symbol in close.columns:
            # Trend strength component
            fast_ma = custom_indicators['fast_ma'][symbol]
            slow_ma = custom_indicators['slow_ma'][symbol]
            trend_strength = abs((fast_ma - slow_ma) / slow_ma)
            trend_score = trend_strength.clip(0, 0.1) * 10  # Normalize to 0-1
            
            # RSI component
            rsi_val = custom_indicators['rsi'][symbol]
            rsi_score = pd.Series(0.5, index=close.index)
            rsi_score[rsi_val < 30] = 1 - (rsi_val[rsi_val < 30] / 30)
            rsi_score[rsi_val > 70] = (rsi_val[rsi_val > 70] - 70) / 30
            
            # Volume component
            volume_score = pd.Series(0.5, index=close.index)
            if 'volume_ma' in custom_indicators and symbol in custom_indicators['volume_ma'].columns:
                if hasattr(self.data, 'volume'):
                    vol_ratio = self.data.volume[symbol] / custom_indicators['volume_ma'][symbol]
                    volume_score = (vol_ratio - 1).clip(0, 1)
            
            # Market breadth component
            breadth_score = pd.Series(0.5, index=close.index)
            if 'market' in self.indicators and 'market_breadth' in self.indicators['market']:
                breadth_score = self.indicators['market']['market_breadth']
            
            # Combine scores
            strength[symbol] = (
                trend_score * 0.3 +
                rsi_score * 0.3 +
                volume_score * 0.2 +
                breadth_score * 0.2
            ).clip(0, 1)
        
        return strength
    
    def get_strategy_description(self) -> str:
        """Get detailed strategy description."""
        return f"""
        Advanced Multi-Symbol Strategy
        
        Features:
        1. Symbol-Specific Parameters:
           - Major cryptos: {self.CRYPTO_MAJOR_PARAMS}
           - Alt cryptos: {self.CRYPTO_ALT_PARAMS}
        
        2. Market Regime Detection:
           - Adapts to bull/bear/ranging markets
           - Lookback: {self.params['market_regime_lookback']} periods
        
        3. Correlation-Based Sizing:
           - Max correlation: {self.params['max_correlation']}
           - Reduction factor: {self.params['position_reduction_factor']}
        
        4. Advanced Filters:
           - Volume confirmation
           - Volatility percentiles: {self.params['min_volatility_percentile']}-{self.params['max_volatility_percentile']}
           - RSI momentum confirmation
           - Trend strength requirements
        
        5. Risk Management:
           - Dynamic ATR-based stops per symbol
           - Position size adjustments based on correlation
           - Market regime filters
        """
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.params.copy()
    
    def validate_parameters(self) -> bool:
        """Validate strategy parameters."""
        try:
            # Validate basic parameters
            if self.params['max_correlation'] < 0 or self.params['max_correlation'] > 1:
                return False
            if self.params['position_reduction_factor'] < 0 or self.params['position_reduction_factor'] > 1:
                return False
            if self.params['min_volatility_percentile'] < 0 or self.params['min_volatility_percentile'] > 100:
                return False
            if self.params['max_volatility_percentile'] < 0 or self.params['max_volatility_percentile'] > 100:
                return False
            if self.params['min_volatility_percentile'] >= self.params['max_volatility_percentile']:
                return False
            
            # Validate symbol parameters
            for symbol, params in self.symbol_params.items():
                if params.fast_period >= params.slow_period:
                    return False
                if params.atr_multiplier_sl <= 0 or params.atr_multiplier_tp <= 0:
                    return False
                if params.volume_factor <= 0:
                    return False
                if params.min_trend_strength < 0:
                    return False
            
            return True
        except Exception:
            return False