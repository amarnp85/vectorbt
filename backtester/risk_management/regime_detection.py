"""
Market Regime Detection Module

Implements various methods to detect market regimes:
- Volatility regimes (low, normal, high)
- Trend regimes (bullish, bearish, ranging)
- Combined regime analysis
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """
    Comprehensive market regime detection.
    
    Combines multiple methods to identify current market conditions.
    """
    
    def __init__(
        self,
        lookback_short: int = 20,
        lookback_long: int = 60,
        volatility_thresholds: Tuple[float, float] = (0.3, 0.7),
        trend_threshold: float = 0.02
    ):
        """
        Initialize market regime detector.
        
        Args:
            lookback_short: Short period for fast-moving indicators
            lookback_long: Long period for slow-moving indicators
            volatility_thresholds: Percentile thresholds for vol regimes
            trend_threshold: Minimum trend strength for trending regime
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.volatility_thresholds = volatility_thresholds
        self.trend_threshold = trend_threshold
    
    def detect_all_regimes(
        self,
        price_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Detect all regime types.
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            DataFrame with regime classifications
        """
        # Detect individual regimes
        vol_regime = VolatilityRegime(
            lookback=self.lookback_short,
            thresholds=self.volatility_thresholds
        ).detect(price_data)
        
        trend_regime = TrendRegime(
            lookback_short=self.lookback_short,
            lookback_long=self.lookback_long,
            threshold=self.trend_threshold
        ).detect(price_data)
        
        # Combine regimes
        combined_regime = self._combine_regimes(vol_regime, trend_regime)
        
        return pd.DataFrame({
            'volatility_regime': vol_regime,
            'trend_regime': trend_regime,
            'combined_regime': combined_regime
        })
    
    def _combine_regimes(
        self,
        vol_regime: pd.Series,
        trend_regime: pd.Series
    ) -> pd.Series:
        """
        Combine volatility and trend regimes into unified classification.
        
        Args:
            vol_regime: Volatility regime series
            trend_regime: Trend regime series
            
        Returns:
            Combined regime classification
        """
        combined = pd.Series(index=vol_regime.index, dtype=str)
        
        # Define combined regime rules
        conditions = [
            # Trending markets
            (trend_regime == 'bullish') & (vol_regime == 'low'),
            (trend_regime == 'bullish') & (vol_regime == 'normal'),
            (trend_regime == 'bullish') & (vol_regime == 'high'),
            (trend_regime == 'bearish') & (vol_regime == 'low'),
            (trend_regime == 'bearish') & (vol_regime == 'normal'),
            (trend_regime == 'bearish') & (vol_regime == 'high'),
            
            # Ranging markets
            (trend_regime == 'ranging') & (vol_regime == 'low'),
            (trend_regime == 'ranging') & (vol_regime == 'normal'),
            (trend_regime == 'ranging') & (vol_regime == 'high'),
        ]
        
        choices = [
            'quiet_uptrend',
            'normal_uptrend',
            'volatile_uptrend',
            'quiet_downtrend',
            'normal_downtrend',
            'volatile_downtrend',
            'quiet_range',
            'normal_range',
            'volatile_range'
        ]
        
        combined = pd.Series(
            np.select(conditions, choices, default='undefined'),
            index=vol_regime.index
        )
        
        return combined


class VolatilityRegime:
    """
    Volatility-based regime detection.
    
    Classifies market into low, normal, or high volatility regimes.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        thresholds: Tuple[float, float] = (0.3, 0.7),
        method: str = 'realized'
    ):
        """
        Initialize volatility regime detector.
        
        Args:
            lookback: Period for volatility calculation
            thresholds: Percentile thresholds (low/normal, normal/high)
            method: 'realized', 'garch', or 'parkinson'
        """
        self.lookback = lookback
        self.thresholds = thresholds
        self.method = method
    
    def detect(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Detect volatility regime.
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            Series with volatility regime labels
        """
        # Calculate volatility
        if self.method == 'realized':
            volatility = self._realized_volatility(price_data)
        elif self.method == 'parkinson':
            volatility = self._parkinson_volatility(price_data)
        else:
            volatility = self._realized_volatility(price_data)
        
        # Calculate rolling percentile rank
        vol_rank = volatility.rolling(252).rank(pct=True)
        
        # Classify regime
        regime = pd.Series('normal', index=volatility.index)
        regime[vol_rank < self.thresholds[0]] = 'low'
        regime[vol_rank > self.thresholds[1]] = 'high'
        
        return regime
    
    def _realized_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate realized volatility."""
        returns = price_data['close'].pct_change()
        return returns.rolling(self.lookback).std() * np.sqrt(252)
    
    def _parkinson_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator."""
        high_low_ratio = np.log(price_data['high'] / price_data['low'])
        return high_low_ratio.rolling(self.lookback).apply(
            lambda x: np.sqrt(np.sum(x**2) / (4 * len(x) * np.log(2)))
        ) * np.sqrt(252)


class TrendRegime:
    """
    Trend-based regime detection.
    
    Classifies market into bullish, bearish, or ranging regimes.
    """
    
    def __init__(
        self,
        lookback_short: int = 20,
        lookback_long: int = 60,
        threshold: float = 0.02,
        method: str = 'ma_cross'
    ):
        """
        Initialize trend regime detector.
        
        Args:
            lookback_short: Short MA period
            lookback_long: Long MA period
            threshold: Minimum trend strength
            method: 'ma_cross', 'adx', or 'linear_regression'
        """
        self.lookback_short = lookback_short
        self.lookback_long = lookback_long
        self.threshold = threshold
        self.method = method
    
    def detect(self, price_data: pd.DataFrame) -> pd.Series:
        """
        Detect trend regime.
        
        Args:
            price_data: DataFrame with OHLC data
            
        Returns:
            Series with trend regime labels
        """
        if self.method == 'ma_cross':
            return self._ma_cross_regime(price_data)
        elif self.method == 'adx':
            return self._adx_regime(price_data)
        elif self.method == 'linear_regression':
            return self._linear_regression_regime(price_data)
        else:
            return self._ma_cross_regime(price_data)
    
    def _ma_cross_regime(self, price_data: pd.DataFrame) -> pd.Series:
        """Detect regime using moving average crossovers."""
        close = price_data['close']
        
        # Calculate moving averages
        ma_short = close.rolling(self.lookback_short).mean()
        ma_long = close.rolling(self.lookback_long).mean()
        
        # Calculate trend strength
        trend_strength = (ma_short - ma_long) / ma_long
        
        # Classify regime
        regime = pd.Series('ranging', index=close.index)
        regime[trend_strength > self.threshold] = 'bullish'
        regime[trend_strength < -self.threshold] = 'bearish'
        
        return regime
    
    def _adx_regime(self, price_data: pd.DataFrame) -> pd.Series:
        """Detect regime using ADX indicator."""
        # Calculate ADX
        high = price_data['high']
        low = price_data['low']
        close = price_data['close']
        
        # Calculate directional movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # True range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smoothed indicators
        atr = tr.rolling(self.lookback_short).mean()
        plus_di = 100 * (plus_dm.rolling(self.lookback_short).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.lookback_short).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(self.lookback_short).mean()
        
        # Classify regime
        regime = pd.Series('ranging', index=close.index)
        trending = adx > 25  # Standard ADX threshold
        
        regime[trending & (plus_di > minus_di)] = 'bullish'
        regime[trending & (plus_di < minus_di)] = 'bearish'
        
        return regime
    
    def _linear_regression_regime(self, price_data: pd.DataFrame) -> pd.Series:
        """Detect regime using linear regression slope."""
        close = price_data['close']
        
        # Calculate rolling linear regression slope
        def calculate_slope(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0] / x.mean()
        
        slope = close.rolling(self.lookback_long).apply(calculate_slope)
        
        # Classify regime
        regime = pd.Series('ranging', index=close.index)
        regime[slope > self.threshold] = 'bullish'
        regime[slope < -self.threshold] = 'bearish'
        
        return regime


class RegimeTransitionDetector:
    """
    Detect regime transitions and their characteristics.
    
    Useful for adjusting strategy behavior during regime changes.
    """
    
    def __init__(self, min_regime_length: int = 20):
        """
        Initialize regime transition detector.
        
        Args:
            min_regime_length: Minimum bars for regime confirmation
        """
        self.min_regime_length = min_regime_length
    
    def detect_transitions(
        self,
        regime: pd.Series
    ) -> pd.DataFrame:
        """
        Detect regime transitions.
        
        Args:
            regime: Series with regime labels
            
        Returns:
            DataFrame with transition information
        """
        # Detect changes
        regime_changes = regime != regime.shift(1)
        
        # Filter out short-lived regimes
        regime_filtered = self._filter_short_regimes(regime)
        
        # Identify transitions
        transitions = []
        current_regime = None
        regime_start = None
        
        for i, (date, reg) in enumerate(regime_filtered.items()):
            if reg != current_regime:
                if current_regime is not None:
                    transitions.append({
                        'date': date,
                        'from_regime': current_regime,
                        'to_regime': reg,
                        'regime_length': i - regime_start
                    })
                current_regime = reg
                regime_start = i
        
        return pd.DataFrame(transitions)
    
    def _filter_short_regimes(self, regime: pd.Series) -> pd.Series:
        """Filter out regimes shorter than minimum length."""
        filtered = regime.copy()
        
        # Group consecutive regimes
        regime_groups = (regime != regime.shift(1)).cumsum()
        
        # Calculate regime lengths
        regime_lengths = regime.groupby(regime_groups).transform('count')
        
        # Replace short regimes with previous regime
        mask = regime_lengths < self.min_regime_length
        filtered[mask] = np.nan
        filtered = filtered.fillna(method='ffill')
        
        return filtered 