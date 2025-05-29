"""
Pairs Trading Strategy

A statistical arbitrage strategy that trades pairs of cointegrated assets.
When the spread between them deviates from the mean, we bet on convergence.
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, Optional, Tuple, List
import logging
from statsmodels.tsa.stattools import coint

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class PairsTradingStrategy(BaseStrategy):
    """
    Pairs trading strategy with cointegration-based pair selection.
    
    Features:
    - Cointegration testing for pair selection
    - Z-score based entry/exit signals
    - OLS-based spread calculation
    - Support for multiple pairs
    
    Note: Position sizing and portfolio management are handled at the portfolio level.
    """
    
    def __init__(
        self,
        data: vbt.Data,
        params: Dict[str, Any],
        pair: Optional[Tuple[str, str]] = None,
        **kwargs
    ):
        """
        Initialize pairs trading strategy.
        
        Args:
            data: VBT Data object with multiple symbols
            params: Strategy parameters including:
                - window: Lookback period for z-score calculation
                - entry_z: Z-score threshold for entry (e.g., 2.0)
                - exit_z: Z-score threshold for exit (e.g., 0.0)
                - coint_pvalue: P-value threshold for cointegration test
                - atr_multiplier_sl: ATR multiplier for stop loss
                - atr_multiplier_tp: ATR multiplier for take profit
            pair: Optional tuple of (symbol1, symbol2) to trade
        """
        # Validate we have multi-symbol data
        if not hasattr(data, 'symbols') or len(data.symbols) < 2:
            raise ValueError("Pairs trading requires at least 2 symbols in the data")
        
        # Set default parameters
        default_params = {
            'window': 30,
            'entry_z': 2.0,
            'exit_z': 0.0,
            'coint_pvalue': 0.05,
            'atr_multiplier_sl': 2.0,
            'atr_multiplier_tp': 3.0
        }
        
        # Merge with provided params
        self.params = {**default_params, **params}
        self.pair = pair
        self.selected_pair = None
        
        super().__init__(data, self.params, **kwargs)
    
    def find_cointegrated_pairs(
        self,
        pvalue_threshold: float = 0.05
    ) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated pairs from available symbols.
        
        Args:
            pvalue_threshold: P-value threshold for cointegration
            
        Returns:
            List of tuples (symbol1, symbol2, pvalue)
        """
        symbols = list(self.data.symbols)
        close_prices = self.data.close
        cointegrated_pairs = []
        
        # Test all pairs
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                s1, s2 = symbols[i], symbols[j]
                
                # Get log prices
                log_s1 = np.log(close_prices[s1].dropna())
                log_s2 = np.log(close_prices[s2].dropna())
                
                # Align indices
                common_idx = log_s1.index.intersection(log_s2.index)
                if len(common_idx) < self.params['window'] * 2:
                    continue
                
                log_s1 = log_s1[common_idx]
                log_s2 = log_s2[common_idx]
                
                # Test cointegration
                try:
                    _, pvalue, _ = coint(log_s1, log_s2)
                    if pvalue < pvalue_threshold:
                        cointegrated_pairs.append((s1, s2, pvalue))
                except Exception as e:
                    logger.debug(f"Cointegration test failed for {s1}-{s2}: {e}")
        
        # Sort by p-value
        cointegrated_pairs.sort(key=lambda x: x[2])
        
        return cointegrated_pairs
    
    def calculate_spread_zscore(
        self,
        s1_prices: pd.Series,
        s2_prices: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate spread and z-score using OLS.
        
        Args:
            s1_prices: Prices of first asset
            s2_prices: Prices of second asset
            
        Returns:
            Tuple of (spread, zscore, hedge_ratio)
        """
        # Use vectorbt's OLS indicator
        ols = vbt.OLS.run(
            s1_prices,
            s2_prices,
            window=self.params['window']
        )
        
        spread = ols.error
        zscore = ols.zscore
        hedge_ratio = ols.coeff
        
        return spread, zscore, hedge_ratio
    
    def init_indicators(self) -> Dict[str, Any]:
        """Calculate pairs trading indicators."""
        # Select pair if not provided
        if self.pair is None:
            logger.info("No pair specified, finding cointegrated pairs...")
            pairs = self.find_cointegrated_pairs(
                pvalue_threshold=self.params['coint_pvalue']
            )
            
            if not pairs:
                raise ValueError("No cointegrated pairs found!")
            
            self.selected_pair = (pairs[0][0], pairs[0][1])
            logger.info(f"Selected pair: {self.selected_pair[0]}-{self.selected_pair[1]} (p-value: {pairs[0][2]:.4f})")
        else:
            self.selected_pair = self.pair
        
        s1, s2 = self.selected_pair
        
        # Get prices
        s1_close = self.data.close[s1]
        s2_close = self.data.close[s2]
        
        # Calculate spread and z-score
        spread, zscore, hedge_ratio = self.calculate_spread_zscore(s1_close, s2_close)
        
        self.indicators['spread'] = spread
        self.indicators['zscore'] = zscore
        self.indicators['hedge_ratio'] = hedge_ratio
        self.indicators['s1_close'] = s1_close
        self.indicators['s2_close'] = s2_close
        
        # Calculate ATR for both assets (for risk management)
        if all(col in self.data.columns.get_level_values(0) for col in ['high', 'low']):
            from ..indicators.simple_indicators import atr
            
            s1_atr = atr(
                self.data.high[s1], 
                self.data.low[s1], 
                self.data.close[s1], 
                window=14
            )
            s2_atr = atr(
                self.data.high[s2], 
                self.data.low[s2], 
                self.data.close[s2], 
                window=14
            )
            
            # Combined ATR (weighted by hedge ratio)
            self.indicators['combined_atr'] = s1_atr + abs(hedge_ratio) * s2_atr
        
        return self.indicators
    
    def generate_pairs_signals(
        self,
        zscore: pd.Series
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Generate trading signals from z-score.
        
        Args:
            zscore: Z-score series
            
        Returns:
            Tuple of (long_spread, short_spread) signals
        """
        entry_z = self.params['entry_z']
        exit_z = self.params['exit_z']
        
        # Initialize signals
        long_spread = pd.Series(False, index=zscore.index)
        short_spread = pd.Series(False, index=zscore.index)
        long_spread_exits = pd.Series(False, index=zscore.index)
        short_spread_exits = pd.Series(False, index=zscore.index)
        
        # Track position
        position = 0
        
        for i in range(len(zscore)):
            if pd.isna(zscore.iloc[i]):
                continue
            
            z = zscore.iloc[i]
            
            if position == 0:
                # Enter positions
                if z > entry_z:
                    short_spread.iloc[i] = True  # Short the spread
                    position = -1
                elif z < -entry_z:
                    long_spread.iloc[i] = True  # Long the spread
                    position = 1
            elif position == 1:
                # Exit long
                if z > -exit_z:
                    long_spread_exits.iloc[i] = True
                    position = 0
            elif position == -1:
                # Exit short
                if z < exit_z:
                    short_spread_exits.iloc[i] = True
                    position = 0
        
        return long_spread, long_spread_exits, short_spread, short_spread_exits
    
    def generate_signals(self) -> Dict[str, pd.Series]:
        """
        Generate trading signals for the pair.
        
        Returns:
            Dictionary with signal arrays
        """
        zscore = self.indicators['zscore']
        s1, s2 = self.selected_pair
        
        # Generate spread signals
        long_spread, long_spread_exits, short_spread, short_spread_exits = self.generate_pairs_signals(zscore)
        
        # Convert to individual asset signals
        # When long spread: long S1, short S2
        # When short spread: short S1, long S2
        
        # Initialize signal DataFrames for all symbols
        all_symbols = list(self.data.symbols)
        signal_index = zscore.index
        
        long_entries = pd.DataFrame(False, index=signal_index, columns=all_symbols)
        long_exits = pd.DataFrame(False, index=signal_index, columns=all_symbols)
        short_entries = pd.DataFrame(False, index=signal_index, columns=all_symbols)
        short_exits = pd.DataFrame(False, index=signal_index, columns=all_symbols)
        
        # Set signals for the pair
        # Long spread = Long S1, Short S2
        long_entries.loc[long_spread, s1] = True
        short_entries.loc[long_spread, s2] = True
        long_exits.loc[long_spread_exits, s1] = True
        short_exits.loc[long_spread_exits, s2] = True
        
        # Short spread = Short S1, Long S2
        short_entries.loc[short_spread, s1] = True
        long_entries.loc[short_spread, s2] = True
        short_exits.loc[short_spread_exits, s1] = True
        long_exits.loc[short_spread_exits, s2] = True
        
        # Calculate stop levels if ATR is available
        sl_levels = pd.DataFrame(np.nan, index=signal_index, columns=all_symbols)
        tp_levels = pd.DataFrame(np.nan, index=signal_index, columns=all_symbols)
        
        if 'combined_atr' in self.indicators:
            atr_values = self.indicators['combined_atr']
            s1_close = self.indicators['s1_close']
            s2_close = self.indicators['s2_close']
            
            # For long spread positions
            long_mask = long_spread
            if long_mask.any():
                # S1 long stops
                sl_levels.loc[long_mask, s1] = s1_close[long_mask] - (
                    self.params['atr_multiplier_sl'] * atr_values[long_mask]
                )
                tp_levels.loc[long_mask, s1] = s1_close[long_mask] + (
                    self.params['atr_multiplier_tp'] * atr_values[long_mask]
                )
                
                # S2 short stops (inverted)
                sl_levels.loc[long_mask, s2] = s2_close[long_mask] + (
                    self.params['atr_multiplier_sl'] * atr_values[long_mask]
                )
                tp_levels.loc[long_mask, s2] = s2_close[long_mask] - (
                    self.params['atr_multiplier_tp'] * atr_values[long_mask]
                )
            
            # For short spread positions
            short_mask = short_spread
            if short_mask.any():
                # S1 short stops
                sl_levels.loc[short_mask, s1] = s1_close[short_mask] + (
                    self.params['atr_multiplier_sl'] * atr_values[short_mask]
                )
                tp_levels.loc[short_mask, s1] = s1_close[short_mask] - (
                    self.params['atr_multiplier_tp'] * atr_values[short_mask]
                )
                
                # S2 long stops
                sl_levels.loc[short_mask, s2] = s2_close[short_mask] - (
                    self.params['atr_multiplier_sl'] * atr_values[short_mask]
                )
                tp_levels.loc[short_mask, s2] = s2_close[short_mask] + (
                    self.params['atr_multiplier_tp'] * atr_values[short_mask]
                )
        
        # Store signals
        self.signals = {
            'long_entries': long_entries,
            'long_exits': long_exits,
            'short_entries': short_entries,
            'short_exits': short_exits,
            'sl_levels': sl_levels if not sl_levels.isna().all().all() else None,
            'tp_levels': tp_levels if not tp_levels.isna().all().all() else None
        }
        
        return self.signals
    
    def get_pair_metadata(self) -> Dict[str, Any]:
        """Get metadata about the selected pair.
        
        Returns:
            Dictionary with pair information
        """
        if self.selected_pair is None:
            return {}
        
        s1, s2 = self.selected_pair
        
        metadata = {
            'pair': f"{s1}-{s2}",
            'symbol1': s1,
            'symbol2': s2,
            'hedge_ratio': float(self.indicators['hedge_ratio'].iloc[-1]) if 'hedge_ratio' in self.indicators else None,
            'current_zscore': float(self.indicators['zscore'].iloc[-1]) if 'zscore' in self.indicators else None,
            'window': self.params['window'],
            'entry_z': self.params['entry_z'],
            'exit_z': self.params['exit_z']
        }
        
        return metadata
    
    def __repr__(self) -> str:
        """String representation of the strategy."""
        pair_str = f"{self.selected_pair[0]}-{self.selected_pair[1]}" if self.selected_pair else "No pair selected"
        return (
            f"PairsTradingStrategy("
            f"pair={pair_str}, "
            f"window={self.params['window']}, "
            f"entry_z={self.params['entry_z']}, "
            f"exit_z={self.params['exit_z']})"
        ) 