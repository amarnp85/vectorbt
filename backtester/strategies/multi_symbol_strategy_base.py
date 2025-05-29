"""
Multi-Symbol Strategy Base Class

Provides a foundation for strategies that operate across multiple symbols simultaneously,
leveraging VectorBTPro's native multi-symbol support for efficient computation.

Key Features:
- Cross-symbol correlation analysis
- Relative strength indicators
- Market regime detection
- Symbol ranking and selection
- Automatic broadcasting across symbols
"""

import pandas as pd
import numpy as np
import vectorbtpro as vbt
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import logging

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MultiSymbolStrategy(BaseStrategy):
    """
    Base class for multi-symbol trading strategies.
    
    Extends BaseStrategy to handle multiple symbols simultaneously,
    enabling cross-symbol analysis and coordinated signal generation.
    
    Note: Position sizing and portfolio management are handled at the portfolio level.
    """
    
    def __init__(self, data: vbt.Data, params: Dict[str, Any]):
        """
        Initialize multi-symbol strategy.
        
        Args:
            data: VBT Data object with multiple symbols
            params: Strategy parameters including:
                - correlation_lookback: Period for correlation calculation (default: 60)
                - relative_strength_lookback: Period for RS calculation (default: 20)
                - use_symbol_ranking: Whether to rank symbols (default: True)
                - max_active_symbols: Max symbols to trade simultaneously (default: all)
        """
        if not hasattr(data, 'symbols') or len(data.symbols) < 2:
            raise ValueError("MultiSymbolStrategy requires vbt.Data with multiple symbols")
            
        # Set default parameters
        default_params = {
            'correlation_lookback': 60,
            'relative_strength_lookback': 20,
            'use_symbol_ranking': True,
            'max_active_symbols': None,  # None means all symbols
            'min_correlation': -0.5,
            'max_correlation': 0.8
        }
        
        # Merge with provided params
        params = {**default_params, **params}
        
        super().__init__(data, params)
        
        self.symbols = list(data.symbols)
        self.n_symbols = len(self.symbols)
        
        # Multi-symbol specific parameters
        self.correlation_lookback = params['correlation_lookback']
        self.relative_strength_lookback = params['relative_strength_lookback']
        self.use_symbol_ranking = params['use_symbol_ranking']
        self.max_active_symbols = params.get('max_active_symbols', self.n_symbols)
        
        logger.info(f"Initialized MultiSymbolStrategy with {self.n_symbols} symbols: {self.symbols}")
    
    def init_indicators(self) -> Dict[str, Any]:
        """
        Initialize indicators for all symbols plus cross-symbol indicators.
        
        Returns:
            Dictionary of indicators including cross-symbol metrics
        """
        # Calculate per-symbol indicators using VBT's broadcasting
        symbol_indicators = self._calculate_all_symbol_indicators()
        
        # Calculate cross-symbol indicators
        cross_indicators = self._calculate_cross_symbol_indicators()
        
        # Calculate market-wide indicators
        market_indicators = self._calculate_market_indicators()
        
        # Combine all indicators
        self.indicators = {
            **symbol_indicators,
            'cross_symbol': cross_indicators,
            'market': market_indicators
        }
        
        return self.indicators
    
    def _calculate_all_symbol_indicators(self) -> Dict[str, Any]:
        """
        Calculate indicators for all symbols at once using VBT broadcasting.
        
        Returns:
            Dictionary of indicators
        """
        # VBT automatically broadcasts calculations across all symbols
        close = self.data.close
        high = self.data.high
        low = self.data.low
        volume = self.data.volume if hasattr(self.data, 'volume') else None
        
        indicators = {}
        
        # Example: SMA for all symbols at once
        indicators['sma_20'] = vbt.talib("SMA").run(close, timeperiod=20).real
        indicators['sma_50'] = vbt.talib("SMA").run(close, timeperiod=50).real
        
        # RSI for all symbols
        indicators['rsi'] = vbt.talib("RSI").run(close, timeperiod=14).real
        
        # ATR for all symbols
        indicators['atr'] = vbt.talib("ATR").run(high, low, close, timeperiod=14).real
        
        # Returns for all symbols
        indicators['returns'] = close.pct_change()
        
        # Call strategy-specific indicator calculation
        custom_indicators = self._calculate_custom_indicators()
        indicators.update(custom_indicators)
        
        return indicators
    
    @abstractmethod
    def _calculate_custom_indicators(self) -> Dict[str, Any]:
        """
        Calculate strategy-specific indicators.
        
        Returns:
            Dictionary of custom indicators
        """
        pass
    
    def _calculate_cross_symbol_indicators(self) -> Dict[str, Any]:
        """
        Calculate cross-symbol indicators like correlations and relative strength.
        
        Returns:
            Dictionary of cross-symbol indicators
        """
        returns = self.indicators['returns']
        
        indicators = {}
        
        # Rolling correlation matrix using VBT
        correlation_matrix = returns.vbt.rolling_corr(
            window=self.correlation_lookback,
            pairwise=True
        )
        indicators['correlation_matrix'] = correlation_matrix
        
        # Average correlation for each symbol
        avg_correlations = {}
        for symbol in self.symbols:
            # Get correlations with other symbols
            symbol_corrs = []
            for other_symbol in self.symbols:
                if symbol != other_symbol:
                    corr = returns[symbol].rolling(self.correlation_lookback).corr(
                        returns[other_symbol]
                    )
                    symbol_corrs.append(corr)
            
            if symbol_corrs:
                avg_correlations[symbol] = pd.concat(symbol_corrs, axis=1).mean(axis=1)
        
        indicators['avg_correlations'] = pd.DataFrame(avg_correlations)
        
        # Relative strength vs equal-weight portfolio
        portfolio_returns = returns.mean(axis=1)
        relative_strength = {}
        
        for symbol in self.symbols:
            # Calculate cumulative returns
            symbol_cumret = (1 + returns[symbol]).rolling(
                self.relative_strength_lookback
            ).apply(lambda x: x.prod())
            
            portfolio_cumret = (1 + portfolio_returns).rolling(
                self.relative_strength_lookback
            ).apply(lambda x: x.prod())
            
            relative_strength[symbol] = symbol_cumret / portfolio_cumret
        
        indicators['relative_strength'] = pd.DataFrame(relative_strength)
        
        # Symbol ranking based on momentum
        if self.use_symbol_ranking:
            momentum_scores = returns.rolling(20).mean()
            rankings = momentum_scores.rank(axis=1, ascending=False)
            indicators['symbol_rankings'] = rankings
        
        return indicators
    
    def _calculate_market_indicators(self) -> Dict[str, Any]:
        """
        Calculate market-wide indicators.
        
        Returns:
            Dictionary of market indicators
        """
        returns = self.indicators['returns']
        
        indicators = {}
        
        # Market breadth (% of symbols with positive returns)
        indicators['market_breadth'] = (returns > 0).sum(axis=1) / self.n_symbols
        
        # Market volatility (average volatility across symbols)
        symbol_vols = returns.rolling(20).std()
        indicators['market_volatility'] = symbol_vols.mean(axis=1) * np.sqrt(252)
        
        # Market trend (using equal-weight portfolio)
        portfolio_returns = returns.mean(axis=1)
        indicators['market_trend'] = vbt.talib("SMA").run(
            portfolio_returns.cumsum(), timeperiod=20
        ).real
        
        # Dispersion (cross-sectional volatility)
        indicators['return_dispersion'] = returns.std(axis=1)
        
        return indicators
    
    def generate_signals(self) -> Dict[str, Any]:
        """
        Generate trading signals for all symbols.
        
        Returns:
            Dictionary containing signals for all symbols
        """
        if self.indicators is None:
            raise ValueError("Indicators not initialized. Call init_indicators() first.")
        
        # Generate base signals for all symbols
        base_signals = self._generate_base_signals()
        
        # Apply cross-symbol filters
        filtered_signals = self._apply_cross_symbol_filters(base_signals)
        
        # Apply symbol selection if enabled
        if self.use_symbol_ranking and self.max_active_symbols < self.n_symbols:
            final_signals = self._apply_symbol_selection(filtered_signals)
        else:
            final_signals = filtered_signals
        
        # Clean signals using VBT
        cleaned_signals = self._clean_multi_symbol_signals(final_signals)
        
        self.signals = cleaned_signals
        return self.signals
    
    @abstractmethod
    def _generate_base_signals(self) -> Dict[str, pd.DataFrame]:
        """
        Generate base signals for all symbols.
        
        Returns:
            Dictionary with signal DataFrames
        """
        pass
    
    def _apply_cross_symbol_filters(
        self, 
        base_signals: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Apply cross-symbol filters like correlation limits.
        
        Args:
            base_signals: Base signals for all symbols
            
        Returns:
            Filtered signals
        """
        filtered_signals = base_signals.copy()
        
        # Apply correlation filter
        if 'avg_correlations' in self.indicators['cross_symbol']:
            avg_corrs = self.indicators['cross_symbol']['avg_correlations']
            
            # Reduce signals for highly correlated symbols
            high_corr_mask = avg_corrs > self.params['max_correlation']
            
            for signal_type in ['long_entries', 'short_entries']:
                if signal_type in filtered_signals:
                    # Mask out signals for highly correlated symbols
                    filtered_signals[signal_type] = (
                        filtered_signals[signal_type] & ~high_corr_mask
                    )
        
        # Apply market regime filter
        if 'market_breadth' in self.indicators['market']:
            breadth = self.indicators['market']['market_breadth']
            
            # Reduce long signals in weak market breadth
            weak_market = breadth < 0.3
            if 'long_entries' in filtered_signals:
                for symbol in self.symbols:
                    filtered_signals['long_entries'][symbol] = (
                        filtered_signals['long_entries'][symbol] & ~weak_market
                    )
        
        return filtered_signals
    
    def _apply_symbol_selection(
        self,
        signals: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Select top-ranked symbols for trading.
        
        Args:
            signals: Signals for all symbols
            
        Returns:
            Signals with only top symbols active
        """
        if 'symbol_rankings' not in self.indicators['cross_symbol']:
            return signals
        
        rankings = self.indicators['cross_symbol']['symbol_rankings']
        selected_signals = signals.copy()
        
        # Keep only top N symbols
        for idx in rankings.index:
            top_symbols = rankings.loc[idx].nsmallest(
                self.max_active_symbols
            ).index.tolist()
            
            # Zero out signals for non-selected symbols
            for signal_type in ['long_entries', 'short_entries']:
                if signal_type in selected_signals:
                    for symbol in self.symbols:
                        if symbol not in top_symbols:
                            selected_signals[signal_type].loc[idx, symbol] = False
        
        return selected_signals
    
    def _clean_multi_symbol_signals(
        self,
        signals: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Clean signals for all symbols using VBT.
        
        Args:
            signals: Raw signals
            
        Returns:
            Cleaned signals
        """
        cleaned_signals = {}
        
        # Clean long signals
        if 'long_entries' in signals and 'long_exits' in signals:
            # VBT can clean multi-symbol signals efficiently
            long_entries = signals['long_entries']
            long_exits = signals['long_exits']
            
            # Clean each symbol's signals
            cleaned_entries = pd.DataFrame(index=long_entries.index)
            cleaned_exits = pd.DataFrame(index=long_exits.index)
            
            for symbol in self.symbols:
                if symbol in long_entries.columns:
                    cleaned_entries[symbol], cleaned_exits[symbol] = \
                        long_entries[symbol].vbt.signals.clean(
                            long_exits[symbol]
                        )
            
            cleaned_signals['long_entries'] = cleaned_entries
            cleaned_signals['long_exits'] = cleaned_exits
        
        # Clean short signals
        if 'short_entries' in signals and 'short_exits' in signals:
            short_entries = signals['short_entries']
            short_exits = signals['short_exits']
            
            cleaned_entries = pd.DataFrame(index=short_entries.index)
            cleaned_exits = pd.DataFrame(index=short_exits.index)
            
            for symbol in self.symbols:
                if symbol in short_entries.columns:
                    cleaned_entries[symbol], cleaned_exits[symbol] = \
                        short_entries[symbol].vbt.signals.clean(
                            short_exits[symbol]
                        )
            
            cleaned_signals['short_entries'] = cleaned_entries
            cleaned_signals['short_exits'] = cleaned_exits
        
        # Copy over stop levels if present
        for key in ['sl_levels', 'tp_levels']:
            if key in signals:
                cleaned_signals[key] = signals[key]
        
        return cleaned_signals
    
    def get_active_symbols(self, timestamp: pd.Timestamp) -> List[str]:
        """
        Get list of symbols that should be traded at a given timestamp.
        
        Args:
            timestamp: Time to check
            
        Returns:
            List of active symbols
        """
        if not self.use_symbol_ranking:
            return self.symbols
        
        if 'symbol_rankings' in self.indicators['cross_symbol']:
            rankings = self.indicators['cross_symbol']['symbol_rankings']
            if timestamp in rankings.index:
                return rankings.loc[timestamp].nsmallest(
                    self.max_active_symbols
                ).index.tolist()
        
        return self.symbols
    
    def __repr__(self) -> str:
        """String representation of multi-symbol strategy."""
        return (
            f"{self.__class__.__name__}("
            f"symbols={self.n_symbols}, "
            f"ranking={'on' if self.use_symbol_ranking else 'off'}, "
            f"max_active={self.max_active_symbols})"
        ) 