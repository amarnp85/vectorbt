"""
Signal Extraction and Processing

This module handles extraction of trading signals from various sources including
VectorBT portfolio objects, strategy-generated signals, and trade records.

Module Purpose:
==============
Trading signals can come from multiple sources:
1. Portfolio trade records (what actually happened)
2. Strategy-generated signals (what the strategy intended)
3. Risk management orders (stop losses, take profits)

This module extracts signals from these sources and converts them into
a unified format for chart rendering and analysis.

Key Features:
============
- Portfolio signal extraction with proper timing
- Strategy signal integration with priority handling
- Stop loss and take profit level extraction
- Signal validation and cleaning
- Multiple signal format support (legacy and unified)

Integration Points:
==================
- Used by: SignalProcessor for signal extraction
- Uses: TimingCalculator for timing corrections
- Data Sources: VectorBT Portfolio, strategy dictionaries
- Output: Unified signal format for rendering

Related Modules:
===============
- timing.py: Provides timing calculations
- validators.py: Validates extracted signals
- ../signals/signal_interface.py: Unified signal format
"""

import numpy as np
import pandas as pd
import vectorbtpro as vbt
from typing import Dict, Any, Optional, List, Tuple, Union
from abc import ABC, abstractmethod

from backtester.analysis.signal_components.timing import TimingCalculator, TimestampNormalizer, TimingConfig
from backtester.signals.signal_interface import SignalFormat, convert_legacy_signals
from backtester.utilities.structured_logging import get_logger

logger = get_logger(__name__)


class ISignalExtractor(ABC):
    """
    Interface for signal extractors.
    
    All signal extractors implement this interface to ensure
    consistent behavior and enable easy swapping of extraction strategies.
    """
    
    @abstractmethod
    def extract_signals(self) -> SignalFormat:
        """Extract signals and return in unified format."""
        pass
    
    @abstractmethod
    def get_extraction_info(self) -> Dict[str, Any]:
        """Get information about the extraction process."""
        pass


class PortfolioSignalExtractor(ISignalExtractor):
    """
    Extracts signals from VectorBT Portfolio trade and order records.
    
    This extractor analyzes the portfolio's trade history to determine:
    - Entry and exit signals based on position changes
    - Signal timing with proper execution delay
    - Entry/exit prices from trade records
    - Stop loss and take profit levels when available
    
    The extractor implements proper timing handling to prevent lookahead bias
    by showing signals at execution time rather than signal generation time.
    
    Usage:
        extractor = PortfolioSignalExtractor(portfolio, data_processor, timing_config)
        signals = extractor.extract_signals()
    """
    
    def __init__(
        self,
        portfolio: vbt.Portfolio,
        data_processor: Any,  # DataProcessor from chart_components
        timing_config: TimingConfig
    ):
        """
        Initialize portfolio signal extractor.
        
        Args:
            portfolio: VectorBT Portfolio object
            data_processor: Data processor for index alignment
            timing_config: Timing configuration for corrections
        """
        self.portfolio = portfolio
        self.data_processor = data_processor
        self.timing_config = timing_config
        
        # Initialize timing utilities
        self.timing_calculator = TimingCalculator(timing_config)
        self.timestamp_normalizer = TimestampNormalizer(
            self.data_processor.get_ohlcv_data().index
        )
        
        self._extraction_info = {}
    
    def extract_signals(self) -> SignalFormat:
        """
        Extract signals from portfolio trade records.
        
        This method analyzes trades to determine entry/exit signals
        with proper timing corrections and price extraction.
        
        Returns:
            Unified signal format with all extracted signals
        """
        logger.debug("Starting portfolio signal extraction")
        
        # Get data index for signal initialization
        index = self.data_processor.get_ohlcv_data().index
        
        # Initialize empty signal format
        signals = SignalFormat(
            long_entries=pd.Series(False, index=index),
            short_entries=pd.Series(False, index=index),
            long_exits=pd.Series(False, index=index),
            short_exits=pd.Series(False, index=index),
            entry_prices=pd.Series(np.nan, index=index),
            exit_prices=pd.Series(np.nan, index=index),
            sl_levels=pd.Series(np.nan, index=index),
            tp_levels=pd.Series(np.nan, index=index),
            sl_price_levels=pd.Series(np.nan, index=index),
            tp_price_levels=pd.Series(np.nan, index=index),
            index=index
        )
        
        # Extract from trades
        trades_processed = self._extract_from_trades(signals)
        
        # Optionally extract from orders (for additional risk management signals)
        orders_processed = False
        if not self.timing_config.mode.value == "trades_only":
            orders_processed = self._extract_from_orders(signals)
        
        # Store extraction info
        self._extraction_info = {
            "trades_processed": trades_processed,
            "orders_processed": orders_processed,
            "timing_mode": self.timing_config.mode.value,
            "execution_delay": self.timing_config.execution_delay,
            "signals_extracted": signals.get_summary()
        }
        
        logger.info(f"Portfolio extraction complete: {signals.get_summary()}")
        return signals
    
    def _extract_from_trades(self, signals: SignalFormat) -> bool:
        """
        Extract signals from trade records.
        
        Args:
            signals: Signal format to populate
            
        Returns:
            True if trades were processed successfully
        """
        try:
            if not hasattr(self.portfolio, 'trades') or len(self.portfolio.trades.records) == 0:
                logger.debug("No trades found in portfolio")
                return False
            
            trades = self.portfolio.trades.records_readable
            logger.debug(f"Processing {len(trades)} trades")
            
            for trade_idx, trade in trades.iterrows():
                self._process_trade_entry(trade, trade_idx, signals)
                self._process_trade_exit(trade, trade_idx, signals)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract signals from trades: {e}")
            return False
    
    def _process_trade_entry(self, trade: pd.Series, trade_idx: int, signals: SignalFormat):
        """Process entry signal from trade record."""
        try:
            # Get and normalize entry timestamp
            entry_timestamp = self.timestamp_normalizer.normalize(trade.get('Entry Index'))
            if entry_timestamp is None or entry_timestamp not in signals.index:
                logger.debug(f"Trade {trade_idx}: Invalid entry timestamp")
                return
            
            # Calculate display timestamp based on timing mode
            display_timestamp = self.timing_calculator.calculate_display_timestamp(
                entry_timestamp, signals.index
            )
            
            if display_timestamp not in signals.index:
                logger.debug(f"Trade {trade_idx}: Display timestamp not in index")
                return
            
            # Extract entry price
            entry_price = self._extract_price(trade, [
                'Entry Price', 'Avg Entry Price', 'Entry Avg Price'
            ])
            
            # Determine trade direction
            direction = self._detect_trade_direction(trade)
            
            # Set appropriate entry signal
            if direction == 'short':
                signals.short_entries.loc[display_timestamp] = True
                signals.long_entries.loc[display_timestamp] = False  # Clear conflicts
                logger.debug(f"Trade {trade_idx}: SHORT entry at {display_timestamp}")
            else:
                signals.long_entries.loc[display_timestamp] = True
                signals.short_entries.loc[display_timestamp] = False  # Clear conflicts
                logger.debug(f"Trade {trade_idx}: LONG entry at {display_timestamp}")
            
            # Set entry price
            if entry_price is not None:
                signals.entry_prices.loc[display_timestamp] = entry_price
                
                # Extract risk levels associated with this entry
                self._extract_risk_levels(trade, signals, display_timestamp, entry_price, direction)
            
        except Exception as e:
            logger.warning(f"Failed to process entry for trade {trade_idx}: {e}")
    
    def _process_trade_exit(self, trade: pd.Series, trade_idx: int, signals: SignalFormat):
        """Process exit signal from trade record."""
        try:
            # Get and normalize exit timestamp
            exit_timestamp = self.timestamp_normalizer.normalize(trade.get('Exit Index'))
            if exit_timestamp is None or exit_timestamp not in signals.index:
                logger.debug(f"Trade {trade_idx}: Invalid exit timestamp")
                return
            
            # Calculate display timestamp
            display_timestamp = self.timing_calculator.calculate_display_timestamp(
                exit_timestamp, signals.index
            )
            
            if display_timestamp not in signals.index:
                logger.debug(f"Trade {trade_idx}: Exit display timestamp not in index")
                return
            
            # Extract exit price
            exit_price = self._extract_price(trade, [
                'Exit Price', 'Avg Exit Price', 'Exit Avg Price'
            ])
            
            # Determine direction for proper exit assignment
            direction = self._detect_trade_direction(trade)
            
            # Set appropriate exit signal
            if direction == 'short':
                signals.short_exits.loc[display_timestamp] = True
                logger.debug(f"Trade {trade_idx}: SHORT exit at {display_timestamp}")
            else:
                signals.long_exits.loc[display_timestamp] = True
                logger.debug(f"Trade {trade_idx}: LONG exit at {display_timestamp}")
            
            # Set exit price
            if exit_price is not None:
                signals.exit_prices.loc[display_timestamp] = exit_price
            
        except Exception as e:
            logger.warning(f"Failed to process exit for trade {trade_idx}: {e}")
    
    def _extract_from_orders(self, signals: SignalFormat) -> bool:
        """
        Extract additional signals from order records.
        
        This is typically used for risk management orders that may not
        appear in the trade records but are visible in order history.
        """
        try:
            if not hasattr(self.portfolio, 'orders') or len(self.portfolio.orders.records) == 0:
                logger.debug("No orders found in portfolio")
                return False
            
            orders = self.portfolio.orders.records_readable
            logger.debug(f"Processing {len(orders)} orders for additional signals")
            
            # Process only risk management orders to avoid duplication
            for order_idx, order in orders.iterrows():
                if self._is_risk_management_order(order):
                    self._process_risk_order(order, order_idx, signals)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract signals from orders: {e}")
            return False
    
    def _extract_risk_levels(
        self,
        trade: pd.Series,
        signals: SignalFormat,
        entry_timestamp: pd.Timestamp,
        entry_price: float,
        direction: str
    ):
        """
        Extract stop loss and take profit levels from trade record.
        
        This method tries multiple fields and calculation methods to
        extract SL/TP information associated with a trade.
        """
        try:
            # Try to extract direct price levels first
            sl_fields = ['Stop Loss', 'SL', 'Stop Price', 'SL Price']
            for field in sl_fields:
                if field in trade.index:
                    sl_value = trade.get(field)
                    if sl_value is not None and not pd.isna(sl_value) and sl_value > 0:
                        signals.sl_price_levels.loc[entry_timestamp] = float(sl_value)
                        logger.debug(f"Extracted SL price: {sl_value}")
                        break
            
            # Try to extract take profit levels
            tp_fields = ['Take Profit', 'TP', 'Profit Price', 'TP Price']
            for field in tp_fields:
                if field in trade.index:
                    tp_value = trade.get(field)
                    if tp_value is not None and not pd.isna(tp_value) and tp_value > 0:
                        signals.tp_price_levels.loc[entry_timestamp] = float(tp_value)
                        logger.debug(f"Extracted TP price: {tp_value}")
                        break
            
            # Try percentage-based levels
            sl_pct_fields = ['SL Pct', 'Stop Loss Pct', 'SL %']
            for field in sl_pct_fields:
                if field in trade.index:
                    sl_pct = trade.get(field)
                    if sl_pct is not None and not pd.isna(sl_pct):
                        signals.sl_levels.loc[entry_timestamp] = float(sl_pct)
                        logger.debug(f"Extracted SL percentage: {sl_pct}%")
                        break
            
            tp_pct_fields = ['TP Pct', 'Take Profit Pct', 'TP %']
            for field in tp_pct_fields:
                if field in trade.index:
                    tp_pct = trade.get(field)
                    if tp_pct is not None and not pd.isna(tp_pct):
                        signals.tp_levels.loc[entry_timestamp] = float(tp_pct)
                        logger.debug(f"Extracted TP percentage: {tp_pct}%")
                        break
            
        except Exception as e:
            logger.debug(f"Failed to extract risk levels: {e}")
    
    def _extract_price(self, record: pd.Series, price_columns: List[str]) -> Optional[float]:
        """Extract price from record with multiple column fallbacks."""
        for col in price_columns:
            if col in record.index:
                price = record.get(col)
                if price is not None and not pd.isna(price):
                    return float(price)
        return None
    
    def _detect_trade_direction(self, trade: pd.Series) -> str:
        """Enhanced trade direction detection with multiple methods."""
        # Method 1: Direction column
        direction = str(trade.get('Direction', '')).lower().strip()
        if direction:
            if 'short' in direction or direction == 'sell':
                return 'short'
            elif 'long' in direction or direction == 'buy':
                return 'long'
        
        # Method 2: Size column (negative = short)
        size = trade.get('Size', 0)
        if isinstance(size, (int, float)) and size < 0:
            return 'short'
        elif isinstance(size, (int, float)) and size > 0:
            return 'long'
        
        # Method 3: Side column
        side = str(trade.get('Side', '')).lower().strip()
        if side in ['short', 'sell', 'sell_to_open']:
            return 'short'
        elif side in ['long', 'buy', 'buy_to_open']:
            return 'long'
        
        # Default to long if direction cannot be determined
        logger.warning(f"Could not determine trade direction, defaulting to 'long'")
        return 'long'
    
    def _is_risk_management_order(self, order: pd.Series) -> bool:
        """Check if order is a risk management order (SL/TP)."""
        order_type = str(order.get('Type', '')).lower()
        stop_type = order.get('Stop Type')
        
        return (
            'stop' in order_type or
            stop_type is not None or
            ('tp' in str(stop_type).lower() if stop_type else False)
        )
    
    def _process_risk_order(self, order: pd.Series, order_idx: int, signals: SignalFormat):
        """Process a risk management order for additional exit signals."""
        # This could be implemented for more sophisticated order analysis
        # For now, we primarily rely on trade records
        pass
    
    def get_extraction_info(self) -> Dict[str, Any]:
        """Get information about the extraction process."""
        return self._extraction_info


class StrategySignalExtractor(ISignalExtractor):
    """
    Extracts and processes signals from strategy-generated dictionaries.
    
    This extractor handles signals that come directly from trading strategies,
    including legacy signal formats and unified signal formats.
    
    It also handles merging strategy signals with portfolio signals,
    giving priority to portfolio signals (what actually happened) over
    strategy signals (what was intended).
    
    Usage:
        extractor = StrategySignalExtractor(strategy_signals, data_index)
        signals = extractor.extract_signals()
    """
    
    def __init__(
        self,
        strategy_signals: Dict[str, Any],
        data_index: pd.Index,
        timing_config: Optional[TimingConfig] = None
    ):
        """
        Initialize strategy signal extractor.
        
        Args:
            strategy_signals: Dictionary of strategy-generated signals
            data_index: Data index for signal alignment
            timing_config: Optional timing configuration
        """
        self.strategy_signals = strategy_signals
        self.data_index = data_index
        self.timing_config = timing_config or TimingConfig()
        self._extraction_info = {}
    
    def extract_signals(self) -> SignalFormat:
        """
        Extract and convert strategy signals to unified format.
        
        Handles both legacy and unified signal formats from strategies.
        
        Returns:
            Unified signal format
        """
        logger.debug("Starting strategy signal extraction")
        
        if not self.strategy_signals:
            logger.debug("No strategy signals provided")
            return SignalFormat.empty(self.data_index)
        
        # Check if signals are in legacy format (unified 'exits')
        if 'exits' in self.strategy_signals and 'long_exits' not in self.strategy_signals:
            logger.debug("Converting legacy signal format")
            signals = convert_legacy_signals(self.strategy_signals, self.data_index)
        else:
            logger.debug("Using unified signal format")
            signals = SignalFormat.from_dict(self.strategy_signals, self.data_index)
        
        # Store extraction info
        self._extraction_info = {
            "format_type": "legacy" if 'exits' in self.strategy_signals else "unified",
            "signals_extracted": signals.get_summary(),
            "original_keys": list(self.strategy_signals.keys())
        }
        
        logger.info(f"Strategy extraction complete: {signals.get_summary()}")
        return signals
    
    def get_extraction_info(self) -> Dict[str, Any]:
        """Get information about the extraction process."""
        return self._extraction_info


class UnifiedSignalExtractor:
    """
    Unified signal extractor that combines portfolio and strategy signals.
    
    This is the main extractor that coordinates extraction from both
    portfolio records and strategy signals, handling priority and merging.
    
    Usage:
        extractor = UnifiedSignalExtractor(portfolio, data_processor, strategy_signals)
        signals = extractor.extract_all_signals()
    """
    
    def __init__(
        self,
        portfolio: vbt.Portfolio,
        data_processor: Any,
        strategy_signals: Optional[Dict[str, Any]] = None,
        timing_config: Optional[TimingConfig] = None
    ):
        """
        Initialize unified signal extractor.
        
        Args:
            portfolio: VectorBT Portfolio object
            data_processor: Data processor for index alignment
            strategy_signals: Optional strategy signals
            timing_config: Timing configuration
        """
        self.portfolio = portfolio
        self.data_processor = data_processor
        self.strategy_signals = strategy_signals or {}
        self.timing_config = timing_config or TimingConfig()
        
        # Initialize extractors
        self.portfolio_extractor = PortfolioSignalExtractor(
            portfolio, data_processor, timing_config
        )
        
        if strategy_signals:
            self.strategy_extractor = StrategySignalExtractor(
                strategy_signals, data_processor.get_ohlcv_data().index, timing_config
            )
        else:
            self.strategy_extractor = None
    
    def extract_all_signals(self) -> SignalFormat:
        """
        Extract signals from all sources and merge with proper priority.
        
        Priority order:
        1. Portfolio signals (highest - what actually happened)
        2. Strategy signals (lower - what was intended)
        
        Returns:
            Merged signal format
        """
        logger.debug("Starting unified signal extraction")
        
        # Extract portfolio signals (highest priority)
        portfolio_signals = self.portfolio_extractor.extract_signals()
        
        # If no strategy signals, return portfolio signals
        if not self.strategy_extractor:
            logger.debug("No strategy signals to merge")
            return portfolio_signals
        
        # Extract strategy signals
        strategy_signals = self.strategy_extractor.extract_signals()
        
        # Merge with portfolio priority
        merged_signals = self._merge_with_priority(portfolio_signals, strategy_signals)
        
        logger.info(f"Unified extraction complete: {merged_signals.get_summary()}")
        return merged_signals
    
    def _merge_with_priority(
        self,
        portfolio_signals: SignalFormat,
        strategy_signals: SignalFormat
    ) -> SignalFormat:
        """
        Merge signals with portfolio signals taking priority.
        
        Args:
            portfolio_signals: Signals from portfolio (highest priority)
            strategy_signals: Signals from strategy (lower priority)
            
        Returns:
            Merged signals
        """
        # Start with portfolio signals
        merged = portfolio_signals
        
        # Check if portfolio has meaningful signals
        has_portfolio_signals = (
            portfolio_signals.long_entries.any() or
            portfolio_signals.short_entries.any() or
            portfolio_signals.long_exits.any() or
            portfolio_signals.short_exits.any()
        )
        
        if not has_portfolio_signals:
            logger.debug("No portfolio signals found, using strategy signals")
            # Use strategy signals as base
            merged.long_entries = strategy_signals.long_entries
            merged.short_entries = strategy_signals.short_entries
            merged.long_exits = strategy_signals.long_exits
            merged.short_exits = strategy_signals.short_exits
            
            # Merge price information where available
            strategy_entry_mask = ~strategy_signals.entry_prices.isna()
            merged.entry_prices.loc[strategy_entry_mask] = strategy_signals.entry_prices.loc[strategy_entry_mask]
            
            strategy_exit_mask = ~strategy_signals.exit_prices.isna()
            merged.exit_prices.loc[strategy_exit_mask] = strategy_signals.exit_prices.loc[strategy_exit_mask]
        
        # Always merge SL/TP levels from strategy (regardless of signal priority)
        # These are important for visualization even if portfolio signals take precedence
        self._merge_risk_levels(merged, strategy_signals)
        
        return merged
    
    def _merge_risk_levels(self, target: SignalFormat, source: SignalFormat):
        """Merge stop loss and take profit levels."""
        # Merge SL price levels where target doesn't have them
        source_sl_mask = ~source.sl_price_levels.isna()
        target_sl_mask = ~target.sl_price_levels.isna()
        merge_mask = source_sl_mask & ~target_sl_mask
        
        if merge_mask.any():
            target.sl_price_levels.loc[merge_mask] = source.sl_price_levels.loc[merge_mask]
            logger.debug(f"Merged {merge_mask.sum()} SL price levels from strategy")
        
        # Merge TP price levels
        source_tp_mask = ~source.tp_price_levels.isna()
        target_tp_mask = ~target.tp_price_levels.isna()
        merge_mask = source_tp_mask & ~target_tp_mask
        
        if merge_mask.any():
            target.tp_price_levels.loc[merge_mask] = source.tp_price_levels.loc[merge_mask]
            logger.debug(f"Merged {merge_mask.sum()} TP price levels from strategy")
        
        # Merge percentage levels
        source_sl_pct_mask = ~source.sl_levels.isna()
        target_sl_pct_mask = ~target.sl_levels.isna()
        merge_mask = source_sl_pct_mask & ~target_sl_pct_mask
        
        if merge_mask.any():
            target.sl_levels.loc[merge_mask] = source.sl_levels.loc[merge_mask]
            logger.debug(f"Merged {merge_mask.sum()} SL percentage levels from strategy")
        
        source_tp_pct_mask = ~source.tp_levels.isna()
        target_tp_pct_mask = ~target.tp_levels.isna()
        merge_mask = source_tp_pct_mask & ~target_tp_pct_mask
        
        if merge_mask.any():
            target.tp_levels.loc[merge_mask] = source.tp_levels.loc[merge_mask]
            logger.debug(f"Merged {merge_mask.sum()} TP percentage levels from strategy")
    
    def get_extraction_info(self) -> Dict[str, Any]:
        """Get comprehensive extraction information."""
        info = {
            "portfolio_info": self.portfolio_extractor.get_extraction_info(),
            "timing_config": {
                "mode": self.timing_config.mode.value,
                "execution_delay": self.timing_config.execution_delay
            }
        }
        
        if self.strategy_extractor:
            info["strategy_info"] = self.strategy_extractor.get_extraction_info()
        
        return info


# Module exports
__all__ = [
    'ISignalExtractor',
    'PortfolioSignalExtractor',
    'StrategySignalExtractor',
    'UnifiedSignalExtractor'
]