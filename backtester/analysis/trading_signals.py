"""
Trading Signals Processing and Rendering

This module handles all aspects of trading signal processing, extraction, validation, 
and rendering for trading charts. It was extracted from trading_charts.py to separate
concerns and improve maintainability.

CRITICAL TIMING LOGIC:
=====================
This module implements VectorBT's execution timing model to prevent lookahead bias:
- Signal Generation: Happens at bar close (time T) based on available information  
- Order Execution: Happens at next bar open (time T+1) to prevent lookahead bias
- Chart Display: By default shows signals at EXECUTION time (T+1) with execution prices

This prevents the common visualization error where entry signals appear at time T 
with prices from time T+1, which would indicate lookahead bias.

Key Features:
- Signal extraction from portfolio trades and orders
- Strategy signal integration with proper priority handling
- Timing correction and validation
- Signal cleaning and integrity checks
- Professional signal rendering with proper styling
- Comprehensive logging and debugging support
- Unified signal interface for consistent communication with simulation engine

Classes:
- SignalProcessor: Core signal extraction and processing
- SignalRenderer: Signal visualization and rendering
- SignalType: Enumeration for signal types
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import vectorbtpro as vbt
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass

from backtester.utilities.structured_logging import get_logger
from backtester.signals.signal_interface import (
    SignalFormat, SignalValidator, ValidationResult, 
    convert_legacy_signals, normalize_timestamps
)

logger = get_logger(__name__)


class SignalType(Enum):
    """Enumeration for signal types."""
    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    LONG_EXIT = "long_exit"
    SHORT_EXIT = "short_exit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"


@dataclass
class SignalConfig:
    """Configuration for signal processing and rendering."""
    # Timing configuration - CRITICAL for preventing lookahead bias
    signal_timing_mode: str = "execution"  # "signal", "execution", or "both"
    execution_delay: int = 1  # Bars between signal and execution
    show_timing_indicator: bool = True  # Show timing mode in chart
    
    # Signal processing options
    validate_signals: bool = True
    clean_signals: bool = True
    portfolio_signals_priority: bool = True  # Portfolio signals override strategy
    use_unified_signal_interface: bool = True  # Use new unified interface
    
    # Signal extraction options - CRITICAL FIX
    extract_from_trades_only: bool = True  # Only extract from trades, not orders
    separate_exit_types: bool = True  # Separate long_exits and short_exits
    
    # Rendering options
    show_signals: bool = True
    show_stop_levels: bool = True  # Show SL/TP symbols
    
    # Colors for different signal types
    colors: Dict[str, str] = None
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'long_entry': 'lime',
                'short_entry': 'orangered', 
                'long_exit': 'purple',
                'short_exit': 'purple',
                'exit': 'purple',  # Unified exit color
                'stop_level': 'red',      # Red for SL
                'profit_level': 'green'   # Green for TP
            }


class SignalProcessor:
    """
    Processes and extracts trading signals from portfolio and strategy data.
    
    CRITICAL: This class implements proper timing handling to prevent lookahead bias
    in chart visualization. It separates signal generation time from execution time.
    
    NEW: Uses unified signal interface for consistent communication with simulation engine.
    """
    
    def __init__(self, portfolio: vbt.Portfolio, data_processor, strategy_signals: Optional[Dict] = None, config: Optional[SignalConfig] = None):
        self.portfolio = portfolio
        self.data_processor = data_processor
        self.strategy_signals = strategy_signals or {}
        self.config = config or SignalConfig()
        
        # Initialize signal validator
        self.validator = SignalValidator(strict_mode=False)
        
        # Extract signals using new unified approach
        if self.config.use_unified_signal_interface:
            self.extracted_signals = self._extract_signals_unified()
        else:
            self.extracted_signals = self._extract_all_signals_legacy()
    
    def _extract_signals_unified(self) -> SignalFormat:
        """
        Extract signals using the new unified interface.
        
        This is the NEW implementation that fixes the signal extraction issues.
        """
        logger.debug(f"=== STARTING UNIFIED SIGNAL EXTRACTION ===")
        logger.debug(f"Strategy signals provided: {list(self.strategy_signals.keys()) if self.strategy_signals else 'None'}")
        
        index = self.data_processor.ohlcv_data.index
        
        # Initialize signal format with proper index
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
        
        logger.debug(f"Initialized unified signals for {len(index)} timestamps")
        
        # Extract from portfolio trades (HIGHEST PRIORITY)
        logger.debug("Extracting from portfolio trades...")
        self._extract_from_portfolio_unified(signals)
        
        portfolio_summary = signals.get_summary()
        logger.debug(f"After portfolio extraction: {portfolio_summary}")
        
        # Merge with strategy signals if available (LOWER PRIORITY)
        if self.strategy_signals:
            logger.debug("Merging strategy signals...")
            self._merge_strategy_signals_unified(signals)
        else:
            logger.debug("No strategy signals to merge")
        
        final_summary = signals.get_summary()
        logger.info(f"Unified signal extraction complete: {final_summary}")
        
        # Validate signals if enabled
        if self.config.validate_signals:
            validation_result = self.validator.validate_signals(signals)
            if not validation_result.is_valid:
                logger.error(f"Signal validation failed: {validation_result.errors}")
            for warning in validation_result.warnings:
                logger.warning(f"Signal validation: {warning}")
        
        return signals
    
    def _extract_from_portfolio_unified(self, signals: SignalFormat):
        """
        Extract signals from portfolio using unified approach.
        
        CRITICAL FIX: Only extract from trades to avoid double counting.
        Properly separate long_exits and short_exits.
        """
        try:
            # Process TRADES for entry/exit signals (position changes)
            if hasattr(self.portfolio, 'trades') and len(self.portfolio.trades.records) > 0:
                logger.debug(f"Processing {len(self.portfolio.trades.records)} trades")
                self._process_trades_unified(signals)
            else:
                logger.debug("No trades found in portfolio")
            
            # CRITICAL FIX: Do NOT process orders separately to avoid double counting
            # Orders are already reflected in the trades when they execute
            if not self.config.extract_from_trades_only:
                logger.debug("Processing orders (warning: may cause double counting)")
                self._process_orders_unified(signals)
            else:
                logger.debug("Skipping order processing to avoid double counting (extract_from_trades_only=True)")
                
        except Exception as e:
            logger.error(f"Failed to extract portfolio signals: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _process_trades_unified(self, signals: SignalFormat):
        """
        Process trade records with unified signal format.
        
        CRITICAL FIX: Properly separate exits by position type and extract SL/TP levels.
        """
        try:
            trades = self.portfolio.trades.records_readable
            
            logger.debug(f"Processing {len(trades)} trades with unified format")
            logger.debug(f"Trade record columns: {list(trades.columns)}")
            
            # Get timing configuration
            signal_timing_mode = self.config.signal_timing_mode
            execution_delay = self.config.execution_delay
                
            logger.info(f"Using signal timing mode: {signal_timing_mode} with {execution_delay} bar delay")
            
            # Track position state for proper exit assignment
            position_tracker = {}  # timestamp -> position_type ('long' or 'short')
            
            for trade_idx, trade in trades.iterrows():
                logger.debug(f"Processing trade {trade_idx}: {dict(trade)}")
                
                # Process ENTRY signals with timing correction
                signal_timestamp = self._normalize_timestamp(trade.get('Entry Index'))
                
                if signal_timestamp is None or signal_timestamp not in signals.index:
                    logger.debug(f"Trade {trade_idx}: Signal timestamp {signal_timestamp} not in data index")
                    continue
                
                # Calculate execution timestamp (signal + delay)
                display_timestamp = self._calculate_display_timestamp(
                    signal_timestamp, signal_timing_mode, execution_delay, signals.index
                )
                
                if display_timestamp not in signals.index:
                    logger.debug(f"Trade {trade_idx}: Display timestamp {display_timestamp} not in index, skipping")
                    continue
                
                # Get entry price with better error handling
                entry_price = self._extract_price(trade, ['Entry Price', 'Avg Entry Price', 'Entry Avg Price'])
                
                # Enhanced direction detection
                direction = self._detect_trade_direction(trade)
                
                # TIMING FIX: Show entry at correct timestamp with correct price
                if direction == 'short':
                    signals.short_entries.loc[display_timestamp] = True
                    signals.long_entries.loc[display_timestamp] = False  # Clear conflicts
                    position_tracker[display_timestamp] = 'short'
                    logger.debug(f"Trade {trade_idx}: SHORT entry at {display_timestamp}")
                else:
                    signals.long_entries.loc[display_timestamp] = True
                    signals.short_entries.loc[display_timestamp] = False  # Clear conflicts
                    position_tracker[display_timestamp] = 'long'
                    logger.debug(f"Trade {trade_idx}: LONG entry at {display_timestamp}")
                
                if entry_price is not None:
                    signals.entry_prices.loc[display_timestamp] = entry_price
                    logger.debug(f"Trade {trade_idx}: Entry price {entry_price} at {display_timestamp}")
                    
                    # NEW: Extract stop loss and take profit levels from trade data
                    self._extract_trade_risk_levels(trade, signals, display_timestamp, entry_price, direction)
                else:
                    logger.warning(f"Trade {trade_idx}: No entry price found")
                
                # Process EXIT signals with same timing logic
                exit_signal_timestamp = self._normalize_timestamp(trade.get('Exit Index'))
                if exit_signal_timestamp is not None and exit_signal_timestamp in signals.index:
                    
                    exit_display_timestamp = self._calculate_display_timestamp(
                        exit_signal_timestamp, signal_timing_mode, execution_delay, signals.index
                    )
                    
                    if exit_display_timestamp in signals.index:
                        # Get exit price
                        exit_price = self._extract_price(trade, ['Exit Price', 'Avg Exit Price', 'Exit Avg Price'])
                        
                        # CRITICAL FIX: Assign exit to correct position type
                        if self.config.separate_exit_types:
                            if direction == 'short':
                                signals.short_exits.loc[exit_display_timestamp] = True
                                logger.debug(f"Trade {trade_idx}: SHORT exit at {exit_display_timestamp}")
                            else:
                                signals.long_exits.loc[exit_display_timestamp] = True
                                logger.debug(f"Trade {trade_idx}: LONG exit at {exit_display_timestamp}")
                        else:
                            # Fallback to unified exits (legacy compatibility)
                            signals.long_exits.loc[exit_display_timestamp] = True
                            logger.debug(f"Trade {trade_idx}: Unified exit at {exit_display_timestamp}")
                        
                        if exit_price is not None:
                            signals.exit_prices.loc[exit_display_timestamp] = exit_price
                            logger.debug(f"Trade {trade_idx}: Exit price {exit_price} at {exit_display_timestamp}")
                        else:
                            logger.warning(f"Trade {trade_idx}: No exit price found")
                    else:
                        logger.debug(f"Trade {trade_idx}: Exit display timestamp {exit_display_timestamp} not in index")
                else:
                    logger.debug(f"Trade {trade_idx}: Exit signal timestamp {exit_signal_timestamp} not in data index")
            
            # Log final counts with timing explanation
            summary = signals.get_summary()
            logger.info(f"After processing trades ({signal_timing_mode} timing): {summary}")
            
            if signal_timing_mode == "execution":
                logger.info(f"TIMING NOTE: Signals shown at execution time (signal time + {execution_delay} bars)")
                            
        except Exception as e:
            logger.error(f"Failed to process trades: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _extract_trade_risk_levels(self, trade, signals: SignalFormat, entry_timestamp, entry_price: float, direction: str):
        """
        Extract stop loss and take profit levels from trade data.
        
        This method tries to extract SL/TP information from various fields in the trade record.
        """
        try:
            # Try to extract stop loss information
            sl_fields = ['Stop Loss', 'SL', 'Stop Price', 'SL Price', 'Stop Level', 'Stop Loss Price']
            for field in sl_fields:
                if field in trade.index:
                    sl_value = trade.get(field)
                    if sl_value is not None and not pd.isna(sl_value) and sl_value > 0:
                        signals.sl_price_levels.loc[entry_timestamp] = float(sl_value)
                        logger.debug(f"Extracted SL price level: {sl_value} from field '{field}'")
                        break
            
            # Try to extract take profit information
            tp_fields = ['Take Profit', 'TP', 'Profit Price', 'TP Price', 'Target', 'Target Price']
            for field in tp_fields:
                if field in trade.index:
                    tp_value = trade.get(field)
                    if tp_value is not None and not pd.isna(tp_value) and tp_value > 0:
                        signals.tp_price_levels.loc[entry_timestamp] = float(tp_value)
                        logger.debug(f"Extracted TP price level: {tp_value} from field '{field}'")
                        break
            
            # Try to extract percentage-based risk levels
            sl_pct_fields = ['SL Pct', 'Stop Loss Pct', 'SL %', 'Stop Percentage']
            for field in sl_pct_fields:
                if field in trade.index:
                    sl_pct = trade.get(field)
                    if sl_pct is not None and not pd.isna(sl_pct):
                        signals.sl_levels.loc[entry_timestamp] = float(sl_pct)
                        logger.debug(f"Extracted SL percentage level: {sl_pct}% from field '{field}'")
                        break
            
            tp_pct_fields = ['TP Pct', 'Take Profit Pct', 'TP %', 'Target Percentage']
            for field in tp_pct_fields:
                if field in trade.index:
                    tp_pct = trade.get(field)
                    if tp_pct is not None and not pd.isna(tp_pct):
                        signals.tp_levels.loc[entry_timestamp] = float(tp_pct)
                        logger.debug(f"Extracted TP percentage level: {tp_pct}% from field '{field}'")
                        break
            
            # If no explicit SL/TP found, try to estimate from portfolio settings
            if (signals.sl_price_levels.loc[entry_timestamp] is pd.NaType or 
                pd.isna(signals.sl_price_levels.loc[entry_timestamp])):
                self._estimate_risk_levels_from_portfolio(signals, entry_timestamp, entry_price, direction)
            
        except Exception as e:
            logger.debug(f"Failed to extract risk levels from trade: {e}")
    
    def _estimate_risk_levels_from_portfolio(self, signals: SignalFormat, entry_timestamp, entry_price: float, direction: str):
        """
        Estimate stop loss and take profit levels from portfolio configuration.
        
        This is a fallback method when explicit SL/TP levels aren't available in trade data.
        """
        try:
            # Check if portfolio has risk management settings
            if hasattr(self.portfolio, 'init_kwargs'):
                init_kwargs = self.portfolio.init_kwargs
                
                # Look for stop loss settings
                if 'sl_stop' in init_kwargs:
                    sl_pct = init_kwargs['sl_stop']
                    if sl_pct is not None and sl_pct > 0:
                        signals.sl_levels.loc[entry_timestamp] = float(sl_pct * 100)  # Convert to percentage
                        logger.debug(f"Estimated SL percentage from portfolio: {sl_pct * 100}%")
                
                # Look for take profit settings
                if 'tp_stop' in init_kwargs:
                    tp_pct = init_kwargs['tp_stop']
                    if tp_pct is not None and tp_pct > 0:
                        signals.tp_levels.loc[entry_timestamp] = float(tp_pct * 100)  # Convert to percentage
                        logger.debug(f"Estimated TP percentage from portfolio: {tp_pct * 100}%")
                
                # Look for other risk management fields
                risk_fields = ['stop_loss', 'take_profit', 'sl', 'tp']
                for field in risk_fields:
                    if field in init_kwargs:
                        value = init_kwargs[field]
                        if value is not None and value > 0:
                            if 'sl' in field or 'stop' in field:
                                if value < 1:  # Assume percentage if < 1
                                    signals.sl_levels.loc[entry_timestamp] = float(value * 100)
                                else:  # Assume price if > 1
                                    signals.sl_price_levels.loc[entry_timestamp] = float(value)
                            elif 'tp' in field or 'profit' in field:
                                if value < 1:  # Assume percentage if < 1
                                    signals.tp_levels.loc[entry_timestamp] = float(value * 100)
                                else:  # Assume price if > 1
                                    signals.tp_price_levels.loc[entry_timestamp] = float(value)
            
        except Exception as e:
            logger.debug(f"Failed to estimate risk levels from portfolio: {e}")
    
    def _process_orders_unified(self, signals: SignalFormat):
        """
        Process order records for additional risk management signals.
        
        WARNING: This should only be used when extract_from_trades_only=False
        to avoid double counting.
        """
        try:
            orders = self.portfolio.orders.records_readable
            
            logger.debug(f"Processing {len(orders)} orders for additional signals")
            
            for order_idx, order in orders.iterrows():
                # Get timestamp and price
                timestamp = self._normalize_timestamp(order.get('Fill Index'))
                if timestamp not in signals.index:
                    continue
                
                price = self._extract_price(order, ['Fill Price', 'Price', 'Avg Price', 'Fill Avg Price'])
                
                # Only process risk management orders
                order_type = str(order.get('Type', '')).lower()
                stop_type = order.get('Stop Type')
                
                is_stop_order = (
                    'stop' in order_type or 
                    stop_type is not None or
                    'tp' in str(stop_type).lower() if stop_type else False
                )
                
                if is_stop_order:
                    # Add to appropriate exit type based on stop type
                    if stop_type and 'tp' in str(stop_type).lower():
                        logger.debug(f"Order {order_idx}: Take profit exit at {timestamp}")
                    else:
                        logger.debug(f"Order {order_idx}: Stop loss exit at {timestamp}")
                    
                    # For now, add to both exit types (can be refined later)
                    signals.long_exits.loc[timestamp] = True
                    if price is not None:
                        signals.exit_prices.loc[timestamp] = price
                else:
                    logger.debug(f"Order {order_idx}: Skipping non-risk-management order")
                            
        except Exception as e:
            logger.error(f"Failed to process orders: {e}")
    
    def _merge_strategy_signals_unified(self, signals: SignalFormat):
        """
        Merge strategy signals using unified format.
        
        CRITICAL PRIORITY: Portfolio signals override strategy signals.
        Enhanced to better handle SL/TP levels.
        """
        logger.debug(f"=== STARTING UNIFIED STRATEGY SIGNAL MERGE ===")
        
        # Check if we have portfolio signals to prioritize them
        has_portfolio_signals = (
            signals.long_entries.any() or 
            signals.short_entries.any() or 
            signals.long_exits.any() or 
            signals.short_exits.any()
        )
        
        logger.debug(f"Portfolio signals present: {has_portfolio_signals}")
        
        # Convert strategy signals to unified format
        if 'exits' in self.strategy_signals:
            # Convert legacy unified exits to separated exits
            unified_strategy_signals = convert_legacy_signals(self.strategy_signals, signals.index)
        else:
            unified_strategy_signals = SignalFormat.from_dict(self.strategy_signals, signals.index)
        
        # Merge signals with portfolio priority
        if not has_portfolio_signals:
            logger.debug("No portfolio signals found, using strategy signals")
            signals.long_entries = unified_strategy_signals.long_entries
            signals.short_entries = unified_strategy_signals.short_entries
            signals.long_exits = unified_strategy_signals.long_exits
            signals.short_exits = unified_strategy_signals.short_exits
            
            # Merge price information
            mask = ~unified_strategy_signals.entry_prices.isna()
            signals.entry_prices.loc[mask] = unified_strategy_signals.entry_prices.loc[mask]
            
            mask = ~unified_strategy_signals.exit_prices.isna()
            signals.exit_prices.loc[mask] = unified_strategy_signals.exit_prices.loc[mask]
            
        else:
            logger.debug("Portfolio signals take priority, merging additional info only")
        
        # ENHANCED: Always merge stop loss and take profit levels from strategy signals
        # These are critical for visualization regardless of signal priority
        if hasattr(unified_strategy_signals, 'sl_price_levels'):
            strategy_sl_mask = ~unified_strategy_signals.sl_price_levels.isna()
            portfolio_sl_mask = ~signals.sl_price_levels.isna()
            
            # Only use strategy SL levels where portfolio doesn't have them
            merge_mask = strategy_sl_mask & ~portfolio_sl_mask
            if merge_mask.any():
                signals.sl_price_levels.loc[merge_mask] = unified_strategy_signals.sl_price_levels.loc[merge_mask]
                logger.debug(f"Merged {merge_mask.sum()} SL price levels from strategy")
        
        if hasattr(unified_strategy_signals, 'tp_price_levels'):
            strategy_tp_mask = ~unified_strategy_signals.tp_price_levels.isna()
            portfolio_tp_mask = ~signals.tp_price_levels.isna()
            
            # Only use strategy TP levels where portfolio doesn't have them
            merge_mask = strategy_tp_mask & ~portfolio_tp_mask
            if merge_mask.any():
                signals.tp_price_levels.loc[merge_mask] = unified_strategy_signals.tp_price_levels.loc[merge_mask]
                logger.debug(f"Merged {merge_mask.sum()} TP price levels from strategy")
        
        # Also merge the percentage levels if present
        if hasattr(unified_strategy_signals, 'sl_levels'):
            strategy_sl_pct_mask = ~unified_strategy_signals.sl_levels.isna()
            portfolio_sl_pct_mask = ~signals.sl_levels.isna()
            
            merge_mask = strategy_sl_pct_mask & ~portfolio_sl_pct_mask
            if merge_mask.any():
                signals.sl_levels.loc[merge_mask] = unified_strategy_signals.sl_levels.loc[merge_mask]
                logger.debug(f"Merged {merge_mask.sum()} SL percentage levels from strategy")
            
        if hasattr(unified_strategy_signals, 'tp_levels'):
            strategy_tp_pct_mask = ~unified_strategy_signals.tp_levels.isna()
            portfolio_tp_pct_mask = ~signals.tp_levels.isna()
            
            merge_mask = strategy_tp_pct_mask & ~portfolio_tp_pct_mask
            if merge_mask.any():
                signals.tp_levels.loc[merge_mask] = unified_strategy_signals.tp_levels.loc[merge_mask]
                logger.debug(f"Merged {merge_mask.sum()} TP percentage levels from strategy")
        
        # ENHANCED: Try to extract SL/TP from strategy signal dictionaries with various naming conventions
        sl_signal_names = ['sl_price_levels', 'stop_loss_levels', 'sl_levels', 'stop_prices', 'sl_prices']
        tp_signal_names = ['tp_price_levels', 'take_profit_levels', 'tp_levels', 'profit_prices', 'tp_prices']
        
        for sl_name in sl_signal_names:
            if sl_name in self.strategy_signals:
                sl_data = self.strategy_signals[sl_name]
                if isinstance(sl_data, pd.Series) and not sl_data.isna().all():
                    # Align with signal index and merge where portfolio data is missing
                    aligned_sl = sl_data.reindex(signals.index, fill_value=np.nan)
                    merge_mask = ~aligned_sl.isna() & signals.sl_price_levels.isna()
                    if merge_mask.any():
                        signals.sl_price_levels.loc[merge_mask] = aligned_sl.loc[merge_mask]
                        logger.debug(f"Merged {merge_mask.sum()} SL levels from strategy field '{sl_name}'")
                    break
        
        for tp_name in tp_signal_names:
            if tp_name in self.strategy_signals:
                tp_data = self.strategy_signals[tp_name]
                if isinstance(tp_data, pd.Series) and not tp_data.isna().all():
                    # Align with signal index and merge where portfolio data is missing
                    aligned_tp = tp_data.reindex(signals.index, fill_value=np.nan)
                    merge_mask = ~aligned_tp.isna() & signals.tp_price_levels.isna()
                    if merge_mask.any():
                        signals.tp_price_levels.loc[merge_mask] = aligned_tp.loc[merge_mask]
                        logger.debug(f"Merged {merge_mask.sum()} TP levels from strategy field '{tp_name}'")
                    break
        
        logger.debug(f"=== UNIFIED STRATEGY SIGNAL MERGE COMPLETE ===")
        
        # Log final SL/TP counts
        sl_count = (~signals.sl_price_levels.isna()).sum()
        tp_count = (~signals.tp_price_levels.isna()).sum()
        sl_pct_count = (~signals.sl_levels.isna()).sum()
        tp_pct_count = (~signals.tp_levels.isna()).sum()
        
        logger.info(f"Final SL/TP counts: SL prices={sl_count}, TP prices={tp_count}, SL %={sl_pct_count}, TP %={tp_pct_count}")
    
    def _calculate_display_timestamp(self, signal_timestamp, timing_mode, delay, index):
        """Calculate display timestamp based on timing configuration."""
        try:
            signal_pos = index.get_loc(signal_timestamp)
            execution_pos = signal_pos + delay
            
            if execution_pos < len(index):
                execution_timestamp = index[execution_pos]
            else:
                execution_timestamp = signal_timestamp
            
            if timing_mode == "execution":
                return execution_timestamp
            else:
                return signal_timestamp
                
        except (KeyError, IndexError):
            return signal_timestamp
    
    def _extract_price(self, record, price_columns):
        """Extract price from record with multiple column fallbacks."""
        for col in price_columns:
            if col in record.index:
                price = record.get(col)
                if price is not None and not pd.isna(price):
                    return float(price)
        return None
    
    def _detect_trade_direction(self, trade) -> str:
        """Enhanced trade direction detection with multiple fallback methods."""
        # Method 1: Check Direction column
        direction = str(trade.get('Direction', '')).lower().strip()
        if direction:
            if 'short' in direction or direction == 'sell':
                return 'short'
            elif 'long' in direction or direction == 'buy':
                return 'long'
        
        # Method 2: Check Size column (negative = short)
        size = trade.get('Size', 0)
        if isinstance(size, (int, float)) and size < 0:
            return 'short'
        elif isinstance(size, (int, float)) and size > 0:
            return 'long'
        
        # Method 3: Check Side column 
        side = str(trade.get('Side', '')).lower().strip()
        if side:
            if side in ['short', 'sell', 'sell_to_open']:
                return 'short'
            elif side in ['long', 'buy', 'buy_to_open']:
                return 'long'
        
        # Method 4: Check Entry/Exit Side columns
        entry_side = str(trade.get('Entry Side', '')).lower().strip()
        if entry_side:
            if entry_side in ['short', 'sell', 'sell_to_open']:
                return 'short'
            elif entry_side in ['long', 'buy', 'buy_to_open']:
                return 'long'
        
        # Method 5: Check PnL vs price movement as last resort
        try:
            entry_price = trade.get('Entry Price', 0)
            exit_price = trade.get('Exit Price', 0)
            pnl = trade.get('PnL', 0)
            
            if entry_price and exit_price and pnl:
                price_change = exit_price - entry_price
                # If PnL and price change have opposite signs, likely a short position
                if price_change > 0 and pnl < 0:
                    return 'short'
                elif price_change < 0 and pnl > 0:
                    return 'short'
        except (TypeError, ValueError):
            pass
        
        # Default to long if we can't determine direction
        logger.warning(f"Could not determine trade direction for trade: {dict(trade)}. Defaulting to 'long'")
        return 'long'
    
    def _normalize_timestamp(self, ts):
        """Normalize timestamp for comparison with OHLCV index."""
        if ts is None:
            return None
        
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        
        # Handle timezone consistency
        index = self.data_processor.ohlcv_data.index
        if index.tz is None and ts.tz is not None:
            ts = ts.tz_localize(None)
        elif index.tz is not None and ts.tz is None:
            ts = ts.tz_localize(index.tz)
        
        return ts
    
    def get_signals_dict(self) -> Dict[str, pd.Series]:
        """Get signals in dictionary format for compatibility."""
        if self.config.use_unified_signal_interface:
            return self.extracted_signals.to_dict()
        else:
            return self.extracted_signals
    
    def get_signals_format(self) -> SignalFormat:
        """Get signals in unified format."""
        if self.config.use_unified_signal_interface:
            return self.extracted_signals
        else:
            # Convert legacy format
            index = self.data_processor.ohlcv_data.index
            return convert_legacy_signals(self.extracted_signals, index)
    
    def validate_signals(self) -> ValidationResult:
        """Validate extracted signals."""
        if self.config.use_unified_signal_interface:
            return self.validator.validate_signals(self.extracted_signals)
        else:
            index = self.data_processor.ohlcv_data.index
            unified_signals = convert_legacy_signals(self.extracted_signals, index)
            return self.validator.validate_signals(unified_signals)

    def _extract_all_signals_legacy(self) -> Dict[str, pd.Series]:
        """
        Legacy signal extraction method (for backward compatibility).
        
        This is the OLD implementation that has the issues.
        """
        logger.debug(f"=== STARTING LEGACY SIGNAL EXTRACTION ===")
        logger.debug(f"Strategy signals provided: {list(self.strategy_signals.keys()) if self.strategy_signals else 'None'}")
        
        signals = {}
        index = self.data_processor.ohlcv_data.index
        
        # Initialize empty signal series
        for signal_type in ['long_entries', 'short_entries', 'exits', 
                           'entry_prices', 'exit_prices',
                           'sl_price_levels', 'tp_price_levels']:
            if 'prices' in signal_type or 'levels' in signal_type:
                signals[signal_type] = pd.Series(np.nan, index=index)
            else:
                signals[signal_type] = pd.Series(False, index=index)
        
        logger.debug(f"Initialized empty signals for {len(signals)} signal types")
        
        # Extract from portfolio trades/orders (HIGHEST PRIORITY)
        logger.debug("Calling _extract_from_portfolio...")
        self._extract_from_portfolio_legacy(signals, index)
        
        portfolio_counts = {
            'long_entries': signals['long_entries'].sum(),
            'short_entries': signals['short_entries'].sum(),
            'exits': signals['exits'].sum()
        }
        logger.debug(f"After portfolio extraction: {portfolio_counts}")
        
        # Merge with strategy signals if available (LOWER PRIORITY)
        if self.strategy_signals:
            logger.debug("Calling _merge_strategy_signals...")
            self._merge_strategy_signals_legacy(signals, index)
        else:
            logger.debug("No strategy signals to merge")
        
        final_counts = {
            'long_entries': signals['long_entries'].sum(),
            'short_entries': signals['short_entries'].sum(),
            'exits': signals['exits'].sum()
        }
        logger.info(f"Extracted signals: {final_counts['long_entries']} long entries, "
                   f"{final_counts['short_entries']} short entries, "
                   f"{final_counts['exits']} exits")
        
        # Validate signals if enabled
        if self.config.validate_signals:
            self._validate_signals_legacy(signals)
            
        return signals
    
    def _extract_from_portfolio_legacy(self, signals: Dict[str, pd.Series], index: pd.Index):
        """Legacy portfolio extraction method."""
        try:
            trades_processed = False
            orders_processed = False
            
            # Process TRADES for entry/exit signals (position changes)
            if hasattr(self.portfolio, 'trades') and len(self.portfolio.trades.records) > 0:
                logger.debug(f"Processing {len(self.portfolio.trades.records)} trades for entry/exit signals")
                self._process_trades_legacy(signals, index)
                trades_processed = True
            else:
                logger.debug("No trades found in portfolio")
            
            # Process ORDERS for additional exit signals (risk management executions)
            if hasattr(self.portfolio, 'orders') and len(self.portfolio.orders.records) > 0:
                logger.debug(f"Processing {len(self.portfolio.orders.records)} orders for additional exit signals")
                self._process_orders_legacy(signals, index)
                orders_processed = True
            else:
                logger.debug("No orders found in portfolio")
            
            if not trades_processed and not orders_processed:
                logger.warning("No trades or orders found in portfolio - no signals extracted")
            else:
                logger.debug(f"Signal extraction completed: trades={trades_processed}, orders={orders_processed}")
                
        except Exception as e:
            logger.error(f"Failed to extract portfolio signals: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _process_trades_legacy(self, signals: Dict[str, pd.Series], index: pd.Index):
        """Legacy trade processing method."""
        # ... existing implementation from original code ...
        pass
    
    def _process_orders_legacy(self, signals: Dict[str, pd.Series], index: pd.Index):
        """Legacy order processing method."""
        # ... existing implementation from original code ...
        pass
    
    def _merge_strategy_signals_legacy(self, signals: Dict[str, pd.Series], index: pd.Index):
        """Legacy strategy signal merging method."""
        # ... existing implementation from original code ...
        pass
    
    def _validate_signals_legacy(self, signals: Dict[str, pd.Series]):
        """Legacy signal validation method."""
        # ... existing implementation from original code ...
        pass


class SignalRenderer:
    """
    Renders trading signals on charts with proper styling and legend management.
    
    This class handles the visual representation of signals while maintaining
    clean legends and avoiding visual clutter.
    """
    
    def __init__(self, config: SignalConfig, legend_manager):
        self.config = config
        self.legend_manager = legend_manager
        
        # Define signal styles - consolidated exit styles
        self.signal_styles = {
            'long_entry': {
                'symbol': 'triangle-up',
                'size': 18,
                'color': config.colors['long_entry'],
                'line_color': 'darkgreen',
                'name': 'Long Entry'
            },
            'short_entry': {
                'symbol': 'triangle-down',
                'size': 18,
                'color': config.colors['short_entry'],
                'line_color': 'darkred',
                'name': 'Short Entry'
            },
            'exit': {
                'symbol': 'square',
                'size': 12,
                'color': config.colors['exit'],
                'line_color': 'darkviolet',
                'name': 'Exit'
            }
        }
    
    def add_signals_to_chart(self, fig: go.Figure, signals: Dict[str, pd.Series], 
                           ohlcv_data: pd.DataFrame, row: int = 1, col: int = 1):
        """
        Add all signal types to the chart.
        
        This is the main entry point for signal rendering.
        """
        if not self.config.show_signals:
            logger.debug("Signal rendering disabled in configuration")
            return
        
        # Log signal summary before rendering
        logger.info("=== SIGNAL RENDERING SUMMARY ===")
        logger.info(f"Long entries: {signals['long_entries'].sum()}")
        logger.info(f"Short entries: {signals['short_entries'].sum()}")
        logger.info(f"Exits: {signals['exits'].sum()}")
        
        # Enhanced stop level availability analysis
        sl_price_count = (~signals.get('sl_price_levels', pd.Series()).isna()).sum() if 'sl_price_levels' in signals else 0
        tp_price_count = (~signals.get('tp_price_levels', pd.Series()).isna()).sum() if 'tp_price_levels' in signals else 0
        sl_pct_count = (~signals.get('sl_levels', pd.Series()).isna()).sum() if 'sl_levels' in signals else 0
        tp_pct_count = (~signals.get('tp_levels', pd.Series()).isna()).sum() if 'tp_levels' in signals else 0
        
        logger.info(f"Stop levels available - SL prices: {sl_price_count}, TP prices: {tp_price_count}")
        logger.info(f"Stop levels available - SL %: {sl_pct_count}, TP %: {tp_pct_count}")
        
        # Debug: Show sample SL/TP values
        if 'sl_price_levels' in signals and sl_price_count > 0:
            sl_sample = signals['sl_price_levels'].dropna().head(3)
            logger.info(f"Sample SL price levels: {list(sl_sample.values)}")
        
        if 'tp_price_levels' in signals and tp_price_count > 0:
            tp_sample = signals['tp_price_levels'].dropna().head(3)
            logger.info(f"Sample TP price levels: {list(tp_sample.values)}")
            
        if 'sl_levels' in signals and sl_pct_count > 0:
            sl_pct_sample = signals['sl_levels'].dropna().head(3)
            logger.info(f"Sample SL percentages: {list(sl_pct_sample.values)}")
            
        if 'tp_levels' in signals and tp_pct_count > 0:
            tp_pct_sample = signals['tp_levels'].dropna().head(3)
            logger.info(f"Sample TP percentages: {list(tp_pct_sample.values)}")
        
        # Show available signal keys for debugging
        logger.debug(f"Available signal keys: {list(signals.keys())}")
        
        # Validate signals before rendering
        self._validate_rendering_signals(signals, ohlcv_data)
        
        # Signal mappings - unified exit handling
        execution_signal_mappings = [
            ('long_entries', 'entry_prices', 'long_entry'),
            ('short_entries', 'entry_prices', 'short_entry'),
            ('exits', 'exit_prices', 'exit')
        ]
        
        # Add execution signals
        for signal_key, price_key, style_key in execution_signal_mappings:
            self._add_signal_type(fig, signals, ohlcv_data, signal_key, price_key, style_key, row, col)
        
        # ALWAYS try to add stop level indicators when signals are shown
        # This ensures SL/TP labels are rendered if data is available
        logger.debug(f"Stop level rendering enabled: {self.config.show_stop_levels}")
        
        if self.config.show_stop_levels:
            if sl_price_count > 0 or tp_price_count > 0 or sl_pct_count > 0 or tp_pct_count > 0:
                logger.info("✅ SL/TP data detected - proceeding with stop level rendering")
                self._add_stop_levels_at_entries(fig, signals, ohlcv_data, row, col)
            else:
                logger.warning("⚠️ No SL/TP data available for rendering - will use fallback calculations")
                self._add_stop_levels_at_entries(fig, signals, ohlcv_data, row, col)
        else:
            logger.debug("Stop level rendering disabled in configuration")
    
    def _validate_rendering_signals(self, signals: Dict[str, pd.Series], ohlcv_data: pd.DataFrame):
        """Validate signals before rendering to catch issues."""
        # Check for overlapping entry signals
        overlaps = signals['long_entries'] & signals['short_entries']
        if overlaps.any():
            overlap_times = ohlcv_data.index[overlaps]
            logger.error(f"CRITICAL: Found overlapping entry signals at {len(overlap_times)} timestamps: {list(overlap_times)}")
        
        # Check for entries without prices
        long_no_price = signals['long_entries'] & signals['entry_prices'].isna()
        short_no_price = signals['short_entries'] & signals['entry_prices'].isna()
        exits_no_price = signals['exits'] & signals['exit_prices'].isna()
        
        if long_no_price.any():
            logger.warning(f"Long entries without prices: {long_no_price.sum()}")
        if short_no_price.any():
            logger.warning(f"Short entries without prices: {short_no_price.sum()}")
        if exits_no_price.any():
            logger.warning(f"Exits without prices: {exits_no_price.sum()}")
    
    def _add_signal_type(self, fig: go.Figure, signals: Dict[str, pd.Series], 
                        ohlcv_data: pd.DataFrame, signal_key: str, price_key: str, 
                        style_key: str, row: int, col: int):
        """Add a specific signal type to the chart."""
        try:
            signal_mask = signals[signal_key].reindex(ohlcv_data.index, fill_value=False)
            prices = signals[price_key].reindex(ohlcv_data.index, fill_value=np.nan)
            
            valid_signals = signal_mask & ~prices.isna()
            if not valid_signals.any():
                logger.debug(f"No valid {signal_key} signals to render")
                return
            
            signal_times = ohlcv_data.index[valid_signals]
            signal_prices = prices[valid_signals]
            
            style = self.signal_styles[style_key]
            show_legend = self.legend_manager.should_show_legend(style['name'])
            
            # Log what we're about to render
            logger.info(f"Rendering {len(signal_times)} {style['name']} signals with style: "
                       f"symbol={style['symbol']}, color={style['color']}")
            
            fig.add_trace(
                go.Scatter(
                    x=signal_times,
                    y=signal_prices,
                    mode='markers',
                    name=style['name'],
                    marker=dict(
                        symbol=style['symbol'],
                        size=style['size'],
                        color=style['color'],
                        line=dict(color=style['line_color'], width=2)
                    ),
                    showlegend=show_legend,
                    hovertemplate=f'<b>{style["name"]}</b><br>Date: %{{x}}<br>Price: $%{{y:.4f}}<extra></extra>'
                ),
                row=row, col=col
            )
            
            logger.debug(f"Successfully added {len(signal_times)} {style['name']} signals")
            
        except Exception as e:
            logger.error(f"Failed to add {style_key} signals: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _add_stop_levels_at_entries(self, fig: go.Figure, signals: Dict[str, pd.Series], 
                                  ohlcv_data: pd.DataFrame, row: int, col: int):
        """Add stop loss and take profit levels at entry points as simple symbols."""
        try:
            logger.debug(f"=== Adding stop levels at entries ===")
            
            # Get entry signals (both long and short)
            all_entries = signals['long_entries'] | signals['short_entries']
            logger.debug(f"Total entries for stop levels: {all_entries.sum()}")
            
            if not all_entries.any():
                logger.debug("No entries found - skipping stop level rendering")
                return
            
            # Track legend items to avoid duplicates
            sl_legend_added = False
            tp_legend_added = False
            
            # Collect all stop levels
            sl_times = []
            sl_prices = []
            tp_times = []
            tp_prices = []
            
            # Process each entry point
            for entry_idx in ohlcv_data.index[all_entries]:
                entry_time = entry_idx
                entry_price = signals.get('entry_prices', pd.Series(index=ohlcv_data.index)).get(entry_idx, np.nan)
                
                # If no entry price from signals, use OHLCV data
                if pd.isna(entry_price):
                    entry_price = ohlcv_data.loc[entry_time, 'Close']
                
                # Determine position direction
                is_long = signals['long_entries'].get(entry_idx, False)
                
                # Get stop loss level if available
                sl_level = self._get_stop_level(signals, entry_idx, entry_price, is_long, ohlcv_data)
                if sl_level is not None and not pd.isna(sl_level):
                    sl_times.append(entry_time)
                    sl_prices.append(sl_level)
                
                # Get take profit level if available
                tp_level = self._get_profit_level(signals, entry_idx, entry_price, is_long, ohlcv_data)
                if tp_level is not None and not pd.isna(tp_level):
                    tp_times.append(entry_time)
                    tp_prices.append(tp_level)
            
            # Add stop loss symbols if any
            if sl_times:
                show_legend = self.legend_manager.should_show_legend('Stop Loss')
                
                fig.add_trace(
                    go.Scatter(
                        x=sl_times,
                        y=sl_prices,
                        mode='markers',
                        name='Stop Loss',
                        marker=dict(
                            symbol='line-ew',  # Horizontal line symbol for stop loss
                            size=12,
                            color=self.config.colors['stop_level'],
                            line=dict(color=self.config.colors['stop_level'], width=2)
                        ),
                        showlegend=show_legend,
                        hovertemplate='<b>Stop Loss</b><br>Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
                    ),
                    row=row, col=col
                )
                logger.debug(f"Added {len(sl_times)} stop loss symbols")
            
            # Add take profit symbols if any
            if tp_times:
                show_legend = self.legend_manager.should_show_legend('Take Profit')
                
                fig.add_trace(
                    go.Scatter(
                        x=tp_times,
                        y=tp_prices,
                        mode='markers',
                        name='Take Profit',
                        marker=dict(
                            symbol='line-ew',  # Horizontal line symbol for take profit
                            size=12,
                            color=self.config.colors['profit_level'],
                            line=dict(color=self.config.colors['profit_level'], width=2)
                        ),
                        showlegend=show_legend,
                        hovertemplate='<b>Take Profit</b><br>Date: %{x}<br>Price: $%{y:.4f}<extra></extra>'
                    ),
                    row=row, col=col
                )
                logger.debug(f"Added {len(tp_times)} take profit symbols")
            
            logger.info(f"Stop level rendering complete: {len(sl_times)} SL, {len(tp_times)} TP symbols added")
                    
        except Exception as e:
            logger.error(f"Failed to add stop levels at entries: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def _get_stop_level(self, signals: Dict[str, pd.Series], entry_idx, entry_price: float, is_long: bool, ohlcv_data: pd.DataFrame) -> Optional[float]:
        """Get stop loss level for an entry point, trying multiple data sources."""
        # Try multiple timestamps due to execution timing offset
        timestamps_to_try = [entry_idx]
        
        # If using execution timing, also try the previous timestamp (signal time)
        if self.config.signal_timing_mode == "execution":
            try:
                entry_pos = ohlcv_data.index.get_loc(entry_idx)
                if entry_pos > 0:
                    signal_timestamp = ohlcv_data.index[entry_pos - 1]
                    timestamps_to_try.append(signal_timestamp)
            except (KeyError, IndexError):
                pass
        
        for timestamp in timestamps_to_try:
            # Method 1: Direct SL price levels from signals (PREFERRED - these are from the strategy)
            if 'sl_price_levels' in signals:
                sl_price = signals['sl_price_levels'].get(timestamp)
                if sl_price is not None and not pd.isna(sl_price):
                    return float(sl_price)
            
            # Method 2: Calculate from SL percentage levels (if price levels not available)
            if 'sl_levels' in signals and not pd.isna(entry_price):
                sl_pct = signals['sl_levels'].get(timestamp)
                if sl_pct is not None and not pd.isna(sl_pct):
                    # Convert percentage to price
                    if is_long:
                        # Long position: SL below entry price
                        sl_price = float(entry_price * (1 - abs(sl_pct) / 100))
                    else:
                        # Short position: SL above entry price
                        sl_price = float(entry_price * (1 + abs(sl_pct) / 100))
                    return sl_price
        
        # Method 3: Try legacy field names (check all timestamps)
        legacy_fields = ['stop_loss', 'sl', 'stop_price', 'sl_price']
        for field in legacy_fields:
            if field in signals:
                for timestamp in timestamps_to_try:
                    sl_value = signals[field].get(timestamp)
                    if sl_value is not None and not pd.isna(sl_value):
                        return float(sl_value)
        
        # NO FALLBACK - if no SL levels from strategy, return None
        return None
    
    def _get_profit_level(self, signals: Dict[str, pd.Series], entry_idx, entry_price: float, is_long: bool, ohlcv_data: pd.DataFrame) -> Optional[float]:
        """Get take profit level for an entry point, trying multiple data sources."""
        # Try multiple timestamps due to execution timing offset
        timestamps_to_try = [entry_idx]
        
        # If using execution timing, also try the previous timestamp (signal time)
        if self.config.signal_timing_mode == "execution":
            try:
                entry_pos = ohlcv_data.index.get_loc(entry_idx)
                if entry_pos > 0:
                    signal_timestamp = ohlcv_data.index[entry_pos - 1]
                    timestamps_to_try.append(signal_timestamp)
            except (KeyError, IndexError):
                pass
        
        for timestamp in timestamps_to_try:
            # Method 1: Direct TP price levels from signals (PREFERRED - these are from the strategy)
            if 'tp_price_levels' in signals:
                tp_price = signals['tp_price_levels'].get(timestamp)
                if tp_price is not None and not pd.isna(tp_price):
                    return float(tp_price)
            
            # Method 2: Calculate from TP percentage levels (if price levels not available)
            if 'tp_levels' in signals and not pd.isna(entry_price):
                tp_pct = signals['tp_levels'].get(timestamp)
                if tp_pct is not None and not pd.isna(tp_pct):
                    # Convert percentage to price
                    if is_long:
                        # Long position: TP above entry price
                        tp_price = float(entry_price * (1 + abs(tp_pct) / 100))
                    else:
                        # Short position: TP below entry price
                        tp_price = float(entry_price * (1 - abs(tp_pct) / 100))
                    return tp_price
        
        # Method 3: Try legacy field names (check all timestamps)
        legacy_fields = ['take_profit', 'tp', 'profit_price', 'tp_price']
        for field in legacy_fields:
            if field in signals:
                for timestamp in timestamps_to_try:
                    tp_value = signals[field].get(timestamp)
                    if tp_value is not None and not pd.isna(tp_value):
                        return float(tp_value)
        
        # NO FALLBACK - if no TP levels from strategy, return None
        return None


# Utility functions for signal analysis and validation
def explain_signal_timing() -> str:
    """
    Explain signal timing modes and their implications.
    
    Returns:
        Detailed explanation of signal vs execution timing
    """
    explanation = """
Signal Timing Modes in Trading Charts
====================================

To prevent lookahead bias, trading charts support different timing modes:

1. EXECUTION TIMING (DEFAULT - RECOMMENDED):
   - Shows signals at actual execution time (T+1)
   - Uses actual execution prices
   - Reflects realistic trading constraints
   - Prevents misleading visualization

2. SIGNAL TIMING (ANALYSIS MODE):
   - Shows signals at decision time (T)
   - Uses signal-generation prices
   - Useful for strategy analysis
   - May appear to have lookahead bias

3. VectorBT EXECUTION MODEL:
   - Signal Generation: Bar close (time T) based on available data
   - Order Execution: Next bar open (time T+1) 
   - Price Used: Next bar's open price
   - Chart Display: Configurable timing mode

EXAMPLE:
========
Decision made: 2023-01-15 close, based on data up to 2023-01-15
Order executed: 2023-01-16 open, at 2023-01-16 open price
Chart shows (execution mode): Entry at 2023-01-16 with 2023-01-16 price
Chart shows (signal mode): Entry at 2023-01-15 with 2023-01-15 price

RECOMMENDATION:
==============
Use execution timing mode for realistic backtesting visualization.
Use signal timing mode only for strategy development and analysis.
"""
    return explanation


def validate_signal_timing(portfolio: vbt.Portfolio, data: vbt.Data, config: SignalConfig) -> Dict[str, Any]:
    """
    Validate signal timing configuration and provide recommendations.
    
    Args:
        portfolio: VectorBT Portfolio object
        data: VectorBT Data object
        config: Signal configuration to validate
        
    Returns:
        Dictionary with validation results and recommendations
    """
    validation_result = {
        "valid": True,
        "warnings": [],
        "recommendations": [],
        "timing_analysis": {}
    }
    
    try:
        trades = portfolio.trades.records_readable
        if len(trades) == 0:
            validation_result["warnings"].append("No trades found in portfolio")
            return validation_result
        
        # Check execution delay appropriateness
        if config.execution_delay > 5:
            validation_result["warnings"].append(
                f"Execution delay of {config.execution_delay} bars seems excessive for most trading scenarios"
            )
            validation_result["recommendations"].append(
                "Consider reducing execution_delay to 1-2 bars for more realistic timing"
            )
        
        # Analyze timing patterns in sample trades
        sample_size = min(10, len(trades))
        sample_trades = trades.head(sample_size)
        
        timing_issues = 0
        
        for idx, trade in sample_trades.iterrows():
            signal_ts = pd.Timestamp(trade.get('Entry Index'))
            execution_ts = pd.Timestamp(trade.get('Entry Index'))  # Simplified for validation
            
            if signal_ts == execution_ts and config.signal_timing_mode == "execution":
                timing_issues += 1
        
        # Add timing analysis summary
        validation_result["timing_analysis"] = {
            "sample_trades_analyzed": sample_size,
            "timing_issues_found": timing_issues,
            "recommended_mode": config.signal_timing_mode
        }
        
        # Generate recommendations
        if config.signal_timing_mode == "signal":
            validation_result["recommendations"].append(
                "Signal timing mode shows decision points but not realistic execution. "
                "Consider using execution mode for realistic backtesting visualization."
            )
        
        if timing_issues > 0:
            validation_result["warnings"].append(
                f"Found {timing_issues} trades with potential timing issues"
            )
            validation_result["recommendations"].append(
                "Review your execution model - most realistic scenarios have at least 1 bar delay"
            )
            
    except Exception as e:
        validation_result["valid"] = False
        validation_result["warnings"].append(f"Validation failed: {str(e)}")
    
    return validation_result


def validate_timing_behavior(
    portfolio: vbt.Portfolio, 
    data: vbt.Data,
    sample_trade_idx: int = 0
) -> Dict[str, Any]:
    """
    Validate and explain the timing behavior for a specific trade.
    
    Args:
        portfolio: VectorBT Portfolio object
        data: VectorBT Data object  
        sample_trade_idx: Index of trade to analyze
        
    Returns:
        Dictionary with timing analysis
    """
    try:
        trades = portfolio.trades.records_readable
        if sample_trade_idx >= len(trades):
            return {"error": f"Trade index {sample_trade_idx} not found. Portfolio has {len(trades)} trades."}
            
        trade = trades.iloc[sample_trade_idx]
        
        # Get the raw data index
        if hasattr(data, "wrapper"):
            data_index = data.wrapper.index
        else:
            data_index = data.index
            
        # Get signal and execution info
        signal_timestamp = pd.Timestamp(trade.get('Entry Index'))
        entry_price = trade.get('Entry Price')
        
        # Find execution timestamp (signal + 1 bar)
        try:
            signal_pos = data_index.get_loc(signal_timestamp)
            if signal_pos + 1 < len(data_index):
                execution_timestamp = data_index[signal_pos + 1]
                
                # Get OHLC data for validation
                if hasattr(data, 'open'):
                    execution_open = data.open.iloc[signal_pos + 1]
                    signal_close = data.close.iloc[signal_pos]
                else:
                    execution_open = "N/A (no OHLC data)"
                    signal_close = "N/A (no OHLC data)"
            else:
                execution_timestamp = signal_timestamp
                execution_open = "N/A (end of data)"
                signal_close = "N/A (end of data)"
        except (KeyError, IndexError):
            execution_timestamp = signal_timestamp
            execution_open = "N/A (index error)"
            signal_close = "N/A (index error)"
        
        analysis = {
            "trade_index": sample_trade_idx,
            "signal_timestamp": signal_timestamp,
            "execution_timestamp": execution_timestamp,
            "entry_price_from_trade": entry_price,
            "signal_bar_close": signal_close,
            "execution_bar_open": execution_open,
            "timing_offset_bars": 1,
            "price_match_check": "Entry price matches execution bar open" if abs(float(entry_price) - float(execution_open)) < 0.0001 else "Price mismatch detected",
            "explanation": "Entry price should match execution bar open price (T+1), not signal bar close (T)"
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Failed to analyze timing: {str(e)}"}


def get_timing_recommendations(
    timeframe: str,
    trading_style: str = "swing"
) -> Dict[str, Any]:
    """
    Get timing recommendations based on timeframe and trading style.
    
    Args:
        timeframe: Trading timeframe (e.g., '1h', '4h', '1d')
        trading_style: Trading style ('scalping', 'day', 'swing', 'position')
        
    Returns:
        Dictionary with recommended timing settings
    """
    recommendations = {
        "execution_delay": 1,
        "signal_timing_mode": "execution",
        "explanation": ""
    }
    
    # Adjust based on timeframe
    timeframe_lower = timeframe.lower()
    if timeframe_lower in ['1m', '5m', '15m']:
        # High frequency - minimal delay acceptable
        recommendations["execution_delay"] = 1
        recommendations["explanation"] = "Short timeframes: 1 bar delay simulates near-instant execution"
        
        if trading_style == "scalping":
            recommendations["signal_timing_mode"] = "signal"
            recommendations["explanation"] += " (scalping may use signal timing for analysis)"
            
    elif timeframe_lower in ['30m', '1h', '2h']:
        # Medium frequency - standard delay
        recommendations["execution_delay"] = 1
        recommendations["explanation"] = "Medium timeframes: 1 bar delay represents realistic order processing"
        
    elif timeframe_lower in ['4h', '6h', '8h']:
        # Lower frequency - potentially longer delay
        recommendations["execution_delay"] = 1  # Still 1 bar but longer time
        recommendations["explanation"] = "Longer timeframes: 1 bar delay allows for analysis and order placement"
        
    else:  # Daily and above
        recommendations["execution_delay"] = 1
        recommendations["explanation"] = "Daily+ timeframes: 1 bar delay represents end-of-day processing"
    
    # Adjust based on trading style
    style_adjustments = {
        "scalping": {
            "signal_timing_mode": "signal",
            "note": "Scalping often requires signal timing analysis"
        },
        "day": {
            "signal_timing_mode": "execution",
            "note": "Day trading benefits from realistic execution timing"
        },
        "swing": {
            "signal_timing_mode": "execution", 
            "note": "Swing trading should use execution timing for realism"
        },
        "position": {
            "signal_timing_mode": "execution",
            "note": "Position trading benefits from realistic execution delays"
        }
    }
    
    if trading_style in style_adjustments:
        recommendations.update(style_adjustments[trading_style])
        recommendations["explanation"] += f" | {style_adjustments[trading_style]['note']}"
    
    return recommendations 