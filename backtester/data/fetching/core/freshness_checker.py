#!/usr/bin/env python3
"""Freshness Checker - Handle data staleness detection and validation.

This module centralizes all logic related to checking if cached data is fresh
enough for use, handling timezone complexities and incomplete candles.
"""

import logging
from typing import List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import vectorbtpro as vbt
from .vbt_data_handler import VBTDataHandler

logger = logging.getLogger(__name__)


class FreshnessChecker:
    """Check and validate data freshness for cached and resampled data."""
    
    @staticmethod
    def parse_timeframe_minutes(timeframe: str) -> int:
        """Convert timeframe string to minutes."""
        tf = timeframe.lower().strip()
        
        try:
            if tf.endswith('s'):
                return int(tf[:-1]) / 60
            elif tf.endswith('m') or tf.endswith('min'):
                return int(tf[:-3] if tf.endswith('min') else tf[:-1])
            elif tf.endswith('h') or tf.endswith('hour'):
                hours = int(tf[:-4] if tf.endswith('hour') else tf[:-1])
                return hours * 60
            elif tf.endswith('d') or tf.endswith('day'):
                days = int(tf[:-3] if tf.endswith('day') else tf[:-1])
                return days * 24 * 60
            elif tf.endswith('w') or tf.endswith('week'):
                weeks = int(tf[:-4] if tf.endswith('week') else tf[:-1])
                return weeks * 7 * 24 * 60
            elif tf.endswith('M'):  # Month
                months = int(tf[:-1])
                return months * 30 * 24 * 60
            elif tf.endswith('y'):  # Year
                years = int(tf[:-1])
                return years * 365 * 24 * 60
            else:
                return 60  # Default to 1 hour
        except (ValueError, TypeError):
            return 60
    
    @staticmethod
    def get_expected_latest_candle(current_time: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        """
        Calculate the exact START time of the latest complete candle.
        
        This provides precise awareness of when candles should be available.
        """
        # Ensure we're working with UTC
        if current_time.tz is None:
            current_time = current_time.tz_localize('UTC')
        elif str(current_time.tz) != 'UTC':
            current_time = current_time.tz_convert('UTC')
        
        tf = timeframe.lower().strip()
        
        if tf.endswith('d') or 'day' in tf:
            # Daily candles complete at 00:00 UTC the next day
            # So if current time is 17:23 on May 24, latest complete is May 23 00:00
            days = int(tf[:-3] if tf.endswith('day') else tf[:-1]) if tf[:-1].isdigit() else 1
            
            # Get the start of today
            start_of_today = current_time.replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Latest complete candle starts (days) day(s) ago
            return start_of_today - timedelta(days=days)
            
        elif tf.endswith('h') or 'hour' in tf:
            # Hourly candles complete at the top of the next hour
            # So if current time is 17:23, latest complete 1h candle is 16:00-17:00 (starts at 16:00)
            hours = int(tf[:-4] if tf.endswith('hour') else tf[:-1]) if tf[:-1].isdigit() else 1
            
            # Calculate how many complete intervals have passed
            current_hour = current_time.hour
            current_minute = current_time.minute
            
            # For 1h: if it's 17:23, we're in the 17:00-18:00 candle which is incomplete
            # The latest complete candle is 16:00-17:00 (starts at 16:00)
            
            # Find the start of the current interval
            intervals_since_midnight = current_hour // hours
            current_interval_start_hour = intervals_since_midnight * hours
            
            # If we're exactly at the start of an interval (e.g., 17:00:00), 
            # check if we have enough time passed to consider it complete
            if current_minute == 0 and current_time.second == 0 and current_time.microsecond == 0:
                # We're exactly at interval boundary - current interval just started, so previous is complete
                latest_complete_hour = current_interval_start_hour - hours
            else:
                # We're inside an interval - current interval is incomplete, previous is complete
                latest_complete_hour = current_interval_start_hour - hours
            
            if latest_complete_hour < 0:
                # Go to previous day
                return (current_time - timedelta(days=1)).replace(
                    hour=24 + latest_complete_hour,
                    minute=0, second=0, microsecond=0
                )
            else:
                return current_time.replace(
                    hour=latest_complete_hour,
                    minute=0, second=0, microsecond=0
                )
                
        elif tf.endswith('m') or 'min' in tf:
            # Minute candles complete at the top of the next minute
            minutes = int(tf[:-3] if tf.endswith('min') else tf[:-1]) if tf[:-1].isdigit() else 1
            
            # Total minutes since start of day
            total_minutes = current_time.hour * 60 + current_time.minute
            
            # Find current interval
            intervals_since_midnight = total_minutes // minutes
            current_interval_start_minutes = intervals_since_midnight * minutes
            
            # If we're exactly at the start of an interval, check if we consider it complete
            if current_time.second == 0 and current_time.microsecond == 0:
                # At exact minute boundary - current interval just started
                latest_complete_minutes = current_interval_start_minutes - minutes
            else:
                # Inside an interval - current is incomplete
                latest_complete_minutes = current_interval_start_minutes - minutes
            
            if latest_complete_minutes < 0:
                # Previous day
                return (current_time - timedelta(days=1)).replace(
                    hour=23, minute=60 + latest_complete_minutes,
                    second=0, microsecond=0
                )
            else:
                return current_time.replace(
                    hour=latest_complete_minutes // 60,
                    minute=latest_complete_minutes % 60,
                    second=0, microsecond=0
                )
        
        else:
            # Default: assume 1 hour ago
            return current_time.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    
    @staticmethod
    def get_safe_end_time_for_fetch(current_time: pd.Timestamp, timeframe: str) -> pd.Timestamp:
        """
        Calculate a safe end time for fetching to avoid incomplete candles.
        
        Returns the end time of the latest complete candle (one period after its start).
        """
        latest_complete_start = FreshnessChecker.get_expected_latest_candle(current_time, timeframe)
        
        # Add one period to get the end time of the complete candle
        tf_minutes = FreshnessChecker.parse_timeframe_minutes(timeframe)
        
        return latest_complete_start + timedelta(minutes=tf_minutes)
    
    @staticmethod
    def is_data_fresh(
        data: vbt.Data,
        target_end: Optional[Union[str, pd.Timestamp]],
        timeframe: str
    ) -> bool:
        """
        Check if data is fresh enough for the target end date.
        
        For multi-symbol data, ALL symbols must be fresh.
        """
        if data is None or target_end is None:
            return True
            
        try:
            # Get target timestamp
            if isinstance(target_end, str):
                if target_end.lower() == "now":
                    target_dt = vbt.utc_timestamp()
                else:
                    target_dt = pd.Timestamp(target_end, tz='UTC')
            else:
                target_dt = pd.Timestamp(target_end)
                if target_dt.tz is None:
                    target_dt = target_dt.tz_localize('UTC')
                elif str(target_dt.tz) != 'UTC':
                    target_dt = target_dt.tz_convert('UTC')
            
            # Get expected latest candle time
            expected_latest = FreshnessChecker.get_expected_latest_candle(target_dt, timeframe)
            
            # Check if data is fresh
            if hasattr(data, 'symbols') and len(data.symbols) > 1:
                # Multi-symbol: check each symbol
                for symbol in data.symbols:
                    symbol_fresh = FreshnessChecker._is_symbol_fresh(
                        data, symbol, expected_latest, timeframe
                    )
                    if not symbol_fresh:
                        return False
                return True
            else:
                # Single symbol or overall check
                return FreshnessChecker._is_symbol_fresh(data, None, expected_latest, timeframe)
                
        except Exception as e:
            logger.debug(f"Error checking freshness: {e}")
            return True  # Assume fresh if can't determine
    
    @staticmethod
    def _is_symbol_fresh(
        data: vbt.Data,
        symbol: Optional[str],
        expected_latest: pd.Timestamp,
        timeframe: str
    ) -> bool:
        """Check if a specific symbol's data is fresh."""
        try:
            # Get symbol data
            if symbol:
                symbol_data = VBTDataHandler.get_symbol_data(data, symbol, 'close')
            else:
                # Use wrapper index for overall check
                if hasattr(data, 'close'):
                    symbol_data = data.close.dropna() if hasattr(data.close, 'dropna') else data.close
                else:
                    return True
            
            if symbol_data is None or len(symbol_data) == 0:
                return False
                
            # Get latest timestamp
            latest_ts = symbol_data.index[-1]
            if latest_ts.tz is None:
                latest_ts = latest_ts.tz_localize('UTC')
            elif str(latest_ts.tz) != 'UTC':
                latest_ts = latest_ts.tz_convert('UTC')
            
            # Check if we have the expected latest candle
            return latest_ts >= expected_latest
            
        except Exception as e:
            logger.debug(f"Error checking symbol freshness: {e}")
            return True
    
    @staticmethod
    def identify_stale_symbols(
        data: vbt.Data,
        symbols: List[str],
        target_end: Optional[Union[str, pd.Timestamp]],
        timeframe: str
    ) -> Tuple[List[str], List[str]]:
        """
        Identify which symbols are stale vs fresh.
        
        Returns:
            Tuple of (stale_symbols, fresh_symbols)
        """
        if data is None or target_end is None:
            return [], symbols
            
        try:
            # Get target timestamp
            if isinstance(target_end, str):
                if target_end.lower() == "now":
                    target_dt = vbt.utc_timestamp()
                else:
                    target_dt = pd.Timestamp(target_end, tz='UTC')
            else:
                target_dt = pd.Timestamp(target_end)
                if target_dt.tz is None:
                    target_dt = target_dt.tz_localize('UTC')
                elif str(target_dt.tz) != 'UTC':
                    target_dt = target_dt.tz_convert('UTC')
            
            expected_latest = FreshnessChecker.get_expected_latest_candle(target_dt, timeframe)
            
            stale_symbols = []
            fresh_symbols = []
            
            for symbol in symbols:
                if FreshnessChecker._is_symbol_fresh(data, symbol, expected_latest, timeframe):
                    fresh_symbols.append(symbol)
                else:
                    stale_symbols.append(symbol)
            
            return stale_symbols, fresh_symbols
            
        except Exception as e:
            logger.debug(f"Error identifying stale symbols: {e}")
            return symbols, []  # Assume all stale if error 