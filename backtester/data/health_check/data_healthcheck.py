#!/usr/bin/env python3
"""Data Health Check Script

Analyzes cached VBT data for quality issues including:
- Date gaps and missing time periods
- Inception coverage (missing early historical data)
- Data completeness (NaN values, missing columns)
- Symbol-specific issues
- Timeframe consistency

Auto-fix capabilities:
- Re-fetch data for critical gaps
- Update data to current time
- Fill missing inception data

Usage:
    python -m backtester.data.health_check.data_healthcheck [options]
    
Examples:
    # Check all data
    python -m backtester.data.health_check.data_healthcheck
    
    # Check specific exchange with auto-fix
    python -m backtester.data.health_check.data_healthcheck --exchange binance --auto-fix
    
    # Check specific timeframe with detailed output
    python -m backtester.data.health_check.data_healthcheck --timeframe 1h --detailed
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess

# Add parent directories to path to import backtester modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtester.data.storage.data_storage import data_storage
from backtester.data.cache_system.cache_manager import cache_manager
import vectorbtpro as vbt

class DataHealthChecker:
    """Comprehensive data health checker for VBT cached data."""
    
    def __init__(self, detailed: bool = False, auto_fix: bool = False, reports_dir: str = None):
        self.detailed = detailed
        self.auto_fix = auto_fix
        self.issues = {
            'critical': [],
            'warning': [],
            'info': []
        }
        self.fixes_applied = []
        
        # Set reports directory
        if reports_dir is None:
            # Default to reports subfolder in this module
            self.reports_dir = Path(__file__).parent / 'reports'
        else:
            self.reports_dir = Path(reports_dir)
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Load cached inception dates from cache system
        self.cached_inception_dates = {}
        self._load_cached_inception_dates()
        
        # Fallback inception dates for major symbols (if not in cache)
        self.fallback_inception_dates = {
            'BTC/USDT': '2017-08-17',
            'ETH/USDT': '2017-08-17', 
            'BNB/USDT': '2017-11-06',
            'ADA/USDT': '2018-04-17',
            'XRP/USDT': '2018-05-04',
            'USDC/USDT': '2018-12-15',
            'DOGE/USDT': '2019-07-05',
            'SOL/USDT': '2020-08-11',
            'AVAX/USDT': '2020-09-22',
            'PEPE/USDT': '2023-05-05',
            'SUI/USDT': '2023-05-03',
            'WLD/USDT': '2023-07-24',
            'FDUSD/USDT': '2023-07-26',
            'WIF/USDT': '2024-03-05',
            'ENA/USDT': '2024-04-02',
            'NEIRO/USDT': '2024-09-16',
            'CETUS/USDT': '2024-11-06',
            'PNUT/USDT': '2024-11-11',
            'TRUMP/USDT': '2025-01-19'
        }
    
    def _load_cached_inception_dates(self):
        """Load inception dates from the cache system."""
        try:
            # Get all exchanges with cached timestamp data
            for exchange in cache_manager.list_exchanges():
                timestamps = cache_manager.get_all_timestamps(exchange)
                if timestamps:
                    # Convert millisecond timestamps to datetime strings
                    self.cached_inception_dates[exchange] = {}
                    for symbol, timestamp_ms in timestamps.items():
                        try:
                            # Convert from milliseconds to datetime
                            dt = datetime.fromtimestamp(timestamp_ms / 1000)
                            self.cached_inception_dates[exchange][symbol] = dt.strftime('%Y-%m-%d')
                        except (ValueError, TypeError) as e:
                            print(f"Warning: Invalid timestamp for {exchange}:{symbol}: {timestamp_ms}")
                    
                    print(f"📅 Loaded {len(self.cached_inception_dates[exchange])} inception dates for {exchange}")
        except Exception as e:
            print(f"Warning: Could not load cached inception dates: {e}")
    
    def get_inception_date(self, exchange: str, symbol: str) -> Optional[str]:
        """Get inception date for a symbol, preferring cached data."""
        # First try cached data
        if exchange in self.cached_inception_dates and symbol in self.cached_inception_dates[exchange]:
            return self.cached_inception_dates[exchange][symbol]
        
        # Fall back to hardcoded dates
        return self.fallback_inception_dates.get(symbol)
    
    def add_issue(self, severity: str, category: str, message: str, details: Dict[str, Any] = None):
        """Add an issue to the findings."""
        issue = {
            'category': category,
            'message': message,
            'details': details or {},
            'timestamp': datetime.now().isoformat()
        }
        self.issues[severity].append(issue)
    
    def apply_fix(self, fix_type: str, fix_command: str, description: str) -> bool:
        """Apply a fix by running a command."""
        if not self.auto_fix:
            return False
            
        try:
            print(f"🔧 Applying fix: {description}")
            # Set working directory to the project root directory
            script_dir = os.path.dirname(os.path.abspath(__file__))  # .../backtester/data/health_check
            backtester_root = os.path.dirname(os.path.dirname(script_dir))  # .../backtester
            project_root = os.path.dirname(backtester_root)  # .../
            
            result = subprocess.run(fix_command, shell=True, capture_output=True, text=True, cwd=project_root)
            
            if result.returncode == 0:
                self.fixes_applied.append({
                    'type': fix_type,
                    'command': fix_command,
                    'description': description,
                    'success': True,
                    'output': result.stdout
                })
                print(f"   ✅ Fix applied successfully")
                return True
            else:
                self.fixes_applied.append({
                    'type': fix_type,
                    'command': fix_command,
                    'description': description,
                    'success': False,
                    'error': result.stderr
                })
                print(f"   ❌ Fix failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   ❌ Error applying fix: {e}")
            return False
    
    def check_date_gaps(self, data: vbt.Data, filename: str, expected_freq: str, exchange: str = None, market_type: str = None) -> List[Dict]:
        """Check for gaps in the date sequence."""
        gaps = []
        
        try:
            # Get the index (timestamps)
            index = data.wrapper.index
            
            if len(index) < 2:
                self.add_issue('warning', 'insufficient_data', 
                             f"{filename}: Less than 2 data points", 
                             {'data_points': len(index)})
                return gaps
            
            # Convert expected frequency to pandas frequency
            freq_map = {
                '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
                '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
                '1d': '1D', '3d': '3D', '1w': '1W', '1M': '1M'
            }
            
            pandas_freq = freq_map.get(expected_freq, expected_freq)
            
            # Create expected date range
            expected_range = pd.date_range(
                start=index[0], 
                end=index[-1], 
                freq=pandas_freq
            )
            
            # Find missing dates
            missing_dates = expected_range.difference(index)
            
            if len(missing_dates) > 0:
                # Group consecutive missing dates into gaps
                if len(missing_dates) > 0:
                    missing_df = pd.DataFrame({'missing': missing_dates}).sort_values('missing')
                    missing_df['group'] = (missing_df['missing'].diff() > pd.Timedelta(pandas_freq)).cumsum()
                    
                    for group_id, group in missing_df.groupby('group'):
                        gap_start = group['missing'].iloc[0]
                        gap_end = group['missing'].iloc[-1]
                        gap_count = len(group)
                        
                        gap_info = {
                            'start': gap_start,
                            'end': gap_end,
                            'duration': gap_end - gap_start,
                            'missing_points': gap_count,
                            'frequency': expected_freq
                        }
                        gaps.append(gap_info)
                        
                        severity = 'critical' if gap_count > 24 else 'warning'
                        self.add_issue(severity, 'date_gaps',
                                     f"{filename}: Data gap from {gap_start} to {gap_end} ({gap_count} missing points)",
                                     gap_info)
                        
                        # Auto-fix for critical gaps
                        if severity == 'critical' and exchange and market_type:
                            symbols = list(data.symbols) if hasattr(data, 'symbols') else []
                            if symbols:
                                # Construct fix command to re-fetch the gap period
                                symbol_list = ','.join(symbols[:5])  # Limit to first 5 symbols
                                fix_cmd = f"python backtester/scripts/fetch_data_cli.py --exchange {exchange} --market {market_type} --timeframe {expected_freq} --symbols {symbol_list} --start {gap_start.strftime('%Y-%m-%d')} --end {gap_end.strftime('%Y-%m-%d')}"
                                self.apply_fix('gap_fill', fix_cmd, f"Fill data gap from {gap_start.date()} to {gap_end.date()}")
            
            # Check for duplicate timestamps
            duplicates = index.duplicated()
            if duplicates.any():
                dup_count = duplicates.sum()
                dup_dates = index[duplicates].unique()
                self.add_issue('critical', 'duplicate_timestamps',
                             f"{filename}: {dup_count} duplicate timestamps found",
                             {'duplicate_dates': [str(d) for d in dup_dates[:10]]})  # Show first 10
                             
        except Exception as e:
            self.add_issue('critical', 'analysis_error',
                         f"{filename}: Error checking date gaps: {e}")
        
        return gaps
    
    def check_data_freshness(self, data: vbt.Data, filename: str, exchange: str = None, market_type: str = None, timeframe: str = None) -> Dict:
        """Check if data is up-to-date (latest candle should be recent)."""
        freshness_info = {}
        
        try:
            data_end = data.wrapper.index[-1]
            now = pd.Timestamp.now(tz=data_end.tz if data_end.tz else 'UTC')
            
            # Calculate time difference
            time_diff = now - data_end
            hours_behind = time_diff.total_seconds() / 3600
            
            freshness_info = {
                'latest_timestamp': data_end,
                'current_time': now,
                'hours_behind': hours_behind
            }
            
            # Determine if data is stale
            if timeframe:
                # Define acceptable staleness based on timeframe
                staleness_thresholds = {
                    '1m': 2, '5m': 6, '15m': 12, '30m': 24,
                    '1h': 3, '2h': 6, '4h': 12, '6h': 18, '8h': 24, '12h': 36,
                    '1d': 48, '3d': 96, '1w': 168
                }
                
                threshold = staleness_thresholds.get(timeframe, 24)  # Default to 24 hours
                
                if hours_behind > threshold:
                    severity = 'critical' if hours_behind > threshold * 2 else 'warning'
                    self.add_issue(severity, 'stale_data',
                                 f"{filename}: Data is {hours_behind:.1f} hours behind (threshold: {threshold}h)",
                                 freshness_info)
                    
                    # Auto-fix for stale data
                    if severity == 'critical' and exchange and market_type and timeframe:
                        symbols = list(data.symbols) if hasattr(data, 'symbols') else []
                        if symbols:
                            symbol_list = ','.join(symbols[:10])  # Limit to first 10 symbols
                            fix_cmd = f"python backtester/scripts/fetch_data_cli.py --exchange {exchange} --market {market_type} --timeframe {timeframe} --symbols {symbol_list} --end now"
                            self.apply_fix('update_stale', fix_cmd, f"Update stale data (behind by {hours_behind:.1f} hours)")
                            
        except Exception as e:
            self.add_issue('critical', 'analysis_error',
                         f"{filename}: Error checking data freshness: {e}")
        
        return freshness_info
    
    def check_inception_coverage(self, data: vbt.Data, filename: str, exchange: str = None, market_type: str = None, timeframe: str = None) -> Dict:
        """Check if data covers expected inception dates using cached data."""
        inception_issues = {}
        
        try:
            data_start = data.wrapper.index[0]
            
            # Extract symbols from data
            symbols = data.symbols if hasattr(data, 'symbols') else []
            
            for symbol in symbols:
                inception_date_str = self.get_inception_date(exchange, symbol)
                if inception_date_str:
                    expected_inception = pd.Timestamp(inception_date_str)
                    
                    # Ensure both timestamps are timezone-aware for comparison
                    if data_start.tz is None and expected_inception.tz is not None:
                        data_start = data_start.tz_localize('UTC')
                    elif data_start.tz is not None and expected_inception.tz is None:
                        expected_inception = expected_inception.tz_localize('UTC')
                    
                    # Allow some tolerance (e.g., within a week)
                    tolerance = pd.Timedelta(days=7)
                    
                    if data_start > (expected_inception + tolerance):
                        missing_days = (data_start - expected_inception).days
                        inception_issues[symbol] = {
                            'expected_inception': expected_inception,
                            'actual_start': data_start,
                            'missing_days': missing_days
                        }
                        
                        severity = 'critical' if missing_days > 365 else 'warning'  # More than a year is critical
                        self.add_issue(severity, 'inception_coverage',
                                     f"{filename}: {symbol} missing {missing_days} days from inception "
                                     f"(expected: {expected_inception.date()}, actual: {data_start.date()})",
                                     inception_issues[symbol])
                        
                        # Auto-fix for missing inception data
                        if severity == 'critical' and exchange and market_type and timeframe:
                            fix_cmd = f"python backtester/scripts/fetch_data_cli.py --exchange {exchange} --market {market_type} --timeframe {timeframe} --symbols {symbol} --start {expected_inception.strftime('%Y-%m-%d')} --end {data_start.strftime('%Y-%m-%d')}"
                            self.apply_fix('inception_fill', fix_cmd, f"Fill missing inception data for {symbol} ({missing_days} days)")
        
        except Exception as e:
            self.add_issue('critical', 'analysis_error',
                         f"{filename}: Error checking inception coverage: {e}")
        
        return inception_issues
    
    def check_data_completeness(self, data: vbt.Data, filename: str) -> Dict:
        """Check for missing values and data completeness."""
        completeness_info = {}
        
        try:
            # Handle VBT Data objects properly
            if not hasattr(data, 'wrapper') or not hasattr(data.wrapper, 'index'):
                self.add_issue('warning', 'data_access_limited',
                             f"{filename}: Cannot access VBT data structure properly")
                return completeness_info
            
            # Get basic info
            total_timesteps = len(data.wrapper.index)
            total_symbols = len(data.symbols) if hasattr(data, 'symbols') else 0
            
            if total_symbols == 0:
                self.add_issue('warning', 'no_symbols',
                             f"{filename}: No symbols found in data")
                return completeness_info
            
            # For VBT Data, check if we have OHLCV attributes
            has_ohlcv = all(hasattr(data, attr) for attr in ['open', 'high', 'low', 'close', 'volume'])
            
            if has_ohlcv:
                # VBT Data with OHLCV attributes
                completeness_info = {
                    'data_type': 'VBT_OHLCV',
                    'total_symbols': total_symbols,
                    'total_timesteps': total_timesteps,
                    'has_all_ohlcv': True
                }
                
                # Check for NaN values in each OHLCV component
                total_cells = 0
                nan_cells = 0
                
                for attr in ['open', 'high', 'low', 'close', 'volume']:
                    if hasattr(data, attr):
                        attr_data = getattr(data, attr)
                        if attr_data is not None and hasattr(attr_data, 'values'):
                            values = attr_data.values
                            if values is not None:
                                attr_total = values.size
                                attr_nan = np.isnan(values).sum() if hasattr(np, 'isnan') else 0
                                
                                total_cells += attr_total
                                nan_cells += attr_nan
                                
                                completeness_info[f'{attr}_cells'] = attr_total
                                completeness_info[f'{attr}_nan'] = attr_nan
                            else:
                                completeness_info[f'{attr}_cells'] = 0
                                completeness_info[f'{attr}_nan'] = 0
                        else:
                            # Attribute exists but is None or has no values
                            completeness_info[f'{attr}_status'] = 'None or no values'
                
                if total_cells > 0:
                    completeness_percentage = ((total_cells - nan_cells) / total_cells) * 100
                    completeness_info.update({
                        'total_cells': total_cells,
                        'nan_cells': nan_cells,
                        'completeness_percentage': completeness_percentage
                    })
                    
                    if completeness_percentage < 95:
                        severity = 'critical' if completeness_percentage < 90 else 'warning'
                        self.add_issue(severity, 'data_completeness',
                                     f"{filename}: Data completeness {completeness_percentage:.1f}% "
                                     f"({nan_cells:,} NaN values out of {total_cells:,} cells)",
                                     completeness_info)
                
                # Basic OHLC validation (check if high >= low)
                try:
                    if hasattr(data, 'high') and hasattr(data, 'low'):
                        high_attr = getattr(data, 'high')
                        low_attr = getattr(data, 'low')
                        
                        # Check if the attributes are not None and have values
                        if (high_attr is not None and hasattr(high_attr, 'values') and 
                            low_attr is not None and hasattr(low_attr, 'values')):
                            
                            high_vals = high_attr.values
                            low_vals = low_attr.values
                            
                            if high_vals is not None and low_vals is not None and high_vals.shape == low_vals.shape:
                                invalid_prices = (high_vals < low_vals).sum()
                                if invalid_prices > 0:
                                    self.add_issue('critical', 'invalid_ohlc',
                                                 f"{filename}: {invalid_prices} instances where High < Low")
                        else:
                            # OHLC attributes exist but are None or don't have values
                            self.add_issue('warning', 'ohlc_access_limited',
                                         f"{filename}: OHLC attributes exist but cannot access values")
                except Exception as e:
                    self.add_issue('warning', 'ohlc_validation_error',
                                 f"{filename}: Could not validate OHLC relationships: {e}")
                                 
            else:
                # Try to get data as DataFrame (fallback)
                try:
                    df = data.get()
                    if df is not None and hasattr(df, 'shape'):
                        completeness_info = {
                            'data_type': 'DataFrame',
                            'total_symbols': total_symbols,
                            'total_timesteps': total_timesteps,
                            'df_shape': df.shape,
                            'has_all_ohlcv': False
                        }
                        
                        # Check DataFrame completeness
                        total_cells = df.size
                        nan_cells = df.isna().sum().sum() if hasattr(df, 'isna') else 0
                        completeness_percentage = ((total_cells - nan_cells) / total_cells) * 100 if total_cells > 0 else 100
                        
                        completeness_info.update({
                            'total_cells': total_cells,
                            'nan_cells': nan_cells,
                            'completeness_percentage': completeness_percentage
                        })
                        
                        if completeness_percentage < 95:
                            severity = 'critical' if completeness_percentage < 90 else 'warning'
                            self.add_issue(severity, 'data_completeness',
                                         f"{filename}: Data completeness {completeness_percentage:.1f}% "
                                         f"({nan_cells:,} NaN values out of {total_cells:,} cells)",
                                         completeness_info)
                    else:
                        self.add_issue('warning', 'data_access_error',
                                     f"{filename}: Cannot access data as DataFrame")
                        completeness_info = {
                            'data_type': 'Unknown',
                            'total_symbols': total_symbols,
                            'total_timesteps': total_timesteps,
                            'has_all_ohlcv': False
                        }
                except Exception as e:
                    self.add_issue('warning', 'data_access_error',
                                 f"{filename}: Error accessing data structure: {e}")
                    completeness_info = {
                        'data_type': 'AccessError',
                        'total_symbols': total_symbols,
                        'total_timesteps': total_timesteps,
                        'error': str(e)
                    }
                                 
        except Exception as e:
            self.add_issue('critical', 'analysis_error',
                         f"{filename}: Error checking data completeness: {e}")
        
        return completeness_info
    
    def check_symbol_consistency(self, data: vbt.Data, filename: str) -> Dict:
        """Check for symbol-specific issues."""
        symbol_info = {}
        
        try:
            symbols = data.symbols if hasattr(data, 'symbols') else []
            
            if not symbols:
                self.add_issue('warning', 'no_symbols',
                             f"{filename}: No symbols detected in data")
                return symbol_info
            
            # For VBT Data objects, analyze symbol coverage
            if hasattr(data, 'wrapper') and hasattr(data.wrapper, 'index'):
                total_timesteps = len(data.wrapper.index)
                
                if len(symbols) > 1:
                    # Check symbol-specific data coverage using VBT structure
                    for symbol in symbols:
                        try:
                            # For VBT data, check if we can access symbol-specific data
                            symbol_data_points = 0
                            missing_points = 0
                            
                            # Try to get close data for this symbol as a proxy for data availability
                            if hasattr(data, 'close') and hasattr(data.close, '__getitem__'):
                                try:
                                    symbol_close = data.close[symbol] if symbol in data.close.columns else None
                                    if symbol_close is not None and hasattr(symbol_close, 'dropna'):
                                        valid_data = symbol_close.dropna()
                                        symbol_data_points = len(valid_data)
                                        missing_points = total_timesteps - symbol_data_points
                                        
                                        symbol_info[symbol] = {
                                            'start_date': str(valid_data.index[0]) if len(valid_data) > 0 else None,
                                            'end_date': str(valid_data.index[-1]) if len(valid_data) > 0 else None,
                                            'data_points': symbol_data_points,
                                            'missing_points': missing_points,
                                            'total_expected': total_timesteps
                                        }
                                    else:
                                        # Symbol exists but no valid data found
                                        symbol_info[symbol] = {
                                            'start_date': None,
                                            'end_date': None,
                                            'data_points': 0,
                                            'missing_points': total_timesteps,
                                            'total_expected': total_timesteps
                                        }
                                        self.add_issue('warning', 'symbol_no_data',
                                                     f"{filename}: Symbol {symbol} has no valid data")
                                except KeyError:
                                    # Symbol not found in close data
                                    symbol_info[symbol] = {
                                        'start_date': None,
                                        'end_date': None,
                                        'data_points': 0,
                                        'missing_points': total_timesteps,
                                        'total_expected': total_timesteps
                                    }
                                    self.add_issue('warning', 'symbol_not_found',
                                                 f"{filename}: Symbol {symbol} not found in close data")
                                except Exception as e:
                                    self.add_issue('warning', 'symbol_access_error',
                                                 f"{filename}: Error accessing symbol {symbol}: {e}")
                            else:
                                # No close data available - use basic info
                                symbol_info[symbol] = {
                                    'start_date': str(data.wrapper.index[0]) if total_timesteps > 0 else None,
                                    'end_date': str(data.wrapper.index[-1]) if total_timesteps > 0 else None,
                                    'data_points': total_timesteps,  # Assume full coverage
                                    'missing_points': 0,
                                    'total_expected': total_timesteps
                                }
                                
                        except Exception as e:
                            self.add_issue('warning', 'symbol_analysis_error',
                                         f"{filename}: Error analyzing symbol {symbol}: {e}")
                    
                    # Check for significant differences in data coverage between symbols
                    if len(symbol_info) > 1:
                        data_points = [info['data_points'] for info in symbol_info.values() 
                                     if info['data_points'] is not None and info['data_points'] > 0]
                        if data_points:
                            min_points = min(data_points)
                            max_points = max(data_points)
                            
                            if len(data_points) > 1 and (max_points - min_points) > max_points * 0.1:  # More than 10% difference
                                self.add_issue('warning', 'symbol_coverage_mismatch',
                                             f"{filename}: Significant variation in symbol coverage "
                                             f"({min_points} to {max_points} points)")
                else:
                    # Single symbol case
                    symbol = symbols[0]
                    symbol_info[symbol] = {
                        'start_date': str(data.wrapper.index[0]) if total_timesteps > 0 else None,
                        'end_date': str(data.wrapper.index[-1]) if total_timesteps > 0 else None,
                        'data_points': total_timesteps,
                        'missing_points': 0,
                        'total_expected': total_timesteps
                    }
            else:
                self.add_issue('warning', 'data_structure_limited',
                             f"{filename}: Cannot analyze symbol consistency - limited data access")
                             
        except Exception as e:
            self.add_issue('critical', 'analysis_error',
                         f"{filename}: Error checking symbol consistency: {e}")
        
        return symbol_info
    
    def analyze_file(self, filepath: Path) -> Dict:
        """Analyze a single data file."""
        filename = filepath.name
        file_info = {
            'filepath': str(filepath),
            'filename': filename,
            'size_mb': filepath.stat().st_size / (1024 * 1024),
            'modified': datetime.fromtimestamp(filepath.stat().st_mtime)
        }
        
        try:
            # Parse filename to extract metadata
            base_name = filename.replace('.pickle.blosc', '').replace('.pickle', '')
            parts = base_name.split('_')
            if len(parts) >= 3:
                exchange = parts[0]
                market_type = parts[1] 
                timeframe = parts[2]
                
                file_info.update({
                    'exchange': exchange,
                    'market_type': market_type,
                    'timeframe': timeframe
                })
            
            # Load the data
            print(f"📊 Analyzing {filename}...")
            
            # Load data using VBT data storage system (handles compression automatically)
            data = None
            try:
                if 'exchange' in file_info and 'timeframe' in file_info and 'market_type' in file_info:
                    # Use the data storage system to load
                    data = data_storage.load_data(
                        file_info['exchange'], 
                        file_info['timeframe'], 
                        market_type=file_info['market_type']
                    )
                else:
                    # Fallback to direct loading
                    import pickle
                    with open(filepath, 'rb') as f:
                        data = pickle.load(f)
            except Exception as e:
                self.add_issue('critical', 'load_error',
                             f"{filename}: Cannot load file: {e}")
                return file_info
            
            if data is None:
                self.add_issue('critical', 'empty_file',
                             f"{filename}: File loaded but contains no data")
                return file_info
            
            # Basic file info
            file_info.update({
                'symbols': list(data.symbols) if hasattr(data, 'symbols') else [],
                'symbol_count': len(data.symbols) if hasattr(data, 'symbols') else 0,
                'data_shape': data.wrapper.shape if hasattr(data, 'wrapper') else None,
                'date_range': (
                    str(data.wrapper.index[0]), 
                    str(data.wrapper.index[-1])
                ) if hasattr(data, 'wrapper') and len(data.wrapper.index) > 0 else None
            })
            
            # Run health checks with auto-fix capabilities
            exchange = file_info.get('exchange')
            market_type = file_info.get('market_type')
            timeframe = file_info.get('timeframe')
            
            # Check data freshness
            freshness_info = self.check_data_freshness(data, filename, exchange, market_type, timeframe)
            file_info['freshness'] = freshness_info
            
            # Check for date gaps
            if timeframe:
                gaps = self.check_date_gaps(data, filename, timeframe, exchange, market_type)
                file_info['gaps'] = gaps
            
            # Check inception coverage
            inception_issues = self.check_inception_coverage(data, filename, exchange, market_type, timeframe)
            file_info['inception_issues'] = inception_issues
            
            # Check data completeness
            completeness_info = self.check_data_completeness(data, filename)
            file_info['completeness'] = completeness_info
            
            # Check symbol consistency
            symbol_info = self.check_symbol_consistency(data, filename)
            file_info['symbol_info'] = symbol_info
            
            print(f"   ✅ Analysis complete")
            
        except Exception as e:
            self.add_issue('critical', 'analysis_error',
                         f"{filename}: Fatal error during analysis: {e}")
            print(f"   ❌ Analysis failed: {e}")
        
        return file_info
    
    def save_report(self, report: str, filename: str = None) -> Path:
        """Save report to the reports directory."""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"health_report_{timestamp}.txt"
        
        report_path = self.reports_dir / filename
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path
    
    def generate_report(self, file_analyses: List[Dict]) -> str:
        """Generate a comprehensive health check report."""
        report = []
        report.append("=" * 80)
        report.append("🔍 DATA HEALTH CHECK REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Files analyzed: {len(file_analyses)}")
        report.append(f"Auto-fix enabled: {'Yes' if self.auto_fix else 'No'}")
        report.append("")
        
        # Cache system info
        if self.cached_inception_dates:
            report.append("📅 CACHED INCEPTION DATA")
            report.append("-" * 40)
            total_cached = sum(len(symbols) for symbols in self.cached_inception_dates.values())
            report.append(f"Total cached inception dates: {total_cached}")
            for exchange, symbols in self.cached_inception_dates.items():
                report.append(f"  {exchange}: {len(symbols)} symbols")
            report.append("")
        
        # Summary statistics
        total_issues = sum(len(issues) for issues in self.issues.values())
        report.append("📋 SUMMARY")
        report.append("-" * 40)
        report.append(f"🔴 Critical issues: {len(self.issues['critical'])}")
        report.append(f"🟡 Warnings: {len(self.issues['warning'])}")
        report.append(f"ℹ️  Info: {len(self.issues['info'])}")
        report.append(f"📊 Total issues: {total_issues}")
        
        if self.fixes_applied:
            successful_fixes = len([f for f in self.fixes_applied if f['success']])
            report.append(f"🔧 Fixes applied: {successful_fixes}/{len(self.fixes_applied)}")
        
        report.append("")
        
        # Auto-fix summary
        if self.fixes_applied:
            report.append("🔧 AUTO-FIX SUMMARY")
            report.append("-" * 40)
            for fix in self.fixes_applied:
                status = "✅" if fix['success'] else "❌"
                report.append(f"{status} {fix['description']}")
                if not fix['success'] and 'error' in fix:
                    report.append(f"   Error: {fix['error']}")
            report.append("")
        
        # File overview
        if file_analyses:
            report.append("📁 FILE OVERVIEW")
            report.append("-" * 40)
            total_size = sum(f.get('size_mb', 0) for f in file_analyses)
            total_symbols = sum(f.get('symbol_count', 0) for f in file_analyses)
            
            report.append(f"Total storage: {total_size:.1f} MB")
            report.append(f"Total symbols: {total_symbols}")
            report.append("")
            
            # Group by exchange and timeframe
            by_exchange = {}
            by_timeframe = {}
            
            for file_info in file_analyses:
                exchange = file_info.get('exchange', 'unknown')
                timeframe = file_info.get('timeframe', 'unknown')
                
                by_exchange[exchange] = by_exchange.get(exchange, 0) + 1
                by_timeframe[timeframe] = by_timeframe.get(timeframe, 0) + 1
            
            report.append("   By Exchange:")
            for exchange, count in sorted(by_exchange.items()):
                report.append(f"     {exchange}: {count} files")
            
            report.append("   By Timeframe:")
            for timeframe, count in sorted(by_timeframe.items()):
                report.append(f"     {timeframe}: {count} files")
            report.append("")
        
        # Issues by category
        if total_issues > 0:
            report.append("🚨 ISSUES FOUND")
            report.append("-" * 40)
            
            for severity in ['critical', 'warning', 'info']:
                if self.issues[severity]:
                    report.append(f"\n{severity.upper()} ISSUES ({len(self.issues[severity])}):")
                    
                    for issue in self.issues[severity]:
                        report.append(f"  • {issue['message']}")
                        
                        if self.detailed and issue['details']:
                            for key, value in issue['details'].items():
                                if isinstance(value, (list, dict)):
                                    report.append(f"    {key}: {str(value)[:100]}...")
                                else:
                                    report.append(f"    {key}: {value}")
        else:
            report.append("✅ NO ISSUES FOUND")
            report.append("All data files passed health checks!")
        
        # Detailed file information
        if self.detailed and file_analyses:
            report.append("\n" + "=" * 80)
            report.append("📋 DETAILED FILE ANALYSIS")
            report.append("=" * 80)
            
            for file_info in file_analyses:
                report.append(f"\n📄 {file_info['filename']}")
                report.append(f"   Size: {file_info.get('size_mb', 0):.2f} MB")
                report.append(f"   Modified: {file_info.get('modified', 'Unknown')}")
                
                if file_info.get('symbols'):
                    report.append(f"   Symbols: {len(file_info['symbols'])} ({', '.join(file_info['symbols'][:5])}{'...' if len(file_info['symbols']) > 5 else ''})")
                
                if file_info.get('date_range'):
                    report.append(f"   Date range: {file_info['date_range'][0]} to {file_info['date_range'][1]}")
                
                if file_info.get('data_shape'):
                    report.append(f"   Shape: {file_info['data_shape']}")
                
                if file_info.get('freshness'):
                    hours_behind = file_info['freshness'].get('hours_behind', 0)
                    report.append(f"   Freshness: {hours_behind:.1f} hours behind")
                
                if file_info.get('gaps'):
                    report.append(f"   Gaps: {len(file_info['gaps'])} found")
                
                if file_info.get('completeness'):
                    comp = file_info['completeness']
                    report.append(f"   Completeness: {comp.get('completeness_percentage', 0):.1f}%")
        
        # Recommendations
        report.append("\n" + "=" * 80)
        report.append("💡 RECOMMENDATIONS")
        report.append("=" * 80)
        
        if len(self.issues['critical']) > 0:
            report.append("🔴 CRITICAL ACTIONS NEEDED:")
            if not self.auto_fix:
                report.append("  1. Run with --auto-fix to automatically resolve critical issues")
            report.append("  2. Address critical data gaps - consider re-fetching affected periods")
            report.append("  3. Fix invalid OHLCV data - these can affect calculations")
            report.append("  4. Resolve loading errors - these files may be corrupted")
        
        if len(self.issues['warning']) > 0:
            report.append("\n🟡 RECOMMENDED ACTIONS:")
            report.append("  1. Consider filling minor data gaps")
            report.append("  2. Review inception coverage for important symbols")
            report.append("  3. Monitor zero volume periods")
            report.append("  4. Update stale data regularly")
        
        if total_issues == 0:
            report.append("🎉 DATA QUALITY EXCELLENT!")
            report.append("  Your cached data appears to be in excellent condition.")
            report.append("  Continue with regular monitoring and updates.")
        
        if not self.auto_fix and total_issues > 0:
            report.append("\n🔧 AUTO-FIX AVAILABLE:")
            report.append("  Run this script with --auto-fix to automatically resolve many issues.")
            report.append("  This will execute commands to re-fetch missing data and update stale data.")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze cached VBT data for quality issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Check all data
  %(prog)s --exchange binance           # Check specific exchange
  %(prog)s --timeframe 1h --detailed    # Check specific timeframe with details
  %(prog)s --auto-fix                   # Check and automatically fix issues
        """
    )
    
    parser.add_argument('--exchange', type=str,
                      help='Filter by specific exchange (e.g., binance)')
    parser.add_argument('--timeframe', type=str,
                      help='Filter by specific timeframe (e.g., 1h, 1d)')
    parser.add_argument('--market', type=str, choices=['spot', 'swap'],
                      help='Filter by market type')
    parser.add_argument('--detailed', action='store_true',
                      help='Show detailed analysis for each file')
    parser.add_argument('--auto-fix', action='store_true',
                      help='Automatically fix discovered issues')
    parser.add_argument('--output', type=str,
                      help='Save report to file (default: auto-generated in reports folder)')
    
    args = parser.parse_args()
    
    print("🔍 Starting Data Health Check...")
    if args.auto_fix:
        print("🔧 Auto-fix mode enabled - will attempt to resolve issues")
    print("=" * 50)
    
    # Initialize health checker
    checker = DataHealthChecker(detailed=args.detailed, auto_fix=args.auto_fix)
    
    # Get storage directory
    storage_dir = Path(data_storage.storage_dir)
    if not storage_dir.exists():
        print(f"❌ Storage directory not found: {storage_dir}")
        return 1
    
    # Find data files
    pickle_files = list(storage_dir.glob("*.pickle")) + list(storage_dir.glob("*.pickle.blosc"))
    
    # Filter files based on arguments
    filtered_files = []
    for filepath in pickle_files:
        filename = filepath.name
        
        # Parse filename: exchange_market_timeframe.pickle[.blosc]
        base_name = filename.replace('.pickle.blosc', '').replace('.pickle', '')
        parts = base_name.split('_')
        if len(parts) < 3:
            continue
            
        exchange, market_type, timeframe = parts[0], parts[1], parts[2]
        
        # Apply filters
        if args.exchange and exchange != args.exchange:
            continue
        if args.timeframe and timeframe != args.timeframe:
            continue
        if args.market and market_type != args.market:
            continue
            
        filtered_files.append(filepath)
    
    if not filtered_files:
        print("❌ No data files found matching criteria")
        return 1
    
    print(f"📁 Found {len(filtered_files)} data files to analyze")
    print()
    
    # Analyze each file
    file_analyses = []
    for filepath in filtered_files:
        try:
            file_info = checker.analyze_file(filepath)
            file_analyses.append(file_info)
        except Exception as e:
            print(f"❌ Failed to analyze {filepath.name}: {e}")
            checker.add_issue('critical', 'analysis_error',
                            f"Failed to analyze {filepath.name}: {e}")
    
    print()
    print("=" * 50)
    print("🔍 Analysis Complete - Generating Report...")
    print()
    
    # Generate report
    report = checker.generate_report(file_analyses)
    
    # Output report
    if args.output:
        report_path = checker.reports_dir / args.output
        checker.save_report(report, args.output)
        print(f"📄 Report saved to: {report_path}")
    else:
        # Auto-generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filters = []
        if args.exchange:
            filters.append(f"exchange_{args.exchange}")
        if args.timeframe:
            filters.append(f"timeframe_{args.timeframe}")
        if args.market:
            filters.append(f"market_{args.market}")
        
        filter_str = "_".join(filters)
        filename = f"health_report_{filter_str}_{timestamp}.txt" if filter_str else f"health_report_{timestamp}.txt"
        
        report_path = checker.save_report(report, filename)
        print(f"📄 Report saved to: {report_path}")
        print(report)
    
    # Return appropriate exit code
    if len(checker.issues['critical']) > 0:
        return 2  # Critical issues found
    elif len(checker.issues['warning']) > 0:
        return 1  # Warnings found
    else:
        return 0  # All good

if __name__ == "__main__":
    exit(main()) 