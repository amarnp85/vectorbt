#!/usr/bin/env python3
"""Streamlined Data Health Check

Focused health analysis for VBT cached data with realistic gap resolution.
Integrates with the new CLI tools and simplified cache system.

Key features:
- Critical gap detection and resolution strategies
- Data freshness monitoring
- Cache integrity validation
- Actionable fix recommendations using available CLI tools

Usage:
    python -m backtester.data.health_check.data_healthcheck [options]
    
Examples:
    # Quick health check with fixes
    python -m backtester.data.health_check.data_healthcheck --auto-fix
    
    # Detailed analysis for specific exchange
    python -m backtester.data.health_check.data_healthcheck --exchange binance --detailed
    
    # Focus on critical issues only
    python -m backtester.data.health_check.data_healthcheck --critical-only
    
    # Fill minor gaps with interpolation
    python -m backtester.data.health_check.data_healthcheck --interpolate --auto-fix
    
    # Use linear interpolation
    python -m backtester.data.health_check.data_healthcheck --interpolate --interpolation-strategy linear
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
from pathlib import Path
import subprocess
import logging

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backtester.data.storage.data_storage import data_storage
from backtester.data.cache_system.cache_manager import SimpleCacheManager
from backtester.data.fetching.core.vbt_data_handler import VBTDataHandler
from backtester.data.fetching.core.freshness_checker import FreshnessChecker
import vectorbtpro as vbt

logger = logging.getLogger(__name__)

class StreamlinedHealthChecker:
    """Streamlined health checker focused on actionable issues and realistic solutions."""
    
    def __init__(self, detailed: bool = False, auto_fix: bool = False, critical_only: bool = False, 
                 enable_interpolation: bool = False, interpolation_strategy: str = 'financial_forward_fill'):
        self.detailed = detailed
        self.auto_fix = auto_fix
        self.critical_only = critical_only
        self.enable_interpolation = enable_interpolation
        self.interpolation_strategy = interpolation_strategy
        
        self.issues = {
            'critical': [],
            'warning': [],
            'info': []
        }
        self.fixes_applied = []
        
        # Initialize cache manager
        self.cache_manager = SimpleCacheManager()
        
        # Reports directory
        self.reports_dir = Path(__file__).parent / 'reports'
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def add_issue(self, severity: str, category: str, message: str, fix_commands: List[str] = None):
        """Add an issue with optional fix commands."""
        if self.critical_only and severity != 'critical':
            return
            
        issue = {
            'category': category,
            'message': message,
            'fix_commands': fix_commands or [],
            'timestamp': datetime.now().isoformat()
        }
        self.issues[severity].append(issue)
    
    def apply_fix(self, description: str, commands: List[str]) -> bool:
        """Apply a fix by running commands."""
        if not self.auto_fix or not commands:
            return False
        
        print(f"🔧 Applying fix: {description}")
        
        success_count = 0
        for cmd in commands:
            try:
                # Set working directory to project root
                project_root = Path(__file__).parent.parent.parent.parent
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=project_root)
                
                if result.returncode == 0:
                    print(f"   ✅ Command successful: {cmd[:50]}...")
                    success_count += 1
                else:
                    print(f"   ❌ Command failed: {cmd[:50]}...")
                    print(f"      Error: {result.stderr[:100]}")
                    
            except Exception as e:
                print(f"   ❌ Exception running command: {e}")
        
        # Record the fix attempt
        self.fixes_applied.append({
            'description': description,
            'commands': commands,
            'success_count': success_count,
            'total_commands': len(commands),
            'success': success_count == len(commands)
        })
        
        return success_count == len(commands)
    
    def check_data_gaps(self, data: vbt.Data, exchange: str, market: str, timeframe: str) -> Dict[str, Any]:
        """Check for critical data gaps and provide realistic solutions."""
        gap_info = {'critical_gaps': [], 'minor_gaps': [], 'total_missing': 0}
        
        try:
            # Extract OHLCV for gap analysis
            ohlcv = VBTDataHandler.extract_ohlcv(data)
            if not ohlcv or 'close' not in ohlcv:
                return gap_info
            
            close_df = ohlcv['close']
            tf_minutes = FreshnessChecker.parse_timeframe_minutes(timeframe)
            
            # Define realistic gap thresholds based on timeframe and data span
            total_symbols = len(close_df.columns)
            
            # Calculate expected timespan to set realistic thresholds
            if len(close_df.index) > 0:
                total_days = (close_df.index[-1] - close_df.index[0]).days
                # For longer timespans, allow more tolerance for gaps
                if total_days > 365:  # More than 1 year of data
                    # Be more lenient - crypto markets have natural gaps
                    critical_threshold = max(5000, total_days * 2)  # ~2 periods per day as critical
                    minor_threshold = max(1000, total_days * 0.5)   # ~0.5 periods per day as minor
                else:
                    # Shorter timespan - be more strict
                    critical_threshold = 2000
                    minor_threshold = 500
            else:
                # Fallback thresholds
                critical_threshold = 5000
                minor_threshold = 1000
            
            # Check each symbol for gaps
            for symbol in close_df.columns:
                symbol_data = close_df[symbol].dropna()
                if len(symbol_data) < 2:
                    continue
                
                # Calculate expected vs actual data points
                time_span = symbol_data.index[-1] - symbol_data.index[0]
                expected_periods = int(time_span.total_seconds() / (tf_minutes * 60)) + 1
                actual_periods = len(symbol_data)
                missing_periods = expected_periods - actual_periods
                
                if missing_periods > 0:
                    gap_info['total_missing'] += missing_periods
                    
                    # Calculate completion rate for context
                    completion_rate = actual_periods / expected_periods if expected_periods > 0 else 1.0
                    
                    # Categorize gaps using dynamic thresholds
                    if missing_periods > critical_threshold:
                        gap_info['critical_gaps'].append({
                            'symbol': symbol,
                            'missing_periods': missing_periods,
                            'start_date': symbol_data.index[0],
                            'end_date': symbol_data.index[-1],
                            'completion_rate': completion_rate
                        })
                    elif missing_periods > minor_threshold:
                        gap_info['minor_gaps'].append({
                            'symbol': symbol,
                            'missing_periods': missing_periods,
                            'completion_rate': completion_rate
                        })
            
            # Add issues and fixes for critical gaps only
            if gap_info['critical_gaps']:
                critical_symbols = [g['symbol'] for g in gap_info['critical_gaps'][:5]]  # Limit to 5 symbols
                symbol_list = ','.join(critical_symbols)
                
                fix_commands = [
                    f"python backtester/scripts/fetch_data_cli.py --exchange {exchange} --market {market} --timeframe {timeframe} --symbols {symbol_list} --inception"
                ]
                
                avg_missing = gap_info['total_missing'] // len(gap_info['critical_gaps'])
                completion_rates = [g['completion_rate'] for g in gap_info['critical_gaps']]
                avg_completion = sum(completion_rates) / len(completion_rates) * 100
                
                self.add_issue(
                    'critical', 
                    'data_gaps',
                    f"{exchange}/{market}/{timeframe}: {len(gap_info['critical_gaps'])} symbols with major gaps (avg {avg_missing} missing, {avg_completion:.1f}% complete)",
                    fix_commands
                )
                
                # Apply fix if auto-fix enabled
                if self.auto_fix:
                    self.apply_fix(f"Fill critical gaps for {len(critical_symbols)} symbols", fix_commands)
            
            elif gap_info['minor_gaps']:
                avg_missing = gap_info['total_missing'] // len(gap_info['minor_gaps']) if gap_info['minor_gaps'] else 0
                
                # Offer interpolation for minor gaps if enabled
                if self.enable_interpolation and self.auto_fix and gap_info['total_missing'] <= 10000:
                    self.add_issue(
                        'warning',
                        'data_gaps', 
                        f"{exchange}/{market}/{timeframe}: {len(gap_info['minor_gaps'])} symbols with minor gaps (avg {avg_missing} missing periods) - attempting interpolation",
                        [f"Interpolate using {self.interpolation_strategy} strategy"]
                    )
                    
                    # Apply interpolation fix
                    if self.apply_interpolation_fix(data, exchange, market, timeframe, {}, gap_info):
                        print(f"   🔧 Applied interpolation fix for minor gaps")
                        self.apply_fix(f"Interpolate {gap_info['total_missing']} missing periods", 
                                     [f"Applied {self.interpolation_strategy} interpolation"])
                        return gap_info  # Return updated gap info
                else:
                    self.add_issue(
                        'warning',
                        'data_gaps', 
                        f"{exchange}/{market}/{timeframe}: {len(gap_info['minor_gaps'])} symbols with minor gaps (avg {avg_missing} missing periods)"
                    )
            
        except Exception as e:
            self.add_issue('critical', 'analysis_error', f"Gap analysis failed for {exchange}/{market}/{timeframe}: {e}")
        
        return gap_info
    
    def check_data_freshness(self, data: vbt.Data, exchange: str, market: str, timeframe: str) -> Dict[str, Any]:
        """Check data freshness with realistic update strategies."""
        freshness_info = {}
        
        try:
            start_date, end_date = VBTDataHandler.get_date_range(data)
            now = datetime.now()
            age_hours = (now - end_date.replace(tzinfo=None)).total_seconds() / 3600
            
            freshness_info = {
                'latest_data': end_date,
                'age_hours': age_hours,
                'is_stale': False
            }
            
            # Define staleness thresholds based on timeframe
            staleness_map = {
                '1m': 2, '5m': 6, '15m': 12, '30m': 24,
                '1h': 6, '2h': 12, '4h': 24, '6h': 36, '8h': 48, '12h': 72,
                '1d': 48, '3d': 96, '1w': 168
            }
            
            threshold = staleness_map.get(timeframe, 24)
            
            if age_hours > threshold:
                freshness_info['is_stale'] = True
                severity = 'critical' if age_hours > threshold * 2 else 'warning'
                
                # Get top symbols for update
                symbols = list(data.symbols)[:10] if hasattr(data, 'symbols') else []
                symbol_list = ','.join(symbols)
                
                fix_commands = [
                    f"python backtester/scripts/fetch_data_cli.py --exchange {exchange} --market {market} --timeframe {timeframe} --symbols {symbol_list} --end now"
                ]
                
                self.add_issue(
                    severity,
                    'stale_data',
                    f"{exchange}/{market}/{timeframe}: Data is {age_hours:.1f} hours stale (threshold: {threshold}h)",
                    fix_commands
                )
                
                # Apply fix for critical staleness
                if severity == 'critical' and self.auto_fix:
                    self.apply_fix(f"Update stale {timeframe} data", fix_commands)
        
        except Exception as e:
            self.add_issue('critical', 'analysis_error', f"Freshness check failed: {e}")
        
        return freshness_info
    
    def check_cache_integrity(self) -> Dict[str, Any]:
        """Check cache system integrity."""
        integrity_info = {'exchanges': [], 'issues': []}
        
        try:
            # Check cache statistics
            cache_stats = self.cache_manager.get_cache_stats()
            
            for exchange in cache_stats['exchanges']:
                exchange_stats = cache_stats['by_exchange'][exchange]
                
                # Check for missing volume cache
                if not self.cache_manager.is_volume_cache_fresh(exchange):
                    fix_commands = [f"python backtester/scripts/fetch_data_cli.py --exchange {exchange} --top 20"]
                    self.add_issue(
                        'warning',
                        'cache_stale',
                        f"{exchange}: Volume cache is stale or missing",
                        fix_commands
                    )
                    
                    # Auto-fix stale volume cache (reasonable warning-level fix)
                    if self.auto_fix:
                        self.apply_fix(f"Refresh volume cache for {exchange}", fix_commands)
                
                # Check for excessive failed symbols
                failed_count = exchange_stats['failed_symbols']
                if failed_count > 10:
                    self.add_issue(
                        'warning',
                        'failed_symbols',
                        f"{exchange}: {failed_count} symbols marked as failed"
                    )
                
                integrity_info['exchanges'].append({
                    'exchange': exchange,
                    'volume_symbols': exchange_stats['volume_symbols'],
                    'timestamp_symbols': exchange_stats['timestamp_symbols'],
                    'failed_symbols': failed_count,
                    'volume_fresh': exchange_stats['volume_cache_fresh']
                })
        
        except Exception as e:
            self.add_issue('critical', 'cache_error', f"Cache integrity check failed: {e}")
        
        return integrity_info
    
    def analyze_storage_file(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single storage file for health issues."""
        exchange = file_info['exchange']
        market = file_info['market']
        timeframe = file_info['timeframe']
        
        print(f"📊 Analyzing {exchange}/{market}/{timeframe}...")
        
        analysis = {
            'file_info': file_info,
            'has_data': False,
            'symbol_count': 0,
            'gaps': {},
            'freshness': {},
            'size_mb': file_info.get('file_size_bytes', 0) / (1024 * 1024),
            'fixes_applied': []
        }
        
        try:
            # Load data
            data = data_storage.load_data(exchange, timeframe, market_type=market)
            
            if data is None:
                self.add_issue('critical', 'load_error', f"Cannot load {exchange}/{market}/{timeframe}")
                return analysis
            
            analysis['has_data'] = True
            analysis['symbol_count'] = len(data.symbols) if hasattr(data, 'symbols') else 0
            
            # Store initial fix count
            initial_fix_count = len(self.fixes_applied)
            
            # Check data gaps (may apply fixes if auto-fix enabled)
            analysis['gaps'] = self.check_data_gaps(data, exchange, market, timeframe)
            
            # Check if fixes were applied for this file
            fixes_applied_count = len(self.fixes_applied) - initial_fix_count
            if fixes_applied_count > 0 and self.auto_fix:
                print(f"   🔄 Reloading data after {fixes_applied_count} fix(es)...")
                
                # Reload data to see improvements
                data_reloaded = data_storage.load_data(exchange, timeframe, market_type=market)
                if data_reloaded is not None:
                    # Re-analyze gaps with fresh data
                    analysis['gaps'] = self.check_data_gaps(data_reloaded, exchange, market, timeframe)
                    data = data_reloaded  # Use reloaded data for freshness check
                    analysis['fixes_applied'] = self.fixes_applied[-fixes_applied_count:]
                    print(f"   ♻️  Data reloaded and re-analyzed")
            
            # Check freshness (using potentially reloaded data)
            analysis['freshness'] = self.check_data_freshness(data, exchange, market, timeframe)
            
            missing_count = analysis['gaps']['total_missing']
            print(f"   ✅ {analysis['symbol_count']} symbols, {missing_count} missing periods")
            
        except Exception as e:
            self.add_issue('critical', 'analysis_error', f"Analysis failed for {exchange}/{market}/{timeframe}: {e}")
            print(f"   ❌ Analysis failed: {e}")
        
        return analysis
    
    def generate_report(self, analyses: List[Dict[str, Any]]) -> str:
        """Generate focused health report with actionable recommendations."""
        lines = []
        lines.append("=" * 70)
        lines.append("🔍 VBT DATA HEALTH CHECK")
        lines.append("=" * 70)
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Files analyzed: {len(analyses)}")
        lines.append(f"Auto-fix enabled: {'Yes' if self.auto_fix else 'No'}")
        lines.append("")
        
        # Summary
        total_issues = sum(len(issues) for issues in self.issues.values())
        lines.append("📋 ISSUE SUMMARY")
        lines.append("-" * 40)
        lines.append(f"🔴 Critical: {len(self.issues['critical'])}")
        lines.append(f"🟡 Warning: {len(self.issues['warning'])}")
        lines.append(f"ℹ️  Info: {len(self.issues['info'])}")
        lines.append("")
        
        # Fixes applied
        if self.fixes_applied:
            lines.append("🔧 FIXES APPLIED")
            lines.append("-" * 40)
            for fix in self.fixes_applied:
                status = "✅" if fix['success'] else "❌"
                lines.append(f"{status} {fix['description']}")
                lines.append(f"   Commands: {fix['success_count']}/{fix['total_commands']} successful")
            lines.append("")
        
        # Critical issues first
        if self.issues['critical']:
            lines.append("🚨 CRITICAL ISSUES")
            lines.append("-" * 40)
            for issue in self.issues['critical']:
                lines.append(f"• {issue['message']}")
                if issue['fix_commands'] and not self.auto_fix:
                    lines.append(f"  Fix: {issue['fix_commands'][0]}")
            lines.append("")
        
        # Warnings (only if not critical-only mode)
        if not self.critical_only and self.issues['warning']:
            lines.append("⚠️  WARNING ISSUES")
            lines.append("-" * 40)
            for issue in self.issues['warning'][:5]:  # Limit to first 5
                lines.append(f"• {issue['message']}")
            
            if len(self.issues['warning']) > 5:
                lines.append(f"... and {len(self.issues['warning']) - 5} more warnings")
            lines.append("")
        
        # Storage overview
        if analyses:
            lines.append("💾 STORAGE OVERVIEW")
            lines.append("-" * 40)
            total_size = sum(a.get('size_mb', 0) for a in analyses)
            total_symbols = sum(a.get('symbol_count', 0) for a in analyses)
            
            lines.append(f"Total files: {len(analyses)}")
            lines.append(f"Total size: {total_size:.1f} MB")
            lines.append(f"Total symbols: {total_symbols}")
            lines.append("")
            
            # Show files with issues
            problem_files = [a for a in analyses if a['gaps']['critical_gaps'] or a['freshness'].get('is_stale', False)]
            if problem_files:
                lines.append("🚨 FILES WITH ISSUES")
                lines.append("-" * 40)
                for analysis in problem_files[:5]:
                    info = analysis['file_info']
                    lines.append(f"📄 {info['exchange']}/{info['market']}/{info['timeframe']}")
                    if analysis['gaps']['critical_gaps']:
                        lines.append(f"   Gaps: {len(analysis['gaps']['critical_gaps'])} symbols")
                    if analysis['freshness'].get('is_stale', False):
                        lines.append(f"   Stale: {analysis['freshness']['age_hours']:.1f}h old")
                lines.append("")
        
        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        lines.append("-" * 40)
        
        if len(self.issues['critical']) > 0:
            lines.append("🔴 IMMEDIATE ACTION REQUIRED:")
            lines.append("  1. Run with --auto-fix to resolve critical issues automatically")
            lines.append("  2. Check storage integrity and reload corrupted files")
            lines.append("  3. Address data gaps for high-priority symbols")
        elif len(self.issues['warning']) > 0:
            lines.append("🟡 RECOMMENDED ACTIONS:")
            lines.append("  1. Update stale data during low-activity periods")
            lines.append("  2. Consider running gap fills for important symbols")
            lines.append("  3. Monitor cache freshness regularly")
            
            # Check if there are minor gaps that could benefit from interpolation
            has_minor_gaps = any('minor gaps' in issue['message'] for issue in self.issues['warning'])
            if has_minor_gaps:
                lines.append("  4. For minor gaps, consider interpolation:")
                lines.append("     --interpolate --auto-fix (financial forward-fill)")
                lines.append("     --interpolate --interpolation-strategy linear")
        else:
            lines.append("✅ DATA HEALTH IS GOOD!")
            lines.append("  Your cached data appears to be in good condition.")
            lines.append("  Continue regular monitoring and updates.")
        
        if total_issues > 0 and not self.auto_fix:
            lines.append("")
            lines.append("🔧 AUTO-FIX AVAILABLE:")
            lines.append("  Run this script with --auto-fix to automatically resolve many issues.")
        
        # Quick commands
        lines.append("")
        lines.append("🚀 QUICK COMMANDS")
        lines.append("-" * 40)
        lines.append("# Check cache status:")
        lines.append("python backtester/scripts/inspect_cache_cli.py")
        lines.append("")
        lines.append("# Update stale data:")
        lines.append("python backtester/scripts/fetch_data_cli.py --exchange binance --top 10")
        lines.append("")
        lines.append("# Fill gaps from inception:")
        lines.append("python backtester/scripts/fetch_data_cli.py --exchange binance --symbols BTC/USDT,ETH/USDT --inception")
        lines.append("")
        lines.append("# Interpolate minor gaps (financial strategy):")
        lines.append("python -m backtester.data.health_check.data_healthcheck --interpolate --auto-fix")
        lines.append("")
        lines.append("# Interpolate with linear strategy:")
        lines.append("python -m backtester.data.health_check.data_healthcheck --interpolate --interpolation-strategy linear --auto-fix")
        lines.append("")
        lines.append("🎯 GAP THRESHOLDS (DYNAMIC)")
        lines.append("-" * 40)
        lines.append("• Critical: >5000 missing periods for multi-year data")
        lines.append("• Minor: >1000 missing periods (normal for crypto markets)")
        lines.append("• Note: Thresholds scale with data timespan for realistic assessment")
        lines.append("")
        lines.append("📊 INTERPOLATION STRATEGIES")
        lines.append("-" * 40)
        lines.append("• financial_forward_fill: OHLC=last_close, Volume=0 (recommended)")
        lines.append("• linear: Linear interpolation between known points")
        lines.append("• time_aware: Time-weighted interpolation considering gaps")
        lines.append("• Max gap size: 10,000 periods (~7 days of 1m data)")
        
        return "\n".join(lines)
    
    def save_report(self, report: str) -> Path:
        """Save report to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"health_check_{timestamp}.txt"
        report_path = self.reports_dir / filename
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        return report_path
    
    def interpolate_missing_data(self, data: vbt.Data, exchange: str, market: str, timeframe: str, 
                                strategy: str = 'financial_forward_fill') -> Optional[vbt.Data]:
        """
        Interpolate missing data points in time series with financial data-appropriate strategies.
        
        Args:
            data: VBT Data object to interpolate
            exchange: Exchange identifier
            market: Market type
            timeframe: Timeframe identifier  
            strategy: Interpolation strategy ('financial_forward_fill', 'linear', 'time_aware')
            
        Returns:
            Interpolated VBT Data object or None if interpolation fails
        """
        try:
            # Extract OHLCV data
            ohlcv = VBTDataHandler.extract_ohlcv(data)
            if not ohlcv or 'close' not in ohlcv:
                print(f"   ❌ Cannot interpolate: Missing OHLCV data")
                return None
            
            # Get the original symbols
            symbols = list(data.symbols) if hasattr(data, 'symbols') else []
            if not symbols:
                print(f"   ❌ Cannot interpolate: No symbols found")
                return None
            
            print(f"   🔄 Interpolating missing data using '{strategy}' strategy...")
            
            # Get the expected frequency for this timeframe
            tf_minutes = FreshnessChecker.parse_timeframe_minutes(timeframe)
            freq_map = {
                1: '1min', 5: '5min', 15: '15min', 30: '30min',
                60: '1H', 120: '2H', 240: '4H', 360: '6H', 480: '8H', 720: '12H',
                1440: '1D', 4320: '3D', 10080: '1W'
            }
            pandas_freq = freq_map.get(tf_minutes, f'{tf_minutes}min')
            
            # Create complete time index
            original_index = ohlcv['close'].index
            if len(original_index) < 2:
                print(f"   ❌ Insufficient data for interpolation")
                return None
            
            # Generate complete time range
            complete_index = pd.date_range(
                start=original_index[0],
                end=original_index[-1], 
                freq=pandas_freq
            )
            
            original_len = len(original_index)
            complete_len = len(complete_index)
            missing_count = complete_len - original_len
            
            if missing_count == 0:
                print(f"   ✅ No missing periods found")
                return data
            
            print(f"   📊 Original: {original_len} periods, Complete: {complete_len} periods (+{missing_count} interpolated)")
            
            # Interpolate each OHLCV component
            interpolated_ohlcv = {}
            for component in ['open', 'high', 'low', 'close', 'volume']:
                if component not in ohlcv:
                    continue
                    
                df = ohlcv[component]
                
                # Reindex to complete time range (creates NaN for missing periods)
                reindexed = df.reindex(complete_index)
                
                if strategy == 'financial_forward_fill':
                    # Financial-appropriate interpolation
                    if component == 'volume':
                        # Volume should be 0 during gaps (no trading)
                        interpolated_ohlcv[component] = reindexed.fillna(0)
                    else:
                        # OHLC: Forward fill (last known price)
                        # During gaps: Open=High=Low=Close=last_close, Volume=0
                        filled = reindexed.ffill()
                        interpolated_ohlcv[component] = filled
                        
                elif strategy == 'linear':
                    # Linear interpolation for all components
                    interpolated_ohlcv[component] = reindexed.interpolate(method='linear')
                    
                elif strategy == 'time_aware':
                    # Time-aware interpolation considering time gaps
                    interpolated_ohlcv[component] = reindexed.interpolate(method='time')
                    
                else:
                    # Default to forward fill
                    interpolated_ohlcv[component] = reindexed.ffill()
            
            # Special handling for financial data consistency
            if strategy == 'financial_forward_fill':
                # Ensure OHLC consistency during interpolated periods
                for symbol in symbols:
                    # Find interpolated periods (where we added data)
                    original_mask = interpolated_ohlcv['close'].index.isin(original_index)
                    interpolated_mask = ~original_mask
                    
                    if interpolated_mask.any() and symbol in interpolated_ohlcv['close'].columns:
                        # For interpolated periods: Open=High=Low=Close=previous_close
                        prev_close = interpolated_ohlcv['close'][symbol].ffill()
                        
                        for component in ['open', 'high', 'low']:
                            if component in interpolated_ohlcv and symbol in interpolated_ohlcv[component].columns:
                                interpolated_ohlcv[component].loc[interpolated_mask, symbol] = prev_close[interpolated_mask]
                        
                        # Volume remains 0 for interpolated periods (already set above)
            
            # CRITICAL FIX: Create VBT Data using the correct symbol-based structure
            try:
                # Convert OHLCV-based structure to symbol-based structure
                symbol_dict = {}
                
                for symbol in symbols:
                    # Create a DataFrame for this symbol with OHLCV as columns
                    symbol_data = pd.DataFrame(index=interpolated_ohlcv['close'].index)
                    
                    # Add each OHLCV component as a column
                    for component in ['open', 'high', 'low', 'close', 'volume']:
                        if component in interpolated_ohlcv and symbol in interpolated_ohlcv[component].columns:
                            # Capitalize column names for VBT convention
                            symbol_data[component.capitalize()] = interpolated_ohlcv[component][symbol]
                    
                    # Only include symbols that have at least OHLC data
                    if len(symbol_data.columns) >= 4:
                        symbol_dict[symbol] = symbol_data
                
                if not symbol_dict:
                    print(f"   ❌ No valid symbol data after interpolation")
                    return None
                
                # Create new VBT data object using the symbol-based structure
                new_data = VBTDataHandler.create_from_dict(symbol_dict)
                
                if new_data is None:
                    print(f"   ❌ Failed to create VBT Data object from interpolated data")
                    return None
                
                print(f"   ✅ Interpolation complete: +{missing_count} periods filled")
                return new_data
                
            except Exception as e:
                print(f"   ❌ Failed to create VBT Data object: {e}")
                import traceback
                print(f"   Debug: {traceback.format_exc()}")
                return None
                
        except Exception as e:
            print(f"   ❌ Interpolation failed: {e}")
            import traceback
            print(f"   Debug: {traceback.format_exc()}")
            return None
    
    def apply_interpolation_fix(self, data: vbt.Data, exchange: str, market: str, timeframe: str, 
                               file_info: Dict[str, Any], gap_info: Dict[str, Any]) -> bool:
        """Apply interpolation to fix data gaps and save the result."""
        try:
            # Only interpolate if gaps are reasonable in size
            total_missing = gap_info.get('total_missing', 0)
            if total_missing == 0:
                return True
                
            # Don't interpolate massive gaps (likely data source issues)
            if total_missing > 10000:  # More than ~7 days of 1m data
                print(f"   ⚠️  Skipping interpolation: {total_missing} missing periods too large")
                return False
            
            # Perform interpolation
            interpolated_data = self.interpolate_missing_data(data, exchange, market, timeframe)
            
            if interpolated_data is None:
                return False
            
            # Save interpolated data
            success = data_storage.save_data(interpolated_data, exchange, timeframe, market)
            
            if success:
                print(f"   💾 Saved interpolated data to storage")
                return True
            else:
                print(f"   ❌ Failed to save interpolated data")
                return False
                
        except Exception as e:
            print(f"   ❌ Interpolation fix failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description="Streamlined VBT data health check with realistic gap resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Basic health check
  %(prog)s --auto-fix                   # Check and fix issues automatically
  %(prog)s --exchange binance           # Check specific exchange
  %(prog)s --critical-only              # Show only critical issues
  %(prog)s --detailed                   # Show detailed analysis
  %(prog)s --interpolate --auto-fix     # Fill minor gaps with interpolation
  %(prog)s --interpolate --interpolation-strategy linear  # Use linear interpolation
        """
    )
    
    parser.add_argument('--exchange', type=str,
                      help='Check specific exchange only')
    parser.add_argument('--timeframe', type=str,
                      help='Check specific timeframe only')
    parser.add_argument('--market', type=str, choices=['spot', 'swap'],
                      help='Check specific market type')
    parser.add_argument('--detailed', action='store_true',
                      help='Show detailed analysis')
    parser.add_argument('--auto-fix', action='store_true',
                      help='Automatically fix critical issues')
    parser.add_argument('--interpolate', action='store_true',
                      help='Enable interpolation to fill minor data gaps')
    parser.add_argument('--interpolation-strategy', type=str, 
                      choices=['financial_forward_fill', 'linear', 'time_aware'], 
                      default='financial_forward_fill',
                      help='Interpolation strategy for filling gaps (default: financial_forward_fill)')
    parser.add_argument('--critical-only', action='store_true',
                      help='Show only critical issues')
    parser.add_argument('--save-report', action='store_true',
                      help='Save report to file')
    
    args = parser.parse_args()
    
    print("🔍 Starting Streamlined Data Health Check...")
    if args.auto_fix:
        if args.interpolate:
            print(f"🔧 Auto-fix enabled with interpolation ({args.interpolation_strategy})")
        else:
            print("🔧 Auto-fix enabled - will resolve critical issues")
    elif args.interpolate:
        print(f"📊 Interpolation analysis enabled ({args.interpolation_strategy}) - use --auto-fix to apply")
    if args.critical_only:
        print("⚠️  Critical-only mode - focusing on urgent issues")
    print("=" * 60)
    
    # Initialize checker
    checker = StreamlinedHealthChecker(
        detailed=args.detailed,
        auto_fix=args.auto_fix,
        critical_only=args.critical_only,
        enable_interpolation=args.interpolate,
        interpolation_strategy=args.interpolation_strategy
    )
    
    # Check cache integrity first
    print("🔍 Checking cache integrity...")
    checker.check_cache_integrity()
    
    # Get available data files
    available_data = data_storage.list_available_data()
    
    if not available_data:
        print("❌ No cached data found")
        return 1
    
    # Filter files based on arguments
    filtered_files = []
    for file_info in available_data:
        if args.exchange and file_info['exchange'] != args.exchange:
            continue
        if args.timeframe and file_info['timeframe'] != args.timeframe:
            continue
        if args.market and file_info['market'] != args.market:
            continue
        filtered_files.append(file_info)
    
    if not filtered_files:
        print("❌ No data files match the specified criteria")
        return 1
    
    print(f"📁 Analyzing {len(filtered_files)} data files...")
    print()
    
    # Analyze each file
    analyses = []
    for file_info in filtered_files:
        analysis = checker.analyze_storage_file(file_info)
        analyses.append(analysis)
    
    print()
    print("=" * 60)
    
    # Generate and display report
    report = checker.generate_report(analyses)
    
    if args.save_report:
        report_path = checker.save_report(report)
        print(f"📄 Report saved to: {report_path}")
    else:
        print(report)
    
    # Return appropriate exit code
    if len(checker.issues['critical']) > 0:
        return 2  # Critical issues
    elif len(checker.issues['warning']) > 0:
        return 1  # Warnings
    else:
        return 0  # All good

if __name__ == "__main__":
    exit(main()) 