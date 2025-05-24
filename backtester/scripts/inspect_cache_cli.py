#!/usr/bin/env python3
"""Cache Inspection CLI - Inspect and validate cached VBT data.

This script provides commands to examine cached data, show symbol information,
and validate cache integrity.
"""

import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import pandas as pd
import vectorbtpro as vbt

from backtester.data.storage.data_storage import data_storage
from backtester.data.fetching.core.vbt_data_handler import VBTDataHandler
from backtester.data.fetching.core.freshness_checker import FreshnessChecker
from backtester.data.cache_system import cache_manager

console = Console()
logger = logging.getLogger(__name__)


def show_comprehensive_symbol_analysis(exchange: str, market: str, timeframe: str):
    """Show comprehensive analysis of all symbols in cache."""
    utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    console.print(
        Panel(
            f"[bold cyan]Comprehensive Symbol Analysis[/bold cyan]\n"
            f"[cyan]{exchange.upper()}/{market.upper()}/{timeframe}[/cyan]\n"
            f"[dim]Analysis Time: {utc_time}[/dim]",
            title="🔍 Complete Cache Analysis",
            expand=False,
            style="cyan"
        )
    )
    
    try:
        # Load the data
        data = data_storage.load_data(exchange, timeframe, [], market)
        
        if data is None:
            console.print(f"[red]No cached data found for {exchange}/{market}/{timeframe}[/red]")
            return
        
        # Get all symbols from cache
        all_symbols = list(data.symbols) if hasattr(data, 'symbols') else []
        
        if not all_symbols:
            console.print(f"[red]No symbols found in cached data[/red]")
            return
        
        # Get inception timestamps for comparison
        cached_timestamps = cache_manager.get_all_timestamps(exchange)
        
        # Create comprehensive analysis table
        analysis_table = Table(title=f"📊 Symbol Analysis ({utc_time})", box=box.ROUNDED)
        analysis_table.add_column("Symbol", style="cyan", no_wrap=True)
        analysis_table.add_column("Inception Check", justify="center")
        analysis_table.add_column("Earliest", style="green")
        analysis_table.add_column("Latest", style="yellow")
        analysis_table.add_column("Candles", justify="right", style="magenta")
        analysis_table.add_column("Expected", justify="right", style="dim")
        analysis_table.add_column("Completeness", justify="center")
        analysis_table.add_column("Data Age", style="dim")
        
        current_time = datetime.utcnow()
        tf_minutes = FreshnessChecker.parse_timeframe_minutes(timeframe)
        
        # Track completeness stats
        complete_count = 0
        gap_count = 0
        error_count = 0
        
        console.print(f"[dim]Analyzing {len(all_symbols)} symbols with {tf_minutes}-minute intervals...[/dim]")
        
        for symbol in sorted(all_symbols):
            try:
                # Extract OHLCV data to access symbol data properly
                ohlcv_data = VBTDataHandler.extract_ohlcv(data)
                
                if not ohlcv_data or 'close' not in ohlcv_data:
                    analysis_table.add_row(
                        symbol, "[red]❌ No Data[/red]", "-", "-", "0", "0", "[red]❌ Empty[/red]", "-"
                    )
                    error_count += 1
                    continue
                
                # Get symbol's close data
                close_df = ohlcv_data['close']
                if symbol not in close_df.columns:
                    analysis_table.add_row(
                        symbol, "[red]❌ Missing[/red]", "-", "-", "0", "0", "[red]❌ No Column[/red]", "-"
                    )
                    error_count += 1
                    continue
                
                # Get actual symbol data (not NaN-filled)
                symbol_series = close_df[symbol].dropna()
                
                if len(symbol_series) == 0:
                    analysis_table.add_row(
                        symbol, "[red]❌ No Data[/red]", "-", "-", "0", "0", "[red]❌ Empty[/red]", "-"
                    )
                    error_count += 1
                    continue
                
                # Get ACTUAL per-symbol date range from the symbol's non-null data
                earliest_ts = symbol_series.index[0]
                latest_ts = symbol_series.index[-1]
                actual_candles = len(symbol_series)
                
                # Verify earliest timestamp by checking inception
                inception_check = "❓ Unknown"
                actual_earliest_str = earliest_ts.strftime('%Y-%m-%d')
                
                if symbol in cached_timestamps:
                    inception_ms = cached_timestamps[symbol]
                    inception_dt = datetime.fromtimestamp(inception_ms / 1000)
                    
                    # Allow 1 day tolerance for inception comparison
                    earliest_dt = earliest_ts.to_pydatetime().replace(tzinfo=None)
                    inception_diff_days = abs((earliest_dt - inception_dt).days)
                    
                    if inception_diff_days <= 1:
                        inception_check = "[green]✅ Matches[/green]"
                    elif inception_diff_days <= 7:
                        inception_check = "[yellow]⚠️ Close[/yellow]"
                        # Show actual difference
                        actual_earliest_str += f" (vs {inception_dt.strftime('%Y-%m-%d')})"
                    else:
                        inception_check = f"[red]❌ Off by {inception_diff_days}d[/red]"
                        actual_earliest_str += f" (vs {inception_dt.strftime('%Y-%m-%d')})"
                
                # Calculate MORE REALISTIC expected candles by examining actual data patterns
                # Instead of assuming continuous data, calculate based on data density
                time_span = latest_ts - earliest_ts
                total_theoretical_periods = int(time_span.total_seconds() / (tf_minutes * 60)) + 1
                
                # Calculate data density (actual vs theoretical)
                density_ratio = actual_candles / total_theoretical_periods if total_theoretical_periods > 0 else 0
                
                # For crypto markets, we expect high density (>95%) unless there are real gaps
                # Use density to determine if gaps are expected (market structure) vs missing data
                if density_ratio >= 0.98:
                    # Very high density - likely complete
                    expected_candles = total_theoretical_periods
                    completeness = "[green]✅ Complete[/green]"
                    complete_count += 1
                elif density_ratio >= 0.90:
                    # Good density - minor gaps acceptable
                    expected_candles = total_theoretical_periods
                    missing_candles = total_theoretical_periods - actual_candles
                    completeness = f"[yellow]⚠️ {missing_candles} gaps[/yellow]"
                    gap_count += 1
                else:
                    # Low density - significant issues or different market structure
                    # Use actual data + small buffer as "expected" since theoretical might be wrong
                    expected_candles = int(actual_candles * 1.02)  # 2% buffer for minor gaps
                    missing_candles = expected_candles - actual_candles
                    
                    if missing_candles <= 0:
                        completeness = "[green]✅ Dense[/green]"
                        complete_count += 1
                    else:
                        completeness = f"[red]❌ {missing_candles} gaps[/red]"
                        gap_count += 1
                
                # Alternative: Check for large gaps in the actual data
                if len(symbol_series) > 1:
                    time_diffs = pd.Series(symbol_series.index).diff().dropna()
                    expected_diff = timedelta(minutes=tf_minutes)
                    large_gaps = time_diffs[time_diffs > expected_diff * 2]  # Gaps > 2x expected
                    
                    if len(large_gaps) > 0:
                        gap_hours = large_gaps.dt.total_seconds().sum() / 3600
                        if gap_hours > 24:  # More than 1 day of gaps
                            completeness = f"[red]❌ {len(large_gaps)} large gaps ({gap_hours:.0f}h)[/red]"
                            gap_count += 1
                            complete_count -= 1 if completeness.startswith("[green]") else 0
                
                # Calculate data age
                latest_dt = latest_ts.to_pydatetime().replace(tzinfo=None)
                age_delta = current_time - latest_dt
                age_hours = age_delta.total_seconds() / 3600
                
                if age_hours < 2:
                    age_str = f"{age_hours:.1f}h"
                elif age_hours < 48:
                    age_str = f"{age_hours:.1f}h"
                else:
                    age_str = f"{age_hours/24:.1f}d"
                
                # Format dates
                latest_str = latest_ts.strftime('%m-%d %H:%M')
                
                analysis_table.add_row(
                    symbol,
                    inception_check,
                    actual_earliest_str,
                    latest_str,
                    str(actual_candles),
                    str(expected_candles),
                    completeness,
                    age_str
                )
                
            except Exception as e:
                analysis_table.add_row(
                    symbol, "[red]❌ Error[/red]", "-", "-", "-", "-", f"[red]Error: {str(e)[:20]}...[/red]", "-"
                )
                error_count += 1
        
        console.print(analysis_table)
        
        # Enhanced summary statistics
        summary_table = Table(title="📋 Analysis Summary", box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", justify="right")
        summary_table.add_column("Details", style="dim")
        
        summary_table.add_row("Total Symbols", str(len(all_symbols)), "In cached data")
        summary_table.add_row("Complete Data", str(complete_count), "No significant gaps")
        summary_table.add_row("With Gaps", str(gap_count), "Missing some candles")
        summary_table.add_row("Errors", str(error_count), "Analysis failed")
        
        # Overall health score
        if len(all_symbols) > 0:
            health_score = (complete_count / len(all_symbols)) * 100
            if health_score >= 90:
                health_status = f"[green]✅ Excellent ({health_score:.1f}%)[/green]"
            elif health_score >= 75:
                health_status = f"[yellow]⚠️ Good ({health_score:.1f}%)[/yellow]"
            else:
                health_status = f"[red]❌ Poor ({health_score:.1f}%)[/red]"
        else:
            health_status = "[red]❌ No Data[/red]"
        
        summary_table.add_row("Cache Health", health_status, "Overall data quality")
        
        # Add timeframe info for context
        summary_table.add_row("Timeframe", f"{timeframe} ({tf_minutes}min)", "Data resolution analyzed")
        
        console.print(summary_table)
        
    except Exception as e:
        console.print(f"[red]Error during comprehensive analysis: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


def list_cached_data():
    """List all available cached data files."""
    utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    console.print(
        Panel(
            f"[bold cyan]Cache Inspection[/bold cyan]\n"
            f"[dim]Timestamp: {utc_time}[/dim]",
            title="💾 VBT Data Cache",
            expand=False,
            style="cyan"
        )
    )
    
    # Get available data files
    try:
        available_data = data_storage.list_available_data()
        
        if not available_data:
            console.print("[yellow]No cached data found[/yellow]")
            return
        
        # Create table of available data
        cache_table = Table(title=f"📊 Available Cached Data ({utc_time})", box=box.ROUNDED)
        cache_table.add_column("Exchange", style="cyan")
        cache_table.add_column("Market", style="green")
        cache_table.add_column("Timeframe", style="yellow")
        cache_table.add_column("Symbols", justify="right")
        cache_table.add_column("Date Range", style="dim")
        cache_table.add_column("File Size", justify="right", style="dim")
        
        for entry in available_data:
            # Load basic info about each cached file
            try:
                data = data_storage.load_data(
                    entry['exchange'], entry['timeframe'], [], entry['market']
                )
                
                if data and hasattr(data, 'symbols'):
                    symbol_count = len(data.symbols)
                    
                    # Get date range
                    try:
                        start_date, end_date = VBTDataHandler.get_date_range(data)
                        date_range = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
                    except Exception:
                        date_range = "Unknown"
                    
                    cache_table.add_row(
                        entry['exchange'].upper(),
                        entry['market'].upper(),
                        entry['timeframe'],
                        str(symbol_count),
                        date_range,
                        entry.get('file_size', 'Unknown')
                    )
                else:
                    cache_table.add_row(
                        entry['exchange'].upper(),
                        entry['market'].upper(),
                        entry['timeframe'],
                        "0",
                        "No data",
                        entry.get('file_size', 'Unknown')
                    )
            except Exception as e:
                cache_table.add_row(
                    entry['exchange'].upper(),
                    entry['market'].upper(),
                    entry['timeframe'],
                    "Error",
                    f"Failed to load: {str(e)[:30]}...",
                    entry.get('file_size', 'Unknown')
                )
        
        console.print(cache_table)
        
    except Exception as e:
        console.print(f"[red]Error listing cached data: {e}[/red]")


def inspect_cache_detail(exchange: str, market: str, timeframe: str, symbols: Optional[List[str]] = None):
    """Inspect detailed cache contents for specific exchange/market/timeframe."""
    utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    console.print(
        Panel(
            f"[bold cyan]Detailed Cache Inspection[/bold cyan]\n"
            f"[cyan]{exchange.upper()}/{market.upper()}/{timeframe}[/cyan]\n"
            f"[dim]Timestamp: {utc_time}[/dim]",
            title="🔍 Cache Details",
            expand=False,
            style="cyan"
        )
    )
    
    try:
        # Load the data
        data = data_storage.load_data(exchange, timeframe, [], market)
        
        if data is None:
            console.print(f"[red]No cached data found for {exchange}/{market}/{timeframe}[/red]")
            return
        
        # Get basic information
        all_symbols = list(data.symbols) if hasattr(data, 'symbols') else []
        
        if symbols:
            # Filter to requested symbols
            available_symbols = [s for s in symbols if s in all_symbols]
            missing_symbols = [s for s in symbols if s not in all_symbols]
            
            if missing_symbols:
                console.print(f"[yellow]Warning: Symbols not found in cache: {missing_symbols}[/yellow]")
            
            if not available_symbols:
                console.print(f"[red]None of the requested symbols found in cache[/red]")
                return
                
            symbols_to_show = available_symbols
        else:
            symbols_to_show = all_symbols
        
        # Create overview table
        overview_table = Table(title=f"📋 Cache Overview ({utc_time})", box=box.ROUNDED)
        overview_table.add_column("Metric", style="cyan")
        overview_table.add_column("Value", justify="right")
        overview_table.add_column("Details", style="dim")
        
        # Basic metrics
        overview_table.add_row("Exchange", exchange.upper(), "Data source")
        overview_table.add_row("Market", market.upper(), "Market type")
        overview_table.add_row("Timeframe", timeframe, "Data resolution")
        overview_table.add_row("Total Symbols", str(len(all_symbols)), "All cached symbols")
        overview_table.add_row("Showing", str(len(symbols_to_show)), "Selected for display")
        
        # Date range and data points
        try:
            start_date, end_date = VBTDataHandler.get_date_range(data)
            total_candles = len(data.index) if hasattr(data, 'index') else 0
            
            overview_table.add_row("Start Date", start_date.strftime('%Y-%m-%d %H:%M'), "First data point")
            overview_table.add_row("End Date", end_date.strftime('%Y-%m-%d %H:%M'), "Last data point")
            overview_table.add_row("Total Candles", str(total_candles), "Data points per symbol")
            
            # Calculate age
            age_hours = (datetime.utcnow().replace(tzinfo=None) - end_date.replace(tzinfo=None)).total_seconds() / 3600
            if age_hours < 1:
                age_str = f"{age_hours * 60:.0f} minutes"
            elif age_hours < 24:
                age_str = f"{age_hours:.1f} hours"
            else:
                age_str = f"{age_hours / 24:.1f} days"
            
            overview_table.add_row("Data Age", age_str, "Time since last update")
            
        except Exception as e:
            overview_table.add_row("Date Info", "Error", f"Could not determine: {e}")
        
        console.print(overview_table)
        
        # Show symbol details only if specific symbols requested
        if symbols:
            show_symbol_tails(data, symbols_to_show[:10])  # Limit to 10 symbols for display
            
            if len(symbols_to_show) > 10:
                console.print(f"[dim]Note: Showing first 10 of {len(symbols_to_show)} symbols. Use --symbols to specify particular symbols.[/dim]")
        else:
            # If no specific symbols requested, show comprehensive analysis for all symbols
            console.print(f"\n[dim]Showing comprehensive analysis for all {len(all_symbols)} symbols...[/dim]")
            show_comprehensive_symbol_analysis(exchange, market, timeframe)
        
    except Exception as e:
        console.print(f"[red]Error inspecting cache: {e}[/red]")


def show_symbol_tails(data: vbt.Data, symbols: List[str], tail_length: int = 5):
    """Show the tail (last N rows) of data for each symbol."""
    utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    console.print(
        Panel(
            f"[bold cyan]Recent Data (Last {tail_length} candles)[/bold cyan]\n"
            f"[dim]Timestamp: {utc_time}[/dim]",
            title="📈 Symbol Data Tails",
            expand=False,
            style="green"
        )
    )
    
    try:
        # Extract OHLCV data
        ohlcv = VBTDataHandler.extract_ohlcv(data)
        
        if not ohlcv:
            console.print("[red]Could not extract OHLCV data[/red]")
            return
        
        for symbol in symbols:
            # Create table for this symbol
            symbol_table = Table(title=f"📊 {symbol} (Last {tail_length} candles)", box=box.MINIMAL_DOUBLE_HEAD)
            symbol_table.add_column("Date", style="cyan")
            symbol_table.add_column("Open", justify="right", style="green")
            symbol_table.add_column("High", justify="right", style="green")
            symbol_table.add_column("Low", justify="right", style="red")
            symbol_table.add_column("Close", justify="right", style="yellow")
            symbol_table.add_column("Volume", justify="right", style="dim")
            
            try:
                # Get symbol data
                symbol_df = pd.DataFrame()
                
                for feature in ['open', 'high', 'low', 'close', 'volume']:
                    if feature in ohlcv and ohlcv[feature] is not None:
                        if symbol in ohlcv[feature].columns:
                            symbol_df[feature.capitalize()] = ohlcv[feature][symbol]
                
                if symbol_df.empty:
                    symbol_table.add_row("No data", "-", "-", "-", "-", "-")
                else:
                    # Get last N rows
                    tail_data = symbol_df.tail(tail_length)
                    
                    for idx, (timestamp, row) in enumerate(tail_data.iterrows()):
                        date_str = timestamp.strftime('%m-%d %H:%M')
                        
                        # Format values
                        open_val = f"{row.get('Open', 0):.4f}" if pd.notna(row.get('Open')) else "-"
                        high_val = f"{row.get('High', 0):.4f}" if pd.notna(row.get('High')) else "-"
                        low_val = f"{row.get('Low', 0):.4f}" if pd.notna(row.get('Low')) else "-"
                        close_val = f"{row.get('Close', 0):.4f}" if pd.notna(row.get('Close')) else "-"
                        volume_val = f"{row.get('Volume', 0):,.0f}" if pd.notna(row.get('Volume')) else "-"
                        
                        symbol_table.add_row(date_str, open_val, high_val, low_val, close_val, volume_val)
                
                console.print(symbol_table)
                
            except Exception as e:
                console.print(f"[red]Error showing data for {symbol}: {e}[/red]")
    
    except Exception as e:
        console.print(f"[red]Error showing symbol tails: {e}[/red]")


def validate_cache_integrity(exchange: str, market: str, timeframe: str):
    """Validate cache integrity and completeness."""
    utc_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
    
    console.print(
        Panel(
            f"[bold cyan]Cache Validation[/bold cyan]\n"
            f"[cyan]{exchange.upper()}/{market.upper()}/{timeframe}[/cyan]\n"
            f"[dim]Timestamp: {utc_time}[/dim]",
            title="🔍 Integrity Check",
            expand=False,
            style="blue"
        )
    )
    
    try:
        # Load the data
        data = data_storage.load_data(exchange, timeframe, [], market)
        
        if data is None:
            console.print(f"[red]No cached data found for {exchange}/{market}/{timeframe}[/red]")
            return
        
        # Create validation table
        validation_table = Table(title=f"🔍 Validation Results ({utc_time})", box=box.ROUNDED)
        validation_table.add_column("Check", style="cyan")
        validation_table.add_column("Status")
        validation_table.add_column("Details", style="dim")
        
        all_symbols = list(data.symbols) if hasattr(data, 'symbols') else []
        
        # Basic structure checks
        if all_symbols:
            validation_table.add_row("✅ Symbols", "Pass", f"{len(all_symbols)} symbols found")
        else:
            validation_table.add_row("❌ Symbols", "Fail", "No symbols in data")
        
        # Index continuity check
        try:
            if hasattr(data, 'index') and len(data.index) > 1:
                # Check for time gaps
                time_diffs = pd.Series(data.index).diff().dropna()
                expected_diff = time_diffs.mode().iloc[0] if not time_diffs.empty else None
                
                if expected_diff:
                    # Allow some tolerance for weekends/holidays
                    tolerance = expected_diff * 2
                    large_gaps = time_diffs[time_diffs > tolerance]
                    
                    if len(large_gaps) == 0:
                        validation_table.add_row("✅ Continuity", "Pass", f"No significant time gaps found")
                    else:
                        validation_table.add_row("⚠️ Continuity", "Warning", f"{len(large_gaps)} potential gaps found")
                else:
                    validation_table.add_row("❌ Continuity", "Fail", "Could not determine expected interval")
            else:
                validation_table.add_row("❌ Continuity", "Fail", "Insufficient data for continuity check")
        except Exception as e:
            validation_table.add_row("❌ Continuity", "Error", f"Check failed: {str(e)[:30]}...")
        
        # Data completeness per symbol
        try:
            ohlcv = VBTDataHandler.extract_ohlcv(data)
            if ohlcv:
                complete_symbols = 0
                incomplete_symbols = 0
                
                for symbol in all_symbols[:5]:  # Check first 5 symbols
                    symbol_complete = True
                    for feature in ['open', 'high', 'low', 'close']:
                        if feature in ohlcv and symbol in ohlcv[feature].columns:
                            null_count = ohlcv[feature][symbol].isnull().sum()
                            if null_count > len(ohlcv[feature]) * 0.1:  # More than 10% null
                                symbol_complete = False
                                break
                    
                    if symbol_complete:
                        complete_symbols += 1
                    else:
                        incomplete_symbols += 1
                
                if incomplete_symbols == 0:
                    validation_table.add_row("✅ Completeness", "Pass", f"All checked symbols complete")
                else:
                    validation_table.add_row("⚠️ Completeness", "Warning", f"{incomplete_symbols} symbols have missing data")
            else:
                validation_table.add_row("❌ Completeness", "Fail", "Could not extract OHLCV data")
        except Exception as e:
            validation_table.add_row("❌ Completeness", "Error", f"Check failed: {str(e)[:30]}...")
        
        # Freshness check
        try:
            start_date, end_date = VBTDataHandler.get_date_range(data)
            age_hours = (datetime.utcnow().replace(tzinfo=None) - end_date.replace(tzinfo=None)).total_seconds() / 3600
            
            if age_hours < 24:
                validation_table.add_row("✅ Freshness", "Fresh", f"Updated {age_hours:.1f} hours ago")
            elif age_hours < 168:  # 1 week
                validation_table.add_row("⚠️ Freshness", "Stale", f"Updated {age_hours/24:.1f} days ago")
            else:
                validation_table.add_row("❌ Freshness", "Old", f"Updated {age_hours/24:.1f} days ago")
        except Exception as e:
            validation_table.add_row("❌ Freshness", "Error", f"Check failed: {str(e)[:30]}...")
        
        console.print(validation_table)
        
    except Exception as e:
        console.print(f"[red]Error validating cache: {e}[/red]")


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Inspect and validate VBT data cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all cached data
    python inspect_cache_cli.py list
    
    # Comprehensive analysis for all symbols (default behavior)
    python inspect_cache_cli.py inspect --exchange binance --market spot --timeframe 1h
    
    # Show recent candle data for specific symbols only
    python inspect_cache_cli.py inspect --exchange binance --market spot --timeframe 1d --symbols BTC/USDT ETH/USDT --tail 10
    
    # Validate cache integrity
    python inspect_cache_cli.py validate --exchange binance --market spot --timeframe 1h
        """
    )
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all cached data files')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Inspect cache contents with comprehensive analysis (default) or specific symbol data')
    inspect_parser.add_argument('--exchange', required=True, help='Exchange ID (e.g., binance)')
    inspect_parser.add_argument('--market', default='spot', help='Market type (default: spot)')
    inspect_parser.add_argument('--timeframe', required=True, help='Timeframe (e.g., 1h, 1d)')
    inspect_parser.add_argument('--symbols', nargs='+', help='Show recent candle data for specific symbols (optional - without this, shows comprehensive analysis for all symbols)')
    inspect_parser.add_argument('--tail', type=int, default=5, help='Number of recent candles to show when using --symbols (default: 5)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate cache integrity')
    validate_parser.add_argument('--exchange', required=True, help='Exchange ID (e.g., binance)')
    validate_parser.add_argument('--market', default='spot', help='Market type (default: spot)')
    validate_parser.add_argument('--timeframe', required=True, help='Timeframe (e.g., 1h, 1d)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'list':
            list_cached_data()
        elif args.command == 'inspect':
            inspect_cache_detail(args.exchange, args.market, args.timeframe, args.symbols)
        elif args.command == 'validate':
            validate_cache_integrity(args.exchange, args.market, args.timeframe)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        logger.exception("Unexpected error occurred")


if __name__ == "__main__":
    main() 