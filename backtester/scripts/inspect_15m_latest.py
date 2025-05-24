#!/usr/bin/env python3
"""Inspect 15m Data Latest Candles

Check if the mixed fetch correctly updated existing symbols and added new ones.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbtpro as vbt
from data.storage.data_storage import data_storage
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from datetime import datetime

console = Console()

def inspect_15m_data():
    """Inspect the 15m data to see if updates worked correctly."""
    
    console.print(Panel(
        "[bold cyan]15m Data Inspection[/bold cyan]\n"
        "Checking latest candles and update status",
        title="🔍 15m Data Inspector",
        style="blue"
    ))
    
    # Load 15m data
    data_15m = data_storage.load_data('binance', '15m', market_type='spot')
    
    if data_15m is None:
        console.print("[red]❌ No 15m data found![/red]")
        return
    
    # Basic info
    console.print(f"\n[cyan]📊 Data shape:[/cyan] {data_15m.wrapper.shape}")
    console.print(f"[cyan]📊 Symbols:[/cyan] {list(data_15m.symbols)}")
    console.print(f"[cyan]📊 Features:[/cyan] {list(data_15m.wrapper.columns)}")
    
    # Check latest timestamps for each symbol
    console.print("\n[bold yellow]⏰ Latest Candles by Symbol:[/bold yellow]")
    
    for symbol in data_15m.symbols:
        console.print(f"\n[bold cyan]{symbol}:[/bold cyan]")
        
        # Get OHLCV data for this symbol
        try:
            # For each OHLCV feature
            features = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Create a table for the last 5 candles
            table = Table(title=f"Last 5 candles for {symbol}", box=box.ROUNDED)
            table.add_column("Timestamp", style="cyan")
            for feature in features:
                table.add_column(feature, style="green" if feature == "Close" else "white")
            
            # Get the data
            last_5_idx = data_15m.wrapper.index[-5:]
            
            for timestamp in last_5_idx:
                row = [str(timestamp)]
                for feature in features:
                    # Access data using VBT's attribute access
                    feature_data = getattr(data_15m, feature.lower())
                    if len(data_15m.symbols) > 1:
                        value = feature_data[symbol].loc[timestamp]
                    else:
                        value = feature_data.loc[timestamp]
                    
                    # Format the value
                    if feature == 'Volume':
                        row.append(f"{value:,.2f}")
                    else:
                        row.append(f"{value:,.2f}")
                
                table.add_row(*row)
            
            console.print(table)
            
            # Print the very latest timestamp
            latest_ts = data_15m.wrapper.index[-1]
            console.print(f"[yellow]Latest timestamp:[/yellow] {latest_ts}")
            
            # Check if this is fresh relative to current time
            now = pd.Timestamp.now('UTC')
            time_diff = now - latest_ts
            console.print(f"[yellow]Time since last candle:[/yellow] {time_diff}")
            
            # Check if we're missing the latest candle
            expected_latest = now.floor('15min')
            if latest_ts < expected_latest - pd.Timedelta(minutes=15):
                console.print(f"[red]⚠️ Missing candles! Expected up to:[/red] {expected_latest - pd.Timedelta(minutes=15)}")
            else:
                console.print(f"[green]✅ Data appears up to date[/green]")
                
        except Exception as e:
            console.print(f"[red]❌ Error accessing {symbol} data: {e}[/red]")
    
    # Check index alignment
    console.print("\n[bold yellow]📅 Index Alignment Check:[/bold yellow]")
    
    # Get unique timestamps count per symbol
    for symbol in data_15m.symbols:
        try:
            # Get close data for this symbol
            if len(data_15m.symbols) > 1:
                symbol_data = data_15m.close[symbol].dropna()
            else:
                symbol_data = data_15m.close.dropna()
            
            first_ts = symbol_data.index[0]
            last_ts = symbol_data.index[-1]
            data_points = len(symbol_data)
            
            console.print(f"\n[cyan]{symbol}:[/cyan]")
            console.print(f"  First timestamp: {first_ts}")
            console.print(f"  Last timestamp: {last_ts}")
            console.print(f"  Data points: {data_points:,}")
            
            # Check for gaps
            expected_points = int((last_ts - first_ts).total_seconds() / 900) + 1  # 900 seconds = 15 minutes
            if data_points < expected_points * 0.95:  # Allow 5% missing for maintenance
                console.print(f"  [yellow]⚠️ Possible gaps: expected ~{expected_points:,} points[/yellow]")
            
        except Exception as e:
            console.print(f"[red]❌ Error checking {symbol}: {e}[/red]")
    
    # Summary
    console.print("\n" + "="*80)
    console.print("[bold green]Summary:[/bold green]")
    console.print(f"- Total symbols: {len(data_15m.symbols)}")
    console.print(f"- Total data points: {len(data_15m.wrapper.index):,}")
    console.print(f"- OHLCV structure: {'✅ Preserved' if len(data_15m.wrapper.columns) == 5 else '❌ Lost'}")
    
    # Check what happened in the logs
    console.print("\n[bold yellow]🔍 Analysis of the fetch behavior:[/bold yellow]")
    console.print("From the logs, the second fetch showed:")
    console.print("- '✅ Cached data is fresh, proceeding with mixed fetch'")
    console.print("- This suggests the freshness check passed even after 15 minutes")
    console.print("- Let's check the freshness logic...")
    
    # Analyze the freshness check
    latest_dt = data_15m.wrapper.index[-1]
    target_dt = pd.Timestamp.now('UTC')
    time_diff = target_dt - latest_dt
    
    console.print(f"\n[cyan]Freshness calculation:[/cyan]")
    console.print(f"- Latest data: {latest_dt}")
    console.print(f"- Current time: {target_dt}")
    console.print(f"- Time difference: {time_diff}")
    console.print(f"- Total seconds: {time_diff.total_seconds():.0f}")
    
    # For 15m data, the freshness check uses 30 minutes threshold
    if time_diff.total_seconds() <= 1800:  # 30 minutes
        console.print("[yellow]⚠️ Data considered 'fresh' (< 30 min old) by current logic[/yellow]")
    else:
        console.print("[green]✅ Data would be considered 'stale' (> 30 min old)[/green]")

if __name__ == "__main__":
    inspect_15m_data() 