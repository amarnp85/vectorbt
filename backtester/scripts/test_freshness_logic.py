#!/usr/bin/env python3
"""Test Freshness Logic

Demonstrate the current freshness thresholds and their impact on updates.
"""

import pandas as pd
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def test_freshness_logic():
    """Test the freshness logic for different timeframes."""
    
    console.print(Panel(
        "[bold cyan]Freshness Logic Test[/bold cyan]\n"
        "Testing when data is considered 'stale' and needs updating",
        title="🕐 Freshness Test",
        style="blue"
    ))
    
    # Current thresholds from _is_data_fresh()
    thresholds = {
        '1m': 1800,   # 30 minutes
        '5m': 1800,   # 30 minutes  
        '15m': 1800,  # 30 minutes
        '30m': 1800,  # 30 minutes
        '1h': 7200,   # 2 hours
        '4h': 7200,   # 2 hours
        '1d': 86400,  # 1 day (same day check)
    }
    
    # Create test scenarios
    table = Table(title="Freshness Threshold Analysis", box=box.ROUNDED)
    table.add_column("Timeframe", style="cyan")
    table.add_column("Current Threshold", style="yellow")
    table.add_column("Candles Before Update", style="magenta")
    table.add_column("Issue?", style="red")
    table.add_column("Suggested Threshold", style="green")
    
    for tf, threshold_seconds in thresholds.items():
        # Parse timeframe
        if tf.endswith('m'):
            minutes = int(tf[:-1])
            candle_seconds = minutes * 60
        elif tf.endswith('h'):
            hours = int(tf[:-1])
            candle_seconds = hours * 3600
        elif tf.endswith('d'):
            days = int(tf[:-1])
            candle_seconds = days * 86400
        else:
            candle_seconds = 60  # Default to 1 minute
        
        # Calculate how many candles before update
        candles_before_update = threshold_seconds / candle_seconds
        
        # Determine if there's an issue
        issue = "YES" if candles_before_update > 1.5 else "NO"
        
        # Suggest better threshold (1 candle + small buffer)
        suggested_threshold = candle_seconds * 1.1  # 10% buffer
        
        # Format for display
        threshold_display = format_duration(threshold_seconds)
        suggested_display = format_duration(suggested_threshold)
        
        table.add_row(
            tf,
            threshold_display,
            f"{candles_before_update:.1f}",
            issue,
            suggested_display
        )
    
    console.print(table)
    
    # Show specific 15m scenario
    console.print("\n[bold yellow]📊 15m Timeframe Scenario (Your Test Case):[/bold yellow]")
    console.print("1. First fetch at 14:00:00 - BTC/USDT fetched")
    console.print("2. Wait for new candle at 14:15:00 (15 minutes later)")
    console.print("3. Second fetch with ETH/USDT added")
    console.print("4. Time gap: 15 minutes < 30 minutes threshold")
    console.print("5. Result: BTC/USDT NOT updated (considered 'fresh')")
    console.print("6. Only ETH/USDT fetched with latest data")
    
    console.print("\n[bold red]❌ Problem:[/bold red] The 30-minute threshold means:")
    console.print("- You need to wait 2+ candles before updates happen")
    console.print("- This causes stale data in mixed fetch scenarios")
    console.print("- Symbols show NaN for the latest candle")
    
    console.print("\n[bold green]✅ Solution:[/bold green] Tighter freshness thresholds:")
    console.print("- 15m: 16.5 minutes (1 candle + buffer)")
    console.print("- 5m: 5.5 minutes (1 candle + buffer)")
    console.print("- 1h: 66 minutes (1 candle + buffer)")

def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds >= 3600:
        hours = seconds / 3600
        return f"{hours:.1f} hours"
    elif seconds >= 60:
        minutes = seconds / 60
        return f"{minutes:.0f} minutes"
    else:
        return f"{seconds:.0f} seconds"

if __name__ == "__main__":
    test_freshness_logic() 