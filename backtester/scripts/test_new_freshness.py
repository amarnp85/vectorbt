#!/usr/bin/env python3
"""Test New Freshness Logic

Test the improved timeframe-aware freshness thresholds.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def parse_timeframe_seconds(tf: str) -> int:
    """Parse timeframe string to seconds (same logic as in _is_data_fresh)."""
    tf = tf.lower().strip()
    
    if tf.endswith('s'):
        return int(tf[:-1])
    elif tf.endswith('m') or tf.endswith('min'):
        minutes = int(tf[:-3] if tf.endswith('min') else tf[:-1])
        return minutes * 60
    elif tf.endswith('h') or tf.endswith('hour'):
        hours = int(tf[:-4] if tf.endswith('hour') else tf[:-1])
        return hours * 3600
    elif tf.endswith('d') or tf.endswith('day'):
        days = int(tf[:-3] if tf.endswith('day') else tf[:-1])
        return days * 86400
    elif tf.endswith('w') or tf.endswith('week'):
        weeks = int(tf[:-4] if tf.endswith('week') else tf[:-1])
        return weeks * 604800  # 7 days
    else:
        # Default fallback for unknown formats
        return 3600  # 1 hour

def test_new_freshness_logic():
    """Test the new timeframe-aware freshness logic."""
    
    console.print(Panel(
        "[bold cyan]New Timeframe-Aware Freshness Logic Test[/bold cyan]\n"
        "Testing improved thresholds that scale with timeframe duration",
        title="🎯 Updated Freshness Test",
        style="green"
    ))
    
    # Test various timeframes
    timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']
    
    # Create comparison table
    table = Table(title="New vs Old Freshness Thresholds", box=box.ROUNDED)
    table.add_column("Timeframe", style="cyan")
    table.add_column("Candle Duration", style="blue")
    table.add_column("NEW Threshold (0.9x)", style="green")
    table.add_column("OLD Threshold", style="red")
    table.add_column("Candles Before Update", style="magenta")
    table.add_column("Improvement", style="green")
    
    for tf in timeframes:
        candle_seconds = parse_timeframe_seconds(tf)
        new_threshold = candle_seconds * 0.9
        
        # Old thresholds
        if tf.endswith('m'):
            old_threshold = 1800  # 30 minutes
        elif tf.endswith('h'):
            old_threshold = 7200  # 2 hours
        elif tf.endswith('d'):
            old_threshold = 86400  # 1 day
        else:
            old_threshold = 3600  # 1 hour
        
        candles_before_update = new_threshold / candle_seconds
        
        # Format durations
        candle_duration = format_duration(candle_seconds)
        new_threshold_str = format_duration(new_threshold)
        old_threshold_str = format_duration(old_threshold)
        
        # Determine improvement
        if new_threshold < old_threshold:
            improvement = "✅ BETTER"
        elif new_threshold == old_threshold:
            improvement = "= SAME"
        else:
            improvement = "⚠️ LONGER"
        
        table.add_row(
            tf,
            candle_duration,
            new_threshold_str,
            old_threshold_str,
            f"{candles_before_update:.1f}",
            improvement
        )
    
    console.print(table)
    
    # Highlight the 15m scenario
    console.print("\n[bold yellow]🎯 15m Scenario Analysis:[/bold yellow]")
    
    tf_15m_seconds = parse_timeframe_seconds("15m")
    new_15m_threshold = tf_15m_seconds * 0.9
    
    console.print(f"- 15m candle duration: {tf_15m_seconds} seconds ({tf_15m_seconds/60:.0f} minutes)")
    console.print(f"- NEW freshness threshold: {new_15m_threshold} seconds ({new_15m_threshold/60:.1f} minutes)")
    console.print(f"- OLD freshness threshold: 1800 seconds (30 minutes)")
    
    console.print("\n[bold green]✅ Expected behavior with NEW logic:[/bold green]")
    console.print("1. First fetch at 14:00:00 - BTC/USDT fetched")
    console.print("2. Wait for new candle at 14:15:00 (15 minutes = 900 seconds)")
    console.print(f"3. Second fetch: 900s > {new_15m_threshold:.0f}s = Data is STALE")
    console.print("4. BTC/USDT will be UPDATED before adding ETH/USDT")
    console.print("5. Both symbols will have latest data!")
    
    console.print("\n[bold red]❌ OLD behavior:[/bold red]")
    console.print("3. Second fetch: 900s < 1800s = Data is FRESH")
    console.print("4. BTC/USDT NOT updated, only ETH/USDT fetched")
    console.print("5. BTC/USDT shows NaN for latest candle")

def format_duration(seconds):
    """Format seconds into human-readable duration."""
    if seconds >= 3600:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    elif seconds >= 60:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        return f"{seconds:.0f}s"

if __name__ == "__main__":
    test_new_freshness_logic() 