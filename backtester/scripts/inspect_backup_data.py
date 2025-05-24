#!/usr/bin/env python3
"""Inspect Backup Data

Check the backup 4h file to understand when data corruption occurred.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbtpro as vbt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def inspect_backup_file():
    """Inspect the backup 4h file."""
    
    backup_path = "/Users/amarpatel/python/backtester/backtester/vbt_data/binance_spot_4h.pickle.blosc.backup_20250523_171607"
    
    console.print(Panel(
        f"[bold cyan]Inspecting Backup File[/bold cyan]\n"
        f"Path: {backup_path}",
        title="🔍 Backup Inspector",
        style="blue"
    ))
    
    if not os.path.exists(backup_path):
        console.print(f"[red]❌ Backup file not found: {backup_path}[/red]")
        return
    
    try:
        # Load the backup data
        backup_data = vbt.Data.load(backup_path)
        
        # Basic info
        info_table = Table(show_header=False)
        info_table.add_column("Property", style="cyan")
        info_table.add_column("Value", style="white")
        
        info_table.add_row("Data Type", str(type(backup_data)))
        
        if hasattr(backup_data, 'symbols'):
            symbols = list(backup_data.symbols)
            info_table.add_row("Symbols Count", str(len(symbols)))
            info_table.add_row("Symbols", str(symbols))
        
        if hasattr(backup_data, 'wrapper'):
            wrapper = backup_data.wrapper
            info_table.add_row("Wrapper Shape", str(wrapper.shape))
            info_table.add_row("Wrapper Columns", str(list(wrapper.columns)))
            info_table.add_row("Index Length", str(len(wrapper.index)))
        
        console.print(info_table)
        
        # Check OHLCV attributes
        ohlcv_attrs = ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_ohlcv = []
        
        for attr in ohlcv_attrs:
            if hasattr(backup_data, attr) and getattr(backup_data, attr) is not None:
                available_ohlcv.append(attr)
        
        console.print(f"\n[green]Available OHLCV attributes: {available_ohlcv}[/green]")
        
        # Check underlying structure
        try:
            underlying = backup_data.get()
            console.print(f"\n[yellow]Underlying type: {type(underlying)}[/yellow]")
            
            if isinstance(underlying, tuple):
                console.print(f"Tuple with {len(underlying)} elements:")
                for i, elem in enumerate(underlying):
                    console.print(f"  Element {i}: {type(elem)} - {elem.shape if hasattr(elem, 'shape') else 'no shape'}")
            elif hasattr(underlying, 'shape'):
                console.print(f"Shape: {underlying.shape}")
                if hasattr(underlying, 'columns'):
                    console.print(f"Columns: {list(underlying.columns)}")
        
        except Exception as e:
            console.print(f"[red]❌ Error getting underlying data: {e}[/red]")
        
        # Compare with current 4h data
        console.print(f"\n[bold]Comparison with Current 4h Data:[/bold]")
        
        current_path = "/Users/amarpatel/python/backtester/backtester/vbt_data/binance_spot_4h.pickle.blosc"
        if os.path.exists(current_path):
            current_data = vbt.Data.load(current_path)
            
            backup_shape = backup_data.wrapper.shape if hasattr(backup_data, 'wrapper') else 'unknown'
            current_shape = current_data.wrapper.shape if hasattr(current_data, 'wrapper') else 'unknown'
            
            backup_cols = list(backup_data.wrapper.columns) if hasattr(backup_data, 'wrapper') else []
            current_cols = list(current_data.wrapper.columns) if hasattr(current_data, 'wrapper') else []
            
            comparison_table = Table(title="Backup vs Current")
            comparison_table.add_column("Aspect", style="cyan")
            comparison_table.add_column("Backup", style="green")
            comparison_table.add_column("Current", style="red")
            
            comparison_table.add_row("Shape", str(backup_shape), str(current_shape))
            comparison_table.add_row("Columns", str(backup_cols), str(current_cols))
            comparison_table.add_row("Symbols", str(len(backup_data.symbols) if hasattr(backup_data, 'symbols') else 0), 
                                   str(len(current_data.symbols) if hasattr(current_data, 'symbols') else 0))
            
            console.print(comparison_table)
        
    except Exception as e:
        console.print(f"[red]❌ Error loading backup file: {e}[/red]")
        import traceback
        console.print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    inspect_backup_file() 