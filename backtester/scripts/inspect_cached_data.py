#!/usr/bin/env python3
"""Inspect Cached Data Structure

This script loads and inspects the cached VBT data files to understand:
1. What data structure is actually stored
2. Whether OHLCV data is properly preserved
3. Symbol availability and naming
4. Data integrity issues
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

console = Console()

def inspect_vbt_data(data, title: str):
    """Inspect VBT data structure and print detailed information."""
    
    console.print(f"\n[bold cyan]🔍 Inspecting {title}[/bold cyan]")
    
    if data is None:
        console.print("[red]❌ Data is None[/red]")
        return
    
    # Basic info
    info_table = Table(show_header=False, box=box.ROUNDED)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="white")
    
    info_table.add_row("Data Type", str(type(data)))
    
    # Check if it has symbols
    if hasattr(data, 'symbols'):
        symbols = list(data.symbols)
        info_table.add_row("Symbols Count", str(len(symbols)))
        info_table.add_row("Symbols", str(symbols))
    else:
        info_table.add_row("Symbols", "❌ No symbols attribute")
    
    # Check wrapper info
    if hasattr(data, 'wrapper'):
        wrapper = data.wrapper
        info_table.add_row("Wrapper Shape", str(wrapper.shape))
        info_table.add_row("Wrapper Columns", str(list(wrapper.columns)))
        info_table.add_row("Index Start", str(wrapper.index[0]))
        info_table.add_row("Index End", str(wrapper.index[-1]))
        info_table.add_row("Index Length", str(len(wrapper.index)))
    else:
        info_table.add_row("Wrapper", "❌ No wrapper attribute")
    
    console.print(info_table)
    
    # Check OHLCV attributes
    ohlcv_attrs = ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume']
    available_ohlcv = []
    missing_ohlcv = []
    
    for attr in ohlcv_attrs:
        if hasattr(data, attr) and getattr(data, attr) is not None:
            available_ohlcv.append(attr)
        else:
            missing_ohlcv.append(attr)
    
    ohlcv_table = Table(title="OHLCV Attributes", box=box.ROUNDED)
    ohlcv_table.add_column("Available", style="green")
    ohlcv_table.add_column("Missing", style="red")
    
    ohlcv_table.add_row(
        "\n".join(available_ohlcv) if available_ohlcv else "None",
        "\n".join(missing_ohlcv) if missing_ohlcv else "None"
    )
    
    console.print(ohlcv_table)
    
    # Check VBT convenience methods
    convenience_methods = {
        'has_ohlc': 'Has OHLC',
        'has_ohlcv': 'Has OHLCV', 
        'has_any_ohlc': 'Has Any OHLC',
        'has_any_ohlcv': 'Has Any OHLCV'
    }
    
    convenience_table = Table(title="VBT Convenience Methods", box=box.ROUNDED)
    convenience_table.add_column("Method", style="cyan")
    convenience_table.add_column("Result", style="white")
    
    for method, description in convenience_methods.items():
        if hasattr(data, method):
            try:
                result = getattr(data, method)()
                convenience_table.add_row(description, f"✅ {result}" if result else f"❌ {result}")
            except Exception as e:
                convenience_table.add_row(description, f"⚠️ Error: {e}")
        else:
            convenience_table.add_row(description, "❌ Method not available")
    
    console.print(convenience_table)
    
    # Try to get underlying data
    console.print(f"\n[bold yellow]📊 Underlying Data Structure[/bold yellow]")
    try:
        underlying = data.get()
        console.print(f"Underlying type: {type(underlying)}")
        
        if isinstance(underlying, tuple):
            console.print(f"Tuple with {len(underlying)} elements:")
            for i, elem in enumerate(underlying):
                console.print(f"  Element {i}: {type(elem)} - {elem.shape if hasattr(elem, 'shape') else 'no shape'}")
        elif hasattr(underlying, 'shape'):
            console.print(f"Shape: {underlying.shape}")
            if hasattr(underlying, 'columns'):
                console.print(f"Columns: {list(underlying.columns)}")
                console.print(f"Column types: {underlying.dtypes}")
                
                # Show sample data
                console.print(f"\nFirst 5 rows:")
                console.print(underlying.head())
                
                # Check for NaN values
                nan_counts = underlying.isnull().sum()
                if nan_counts.sum() > 0:
                    console.print(f"\nNaN counts per column:")
                    console.print(nan_counts)
                else:
                    console.print("\n✅ No NaN values found")
        else:
            console.print(f"Unknown data structure: {underlying}")
            
    except Exception as e:
        console.print(f"❌ Error getting underlying data: {e}")
    
    console.print("\n" + "="*80)

def main():
    """Main inspection function."""
    
    console.print(Panel(
        "[bold cyan]VBT Cached Data Inspector[/bold cyan]\n"
        "Analyzing cached data structure and integrity",
        title="🔍 Data Inspector",
        box=box.DOUBLE
    ))
    
    # Initialize data storage
    try:
        console.print(f"📁 Data storage path: {data_storage.storage_dir}")
    except Exception as e:
        console.print(f"❌ Error initializing data storage: {e}")
        return
    
    # Check what files exist
    files_to_check = [
        ('binance', '4h', 'spot'),
        ('binance', '1d', 'spot')
    ]
    
    for exchange_id, timeframe, market_type in files_to_check:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Loading {exchange_id.upper()} {timeframe} {market_type.upper()} data[/bold]")
        
        try:
            # Load the data
            data = data_storage.load_data(exchange_id, timeframe, market_type=market_type)
            
            # Inspect it
            inspect_vbt_data(data, f"{exchange_id.upper()} {timeframe} {market_type.upper()}")
            
        except Exception as e:
            console.print(f"❌ Error loading {exchange_id} {timeframe}: {e}")
            import traceback
            console.print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main() 