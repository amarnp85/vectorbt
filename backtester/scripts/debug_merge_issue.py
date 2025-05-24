#!/usr/bin/env python3
"""Debug Merge Issue

Analyze exactly what happens during VBT data merging to understand symbol loss.
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

console = Console()

def debug_vbt_data_extraction():
    """Debug what happens when we extract DataFrames from VBT data."""
    
    console.print(Panel(
        "[bold cyan]VBT Data Extraction Debug[/bold cyan]\n"
        "Understanding how .get() method works with OHLCV data",
        title="🔍 Debug Inspector",
        style="blue"
    ))
    
    # Load the current 4h data
    data_4h = data_storage.load_data('binance', '4h', market_type='spot')
    
    if data_4h is None:
        console.print("[red]❌ No 4h data found[/red]")
        return
    
    console.print(f"[bold]4h Data Overview[/bold]")
    console.print(f"Symbols: {list(data_4h.symbols)}")
    console.print(f"Wrapper shape: {data_4h.wrapper.shape}")
    console.print(f"Wrapper columns: {list(data_4h.wrapper.columns)}")
    
    # Try different extraction methods
    console.print(f"\n[bold yellow]Testing .get() method[/bold yellow]")
    try:
        extracted = data_4h.get()
        console.print(f"get() returns: {type(extracted)}")
        
        if isinstance(extracted, tuple):
            console.print(f"Tuple with {len(extracted)} elements:")
            for i, elem in enumerate(extracted):
                if hasattr(elem, 'shape') and hasattr(elem, 'columns'):
                    console.print(f"  Element {i} ({data_4h.wrapper.columns[i]}): {elem.shape}, columns: {list(elem.columns)}")
                    console.print(f"    Sample columns: {elem.columns[:5]}")
                else:
                    console.print(f"  Element {i}: {type(elem)}")
        else:
            console.print(f"Single object: {type(extracted)}")
            if hasattr(extracted, 'shape'):
                console.print(f"Shape: {extracted.shape}")
            if hasattr(extracted, 'columns'):
                console.print(f"Columns: {list(extracted.columns)}")
    
    except Exception as e:
        console.print(f"[red]❌ get() failed: {e}[/red]")
    
    # Test individual OHLCV access
    console.print(f"\n[bold yellow]Testing individual OHLCV access[/bold yellow]")
    ohlcv_attrs = ['open', 'high', 'low', 'close', 'volume']
    
    for attr in ohlcv_attrs:
        try:
            if hasattr(data_4h, attr):
                ohlcv_data = getattr(data_4h, attr)
                console.print(f"{attr}: {type(ohlcv_data)}, shape: {ohlcv_data.shape if hasattr(ohlcv_data, 'shape') else 'no shape'}")
                if hasattr(ohlcv_data, 'columns'):
                    console.print(f"  Columns: {list(ohlcv_data.columns)}")
        except Exception as e:
            console.print(f"[red]{attr}: Error - {e}[/red]")
    
    # Test what happens with pandas concat
    console.print(f"\n[bold yellow]Testing pandas concat behavior[/bold yellow]")
    try:
        # Simulate what happens in the merge
        if hasattr(data_4h, 'open') and hasattr(data_4h, 'high'):
            open_data = data_4h.open
            high_data = data_4h.high
            
            console.print(f"Open data shape: {open_data.shape}, columns: {list(open_data.columns)}")
            console.print(f"High data shape: {high_data.shape}, columns: {list(high_data.columns)}")
            
            # Try concatenating them
            combined = pd.concat([open_data, high_data], axis=1)
            console.print(f"Combined shape: {combined.shape}, columns: {list(combined.columns)}")
            
            # Check for duplicates
            if combined.columns.duplicated().any():
                dup_cols = combined.columns[combined.columns.duplicated()].tolist()
                console.print(f"[red]Duplicate columns found: {dup_cols}[/red]")
            
    except Exception as e:
        console.print(f"[red]❌ Pandas concat test failed: {e}[/red]")
    
    # Test VBT's internal structure
    console.print(f"\n[bold yellow]Testing VBT internal structure[/bold yellow]")
    try:
        if hasattr(data_4h, 'wrapper'):
            wrapper = data_4h.wrapper
            console.print(f"Wrapper type: {type(wrapper)}")
            console.print(f"Wrapper columns: {list(wrapper.columns)}")
            console.print(f"Wrapper index: {wrapper.index.dtype}, length: {len(wrapper.index)}")
            
            # Check if it's MultiIndex
            if isinstance(wrapper.columns, pd.MultiIndex):
                console.print(f"MultiIndex levels: {wrapper.columns.nlevels}")
                console.print(f"Level names: {wrapper.columns.names}")
                console.print(f"Level 0 unique: {wrapper.columns.get_level_values(0).unique()}")
                if wrapper.columns.nlevels > 1:
                    console.print(f"Level 1 unique: {wrapper.columns.get_level_values(1).unique()}")
            else:
                console.print(f"Simple Index: {list(wrapper.columns)}")
    
    except Exception as e:
        console.print(f"[red]❌ Wrapper inspection failed: {e}[/red]")

def debug_symbol_reconstruction():
    """Debug the symbol reconstruction logic specifically."""
    
    console.print(f"\n" + "="*60)
    console.print(Panel(
        "[bold red]Symbol Reconstruction Debug[/bold red]\n"
        "Testing the exact logic used in mixed fetch",
        title="🔧 Reconstruction Debug", 
        style="red"
    ))
    
    # Load the data
    data_4h = data_storage.load_data('binance', '4h', market_type='spot')
    
    if data_4h is None:
        console.print("[red]❌ No data to debug[/red]")
        return
    
    console.print(f"Current symbols: {list(data_4h.symbols)}")
    console.print(f"Current wrapper columns: {list(data_4h.wrapper.columns)}")
    
    # Simulate the extraction process from _fetch_missing_symbols_and_merge
    try:
        extracted_data = data_4h.get()
        console.print(f"Extracted data type: {type(extracted_data)}")
        
        if isinstance(extracted_data, tuple):
            # This is what's happening in the merge logic
            first_element = extracted_data[0]
            console.print(f"First tuple element: {type(first_element)}, shape: {first_element.shape}")
            console.print(f"First element columns: {list(first_element.columns)}")
            console.print(f"Columns are symbols: {list(first_element.columns) == list(data_4h.symbols)}")
            
            # Test if this is actually a symbol-per-column format  
            if list(first_element.columns) == list(data_4h.symbols):
                console.print("[green]✅ Confirmed: Each column represents a symbol[/green]")
                console.print(f"Sample values from first symbol ({first_element.columns[0]}):")
                console.print(first_element.iloc[0:3, 0])
            else:
                console.print("[red]❌ Column names don't match symbols[/red]")
        
        # Test the symbol dictionary creation logic
        console.print(f"\n[bold]Testing Symbol Dictionary Logic[/bold]")
        if isinstance(extracted_data, tuple) and len(extracted_data) >= 5:
            # Simulate creating symbol dictionary with proper OHLCV
            feature_names = ['Open', 'High', 'Low', 'Close', 'Volume']
            symbols = list(data_4h.symbols)
            
            console.print(f"Would create dict for {len(symbols)} symbols with features: {feature_names}")
            
            symbol_dict = {}
            for i, symbol in enumerate(symbols):
                # Create a DataFrame with OHLCV features for this symbol
                symbol_ohlcv = pd.DataFrame(index=extracted_data[0].index)
                for j, feature in enumerate(feature_names):
                    if j < len(extracted_data):
                        symbol_ohlcv[feature] = extracted_data[j].iloc[:, i]
                
                symbol_dict[symbol] = symbol_ohlcv
                console.print(f"Created {symbol}: {symbol_ohlcv.shape} with columns {list(symbol_ohlcv.columns)}")
                
                # Show sample data
                if i == 0:  # Just for first symbol
                    console.print("Sample data:")
                    console.print(symbol_ohlcv.head(2))
                
                if i >= 2:  # Limit output
                    console.print(f"... (showing first 3 symbols only)")
                    break
            
            # Test VBT reconstruction
            console.print(f"\n[bold]Testing VBT Data Reconstruction[/bold]")
            try:
                reconstructed = vbt.Data.from_data(symbol_dict)
                console.print(f"✅ Reconstruction successful!")
                console.print(f"Reconstructed symbols: {list(reconstructed.symbols)}")
                console.print(f"Reconstructed shape: {reconstructed.wrapper.shape}")
                console.print(f"Reconstructed columns: {list(reconstructed.wrapper.columns)}")
            except Exception as recon_error:
                console.print(f"[red]❌ Reconstruction failed: {recon_error}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Symbol reconstruction debug failed: {e}[/red]")
        import traceback
        console.print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    debug_vbt_data_extraction()
    debug_symbol_reconstruction() 