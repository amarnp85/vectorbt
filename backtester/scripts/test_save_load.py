#!/usr/bin/env python3
"""Test Save/Load Process

Test where OHLCV data gets corrupted during save/load.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbtpro as vbt
from backtester.data.storage.data_storage import data_storage

def analyze_ohlcv_data(data, stage_name):
    """Analyze OHLCV data structure at a specific stage."""
    
    print(f"\n📊 {stage_name}")
    print("-" * 40)
    
    print(f"   Type: {type(data)}")
    print(f"   Symbols: {list(data.symbols) if hasattr(data, 'symbols') else 'No symbols'}")
    print(f"   Shape: {data.wrapper.shape if hasattr(data, 'wrapper') else 'No wrapper'}")
    
    if hasattr(data, 'wrapper'):
        print(f"   Columns: {list(data.wrapper.columns)}")
    
    # Check OHLCV attributes
    print(f"   OHLCV Status:")
    ohlcv_attrs = ['open', 'high', 'low', 'close', 'volume']
    
    for attr in ohlcv_attrs:
        has_attr = hasattr(data, attr)
        if has_attr:
            attr_data = getattr(data, attr)
            if attr_data is not None:
                attr_type = type(attr_data)
                attr_shape = attr_data.shape if hasattr(attr_data, 'shape') else 'No shape'
                print(f"     ✅ {attr}: {attr_type} - {attr_shape}")
            else:
                print(f"     ⚠️ {attr}: None")
        else:
            print(f"     ❌ {attr}: Not found")
    
    # Check properties
    if hasattr(data, 'has_ohlcv'):
        print(f"   Has OHLCV: {data.has_ohlcv}")
    if hasattr(data, 'has_ohlc'):
        print(f"   Has OHLC: {data.has_ohlc}")

def test_save_load_process():
    """Test the complete save/load process to find where OHLCV gets corrupted."""
    
    print("🧪 Testing Save/Load Process")
    print("=" * 50)
    
    try:
        # Step 1: Fetch fresh data
        print("\n🔴 STEP 1: Fresh VBT Fetch")
        fresh_data = vbt.CCXTData.pull(
            "BTC/USDT",
            exchange="binance",
            timeframe="4h",
            start="2024-01-01",
            end="2024-01-10",
            show_progress=False
        )
        analyze_ohlcv_data(fresh_data, "Fresh VBT Data")
        
        # Step 2: Save the data
        print("\n🟡 STEP 2: Saving Data")
        success = data_storage.save_data(fresh_data, 'binance', '4h_test', 'spot')
        print(f"   Save success: {success}")
        
        # Step 3: Load the data back
        print("\n🟢 STEP 3: Loading Data Back")
        loaded_data = data_storage.load_data('binance', '4h_test', market_type='spot')
        
        if loaded_data is not None:
            analyze_ohlcv_data(loaded_data, "Loaded VBT Data")
            
            # Compare before and after
            print("\n🔍 COMPARISON:")
            print(f"   Fresh data OHLCV: {fresh_data.has_ohlcv if hasattr(fresh_data, 'has_ohlcv') else 'Unknown'}")
            print(f"   Loaded data OHLCV: {loaded_data.has_ohlcv if hasattr(loaded_data, 'has_ohlcv') else 'Unknown'}")
            
            # Test .get() on both
            print("\n🔧 Testing .get() method:")
            try:
                fresh_raw = fresh_data.get()
                loaded_raw = loaded_data.get()
                
                print(f"   Fresh .get() type: {type(fresh_raw)}")
                print(f"   Fresh .get() shape: {fresh_raw.shape if hasattr(fresh_raw, 'shape') else 'No shape'}")
                if hasattr(fresh_raw, 'columns'):
                    print(f"   Fresh .get() columns: {list(fresh_raw.columns)}")
                
                print(f"   Loaded .get() type: {type(loaded_raw)}")
                print(f"   Loaded .get() shape: {loaded_raw.shape if hasattr(loaded_raw, 'shape') else 'No shape'}")
                if hasattr(loaded_raw, 'columns'):
                    print(f"   Loaded .get() columns: {list(loaded_raw.columns)}")
                else:
                    print(f"   Loaded .get() has no columns")
                
            except Exception as get_error:
                print(f"   ❌ .get() comparison failed: {get_error}")
            
        else:
            print("   ❌ Failed to load data back")
        
        # Step 4: Cleanup
        print("\n🗑️ STEP 4: Cleanup")
        import os
        test_files = [
            '/Users/amarpatel/python/backtester/backtester/vbt_data/binance_spot_4h_test.pickle',
            '/Users/amarpatel/python/backtester/backtester/vbt_data/binance_spot_4h_test.pickle.blosc'
        ]
        
        for test_file in test_files:
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"   Removed: {test_file}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_save_load_process() 