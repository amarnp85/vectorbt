#!/usr/bin/env python3
"""Test VBT Fetch

Test what VBT CCXTData.pull() actually fetches.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbtpro as vbt

def test_vbt_fetch():
    """Test VBT fetch to see what data structure is created."""
    
    print("🧪 Testing VBT CCXTData.pull()")
    print("=" * 50)
    
    try:
        # Test single symbol fetch
        print("\n📊 Fetching single symbol (BTC/USDT)...")
        data = vbt.CCXTData.pull(
            "BTC/USDT",
            exchange="binance",
            timeframe="1d",
            start="2024-01-01",
            end="2024-01-10",
            show_progress=True
        )
        
        print(f"✅ Fetch successful")
        print(f"   Type: {type(data)}")
        print(f"   Symbols: {list(data.symbols) if hasattr(data, 'symbols') else 'No symbols'}")
        print(f"   Shape: {data.wrapper.shape if hasattr(data, 'wrapper') else 'No wrapper'}")
        
        if hasattr(data, 'wrapper'):
            print(f"   Columns: {list(data.wrapper.columns)}")
        
        # Check OHLCV attributes
        print("\n🔧 Checking OHLCV attributes...")
        ohlcv_attrs = ['open', 'high', 'low', 'close', 'volume']
        
        for attr in ohlcv_attrs:
            has_attr = hasattr(data, attr)
            if has_attr:
                attr_data = getattr(data, attr)
                if attr_data is not None:
                    attr_type = type(attr_data)
                    attr_shape = attr_data.shape if hasattr(attr_data, 'shape') else 'No shape'
                    print(f"   ✅ {attr}: {attr_type} - {attr_shape}")
                else:
                    print(f"   ⚠️ {attr}: None")
            else:
                print(f"   ❌ {attr}: Not found")
        
        # Check .get() method
        print("\n🔧 Testing .get() method...")
        try:
            raw_data = data.get()
            print(f"   ✅ .get() successful")
            print(f"   Type: {type(raw_data)}")
            
            if isinstance(raw_data, tuple):
                print(f"   Tuple length: {len(raw_data)}")
                for i, item in enumerate(raw_data):
                    print(f"     Item {i}: {type(item)} - {item.shape if hasattr(item, 'shape') else 'No shape'}")
                    if hasattr(item, 'columns'):
                        print(f"       Columns: {list(item.columns)}")
            else:
                print(f"   Shape: {raw_data.shape if hasattr(raw_data, 'shape') else 'No shape'}")
                if hasattr(raw_data, 'columns'):
                    print(f"   Columns: {list(raw_data.columns)}")
                    if len(raw_data) > 0:
                        print(f"   Sample data:")
                        print(raw_data.head())
                        
        except Exception as e:
            print(f"   ❌ .get() failed: {e}")
        
        # Check has_ohlcv
        if hasattr(data, 'has_ohlcv'):
            print(f"\n🎯 Has OHLCV: {data.has_ohlcv}")
        if hasattr(data, 'has_ohlc'):
            print(f"🎯 Has OHLC: {data.has_ohlc}")
        
        # Check data attribute
        if hasattr(data, 'data'):
            print(f"\n📂 Data attribute type: {type(data.data)}")
            if hasattr(data.data, 'keys'):
                print(f"   Keys: {list(data.data.keys())}")
                for key in list(data.data.keys())[:3]:  # First 3 keys
                    value = data.data[key]
                    print(f"     {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else 'No shape'}")
                    if hasattr(value, 'columns'):
                        print(f"       Columns: {list(value.columns)}")
        
    except Exception as e:
        print(f"❌ VBT fetch failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    test_vbt_fetch() 