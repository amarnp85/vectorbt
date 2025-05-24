#!/usr/bin/env python3
"""Debug VBT Data Structure

Examine VBT data structure to understand storage resampling issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbtpro as vbt
from backtester.data.storage.data_storage import data_storage

def examine_vbt_structure():
    """Examine VBT data structure for debugging storage resampling."""
    
    print("🔍 Examining VBT Data Structure")
    print("=" * 50)
    
    # Load 4h data
    print("\n📊 Loading 4h data...")
    data_4h = data_storage.load_data('binance', '4h', market_type='spot')
    
    if data_4h is None:
        print("❌ No 4h data found")
        return
    
    print(f"✅ 4h data loaded")
    print(f"   Type: {type(data_4h)}")
    print(f"   Symbols: {list(data_4h.symbols) if hasattr(data_4h, 'symbols') else 'No symbols'}")
    print(f"   Shape: {data_4h.wrapper.shape if hasattr(data_4h, 'wrapper') else 'No wrapper'}")
    
    if hasattr(data_4h, 'wrapper'):
        print(f"   Columns: {list(data_4h.wrapper.columns)}")
        
    # Check for OHLCV attributes
    print("\n🔧 Checking OHLCV attributes...")
    ohlcv_attrs = ['open', 'high', 'low', 'close', 'volume', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for attr in ohlcv_attrs:
        has_attr = hasattr(data_4h, attr)
        if has_attr:
            attr_data = getattr(data_4h, attr)
            attr_type = type(attr_data)
            attr_shape = attr_data.shape if hasattr(attr_data, 'shape') else 'No shape'
            print(f"   ✅ {attr}: {attr_type} - {attr_shape}")
        else:
            print(f"   ❌ {attr}: Not found")
    
    # Check wrapper columns
    print("\n📋 Wrapper analysis...")
    if hasattr(data_4h, 'wrapper'):
        wrapper = data_4h.wrapper
        print(f"   Index type: {type(wrapper.index)}")
        print(f"   Index length: {len(wrapper.index)}")
        print(f"   Columns type: {type(wrapper.columns)}")
        print(f"   Columns: {list(wrapper.columns)}")
        
        if hasattr(wrapper, 'grouper'):
            print(f"   Grouper: {wrapper.grouper}")
    
    # Try get() method
    print("\n🔧 Testing .get() method...")
    try:
        raw_data = data_4h.get()
        print(f"   ✅ .get() successful")
        print(f"   Type: {type(raw_data)}")
        print(f"   Shape: {raw_data.shape if hasattr(raw_data, 'shape') else 'No shape'}")
        
        if hasattr(raw_data, 'columns'):
            print(f"   Columns: {list(raw_data.columns)}")
            print(f"   Column types: {type(raw_data.columns)}")
            
            if hasattr(raw_data.columns, 'nlevels'):
                print(f"   MultiIndex levels: {raw_data.columns.nlevels}")
                if raw_data.columns.nlevels > 1:
                    for level in range(raw_data.columns.nlevels):
                        level_values = raw_data.columns.get_level_values(level)
                        print(f"     Level {level}: {list(set(level_values))}")
        else:
            print(f"   No columns attribute")
            
    except Exception as e:
        print(f"   ❌ .get() failed: {e}")
    
    # Check specific features that should exist
    print("\n🎯 Feature-specific checks...")
    
    # Check if it's a features-oriented data
    if hasattr(data_4h, 'feature_oriented') and data_4h.feature_oriented:
        print("   📊 Data is feature-oriented")
        if hasattr(data_4h, 'features'):
            print(f"   Features: {list(data_4h.features)}")
    
    # Check if it's symbol-oriented data  
    if hasattr(data_4h, 'symbol_oriented') and data_4h.symbol_oriented:
        print("   📊 Data is symbol-oriented")
    
    # Check ohlcv properties
    if hasattr(data_4h, 'has_ohlcv'):
        print(f"   Has OHLCV: {data_4h.has_ohlcv}")
        
    if hasattr(data_4h, 'has_ohlc'):
        print(f"   Has OHLC: {data_4h.has_ohlc}")
        
    # Check data attribute
    print("\n📂 Checking .data attribute...")
    if hasattr(data_4h, 'data'):
        data_attr = data_4h.data
        print(f"   Type: {type(data_attr)}")
        if isinstance(data_attr, dict):
            print(f"   Keys: {list(data_attr.keys())}")
            for key, value in data_attr.items():
                print(f"     {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else 'No shape'}")
        else:
            print(f"   Shape: {data_attr.shape if hasattr(data_attr, 'shape') else 'No shape'}")
            if hasattr(data_attr, 'columns'):
                print(f"   Columns: {list(data_attr.columns)}")
    
    print("\n" + "=" * 50)
    print("🎯 Analysis complete")

if __name__ == "__main__":
    examine_vbt_structure() 