#!/usr/bin/env python3
"""Debug multi-symbol data structures"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vectorbtpro as vbt
from backtester.data import fetch_data

# Load multi-symbol data
symbols = ["BTC/USDT", "ETH/USDT"]
data = fetch_data(symbols=symbols, exchange="binance", timeframe="1d", start_date="2023-01-01", end_date="2023-01-31")

print("=== DATA STRUCTURE ===")
print(f"data type: {type(data)}")
print(f"data.close type: {type(data.close)}")
print(f"data.close shape: {data.close.shape}")
print(f"data.close columns: {list(data.close.columns)}")
print(f"data.close index: {data.close.index[:5]}")

print("\n=== MOVING AVERAGES ===")
close = data.close
print(f"close type: {type(close)}")
print(f"close shape: {close.shape}")
print(f"close columns: {list(close.columns)}")

# Test MA calculation
fast_ma = vbt.MA.run(close, window=20)
print(f"fast_ma type: {type(fast_ma)}")
print(f"fast_ma.ma type: {type(fast_ma.ma)}")
print(f"fast_ma.ma shape: {fast_ma.ma.shape}")
print(f"fast_ma.ma columns: {list(fast_ma.ma.columns)}")

slow_ma = vbt.MA.run(close, window=50)
print(f"slow_ma.ma type: {type(slow_ma.ma)}")
print(f"slow_ma.ma shape: {slow_ma.ma.shape}")  
print(f"slow_ma.ma columns: {list(slow_ma.ma.columns)}")

print("\n=== ALIGNMENT CHECK ===")
print(f"fast_ma.ma.columns == slow_ma.ma.columns: {list(fast_ma.ma.columns) == list(slow_ma.ma.columns)}")
print(f"fast_ma.ma.index == slow_ma.ma.index: {fast_ma.ma.index.equals(slow_ma.ma.index)}")

# Test simple comparison
try:
    comparison = fast_ma.ma > slow_ma.ma
    print(f"Comparison successful: {comparison.shape}")
except Exception as e:
    print(f"Comparison failed: {e}")
    print(f"fast_ma columns: {fast_ma.ma.columns.tolist()}")
    print(f"slow_ma columns: {slow_ma.ma.columns.tolist()}")
    print(f"fast_ma index sample: {fast_ma.ma.index[:3].tolist()}")
    print(f"slow_ma index sample: {slow_ma.ma.index[:3].tolist()}")