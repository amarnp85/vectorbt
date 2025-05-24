#!/usr/bin/env python3
"""Test script to verify selective staleness update logic"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetching.data_fetcher import fetch_data, _identify_stale_symbols
from data.storage.data_storage import data_storage
import vectorbtpro as vbt
import pandas as pd
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')

def test_selective_staleness():
    """Test selective staleness detection and updating"""
    
    print("\n" + "="*60)
    print("Testing Selective Staleness Detection")
    print("="*60)
    
    # Step 1: Fetch some fresh data to establish a baseline
    print("\nStep 1: Fetching fresh data to establish baseline...")
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT', 'PEPE/USDT']
    
    fresh_data = fetch_data(
        symbols=symbols,
        exchange_id='binance',
        timeframe='1h',
        start_date=None,
        end_date='now',
        use_cache=True,
        market_type='spot'
    )
    
    if fresh_data is None:
        print("❌ Failed to fetch baseline data")
        return
        
    print(f"✅ Baseline data fetched: {len(fresh_data.symbols)} symbols")
    
    # Step 2: Artificially age some symbols in the cache to simulate staleness
    print("\nStep 2: Artificially aging some symbols to simulate staleness...")
    
    try:
        # Load the cached data
        cached_data = data_storage.load_data('binance', '1h', market_type='spot')
        if cached_data is None:
            print("❌ No cached data found")
            return
            
        # Get the raw DataFrame
        df = cached_data.get()
        print(f"Original data shape: {df.shape}")
        print(f"Original date range: {df.index[0]} to {df.index[-1]}")
        
        # Artificially age BTC/USDT and ETH/USDT by removing recent data
        # This simulates them being stale
        cutoff_time = df.index[-1] - timedelta(hours=4)  # Make them 4 hours stale
        
        symbols_to_age = ['BTC/USDT', 'ETH/USDT']
        fresh_symbols = ['SOL/USDT', 'DOGE/USDT', 'PEPE/USDT']
        
        print(f"Aging symbols: {symbols_to_age}")
        print(f"Keeping fresh: {fresh_symbols}")
        print(f"Cutoff time: {cutoff_time}")
        
        # Create aged data by truncating some symbols
        if isinstance(df.columns, pd.MultiIndex):
            # MultiIndex case: (symbol, feature)
            aged_df = df.copy()
            for symbol in symbols_to_age:
                if symbol in df.columns.get_level_values(0):
                    # Remove recent data for this symbol
                    symbol_mask = aged_df.index <= cutoff_time
                    for feature in ['Open', 'High', 'Low', 'Close', 'Volume']:
                        if (symbol, feature) in aged_df.columns:
                            aged_df.loc[~symbol_mask, (symbol, feature)] = None
        else:
            # Simple case: if we can't easily manipulate, create artificial aging
            print("⚠️ Simple column structure - using timestamp manipulation")
            aged_df = df.copy()
            
        # Save the aged data back to cache
        aged_vbt_data = vbt.Data(aged_df)
        success = data_storage.save_data(aged_vbt_data, 'binance', '1h', 'spot')
        
        if success:
            print("✅ Successfully created artificially aged cache")
        else:
            print("❌ Failed to save aged cache")
            return
            
    except Exception as e:
        print(f"❌ Error creating aged cache: {e}")
        return
    
    # Step 3: Test staleness detection
    print("\nStep 3: Testing staleness detection...")
    
    try:
        # Load the aged cache and test staleness detection
        aged_cache = data_storage.load_data('binance', '1h', market_type='spot')
        
        stale_symbols, fresh_symbols_found = _identify_stale_symbols(
            aged_cache, symbols, 'now', '1h'
        )
        
        print(f"Staleness detection results:")
        print(f"   Stale symbols: {stale_symbols}")
        print(f"   Fresh symbols: {fresh_symbols_found}")
        
    except Exception as e:
        print(f"❌ Error testing staleness detection: {e}")
    
    # Step 4: Test selective fetch
    print("\nStep 4: Testing selective fetch with aged cache...")
    
    selective_data = fetch_data(
        symbols=symbols,
        exchange_id='binance',
        timeframe='1h',
        start_date=None,
        end_date='now',
        use_cache=True,
        market_type='spot'
    )
    
    if selective_data is not None:
        print(f"✅ Selective fetch successful: {len(selective_data.symbols)} symbols")
        
        # Check timestamps for each symbol
        print("\nPer-symbol latest timestamps after selective update:")
        for symbol in selective_data.symbols:
            try:
                if len(selective_data.symbols) > 1:
                    symbol_data = selective_data.close[symbol].dropna()
                else:
                    symbol_data = selective_data.close.dropna()
                    
                if len(symbol_data) > 0:
                    latest = symbol_data.index[-1]
                    print(f"   - {symbol}: {latest}")
            except Exception as e:
                print(f"   - {symbol}: Error - {e}")
    else:
        print("❌ Selective fetch failed")
    
    print("\n" + "="*60)
    print("Selective Staleness Test Complete")
    print("="*60)

if __name__ == "__main__":
    test_selective_staleness() 