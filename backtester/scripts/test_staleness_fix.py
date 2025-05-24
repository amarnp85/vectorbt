#!/usr/bin/env python3
"""Test script to verify staleness detection and resampling fixes"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetching.data_fetcher import fetch_data, fetch_top_symbols
import logging

# Set up logging to see debug messages
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s',
    handlers=[
        logging.FileHandler('staleness_test.log'),
        logging.StreamHandler()
    ]
)

def test_staleness_detection():
    """Test that stale cache is properly detected and handled"""
    
    print("\n" + "="*60)
    print("Testing Staleness Detection and Resampling Fix")
    print("="*60)
    
    # Test 1: Fetch data with potential stale cache
    print("\nTest 1: Fetching top 10 symbols with 'now' as end date")
    print("-" * 40)
    
    data = fetch_top_symbols(
        exchange_id='binance',
        quote_currency='USDT',
        market_type='spot',
        limit=10,
        timeframe='1h',
        start_date=None,  # Inception
        end_date='now',   # Current time - will detect staleness
        use_cache=True
    )
    
    if data is not None:
        print(f"\n✅ Data fetched successfully")
        print(f"   Symbols: {len(data.symbols)}")
        print(f"   Shape: {data.wrapper.shape}")
        
        # Check individual symbol timestamps
        print("\n   Per-symbol latest timestamps:")
        for symbol in data.symbols[:5]:  # Show first 5
            try:
                if len(data.symbols) > 1:
                    symbol_data = data.close[symbol].dropna()
                else:
                    symbol_data = data.close.dropna()
                    
                if len(symbol_data) > 0:
                    latest = symbol_data.index[-1]
                    print(f"   - {symbol}: {latest}")
            except:
                pass
    else:
        print(f"\n❌ Data fetch failed")
    
    # Test 2: Test specific scenario with mixed symbols
    print("\n\nTest 2: Testing specific symbols with mixed availability")
    print("-" * 40)
    
    test_symbols = ['BTC/USDT', 'ETH/USDT', 'DOGE/USDT', 'NONEXISTENT/USDT']
    
    data = fetch_data(
        symbols=test_symbols,
        exchange_id='binance',
        timeframe='1h',
        start_date=None,
        end_date='now',
        use_cache=True,
        market_type='spot',
        prefer_resampling=True  # This should be disabled if cache is stale
    )
    
    if data is not None:
        print(f"\n✅ Mixed symbol fetch successful")
        print(f"   Requested: {test_symbols}")
        print(f"   Received: {list(data.symbols)}")
    else:
        print(f"\n❌ Mixed symbol fetch failed")
    
    # Test 3: Force resampling scenario
    print("\n\nTest 3: Testing forced resampling (5m to 1h)")
    print("-" * 40)
    
    # First ensure we have 5m data cached
    print("   Fetching 5m data first...")
    data_5m = fetch_top_symbols(
        exchange_id='binance',
        limit=5,
        timeframe='5m',
        end_date='now',
        use_cache=True
    )
    
    if data_5m is not None:
        print(f"   ✅ 5m data cached: {len(data_5m.symbols)} symbols")
        
        # Now try to get 1h data which should resample from 5m
        print("   Attempting to resample to 1h...")
        
        # Clear 1h cache to force resampling
        from data.storage.data_storage import data_storage
        import os
        cache_file = os.path.join(data_storage.storage_dir, 'binance_spot_1h.pickle')
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print("   Cleared 1h cache to force resampling")
        
        data_1h = fetch_data(
            symbols=list(data_5m.symbols),
            exchange_id='binance',
            timeframe='1h',
            end_date='now',
            use_cache=True,
            prefer_resampling=True
        )
        
        if data_1h is not None:
            print(f"   ✅ Resampling successful: {len(data_1h.symbols)} symbols")
            # Check if timestamps are fresh
            if len(data_1h.symbols) > 0:
                symbol = list(data_1h.symbols)[0]
                try:
                    if len(data_1h.symbols) > 1:
                        latest = data_1h.close[symbol].dropna().index[-1]
                    else:
                        latest = data_1h.close.dropna().index[-1]
                    print(f"   Latest timestamp after resample: {latest}")
                except:
                    pass
        else:
            print(f"   ❌ Resampling failed (likely due to staleness check)")
    
    print("\n" + "="*60)
    print("Test Complete - Check logs for detailed debug info")
    print("="*60)

if __name__ == "__main__":
    test_staleness_detection() 