#!/usr/bin/env python3
"""Simple test to verify the selective staleness fix"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.fetching.data_fetcher import fetch_data, _identify_stale_symbols
from data.storage.data_storage import data_storage
import vectorbtpro as vbt
from datetime import datetime, timedelta

def test_current_behavior():
    """Test current staleness behavior"""
    
    print("\n" + "="*60)
    print("Testing Current Staleness Behavior")
    print("="*60)
    
    # Test with a few symbols to see current behavior
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']
    
    print(f"\nTesting with symbols: {symbols}")
    print("Requesting data with end_date='now' to trigger staleness checks...")
    
    data = fetch_data(
        symbols=symbols,
        exchange_id='binance',
        timeframe='1h',
        start_date=None,
        end_date='now',
        use_cache=True,
        market_type='spot'
    )
    
    if data is not None:
        print(f"\n✅ Data fetch successful: {len(data.symbols)} symbols")
        
        # Check individual symbol timestamps
        print("\nPer-symbol analysis:")
        for symbol in data.symbols:
            try:
                if len(data.symbols) > 1:
                    symbol_data = data.close[symbol].dropna()
                else:
                    symbol_data = data.close.dropna()
                    
                if len(symbol_data) > 0:
                    latest = symbol_data.index[-1]
                    now = datetime.now()
                    time_diff = now - latest.replace(tzinfo=None)
                    
                    print(f"   {symbol}:")
                    print(f"      Latest: {latest}")
                    print(f"      Age: {time_diff}")
                    print(f"      Fresh: {'✓' if time_diff.total_seconds() < 7200 else '✗'}")  # 2 hours for 1h timeframe
            except Exception as e:
                print(f"   {symbol}: Error - {e}")
                
        # Test staleness detection function directly
        print(f"\nTesting _identify_stale_symbols function:")
        try:
            stale, fresh = _identify_stale_symbols(data, list(data.symbols), 'now', '1h')
            print(f"   Stale symbols: {stale}")
            print(f"   Fresh symbols: {fresh}")
        except Exception as e:
            print(f"   Error in staleness detection: {e}")
    else:
        print("❌ Data fetch failed")
    
    print("\n" + "="*60)
    print("Test Complete")
    print("="*60)
    
    # Expected behavior after fix:
    print("\n📋 Expected Behavior After Fix:")
    print("- If some symbols are stale and some are fresh:")
    print("  → Should see: 'Selective Update (X stale, Y fresh)'")
    print("  → Fresh symbols should keep their original timestamps")
    print("  → Only stale symbols should be updated")
    print("- If all symbols are fresh:")
    print("  → Should see: 'Complete' (cache hit)")
    print("- If all symbols are stale:")
    print("  → Should see: 'Stale (all symbols stale)'")

if __name__ == "__main__":
    test_current_behavior() 